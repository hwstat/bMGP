#!/usr/bin/env Rscript
## =========================================================================
## Table 2 and Table 7, Bayes and BayesBag rows: the one renderer.  Reads
## ridge_bayesbag_summary.csv (and the timing CSV if it is there) and produces
## every formatted view of those numbers:
##
##   stdout                              the Markdown tables
##   ridge_bayesbag_table.tex            the LaTeX tabular
##   ridge_bayesbag_paper_style.csv      the cells in the paper's format
##
## Run_Table2_Bayes_BayesBag.R writes the CSVs and no formatting, so a cell
## appears in exactly one place in the code.
##
## Usage: Rscript Make_Table2_Bayes_BayesBag.R [--out=<dir with the CSVs>]
## =========================================================================

script_dir_of_this_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
}

SCRIPT_DIR <- script_dir_of_this_file()

args <- commandArgs(trailingOnly = TRUE)
keyed <- grep("^--[^=]+=", args, value = TRUE)
cli <- as.list(sub("^--[^=]+=", "", keyed))
names(cli) <- sub("^--([^=]+)=.*$", "\\1", keyed)
out_dir <- if (!is.null(cli[["out"]])) cli[["out"]] else
  file.path(SCRIPT_DIR, "output", "table2_bayes_bayesbag")

agg <- utils::read.csv(file.path(out_dir, "ridge_bayesbag_summary.csv"),
                       stringsAsFactors = FALSE)
agg <- agg[agg$parameter_set %in% c("active", "inactive"), , drop = FALSE]

f3 <- function(x) sprintf("%.3f", x)
cell <- function(cov, bias) sprintf("%s (%s)", f3(cov), f3(bias))

get <- function(p, scen, method, group, column) {
  hit <- agg[agg$p == p & agg$scenario == scen & agg$method == method &
               agg$parameter_set == group, column]
  if (length(hit) != 1L) NA_real_ else hit
}

methods <- c("Bayes", "BayesBag", "MGP", "bMGP")
methods <- methods[methods %in% unique(agg$method)]
p_grid <- sort(unique(agg$p))
scen_grid <- intersect(c("well", "miss"), unique(agg$scenario))
scen_label <- c(well = "m = 0 (homoskedastic)",
                miss = "m = 4 (heteroskedastic)")

cat("\n## Paper-layout table\n\n")
cat("| Method | Active Cov. (Bias) | Active Len. |",
    "Inactive Cov. (Bias) | Inactive Len. |\n")
cat("|---|---|---|---|---|\n")
for (p in p_grid) {
  for (scen in scen_grid) {
    cat(sprintf("| **p = %d, %s** | | | | |\n", p, scen_label[[scen]]))
    for (m in methods) {
      cat(sprintf("| %s | %s | %s | %s | %s |\n", m,
                  cell(get(p, scen, m, "active", "coverage"),
                       get(p, scen, m, "active", "signed_bias")),
                  f3(get(p, scen, m, "active", "mean_interval_length")),
                  cell(get(p, scen, m, "inactive", "coverage"),
                       get(p, scen, m, "inactive", "signed_bias")),
                  f3(get(p, scen, m, "inactive", "mean_interval_length"))))
    }
  }
}

## The replication count is read from the CSV rather than restated, so a short
## pilot run does not print the label of the full one.
cat(sprintf("\n## Monte Carlo standard errors (R = %s replicated data sets)\n\n",
            paste(sort(unique(agg$repeats)), collapse = ", ")))
cat("| p | DGP | Method | Group | Coverage | MC SE | Bias | MC SE |",
    "Length | MC SE |\n")
cat("|---|---|---|---|---|---|---|---|---|---|\n")
for (p in p_grid) {
  for (scen in scen_grid) {
    for (m in methods) {
      for (g in c("active", "inactive")) {
        cat(sprintf("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                    p, if (scen == "well") "m = 0" else "m = 4", m, g,
                    f3(get(p, scen, m, g, "coverage")),
                    f3(get(p, scen, m, g, "coverage_se")),
                    f3(get(p, scen, m, g, "signed_bias")),
                    f3(get(p, scen, m, g, "signed_bias_se")),
                    f3(get(p, scen, m, g, "mean_interval_length")),
                    f3(get(p, scen, m, g, "interval_length_se"))))
      }
    }
  }
}

timing_path <- file.path(out_dir, "ridge_bayesbag_timing.csv")
if (file.exists(timing_path)) {
  tm <- utils::read.csv(timing_path, stringsAsFactors = FALSE)
  cat("\n## Timing, one replicated data set, single core\n\n")
  cat("| p | DGP | Method | Elapsed (s) | User (s) | BLAS threads |\n")
  cat("|---|---|---|---|---|---|\n")
  for (i in seq_len(nrow(tm))) {
    cat(sprintf("| %d | %s | %s | %.3f | %.3f | %s |\n",
                tm$p[i], if (tm$scenario[i] == "well") "m = 0" else "m = 4",
                tm$method[i], tm$elapsed_sec[i], tm$user_sec[i],
                as.character(tm$blas_threads[i])))
  }
}

## --- the LaTeX tabular and the paper-style CSV ---------------------------
## Same numbers, same rounding, read from the same data frame as the Markdown
## above.

fmt_tex <- function(x) {
  s <- sprintf("%.3f", x)
  ifelse(substr(s, 1L, 1L) == "-", paste0("$", s, "$"), s)
}

scenario_m <- c(well = 0, miss = 4)

latex_table <- function(agg, methods_shown) {
  pick <- function(p_value, scenario, method, group, column) {
    hit <- agg[agg$p == p_value & agg$scenario == scenario &
                 agg$method == method & agg$parameter_set == group, column]
    if (length(hit) != 1L) {
      stop("Expected one row for p=", p_value, ", ", scenario, ", ", method,
           ", ", group, call. = FALSE)
    }
    hit
  }
  row_cells <- function(p_value, scenario, method) {
    c(
      method,
      sprintf("%s (%s)",
              fmt_tex(pick(p_value, scenario, method, "active", "coverage")),
              fmt_tex(pick(p_value, scenario, method, "active", "signed_bias"))),
      fmt_tex(pick(p_value, scenario, method, "active", "mean_interval_length")),
      sprintf("%s (%s)",
              fmt_tex(pick(p_value, scenario, method, "inactive", "coverage")),
              fmt_tex(pick(p_value, scenario, method, "inactive", "signed_bias"))),
      fmt_tex(pick(p_value, scenario, method, "inactive",
                   "mean_interval_length"))
    )
  }

  L <- c(
    "\\begin{tabular}{llcccc}",
    "\\toprule",
    "$p$ & Method",
    "& Active Cov. (Bias) & Active Len.",
    "& Inactive Cov. (Bias) & Inactive Len. \\\\"
  )
  for (scenario in scen_grid) {
    L <- c(L, "\\midrule",
           sprintf("\\multicolumn{6}{l}{\\textit{%s ($m=%d$)}} \\\\",
                   if (scenario == "well") "Homoskedastic" else "Heteroskedastic",
                   scenario_m[[scenario]]),
           "\\addlinespace[2pt]")
    first_p <- TRUE
    for (p_value in p_grid) {
      if (!first_p) L <- c(L, "\\addlinespace")
      for (i in seq_along(methods_shown)) {
        cells <- row_cells(p_value, scenario, methods_shown[i])
        stub <- if (i == 1L) sprintf("$%d$", p_value) else ""
        L <- c(L, paste0(paste(c(stub, cells), collapse = " & "), " \\\\"))
      }
      first_p <- FALSE
    }
  }
  c(L, "\\bottomrule", "\\end{tabular}")
}

paper_style_rows <- function(agg) {
  out <- agg
  out$coverage_bias <- sprintf("%.3f (%.3f)", out$coverage, out$signed_bias)
  out$length <- sprintf("%.3f", out$mean_interval_length)
  cols <- c("scenario", "m", "n", "p", "method", "parameter_set", "repeats",
            "coverage", "coverage_se", "signed_bias", "signed_bias_se",
            "mean_interval_length", "interval_length_se",
            "coverage_bias", "length")
  out[cols]
}

shown <- intersect(c("Bayes", "BayesBag"), methods)
if (length(shown) > 0L) {
  tex_path <- file.path(out_dir, "ridge_bayesbag_table.tex")
  tex <- latex_table(agg, shown)
  writeLines(tex, tex_path)
  cat("\n## LaTeX tabular\n\n```latex\n")
  cat(paste(tex, collapse = "\n"), "\n```\n", sep = "")
  message("Wrote ", tex_path)
}

style_path <- file.path(out_dir, "ridge_bayesbag_paper_style.csv")
utils::write.csv(paper_style_rows(agg), style_path, row.names = FALSE)
message("Wrote ", style_path)
