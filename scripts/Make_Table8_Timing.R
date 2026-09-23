#!/usr/bin/env Rscript
## Table 9: assembles the timing table and the mixing table from the raw CSVs
## written by Run_Table8_Timing_SpikeSlab.R, and stacks every raw CSV into one
## long-format file. Writes nothing that is not read from those CSVs.
## Usage: Rscript Make_Table8_Timing.R [--p=200 --tag=gibbs --bags_tag=bags50 --out=DIR]
##
## Every number in the output is read from the raw CSVs or derived from them by
## a rule recorded in the `basis` column, so a measured number is
## distinguishable from an averaged one.
##
## Rows are selected by tag, scenario and kind before anything is summed. The
## timing scripts append to their CSVs, so summing a whole file would pool
## scenarios and reruns.
##
## Options (defaults in brackets)
##   --p=200           [200]      covariate count to report on
##   --tag=gibbs       [gibbs]    tag of the single-fit and MGP/bMGP rows
##   --bags_tag=bags50 [bags50]   tag of the bagged-fit rows
##   --families=linear,studentt   [both]  panels to write, in order
##   --chain=kept      [kept]     kept or thinned, for the mixing table
##   --collect=true    [true]     also write all_timings.csv
##   --out=<script dir>/output/table8
##
## Outputs, all under --out
##   timing_table.csv, timing_table.tex     the Table 9 body
##   mixing_table.csv, mixing_table.tex     effective sample sizes
##   all_timings.csv                        every raw row in one file

script_dir_of_this_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
}

SCRIPT_DIR <- script_dir_of_this_file()
ENGINE_DIR <- file.path(SCRIPT_DIR, "..", "src", "high_dimension")

source(file.path(ENGINE_DIR, "timing_common.R"))

FAMILY_LABEL <- c(
  linear = "Linear spike-and-slab",
  studentt = "Student-$t$ spike-and-slab"
)

read_timings <- function(path, required = TRUE) {
  if (!file.exists(path)) {
    if (required) stop("Missing timing CSV: ", path, call. = FALSE)
    return(NULL)
  }
  utils::read.csv(path, stringsAsFactors = FALSE)
}

fmt_seconds <- function(x) {
  if (is.na(x)) return("--")
  if (x < 1) return(sprintf("%.2f s", x))
  if (x < 120) return(sprintf("%.1f s", x))
  if (x < 7200) return(sprintf("%.1f min", x / 60))
  sprintf("%.1f h", x / 3600)
}

fmt_ess <- function(x) if (is.na(x)) "--" else sprintf("%.0f", x)

## Mean over the data-generating processes that were timed, with the per
## scenario values written out so the averaging stays visible in the CSV.
across_scenarios <- function(values, scenarios) {
  keep <- !is.na(values)
  list(mean = if (any(keep)) mean(values[keep]) else NA_real_,
       detail = paste(sprintf("%s %.2f s", scenarios[keep], values[keep]),
                      collapse = "; "),
       n = sum(keep))
}

latex_escape <- function(x) gsub("&", "\\\\&", x)

family_rows <- function(family, p_value, tag, bags_tag, gibbs, bags, mgp) {
  in_family <- function(df) {
    if (is.null(df) || nrow(df) == 0L) return(df)
    df[df$family == family & df$p == p_value, , drop = FALSE]
  }

  single <- in_family(gibbs)
  single <- single[single$tag == tag & single$kind == "single_fit", ,
                   drop = FALSE]
  mgp_f <- in_family(mgp)
  mgp_f <- mgp_f[mgp_f$tag == tag & mgp_f$component == "total_wall", ,
                 drop = FALSE]
  if (nrow(single) == 0L && nrow(mgp_f) == 0L) return(NULL)

  ## Gibbs, one posterior fit: mean over repetitions within a scenario, then
  ## mean over scenarios.
  per_scenario <- tapply(single$seconds, single$scenario, mean)
  gibbs_one <- across_scenarios(as.numeric(per_scenario), names(per_scenario))
  burnin <- unique(single$burnin)[1]
  n_keep <- unique(single$n_keep)[1]
  thin <- unique(single$thin)[1]

  ## BayesBag: sum the bagged fits WITHIN a scenario and tag, then mean over
  ## scenarios.
  bag_mean <- NA_real_; bag_detail <- "not run"; bag_count <- 0L
  bag_per_fit <- c(NA_real_, NA_real_)
  bags_f <- in_family(bags)
  if (!is.null(bags_f) && nrow(bags_f) > 0L) {
    bags_f <- bags_f[bags_f$tag == bags_tag & bags_f$kind == "bagged_fit", ,
                     drop = FALSE]
  }
  if (!is.null(bags_f) && nrow(bags_f) > 0L) {
    totals <- tapply(bags_f$seconds, bags_f$scenario, sum)
    counts <- tapply(bags_f$seconds, bags_f$scenario, length)
    bag <- across_scenarios(as.numeric(totals), names(totals))
    bag_mean <- bag$mean
    bag_count <- as.integer(stats::median(as.numeric(counts)))
    bag_per_fit <- c(min(bags_f$seconds), max(bags_f$seconds))
    bag_detail <- sprintf(
      "measured, %s bagged fits per scenario (%s); per fit %.1f to %.1f s",
      paste(sprintf("%s: %d", names(counts), as.integer(counts)),
            collapse = ", "),
      bag$detail, bag_per_fit[1], bag_per_fit[2])
  }

  wall <- mgp_f
  mgp_by <- wall[wall$method == "MGP", c("scenario", "seconds")]
  bmgp_by <- wall[wall$method == "bMGP", c("scenario", "seconds")]
  mgp_one <- across_scenarios(mgp_by$seconds, mgp_by$scenario)
  bmgp_one <- across_scenarios(bmgp_by$seconds, bmgp_by$scenario)
  total_paths <- unique(wall$total_paths_mgp)[1]
  n_boot <- unique(wall$n_boot_used)[1]

  data.frame(
    family = family,
    panel = FAMILY_LABEL[[family]],
    method = c(
      sprintf("Gibbs, one posterior fit (%s burn-in $+$ %s kept, thinned to %s)",
              format(burnin, big.mark = ","), format(n_keep, big.mark = ","),
              format(thin, big.mark = ",")),
      sprintf("BayesBag, %d Gibbs fits", if (bag_count > 0L) bag_count else 50L),
      sprintf("MGP, 1 initialization and %s paths",
              format(total_paths, big.mark = ",")),
      sprintf("bMGP, %d initializations and %s paths", n_boot,
              format(total_paths, big.mark = ","))
    ),
    seconds = c(gibbs_one$mean, bag_mean, mgp_one$mean, bmgp_one$mean),
    basis = c(
      sprintf("measured, %d repetitions per scenario (%s)",
              as.integer(stats::median(table(single$scenario))),
              gibbs_one$detail),
      bag_detail,
      sprintf("measured (%s)", mgp_one$detail),
      sprintf("measured (%s)", bmgp_one$detail)
    ),
    scenarios_averaged = c(gibbs_one$n, if (is.na(bag_mean)) 0L else
      length(unique(bags_f$scenario)), mgp_one$n, bmgp_one$n),
    p = p_value,
    tag = tag,
    stringsAsFactors = FALSE
  )
}

mixing_rows <- function(ess, p_value, tag, families, chain_kind) {
  if (is.null(ess) || nrow(ess) == 0L) return(NULL)
  ess <- ess[ess$tag == tag & ess$p == p_value & ess$chain == chain_kind, ,
             drop = FALSE]
  ess <- ess[ess$family %in% families, , drop = FALSE]
  if (nrow(ess) == 0L) return(NULL)
  ess <- ess[order(match(ess$family, families), ess$m), , drop = FALSE]
  data.frame(
    family = ess$family,
    panel = unname(FAMILY_LABEL[ess$family]),
    scenario = ess$scenario,
    m = ess$m,
    chain = ess$chain,
    n_draws = ess$n_draws,
    p = ess$p,
    n_active = ess$n_active,
    n_inactive = ess$n_inactive,
    ess_active_min = ess$ess_active_min,
    ess_active_median = ess$ess_active_median,
    ess_inactive_min = ess$ess_inactive_min,
    ess_inactive_median = ess$ess_inactive_median,
    ess_all_min = ess$ess_all_min,
    ess_all_median = ess$ess_all_median,
    mean_inclusion_active = ess$mean_inclusion_active,
    mean_inclusion_inactive = ess$mean_inclusion_inactive,
    stringsAsFactors = FALSE
  )
}

## Every raw CSV in one long-format file. Columns a given source file does not
## have are filled with NA and the file name is kept in `source_file`.
collect_raw <- function(dir_path, out_path) {
  pad_columns <- function(df, all_names) {
    for (name in setdiff(all_names, names(df))) df[[name]] <- NA
    df[all_names]
  }
  files <- list.files(dir_path, pattern = "\\.csv$", full.names = TRUE)
  files <- files[!grepl("all_timings|timing_table|mixing_table|_ess",
                        basename(files))]
  if (length(files) == 0L) {
    message("No raw timing CSVs in ", dir_path, "; all_timings.csv skipped.")
    return(invisible(NULL))
  }
  pieces <- lapply(files, function(path) {
    df <- utils::read.csv(path, stringsAsFactors = FALSE)
    df$source_file <- basename(path)
    df
  })
  all_names <- unique(unlist(lapply(pieces, names)))
  all_names <- c("source_file", setdiff(all_names, "source_file"))
  stacked <- do.call(rbind, lapply(pieces, pad_columns, all_names = all_names))
  utils::write.csv(stacked, out_path, row.names = FALSE)
  message("Wrote ", out_path, " (", nrow(stacked), " rows from ",
          length(files), " files)")
  invisible(stacked)
}

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))

  out_dir <- cli_value(cli, "out", file.path(SCRIPT_DIR, "output", "table8"))
  p_value <- as.integer(cli_value(cli, "p", "200"))
  tag <- cli_value(cli, "tag", "gibbs")
  bags_tag <- cli_value(cli, "bags_tag", "bags50")
  families <- strsplit(cli_value(cli, "families", "linear,studentt"), ",")[[1]]
  chain_kind <- cli_value(cli, "chain", "kept")

  path_of <- function(key, default) {
    out_path_of(cli_value(cli, key, default), out_dir)
  }

  gibbs <- read_timings(path_of("gibbs_csv", "gibbs_timings.csv"))
  bags <- read_timings(path_of("bags_csv", "gibbs_bags.csv"), required = FALSE)
  mgp <- read_timings(path_of("mgp_csv", "mgp_bmgp_timings.csv"))
  ess <- read_timings(path_of("ess_csv", "gibbs_timings_ess.csv"),
                      required = FALSE)

  out_csv <- path_of("out_csv", "timing_table.csv")
  out_tex <- path_of("tex", "timing_table.tex")
  mix_csv <- path_of("mixing_out", "mixing_table.csv")
  mix_tex <- path_of("mixing_tex", "mixing_table.tex")

  panels <- lapply(families, family_rows, p_value = p_value, tag = tag,
                   bags_tag = bags_tag, gibbs = gibbs, bags = bags, mgp = mgp)
  panels <- panels[!vapply(panels, is.null, logical(1))]
  if (length(panels) == 0L) {
    stop("No rows for tag=", tag, ", p=", p_value, call. = FALSE)
  }
  rows <- do.call(rbind, panels)
  utils::write.csv(rows, out_csv, row.names = FALSE)
  message("Wrote ", out_csv)

  tex <- c(
    sprintf("%% Timing of one replication of the spike-and-slab experiment at n = 250, p = %d.",
            p_value),
    "% Generated by Make_Table8_Timing.R -- do not edit by hand.",
    "% Requires \\usepackage{booktabs}.",
    "\\begin{tabular}{lr}",
    "\\toprule",
    "Method & One replication \\\\"
  )
  for (family in unique(rows$family)) {
    block <- rows[rows$family == family, , drop = FALSE]
    tex <- c(tex, "\\midrule",
             sprintf("\\multicolumn{2}{l}{\\emph{%s}} \\\\",
                     FAMILY_LABEL[[family]]))
    for (i in seq_len(nrow(block))) {
      tex <- c(tex, sprintf("\\quad %s & %s \\\\",
                            latex_escape(block$method[i]),
                            fmt_seconds(block$seconds[i])))
    }
  }
  tex <- c(tex, "\\bottomrule", "\\end{tabular}")
  writeLines(tex, out_tex)
  message("Wrote ", out_tex)

  mix <- mixing_rows(ess, p_value, tag, families, chain_kind)
  if (!is.null(mix)) {
    utils::write.csv(mix, mix_csv, row.names = FALSE)
    message("Wrote ", mix_csv)
    mtex <- c(
      "% Mixing of the Gibbs chain, all p coefficients, no subsetting.",
      "% Generated by Make_Table8_Timing.R -- do not edit by hand.",
      "% Requires \\usepackage{booktabs}.",
      "\\begin{tabular}{llrrrr}",
      "\\toprule",
      "& & \\multicolumn{2}{c}{Active} & \\multicolumn{2}{c}{Inactive} \\\\",
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
      "Model & DGP & Min & Median & Min & Median \\\\",
      "\\midrule"
    )
    for (i in seq_len(nrow(mix))) {
      mtex <- c(mtex, sprintf(
        "%s & $m = %g$ & %s & %s & %s & %s \\\\",
        mix$panel[i], mix$m[i],
        fmt_ess(mix$ess_active_min[i]), fmt_ess(mix$ess_active_median[i]),
        fmt_ess(mix$ess_inactive_min[i]), fmt_ess(mix$ess_inactive_median[i])
      ))
    }
    mtex <- c(mtex, "\\bottomrule", "\\end{tabular}")
    writeLines(mtex, mix_tex)
    message("Wrote ", mix_tex)
  } else {
    message("No ESS rows for chain=", chain_kind, "; mixing table skipped.")
  }

  if (cli_flag(cli, "collect", TRUE)) {
    collect_raw(out_dir, out_path_of("all_timings.csv", out_dir))
  }

  show <- rows[c("panel", "method", "seconds")]
  show$one_replication <- vapply(rows$seconds, fmt_seconds, character(1))
  print(show[c("panel", "method", "one_replication")], row.names = FALSE)
  invisible(rows)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
