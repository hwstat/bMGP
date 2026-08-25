#!/usr/bin/env Rscript
## Figure 4: ACTG175 replicate-density PPC figure, from saved results.
## Usage: Rscript Plot_PPC_Density.R [--source=bagged --input=... --out=...]

OPTS <- list(
  source_set = "standard",            # standard | bagged
  n_density  = 512L,                  # stats::density(n = ...), its own default
  cut        = 3,                     # bandwidths of evaluation past the data
  diagnostics = c("absolute_residual_tail", "chi_square_discrepancy"),

  panel_w_cm = 6.01,                  # -> ~15.6 cm, the paper's common width
  panel_h_cm = 4.40,
  hsep_cm    = 1.55,                  # room for the right panel's own y ticks
  pad_frac_x = 0.04,                  # base R's default axis expansion
  pad_frac_y = 0.06,

  frame_colour = "#4D4D4D",
  obs_colour   = "#4D4D4D",
  line_pt      = 0.80,
  obs_pt       = 0.60,

  ylab_shift_mm  = -11,
  legend_drop_mm = -11,
  prefix = "ppcd"
)

DIAG_XLAB <- c(
  absolute_residual_tail = "$S_{\\mathrm{tail}}$",
  chi_square_discrepancy = "$S_{\\chi^2}$"
)

ENGINES <- list(
  list(slot = "gaussian_mp", std = "GPE",  bag = "bGPE",  dash = "solid"),
  list(slot = "t_mp",        std = "TPE",  bag = "bTPE",  dash = "dash pattern=on 4pt off 4pt"),
  list(slot = "bayes_t",     std = "BTPE", bag = "bBTPE", dash = "dash pattern=on 1pt off 2.5pt")
)

OBS_DASH <- "dash pattern=on 5pt off 2pt on 1pt off 2pt"

find_ppc_result_by_class <- function(ppc_single, diagnostic_class) {
  idx <- vapply(ppc_single$results, function(z) z$diagnostic_class, character(1))
  hit <- which(idx == diagnostic_class)
  if (length(hit) != 1L) {
    stop("Could not uniquely identify diagnostic_class = '", diagnostic_class,
         "'. Present: ", paste(unique(idx), collapse = ", "), call. = FALSE)
  }
  ppc_single$results[[hit]]
}

default_input_paths <- function(script_dir) {
  c(
    file.path(script_dir, "application_results", "aids_application_results.rds"),
    file.path(script_dir, "double_bootstrap_results",
              "aids_double_bootstrap_calibration.rds")
  )
}

locate_ppc_single <- function(obj, source_set) {
  has_results <- function(z) is.list(z) && !is.null(z$results)

  if (source_set == "bagged") {
    if (has_results(obj$bagged_ppc)) return(obj$bagged_ppc)
    stop("--source=bagged needs a $bagged_ppc component; this file has none. ",
         "It is written by double_bootstrap_calibration.R with ",
         "compute_bagged_ppc = TRUE.", call. = FALSE)
  }

  if (has_results(obj$ppc_single)) return(obj$ppc_single)
  if (has_results(obj$standard_app$ppc_single)) return(obj$standard_app$ppc_single)
  stop("No $ppc_single found. Expected the object written by run.R, or the ",
       "one written by double_bootstrap_calibration.R (which carries it as ",
       "$standard_app$ppc_single).", call. = FALSE)
}

build_panel <- function(res, diagnostic_class, opts, label_field) {
  S_obs <- res$gaussian_mp$S_obs

  reps <- lapply(ENGINES, function(e) {
    z <- res[[e$slot]]
    if (is.null(z)) NULL else as.numeric(z$S_rep)
  })
  keep <- !vapply(reps, is.null, logical(1))
  if (!any(keep)) {
    stop("No engine had an S_rep vector for ", diagnostic_class, call. = FALSE)
  }

  rng <- range(c(S_obs, unlist(reps[keep], use.names = FALSE)))

  bws <- vapply(which(keep), function(k) stats::bw.nrd0(reps[[k]]), numeric(1))
  ext <- opts$cut * max(bws)
  from <- rng[1] - ext
  to   <- rng[2] + ext

  curves <- list()
  for (k in which(keep)) {
    d <- stats::density(reps[[k]], from = from, to = to, n = opts$n_density)
    curves[[length(curves) + 1L]] <- list(
      label = ENGINES[[k]][[label_field]],
      dash  = ENGINES[[k]]$dash,
      x     = as.numeric(d$x),
      y     = as.numeric(d$y),
      bw    = d$bw,
      p_value = res[[ENGINES[[k]]$slot]]$p_value
    )
  }

  ymax <- max(vapply(curves, function(z) max(z$y), numeric(1)))
  edge <- max(vapply(curves, function(z) max(z$y[1], z$y[length(z$y)]),
                     numeric(1)))

  xlim <- if (opts$cut > 0) {
    c(from, to)
  } else {
    rng + c(-1, 1) * opts$pad_frac_x * diff(rng)
  }

  list(
    diagnostic_class = diagnostic_class,
    xlab = if (diagnostic_class %in% names(DIAG_XLAB)) {
      DIAG_XLAB[[diagnostic_class]]
    } else {
      "replicate statistic"
    },
    S_obs = S_obs,
    curves = curves,
    edge_frac = edge / ymax,
    xlim = xlim,
    ylim = c(0, ymax * (1 + opts$pad_frac_y))
  )
}

load_panels <- function(input_file, opts) {
  obj <- readRDS(input_file)
  ppc <- locate_ppc_single(obj, opts$source_set)
  label_field <- if (opts$source_set == "bagged") "bag" else "std"

  present <- vapply(ppc$results, function(z) z$diagnostic_class, character(1))
  wanted <- opts$diagnostics[opts$diagnostics %in% present]
  if (length(wanted) == 0L) {
    stop("None of the requested diagnostics is in the results file. ",
         "Requested: ", paste(opts$diagnostics, collapse = ", "),
         "; present: ", paste(unique(present), collapse = ", "), call. = FALSE)
  }

  lapply(wanted, function(cl) {
    build_panel(find_ppc_result_by_class(ppc, cl), cl, opts, label_field)
  })
}

selftest_panels <- function(opts, seed = 2026) {
  set.seed(seed)
  fake <- function(S_obs, means, sds) {
    list(
      gaussian_mp = list(S_obs = S_obs,
                         S_rep = stats::rnorm(100, means[1], sds[1]),
                         p_value = 0),
      t_mp        = list(S_obs = S_obs,
                         S_rep = stats::rnorm(100, means[2], sds[2]),
                         p_value = 0.47),
      bayes_t     = list(S_obs = S_obs,
                         S_rep = stats::rnorm(100, means[3], sds[3]),
                         p_value = 0.60)
    )
  }
  list(
    build_panel(fake(3.710, c(2.80, 3.71, 3.75), c(0.10, 0.25, 0.27)),
                "absolute_residual_tail", opts, "std"),
    build_panel(fake(1.000, c(1.00, 1.04, 1.05), c(0.04, 0.06, 0.06)),
                "chi_square_discrepancy", opts, "std")
  )
}

num <- function(x, d = 5) formatC(x, format = "f", digits = d)

hex_of <- function(colour) toupper(sub("^#", "", colour))

coord_block <- function(x, y, indent = "      ", per_line = 4) {
  pairs <- sprintf("(%s,%s)", num(x), num(y))
  vapply(split(pairs, ceiling(seq_along(pairs) / per_line)),
         function(z) paste0(indent, paste(z, collapse = " ")), character(1))
}

axis_ticks <- function(limits, n = 4) {
  br <- pretty(limits, n = n)
  br[br > limits[1] & br < limits[2]]
}

tick_precision <- function(ticks, max_dp = 4L) {
  for (d in 0:max_dp) {
    if (all(abs(ticks - round(ticks, d)) < 1e-9)) return(d)
  }
  max_dp
}

tick_label_style <- function(axis, ticks) {
  sprintf("%sticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=%d}",
          axis, tick_precision(ticks))
}

write_density_tikz <- function(panels, path, opts) {
  px <- opts$prefix
  n_col <- length(panels)

  legend_curves <- panels[[1]]$curves

  L <- c(
    "% =========================================================================",
    "%  ACTG175: replicate-statistic densities under GPE / TPE / BTPE,",
    "%  one panel per PPC diagnostic, with the observed value marked.",
    "%  Generated by plot_ppc_density_tikz.R -- do not edit by hand.",
    "%",
    "%  Preamble requirements (main.tex):",
    "%      \\usepackage{pgfplots}",
    "%      \\pgfplotsset{compat=1.18}",
    "%      \\usepgfplotslibrary{groupplots}",
    "% =========================================================================",
    "\\begin{tikzpicture}",
    sprintf("  \\definecolor{%sFrame}{HTML}{%s}", px, hex_of(opts$frame_colour)),
    sprintf("  \\definecolor{%sObs}{HTML}{%s}", px, hex_of(opts$obs_colour)),
    "  \\pgfplotsset{",
    sprintf("    %sPanel/.style={", px),
    sprintf("      width=%scm, height=%scm, scale only axis,",
            num(opts$panel_w_cm, 2), num(opts$panel_h_cm, 2)),
    sprintf("      axis line style={draw=%sFrame, line width=0.45pt},", px),
    sprintf("      tick style={draw=%sFrame, line width=0.45pt},", px),
    "      tick label style={font=\\small}, label style={font=\\small},",
    "      enlargelimits=false, clip mode=individual,",
    "      every axis plot/.append style={line join=round, line cap=round},",
    "    },",
    sprintf("    %sObsLine/.style={draw=%sObs, line width=%spt, %s},",
            px, px, num(opts$obs_pt, 2), OBS_DASH),
    "  }",
    "  \\begin{groupplot}[",
    sprintf("    %sPanel,", px),
    sprintf("    group style={group name=%sgrp, group size=%d by 1,", px, n_col),
    sprintf("      horizontal sep=%scm},", num(opts$hsep_cm, 2)),
    "  ]"
  )

  for (pn in panels) {
    xt <- axis_ticks(pn$xlim)
    yt <- axis_ticks(pn$ylim)
    L <- c(L, "", sprintf("  %% ---------- %s ----------", pn$diagnostic_class),
           sprintf(paste0("  \\nextgroupplot[xmin=%s, xmax=%s, ymin=%s, ymax=%s,",
                          " xlabel={%s},"),
                   num(pn$xlim[1]), num(pn$xlim[2]),
                   num(pn$ylim[1]), num(pn$ylim[2]), pn$xlab),
           sprintf("    xtick={%s}, %s,",
                   paste(num(xt, 3), collapse = ","), tick_label_style("x", xt)),
           sprintf("    ytick={%s}, %s]",
                   paste(num(yt, 3), collapse = ","), tick_label_style("y", yt)))

    L <- c(L, sprintf("  \\addplot[%sObsLine] coordinates {(%s,%s) (%s,%s)};",
                      px, num(pn$S_obs), num(pn$ylim[1]),
                      num(pn$S_obs), num(pn$ylim[2])))

    for (cv in pn$curves) {
      L <- c(L, sprintf("  %% %s   p = %s,  bw = %s", cv$label,
                        num(cv$p_value, 3), num(cv$bw, 5)))
      L <- c(L, sprintf("  \\addplot[draw=black, line width=%spt, %s] coordinates {",
                        num(opts$line_pt, 2), cv$dash))
      L <- c(L, coord_block(cv$x, cv$y), "  };")
    }
  }
  L <- c(L, "  \\end{groupplot}")

  if (n_col %% 2L == 1L) {
    anchor_node <- sprintf("%sgrp c%dr1.south", px, (n_col + 1L) %/% 2L)
    anchor_xshift <- 0
  } else {
    anchor_node <- sprintf("%sgrp c%dr1.south east", px, n_col %/% 2L)
    anchor_xshift <- 10 * opts$hsep_cm / 2
  }
  legend_bits <- vapply(seq_along(legend_curves), function(i) {
    cv <- legend_curves[[i]]
    sprintf(paste0("    \\tikz[baseline=-0.6ex]{\\draw[black, line width=%spt, %s]",
                   " (0,0) -- (0.7,0);}~%s\\qquad"),
            num(opts$line_pt, 2), cv$dash, cv$label)
  }, character(1))

  L <- c(
    L,
    "  % ---- shared y label and legend --------------------------------------",
    sprintf(paste0("  \\node[anchor=south, rotate=90, font=\\small] at ",
                   "([xshift=%dmm] %sgrp c1r1.west) {Density};"),
            opts$ylab_shift_mm, px),
    sprintf(paste0("  \\node[anchor=north, font=\\small] at ",
                   "([xshift=%smm, yshift=%dmm] %s) {"),
            num(anchor_xshift, 2), opts$legend_drop_mm, anchor_node),
    legend_bits,
    sprintf(paste0("    \\tikz[baseline=-0.6ex]{\\draw[%sObs, line width=%spt, %s]",
                   " (0,0) -- (0.7,0);}~Observed};"),
            px, num(opts$obs_pt, 2), OBS_DASH),
    "\\end{tikzpicture}"
  )

  writeLines(L, path)
  path
}

write_standalone <- function(figure_path, path) {
  writeLines(c(
    "% Minimal wrapper: compile this to preview the figure on its own.",
    "\\documentclass[border=4pt]{standalone}",
    "\\usepackage[T1]{fontenc}",
    "\\usepackage{amsmath}",
    "\\usepackage{pgfplots}",
    "\\pgfplotsset{compat=1.18}",
    "\\usepgfplotslibrary{groupplots}",
    "\\begin{document}",
    sprintf("\\input{%s}", tools::file_path_sans_ext(basename(figure_path))),
    "\\end{document}"
  ), path)
  path
}

parse_cli <- function(args) {
  keyed <- grep("^--[^=]+=", args, value = TRUE)
  out <- as.list(sub("^--[^=]+=", "", keyed))
  names(out) <- sub("^--([^=]+)=.*$", "\\1", keyed)
  out
}
cli_flag <- function(cli, key, default = FALSE) {
  v <- cli[[key]]
  if (is.null(v) || !nzchar(v)) default else tolower(v) %in% c("true", "t", "yes", "1")
}
script_dir_of_this_file <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) == 1L) dirname(normalizePath(sub("^--file=", "", fa))) else getwd()
}

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))
  script_dir <- script_dir_of_this_file()
  opts <- OPTS

  if (!is.null(cli[["source"]])) {
    if (!cli[["source"]] %in% c("standard", "bagged")) {
      stop("--source must be 'standard' or 'bagged'.", call. = FALSE)
    }
    opts$source_set <- cli[["source"]]
  }
  for (key in c("panel_w", "panel_h", "hsep")) {
    if (!is.null(cli[[key]])) {
      v <- as.numeric(cli[[key]])
      if (!is.finite(v) || v <= 0) {
        stop("--", key, " must be a positive length in cm.", call. = FALSE)
      }
      opts[[paste0(key, "_cm")]] <- v
    }
  }
  if (!is.null(cli[["n_density"]])) {
    opts$n_density <- as.integer(cli[["n_density"]])
  }
  if (!is.null(cli[["cut"]])) {
    v <- as.numeric(cli[["cut"]])
    if (!is.finite(v) || v < 0) {
      stop("--cut must be a non-negative number of bandwidths.", call. = FALSE)
    }
    opts$cut <- v
  }

  out_dir <- if (!is.null(cli[["out"]])) cli[["out"]] else file.path(script_dir, "output")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (cli_flag(cli, "selftest", FALSE)) {
    message("selftest: synthetic replicate statistics, real drawing path")
    panels <- selftest_panels(opts)
    stem <- "ppc_pe_density_two_panel_selftest"
  } else {
    input_file <- cli[["input"]]
    if (is.null(input_file) || !nzchar(input_file)) {
      candidates <- default_input_paths(script_dir)
      hit <- candidates[file.exists(candidates)]
      if (length(hit) == 0L) {
        stop("No results file found. Looked for:\n  ",
             paste(normalizePath(candidates, mustWork = FALSE), collapse = "\n  "),
             "\nPass --input=<file>, or --selftest=true to check the writer ",
             "on synthetic data.", call. = FALSE)
      }
      input_file <- hit[1]
    }
    if (!file.exists(input_file)) {
      stop("Results file not found: ", input_file, call. = FALSE)
    }
    message("reading ", normalizePath(input_file))
    panels <- load_panels(input_file, opts)
    stem <- if (opts$source_set == "bagged") {
      "ppc_pe_density_two_panel_bagged"
    } else {
      "ppc_pe_density_two_panel"
    }
  }

  for (pn in panels) {
    message(sprintf(
      "  %-24s S_obs = %.4f   x [%.4f, %.4f]   peak %.3f   edge %.1f%% of peak",
      pn$diagnostic_class, pn$S_obs, pn$xlim[1], pn$xlim[2],
      max(vapply(pn$curves, function(z) max(z$y), numeric(1))),
      100 * pn$edge_frac))
    for (cv in pn$curves) {
      message(sprintf("      %-6s p = %.3f   bw = %.5f   %d points",
                      cv$label, cv$p_value, cv$bw, length(cv$x)))
    }
  }

  fig <- write_density_tikz(panels,
                            file.path(out_dir, paste0(stem, "_pgfplots.tex")),
                            opts)
  wrap <- write_standalone(fig, file.path(out_dir, paste0(stem, "_standalone.tex")))

  message("\nWrote:\n  ", fig, "\n  ", wrap)
  invisible(panels)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
