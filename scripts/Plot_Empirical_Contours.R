#!/usr/bin/env Rscript
## Figure 3: joint credible contours for the ACTG175 treatment effects, MGP vs
## bMGP under the GPE, TPE and BTPE engines, from saved results.
## Usage: Rscript Plot_Empirical_Contours.R

SCRIPT_DIR <- local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
})

OPTS <- list(
  probs = 0.95,
  zero_lines = FALSE,
  prob = 0.95,             # kept for the single-level call path
  n_grid = 120,
  coef_x = "trt_2",
  coef_y = "trt_3",
  engine_order = c("GPE", "TPE", "BTPE"),

  frame_colour = "#4D4D4D",
  grid_colour  = "#E8E8E8",

  shared_limits = TRUE,  # one window for all engines -- see README
  equal_aspect = TRUE,   # beta_2 and beta_3 are the same kind of quantity, so
  panel_w_cm = 4.31,     # -> ~15.6 cm total, the paper's common figure width
  panel_h_cm = 3.70,     # only used when equal_aspect = FALSE
  hsep_cm    = 0.55,     # no repeated y descriptions, so the panels can sit close
  line_pt    = 0.7,
  pad_frac   = 0.08,     # matches the ggplot expansion(mult = 0.08)
  xlab_drop_mm   = -7,
  legend_drop_mm = -13,
  prefix = "dbc"
)

CALIB_LABELS <- list(
  mgp = c(PBP = "MGP", bPBP = "bMGP"),
  pbp = c(PBP = "PBP", bPBP = "bPBP")
)
CALIB_DASH <- c(PBP = "solid", bPBP = "dash pattern=on 5pt off 3pt")

load_contour_data <- function(results_file, opts) {
  if (!file.exists(results_file)) {
    stop("Double-bootstrap results not found: ", results_file,
         "\n  Produce it with Run_Empirical_Bagged.R, or pass ",
         "--selftest=true to check the writer on synthetic draws.",
         call. = FALSE)
  }
  out <- readRDS(results_file)
  include_bayes <- !is.null(out$bagged_fits$bayes)
  pieces <- lapply(opts$probs, function(pr) {
    d <- make_double_bootstrap_contour_data(
      standard_app = out$standard_app,
      bagged_fits = out$bagged_fits,
      include_bayes = include_bayes,
      coef_x = opts$coef_x,
      coef_y = opts$coef_y,
      n_grid = opts$n_grid,
      prob = pr
    )
    d$prob <- pr
    d$path <- paste0(d$path, "_p", round(100 * pr))
    d
  })
  do.call(rbind, pieces)
}

selftest_contour_data <- function(opts, seed = 2026) {
  set.seed(seed)
  fake_fit <- function(n, mu, sd_x, sd_y, rho) {
    z1 <- stats::rnorm(n)
    z2 <- rho * z1 + sqrt(1 - rho^2) * stats::rnorm(n)
    beta <- cbind(mu[1] + sd_x * z1, mu[2] + sd_y * z2, stats::rnorm(n))
    colnames(beta) <- c(opts$coef_x, opts$coef_y, "other")
    list(beta_draws = beta, coef_names = colnames(beta))
  }
  spec <- list(
    GPE  = list(c(0.10, 0.16), 0.055, 0.052, 0.25, 0.082, 0.079),
    TPE  = list(c(0.11, 0.17), 0.050, 0.048, 0.30, 0.079, 0.075),
    BTPE = list(c(0.09, 0.15), 0.048, 0.046, 0.20, 0.071, 0.069)
  )
  panels <- list()
  for (pr in opts$probs) {
    for (engine in names(spec)) {
      s <- spec[[engine]]
      d <- make_standard_vs_bagged_contour_panel(
        fake_fit(4000, s[[1]], s[[2]], s[[3]], s[[4]]),
        fake_fit(4000, s[[1]], s[[5]], s[[6]], s[[4]]),
        engine = engine, coef_x = opts$coef_x, coef_y = opts$coef_y,
        n_grid = opts$n_grid, prob = pr)
      d$prob <- pr
      d$path <- paste0(d$path, "_p", round(100 * pr))
      panels[[length(panels) + 1L]] <- d
    }
  }
  out <- do.call(rbind, panels)
  out$engine <- factor(out$engine, levels = opts$engine_order)
  out$calibration <- factor(out$calibration, levels = c("PBP", "bPBP"))
  out
}

panel_limits <- function(values, frac) {
  r <- range(values, finite = TRUE)
  if (diff(r) <= 0) r <- r + c(-0.5, 0.5)
  r + c(-1, 1) * frac * diff(r)
}

axis_ticks <- function(limits, n = 3) {
  br <- pretty(limits, n = n)
  br[br > limits[1] & br < limits[2]]
}

common_ticks <- function(xlim, ylim, n = 4) {
  span <- max(diff(xlim), diff(ylim))
  step <- diff(pretty(c(0, span), n = n))[1]
  grid_from <- function(lim) {
    k <- seq(floor(lim[1] / step), ceiling(lim[2] / step))
    v <- k * step
    v[v > lim[1] & v < lim[2]]
  }
  list(x = grid_from(xlim), y = grid_from(ylim))
}

num <- function(x, d = 5) formatC(x, format = "f", digits = d)

hex_of <- function(colour) toupper(sub("^#", "", colour))

coord_block <- function(x, y, indent = "      ", per_line = 4) {
  pairs <- sprintf("(%s,%s)", num(x), num(y))
  vapply(split(pairs, ceiling(seq_along(pairs) / per_line)),
         function(z) paste0(indent, paste(z, collapse = " ")), character(1))
}

write_contour_tikz <- function(contour_df, path, opts, label_set = "mgp") {
  px <- opts$prefix
  labels <- CALIB_LABELS[[label_set]]
  engines <- levels(droplevels(contour_df$engine))
  n_col <- length(engines)

  shared_x <- panel_limits(contour_df$x, opts$pad_frac)
  shared_y <- panel_limits(contour_df$y, opts$pad_frac)

  panel_h <- if (isTRUE(opts$equal_aspect) && isTRUE(opts$shared_limits)) {
    opts$panel_w_cm * diff(shared_y) / diff(shared_x)
  } else {
    opts$panel_h_cm
  }

  L <- c(
    "% =========================================================================",
    "%  ACTG175: 95% joint credible contours for (beta_2, beta_3),",
    "%  standard against bagged, one panel per predictive engine.",
    "%  Generated by Plot_Empirical_Contours.R -- do not edit by hand.",
    "%",
    "%  Preamble requirements (main.tex):",
    "%      \\usepackage{pgfplots}",
    "%      \\pgfplotsset{compat=1.18}",
    "%      \\usepgfplotslibrary{groupplots}",
    "% =========================================================================",
    "\\begin{tikzpicture}",
    sprintf("  \\definecolor{%sFrame}{HTML}{%s}", px, hex_of(opts$frame_colour)),
    sprintf("  \\definecolor{%sGrid}{HTML}{%s}", px, hex_of(opts$grid_colour)),
    sprintf("  \\definecolor{%sZero}{HTML}{9A9A9A}", px),
    "  \\pgfplotsset{",
    sprintf("    %sPanel/.style={", px),
    sprintf("      width=%scm, height=%scm, scale only axis,",
            num(opts$panel_w_cm, 2), num(panel_h, 2)),
    sprintf("      axis line style={draw=%sFrame, line width=0.45pt},", px),
    sprintf("      tick style={draw=%sFrame, line width=0.45pt},", px),
    "      tick label style={font=\\small}, title style={font=\\small, yshift=-2pt},",
    "      label style={font=\\small}, ylabel={$\\beta_3$},",
    "      enlargelimits=false, clip mode=individual,",
    "      every axis plot/.append style={line join=round, line cap=round},",
    "    },",
    sprintf("    %sStd/.style={draw=black, line width=%spt, %s},",
            px, num(opts$line_pt, 2), CALIB_DASH[["PBP"]]),
    sprintf("    %sBag/.style={draw=black, line width=%spt, %s},",
            px, num(opts$line_pt, 2), CALIB_DASH[["bPBP"]]),
    sprintf("    %sInner/.style={line width=%spt},", px, num(0.45, 2)),
    sprintf("    %sZero/.style={draw=%sZero, line width=0.4pt,", px, px),
    "      dash pattern=on 2pt off 2pt},",
    "  }",
    "  \\begin{groupplot}[",
    sprintf("    %sPanel,", px),
    sprintf("    group style={group name=%sgrp, group size=%d by 1,", px, n_col),
    sprintf("      horizontal sep=%scm,", num(opts$hsep_cm, 2)),
    "      y descriptions at=edge left},",
    "  ]"
  )

  shared_x <- panel_limits(contour_df$x, opts$pad_frac)
  shared_y <- panel_limits(contour_df$y, opts$pad_frac)

  for (engine in engines) {
    block <- contour_df[contour_df$engine == engine, , drop = FALSE]
    if (isTRUE(opts$shared_limits)) {
      xlim <- shared_x
      ylim <- shared_y
    } else {
      xlim <- panel_limits(block$x, opts$pad_frac)
      ylim <- panel_limits(block$y, opts$pad_frac)
    }
    ticks <- if (isTRUE(opts$equal_aspect) && isTRUE(opts$shared_limits)) {
      common_ticks(xlim, ylim)
    } else {
      list(x = axis_ticks(xlim), y = axis_ticks(ylim))
    }
    L <- c(L, "", sprintf("  %% ---------- %s ----------", engine),
           sprintf(paste0("  \\nextgroupplot[xmin=%s, xmax=%s, ymin=%s, ymax=%s,",
                          " xtick={%s}, ytick={%s}, title={%s}]"),
                   num(xlim[1]), num(xlim[2]), num(ylim[1]), num(ylim[2]),
                   paste(num(ticks$x, 3), collapse = ","),
                   paste(num(ticks$y, 3), collapse = ","),
                   engine))
    if (isTRUE(opts$zero_lines)) {
      L <- c(L,
        sprintf("  \\addplot[%sZero] coordinates {(%s,0) (%s,0)};",
                px, num(xlim[1]), num(xlim[2])),
        sprintf("  \\addplot[%sZero] coordinates {(0,%s) (0,%s)};",
                px, num(ylim[1]), num(ylim[2])))
    }
    for (pr in sort(unique(block$prob), decreasing = TRUE)) {
      for (calib in c("PBP", "bPBP")) {
        style <- if (calib == "PBP") paste0(px, "Std") else paste0(px, "Bag")
        if (pr < max(block$prob)) style <- paste0(style, ", ", px, "Inner")
        sub <- block[block$calibration == calib & block$prob == pr, , drop = FALSE]
        for (one_path in split(sub, sub$path, drop = TRUE)) {
          if (nrow(one_path) < 2L) next
          L <- c(L, sprintf("  \\addplot[%s] coordinates {", style))
          L <- c(L, coord_block(one_path$x, one_path$y), "  };")
        }
      }
    }
  }
  L <- c(L, "  \\end{groupplot}")

  mid <- ceiling(n_col / 2)
  L <- c(
    L,
    "  % ---- shared x label and legend --------------------------------------",
    sprintf(paste0("  \\node[anchor=north, font=\\small] at ",
                   "([yshift=%dmm] %sgrp c%dr1.south) {$\\beta_2$};"),
            opts$xlab_drop_mm, px, mid),
    sprintf(paste0("  \\node[anchor=north, font=\\small] at ",
                   "([yshift=%dmm] %sgrp c%dr1.south) {"),
            opts$legend_drop_mm, px, mid),
    sprintf(paste0("    \\tikz[baseline=-0.6ex]{\\draw[black, line width=%spt, %s]",
                   " (0,0) -- (0.75,0);}~%s\\qquad"),
            num(opts$line_pt, 2), CALIB_DASH[["PBP"]], labels[["PBP"]]),
    sprintf(paste0("    \\tikz[baseline=-0.6ex]{\\draw[black, line width=%spt, %s]",
                   " (0,0) -- (0.75,0);}~%s};"),
            num(opts$line_pt, 2), CALIB_DASH[["bPBP"]], labels[["bPBP"]]),
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
cli_value <- function(cli, key, default) {
  if (!is.null(cli[[key]]) && nzchar(cli[[key]])) cli[[key]] else default
}
cli_flag <- function(cli, key, default = FALSE) {
  v <- cli_value(cli, key, NA_character_)
  if (is.na(v)) default else tolower(v) %in% c("true", "t", "yes", "1")
}

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))
  opts <- OPTS
  label_set <- match.arg(cli_value(cli, "labels", "mgp"), c("mgp", "pbp"))
  if (!is.null(cli[["probs"]])) {
    opts$probs <- as.numeric(strsplit(cli[["probs"]], ",", fixed = TRUE)[[1]])
    if (anyNA(opts$probs) || any(opts$probs <= 0 | opts$probs >= 1)) {
      stop("--probs must be comma separated values strictly between 0 and 1.",
           call. = FALSE)
    }
  }
  opts$zero_lines <- cli_flag(cli, "zero_lines", opts$zero_lines)
  if (!is.null(cli[["panel_w"]])) {
    opts$panel_w_cm <- as.numeric(cli[["panel_w"]])
    if (!is.finite(opts$panel_w_cm) || opts$panel_w_cm <= 0) {
      stop("--panel_w must be a positive width in cm.", call. = FALSE)
    }
  }
  if (cli_flag(cli, "free_scales", FALSE)) {
    opts$shared_limits <- FALSE
    opts$equal_aspect <- FALSE
  }
  out_dir <- cli_value(cli, "out", file.path(SCRIPT_DIR, "output"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  source(file.path(SCRIPT_DIR, "Run_Empirical_Bagged.R"), chdir = TRUE)

  contour_df <- if (cli_flag(cli, "selftest", FALSE)) {
    message("Self-test: synthetic draws through the project's own contour code")
    selftest_contour_data(opts)
  } else {
    results_file <- cli_value(
      cli, "results",
      file.path(SCRIPT_DIR, "double_bootstrap_results",
                "aids_double_bootstrap_calibration.rds")
    )
    message("Reading: ", results_file)
    load_contour_data(results_file, opts)
  }
  if (nrow(contour_df) == 0L) stop("No contour lines were generated.", call. = FALSE)

  stem <- "double_bootstrap_contours"
  fig <- write_contour_tikz(contour_df, file.path(out_dir, paste0(stem, "_pgfplots.tex")),
                            opts, label_set = label_set)
  wrap <- write_standalone(fig, file.path(out_dir, paste0(stem, "_standalone.tex")))

  message("engines: ", paste(levels(droplevels(contour_df$engine)), collapse = ", "),
          " | contour vertices: ", nrow(contour_df))
  message("Wrote:\n  ", fig, "\n  ", wrap)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
