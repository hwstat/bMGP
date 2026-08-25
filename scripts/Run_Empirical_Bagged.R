## ACTG175 bagged calibration (Section 4.4): B = 50 bootstrap data sets, each
## engine refitted per bag (one CmdStan fit per bag for BTPE); writes
## double_bootstrap_results/. Needs Run_Empirical_ACTG175.R results first.
## Usage: Rscript Run_Empirical_Bagged.R
source("Run_Empirical_ACTG175.R")

make_bootstrap_prepared <- function(prep, index) {
  X <- prep$X[index, , drop = FALSE]
  y <- prep$y[index]
  colnames(X) <- prep$coef_names

  list(
    y = as.numeric(y),
    X = X,
    n_obs = length(y),
    p = ncol(X),
    coef_names = prep$coef_names
  )
}

take_draws <- function(fit, n_draws, seed) {
  if (nrow(fit$beta_draws) <= n_draws) {
    return(fit)
  }

  set.seed(seed)
  idx <- sample.int(nrow(fit$beta_draws), size = n_draws, replace = FALSE)
  fit$beta_draws <- fit$beta_draws[idx, , drop = FALSE]
  fit$var_draws <- fit$var_draws[idx]
  fit
}

fit_bootstrap_predictive_engines <- function(prep_boot,
                                             paths_per_boot = 20,
                                             df_model = 5,
                                             N_add_hybrid = 100,
                                             seed = 9001,
                                             n_restart_boot = 10,
                                             include_bayes = TRUE,
                                             bayes_iter_sampling_boot = 1000,
                                             bayes_iter_warmup_boot = 500,
                                             bayes_chains_boot = 1,
                                             bayes_parallel_chains_boot = 1) {
  fit_g <- gaussian_regression_mp(
    y_obs = prep_boot$y,
    X_obs = prep_boot$X,
    B = paths_per_boot,
    N_add = N_add_hybrid,
    seed = seed
  )

  fit_t <- student_t_regression_mp(
    y_obs = prep_boot$y,
    X_obs = prep_boot$X,
    B = paths_per_boot,
    N_add = N_add_hybrid,
    df = df_model,
    n_restart = n_restart_boot,
    seed = seed
  )

  fit_b <- NULL
  if (isTRUE(include_bayes)) {
    fit_b_cmd <- fit_bayes_student_t_cmdstan(
      y_obs = prep_boot$y,
      X_obs = prep_boot$X,
      df_fixed = df_model,
      sigma0 = 10,
      iter_sampling = bayes_iter_sampling_boot,
      iter_warmup = bayes_iter_warmup_boot,
      chains = bayes_chains_boot,
      parallel_chains = bayes_parallel_chains_boot,
      seed = seed
    )
    fit_b <- extract_cmdstan_draws(fit_b_cmd, coef_names = prep_boot$coef_names)
    fit_b <- take_draws(fit_b, n_draws = paths_per_boot, seed = seed + 77L)
  }

  list(gaussian = fit_g, student_t = fit_t, bayes = fit_b)
}

combine_bagged_fits <- function(fits, engine = c("gaussian", "student_t", "bayes"),
                                display_label = NULL) {
  engine <- match.arg(engine)
  engine_fits <- lapply(fits, function(z) z[[engine]])
  engine_fits <- Filter(Negate(is.null), engine_fits)
  if (length(engine_fits) == 0L) {
    return(NULL)
  }

  beta_draws <- do.call(rbind, lapply(engine_fits, function(z) z$beta_draws))
  var_draws <- unlist(lapply(engine_fits, function(z) z$var_draws), use.names = FALSE)
  coef_names <- engine_fits[[1]]$coef_names
  colnames(beta_draws) <- coef_names

  if (is.null(display_label)) {
    display_label <- switch(
      engine,
      gaussian = "DB-MGP-Gaussian-hybrid",
      student_t = "DB-MGP-Student-t-hybrid",
      bayes = "DB-Bayes-Student-t"
    )
  }

  list(
    beta_draws = beta_draws,
    var_draws = var_draws,
    coef_names = coef_names,
    model = display_label,
    display_label = display_label,
    n_bootstrap_fits = length(engine_fits),
    n_draws = nrow(beta_draws)
  )
}

make_bootstrap_error <- function(b, err) {
  structure(
    list(
      bootstrap_index = b,
      message = conditionMessage(err),
      call = conditionCall(err)
    ),
    class = "bootstrap_error"
  )
}

is_bootstrap_error <- function(x) {
  inherits(x, "bootstrap_error") || inherits(x, "try-error")
}

bootstrap_error_message <- function(x, fallback_index = NA_integer_) {
  if (inherits(x, "bootstrap_error")) {
    idx <- x$bootstrap_index
    msg <- x$message
  } else {
    idx <- fallback_index
    msg <- paste(as.character(x), collapse = "\n")
  }
  paste0("bootstrap ", idx, ": ", msg)
}

validate_bootstrap_fits <- function(fits, include_bayes = TRUE) {
  failed <- which(vapply(fits, is_bootstrap_error, logical(1)))
  if (length(failed) > 0L) {
    details <- vapply(
      failed,
      function(i) bootstrap_error_message(fits[[i]], fallback_index = i),
      character(1)
    )
    stop(
      "One or more bootstrap replications failed before aggregation:\n",
      paste(details, collapse = "\n"),
      call. = FALSE
    )
  }

  required_names <- c("gaussian", "student_t", "bayes")
  malformed <- which(vapply(
    fits,
    function(z) !is.list(z) || !all(required_names %in% names(z)),
    logical(1)
  ))
  if (length(malformed) > 0L) {
    stop(
      "Malformed bootstrap fit object(s) at index: ",
      paste(malformed, collapse = ", "),
      call. = FALSE
    )
  }

  if (isTRUE(include_bayes)) {
    missing_bayes <- which(vapply(fits, function(z) is.null(z$bayes), logical(1)))
    if (length(missing_bayes) > 0L) {
      stop(
        "Bayes bootstrap fit is missing at index: ",
        paste(missing_bayes, collapse = ", "),
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}

summarize_bagged_variance_components <- function(fits,
                                                 engines = c("gaussian", "student_t", "bayes"),
                                                 terms = c("trt_2", "trt_3")) {
  rows <- list()

  for (engine in engines) {
    engine_fits <- lapply(fits, function(z) z[[engine]])
    engine_fits <- Filter(Negate(is.null), engine_fits)
    if (length(engine_fits) == 0L) {
      next
    }

    label <- switch(
      engine,
      gaussian = "DB-MGP-Gaussian-hybrid",
      student_t = "DB-MGP-Student-t-hybrid",
      bayes = "DB-Bayes-Student-t"
    )

    coef_names <- engine_fits[[1]]$coef_names
    for (term in terms) {
      idx <- match(term, coef_names)
      if (is.na(idx)) {
        stop("term not found in coef_names: ", term)
      }

      boot_means <- vapply(engine_fits, function(z) mean(z$beta_draws[, idx]), numeric(1))
      boot_vars <- vapply(engine_fits, function(z) stats::var(z$beta_draws[, idx]), numeric(1))
      all_draws <- unlist(lapply(engine_fits, function(z) z$beta_draws[, idx]), use.names = FALSE)

      rows[[length(rows) + 1L]] <- data.frame(
        model = label,
        term = term,
        total_sd = stats::sd(all_draws),
        total_var = stats::var(all_draws),
        mean_within_bootstrap_var = mean(boot_vars, na.rm = TRUE),
        between_bootstrap_mean_var = stats::var(boot_means),
        between_share_of_total = stats::var(boot_means) / stats::var(all_draws),
        n_bootstrap = length(engine_fits),
        draws_per_bootstrap = nrow(engine_fits[[1]]$beta_draws),
        row.names = NULL
      )
    }
  }

  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

summarize_bagged_coefficients <- function(fit_g_bagged, fit_t_bagged,
                                          fit_b_bagged = NULL,
                                          coef_targets = c("trt_2", "trt_3"),
                                          level = 0.95) {
  tabs <- list(
    extract_coef_summary_table(
      fit_g_bagged$beta_draws,
      fit_g_bagged$coef_names,
      fit_g_bagged$display_label,
      targets = coef_targets,
      level = level
    ),
    extract_coef_summary_table(
      fit_t_bagged$beta_draws,
      fit_t_bagged$coef_names,
      fit_t_bagged$display_label,
      targets = coef_targets,
      level = level
    )
  )

  if (!is.null(fit_b_bagged)) {
    tabs[[length(tabs) + 1L]] <- extract_coef_summary_table(
      fit_b_bagged$beta_draws,
      fit_b_bagged$coef_names,
      fit_b_bagged$display_label,
      targets = coef_targets,
      level = level
    )
  }

  out <- do.call(rbind, tabs)
  rownames(out) <- NULL
  out
}

contour_level <- function(kd, prob = 0.95) {
  z <- as.vector(kd$z)
  z_sorted <- z[order(z, decreasing = TRUE)]
  mass <- cumsum(z_sorted) / sum(z_sorted)
  z_sorted[min(which(mass >= prob))]
}

expand_degenerate_range <- function(x) {
  x <- range(x, finite = TRUE)
  if (diff(x) > 0) {
    return(x)
  }
  x + c(-0.5, 0.5)
}

contour_line_data <- function(kd, engine, calibration, prob = 0.95) {
  level <- contour_level(kd, prob = prob)
  lines <- grDevices::contourLines(kd$x, kd$y, kd$z, levels = level)
  if (length(lines) == 0L) {
    return(data.frame())
  }

  out <- lapply(seq_along(lines), function(i) {
    data.frame(
      engine = engine,
      calibration = calibration,
      path = paste(engine, calibration, i, sep = "_"),
      x = lines[[i]]$x,
      y = lines[[i]]$y,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

make_standard_vs_bagged_contour_panel <- function(standard_fit, bagged_fit,
                                                  engine,
                                                  coef_x = "trt_2",
                                                  coef_y = "trt_3",
                                                  n_grid = 120,
                                                  prob = 0.95) {
  coef_names <- standard_fit$coef_names
  ix <- match(coef_x, coef_names)
  iy <- match(coef_y, coef_names)
  if (any(is.na(c(ix, iy)))) {
    stop("coef_x / coef_y not found in coef_names.")
  }

  xlim <- expand_degenerate_range(c(standard_fit$beta_draws[, ix], bagged_fit$beta_draws[, ix]))
  ylim <- expand_degenerate_range(c(standard_fit$beta_draws[, iy], bagged_fit$beta_draws[, iy]))

  kd_std <- MASS::kde2d(
    standard_fit$beta_draws[, ix],
    standard_fit$beta_draws[, iy],
    n = n_grid,
    lims = c(xlim, ylim)
  )
  kd_bag <- MASS::kde2d(
    bagged_fit$beta_draws[, ix],
    bagged_fit$beta_draws[, iy],
    n = n_grid,
    lims = c(xlim, ylim)
  )

  rbind(
    contour_line_data(kd_std, engine = engine, calibration = "PBP", prob = prob),
    contour_line_data(kd_bag, engine = engine, calibration = "bPBP", prob = prob)
  )
}

make_double_bootstrap_contour_data <- function(standard_app, bagged_fits,
                                               include_bayes = TRUE,
                                               coef_x = "trt_2",
                                               coef_y = "trt_3",
                                               n_grid = 120,
                                               prob = 0.95) {
  panels <- list(
    make_standard_vs_bagged_contour_panel(
      standard_fit = standard_app$gaussian_hybrid,
      bagged_fit = bagged_fits$gaussian,
      engine = "GPE",
      coef_x = coef_x,
      coef_y = coef_y,
      n_grid = n_grid,
      prob = prob
    ),
    make_standard_vs_bagged_contour_panel(
      standard_fit = standard_app$t_hybrid,
      bagged_fit = bagged_fits$student_t,
      engine = "TPE",
      coef_x = coef_x,
      coef_y = coef_y,
      n_grid = n_grid,
      prob = prob
    )
  )

  if (isTRUE(include_bayes) && !is.null(bagged_fits$bayes)) {
    panels[[length(panels) + 1L]] <- make_standard_vs_bagged_contour_panel(
      standard_fit = standard_app$bayes_t,
      bagged_fit = bagged_fits$bayes,
      engine = "BTPE",
      coef_x = coef_x,
      coef_y = coef_y,
      n_grid = n_grid,
      prob = prob
    )
  }

  out <- do.call(rbind, panels)
  out$engine <- factor(out$engine, levels = c("GPE", "TPE", "BTPE"))
  out$calibration <- factor(out$calibration, levels = c("PBP", "bPBP"))
  out
}

plot_double_bootstrap_contour_facets <- function(standard_app, bagged_fits,
                                                 include_bayes = TRUE,
                                                 coef_x = "trt_2",
                                                 coef_y = "trt_3",
                                                 n_grid = 120,
                                                 prob = 0.95,
                                                 base_size = 10) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required to draw the faceted contour figure.")
  }

  contour_df <- make_double_bootstrap_contour_data(
    standard_app = standard_app,
    bagged_fits = bagged_fits,
    include_bayes = include_bayes,
    coef_x = coef_x,
    coef_y = coef_y,
    n_grid = n_grid,
    prob = prob
  )
  if (nrow(contour_df) == 0L) {
    stop("No contour lines were generated.")
  }

  ggplot2::ggplot(
    contour_df,
    ggplot2::aes(
      x = x,
      y = y,
      group = path,
      linetype = calibration
    )
  ) +
    ggplot2::geom_path(
      linewidth = 0.5,
      colour = "black",
      lineend = "round",
      linejoin = "round"
    ) +
    ggplot2::facet_wrap(ggplot2::vars(engine), nrow = 1, scales = "free") +
    ggplot2::scale_linetype_manual(
      values = c("PBP" = "solid", "bPBP" = "longdash"),
      breaks = c("PBP", "bPBP")
    ) +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.08)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = 0.08)) +
    ggplot2::labs(
      x = coef_axis_label(coef_x),
      y = coef_axis_label(coef_y),
      linetype = NULL
    ) +
    ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(fill = NA, colour = "black", linewidth = 0.3),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_size,
        face = "plain",
        margin = ggplot2::margin(0, 0, 5, 0)
      ),
      axis.line = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_line(colour = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(2, "pt"),
      axis.title = ggplot2::element_text(size = base_size + 0.5),
      axis.text = ggplot2::element_text(size = base_size - 1, colour = "black"),
      legend.position = "bottom",
      legend.text = ggplot2::element_text(size = base_size - 1.5),
      legend.key.width = grid::unit(1.1, "cm"),
      legend.key.height = grid::unit(0.25, "cm"),
      legend.spacing.x = grid::unit(0.3, "cm"),
      legend.margin = ggplot2::margin(-2, 0, 0, 0),
      plot.margin = ggplot2::margin(4, 14, 2, 4)
    ) +
    ggplot2::guides(
      linetype = ggplot2::guide_legend(
        override.aes = list(linewidth = 0.55),
        nrow = 1
      )
    )
}

save_double_bootstrap_contours <- function(standard_app, bagged_fits,
                                           figure_dir,
                                           save_png = TRUE,
                                           include_bayes = TRUE,
                                           base_size = 10) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  has_bayes <- isTRUE(include_bayes) && !is.null(bagged_fits$bayes)
  figure_width <- if (has_bayes) 7.2 else 5.0
  figure_height <- 2.7
  p <- plot_double_bootstrap_contour_facets(
    standard_app = standard_app,
    bagged_fits = bagged_fits,
    include_bayes = include_bayes,
    base_size = base_size
  )

  pdf_file <- file.path(figure_dir, "double_bootstrap_standard_vs_bagged_contours.pdf")
  ggplot2::ggsave(
    filename = pdf_file,
    plot = p,
    width = figure_width,
    height = figure_height,
    units = "in",
    device = grDevices::pdf
  )

  if (isTRUE(save_png)) {
    png_file <- file.path(figure_dir, "double_bootstrap_standard_vs_bagged_contours.png")
    ggplot2::ggsave(
      filename = png_file,
      plot = p,
      width = figure_width,
      height = figure_height,
      units = "in",
      dpi = 300
    )
  }

  invisible(TRUE)
}

redraw_double_bootstrap_contours_from_saved <- function(
    results_file = file.path("double_bootstrap_results", "aids_double_bootstrap_calibration.rds"),
    figure_dir = file.path(dirname(results_file), "figures"),
    save_png = TRUE,
    include_bayes = NULL,
    base_size = 10) {
  if (!file.exists(results_file)) {
    stop("Double-bootstrap results not found: ", results_file)
  }

  out <- readRDS(results_file)
  if (is.null(include_bayes)) {
    include_bayes <- !is.null(out$bagged_fits$bayes)
  }

  save_double_bootstrap_contours(
    standard_app = out$standard_app,
    bagged_fits = out$bagged_fits,
    figure_dir = figure_dir,
    save_png = save_png,
    include_bayes = include_bayes,
    base_size = base_size
  )

  invisible(out)
}

generate_pseudo_sample_from_bagged_draws <- function(X_obs,
                                                     beta_draws,
                                                     var_draws,
                                                     n_rep,
                                                     error = c("gaussian", "student_t"),
                                                     df = 5,
                                                     seed = 1234) {
  error <- match.arg(error)
  set.seed(seed)

  n_obs <- nrow(X_obs)
  p <- ncol(X_obs)
  draw_idx <- sample.int(nrow(beta_draws), size = 1)
  beta_draw <- as.numeric(beta_draws[draw_idx, ])
  var_draw <- max(as.numeric(var_draws[draw_idx]), 1e-8)

  idx_seq <- sample.int(n_obs, size = n_rep, replace = TRUE)
  X_rep <- X_obs[idx_seq, , drop = FALSE]
  colnames(X_rep) <- colnames(X_obs)
  eps <- switch(
    error,
    gaussian = rnorm(n_rep),
    student_t = rt(n_rep, df = df)
  )
  y_rep <- as.vector(X_rep %*% beta_draw) + sqrt(var_draw) * eps

  list(y = y_rep, X = matrix(X_rep, nrow = n_rep, ncol = p, dimnames = dimnames(X_rep)))
}

run_bagged_pbppc_engines <- function(y_obs, X_obs,
                                     fit_g_bagged,
                                     fit_t_bagged,
                                     fit_b_bagged = NULL,
                                     fit_g_reference = NULL,
                                     functional = c("chi2", "tail"),
                                     df = 5,
                                     q_tail = 0.995,
                                     n_mc_pvalue = 100,
                                     m_future_ppc = NULL,
                                     p_value_type = NULL,
                                     functional_restarts = 10,
                                     parallel = FALSE,
                                     n_cores = 1L,
                                     seed = 4040) {
  functional <- match.arg(functional)
  n_obs <- length(y_obs)
  if (is.null(m_future_ppc)) {
    m_future_ppc <- 2L * n_obs
  }
  if (m_future_ppc < n_obs) {
    stop("m_future_ppc must be at least n_obs.")
  }

  fun_obj <- build_pbppc_functional(
    y_obs = y_obs,
    X_obs = X_obs,
    fit_g = fit_g_reference,
    functional = functional,
    df = df,
    q_tail = q_tail,
    functional_restarts = functional_restarts
  )
  if (is.null(p_value_type)) {
    p_value_type <- fun_obj$p_value_type
  }
  S_obs <- fun_obj$S_obs
  get_S <- fun_obj$eval_stat
  include_bayes <- !is.null(fit_b_bagged)

  one_rep <- function(b) {
    dat_g <- generate_pseudo_sample_from_bagged_draws(
      X_obs = X_obs,
      beta_draws = fit_g_bagged$beta_draws,
      var_draws = fit_g_bagged$var_draws,
      n_rep = m_future_ppc,
      error = "gaussian",
      df = df,
      seed = seed + 1000L + b
    )
    dat_t <- generate_pseudo_sample_from_bagged_draws(
      X_obs = X_obs,
      beta_draws = fit_t_bagged$beta_draws,
      var_draws = fit_t_bagged$var_draws,
      n_rep = m_future_ppc,
      error = "student_t",
      df = df,
      seed = seed + 3000L + b
    )

    out <- list(
      g_rep = get_S(dat_g$y[seq_len(n_obs)], dat_g$X[seq_len(n_obs), , drop = FALSE]),
      g_center = get_S(c(y_obs, dat_g$y), rbind(X_obs, dat_g$X)),
      t_rep = get_S(dat_t$y[seq_len(n_obs)], dat_t$X[seq_len(n_obs), , drop = FALSE]),
      t_center = get_S(c(y_obs, dat_t$y), rbind(X_obs, dat_t$X))
    )

    if (include_bayes) {
      dat_b <- generate_pseudo_sample_from_bagged_draws(
        X_obs = X_obs,
        beta_draws = fit_b_bagged$beta_draws,
        var_draws = fit_b_bagged$var_draws,
        n_rep = m_future_ppc,
        error = "student_t",
        df = df,
        seed = seed + 5000L + b
      )
      out$b_rep <- get_S(dat_b$y[seq_len(n_obs)], dat_b$X[seq_len(n_obs), , drop = FALSE])
      out$b_center <- get_S(c(y_obs, dat_b$y), rbind(X_obs, dat_b$X))
    }

    out
  }

  reps <- safe_mc_lapply(
    X = seq_len(n_mc_pvalue),
    FUN = one_rep,
    n_cores = if (isTRUE(parallel)) n_cores else 1L
  )

  out_g <- pbppc_from_statistics(
    S_obs = S_obs,
    S_rep = vapply(reps, function(z) z$g_rep, numeric(1)),
    S_center = vapply(reps, function(z) z$g_center, numeric(1)),
    n_obs = n_obs,
    model_label = fit_g_bagged$display_label,
    functional_label = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    p_value_type = p_value_type,
    reference_label = fun_obj$reference_label,
    n_mc_pvalue = n_mc_pvalue,
    m_future_ppc = m_future_ppc
  )
  out_t <- pbppc_from_statistics(
    S_obs = S_obs,
    S_rep = vapply(reps, function(z) z$t_rep, numeric(1)),
    S_center = vapply(reps, function(z) z$t_center, numeric(1)),
    n_obs = n_obs,
    model_label = fit_t_bagged$display_label,
    functional_label = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    p_value_type = p_value_type,
    reference_label = fun_obj$reference_label,
    n_mc_pvalue = n_mc_pvalue,
    m_future_ppc = m_future_ppc
  )

  tab <- rbind(out_g$summary, out_t$summary)
  out <- list(
    functional = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    reference = fun_obj$reference_label,
    gaussian_mp = out_g,
    t_mp = out_t,
    bayes_t = NULL
  )

  if (include_bayes) {
    out_b <- pbppc_from_statistics(
      S_obs = S_obs,
      S_rep = vapply(reps, function(z) z$b_rep, numeric(1)),
      S_center = vapply(reps, function(z) z$b_center, numeric(1)),
      n_obs = n_obs,
      model_label = fit_b_bagged$display_label,
      functional_label = fun_obj$functional_label,
      diagnostic_class = fun_obj$diagnostic_class,
      p_value_type = p_value_type,
      reference_label = fun_obj$reference_label,
      n_mc_pvalue = n_mc_pvalue,
      m_future_ppc = m_future_ppc
    )
    out$bayes_t <- out_b
    tab <- rbind(tab, out_b$summary)
  }

  rownames(tab) <- NULL
  out$table <- tab
  out
}

run_bagged_pbppc_suite_once <- function(standard_app, bagged_fits,
                                        diagnostics = NULL,
                                        df = 5,
                                        q_tail = 0.995,
                                        functional_restarts = 10,
                                        parallel = FALSE,
                                        n_cores = 1L,
                                        seed0 = 4040) {
  prep <- standard_app$prepared
  y_obs <- prep$y
  X_obs <- prep$X
  if (is.null(diagnostics)) {
    diagnostics <- default_aids_ppc_diagnostics(
      n_obs = length(y_obs),
      q_tail = q_tail
    )
  }

  res_list <- vector("list", length(diagnostics))
  tab_list <- vector("list", length(diagnostics))
  for (j in seq_along(diagnostics)) {
    spec <- diagnostics[[j]]
    spec_full <- modifyList(
      list(
        y_obs = y_obs,
        X_obs = X_obs,
        fit_g_bagged = bagged_fits$gaussian,
        fit_t_bagged = bagged_fits$student_t,
        fit_b_bagged = bagged_fits$bayes,
        fit_g_reference = standard_app$gaussian_hybrid,
        df = df,
        functional_restarts = functional_restarts,
        parallel = parallel,
        n_cores = n_cores,
        seed = seed0 + 100000L * j
      ),
      spec
    )
    res_j <- do.call(run_bagged_pbppc_engines, spec_full)
    res_list[[j]] <- res_j
    tab_j <- res_j$table
    tab_j$diagnostic_id <- j
    tab_list[[j]] <- tab_j
  }

  table_all <- do.call(rbind, tab_list)
  rownames(table_all) <- NULL
  list(results = res_list, table = table_all, diagnostics = diagnostics)
}

make_bagged_ppc_summary_table <- function(tab) {
  keep <- tab[, c("diagnostic_class", "model", "p_value", "delta_mean", "delta_sd")]
  chi <- keep[keep$diagnostic_class == "chi_square_discrepancy", , drop = FALSE]
  tail <- keep[keep$diagnostic_class == "absolute_residual_tail", , drop = FALSE]

  model_map <- data.frame(
    model = c("DB-MGP-Gaussian-hybrid", "DB-MGP-Student-t-hybrid", "DB-Bayes-Student-t"),
    Model = c("bGPE", "bTPE", "bBTPE"),
    stringsAsFactors = FALSE
  )
  model_map <- model_map[model_map$model %in% unique(keep$model), , drop = FALSE]

  out <- data.frame(
    Model = model_map$Model,
    S_chi2_p_value = NA_real_,
    S_chi2_AvgDiff = NA_real_,
    S_chi2_StdDiff = NA_real_,
    S_tail_p_value = NA_real_,
    S_tail_AvgDiff = NA_real_,
    S_tail_StdDiff = NA_real_,
    row.names = NULL
  )

  for (i in seq_len(nrow(model_map))) {
    chi_i <- chi[chi$model == model_map$model[i], , drop = FALSE]
    tail_i <- tail[tail$model == model_map$model[i], , drop = FALSE]
    if (nrow(chi_i) != 1L || nrow(tail_i) != 1L) {
      stop("Could not construct the bagged PPC summary row for model: ", model_map$model[i])
    }
    out$S_chi2_p_value[i] <- chi_i$p_value
    out$S_chi2_AvgDiff[i] <- chi_i$delta_mean
    out$S_chi2_StdDiff[i] <- chi_i$delta_sd
    out$S_tail_p_value[i] <- tail_i$p_value
    out$S_tail_AvgDiff[i] <- tail_i$delta_mean
    out$S_tail_StdDiff[i] <- tail_i$delta_sd
  }

  out
}

plot_bagged_ppc_stat_density <- function(S_obs, S_list, labels,
                                         main = "",
                                         xlab = "replicate statistic") {
  values <- c(S_obs, unlist(S_list, use.names = FALSE))
  rng <- range(values)
  densities <- lapply(S_list, function(z) density(z, from = rng[1], to = rng[2]))
  ymax <- max(vapply(densities, function(z) max(z$y), numeric(1)))
  ltys <- seq_along(S_list)

  plot(
    densities[[1]],
    xlim = rng,
    ylim = c(0, ymax),
    lty = ltys[1],
    lwd = 2,
    main = if (is.null(main)) "" else main,
    xlab = xlab,
    ylab = "Density"
  )
  if (length(densities) > 1L) {
    for (i in 2:length(densities)) {
      lines(densities[[i]], lty = ltys[i], lwd = 2)
    }
  }
  abline(v = S_obs, lty = 2, lwd = 2)
  legend(
    "topright",
    legend = c(labels, "Observed"),
    lty = c(ltys, 2),
    lwd = 2,
    bty = "n"
  )
}

save_bagged_ppc_density_figure <- function(bagged_ppc,
                                           figure_dir,
                                           save_png = TRUE) {
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  chi2_res <- find_ppc_result_by_class(bagged_ppc, "chi_square_discrepancy")
  tail_res <- find_ppc_result_by_class(bagged_ppc, "absolute_residual_tail")
  include_bayes <- !is.null(tail_res$bayes_t)
  labels <- if (include_bayes) c("bGPE", "bTPE", "bBTPE") else c("bGPE", "bTPE")

  collect_reps <- function(res) {
    vals <- list(res$gaussian_mp$S_rep, res$t_mp$S_rep)
    if (!is.null(res$bayes_t)) {
      vals[[length(vals) + 1L]] <- res$bayes_t$S_rep
    }
    vals
  }

  save_plot_multi(
    plot_fun = function() {
      oldpar <- par(no.readonly = TRUE)
      on.exit(par(oldpar), add = TRUE)
      par(
        mfrow = c(1, 2),
        mar = c(4, 4, 1, 1),
        oma = c(0, 0, 0, 0),
        cex = 1.2,
        cex.lab = 1.2,
        cex.axis = 1.1
      )
      plot_bagged_ppc_stat_density(
        S_obs = tail_res$gaussian_mp$S_obs,
        S_list = collect_reps(tail_res),
        labels = labels,
        xlab = expression(S[tail])
      )
      plot_bagged_ppc_stat_density(
        S_obs = chi2_res$gaussian_mp$S_obs,
        S_list = collect_reps(chi2_res),
        labels = labels,
        xlab = expression(S[chi^2])
      )
    },
    stem = "double_bootstrap_ppc_replicate_density_two_panel",
    figure_dir = figure_dir,
    width = 12,
    height = 5,
    save_png = save_png,
    aliases = c("double_bootstrap_ppc_pe_density_two_panel")
  )

  invisible(TRUE)
}

save_bagged_ppc_outputs <- function(bagged_ppc,
                                    bagged_ppc_table_paper,
                                    bagged_ppc_summary,
                                    output_dir,
                                    save_png = TRUE) {
  write.csv(
    bagged_ppc$table,
    file = file.path(output_dir, "double_bootstrap_ppc_single_table_raw.csv"),
    row.names = FALSE
  )
  write.csv(
    bagged_ppc_table_paper,
    file = file.path(output_dir, "double_bootstrap_ppc_single_table_paper.csv"),
    row.names = FALSE
  )
  write.csv(
    bagged_ppc_summary,
    file = file.path(output_dir, "double_bootstrap_ppc_pe_summary.csv"),
    row.names = FALSE
  )
  writeLines(
    make_ppc_summary_latex(
      bagged_ppc_summary,
      caption = paste(
        "Double-bootstrap PPC-PE results by model. bGPE and bTPE denote",
        "bagged predictive engines based on Gaussian and Student-t regression models,",
        "respectively; bBTPE denotes the bagged Bayes posterior predictive engine."
      ),
      label = "tab:double-bootstrap-ppc-pe-aids"
    ),
    con = file.path(output_dir, "double_bootstrap_ppc_pe_summary.tex")
  )
  save_bagged_ppc_density_figure(
    bagged_ppc = bagged_ppc,
    figure_dir = file.path(output_dir, "figures"),
    save_png = save_png
  )
  invisible(TRUE)
}

setting_value <- function(settings, field, default = NA_character_) {
  if (is.null(settings) || !(field %in% settings$field)) {
    return(default)
  }
  settings$value[match(field, settings$field)]
}

add_bagged_ppc_to_saved_double_bootstrap_results <- function(
    results_file = file.path("double_bootstrap_results", "aids_double_bootstrap_calibration.rds"),
    output_dir = dirname(results_file),
    q_tail = 0.995,
    df_model = NULL,
    functional_restarts = NULL,
    parallel = FALSE,
    n_cores = recommended_ppc_cores(4L),
    seed0 = 42026,
    save_png = TRUE) {
  if (!file.exists(results_file)) {
    stop("Double-bootstrap results not found: ", results_file)
  }

  out <- readRDS(results_file)
  if (is.null(df_model)) {
    df_model <- as.numeric(setting_value(out$settings, "df_student_t", "5"))
  }
  if (is.null(functional_restarts)) {
    functional_restarts <- as.integer(setting_value(out$settings, "n_restart_boot", "10"))
  }

  n <- out$standard_app$prepared$n_obs
  bagged_ppc <- run_bagged_pbppc_suite_once(
    standard_app = out$standard_app,
    bagged_fits = out$bagged_fits,
    diagnostics = default_aids_ppc_diagnostics(n_obs = n, q_tail = q_tail),
    df = df_model,
    q_tail = q_tail,
    functional_restarts = functional_restarts,
    parallel = parallel,
    n_cores = n_cores,
    seed0 = seed0
  )
  bagged_ppc_table_paper <- format_ppc_table_for_paper(bagged_ppc$table)
  bagged_ppc_summary <- make_bagged_ppc_summary_table(bagged_ppc$table)

  out$bagged_ppc <- bagged_ppc
  out$bagged_ppc_table_paper <- bagged_ppc_table_paper
  out$bagged_ppc_summary <- bagged_ppc_summary
  saveRDS(out, file = results_file)
  save_bagged_ppc_outputs(
    bagged_ppc = bagged_ppc,
    bagged_ppc_table_paper = bagged_ppc_table_paper,
    bagged_ppc_summary = bagged_ppc_summary,
    output_dir = output_dir,
    save_png = save_png
  )

  invisible(out)
}

run_aids_double_bootstrap_calibration <- function(
    standard_results_file = file.path("application_results", "aids_application_results.rds"),
    output_dir = "double_bootstrap_results",
    B_bootstrap = 50,
    paths_per_boot = 20,
    bootstrap_size = NULL,
    df_model = 5,
    N_add_hybrid = 100,
    n_restart_boot = 10,
    include_bayes = TRUE,
    bayes_iter_sampling_boot = 1000,
    bayes_iter_warmup_boot = 500,
    bayes_chains_boot = 1,
    bayes_parallel_chains_boot = 1,
    parallel = FALSE,
    n_cores = recommended_ppc_cores(4L),
    mc_preschedule = FALSE,
    compute_bagged_ppc = TRUE,
    bagged_ppc_parallel = FALSE,
    bagged_ppc_n_cores = recommended_ppc_cores(4L),
    bagged_ppc_seed = 42026,
    seed = 92025) {
  if (!file.exists(standard_results_file)) {
    stop(
      "Standard application results not found: ", standard_results_file,
      ". Run Rscript run.R first, or pass a valid standard_results_file."
    )
  }

  standard_app <- readRDS(standard_results_file)
  prep <- standard_app$prepared
  n <- prep$n_obs
  if (is.null(bootstrap_size)) {
    bootstrap_size <- n
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  one_bootstrap <- function(b) {
    set.seed(seed + b)
    idx <- sample.int(n, size = bootstrap_size, replace = TRUE)
    prep_boot <- make_bootstrap_prepared(prep, idx)
    fit_bootstrap_predictive_engines(
      prep_boot = prep_boot,
      paths_per_boot = paths_per_boot,
      df_model = df_model,
      N_add_hybrid = N_add_hybrid,
      seed = seed + 10000L * b,
      n_restart_boot = n_restart_boot,
      include_bayes = include_bayes,
      bayes_iter_sampling_boot = bayes_iter_sampling_boot,
      bayes_iter_warmup_boot = bayes_iter_warmup_boot,
      bayes_chains_boot = bayes_chains_boot,
      bayes_parallel_chains_boot = bayes_parallel_chains_boot
    )
  }

  one_bootstrap_safe <- function(b) {
    tryCatch(
      one_bootstrap(b),
      error = function(e) make_bootstrap_error(b, e)
    )
  }

  if (isTRUE(include_bayes) && isTRUE(parallel)) {
    invisible(get_bayes_student_t_stan_model())
  }

  boot_fits <- safe_mc_lapply(
    X = seq_len(B_bootstrap),
    FUN = one_bootstrap_safe,
    n_cores = if (isTRUE(parallel)) n_cores else 1L,
    mc_preschedule = mc_preschedule
  )

  failed <- which(vapply(boot_fits, is_bootstrap_error, logical(1)))
  if (length(failed) > 0L && isTRUE(parallel)) {
    message(
      "Retrying failed bootstrap replication(s) sequentially: ",
      paste(failed, collapse = ", ")
    )
    boot_fits[failed] <- lapply(failed, one_bootstrap_safe)
  }
  validate_bootstrap_fits(boot_fits, include_bayes = include_bayes)

  bagged_fits <- list(
    gaussian = combine_bagged_fits(boot_fits, engine = "gaussian"),
    student_t = combine_bagged_fits(boot_fits, engine = "student_t"),
    bayes = combine_bagged_fits(boot_fits, engine = "bayes")
  )
  variance_components <- summarize_bagged_variance_components(boot_fits)

  coefficient_table <- summarize_bagged_coefficients(
    fit_g_bagged = bagged_fits$gaussian,
    fit_t_bagged = bagged_fits$student_t,
    fit_b_bagged = bagged_fits$bayes
  )
  coefficient_table_paper <- format_coefficient_table_for_paper(coefficient_table)

  coefficient_comparison_table <- NULL
  if (!is.null(standard_app$coefficient_table)) {
    standard_coef <- standard_app$coefficient_table
    standard_coef$calibration <- "PBP"
    bagged_coef <- coefficient_table
    bagged_coef$calibration <- "bPBP"
    coefficient_comparison_table <- rbind(standard_coef, bagged_coef)
    rownames(coefficient_comparison_table) <- NULL
  }

  bagged_ppc <- NULL
  bagged_ppc_table_paper <- NULL
  bagged_ppc_summary <- NULL
  if (isTRUE(compute_bagged_ppc)) {
    bagged_ppc <- run_bagged_pbppc_suite_once(
      standard_app = standard_app,
      bagged_fits = bagged_fits,
      diagnostics = default_aids_ppc_diagnostics(n_obs = n, q_tail = 0.995),
      df = df_model,
      q_tail = 0.995,
      functional_restarts = n_restart_boot,
      parallel = bagged_ppc_parallel,
      n_cores = bagged_ppc_n_cores,
      seed0 = bagged_ppc_seed
    )
    bagged_ppc_table_paper <- format_ppc_table_for_paper(bagged_ppc$table)
    bagged_ppc_summary <- make_bagged_ppc_summary_table(bagged_ppc$table)
  }

  settings <- data.frame(
    field = c(
      "method",
      "bootstrap_datasets",
      "paths_per_bootstrap",
      "total_bagged_draws_per_engine",
      "bootstrap_size",
      "df_student_t",
      "hybrid_future_steps",
      "n_restart_boot",
      "include_bayes",
      "bayes_iter_sampling_boot",
      "bayes_iter_warmup_boot",
      "parallel",
      "n_cores",
      "mc_preschedule",
      "compute_bagged_ppc",
      "bagged_ppc_parallel",
      "bagged_ppc_n_cores",
      "bagged_ppc_seed",
      "seed"
    ),
    value = c(
      "Double-bootstrap / bagged predictive Bayes posterior",
      as.character(B_bootstrap),
      as.character(paths_per_boot),
      as.character(B_bootstrap * paths_per_boot),
      as.character(bootstrap_size),
      as.character(df_model),
      as.character(N_add_hybrid),
      as.character(n_restart_boot),
      as.character(include_bayes),
      as.character(bayes_iter_sampling_boot),
      as.character(bayes_iter_warmup_boot),
      as.character(parallel),
      as.character(if (isTRUE(parallel)) n_cores else 1L),
      as.character(mc_preschedule),
      as.character(compute_bagged_ppc),
      as.character(bagged_ppc_parallel),
      as.character(if (isTRUE(bagged_ppc_parallel)) bagged_ppc_n_cores else 1L),
      as.character(bagged_ppc_seed),
      as.character(seed)
    ),
    stringsAsFactors = FALSE
  )

  out <- list(
    standard_results_file = standard_results_file,
    settings = settings,
    standard_app = standard_app,
    bagged_fits = bagged_fits,
    variance_components = variance_components,
    coefficient_table = coefficient_table,
    coefficient_table_paper = coefficient_table_paper,
    coefficient_comparison_table = coefficient_comparison_table,
    bagged_ppc = bagged_ppc,
    bagged_ppc_table_paper = bagged_ppc_table_paper,
    bagged_ppc_summary = bagged_ppc_summary
  )

  saveRDS(out, file = file.path(output_dir, "aids_double_bootstrap_calibration.rds"))
  write.csv(settings, file = file.path(output_dir, "double_bootstrap_settings.csv"),
            row.names = FALSE)
  write.csv(coefficient_table,
            file = file.path(output_dir, "double_bootstrap_coefficient_table_raw.csv"),
            row.names = FALSE)
  write.csv(coefficient_table_paper,
            file = file.path(output_dir, "double_bootstrap_coefficient_table_paper.csv"),
            row.names = FALSE)
  write.csv(variance_components,
            file = file.path(output_dir, "double_bootstrap_variance_decomposition.csv"),
            row.names = FALSE)
  if (!is.null(coefficient_comparison_table)) {
    write.csv(
      coefficient_comparison_table,
      file = file.path(output_dir, "double_bootstrap_standard_vs_bagged_coefficient_table.csv"),
      row.names = FALSE
    )
  }
  if (!is.null(bagged_ppc)) {
    save_bagged_ppc_outputs(
      bagged_ppc = bagged_ppc,
      bagged_ppc_table_paper = bagged_ppc_table_paper,
      bagged_ppc_summary = bagged_ppc_summary,
      output_dir = output_dir,
      save_png = TRUE
    )
  }

  save_double_bootstrap_contours(
    standard_app = standard_app,
    bagged_fits = bagged_fits,
    figure_dir = file.path(output_dir, "figures"),
    save_png = TRUE,
    include_bayes = include_bayes
  )

  invisible(out)
}

main <- function() {
  out <- run_aids_double_bootstrap_calibration(
    standard_results_file = file.path("application_results", "aids_application_results.rds"),
    output_dir = "double_bootstrap_results",
    B_bootstrap = 50,
    paths_per_boot = 20,
    include_bayes = TRUE,
    parallel = FALSE,
    n_cores = 1
  )

  print(out$settings)
  print(out$coefficient_table_paper)
  invisible(out)
}

if (sys.nframe() == 0) {
  main()
}
