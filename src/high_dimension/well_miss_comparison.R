## Table assembly: well- vs mis-specified comparison of coverage, bias, length.
## Sourced by scripts/Run_Tables23_HighDimension.R.
comparison_model_ids <- function() {
  c(
    "linear_ridge_well",
    "linear_sas_well",
    "linear_ridge_miss",
    "linear_sas_miss",
    "studentt_ridge_well",
    "studentt_sas_well",
    "studentt_ridge_miss",
    "studentt_sas_miss"
  )
}

comparison_model_spec <- function(model_id) {
  specs <- list(
    linear_ridge_well = list(family = "linear", prior = "ridge", dgp = "well"),
    linear_sas_well = list(family = "linear", prior = "sas", dgp = "well"),
    linear_ridge_miss = list(family = "linear", prior = "ridge", dgp = "miss"),
    linear_sas_miss = list(family = "linear", prior = "sas", dgp = "miss"),
    studentt_ridge_well = list(family = "studentt", prior = "ridge", dgp = "well"),
    studentt_sas_well = list(family = "studentt", prior = "sas", dgp = "well"),
    studentt_ridge_miss = list(family = "studentt", prior = "ridge", dgp = "miss"),
    studentt_sas_miss = list(family = "studentt", prior = "sas", dgp = "miss")
  )
  spec <- specs[[model_id]]
  if (is.null(spec)) stop("Unknown comparison model: ", model_id, call. = FALSE)
  spec$model_id <- model_id
  spec
}

selected_comparison_models <- function(requested) {
  requested <- tolower(requested)
  ids <- comparison_model_ids()
  selected <- switch(
    requested,
    all = ids,
    linear = ids[startsWith(ids, "linear_")],
    studentt = ids[startsWith(ids, "studentt_")],
    well = ids[endsWith(ids, "_well")],
    miss = ids[endsWith(ids, "_miss")],
    misspecified = ids[endsWith(ids, "_miss")],
    ridge = ids[grepl("_ridge_", ids, fixed = TRUE)],
    sas = ids[grepl("_sas_", ids, fixed = TRUE)],
    if (requested %in% ids) requested else NULL
  )
  if (is.null(selected)) {
    stop(
      "Unknown --model. Use all, linear, studentt, well, miss, ridge, sas, or one of: ",
      paste(ids, collapse = ", "),
      call. = FALSE
    )
  }
  selected
}

family_config_value <- function(config, family, name, default = NULL) {
  family_name <- paste0(family, "_", name)
  if (!is.null(config$params[[family_name]])) return(config$params[[family_name]])
  if (!is.null(config$params[[name]])) return(config$params[[name]])
  default
}

family_int <- function(config, family, name, default) {
  as.integer(family_config_value(config, family, name, default))
}

family_num <- function(config, family, name, default) {
  as.numeric(family_config_value(config, family, name, default))
}

family_bool <- function(config, family, name, default = FALSE) {
  value <- family_config_value(config, family, name, if (default) "true" else "false")
  tolower(as.character(value)) %in% c("1", "true", "yes", "y")
}

family_text <- function(config, family, name, default = "") {
  as.character(family_config_value(config, family, name, default))
}

comparison_profile_defaults <- function(profile, family) {
  smoke <- identical(tolower(profile), "smoke")
  if (identical(family, "linear")) {
    if (smoke) {
      return(list(repeats = 2L, n_boot = 2L, paths_per_boot = 2L,
                  B = 4L, T = 8L, n = 40L, p = 20L, p_star = 5L))
    }
    return(list(repeats = 80L, n_boot = 50L, paths_per_boot = 20L,
                B = 1000L, T = 5000L, n = 250L, p = 200L, p_star = 50L))
  }

  if (smoke) {
    return(list(repeats = 2L, n_boot = 2L, paths_per_boot = 2L,
                B = 4L, T = 8L, n = 50L, p = 20L, p_star = 3L))
  }
  list(repeats = 80L, n_boot = 50L, paths_per_boot = 20L,
       B = 1000L, T = 300L, n = 250L, p = 200L, p_star = 5L)
}

comparison_variant_config <- function(config, model_id) {
  spec <- comparison_model_spec(model_id)
  family <- spec$family
  defaults <- comparison_profile_defaults(config$profile, family)

  n_boot <- family_int(config, family, "n_boot", defaults$n_boot)
  paths_per_boot <- family_int(
    config, family, "paths_per_boot", defaults$paths_per_boot
  )
  p <- family_int(config, family, "p", defaults$p)
  p_star <- family_int(config, family, "p_star", defaults$p_star)
  if (p_star < 1L || p_star > p) {
    stop(family, "_p_star must be between 1 and ", p, ".", call. = FALSE)
  }

  common <- list(
    repeats = family_int(config, family, "repeats", defaults$repeats),
    n_boot = n_boot,
    paths_per_boot = paths_per_boot,
    B = family_int(config, family, "B", n_boot * paths_per_boot),
    T = family_int(config, family, "T", defaults$T),
    n = family_int(config, family, "n", defaults$n),
    p = p,
    p_star = p_star,
    hetero_multiplier = family_num(config, family, "hetero_multiplier", 4),
    hetero_threshold = family_num(config, family, "hetero_threshold", 1),
    hetero_index = family_int(config, family, "hetero_index", 1L),
    em_max_iter = family_int(config, family, "em_max_iter", 500L),
    em_tol = family_num(config, family, "em_tol", 1e-6),
    save_draws = family_text(
      config, family, "save_draws",
      if (identical(config$profile, "smoke")) "false" else "true"
    ),
    write_draws_rds = family_text(
      config, family, "write_draws_rds",
      family_text(
        config, family, "save_draws",
        if (identical(config$profile, "smoke")) "false" else "true"
      )
    ),
    save_data = family_text(config, family, "save_data", "false"),
    save_fit_objects = family_text(config, family, "save_fit_objects", "false"),
    workers = family_text(config, family, "workers", "auto"),
    max_workers = family_int(
      config, family, "max_workers",
      if (identical(family, "linear")) 12L else 10L
    ),
    reserve_cores = family_int(config, family, "reserve_cores", 0L),
    parallel = family_text(config, family, "parallel", "true"),
    single_thread_blas = family_text(
      config, family, "single_thread_blas", "true"
    ),
    scenario = spec$dgp,
    prior = spec$prior
  )

  if (identical(family, "linear")) {
    common <- utils::modifyList(common, list(
      sigma2 = family_num(config, family, "sigma2", 1),
      tau2 = family_num(config, family, "tau2", 1),
      beta_scale = family_num(config, family, "beta_scale", 3),
      active_ratio = p_star / p,
      v1 = family_num(config, family, "v1", 9),
      w = family_num(config, family, "w", p_star / p),
      v0_grid = family_text(
        config, family, "v0_grid",
        "1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005"
      )
    ))
  } else {
    common <- utils::modifyList(common, list(
      rho = family_num(config, family, "rho", 0),
      lambda = family_num(config, family, "lambda", 20),
      nu = family_num(config, family, "nu", 4),
      sigma = family_num(config, family, "sigma", 1),
      kappa = family_num(config, family, "kappa", 10),
      tau2 = family_num(config, family, "tau2", 1),
      v1 = family_num(config, family, "v1", 1),
      w = family_num(config, family, "w", 0.1),
      v0_grid = family_text(
        config, family, "v0_grid",
        "0.5,0.1,0.05,0.01,0.005,0.001,0.0005"
      )
    ))
  }

  simulation_config(
    model = model_id,
    profile = config$profile,
    design = if (identical(family, "linear")) config$design else "normal",
    out_dir = file.path(config$out_dir, model_id),
    seed = config$seed,
    run_gibbs = FALSE,
    params = common
  )
}

generate_linear_comparison_base <- function(config, repeat_seed) {
  p <- config_int(config, "p", 200L)
  n <- config_int(config, "n", 100L)
  p_star <- config_int(config, "p_star", 50L)
  beta_scale <- config_num(config, "beta_scale", 3)
  sigma2 <- config_num(config, "sigma2", 1)

  set.seed(repeat_seed)
  z <- sample(c(-1, 1), n, replace = TRUE)
  X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  X <- standardize_columns(X + z)
  active_indices <- sort(sample.int(p, p_star, replace = FALSE))
  beta <- numeric(p)
  beta[active_indices] <- beta_scale * stats::rnorm(p_star)
  epsilon <- stats::rnorm(n)

  list(
    X = X,
    beta = beta,
    active_indices = active_indices,
    epsilon = epsilon,
    sigma2 = sigma2
  )
}

linear_comparison_data_from_config <- function(config, repeat_seed) {
  base <- generate_linear_comparison_base(config, repeat_seed)
  scenario <- tolower(as.character(config_value(config, "scenario", "well")))
  location <- as.vector(base$X %*% base$beta)
  n <- nrow(base$X)

  scale_factor <- rep(1, n)
  if (identical(scenario, "miss")) {
    multiplier <- config_num(config, "hetero_multiplier", 4)
    threshold <- config_num(config, "hetero_threshold", 1)
    hetero_index <- config_int(config, "hetero_index", 1L)
    hetero_index <- max(1L, min(ncol(base$X), hetero_index))
    scale_factor <- sqrt(
      1 + multiplier * as.numeric(abs(base$X[, hetero_index]) > threshold)
    )
  }

  y <- location + sqrt(base$sigma2) * scale_factor * base$epsilon
  list(
    X = base$X,
    y = y,
    beta = base$beta,
    sigma2 = base$sigma2,
    active_indices = base$active_indices,
    dgp = if (identical(scenario, "miss")) {
      "heteroskedastic_gaussian"
    } else {
      "homoskedastic_gaussian"
    },
    conditional_scale = sqrt(base$sigma2) * scale_factor
  )
}

generate_studentt_comparison_base <- function(config, repeat_seed) {
  p <- config_int(config, "p", 200L)
  n <- config_int(config, "n", 250L)
  p_star <- config_int(config, "p_star", 5L)
  rho <- config_num(config, "rho", 0)
  lambda <- config_num(config, "lambda", 20)
  nu <- config_num(config, "nu", 4)
  sigma <- config_num(config, "sigma", 1)

  set.seed(repeat_seed)
  Sigma <- equicorr_matrix(p, rho)
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  X <- standardize_columns(X)
  active_indices <- seq_len(p_star)
  beta <- numeric(p)
  beta[active_indices] <- sample(c(-1, 1), p_star, replace = TRUE) *
    sqrt(lambda / p_star)
  epsilon <- stats::rt(n, df = nu)

  list(
    X = X,
    beta = beta,
    active_indices = active_indices,
    epsilon = epsilon,
    nu = nu,
    sigma = sigma
  )
}

studentt_comparison_data_from_config <- function(config, repeat_seed) {
  base <- generate_studentt_comparison_base(config, repeat_seed)
  scenario <- tolower(as.character(config_value(config, "scenario", "well")))
  location <- as.vector(base$X %*% base$beta)
  n <- nrow(base$X)

  scale_factor <- rep(1, n)
  if (identical(scenario, "miss")) {
    multiplier <- config_num(config, "hetero_multiplier", 4)
    threshold <- config_num(config, "hetero_threshold", 1)
    hetero_index <- config_int(config, "hetero_index", 1L)
    hetero_index <- max(1L, min(ncol(base$X), hetero_index))
    scale_factor <- sqrt(
      1 + multiplier * as.numeric(abs(base$X[, hetero_index]) > threshold)
    )
  }

  y <- location + base$sigma * scale_factor * base$epsilon
  list(
    X = base$X,
    y = y,
    beta = base$beta,
    active_indices = base$active_indices,
    nu = base$nu,
    sigma = base$sigma,
    dgp = if (identical(scenario, "miss")) {
      "heteroskedastic_studentt"
    } else {
      "homoskedastic_studentt"
    },
    conditional_scale = base$sigma * scale_factor
  )
}

map_linear_sas_em <- function(X, y, sigma2, v0, v1, w,
                              beta_init = NULL, max_iter = 500L,
                              tol = 1e-6) {
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)
  beta <- if (is.null(beta_init)) numeric(p) else as.vector(beta_init)
  w <- min(max(w, 1e-6), 1 - 1e-6)
  XtX_scaled <- crossprod(X) / sigma2
  Xty_scaled <- as.vector(crossprod(X, y) / sigma2)

  for (iter in seq_len(max_iter)) {
    beta_old <- beta
    slab_log <- log(w) + stats::dnorm(beta, 0, sqrt(v1), log = TRUE)
    spike_log <- log1p(-w) + stats::dnorm(beta, 0, sqrt(v0), log = TRUE)
    slab_prob <- 1 / (1 + exp(spike_log - slab_log))
    prior_precision <- slab_prob / v1 + (1 - slab_prob) / v0
    beta <- solve_spd(XtX_scaled + diag(prior_precision, p), Xty_scaled)
    if (sqrt(sum((beta - beta_old)^2)) < tol) return(beta)
  }
  beta
}

continuation_linear_sas_em <- function(X, y, sigma2, v0_grid, v1, w,
                                       max_iter = 500L, tol = 1e-6) {
  beta <- numeric(ncol(X))
  for (v0 in v0_grid) {
    beta <- map_linear_sas_em(
      X, y, sigma2, v0, v1, w,
      beta_init = beta, max_iter = max_iter, tol = tol
    )
  }
  beta
}

draws_linear_sas_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  sigma2 <- data$sigma2
  v0_grid <- config_num_vector(
    config, "v0_grid", c(1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
  )
  v0 <- tail(v0_grid, 1L)
  v1 <- config_num(config, "v1", 9)
  w <- config_num(config, "w", config_num(config, "active_ratio", 0.25))
  beta_n <- continuation_linear_sas_em(
    data$X, data$y, sigma2, v0_grid, v1, w,
    max_iter = config_int(config, "em_max_iter", 500L),
    tol = config_num(config, "em_tol", 1e-6)
  )

  D_n <- regularized_fisher(beta_n, 1 - w, v0, v1)
  Sigma_n <- solve_spd(crossprod(data$X) / sigma2 + D_n)
  set.seed(seed + 1L)
  if (identical(config$design, "empirical")) {
    X_samples <- sample_X_empirical(data$X, T_steps)
  } else if (identical(config$design, "mixture")) {
    X_samples <- sample_linear_mixture(p, T_steps, seed = seed + 1L)
  } else {
    X_samples <- matrix(stats::rnorm(p * T_steps), nrow = p)
  }
  paths <- precompute_paths_linear(X_samples, Sigma_n, sigma2)
  draws <- recursive_update_ridge_exact_bayes(
    matrix(beta_n, nrow = p, ncol = n_paths),
    sigma2,
    paths$Sigma_x_samples,
    paths$xSigma_x_samples,
    seed = seed + 2L
  )
  list(draws = draws, beta_n = beta_n, Sigma_n = Sigma_n)
}

map_studentt_ridge_em <- function(X, y, nu, sigma, tau2,
                                  beta_init = NULL, max_iter = 500L,
                                  tol = 1e-6) {
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)
  if (tau2 <= 0) stop("tau2 must be positive.", call. = FALSE)

  if (is.null(beta_init)) {
    beta <- solve_spd(
      crossprod(X) / sigma^2 + diag(1 / tau2, p),
      as.vector(crossprod(X, y) / sigma^2)
    )
  } else {
    beta <- as.vector(beta_init)
  }

  for (iter in seq_len(max_iter)) {
    beta_old <- beta
    resid <- y - as.vector(X %*% beta)
    lambda_exp <- (nu + 1) / (nu + resid^2 / sigma^2)
    row_weight <- lambda_exp / sigma^2
    Xw <- sweep(X, 1, sqrt(row_weight), "*")
    precision <- crossprod(Xw) + diag(1 / tau2, p)
    score <- as.vector(crossprod(X, row_weight * y))
    beta <- solve_spd(precision, score)
    if (sqrt(sum((beta - beta_old)^2)) < tol) return(beta)
  }
  beta
}

draws_studentt_ridge_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  nu <- data$nu
  sigma <- data$sigma
  tau2 <- config_num(config, "tau2", 1)
  kappa <- config_num(config, "kappa", 10)
  beta_n <- map_studentt_ridge_em(
    data$X, data$y, nu, sigma, tau2,
    max_iter = config_int(config, "em_max_iter", 500L),
    tol = config_num(config, "em_tol", 1e-6)
  )

  cnu <- (nu + 1) / ((nu + 3) * sigma^2)
  D_n <- diag(1 / tau2, p)
  Sigma_n <- solve_spd(cnu * crossprod(data$X) + D_n)
  set.seed(seed + 1L)
  X_samples <- matrix(stats::rnorm(p * T_steps, sd = kappa), nrow = p)
  paths <- precompute_paths_studentt(Sigma_n, T_steps, nu, sigma, X_samples)
  draws <- recursive_update_studentt(
    matrix(beta_n, nrow = p, ncol = n_paths),
    paths$Sigma_x_samples,
    paths$quad_samples,
    nu,
    sigma,
    seed = seed + 2L
  )
  list(draws = draws, beta_n = beta_n, Sigma_n = Sigma_n)
}

comparison_data_function <- function(spec) {
  if (identical(spec$family, "linear")) {
    linear_comparison_data_from_config
  } else {
    studentt_comparison_data_from_config
  }
}

comparison_fit_function <- function(spec) {
  if (identical(spec$family, "linear") && identical(spec$prior, "ridge")) {
    return(draws_linear_from_data)
  }
  if (identical(spec$family, "linear")) return(draws_linear_sas_from_data)
  if (identical(spec$prior, "ridge")) return(draws_studentt_ridge_from_data)
  draws_studentt_from_data
}

comparison_settings_row <- function(config, model_id) {
  spec <- comparison_model_spec(model_id)
  cfg <- comparison_variant_config(config, model_id)
  p <- config_int(cfg, "p", NA_integer_)
  p_star <- config_int(cfg, "p_star", NA_integer_)
  row <- data.frame(
    model = model_id,
    family = spec$family,
    prior = spec$prior,
    dgp = spec$dgp,
    profile = config$profile,
    repeats = config_int(cfg, "repeats", NA_integer_),
    n_boot = config_int(cfg, "n_boot", NA_integer_),
    paths_per_boot = config_int(cfg, "paths_per_boot", NA_integer_),
    ordinary_B = config_int(cfg, "B", NA_integer_),
    pooled_bpbp_draws = config_int(cfg, "n_boot", NA_integer_) *
      config_int(cfg, "paths_per_boot", NA_integer_),
    T = config_int(cfg, "T", NA_integer_),
    n = config_int(cfg, "n", NA_integer_),
    p = p,
    p_star = p_star,
    active_ratio = p_star / p,
    hetero_multiplier = config_num(cfg, "hetero_multiplier", NA_real_),
    hetero_threshold = config_num(cfg, "hetero_threshold", NA_real_),
    hetero_index = config_int(cfg, "hetero_index", NA_integer_),
    ridge_tau2 = if (identical(spec$prior, "ridge")) {
      config_num(cfg, "tau2", NA_real_)
    } else {
      NA_real_
    },
    max_workers = config_int(cfg, "max_workers", NA_integer_),
    save_draws = config_bool(cfg, "save_draws", FALSE),
    stringsAsFactors = FALSE
  )
  if (identical(spec$family, "linear")) {
    row$error_family <- "Gaussian"
    row$observed_design <- "standardized 0.5 N(-1,I) + 0.5 N(1,I)"
    row$predictive_design <- if (identical(cfg$design, "normal")) {
      "N(0,I)"
    } else {
      cfg$design
    }
    row$signal <- sprintf(
      "random active beta_j = %.8g * eta_j, eta_j iid N(0,1)",
      config_num(cfg, "beta_scale", 3)
    )
    row$beta_scale <- config_num(cfg, "beta_scale", NA_real_)
    row$sigma2 <- config_num(cfg, "sigma2", NA_real_)
    row$rho <- NA_real_
    row$lambda <- NA_real_
    row$sigma <- NA_real_
    row$nu <- NA_real_
    row$kappa <- NA_real_
  } else {
    row$error_family <- "Student-t"
    row$observed_design <- "standardized equicorrelated Gaussian"
    row$predictive_design <- "N(0,kappa^2 I)"
    row$signal <- sprintf(
      paste0(
        "first p_star active; GitHub Rademacher signs * ",
        "sqrt(%.8g/p_star)"
      ),
      config_num(cfg, "lambda", 20)
    )
    row$beta_scale <- NA_real_
    row$sigma2 <- NA_real_
    row$rho <- config_num(cfg, "rho", NA_real_)
    row$lambda <- config_num(cfg, "lambda", NA_real_)
    row$sigma <- config_num(cfg, "sigma", NA_real_)
    row$nu <- config_num(cfg, "nu", NA_real_)
    row$kappa <- config_num(cfg, "kappa", NA_real_)
  }
  row$sas_v1 <- if (identical(spec$prior, "sas")) {
    config_num(cfg, "v1", NA_real_)
  } else {
    NA_real_
  }
  row$sas_w <- if (identical(spec$prior, "sas")) {
    config_num(cfg, "w", NA_real_)
  } else {
    NA_real_
  }
  row$sas_v0_grid <- if (identical(spec$prior, "sas")) {
    as.character(config_value(cfg, "v0_grid", ""))
  } else {
    NA_character_
  }
  row
}

same_settings <- function(x, y) {
  rownames(x) <- NULL
  rownames(y) <- NULL
  identical(x, y)
}

augment_comparison <- function(table, spec) {
  table$scenario <- spec$model_id
  table$family <- spec$family
  table$prior <- spec$prior
  table$dgp <- spec$dgp
  first <- c("scenario", "family", "prior", "dgp")
  table[c(first, setdiff(names(table), first))]
}

well_miss_paper_style_table <- function(comparison) {
  out <- comparison[c(
    "family", "prior", "dgp", "method", "parameter_set", "repeats"
  )]
  out$coverage_bias <- sprintf(
    "%.3f (%.3f)", comparison$coverage, comparison$signed_bias
  )
  out$mean_interval_length <- sprintf(
    "%.3f", comparison$mean_interval_length
  )
  out
}

write_combined_comparison_outputs <- function(completed, out_dir) {
  if (length(completed) == 0L) return(invisible(NULL))
  comparison <- do.call(rbind, lapply(completed, `[[`, "comparison"))
  raw <- do.call(rbind, lapply(completed, `[[`, "raw_comparison"))
  rownames(comparison) <- NULL
  rownames(raw) <- NULL

  utils::write.csv(
    comparison,
    file.path(out_dir, "all_comparison_table.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    raw,
    file.path(out_dir, "all_comparison_raw.csv"),
    row.names = FALSE
  )
  utils::write.csv(
    well_miss_paper_style_table(comparison),
    file.path(out_dir, "all_paper_style_table.csv"),
    row.names = FALSE
  )

  for (family in unique(comparison$family)) {
    take <- comparison$family == family
    utils::write.csv(
      comparison[take, , drop = FALSE],
      file.path(out_dir, paste0(family, "_comparison_table.csv")),
      row.names = FALSE
    )
    take_raw <- raw$family == family
    utils::write.csv(
      raw[take_raw, , drop = FALSE],
      file.path(out_dir, paste0(family, "_comparison_raw.csv")),
      row.names = FALSE
    )
  }
  invisible(list(comparison = comparison, raw = raw))
}

run_well_miss_prior_comparison <- function(config = list()) {
  config <- normalize_simulation_config(config)
  configure_parallel_environment(config)
  dir.create(config$out_dir, recursive = TRUE, showWarnings = FALSE)
  model_ids <- selected_comparison_models(config$model)
  settings <- do.call(rbind, lapply(
    model_ids, function(id) comparison_settings_row(config, id)
  ))
  rownames(settings) <- NULL
  settings_path <- file.path(config$out_dir, "scenario_settings.csv")
  utils::write.csv(settings, settings_path, row.names = FALSE)
  message("Wrote ", settings_path)
  capture.output(
    utils::sessionInfo(),
    file = file.path(config$out_dir, "session_info.txt")
  )

  resume <- config_bool(config, "resume", TRUE)
  overwrite <- config_bool(config, "overwrite", FALSE)
  completed <- list()

  for (model_id in model_ids) {
    spec <- comparison_model_spec(model_id)
    cfg <- comparison_variant_config(config, model_id)
    setting <- settings[settings$model == model_id, , drop = FALSE]
    marker <- file.path(cfg$out_dir, "scenario_complete.rds")

    if (resume && !overwrite && file.exists(marker)) {
      cached <- readRDS(marker)
      if (!same_settings(cached$settings, setting)) {
        stop(
          "Existing completed result has different settings: ", marker,
          ". Use a new --out directory or pass --overwrite=true.",
          call. = FALSE
        )
      }
      message("Skipping completed scenario=", model_id)
      completed[[model_id]] <- cached
      write_combined_comparison_outputs(completed, config$out_dir)
      next
    }

    message("Running scenario=", model_id)
    result <- run_one_bpbp_model(
      cfg,
      model_id,
      comparison_data_function(spec),
      comparison_fit_function(spec)
    )
    cached <- list(
      model = model_id,
      settings = setting,
      comparison = augment_comparison(result$comparison, spec),
      raw_comparison = augment_comparison(result$raw_comparison, spec),
      files = result$files,
      rds_files = result$rds_files,
      completed_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
    )
    saveRDS(cached, marker)
    message("Wrote ", marker)
    completed[[model_id]] <- cached
    write_combined_comparison_outputs(completed, config$out_dir)
    rm(result)
    invisible(gc(FALSE))
  }

  combined <- write_combined_comparison_outputs(completed, config$out_dir)
  list(settings = settings, completed = completed, combined = combined)
}

validate_comparison_pairing <- function(config, repeat_seed = 123L) {
  config <- normalize_simulation_config(config)
  rows <- list()
  i <- 0L
  for (family in c("linear", "studentt")) {
    for (dgp in c("well", "miss")) {
      ridge_id <- paste(family, "ridge", dgp, sep = "_")
      sas_id <- paste(family, "sas", dgp, sep = "_")
      ridge_cfg <- comparison_variant_config(config, ridge_id)
      sas_cfg <- comparison_variant_config(config, sas_id)
      data_fun <- comparison_data_function(comparison_model_spec(ridge_id))
      ridge_data <- data_fun(ridge_cfg, repeat_seed)
      sas_data <- data_fun(sas_cfg, repeat_seed)
      i <- i + 1L
      rows[[i]] <- data.frame(
        family = family,
        dgp = dgp,
        same_X = identical(ridge_data$X, sas_data$X),
        same_y = identical(ridge_data$y, sas_data$y),
        same_beta = identical(ridge_data$beta, sas_data$beta),
        stringsAsFactors = FALSE
      )
    }
  }
  result <- do.call(rbind, rows)
  if (!all(result$same_X & result$same_y & result$same_beta)) {
    stop("Ridge and SAS are not using identical datasets.", call. = FALSE)
  }
  result
}
