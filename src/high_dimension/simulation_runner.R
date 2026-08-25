## Simulation driver for the ordinary MGP runs.
## Sourced by scripts/Run_Tables23_HighDimension.R.
simulation_config <- function(model = "studentt",
                              profile = "smoke",
                              design = "normal",
                              out_dir = file.path(getwd(), "results"),
                              seed = 123,
                              run_gibbs = FALSE,
                              params = list()) {
  if (is.null(params)) params <- list()
  list(
    model = tolower(model),
    profile = tolower(profile),
    design = tolower(design),
    out_dir = out_dir,
    seed = as.integer(seed),
    run_gibbs = isTRUE(run_gibbs),
    params = params
  )
}

normalize_simulation_config <- function(config = list()) {
  defaults <- simulation_config()
  if (is.null(config)) config <- list()
  for (name in names(defaults)) {
    if (is.null(config[[name]])) config[[name]] <- defaults[[name]]
  }
  if (is.null(config$params)) config$params <- list()
  config$model <- tolower(as.character(config$model))
  config$profile <- tolower(as.character(config$profile))
  config$design <- tolower(as.character(config$design))
  config$seed <- as.integer(config$seed)
  config$run_gibbs <- isTRUE(config$run_gibbs)
  config
}

parse_simulation_args <- function(args = commandArgs(trailingOnly = TRUE),
                                  repo_root = getwd()) {
  values <- list()
  for (arg in args) {
    if (!startsWith(arg, "--") || !grepl("=", arg, fixed = TRUE)) next
    key <- sub("^--", "", sub("=.*$", "", arg))
    value <- sub("^[^=]*=", "", arg)
    values[[key]] <- value
  }

  arg_value <- function(name, default = NULL) {
    if (is.null(values[[name]])) default else values[[name]]
  }
  arg_logical <- function(name, default = FALSE) {
    value <- tolower(arg_value(name, if (default) "true" else "false"))
    value %in% c("1", "true", "yes", "y")
  }

  known <- c("model", "profile", "design", "out", "seed", "gibbs")
  params <- values[setdiff(names(values), known)]
  if (!is.null(params[["obs"]]) && is.null(params[["n"]])) {
    params[["n"]] <- params[["obs"]]
  }

  simulation_config(
    model = arg_value("model", "studentt"),
    profile = arg_value("profile", "smoke"),
    design = arg_value("design", "normal"),
    out_dir = arg_value("out", file.path(repo_root, "results")),
    seed = as.integer(arg_value("seed", "123")),
    run_gibbs = arg_logical("gibbs", FALSE),
    params = params
  )
}

config_value <- function(config, name, default = NULL) {
  if (!is.null(config$params[[name]])) config$params[[name]] else default
}

config_int <- function(config, name, default) {
  as.integer(config_value(config, name, default))
}

config_num <- function(config, name, default) {
  as.numeric(config_value(config, name, default))
}

config_bool <- function(config, name, default = FALSE) {
  value <- config_value(config, name, if (default) "true" else "false")
  tolower(as.character(value)) %in% c("1", "true", "yes", "y")
}

config_num_vector <- function(config, name, default) {
  value <- config_value(config, name, NULL)
  if (is.null(value)) return(default)
  as.numeric(strsplit(as.character(value), ",", fixed = TRUE)[[1]])
}

sample_truncnorm <- function(n, sd = 3, lower = -6, upper = 6) {
  out <- numeric(n)
  filled <- 0L
  while (filled < n) {
    draw <- stats::rnorm((n - filled) * 2L, sd = sd)
    draw <- draw[draw >= lower & draw <= upper]
    take <- min(length(draw), n - filled)
    if (take > 0L) {
      out[(filled + 1L):(filled + take)] <- draw[seq_len(take)]
      filled <- filled + take
    }
  }
  out
}

summarize_draws <- function(samples_by_col, beta_true = NULL) {
  samples_by_col <- as.matrix(samples_by_col)
  out <- data.frame(
    index = seq_len(ncol(samples_by_col)),
    mean = colMeans(samples_by_col),
    sd = apply(samples_by_col, 2, stats::sd),
    q025 = apply(samples_by_col, 2, stats::quantile, 0.025, names = FALSE),
    q975 = apply(samples_by_col, 2, stats::quantile, 0.975, names = FALSE)
  )
  if (!is.null(beta_true)) out$truth <- beta_true
  out
}

write_simulation_summary <- function(out_dir, name, summary) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0(name, "_summary.csv"))
  utils::write.csv(summary, path, row.names = FALSE)
  message("Wrote ", path)
  path
}

write_simulation_rds <- function(out_dir, name, object) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0(name, "_draws.rds"))
  saveRDS(object, path)
  message("Wrote ", path)
  path
}

run_linear_simulation <- function(config) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 400 else 20)
  n <- config_int(config, "n", if (paper) 100 else 50)
  T_steps <- config_int(config, "T", if (paper) 10000 else 200)
  B <- config_int(config, "B", if (paper) 5000 else 500)
  sigma2 <- config_num(config, "sigma2", 1)
  tau2 <- config_num(config, "tau2", 1)
  active_ratio <- config_num(config, "active_ratio", 100 / 400)
  beta_scale <- config_num(config, "beta_scale", 3)

  data <- generate_linear_data(
    p = p, n = n, sigma2 = sigma2, active_ratio = active_ratio,
    beta_scale = beta_scale, seed = config$seed
  )
  X <- data$X
  y <- data$y
  post <- analytical_ridge_posterior(X, y, sigma2, tau2)
  beta_n <- post$mu
  Sigma_n <- post$Sigma

  set.seed(config$seed + 1L)
  if (identical(config$design, "empirical")) {
    X_samples <- sample_X_empirical(X, T_steps)
  } else if (identical(config$design, "mixture")) {
    X_samples <- sample_linear_mixture(p, T_steps, seed = config$seed + 1L)
  } else {
    X_samples <- matrix(stats::rnorm(p * T_steps), nrow = p)
  }

  paths <- precompute_paths_linear(X_samples, Sigma_n, sigma2)
  draws <- recursive_update_ridge_exact_bayes(
    matrix(beta_n, nrow = p, ncol = B),
    sigma2,
    paths$Sigma_x_samples,
    paths$xSigma_x_samples,
    seed = config$seed + 2L
  )

  summary <- summarize_draws(t(draws), data$beta)
  summary$analytic_mean <- post$mu
  summary$analytic_sd <- sqrt(diag(post$Sigma))
  files <- c(linear = write_simulation_summary(config$out_dir, "linear", summary))
  rds_path <- NULL
  if (config_bool(config, "save_draws", TRUE)) {
    rds_path <- write_simulation_rds(config$out_dir, "linear", list(
      model = "linear",
      config = config,
      draws = draws,
      summary = summary,
      beta_true = data$beta,
      active_indices = data$active_indices,
      beta_n = beta_n,
      Sigma_n = Sigma_n,
      data = data
    ))
    files <- c(files, linear_draws = rds_path)
  }

  list(model = "linear", summaries = list(linear = summary), files = files,
       draws = draws, data = data, beta_n = beta_n, Sigma_n = Sigma_n,
       rds_path = rds_path)
}

run_gamma_simulation <- function(config) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", 10)
  n <- config_int(config, "n", if (paper) 10 else 40)
  p_star <- config_int(config, "p_star", min(5, p))
  T_steps <- config_int(config, "T", if (paper) 1000 else 200)
  B <- config_int(config, "B", if (paper) 5000 else 500)
  kappa <- config_num(config, "kappa", if (paper) 2 else 1)
  rho <- config_num(config, "rho", 0)
  lambda <- config_num(config, "lambda", 1)
  shape <- config_num(config, "shape", 1)
  lambda_ridge <- config_num(config, "lambda_ridge", 1)

  data <- generate_gamma_loglink_data(
    n = n, p = p, p_star = p_star, rho = rho, lambda = lambda,
    shape = shape, seed = config$seed
  )
  X <- data$X
  y <- data$y
  alpha <- data$shape
  beta_n <- map_gamma_loglink_optim(X, y, lambda_ridge, alpha)
  Sigma_n <- solve_spd(alpha * crossprod(X) + diag(lambda_ridge, p))

  set.seed(config$seed + 1L)
  X_samples <- matrix(stats::rnorm(p * T_steps, sd = kappa), nrow = p)
  paths <- precompute_paths_gamma(Sigma_n, T_steps, alpha, X_samples)
  draws <- recursive_update_gamma(
    matrix(beta_n, nrow = p, ncol = B),
    paths$Sigma_x_samples,
    paths$quad_samples,
    alpha,
    seed = config$seed + 2L
  )

  summary <- summarize_draws(t(draws), data$beta)
  files <- c(gamma = write_simulation_summary(config$out_dir, "gamma", summary))
  rds_path <- NULL
  if (config_bool(config, "save_draws", TRUE)) {
    rds_path <- write_simulation_rds(config$out_dir, "gamma", list(
      model = "gamma",
      config = config,
      draws = draws,
      summary = summary,
      beta_true = data$beta,
      active_indices = data$active_indices,
      beta_n = beta_n,
      Sigma_n = Sigma_n,
      data = data
    ))
    files <- c(files, gamma_draws = rds_path)
  }

  list(model = "gamma", summaries = list(gamma = summary), files = files,
       draws = draws, data = data, beta_n = beta_n, Sigma_n = Sigma_n,
       rds_path = rds_path)
}

run_logistic_simulation <- function(config) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 500 else 12)
  n <- config_int(config, "n", if (paper) 1000 else 100)
  p_star <- config_int(config, "p_star", min(5, p))
  T_steps <- config_int(config, "T", if (paper) 5000 else 200)
  B <- config_int(config, "B", if (paper) 5000 else 500)
  rho <- config_num(config, "rho", 0)
  lambda <- config_num(config, "lambda", 1)
  lambda_ridge <- config_num(config, "lambda_ridge", 0.1)

  data <- generate_logistic_data(
    n = n, p = p, p_star = p_star, rho = rho, lambda = lambda,
    seed = config$seed
  )
  X <- data$X
  y <- data$y
  beta_n <- map_logistic_ridge_optim(X, y, lambda_ridge)
  prob <- stats::plogis(as.vector(X %*% beta_n))
  w_n <- prob * (1 - prob)
  Sigma_n <- solve_spd(compute_stable_XtDX(X, w_n) +
                         diag(lambda_ridge * n, p))

  set.seed(config$seed + 1L)
  if (identical(config$design, "truncnorm")) {
    X_samples <- matrix(sample_truncnorm(p * T_steps, sd = 3), nrow = p)
  } else {
    X_samples <- matrix(stats::runif(p * T_steps, -6, 6), nrow = p)
  }

  paths <- precompute_paths_logistic(beta_n, X_samples, Sigma_n)
  draws <- recursive_update_logistic(
    matrix(beta_n, nrow = p, ncol = B),
    X_samples,
    paths$Sigma_x_samples,
    paths$xSigma_x_samples,
    paths$w_samples,
    paths$log_w_samples,
    seed = config$seed + 2L
  )
  pmp_summary <- summarize_draws(t(draws), data$beta)
  summaries <- list(logistic_pmp = pmp_summary)
  gibbs <- NULL
  files <- c(logistic_pmp = write_simulation_summary(
    config$out_dir, "logistic_pmp", pmp_summary
  ))

  if (isTRUE(config$run_gibbs)) {
    n_iter <- config_int(config, "n_iter", if (paper) 12000 else 200)
    burnin <- config_int(config, "burnin", if (paper) 6000 else 100)
    pg_trunc <- config_int(config, "pg_trunc", if (paper) 200 else 80)
    gibbs <- bayes_logistic_ridge_gibbs(
      X, y, tau2 = 1 / (lambda_ridge * n),
      n_iter = n_iter, burnin = burnin, beta_init = beta_n,
      seed = config$seed + 3L, pg_trunc = pg_trunc
    )
    summaries$logistic_gibbs <- summarize_draws(gibbs$beta_samples, data$beta)
    files <- c(files, logistic_gibbs = write_simulation_summary(
      config$out_dir, "logistic_gibbs", summaries$logistic_gibbs
    ))
  }
  rds_path <- NULL
  if (config_bool(config, "save_draws", TRUE)) {
    rds_path <- write_simulation_rds(config$out_dir, "logistic", list(
      model = "logistic",
      config = config,
      pmp_draws = draws,
      gibbs_samples = if (is.null(gibbs)) NULL else gibbs$beta_samples,
      summaries = summaries,
      beta_true = data$beta,
      active_indices = data$active_indices,
      beta_n = beta_n,
      Sigma_n = Sigma_n,
      data = data
    ))
    files <- c(files, logistic_draws = rds_path)
  }

  list(model = "logistic", summaries = summaries, files = files,
       draws = draws, data = data, beta_n = beta_n, Sigma_n = Sigma_n,
       rds_path = rds_path)
}

run_studentt_simulation <- function(config) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 500 else 12)
  n <- config_int(config, "n", if (paper) 250 else 60)
  p_star <- config_int(config, "p_star", min(5, p))
  T_steps <- config_int(config, "T", if (paper) p + 100 else 200)
  B <- config_int(config, "B", if (paper) 5000 else 500)
  kappa <- config_num(config, "kappa", if (paper) 10 else 3)
  rho <- config_num(config, "rho", 0)
  lambda <- config_num(config, "lambda", 20)
  nu <- config_num(config, "nu", 4)
  sigma <- config_num(config, "sigma", 1)
  v0_grid <- config_num_vector(
    config,
    "v0_grid",
    if (paper) {
      c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    } else {
      c(0.5, 0.2, 0.1)
    }
  )
  v1 <- config_num(config, "v1", 1)
  w <- config_num(config, "w", 0.1)

  data <- generate_robust_data(
    n = n, p = p, p_star = p_star, rho = rho, lambda = lambda,
    nu = nu, sigma = sigma, seed = config$seed
  )
  X <- data$X
  y <- data$y
  v0 <- tail(v0_grid, 1)
  beta_n <- continuation_em_studentt_spikeslab(
    X, y, v0_grid, v1, w, nu, sigma,
    max_iter = config_int(config, "em_max_iter", if (paper) 500 else 100)
  )
  cnu <- (nu + 1) / ((nu + 3) * sigma^2)
  D_n <- regularized_fisher(beta_n, 1 - w, v0, v1)
  Sigma_n <- solve_spd(cnu * crossprod(X) + D_n)

  set.seed(config$seed + 1L)
  X_samples <- matrix(stats::rnorm(p * T_steps, sd = kappa), nrow = p)
  paths <- precompute_paths_studentt(Sigma_n, T_steps, nu, sigma, X_samples)
  draws <- recursive_update_studentt(
    matrix(beta_n, nrow = p, ncol = B),
    paths$Sigma_x_samples,
    paths$quad_samples,
    nu,
    sigma,
    seed = config$seed + 2L
  )

  pmp_summary <- summarize_draws(t(draws), data$beta)
  summaries <- list(studentt_pmp = pmp_summary)
  gibbs <- NULL
  files <- c(studentt_pmp = write_simulation_summary(
    config$out_dir, "studentt_pmp", pmp_summary
  ))

  if (isTRUE(config$run_gibbs)) {
    burnin <- config_int(config, "burnin", if (paper) 10000 else 100)
    n_keep <- config_int(config, "n_keep", if (paper) 25000 else 200)
    gibbs <- gibbs_studentt_spikeslab(
      X, y, v0, v1, w, nu, sigma,
      burnin = burnin, n_keep = n_keep, seed = config$seed + 3L
    )
    summaries$studentt_gibbs <- summarize_draws(gibbs$beta_samples, data$beta)
    files <- c(files, studentt_gibbs = write_simulation_summary(
      config$out_dir, "studentt_gibbs", summaries$studentt_gibbs
    ))
  }
  rds_path <- NULL
  if (config_bool(config, "save_draws", TRUE)) {
    rds_path <- write_simulation_rds(config$out_dir, "studentt", list(
      model = "studentt",
      config = config,
      pmp_draws = draws,
      gibbs_samples = if (is.null(gibbs)) NULL else gibbs$beta_samples,
      summaries = summaries,
      beta_true = data$beta,
      active_indices = data$active_indices,
      beta_n = beta_n,
      Sigma_n = Sigma_n,
      data = data
    ))
    files <- c(files, studentt_draws = rds_path)
  }

  list(model = "studentt", summaries = summaries, files = files,
       draws = draws, data = data, beta_n = beta_n, Sigma_n = Sigma_n,
       rds_path = rds_path)
}

run_simulation <- function(config = list()) {
  config <- normalize_simulation_config(config)
  dir.create(config$out_dir, recursive = TRUE, showWarnings = FALSE)
  message("Model: ", config$model,
          "; profile: ", config$profile,
          "; design: ", config$design)

  switch(
    config$model,
    linear = run_linear_simulation(config),
    gamma = run_gamma_simulation(config),
    logistic = run_logistic_simulation(config),
    studentt = run_studentt_simulation(config),
    stop("Unknown model. Use linear, gamma, logistic, or studentt.", call. = FALSE)
  )
}
