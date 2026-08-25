## Bagged (double-bootstrap) simulation driver.
## Sourced by scripts/Run_Tables23_HighDimension.R.
parse_double_bootstrap_args <- function(args = commandArgs(trailingOnly = TRUE),
                                        repo_root = getwd()) {
  config <- parse_simulation_args(args, repo_root)
  if (!any(startsWith(args, "--model="))) config$model <- "all"
  config
}

bootstrap_observed_data <- function(data, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  idx <- sample.int(nrow(data$X), nrow(data$X), replace = TRUE)
  out <- data
  out$X <- data$X[idx, , drop = FALSE]
  out$y <- data$y[idx]
  out$bootstrap_indices <- idx
  out
}

posterior_summary_from_draws <- function(draws, beta_true, level = 0.95) {
  draws <- as.matrix(draws)
  samples <- t(draws)
  alpha <- 1 - level
  lower <- apply(samples, 2, stats::quantile, probs = alpha / 2, names = FALSE)
  upper <- apply(samples, 2, stats::quantile, probs = 1 - alpha / 2, names = FALSE)
  mean_est <- colMeans(samples)

  data.frame(
    index = seq_len(ncol(samples)),
    mean = mean_est,
    sd = apply(samples, 2, stats::sd),
    lower = lower,
    upper = upper,
    truth = beta_true,
    covered = lower <= beta_true & beta_true <= upper,
    bias = mean_est - beta_true,
    abs_bias = abs(mean_est - beta_true),
    interval_length = upper - lower
  )
}

summarize_parameter_sets <- function(summary, model, method, repeat_id,
                                     active_indices = NULL) {
  p <- nrow(summary)
  if (is.null(active_indices)) active_indices <- integer(0)
  active_indices <- sort(unique(active_indices[active_indices >= 1 & active_indices <= p]))
  inactive_indices <- setdiff(seq_len(p), active_indices)

  groups <- list(all = seq_len(p))
  if (length(active_indices) > 0L) groups$active <- active_indices
  if (length(inactive_indices) > 0L) groups$inactive <- inactive_indices

  rows <- lapply(names(groups), function(group_name) {
    idx <- groups[[group_name]]
    data.frame(
      model = model,
      method = method,
      parameter_set = group_name,
      repeat_id = repeat_id,
      n_params = length(idx),
      coverage = mean(summary$covered[idx]),
      signed_bias = mean(summary$bias[idx]),
      mean_abs_bias = mean(summary$abs_bias[idx]),
      mean_interval_length = mean(summary$interval_length[idx])
    )
  })
  do.call(rbind, rows)
}

aggregate_comparison_rows <- function(rows) {
  keys <- unique(rows[c("model", "method", "parameter_set")])
  out <- vector("list", nrow(keys))

  for (i in seq_len(nrow(keys))) {
    key <- keys[i, , drop = FALSE]
    idx <- rows$model == key$model &
      rows$method == key$method &
      rows$parameter_set == key$parameter_set
    sub <- rows[idx, , drop = FALSE]
    repeats <- length(unique(sub$repeat_id))

    out[[i]] <- data.frame(
      model = key$model,
      method = key$method,
      parameter_set = key$parameter_set,
      repeats = repeats,
      n_params = sub$n_params[1],
      coverage = mean(sub$coverage),
      coverage_se = stats::sd(sub$coverage) / sqrt(max(1, repeats)),
      signed_bias = mean(sub$signed_bias),
      signed_bias_se = stats::sd(sub$signed_bias) / sqrt(max(1, repeats)),
      mean_abs_bias = mean(sub$mean_abs_bias),
      mean_interval_length = mean(sub$mean_interval_length),
      interval_length_se = stats::sd(sub$mean_interval_length) /
        sqrt(max(1, repeats))
    )
  }

  result <- do.call(rbind, out)
  result$coverage_se[is.na(result$coverage_se)] <- 0
  result$signed_bias_se[is.na(result$signed_bias_se)] <- 0
  result$interval_length_se[is.na(result$interval_length_se)] <- 0
  result
}

paper_style_comparison_table <- function(aggregate_table) {
  cols <- intersect(c("scenario", "model", "method", "parameter_set", "repeats"),
                    names(aggregate_table))
  out <- aggregate_table[cols]
  out$coverage_bias <- sprintf("%.3f (%.3f)",
                               aggregate_table$coverage,
                               aggregate_table$signed_bias)
  out$mean_interval_length <- sprintf("%.3f",
                                      aggregate_table$mean_interval_length)
  out
}

write_double_bootstrap_outputs <- function(out_dir, model, ordinary_summary,
                                           bpbp_summary, raw_rows,
                                           aggregate_rows) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  ordinary_path <- file.path(out_dir, paste0(model, "_pbp_summary.csv"))
  bpbp_path <- file.path(out_dir, paste0(model, "_bpbp_summary.csv"))
  raw_path <- file.path(out_dir, paste0(model, "_comparison_raw.csv"))
  aggregate_path <- file.path(out_dir, paste0(model, "_comparison_table.csv"))
  paper_path <- file.path(out_dir, paste0(model, "_paper_style_table.csv"))

  if (!is.null(ordinary_summary)) {
    utils::write.csv(ordinary_summary, ordinary_path, row.names = FALSE)
    message("Wrote ", ordinary_path)
  }
  if (!is.null(bpbp_summary)) {
    utils::write.csv(bpbp_summary, bpbp_path, row.names = FALSE)
    message("Wrote ", bpbp_path)
  }
  utils::write.csv(raw_rows, raw_path, row.names = FALSE)
  utils::write.csv(aggregate_rows, aggregate_path, row.names = FALSE)
  utils::write.csv(paper_style_comparison_table(aggregate_rows), paper_path,
                   row.names = FALSE)
  message("Wrote ", aggregate_path)
  message("Wrote ", paper_path)

  c(pbp_summary = ordinary_path,
    bpbp_summary = bpbp_path,
    comparison_raw = raw_path,
    comparison_table = aggregate_path,
    paper_style_table = paper_path)
}

write_bpbp_draws_rds <- function(out_dir, model, object) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0(model, "_bpbp_draws.rds"))
  saveRDS(object, path)
  message("Wrote ", path)
  path
}

configure_parallel_environment <- function(config) {
  if (!config_bool(config, "single_thread_blas", TRUE)) return(invisible(FALSE))

  vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
  missing_vars <- vars[vapply(vars, function(var) {
    identical(Sys.getenv(var, unset = ""), "")
  }, logical(1))]
  if (length(missing_vars) > 0L) {
    do.call(Sys.setenv, as.list(stats::setNames(rep("1", length(missing_vars)),
                                                missing_vars)))
  }
  invisible(TRUE)
}

detect_logical_cores <- function() {
  cores <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (!is.na(cores) && cores >= 1L) return(as.integer(cores))

  read_core_count <- function(command, args) {
    executable <- if (grepl("/", command, fixed = TRUE)) {
      if (file.exists(command)) command else ""
    } else {
      Sys.which(command)
    }
    if (!nzchar(executable)) return(NA_integer_)

    value <- tryCatch(
      suppressWarnings(system2(executable, args, stdout = TRUE, stderr = FALSE)),
      error = function(e) character(0)
    )
    if (length(value) == 0L || !is.null(attr(value, "status"))) return(NA_integer_)
    out <- suppressWarnings(as.integer(value[1]))
    if (!is.na(out) && out >= 1L) out else NA_integer_
  }

  for (args in list("_NPROCESSORS_ONLN", "NPROCESSORS_ONLN")) {
    cores <- read_core_count("getconf", args)
    if (!is.na(cores)) return(cores)
  }

  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    for (command in c("/usr/sbin/sysctl", "sysctl")) {
      cores <- read_core_count(command, c("-n", "hw.logicalcpu"))
      if (!is.na(cores)) return(cores)
    }
  }

  cores <- read_core_count("nproc", character(0))
  if (!is.na(cores)) return(cores)

  1L
}

bpbp_worker_count <- function(config, n_tasks) {
  if (n_tasks <= 1L || !config_bool(config, "parallel", TRUE)) return(1L)

  requested <- tolower(as.character(config_value(config, "workers", "auto")))
  cores <- detect_logical_cores()
  reserve <- max(0L, config_int(config, "reserve_cores", 0L))
  max_workers <- config_int(config, "max_workers", cores)

  workers <- if (identical(requested, "auto")) {
    max(1L, cores - reserve)
  } else {
    suppressWarnings(as.integer(requested))
  }

  if (is.na(workers) || workers < 1L) workers <- 1L
  if (is.na(max_workers) || max_workers < 1L) max_workers <- workers
  min(workers, max_workers, n_tasks)
}

bpbp_chunk_size <- function(config, workers, repeats) {
  default_chunk <- max(1L, min(repeats, workers * 4L))
  chunk_size <- config_int(config, "chunk_size", default_chunk)
  if (is.na(chunk_size) || chunk_size < 1L) chunk_size <- default_chunk
  chunk_size <- min(chunk_size, repeats)
  if (workers > 1L && chunk_size < workers) {
    message("Increasing chunk_size from ", chunk_size, " to ", workers,
            " so all workers can be used.")
    chunk_size <- workers
  }
  chunk_size
}

parallel_lapply_lb <- function(X, FUN, workers, config) {
  if (workers <= 1L || length(X) <= 1L) return(lapply(X, FUN))

  configure_parallel_environment(config)

  if (!identical(.Platform$OS.type, "windows")) {
    out <- parallel::mclapply(
      X, FUN,
      mc.cores = workers,
      mc.preschedule = FALSE,
      mc.set.seed = FALSE
    )
  } else {
    cl <- parallel::makeCluster(workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, {
      Sys.setenv(
        OMP_NUM_THREADS = "1",
        OPENBLAS_NUM_THREADS = "1",
        MKL_NUM_THREADS = "1",
        VECLIB_MAXIMUM_THREADS = "1",
        NUMEXPR_NUM_THREADS = "1"
      )
      NULL
    })
    parallel::clusterExport(cl, ls(envir = globalenv()), envir = globalenv())
    out <- parallel::parLapplyLB(cl, X, FUN)
  }

  failed <- vapply(seq_along(out), function(i) {
    item <- out[[i]]
    is.null(item) ||
      inherits(item, "try-error") ||
      !is.list(item) ||
      is.null(item$repeat_id) ||
      length(item$repeat_id) != 1L ||
      is.na(item$repeat_id)
  }, logical(1))
  if (any(failed)) {
    failed_tasks <- paste(X[failed], collapse = ", ")
    first_failed <- out[[which(failed)[1L]]]
    detail <- if (inherits(first_failed, "try-error")) {
      conditionMessage(attr(first_failed, "condition"))
    } else {
      "worker returned no valid result; it was likely killed by memory pressure"
    }
    stop("Parallel bPBP worker failed for repeat(s): ", failed_tasks,
         ". Details: ", detail,
         ". Reduce --workers/--max_workers or lower p/repeats/n_boot.",
         call. = FALSE)
  }
  out
}

write_standard_bpbp_draws_rds <- function(out_dir, results) {
  has_draws <- vapply(results, function(x) !is.null(x$draws_object), logical(1))
  draw_results <- results[has_draws]
  if (length(draw_results) == 0L) return(character(0))

  model_names <- unique(vapply(draw_results, `[[`, character(1), "model"))
  paths <- character(0)

  for (model in model_names) {
    scenario_names <- names(draw_results)[
      vapply(draw_results, function(x) identical(x$model, model), logical(1))
    ]
    scenarios <- lapply(scenario_names, function(name) {
      object <- draw_results[[name]]$draws_object
      object$scenario <- name
      object
    })
    names(scenarios) <- scenario_names

    object <- list(
      model = model,
      scenarios = scenarios,
      comparison = do.call(rbind, lapply(
        scenario_names,
        function(name) draw_results[[name]]$comparison
      )),
      raw_comparison = do.call(rbind, lapply(
        scenario_names,
        function(name) draw_results[[name]]$raw_comparison
      ))
    )

    path <- write_bpbp_draws_rds(out_dir, model, object)
    paths <- c(paths, setNames(path, paste0(model, "_draws")))
  }

  paths
}

standard_bpbp_scenarios <- function(config) {
  user_params <- config$params
  standard_repeats <- config_int(config, "repeats", 200)
  standard_n_boot <- config_int(config, "n_boot", 50)
  standard_paths <- config_int(config, "paths_per_boot", 20)
  standard_B <- config_int(config, "B", standard_n_boot * standard_paths)

  scenario <- function(name, model, design = "normal", params = list()) {
    defaults <- c(
      list(
        repeats = standard_repeats,
        n_boot = standard_n_boot,
        paths_per_boot = standard_paths,
        B = standard_B,
        save_draws = "false",
        write_draws_rds = "false",
        save_data = "false",
        save_fit_objects = "false"
      ),
      params
    )
    list(
      name = name,
      model = model,
      config = simulation_config(
        model = model,
        profile = "paper",
        design = design,
        out_dir = file.path(config$out_dir, name),
        seed = config$seed,
        run_gibbs = FALSE,
        params = utils::modifyList(defaults, user_params)
      )
    )
  }

  list(
    scenario(
      "linear_normal",
      "linear",
      params = list(T = 10000)
    ),
    scenario(
      "gamma_kappa1",
      "gamma",
      params = list(T = 1000, kappa = 1)
    ),
    scenario(
      "gamma_kappa2",
      "gamma",
      params = list(T = 1000, kappa = 2)
    ),
    scenario(
      "logistic_uniform",
      "logistic",
      design = "uniform",
      params = list(T = 5000)
    ),
    scenario(
      "logistic_truncnorm",
      "logistic",
      design = "truncnorm",
      params = list(T = 5000)
    ),
    scenario(
      "studentt_kappa10",
      "studentt",
      params = list(T = 600, kappa = 10)
    )
  )
}

linear_data_from_config <- function(config, repeat_seed) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 400 else 20)
  n <- config_int(config, "n", if (paper) 100 else 50)
  generate_linear_data(
    p = p,
    n = n,
    sigma2 = config_num(config, "sigma2", 1),
    active_ratio = config_num(config, "active_ratio", 100 / 400),
    beta_scale = config_num(config, "beta_scale", 3),
    seed = repeat_seed
  )
}

gamma_data_from_config <- function(config, repeat_seed) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", 10)
  n <- config_int(config, "n", if (paper) 10 else 40)
  generate_gamma_loglink_data(
    n = n,
    p = p,
    p_star = config_int(config, "p_star", min(5, p)),
    rho = config_num(config, "rho", 0),
    lambda = config_num(config, "lambda", 1),
    shape = config_num(config, "shape", 1),
    seed = repeat_seed
  )
}

logistic_data_from_config <- function(config, repeat_seed) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 500 else 12)
  n <- config_int(config, "n", if (paper) 1000 else 100)
  generate_logistic_data(
    n = n,
    p = p,
    p_star = config_int(config, "p_star", min(5, p)),
    rho = config_num(config, "rho", 0),
    lambda = config_num(config, "lambda", 1),
    seed = repeat_seed
  )
}

studentt_data_from_config <- function(config, repeat_seed) {
  paper <- identical(config$profile, "paper")
  p <- config_int(config, "p", if (paper) 500 else 12)
  n <- config_int(config, "n", if (paper) 250 else 60)
  generate_robust_data(
    n = n,
    p = p,
    p_star = config_int(config, "p_star", min(5, p)),
    rho = config_num(config, "rho", 0),
    lambda = config_num(config, "lambda", 20),
    nu = config_num(config, "nu", 4),
    sigma = config_num(config, "sigma", 1),
    seed = repeat_seed
  )
}

draws_linear_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  sigma2 <- data$sigma2
  tau2 <- config_num(config, "tau2", 1)
  post <- analytical_ridge_posterior(data$X, data$y, sigma2, tau2)
  beta_n <- post$mu
  Sigma_n <- post$Sigma

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

draws_gamma_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  alpha <- data$shape
  lambda_ridge <- config_num(config, "lambda_ridge", 1)
  kappa <- config_num(config, "kappa",
                      if (identical(config$profile, "paper")) 2 else 1)

  beta_n <- map_gamma_loglink_optim(data$X, data$y, lambda_ridge, alpha)
  Sigma_n <- solve_spd(alpha * crossprod(data$X) + diag(lambda_ridge, p))

  set.seed(seed + 1L)
  X_samples <- matrix(stats::rnorm(p * T_steps, sd = kappa), nrow = p)
  paths <- precompute_paths_gamma(Sigma_n, T_steps, alpha, X_samples)
  draws <- recursive_update_gamma(
    matrix(beta_n, nrow = p, ncol = n_paths),
    paths$Sigma_x_samples,
    paths$quad_samples,
    alpha,
    seed = seed + 2L
  )
  list(draws = draws, beta_n = beta_n, Sigma_n = Sigma_n)
}

draws_logistic_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  n <- nrow(data$X)
  lambda_ridge <- config_num(config, "lambda_ridge", 0.1)
  beta_n <- map_logistic_ridge_optim(data$X, data$y, lambda_ridge)
  prob <- stats::plogis(as.vector(data$X %*% beta_n))
  w_n <- prob * (1 - prob)
  Sigma_n <- solve_spd(compute_stable_XtDX(data$X, w_n) +
                         diag(lambda_ridge * n, p))

  set.seed(seed + 1L)
  if (identical(config$design, "truncnorm")) {
    X_samples <- matrix(sample_truncnorm(p * T_steps, sd = 3), nrow = p)
  } else {
    X_samples <- matrix(stats::runif(p * T_steps, -6, 6), nrow = p)
  }

  paths <- precompute_paths_logistic(beta_n, X_samples, Sigma_n)
  draws <- recursive_update_logistic(
    matrix(beta_n, nrow = p, ncol = n_paths),
    X_samples,
    paths$Sigma_x_samples,
    paths$xSigma_x_samples,
    paths$w_samples,
    paths$log_w_samples,
    seed = seed + 2L
  )
  list(draws = draws, beta_n = beta_n, Sigma_n = Sigma_n)
}

draws_studentt_from_data <- function(data, config, n_paths, T_steps, seed) {
  p <- ncol(data$X)
  paper <- identical(config$profile, "paper")
  nu <- data$nu
  sigma <- data$sigma
  v0_grid <- config_num_vector(
    config,
    "v0_grid",
    if (paper) {
      c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    } else {
      c(0.5, 0.2, 0.1)
    }
  )
  v0 <- tail(v0_grid, 1)
  v1 <- config_num(config, "v1", 1)
  w <- config_num(config, "w", 0.1)
  kappa <- config_num(config, "kappa", if (paper) 10 else 3)

  beta_n <- continuation_em_studentt_spikeslab(
    data$X, data$y, v0_grid, v1, w, nu, sigma,
    max_iter = config_int(config, "em_max_iter", if (paper) 500 else 100)
  )
  cnu <- (nu + 1) / ((nu + 3) * sigma^2)
  D_n <- regularized_fisher(beta_n, 1 - w, v0, v1)
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

pool_bpbp_draws <- function(data, config, fit_draws_fun, n_boot,
                            paths_per_boot, T_steps, seed) {
  p <- ncol(data$X)
  pooled <- matrix(NA_real_, p, n_boot * paths_per_boot)
  beta_init <- matrix(NA_real_, p, n_boot)
  col_start <- 1L

  for (b in seq_len(n_boot)) {
    boot_data <- bootstrap_observed_data(data, seed = seed + 1000L * b)
    fit <- fit_draws_fun(
      boot_data, config,
      n_paths = paths_per_boot,
      T_steps = T_steps,
      seed = seed + 2000L * b
    )
    cols <- col_start:(col_start + paths_per_boot - 1L)
    pooled[, cols] <- fit$draws
    beta_init[, b] <- fit$beta_n
    col_start <- col_start + paths_per_boot
  }

  list(draws = pooled, bootstrap_beta_init = beta_init)
}

run_bpbp_repeat <- function(r, config, model, data_fun, fit_draws_fun,
                            total_paths, n_boot, paths_per_boot, T_steps,
                            level, save_draws, save_data, save_fit_objects) {
  repeat_seed <- config$seed + 100000L * (r - 1L)
  data <- data_fun(config, repeat_seed)
  active_indices <- data$active_indices

  ordinary <- fit_draws_fun(
    data, config,
    n_paths = total_paths,
    T_steps = T_steps,
    seed = repeat_seed + 10L
  )
  bpbp <- pool_bpbp_draws(
    data, config, fit_draws_fun,
    n_boot = n_boot,
    paths_per_boot = paths_per_boot,
    T_steps = T_steps,
    seed = repeat_seed + 20L
  )

  ordinary_summary <- posterior_summary_from_draws(
    ordinary$draws, data$beta, level = level
  )
  bpbp_summary <- posterior_summary_from_draws(
    bpbp$draws, data$beta, level = level
  )

  raw_rows <- rbind(
    summarize_parameter_sets(ordinary_summary, model, "PBP", r, active_indices),
    summarize_parameter_sets(bpbp_summary, model, "bPBP", r, active_indices)
  )

  draw_object <- NULL
  if (save_draws) {
    draw_object <- list(
      repeat_id = r,
      data_seed = repeat_seed,
      ordinary_draws = ordinary$draws,
      bpbp_draws = bpbp$draws,
      bootstrap_beta_init = bpbp$bootstrap_beta_init,
      ordinary_beta_n = ordinary$beta_n,
      beta_true = data$beta,
      active_indices = active_indices,
      ordinary_summary = ordinary_summary,
      bpbp_summary = bpbp_summary
    )
    if (save_fit_objects) {
      draw_object$ordinary_Sigma_n <- ordinary$Sigma_n
    }
    if (save_data) {
      draw_object$data <- data
    }
  }

  list(
    repeat_id = r,
    raw_rows = raw_rows,
    draw_object = draw_object,
    ordinary_summary = ordinary_summary,
    bpbp_summary = bpbp_summary
  )
}

run_one_bpbp_model <- function(config, model, data_fun, fit_draws_fun) {
  config$model <- model
  config <- normalize_simulation_config(config)
  paper <- identical(config$profile, "paper")
  repeats <- config_int(config, "repeats", 1)
  n_boot <- config_int(config, "n_boot", if (paper) 50 else 10)
  paths_per_boot <- config_int(config, "paths_per_boot", if (paper) 20 else 20)
  total_paths <- config_int(config, "B", n_boot * paths_per_boot)
  level <- config_num(config, "level", 0.95)

  default_T <- switch(
    model,
    linear = if (paper) 10000 else 200,
    gamma = if (paper) 1000 else 200,
    logistic = if (paper) 5000 else 200,
    studentt = {
      p_default <- config_int(config, "p", if (paper) 500 else 12)
      if (paper) p_default + 100 else 200
    }
  )
  T_steps <- config_int(config, "T", default_T)

  save_draws <- config_bool(config, "save_draws", TRUE)
  save_data <- config_bool(config, "save_data", !paper)
  save_fit_objects <- config_bool(config, "save_fit_objects", !paper)
  workers <- bpbp_worker_count(config, repeats)
  chunk_size <- bpbp_chunk_size(config, workers, repeats)

  message("bPBP model=", model,
          "; repeats=", repeats,
          "; n_boot=", n_boot,
          "; paths_per_boot=", paths_per_boot,
          "; B=", total_paths,
          "; T=", T_steps,
          "; workers=", workers,
          "; chunk_size=", chunk_size,
          "; save_draws=", save_draws)

  raw_rows <- list()
  repeat_draw_objects <- vector("list", repeats)
  rds_files <- character(0)
  draws_object <- NULL
  last_ordinary_summary <- NULL
  last_bpbp_summary <- NULL

  repeat_ids <- seq_len(repeats)
  chunks <- split(repeat_ids, ceiling(seq_along(repeat_ids) / chunk_size))
  completed <- 0L

  for (chunk in chunks) {
    chunk_results <- parallel_lapply_lb(
      chunk,
      function(r) {
        run_bpbp_repeat(
          r, config, model, data_fun, fit_draws_fun,
          total_paths = total_paths,
          n_boot = n_boot,
          paths_per_boot = paths_per_boot,
          T_steps = T_steps,
          level = level,
          save_draws = save_draws,
          save_data = save_data,
          save_fit_objects = save_fit_objects
        )
      },
      workers = min(workers, length(chunk)),
      config = config
    )

    for (result in chunk_results) {
      raw_rows[[length(raw_rows) + 1L]] <- result$raw_rows
      if (save_draws) {
        repeat_draw_objects[[result$repeat_id]] <- result$draw_object
      }
      last_ordinary_summary <- result$ordinary_summary
      last_bpbp_summary <- result$bpbp_summary
    }

    completed <- completed + length(chunk)
    message("[", model, "] completed repeats ", completed, " / ", repeats,
            " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  }

  raw_rows <- do.call(rbind, raw_rows)
  aggregate_rows <- aggregate_comparison_rows(raw_rows)
  files <- write_double_bootstrap_outputs(
    config$out_dir,
    model,
    last_ordinary_summary,
    last_bpbp_summary,
    raw_rows,
    aggregate_rows
  )
  if (save_draws) {
    draws_object <- list(
      model = model,
      config = config,
      repeats = repeat_draw_objects,
      raw_comparison = raw_rows,
      comparison = aggregate_rows
    )
    if (config_bool(config, "write_draws_rds", TRUE)) {
      rds_files <- write_bpbp_draws_rds(
        config$out_dir,
        model,
        draws_object
      )
    }
  }

  list(model = model,
       raw_comparison = raw_rows,
       comparison = aggregate_rows,
       ordinary_summary = last_ordinary_summary,
       bpbp_summary = last_bpbp_summary,
       draws_object = draws_object,
       rds_files = rds_files,
       files = c(files, rds_files))
}

run_double_bootstrap <- function(config = list()) {
  config <- normalize_simulation_config(config)
  configure_parallel_environment(config)
  dir.create(config$out_dir, recursive = TRUE, showWarnings = FALSE)

  model_specs <- list(
    linear = list(data = linear_data_from_config,
                  fit = draws_linear_from_data),
    gamma = list(data = gamma_data_from_config,
                 fit = draws_gamma_from_data),
    logistic = list(data = logistic_data_from_config,
                    fit = draws_logistic_from_data),
    studentt = list(data = studentt_data_from_config,
                    fit = draws_studentt_from_data)
  )

  if (identical(config$model, "standard")) {
    scenarios <- standard_bpbp_scenarios(config)
    message("Standard double-bootstrap bPBP run with ",
            length(scenarios), " scenarios; out_dir=", config$out_dir)
    results <- list()
    pending_draw_results <- list()
    standard_rds_files <- character(0)

    for (i in seq_along(scenarios)) {
      item <- scenarios[[i]]
      spec <- model_specs[[item$model]]
      res <- run_one_bpbp_model(item$config, item$model, spec$data, spec$fit)
      res$scenario <- item$name
      res$comparison$scenario <- item$name
      res$raw_comparison$scenario <- item$name

      if (!is.null(res$draws_object)) {
        pending_draw_results[[item$name]] <- res
      }

      stored_res <- res
      stored_res$draws_object <- NULL
      results[[item$name]] <- stored_res

      next_model <- if (i < length(scenarios)) scenarios[[i + 1L]]$model else NA_character_
      if (length(pending_draw_results) > 0L &&
          (i == length(scenarios) || !identical(item$model, next_model))) {
        standard_rds_files <- c(
          standard_rds_files,
          write_standard_bpbp_draws_rds(config$out_dir, pending_draw_results)
        )
        pending_draw_results <- list()
      }
    }

    combined <- do.call(rbind, lapply(results, function(x) x$comparison))
    combined <- combined[c("scenario", setdiff(names(combined), "scenario"))]
    raw <- do.call(rbind, lapply(results, function(x) x$raw_comparison))
    raw <- raw[c("scenario", setdiff(names(raw), "scenario"))]

    combined_path <- file.path(config$out_dir, "standard_comparison_table.csv")
    raw_path <- file.path(config$out_dir, "standard_comparison_raw.csv")
    paper_path <- file.path(config$out_dir, "standard_paper_style_table.csv")
    utils::write.csv(combined, combined_path, row.names = FALSE)
    utils::write.csv(raw, raw_path, row.names = FALSE)
    utils::write.csv(paper_style_comparison_table(combined), paper_path,
                     row.names = FALSE)
    message("Wrote ", combined_path)
    message("Wrote ", raw_path)
    message("Wrote ", paper_path)

    results$standard_comparison <- combined
    results$standard_files <- c(comparison_table = combined_path,
                                comparison_raw = raw_path,
                                paper_style_table = paper_path,
                                standard_rds_files)
    return(results)
  }

  models <- if (identical(config$model, "all")) {
    names(model_specs)
  } else {
    config$model
  }

  unknown <- setdiff(models, names(model_specs))
  if (length(unknown) > 0L) {
    stop("Unknown model. Use standard, all, linear, gamma, logistic, or studentt.",
         call. = FALSE)
  }

  message("Double-bootstrap bPBP run: model=", paste(models, collapse = ", "),
          "; profile=", config$profile,
          "; design=", config$design)

  results <- lapply(models, function(model) {
    spec <- model_specs[[model]]
    run_one_bpbp_model(config, model, spec$data, spec$fit)
  })
  names(results) <- models

  if (length(results) == 1L) {
    results[[1L]]
  } else {
    combined <- do.call(rbind, lapply(results, function(x) x$comparison))
    combined_path <- file.path(config$out_dir, "all_models_comparison_table.csv")
    paper_path <- file.path(config$out_dir, "all_models_paper_style_table.csv")
    utils::write.csv(combined, combined_path, row.names = FALSE)
    utils::write.csv(paper_style_comparison_table(combined), paper_path,
                     row.names = FALSE)
    message("Wrote ", combined_path)
    message("Wrote ", paper_path)
    results$all_models_comparison <- combined
    results$all_models_files <- c(comparison_table = combined_path,
                                  paper_style_table = paper_path)
    results
  }
}
