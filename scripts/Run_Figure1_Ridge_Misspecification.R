#!/usr/bin/env Rscript
## Figure 1: posterior of a single regression coefficient under increasing
## heteroskedastic misspecification, MGP (top) vs bMGP (bottom) half-eye panels.
## Usage: Rscript Run_Figure1_Ridge_Misspecification.R

need_packages <- function(pkgs, what) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop(what, " requires the package(s): ", paste(missing, collapse = ", "),
         "\n  install.packages(c(",
         paste0("\"", missing, "\"", collapse = ", "), "))", call. = FALSE)
  }
  invisible(TRUE)
}

standardize_columns <- function(X) {
  X <- as.matrix(X)
  centers <- colMeans(X)
  scales <- apply(X, 2, stats::sd)
  scales[scales == 0] <- 1
  sweep(sweep(X, 2, centers, "-"), 2, scales, "/")
}

chol_with_jitter <- function(A, max_tries = 7L) {
  S <- (A + t(A)) / 2
  scale <- max(1, mean(abs(diag(S))))
  for (i in seq_len(max_tries)) {
    jitter <- if (i == 1L) 0 else scale * 10^(-12 + i)
    R <- tryCatch(chol(S + diag(jitter, nrow(S))), error = function(e) NULL)
    if (!is.null(R)) return(R)
  }
  chol(S + diag(scale * 1e-4, nrow(S)))
}

solve_spd <- function(A, b = NULL) {
  R <- chol_with_jitter(A)
  if (is.null(b)) {
    chol2inv(R)
  } else {
    backsolve(R, forwardsolve(t(R), b))
  }
}

interval_length <- function(lower, upper) {
  upper - lower
}

sample_linear_mixture <- function(p, T, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z <- sample(c(-1, 1), T, replace = TRUE)
  X_samples <- matrix(stats::rnorm(p * T), nrow = p, ncol = T)
  sweep(X_samples, 2, z, "+")
}

analytical_ridge_posterior <- function(X, y, sigma2, tau2) {
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)

  Sigma_inv <- crossprod(X) / sigma2 + diag(1 / tau2, p)
  Sigma <- solve_spd(Sigma_inv)
  mu <- as.vector(Sigma %*% (crossprod(X, y) / sigma2))

  list(mu = mu, Sigma = Sigma)
}

precompute_paths_linear <- function(X_samples, Sigma_n, sigma2) {
  X_samples <- as.matrix(X_samples)
  p <- nrow(X_samples)
  T <- ncol(X_samples)
  inv_sigma2 <- 1 / sigma2

  Sigma_x_samples <- matrix(NA_real_, p, T)
  xSigma_x_samples <- numeric(T)
  Sigma <- as.matrix(Sigma_n)

  for (t in seq_len(T)) {
    x <- X_samples[, t]
    v <- as.vector(Sigma %*% x)
    s <- sum(x * v)
    denom <- 1 + inv_sigma2 * s

    Sigma <- Sigma - (inv_sigma2 / denom) * tcrossprod(v)
    Sigma_x_samples[, t] <- v / denom
    xSigma_x_samples[t] <- s
  }

  list(Sigma_x_samples = Sigma_x_samples,
       xSigma_x_samples = xSigma_x_samples)
}

recursive_update_ridge_exact_bayes <- function(beta_matrix,
                                               sigma2,
                                               Sigma_x_samples,
                                               xSigma_x_samples,
                                               seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  beta_matrix <- as.matrix(beta_matrix)
  T <- ncol(Sigma_x_samples)
  B <- ncol(beta_matrix)

  E <- matrix(stats::rnorm(T * B), nrow = T, ncol = B)
  E <- sweep(E, 1, sqrt(sigma2 + xSigma_x_samples) / sigma2, "*")

  beta_matrix + Sigma_x_samples %*% E
}

sample_X_empirical <- function(X, T, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- as.matrix(X)
  idx <- sample.int(nrow(X), T, replace = TRUE)
  t(X[idx, , drop = FALSE])
}

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

single_linear_sas_defaults <- function(config) {
  defaults <- list(
    repeats = 1L,
    n_boot = 50L,
    paths_per_boot = 20L,
    B = 1000L,
    T = 5000L,
    n = 250L,
    p = 200L,
    p_star = 50L,
    sigma2 = 1,
    tau2 = 1,
    beta_scale = 3,
    v1 = 9,
    w = 0.25,
    v0_grid = "1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005",
    hetero_threshold = 1,
    hetero_index = 1L,
    em_max_iter = 500L,
    em_tol = 1e-6,
    save_draws = "true",
    write_draws_rds = "false",
    save_data = "false",
    save_fit_objects = "false",
    workers = "1",
    max_workers = 1L,
    scenario_workers = 5L,
    parallel = "true",
    single_thread_blas = "true"
  )
  for (name in names(defaults)) {
    linear_name <- paste0("linear_", name)
    if (is.null(config$params[[name]]) && is.null(config$params[[linear_name]])) {
      config$params[[name]] <- defaults[[name]]
    }
  }
  config
}

linear_sas_hetero_data <- function(config, repeat_seed) {
  base <- generate_linear_comparison_base(config, repeat_seed)
  multiplier <- config_num(config, "hetero_multiplier", 0)
  threshold <- config_num(config, "hetero_threshold", 1)
  hetero_index <- config_int(config, "hetero_index", 1L)
  hetero_index <- max(1L, min(ncol(base$X), hetero_index))
  scale_factor <- sqrt(
    1 + multiplier * as.numeric(abs(base$X[, hetero_index]) > threshold)
  )

  list(
    X = base$X,
    y = as.vector(base$X %*% base$beta) +
      sqrt(base$sigma2) * scale_factor * base$epsilon,
    beta = base$beta,
    sigma2 = base$sigma2,
    active_indices = base$active_indices,
    dgp = if (multiplier == 0) {
      "homoskedastic_gaussian"
    } else {
      "heteroskedastic_gaussian"
    },
    conditional_scale = sqrt(base$sigma2) * scale_factor
  )
}

extract_linear_selected_draws <- function(draw_object, multiplier, coefficient) {
  if (coefficient < 1L || coefficient > length(draw_object$beta_true)) {
    stop("Selected coefficient is outside 1:p.", call. = FALSE)
  }
  truth <- draw_object$beta_true[coefficient]
  if (!coefficient %in% draw_object$active_indices) {
    warning("Selected coefficient beta_", coefficient, " is inactive.")
  }

  method_draws <- list(
    PBP = draw_object$ordinary_draws[coefficient, ],
    bPBP = draw_object$bpbp_draws[coefficient, ]
  )
  method_initial <- c(
    PBP = draw_object$ordinary_beta_n[coefficient],
    bPBP = mean(draw_object$bootstrap_beta_init[coefficient, ])
  )

  rows <- lapply(names(method_draws), function(method) {
    values <- as.numeric(method_draws[[method]])
    data.frame(
      multiplier = multiplier,
      repeat_id = draw_object$repeat_id,
      coefficient = coefficient,
      active = coefficient %in% draw_object$active_indices,
      method = method,
      draw_id = seq_along(values),
      draw = values,
      truth = truth,
      error = values - truth,
      initialization = unname(method_initial[[method]]),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

linear_selected_summary <- function(data) {
  keys <- unique(data[c("multiplier", "method")])
  rows <- lapply(seq_len(nrow(keys)), function(i) {
    take <- data$multiplier == keys$multiplier[i] &
      data$method == keys$method[i]
    values <- data$draw[take]
    truth <- unique(data$truth[take])
    data.frame(
      multiplier = keys$multiplier[i],
      method = keys$method[i],
      coefficient = unique(data$coefficient[take]),
      truth = truth,
      initialization = unique(data$initialization[take]),
      mean = mean(values),
      median = stats::median(values),
      mean_bias = mean(values) - truth,
      median_bias = stats::median(values) - truth,
      sd = stats::sd(values),
      q025 = stats::quantile(values, 0.025, names = FALSE),
      q975 = stats::quantile(values, 0.975, names = FALSE),
      covered = stats::quantile(values, 0.025, names = FALSE) <= truth &
        truth <= stats::quantile(values, 0.975, names = FALSE)
    )
  })
  do.call(rbind, rows)
}

linear_ridge_hetero_settings_row <- function(config, multiplier, coefficient) {
  row <- comparison_settings_row(config, "linear_ridge_miss")
  row$model <- paste0("linear_ridge_hm", format(multiplier, trim = TRUE))
  row$dgp <- if (multiplier == 0) "well" else "miss"
  row$repeats <- 1L
  row$hetero_multiplier <- multiplier
  row$selected_coefficient <- coefficient
  row$monte_carlo_mode <- "single shared realization"
  row
}

linear_ridge_hetero_variant_config <- function(config, multiplier) {
  cfg <- comparison_variant_config(config, "linear_ridge_miss")
  cfg$model <- paste0("linear_ridge_hm", format(multiplier, trim = TRUE))
  cfg$out_dir <- file.path(config$out_dir, cfg$model)
  cfg$params$repeats <- 1L
  cfg$params$hetero_multiplier <- multiplier
  cfg$params$scenario <- if (multiplier == 0) "well" else "miss"
  cfg$params$save_draws <- "true"
  cfg$params$write_draws_rds <- "false"
  cfg$params$save_data <- "false"
  cfg$params$save_fit_objects <- "false"
  cfg$params$workers <- "1"
  cfg$params$max_workers <- 1L
  cfg
}

augment_linear_ridge_result <- function(table, multiplier) {
  table$family <- "linear"
  table$prior <- "ridge"
  table$dgp <- if (multiplier == 0) "well" else "miss"
  table$hetero_multiplier <- multiplier
  first <- c("family", "prior", "dgp", "hetero_multiplier")
  table[c(first, setdiff(names(table), first))]
}

run_one_linear_ridge_scenario <- function(config, multiplier, coefficient,
                                          setting) {
  cfg <- linear_ridge_hetero_variant_config(config, multiplier)
  model_id <- cfg$model
  dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)
  marker <- file.path(cfg$out_dir, "scenario_complete.rds")
  resume <- config_bool(config, "resume", TRUE)
  overwrite <- config_bool(config, "overwrite", FALSE)

  if (resume && !overwrite && file.exists(marker)) {
    cached <- readRDS(marker)
    if (!same_settings(cached$settings, setting)) {
      stop("Existing result has different settings: ", marker,
           ". Use a new --out directory or --overwrite=true.", call. = FALSE)
    }
    message("Skipping completed ridge multiplier=", multiplier)
    return(cached)
  }

  message("Running linear ridge with hetero_multiplier=", multiplier)
  result <- run_one_bpbp_model(
    cfg, model_id, linear_sas_hetero_data, draws_linear_from_data
  )
  draw_object <- result$draws_object$repeats[[1L]]
  selected <- extract_linear_selected_draws(
    draw_object, multiplier, coefficient
  )
  cached <- list(
    model = model_id,
    settings = setting,
    comparison = augment_linear_ridge_result(
      result$comparison, multiplier
    ),
    raw_comparison = augment_linear_ridge_result(
      result$raw_comparison, multiplier
    ),
    selected_draws = selected,
    active_indices = draw_object$active_indices,
    beta_true = draw_object$beta_true,
    completed_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  saveRDS(cached, marker, compress = "xz")
  message("Wrote ", marker)
  cached
}

write_linear_ridge_outputs <- function(completed, out_dir) {
  comparison <- do.call(rbind, lapply(completed, `[[`, "comparison"))
  raw <- do.call(rbind, lapply(completed, `[[`, "raw_comparison"))
  selected <- do.call(rbind, lapply(completed, `[[`, "selected_draws"))
  rownames(comparison) <- NULL
  rownames(raw) <- NULL
  rownames(selected) <- NULL
  summary <- linear_selected_summary(selected)
  coefficient <- unique(selected$coefficient)
  if (length(coefficient) != 1L) {
    stop("Combined draws must contain one selected coefficient.", call. = FALSE)
  }

  comparison_path <- file.path(
    out_dir, "linear_ridge_single_comparison_table.csv"
  )
  raw_path <- file.path(out_dir, "linear_ridge_single_comparison_raw.csv")
  draws_path <- file.path(out_dir, "linear_ridge_single_selected_draws.rds")
  summary_path <- file.path(
    out_dir, paste0("linear_ridge_single_beta", coefficient, "_summary.csv")
  )
  utils::write.csv(comparison, comparison_path, row.names = FALSE)
  utils::write.csv(raw, raw_path, row.names = FALSE)
  saveRDS(selected, draws_path, compress = "xz")
  utils::write.csv(summary, summary_path, row.names = FALSE)

  list(
    comparison = comparison,
    raw_comparison = raw,
    selected_draws = selected,
    selected_summary = summary,
    comparison_path = comparison_path,
    raw_path = raw_path,
    draws_path = draws_path,
    summary_path = summary_path
  )
}

run_linear_ridge_hetero_single <- function(config, multipliers,
                                           coefficient = 1L) {
  config <- normalize_simulation_config(config)
  config <- single_linear_sas_defaults(config)
  configure_parallel_environment(config)
  dir.create(config$out_dir, recursive = TRUE, showWarnings = FALSE)

  settings <- do.call(rbind, lapply(multipliers, function(multiplier) {
    linear_ridge_hetero_settings_row(config, multiplier, coefficient)
  }))
  rownames(settings) <- NULL
  settings_path <- file.path(config$out_dir, "linear_ridge_single_settings.csv")
  utils::write.csv(settings, settings_path, row.names = FALSE)

  scenario_workers <- min(
    length(multipliers),
    max(1L, config_int(config, "scenario_workers", 5L))
  )
  message("Running ", length(multipliers), " ridge scenarios with ",
          scenario_workers, " scenario workers")
  run_scenario <- function(i) {
    run_one_linear_ridge_scenario(
      config,
      multipliers[i],
      coefficient,
      settings[i, , drop = FALSE]
    )
  }
  completed <- if (scenario_workers > 1L &&
                   !identical(.Platform$OS.type, "windows")) {
    parallel::mclapply(
      seq_along(multipliers),
      run_scenario,
      mc.cores = scenario_workers,
      mc.preschedule = FALSE,
      mc.set.seed = FALSE
    )
  } else {
    lapply(seq_along(multipliers), run_scenario)
  }
  failed <- vapply(completed, function(item) {
    is.null(item) || inherits(item, "try-error") || is.null(item$model)
  }, logical(1))
  if (any(failed)) {
    stop(
      "Ridge scenario worker failed for multiplier(s): ",
      paste(multipliers[failed], collapse = ", "),
      call. = FALSE
    )
  }
  names(completed) <- vapply(completed, `[[`, character(1), "model")
  combined <- write_linear_ridge_outputs(completed, config$out_dir)
  list(settings = settings, completed = completed, combined = combined)
}

script_dir_of_this_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
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
  value <- cli_value(cli, key, NA_character_)
  if (is.na(value)) return(default)
  tolower(value) %in% c("true", "t", "yes", "1")
}

parse_multiplier_grid <- function(value = "0,5,10,15") {
  parts <- trimws(strsplit(as.character(value), ",", fixed = TRUE)[[1]])
  parts <- parts[nzchar(parts)]
  grid <- suppressWarnings(as.numeric(parts))
  if (any(is.na(grid))) {
    stop("multipliers must be a comma separated list of numbers.", call. = FALSE)
  }
  sort(unique(grid))
}

PLOT_OPTS <- list(
  multipliers   = c(0, 5, 10, 15),
  method_levels = c("PBP", "bPBP"),      # order of the panels
  method_labels = c(PBP = "MGP", bPBP = "bMGP"),

  adjust        = 0.9,     # kernel bandwidth multiplier (as in the original)
  slab_width    = 0.72,    # horizontal extent of a slab, in x units
  normalize     = "all",
  show_intervals = FALSE,  # TRUE adds the median dot + interval stem
  intervals     = c(0.66, 0.95),
  slab_alpha    = 0.88,

  palette_stops = c("#246A73", "#368F8B", "#78A65A",
                    "#D9A441", "#C96B3B", "#8F3B52"),
  ink           = "#1A1C1E",
  rule_colour   = "#202124",
  frame_colour  = "#4D4D4D",
  grid_colour   = "#E8E8E8",

  x_label       = "Degree of misspecification",
  truth_label   = TRUE,
  index_in_label = FALSE,
  y_break_n     = 6,

  fig_width_in  = 7.4,
  fig_height_in = 6.6,

  tikz_width_cm  = 14.0,   # width of one panel
  tikz_height_cm = 4.5,    # height of one panel
  tikz_sep_cm    = 1.25,   # vertical gap between the two panels
  tikz_points    = 220,    # samples kept per density outline
  tikz_prefix    = "mgp"   # prefix for every generated LaTeX name
)

load_ridge_draws <- function(path, opts) {
  if (!file.exists(path)) {
    stop("Draws file not found: ", path, call. = FALSE)
  }
  data <- readRDS(path)
  needed <- c("multiplier", "coefficient", "method", "draw", "truth")
  missing_cols <- setdiff(needed, names(data))
  if (length(missing_cols) > 0L) {
    stop("Draws file is missing column(s): ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }

  missing_m <- setdiff(opts$multipliers, unique(data$multiplier))
  if (length(missing_m) > 0L) {
    stop("No ridge draws found for m = ", paste(missing_m, collapse = ", "),
         call. = FALSE)
  }
  data <- data[data$multiplier %in% opts$multipliers, , drop = FALSE]

  coefficient <- unique(data$coefficient)
  if (length(coefficient) != 1L) {
    stop("Plot data must contain exactly one selected coefficient.",
         call. = FALSE)
  }
  truth <- unique(data$truth)
  if (length(truth) != 1L) {
    stop("The true beta must be shared across all multipliers.", call. = FALSE)
  }

  data$method <- factor(data$method, levels = opts$method_levels)
  if (anyNA(data$method)) {
    stop("Unexpected method label in the draws file.", call. = FALSE)
  }
  levels_m <- sort(unique(data$multiplier))
  data$multiplier_label <- factor(
    data$multiplier,
    levels = levels_m,
    labels = format(levels_m, trim = TRUE)
  )

  attr(data, "coefficient") <- coefficient
  attr(data, "truth") <- truth
  data
}

halfeye_palette <- function(levels, opts) {
  colours <- grDevices::colorRampPalette(opts$palette_stops)(length(levels))
  stats::setNames(colours, levels)
}

beta_math <- function(coefficient, opts) {
  if (isTRUE(opts$index_in_label)) {
    sprintf("\\beta_{%s}", coefficient)
  } else {
    "\\beta"
  }
}

y_breaks <- function(limits, opts) {
  breaks <- scales::extended_breaks(n = opts$y_break_n)(limits)
  breaks[breaks >= limits[1] & breaks <= limits[2]]
}

theme_halfeye <- function(base_size = 12, ink = "#1A1C1E",
                          frame_colour = PLOT_OPTS$frame_colour,
                          grid_colour = PLOT_OPTS$grid_colour) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor   = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(
        colour = grid_colour, linewidth = 0.3
      ),
      panel.border  = ggplot2::element_rect(colour = frame_colour, linewidth = 0.5),
      panel.spacing = grid::unit(1.6, "lines"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_size, colour = ink,
        margin = ggplot2::margin(0, 0, 5, 0)
      ),
      axis.title.x = ggplot2::element_text(
        margin = ggplot2::margin(t = 8), colour = ink
      ),
      axis.title.y = ggplot2::element_text(
        margin = ggplot2::margin(r = 8), colour = ink
      ),
      axis.text  = ggplot2::element_text(colour = ink),
      axis.ticks = ggplot2::element_line(colour = frame_colour, linewidth = 0.4),
      plot.margin = ggplot2::margin(6, 10, 6, 6)
    )
}

build_halfeye_plot <- function(data, opts, tex = FALSE) {
  coefficient <- attr(data, "coefficient")
  truth <- attr(data, "truth")
  palette <- halfeye_palette(levels(data$multiplier_label), opts)

  y_label <- if (tex) {
    sprintf("$%s$", beta_math(coefficient, opts))
  } else if (isTRUE(opts$index_in_label)) {
    bquote(beta[.(coefficient)])
  } else {
    bquote(beta)
  }
  x_label <- if (tex) paste0(opts$x_label, " $m$") else {
    bquote(.(opts$x_label) ~ italic(m))
  }

  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(
      x = .data$multiplier_label,
      y = .data$draw,
      fill = .data$multiplier_label
    )
  ) +
    do.call(ggdist::stat_halfeye, c(
      list(
        adjust        = opts$adjust,
        width         = opts$slab_width,
        normalize     = opts$normalize,
        side          = "right",
        justification = -0.02,
        slab_alpha    = opts$slab_alpha,
        slab_colour   = NA
      ),
      if (isTRUE(opts$show_intervals)) {
        list(
          .width = opts$intervals,
          point_interval = "median_qi",
          interval_colour = opts$ink,
          point_colour = opts$ink,
          point_size = 1.7,
          interval_size_range = c(0.5, 1.6)
        )
      } else {
        list(.width = NA, point_interval = NULL)
      }
    )) +
    ggplot2::geom_hline(
      yintercept = truth,
      linetype = "dashed",
      linewidth = 0.5,
      colour = opts$rule_colour
    ) +
    ggplot2::facet_wrap(
      ggplot2::vars(.data$method),
      ncol = 1,
      labeller = ggplot2::as_labeller(opts$method_labels)
    ) +
    ggplot2::scale_fill_manual(values = palette, guide = "none") +
    ggplot2::scale_x_discrete(
      expand = ggplot2::expansion(add = c(0.35, 0.95))
    ) +
    ggplot2::scale_y_continuous(
      breaks = y_breaks(range(data$draw), opts),
      expand = ggplot2::expansion(mult = c(0.05, 0.08))
    ) +
    ggplot2::labs(title = NULL, x = x_label, y = y_label) +
    theme_halfeye(base_size = 12, ink = opts$ink)

  if (isTRUE(opts$truth_label)) {
    label <- if (tex) {
      sprintf("$%s = %.2f$", beta_math(coefficient, opts), truth)
    } else if (isTRUE(opts$index_in_label)) {
      as.character(
        as.expression(bquote(beta[.(coefficient)] == .(sprintf("%.2f", truth))))
      )
    } else {
      as.character(as.expression(bquote(beta == .(sprintf("%.2f", truth)))))
    }
    annotation <- data.frame(
      method = factor(opts$method_levels[1], levels = opts$method_levels),
      multiplier_label = factor(
        levels(data$multiplier_label)[1],
        levels = levels(data$multiplier_label)
      ),
      x = nlevels(data$multiplier_label) + 0.9,
      y = truth,
      label = label,
      stringsAsFactors = FALSE
    )
    plot <- plot + ggplot2::geom_text(
      data = annotation,
      mapping = ggplot2::aes(x = .data$x, y = .data$y, label = .data$label),
      inherit.aes = FALSE,
      parse = !tex,
      hjust = 1, vjust = -0.55, size = 3.2, colour = opts$rule_colour
    )
  }
  plot
}

save_raster_outputs <- function(plot, out_dir, stem, opts) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  ggplot2::ggsave(
    png_path, plot, width = opts$fig_width_in, height = opts$fig_height_in,
    units = "in", dpi = 320, bg = "white"
  )
  ggplot2::ggsave(
    pdf_path, plot, width = opts$fig_width_in, height = opts$fig_height_in,
    units = "in", device = grDevices::cairo_pdf
  )
  c(png_path, pdf_path)
}

halfeye_geometry <- function(plot, data, opts) {
  built <- ggplot2::ggplot_build(plot)
  idx <- which(vapply(
    built$data, function(d) "datatype" %in% names(d), logical(1)
  ))
  if (length(idx) == 0L) {
    stop("Could not locate the stat_halfeye layer in the built plot.",
         call. = FALSE)
  }
  layer <- built$data[[idx[1]]]
  layout <- built$layout$layout
  layer$method <- as.character(layout$method[match(layer$PANEL, layout$PANEL)])

  m_levels <- levels(data$multiplier_label)
  layer$m_index <- round(layer$x)
  layer$m_label <- m_levels[layer$m_index]

  slab <- layer[layer$datatype == "slab", , drop = FALSE]
  interval <- layer[layer$datatype == "interval", , drop = FALSE]

  scope <- switch(
    opts$normalize,
    all = rep("all", nrow(slab)),
    panels = slab$method,
    paste(slab$method, slab$m_label, sep = "|")
  )
  slab$thick <- unlist(lapply(split(slab$thickness, scope), function(v) {
    top <- max(v, na.rm = TRUE)
    if (!is.finite(top) || top <= 0) v * 0 else v / top
  }))[order(order(scope))]

  keys <- unique(slab[c("method", "m_index", "m_label")])
  keys <- keys[order(match(keys$method, opts$method_levels), keys$m_index), ]

  outlines <- lapply(seq_len(nrow(keys)), function(i) {
    take <- slab$method == keys$method[i] & slab$m_index == keys$m_index[i]
    piece <- slab[take, c("y", "thick"), drop = FALSE]
    piece <- piece[order(piece$y), , drop = FALSE]
    piece <- piece[is.finite(piece$y) & is.finite(piece$thick), , drop = FALSE]
    n_out <- min(opts$tikz_points, nrow(piece))
    pick <- unique(round(seq(1, nrow(piece), length.out = n_out)))
    piece <- piece[pick, , drop = FALSE]
    list(
      method = keys$method[i],
      m_index = keys$m_index[i],
      m_label = keys$m_label[i],
      y = piece$y,
      x = keys$m_index[i] + piece$thick * opts$slab_width
    )
  })

  intervals <- if (nrow(interval) == 0L) list() else {
    lapply(seq_len(nrow(keys)), function(i) {
      take <- interval$method == keys$method[i] &
        interval$m_index == keys$m_index[i]
      piece <- interval[take, , drop = FALSE]
      piece <- piece[order(piece$.width), , drop = FALSE]
      list(
        method = keys$method[i],
        m_index = keys$m_index[i],
        point = piece$y[1],
        bands = data.frame(
          width = piece$.width, lower = piece$ymin, upper = piece$ymax
        )
      )
    })
  }

  y_values <- c(
    unlist(lapply(outlines, `[[`, "y")),
    unlist(lapply(intervals, function(iv) c(iv$bands$lower, iv$bands$upper)))
  )
  list(
    outlines = outlines,
    intervals = intervals,
    m_levels = m_levels,
    methods = opts$method_levels,
    coefficient = attr(data, "coefficient"),
    truth = attr(data, "truth"),
    y_min = min(y_values, na.rm = TRUE),
    y_max = max(y_values, na.rm = TRUE)
  )
}

num <- function(x, digits = 4) formatC(x, format = "f", digits = digits)

coordinate_block <- function(x, y, per_line = 4) {
  pairs <- sprintf("(%s,%s)", num(x), num(y))
  chunks <- split(pairs, ceiling(seq_along(pairs) / per_line))
  paste0("    ", vapply(chunks, paste, character(1), collapse = " "))
}

write_pgfplots_tikz <- function(geom, path, opts) {
  px <- opts$tikz_prefix
  palette <- halfeye_palette(geom$m_levels, opts)
  colour_names <- paste0(px, "Slab", LETTERS[seq_along(palette)])
  names(colour_names) <- geom$m_levels

  span <- geom$y_max - geom$y_min
  y_min <- geom$y_min - 0.05 * span
  y_max <- geom$y_max + 0.08 * span
  x_min <- 0.55
  x_max <- length(geom$m_levels) + 1.00
  ticks <- y_breaks(c(geom$y_min, geom$y_max), opts)

  L <- c(
    "% =========================================================================",
    "%  linear_ridge_halfeye_degree_misspecification -- pgfplots/TikZ version",
    "%  Generated by plot_ridge_degree_misspecification_tikz.R -- do not edit.",
    "%",
    "%  Preamble requirements (main.tex):",
    "%      \\usepackage{pgfplots}",
    "%      \\pgfplotsset{compat=1.18}",
    "%      \\usepgfplotslibrary{groupplots}",
    "%  Then:  \\input{figures/linear_ridge_halfeye_degree_misspecification_pgfplots}",
    "% =========================================================================",
    "\\begin{tikzpicture}"
  )

  L <- c(L, "  % ---- colours --------------------------------------------------------")
  for (i in seq_along(palette)) {
    L <- c(L, sprintf(
      "  \\definecolor{%s}{HTML}{%s}",
      colour_names[i], toupper(sub("^#", "", substr(palette[i], 1, 7)))
    ))
  }
  L <- c(L, sprintf("  \\definecolor{%sInk}{HTML}{%s}", px,
                    toupper(sub("^#", "", opts$ink))))
  L <- c(L, sprintf("  \\definecolor{%sRule}{HTML}{%s}", px,
                    toupper(sub("^#", "", opts$rule_colour))))
  L <- c(L, sprintf("  \\definecolor{%sGrid}{HTML}{%s}", px,
                    toupper(sub("^#", "", opts$grid_colour))))
  L <- c(L, sprintf("  \\definecolor{%sFrame}{HTML}{%s}", px,
                    toupper(sub("^#", "", opts$frame_colour))))

  L <- c(
    L,
    "  % ---- shared styles --------------------------------------------------",
    "  \\pgfplotsset{",
    sprintf("    %sPanel/.style={", px),
    sprintf("      width=%scm, height=%scm, scale only axis,",
            num(opts$tikz_width_cm, 2), num(opts$tikz_height_cm, 2)),
    sprintf("      axis background/.style={fill=white},"),
    sprintf("      axis line style={draw=%sFrame, line width=0.45pt},", px),
    sprintf("      tick style={draw=%sFrame, line width=0.45pt},", px),
    sprintf("      grid=major, ymajorgrids=true, xmajorgrids=false,"),
    sprintf("      grid style={draw=%sGrid, line width=0.3pt},", px),
    sprintf("      xmin=%s, xmax=%s, ymin=%s, ymax=%s,",
            num(x_min, 2), num(x_max, 2), num(y_min, 3), num(y_max, 3)),
    sprintf("      xtick={%s}, xticklabels={%s},",
            paste(seq_along(geom$m_levels), collapse = ","),
            paste(geom$m_levels, collapse = ",")),
    sprintf("      ytick={%s},",
            paste(num(ticks, 3), collapse = ",")),
    "      scaled y ticks=false, y tick label style={/pgf/number format/fixed},",
    sprintf("      ylabel={$%s$},", beta_math(geom$coefficient, opts)),
    "      ylabel style={font=\\small},",
    "      xlabel style={font=\\small},",
    "      tick label style={font=\\small},",
    "      every axis plot/.append style={line join=round},",
    "      clip mode=individual,",
    sprintf("      title style={font=\\small, text=%sInk, yshift=-2pt},", px),
    "    },",
    sprintf("    %sSlab/.style={draw=none, fill opacity=%s},",
            px, num(opts$slab_alpha, 2)),
    sprintf("    %sTruth/.style={draw=%sRule, dashed, line width=0.7pt},",
            px, px)
  )
  if (length(geom$intervals) > 0L) {
    L <- c(
      L,
      sprintf(
        "    %sInner/.style={draw=%sInk, line width=1.5pt, line cap=round},",
        px, px
      ),
      sprintf(
        "    %sOuter/.style={draw=%sInk, line width=0.55pt, line cap=round},",
        px, px
      ),
      sprintf(paste0("    %sPoint/.style={draw=none, only marks, mark=*, ",
                     "mark size=1.5pt, mark options={fill=white, draw=%sInk, ",
                     "line width=0.6pt}},"), px, px)
    )
  }
  L <- c(
    L,
    "  }",
    "  % ---- panels ---------------------------------------------------------",
    "  \\begin{groupplot}[",
    sprintf("    %sPanel,", px),
    "    group style={",
    "      group size=1 by 2,",
    sprintf("      vertical sep=%scm,", num(opts$tikz_sep_cm, 2)),
    "      x descriptions at=edge bottom,",
    "    },",
    sprintf("    xlabel={%s $m$},", opts$x_label),
    "  ]"
  )

  for (method in geom$methods) {
    L <- c(L, "", sprintf("  %% ---------- panel: %s ----------",
                          opts$method_labels[[method]]))
    L <- c(L, sprintf("  \\nextgroupplot[title={%s}]",
                      opts$method_labels[[method]]))

    for (piece in geom$outlines) {
      if (piece$method != method) next
      L <- c(L, sprintf(
        "  %% m = %s", piece$m_label
      ), sprintf(
        "  \\addplot[%sSlab, fill=%s] coordinates {",
        px, colour_names[[piece$m_label]]
      ))
      x <- c(piece$m_index, piece$x, piece$m_index)
      y <- c(piece$y[1], piece$y, piece$y[length(piece$y)])
      L <- c(L, coordinate_block(x, y), "  };")
    }

    for (iv in geom$intervals) {
      if (iv$method != method) next
      bands <- iv$bands[order(-iv$bands$width), , drop = FALSE]
      for (b in seq_len(nrow(bands))) {
        style <- if (b == 1L) paste0(px, "Outer") else paste0(px, "Inner")
        L <- c(L, sprintf(
          "  \\addplot[%s] coordinates {(%s,%s) (%s,%s)};",
          style, num(iv$m_index, 2), num(bands$lower[b]),
          num(iv$m_index, 2), num(bands$upper[b])
        ))
      }
      L <- c(L, sprintf(
        "  \\addplot[%sPoint] coordinates {(%s,%s)};",
        px, num(iv$m_index, 2), num(iv$point)
      ))
    }

    L <- c(L, "  % true coefficient", sprintf(
      "  \\addplot[%sTruth] coordinates {(%s,%s) (%s,%s)};",
      px, num(x_min, 2), num(geom$truth), num(x_max, 2), num(geom$truth)
    ))
    if (isTRUE(opts$truth_label) && identical(method, geom$methods[1])) {
      L <- c(L, sprintf(
        paste0("  \\node[anchor=south east, font=\\scriptsize, text=%sRule, ",
               "inner xsep=1pt, inner ysep=2pt] at (axis cs:%s,%s) ",
               "{$%s = %.2f$};"),
        px, num(x_max - 0.12, 2), num(geom$truth),
        beta_math(geom$coefficient, opts), geom$truth
      ))
    }
  }

  L <- c(L, "  \\end{groupplot}", "\\end{tikzpicture}")
  writeLines(L, path)
  path
}

write_standalone_wrapper <- function(figure_path, path) {
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

ensure_latex_on_path <- function() {
  register <- function(binary) {
    Sys.setenv(PATH = paste(dirname(binary), Sys.getenv("PATH"),
                            sep = .Platform$path.sep))
    options(tikzLatex = binary, tikzDefaultEngine = "pdftex")
    TRUE
  }
  found <- Sys.which("pdflatex")
  if (nzchar(found)) return(register(unname(found)))
  candidates <- Sys.glob(c(
    file.path(path.expand("~"), "Library/TinyTeX/bin/*"),
    "/Library/TeX/texbin",
    "/usr/local/texlive/*/bin/*",
    "C:/texlive/*/bin/*"
  ))
  for (dir in candidates) {
    binary <- Sys.glob(file.path(dir, "pdflatex*"))
    if (length(binary) > 0L) return(register(binary[1]))
  }
  FALSE
}

write_tikzdevice_tex <- function(plot, path, opts) {
  if (!ensure_latex_on_path()) {
    message("No LaTeX engine found; skipping the tikzDevice translation.")
    return(invisible(NULL))
  }
  if (!suppressWarnings(requireNamespace("tikzDevice", quietly = TRUE))) {
    message("tikzDevice is not installed; skipping the automatic translation.")
    return(invisible(NULL))
  }
  tikzDevice::tikz(
    file = path,
    width = opts$fig_width_in,
    height = opts$fig_height_in,
    standAlone = FALSE,
    sanitize = FALSE,
    documentDeclaration = "\\documentclass[11pt]{article}\n"
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  print(plot)
  path
}

run_ridge_simulation <- function(args, script_dir, out_root, multipliers,
                                 coefficient) {
  need_packages("MASS", "The simulation")
  config <- parse_simulation_args(args, script_dir)
  if (!any(startsWith(args, "--profile="))) config$profile <- "formal"
  if (!any(startsWith(args, "--design="))) config$design <- "normal"
  config$out_dir <- file.path(out_root, "linear_ridge")

  probe <- single_linear_sas_defaults(normalize_simulation_config(config))
  p <- config_int(probe, "p", 200L)
  if (coefficient < 1L || coefficient > p) {
    stop("--coefficient=", coefficient, " is outside 1:p with p = ", p,
         ".\n  Either raise --p= or pick a smaller --coefficient=.",
         call. = FALSE)
  }

  message("Simulating ", length(multipliers), " ridge scenarios (m = ",
          paste(multipliers, collapse = ", "), ") for beta_", coefficient,
          " -- this is the slow part")
  result <- run_linear_ridge_hetero_single(config, multipliers, coefficient)
  message("Wrote draws: ", result$combined$draws_path)
  result$combined$draws_path
}

main <- function() {
  script_dir <- script_dir_of_this_file()
  args <- commandArgs(trailingOnly = TRUE)
  cli <- parse_cli(args)

  out_root <- cli_value(
    cli, "out", file.path(script_dir, "output")
  )
  figure_dir <- cli_value(cli, "figures", file.path(out_root, "figures"))
  coefficient <- as.integer(cli_value(cli, "coefficient", "49"))

  opts <- PLOT_OPTS
  opts$multipliers <- parse_multiplier_grid(
    cli_value(cli, "multipliers", paste(opts$multipliers, collapse = ","))
  )
  opts$normalize <- match.arg(
    cli_value(cli, "normalize", opts$normalize), c("all", "panels", "groups")
  )
  opts$show_intervals <- cli_flag(cli, "intervals", opts$show_intervals)

  default_draws <- file.path(
    out_root, "linear_ridge", "linear_ridge_single_selected_draws.rds"
  )
  draws_path <- cli_value(cli, "input", default_draws)
  if (cli_flag(cli, "simulate", FALSE)) {
    draws_path <- run_ridge_simulation(
      args, script_dir, out_root, opts$multipliers, coefficient
    )
  } else if (!file.exists(draws_path)) {
    stop(
      "No draws found at: ", draws_path,
      "\n  Either point --input= at an existing ",
      "linear_ridge_single_selected_draws.rds,",
      "\n  or add --simulate=true to generate it from scratch.",
      call. = FALSE
    )
  }

  need_packages(c("ggplot2", "ggdist", "scales"), "The figure")
  stem <- "linear_ridge_halfeye_degree_misspecification"
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

  data <- load_ridge_draws(draws_path, opts)
  message("Loaded ", nrow(data), " draws for beta_", attr(data, "coefficient"),
          " at m = ", paste(opts$multipliers, collapse = ", "))

  plot <- build_halfeye_plot(data, opts, tex = FALSE)
  raster_paths <- save_raster_outputs(plot, figure_dir, stem, opts)

  geom <- halfeye_geometry(plot, data, opts)
  tikz_path <- write_pgfplots_tikz(
    geom, file.path(figure_dir, paste0(stem, "_pgfplots.tex")), opts
  )
  wrapper_path <- write_standalone_wrapper(
    tikz_path, file.path(figure_dir, paste0(stem, "_standalone.tex"))
  )

  written <- c(raster_paths, tikz_path, wrapper_path)
  if (cli_flag(cli, "tikzdevice", FALSE)) {
    tex_plot <- build_halfeye_plot(data, opts, tex = TRUE)
    device_path <- write_tikzdevice_tex(
      tex_plot, file.path(figure_dir, paste0(stem, "_tikzdevice.tex")), opts
    )
    written <- c(written, device_path)
  }

  message("Wrote:\n  ", paste(written, collapse = "\n  "))
  invisible(written)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
