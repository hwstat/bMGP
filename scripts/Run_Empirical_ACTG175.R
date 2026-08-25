## ACTG175 application, standard fits (Section 4.4): GPE, TPE and BTPE engines
## on data/AIDS.csv; writes application_results/. Run from scripts/.
## Usage: Rscript Run_Empirical_ACTG175.R
safe_solve <- function(A, ridge = 1e-8) {
  A <- as.matrix(A)
  p <- ncol(A)
  solve(A + diag(ridge, p))
}

make_psd <- function(S, eps = 1e-8) {
  S <- (S + t(S)) / 2
  ee <- eigen(S, symmetric = TRUE)
  vals <- pmax(ee$values, eps)
  ee$vectors %*% diag(vals, nrow = length(vals)) %*% t(ee$vectors)
}

tail_sum_sq <- function(M_total) {
  psigamma(M_total + 1, deriv = 1)
}

recommended_ppc_cores <- function(default = 4L) {
  dc <- parallel::detectCores(logical = FALSE)
  if (is.na(dc) || dc < 1L) {
    return(as.integer(default))
  }
  as.integer(max(1L, min(default, dc - 1L)))
}

safe_mc_lapply <- function(X, FUN, n_cores = 1L, mc_preschedule = TRUE) {
  n_cores <- as.integer(n_cores)
  if (is.na(n_cores) || n_cores <= 1L || .Platform$OS.type == "windows") {
    return(lapply(X, FUN))
  }
  parallel::mclapply(X, FUN, mc.cores = n_cores, mc.preschedule = mc_preschedule)
}

time_with_value <- function(expr) {
  tm <- system.time(val <- eval.parent(substitute(expr)))
  list(value = val, elapsed = unname(tm["elapsed"]))
}

read_aids_data <- function(filepath = "../data/AIDS.csv",
                           outcome_name = "cd420",
                           strict = TRUE) {
  dat <- read.csv(filepath, check.names = FALSE)

  first_name <- names(dat)[1]
  first_col <- dat[[1]]
  n <- nrow(dat)

  is_seq0 <- is.numeric(first_col) && length(first_col) == n &&
    all(first_col == seq.int(0, n - 1))
  is_seq1 <- is.numeric(first_col) && length(first_col) == n &&
    all(first_col == seq_len(n))

  drop_first <- isTRUE(
    is.na(first_name) ||
      trimws(first_name) == "" ||
      identical(first_name, "X") ||
      identical(first_name, "Unnamed: 0") ||
      grepl("^Unnamed:?\\s*0$", first_name) ||
      is_seq0 ||
      is_seq1
  )

  if (drop_first) {
    dat <- dat[, -1, drop = FALSE]
  }

  dat <- dat[complete.cases(dat), , drop = FALSE]

  if (!(outcome_name %in% names(dat))) {
    stop("Expected outcome column '", outcome_name, "' not found in AIDS.csv.")
  }

  x_names <- setdiff(names(dat), outcome_name)
  y <- as.numeric(dat[[outcome_name]])
  X_no_intercept <- as.matrix(dat[, x_names, drop = FALSE])
  storage.mode(X_no_intercept) <- "double"

  X <- cbind(Intercept = 1, X_no_intercept)
  storage.mode(X) <- "double"

  if (strict) {
    req_trt <- c("trt_1", "trt_2", "trt_3")
    if (!all(req_trt %in% colnames(X))) {
      stop("Treatment dummy columns trt_1/trt_2/trt_3 are missing.")
    }
    if (ncol(X) != 16) {
      stop("Expected 16 regression coefficients including intercept, got ", ncol(X), ".")
    }
    if (length(y) != nrow(X)) {
      stop("Length of y does not match number of rows in X.")
    }
  }

  list(
    data = dat,
    y = y,
    X = X,
    outcome_name = outcome_name,
    pred_names = x_names,
    coef_names = colnames(X),
    n_obs = length(y),
    p = ncol(X)
  )
}

fit_gaussian_regression_mle <- function(y, X, var_floor = 1e-8) {
  beta_hat <- tryCatch(
    as.vector(qr.solve(X, y)),
    error = function(e) rep(0, ncol(X))
  )

  resid <- as.vector(y - X %*% beta_hat)
  sigma2_hat <- max(mean(resid^2), var_floor)

  list(beta = beta_hat, sigma2 = sigma2_hat)
}

gaussian_regression_mp <- function(y_obs, X_obs,
                                   B = 10000,
                                   N_add = 100,
                                   sigma2_floor = 1e-8,
                                   ridge = 1e-8,
                                   seed = 383) {
  y_obs <- as.numeric(y_obs)
  X_obs <- as.matrix(X_obs)
  n_obs <- length(y_obs)
  p <- ncol(X_obs)

  init_fit <- fit_gaussian_regression_mle(y_obs, X_obs, var_floor = sigma2_floor)
  beta0 <- init_fit$beta
  sigma20 <- init_fit$sigma2

  Sigma_nx <- crossprod(X_obs) / n_obs
  Sigma_nx_inv <- safe_solve(Sigma_nx, ridge = ridge)

  set.seed(seed)
  beta_mat <- matrix(rep(beta0, each = B), nrow = B, ncol = p, byrow = FALSE)
  sigma2_vec <- rep(sigma20, B)

  idx_mat <- matrix(
    sample.int(n_obs, size = B * N_add, replace = TRUE),
    nrow = B, ncol = N_add
  )
  eps_mat <- matrix(rnorm(B * N_add), nrow = B, ncol = N_add)

  for (tt in seq_len(N_add)) {
    Ncur <- n_obs + tt
    idx <- idx_mat[, tt]
    X_new <- X_obs[idx, , drop = FALSE]
    e_new <- sqrt(pmax(sigma2_vec, sigma2_floor)) * eps_mat[, tt]

    XSinv <- X_new %*% Sigma_nx_inv
    beta_mat <- beta_mat + (XSinv * e_new) / Ncur
    sigma2_vec <- pmax(
      sigma2_vec + (e_new^2 - sigma2_vec) / Ncur,
      sigma2_floor
    )
  }

  rtail <- sqrt(tail_sum_sq(n_obs + N_add))
  L_inv <- t(chol(make_psd(Sigma_nx_inv)))

  z1 <- matrix(rnorm(p * B), nrow = p, ncol = B)
  z1_cov <- L_inv %*% z1
  z2 <- rnorm(B)

  for (i in seq_len(B)) {
    beta_mat[i, ] <- beta_mat[i, ] +
      rtail * sqrt(sigma2_vec[i]) * z1_cov[, i]
    sigma2_vec[i] <- max(
      sigma2_floor,
      sigma2_vec[i] + rtail * sqrt(2 * sigma2_vec[i]^2) * z2[i]
    )
  }

  colnames(beta_mat) <- colnames(X_obs)

  list(
    beta_draws = beta_mat,
    var_draws = sigma2_vec,
    coef_names = colnames(X_obs),
    init = init_fit,
    model = "MGP-Gaussian-hybrid",
    display_label = "MGP-Gaussian-hybrid",
    role = "misspec_control",
    N_add = N_add
  )
}

generate_pseudo_sample_gaussian_mp <- function(y_obs, X_obs, n_rep,
                                               init_fit = NULL,
                                               sigma2_floor = 1e-8,
                                               ridge = 1e-8,
                                               seed = 1234) {
  y_obs <- as.numeric(y_obs)
  X_obs <- as.matrix(X_obs)
  n_obs <- length(y_obs)

  if (is.null(init_fit)) {
    init_fit <- fit_gaussian_regression_mle(y_obs, X_obs, var_floor = sigma2_floor)
  }

  beta <- init_fit$beta
  sigma2 <- max(init_fit$sigma2, sigma2_floor)

  Sigma_nx <- crossprod(X_obs) / n_obs
  Sigma_nx_inv <- safe_solve(Sigma_nx, ridge = ridge)

  set.seed(seed)
  idx_seq <- sample.int(n_obs, size = n_rep, replace = TRUE)
  eps_seq <- rnorm(n_rep)

  p <- ncol(X_obs)
  X_rep <- matrix(NA_real_, nrow = n_rep, ncol = p)
  Y_rep <- numeric(n_rep)
  colnames(X_rep) <- colnames(X_obs)

  for (k in seq_len(n_rep)) {
    i_cur <- n_obs + k
    x_new <- X_obs[idx_seq[k], ]
    e_new <- sqrt(max(sigma2, sigma2_floor)) * eps_seq[k]
    y_new <- sum(beta * x_new) + e_new

    X_rep[k, ] <- x_new
    Y_rep[k] <- y_new

    z_beta <- as.numeric(Sigma_nx_inv %*% x_new) * e_new
    z_sig2 <- e_new^2 - sigma2

    beta <- beta + z_beta / i_cur
    sigma2 <- max(sigma2_floor, sigma2 + z_sig2 / i_cur)
  }

  list(y = Y_rep, X = X_rep)
}

fit_treg_one <- function(y, X, df = 5, seed = 38134,
                         max_iter = 500, xrtol = 1e-5) {
  n <- nrow(X)
  p <- ncol(X)

  Sigma_nx <- crossprod(X) / n
  Z <- safe_solve(Sigma_nx, ridge = 1e-8) %*% t(X)

  set.seed(seed)
  theta <- rnorm(p + 1, 0, 1)
  theta[p + 1] <- rexp(1, rate = 1 / 2)
  step <- numeric(p + 1)

  for (iter in seq_len(max_iter)) {
    tau2 <- max(theta[p + 1], 1e-6)
    Rvec <- as.vector((y - X %*% theta[1:p]) / sqrt(tau2))

    step[] <- 0
    for (i in seq_len(n)) {
      step[1:p] <- step[1:p] +
        (1 / n) *
        ((sqrt(tau2) * (df + 3) * Rvec[i]) / (df + Rvec[i]^2)) *
        Z[, i]

      step[p + 1] <- step[p + 1] +
        (1 / n) *
        ((tau2 * (df + 3) * (Rvec[i]^2 - 1)) / (df + Rvec[i]^2))
    }

    theta <- theta + step
    theta[p + 1] <- max(theta[p + 1], 1e-6)

    if (sqrt(sum(step^2)) < sqrt(sum(theta^2)) * xrtol) {
      break
    }
  }

  tau2 <- max(theta[p + 1], 1e-6)
  Rvec <- as.vector((y - X %*% theta[1:p]) / sqrt(tau2))
  opt_loglik <- mean(dt(Rvec, df = df, log = TRUE))

  list(theta = theta, opt_loglik = opt_loglik)
}

fit_treg_restart <- function(y, X, df = 5, n_restart = 10, base_seed = 38133) {
  p <- ncol(X)
  theta_rest <- matrix(NA_real_, nrow = p + 1, ncol = n_restart)
  ll_rest <- numeric(n_restart)

  for (i in seq_len(n_restart)) {
    tmp <- fit_treg_one(y, X, df = df, seed = base_seed + i)
    theta_rest[, i] <- tmp$theta
    ll_rest[i] <- tmp$opt_loglik
  }

  theta_rest[, which.max(ll_rest)]
}

fit_student_t_regression_julia_style <- function(y, X, df = 5, n_restart = 10) {
  theta <- fit_treg_restart(y, X, df = df, n_restart = n_restart)
  p <- ncol(X)

  list(
    beta = theta[1:p],
    tau2 = theta[p + 1]
  )
}

student_t_regression_mp <- function(y_obs, X_obs,
                                    B = 10000,
                                    N_add = 100,
                                    df = 5,
                                    n_restart = 10,
                                    tau2_floor = 1e-8,
                                    seed = 383) {
  y_obs <- as.numeric(y_obs)
  X_obs <- as.matrix(X_obs)
  n_obs <- length(y_obs)
  p <- ncol(X_obs)

  init_fit <- fit_student_t_regression_julia_style(
    y = y_obs, X = X_obs, df = df, n_restart = n_restart
  )

  beta0 <- init_fit$beta
  tau20 <- max(init_fit$tau2, tau2_floor)

  theta_mat <- matrix(NA_real_, nrow = p + 1, ncol = B)
  theta_mat[1:p, ] <- beta0
  theta_mat[p + 1, ] <- tau20

  Sigma_nx <- crossprod(X_obs) / n_obs
  Zmat <- safe_solve(Sigma_nx, ridge = 1e-8) %*% t(X_obs)

  set.seed(seed)
  Rmat <- matrix(rt(B * N_add, df = df), nrow = B, ncol = N_add)
  idx_mat <- matrix(
    sample.int(n_obs, size = B * N_add, replace = TRUE),
    nrow = B, ncol = N_add
  )

  for (tt in seq_len(N_add)) {
    Ncur <- n_obs + tt
    Rcol <- Rmat[, tt]
    idx <- idx_mat[, tt]

    tau2_vec <- pmax(theta_mat[p + 1, ], tau2_floor)
    coef_beta <- (sqrt(tau2_vec) * (df + 3) * Rcol) / (df + Rcol^2)

    Zsel <- Zmat[, idx, drop = FALSE]
    beta_update <- t(t(Zsel) * coef_beta) / Ncur
    theta_mat[1:p, ] <- theta_mat[1:p, ] + beta_update

    tau2_old <- pmax(theta_mat[p + 1, ], tau2_floor)
    theta_mat[p + 1, ] <- pmax(
      tau2_floor,
      theta_mat[p + 1, ] +
        (1 / Ncur) *
        (tau2_old * (df + 3) * (Rcol^2 - 1)) / (df + Rcol^2)
    )
  }

  rtail <- sqrt(tail_sum_sq(n_obs + N_add))
  Sigma_nx_inv <- safe_solve(Sigma_nx, ridge = 1e-8)
  L_inv <- t(chol(make_psd(Sigma_nx_inv)))

  z1 <- matrix(rnorm(p * B), nrow = p, ncol = B)
  z1_cov <- L_inv %*% z1
  z2 <- rnorm(B)

  for (i in seq_len(B)) {
    tau2i <- max(theta_mat[p + 1, i], tau2_floor)
    theta_mat[1:p, i] <- theta_mat[1:p, i] +
      rtail * sqrt(((df + 3) * tau2i) / (df + 1)) * z1_cov[, i]
    theta_mat[p + 1, i] <- max(
      tau2_floor,
      theta_mat[p + 1, i] +
        rtail * sqrt(2 * tau2i^2 * (df + 3) / df) * z2[i]
    )
  }

  beta_mat <- t(theta_mat[1:p, , drop = FALSE])
  colnames(beta_mat) <- colnames(X_obs)
  tau2_vec <- as.numeric(theta_mat[p + 1, ])

  list(
    beta_draws = beta_mat,
    var_draws = tau2_vec,
    coef_names = colnames(X_obs),
    init = init_fit,
    model = "MGP-Student-t-hybrid",
    display_label = "MGP-Student-t-hybrid",
    N_add = N_add
  )
}

generate_pseudo_sample_t_mp <- function(y_obs, X_obs, n_rep,
                                        init_fit = NULL,
                                        df = 5,
                                        n_restart = 10,
                                        tau2_floor = 1e-8,
                                        seed = 1234) {
  y_obs <- as.numeric(y_obs)
  X_obs <- as.matrix(X_obs)
  n_obs <- length(y_obs)
  p <- ncol(X_obs)

  if (is.null(init_fit)) {
    init_fit <- fit_student_t_regression_julia_style(
      y = y_obs, X = X_obs, df = df, n_restart = n_restart
    )
  }

  theta <- c(init_fit$beta, max(init_fit$tau2, tau2_floor))
  Sigma_nx <- crossprod(X_obs) / n_obs
  Zmat <- safe_solve(Sigma_nx, ridge = 1e-8) %*% t(X_obs)

  set.seed(seed)
  Rseq <- rt(n_rep, df = df)
  idx_seq <- sample.int(n_obs, size = n_rep, replace = TRUE)

  X_rep <- matrix(NA_real_, nrow = n_rep, ncol = p)
  Y_rep <- numeric(n_rep)
  colnames(X_rep) <- colnames(X_obs)

  for (k in seq_len(n_rep)) {
    i_cur <- n_obs + k
    x_idx <- idx_seq[k]
    x_new <- X_obs[x_idx, ]
    tau2 <- max(theta[p + 1], tau2_floor)
    e_new <- sqrt(tau2) * Rseq[k]
    y_new <- sum(theta[1:p] * x_new) + e_new

    X_rep[k, ] <- x_new
    Y_rep[k] <- y_new

    theta[1:p] <- theta[1:p] +
      (1 / i_cur) *
      ((sqrt(tau2) * (df + 3) * Rseq[k]) / (df + Rseq[k]^2)) *
      Zmat[, x_idx]

    theta[p + 1] <- max(
      tau2_floor,
      theta[p + 1] +
        (1 / i_cur) *
        ((tau2 * (df + 3) * (Rseq[k]^2 - 1)) / (df + Rseq[k]^2))
    )
  }

  list(y = Y_rep, X = X_rep)
}

.bayes_student_t_stan_model_cache <- new.env(parent = emptyenv())

get_bayes_student_t_stan_model <- function(force_recompile = FALSE) {
  cache_key <- "student_t_regression_df_fixed"
  if (!isTRUE(force_recompile) &&
      exists(cache_key, envir = .bayes_student_t_stan_model_cache, inherits = FALSE)) {
    return(get(cache_key, envir = .bayes_student_t_stan_model_cache, inherits = FALSE))
  }

  stan_code <- "
data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  vector[N] y;
  real<lower=1> df_fixed;
  real<lower=0> sigma0;
}
parameters {
  vector[P] coefficients;
  real<lower=1e-4> sigma;
}
model {
  coefficients ~ normal(0, sigma0);
  sigma ~ cauchy(0, 5);
  for (i in 1:N) {
    y[i] ~ student_t(df_fixed, dot_product(row(X, i), coefficients), sigma);
  }
}
"
  stan_file <- cmdstanr::write_stan_file(stan_code)
  model <- cmdstanr::cmdstan_model(stan_file, force_recompile = force_recompile, quiet = TRUE)
  assign(cache_key, model, envir = .bayes_student_t_stan_model_cache)
  model
}

fit_bayes_student_t_cmdstan <- function(y_obs, X_obs,
                                        df_fixed = 5,
                                        sigma0 = 10,
                                        iter_sampling = 10000,
                                        iter_warmup = 2000,
                                        chains = 1,
                                        parallel_chains = 1,
                                        seed = 381,
                                        force_recompile = FALSE) {
  model <- get_bayes_student_t_stan_model(force_recompile = force_recompile)

  dat_list <- list(
    N = nrow(X_obs),
    P = ncol(X_obs),
    X = X_obs,
    y = as.numeric(y_obs),
    df_fixed = df_fixed,
    sigma0 = sigma0
  )

  model$sample(
    data = dat_list,
    seed = seed,
    chains = chains,
    parallel_chains = parallel_chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    adapt_delta = 0.95,
    max_treedepth = 12,
    refresh = 0
  )
}

extract_cmdstan_draws <- function(fit, coef_names) {
  dm <- posterior::as_draws_matrix(
    fit$draws(variables = c("coefficients", "sigma"))
  )

  beta_cols <- paste0("coefficients[", seq_along(coef_names), "]")
  miss <- setdiff(beta_cols, colnames(dm))
  if (length(miss) > 0) {
    stop("Missing coefficient columns in cmdstan draws: ", paste(miss, collapse = ", "))
  }
  if (!("sigma" %in% colnames(dm))) {
    stop("Column 'sigma' not found in cmdstan draws.")
  }

  beta_mat <- as.matrix(dm[, beta_cols, drop = FALSE])
  colnames(beta_mat) <- coef_names
  tau2_vec <- as.numeric(dm[, "sigma"])^2

  list(
    beta_draws = beta_mat,
    var_draws = tau2_vec,
    coef_names = coef_names,
    model = "Bayes-Student-t",
    display_label = "Bayes-Student-t"
  )
}

generate_pseudo_sample_bayes_cmdstan <- function(X_obs,
                                                 beta_draws,
                                                 tau2_draws,
                                                 n_rep,
                                                 df = 5,
                                                 seed = 1234) {
  set.seed(seed)

  n_obs <- nrow(X_obs)
  p <- ncol(X_obs)

  b <- sample.int(nrow(beta_draws), size = 1)
  beta_b <- beta_draws[b, ]
  tau2_b <- tau2_draws[b]

  idx_seq <- sample.int(n_obs, size = n_rep, replace = TRUE)
  Rseq <- rt(n_rep, df = df)

  X_rep <- matrix(NA_real_, nrow = n_rep, ncol = p)
  Y_rep <- numeric(n_rep)
  colnames(X_rep) <- colnames(X_obs)

  for (k in seq_len(n_rep)) {
    x_new <- X_obs[idx_seq[k], ]
    y_new <- sum(beta_b * x_new) + sqrt(tau2_b) * Rseq[k]
    X_rep[k, ] <- x_new
    Y_rep[k] <- y_new
  }

  list(y = Y_rep, X = X_rep)
}

functional_coef_t <- function(y, X, target_coef = "trt_2",
                              df = 5, n_restart = 10) {
  fit <- fit_student_t_regression_julia_style(
    y = y, X = X, df = df, n_restart = n_restart
  )
  idx <- match(target_coef, colnames(X))
  if (is.na(idx)) stop("target_coef not found in X.")
  fit$beta[idx]
}

get_gaussian_reference_fit <- function(y_obs, X_obs, fit_g = NULL) {
  if (!is.null(fit_g) && !is.null(fit_g$init)) {
    return(list(
      beta = as.numeric(fit_g$init$beta),
      sigma2 = as.numeric(max(fit_g$init$sigma2, 1e-8))
    ))
  }
  fit_gaussian_regression_mle(y = y_obs, X = X_obs)
}

standardized_residuals_from_reference <- function(y, X, beta_ref, sigma2_ref) {
  mu_ref <- as.vector(X %*% beta_ref)
  (as.numeric(y) - mu_ref) / sqrt(max(sigma2_ref, 1e-8))
}

functional_abs_stdresid_tail_ref <- function(y, X, beta_ref, sigma2_ref,
                                             q = 0.995) {
  r_std <- standardized_residuals_from_reference(y, X, beta_ref, sigma2_ref)
  as.numeric(quantile(abs(r_std), probs = q, names = FALSE, type = 7))
}

functional_chisq_ref <- function(y, X, beta_ref, sigma2_ref) {
  r_std <- standardized_residuals_from_reference(y, X, beta_ref, sigma2_ref)
  mean(r_std^2)
}

build_pbppc_functional <- function(y_obs, X_obs,
                                   fit_g = NULL,
                                   functional = c("coef", "chi2", "tail"),
                                   target_coef = "trt_2",
                                   df = 5,
                                   q_tail = 0.995,
                                   functional_restarts = 10) {
  functional <- match.arg(functional)

  if (functional == "coef") {
    eval_stat <- function(y, X) {
      functional_coef_t(
        y = y,
        X = X,
        target_coef = target_coef,
        df = df,
        n_restart = functional_restarts
      )
    }

    return(list(
      S_obs = eval_stat(y_obs, X_obs),
      eval_stat = eval_stat,
      functional_label = paste0("coef_", target_coef),
      diagnostic_class = "coefficient",
      p_value_type = "two_sided",
      reference_label = "Student-t refit coefficient functional"
    ))
  }

  ref <- get_gaussian_reference_fit(y_obs = y_obs, X_obs = X_obs, fit_g = fit_g)

  if (functional == "chi2") {
    eval_stat <- function(y, X) {
      functional_chisq_ref(
        y = y, X = X,
        beta_ref = ref$beta,
        sigma2_ref = ref$sigma2
      )
    }

    return(list(
      S_obs = eval_stat(y_obs, X_obs),
      eval_stat = eval_stat,
      functional_label = "chi2_gaussian_ref",
      diagnostic_class = "chi_square_discrepancy",
      p_value_type = "upper",
      reference_label = "Normalized chi-square discrepancy using the fixed observed-data Gaussian working fit"
    ))
  }

  eval_stat <- function(y, X) {
    functional_abs_stdresid_tail_ref(
      y = y, X = X,
      beta_ref = ref$beta,
      sigma2_ref = ref$sigma2,
      q = q_tail
    )
  }

  list(
    S_obs = eval_stat(y_obs, X_obs),
    eval_stat = eval_stat,
    functional_label = paste0("abs_stdresid_q", q_tail, "_gaussian_ref"),
    diagnostic_class = "absolute_residual_tail",
    p_value_type = "upper",
    reference_label = paste0(
      "Absolute standardized-residual quantile using the fixed observed-data Gaussian working fit; q=",
      q_tail
    )
  )
}

pbppc_from_statistics <- function(S_obs, S_rep, S_center, n_obs,
                                  model_label,
                                  functional_label,
                                  diagnostic_class,
                                  p_value_type = c("two_sided", "upper", "lower"),
                                  reference_label = "",
                                  n_mc_pvalue = length(S_rep),
                                  m_future_ppc = NA_integer_) {
  p_value_type <- match.arg(p_value_type)

  p_up <- mean(S_rep >= S_obs)
  p_low <- mean(S_rep <= S_obs)

  p_val_raw <- switch(
    p_value_type,
    two_sided = 2 * min(p_up, 1 - p_up),
    upper = p_up,
    lower = p_low
  )
  p_val <- min(1, p_val_raw)

  delta <- sqrt(n_obs) * (S_center - S_obs)
  mc_se <- sqrt(max(p_up * (1 - p_up), 0) / max(n_mc_pvalue, 1))

  list(
    S_obs = S_obs,
    S_rep = S_rep,
    S_center = S_center,
    p_value = p_val,
    delta = delta,
    summary = data.frame(
      model = model_label,
      functional = functional_label,
      diagnostic_class = diagnostic_class,
      p_value_type = p_value_type,
      reference = reference_label,
      n_mc_pvalue = n_mc_pvalue,
      m_future_ppc = m_future_ppc,
      S_obs = S_obs,
      mc_prob_ge = p_up,
      mc_prob_ge_se = mc_se,
      p_value = p_val,
      S_rep_mean = mean(S_rep),
      S_center_mean = mean(S_center),
      delta_mean = mean(delta),
      delta_sd = sd(delta),
      pr_delta_gt_0 = mean(delta > 0),
      delta_q025 = quantile(delta, 0.025, names = FALSE),
      delta_q975 = quantile(delta, 0.975, names = FALSE),
      row.names = NULL
    )
  )
}

run_pbppc_three_engines <- function(y_obs, X_obs,
                                    fit_g, fit_t, fit_b,
                                    functional = c("coef", "chi2", "tail"),
                                    target_coef = "trt_2",
                                    df = 5,
                                    q_tail = 0.995,
                                    n_mc_pvalue = 100,
                                    m_future_ppc = NULL,
                                    functional_restarts = 10,
                                    p_value_type = NULL,
                                    parallel = FALSE,
                                    n_cores = 1L,
                                    seed = 2026) {
  functional <- match.arg(functional)
  n_obs <- length(y_obs)

  if (is.null(m_future_ppc)) {
    m_future_ppc <- if (functional == "coef") n_obs else 2L * n_obs
  }
  if (m_future_ppc < n_obs) {
    stop("m_future_ppc must be at least n_obs.")
  }

  fun_obj <- build_pbppc_functional(
    y_obs = y_obs,
    X_obs = X_obs,
    fit_g = fit_g,
    functional = functional,
    target_coef = target_coef,
    df = df,
    q_tail = q_tail,
    functional_restarts = functional_restarts
  )

  if (is.null(p_value_type)) {
    p_value_type <- fun_obj$p_value_type
  }

  S_obs <- fun_obj$S_obs
  get_S <- fun_obj$eval_stat

  one_rep <- function(b) {
    dat_g <- generate_pseudo_sample_gaussian_mp(
      y_obs = y_obs,
      X_obs = X_obs,
      n_rep = m_future_ppc,
      init_fit = fit_g$init,
      seed = seed + 1000 + b
    )
    g_rep <- get_S(
      dat_g$y[seq_len(n_obs)],
      dat_g$X[seq_len(n_obs), , drop = FALSE]
    )
    g_center <- get_S(
      c(y_obs, dat_g$y),
      rbind(X_obs, dat_g$X)
    )

    dat_t <- generate_pseudo_sample_t_mp(
      y_obs = y_obs,
      X_obs = X_obs,
      n_rep = m_future_ppc,
      init_fit = fit_t$init,
      df = df,
      n_restart = functional_restarts,
      seed = seed + 3000 + b
    )
    t_rep <- get_S(
      dat_t$y[seq_len(n_obs)],
      dat_t$X[seq_len(n_obs), , drop = FALSE]
    )
    t_center <- get_S(
      c(y_obs, dat_t$y),
      rbind(X_obs, dat_t$X)
    )

    dat_b <- generate_pseudo_sample_bayes_cmdstan(
      X_obs = X_obs,
      beta_draws = fit_b$beta_draws,
      tau2_draws = fit_b$var_draws,
      n_rep = m_future_ppc,
      df = df,
      seed = seed + 5000 + b
    )
    b_rep <- get_S(
      dat_b$y[seq_len(n_obs)],
      dat_b$X[seq_len(n_obs), , drop = FALSE]
    )
    b_center <- get_S(
      c(y_obs, dat_b$y),
      rbind(X_obs, dat_b$X)
    )

    list(
      g_rep = g_rep,
      g_center = g_center,
      t_rep = t_rep,
      t_center = t_center,
      b_rep = b_rep,
      b_center = b_center
    )
  }

  reps <- safe_mc_lapply(
    X = seq_len(n_mc_pvalue),
    FUN = one_rep,
    n_cores = if (isTRUE(parallel)) n_cores else 1L
  )

  S_rep_g <- vapply(reps, function(z) z$g_rep, numeric(1))
  S_center_g <- vapply(reps, function(z) z$g_center, numeric(1))
  S_rep_t <- vapply(reps, function(z) z$t_rep, numeric(1))
  S_center_t <- vapply(reps, function(z) z$t_center, numeric(1))
  S_rep_b <- vapply(reps, function(z) z$b_rep, numeric(1))
  S_center_b <- vapply(reps, function(z) z$b_center, numeric(1))

  out_g <- pbppc_from_statistics(
    S_obs = S_obs,
    S_rep = S_rep_g,
    S_center = S_center_g,
    n_obs = n_obs,
    model_label = fit_g$display_label,
    functional_label = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    p_value_type = p_value_type,
    reference_label = fun_obj$reference_label,
    n_mc_pvalue = n_mc_pvalue,
    m_future_ppc = m_future_ppc
  )

  out_t <- pbppc_from_statistics(
    S_obs = S_obs,
    S_rep = S_rep_t,
    S_center = S_center_t,
    n_obs = n_obs,
    model_label = fit_t$display_label,
    functional_label = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    p_value_type = p_value_type,
    reference_label = fun_obj$reference_label,
    n_mc_pvalue = n_mc_pvalue,
    m_future_ppc = m_future_ppc
  )

  out_b <- pbppc_from_statistics(
    S_obs = S_obs,
    S_rep = S_rep_b,
    S_center = S_center_b,
    n_obs = n_obs,
    model_label = fit_b$display_label,
    functional_label = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    p_value_type = p_value_type,
    reference_label = fun_obj$reference_label,
    n_mc_pvalue = n_mc_pvalue,
    m_future_ppc = m_future_ppc
  )

  tab <- rbind(out_g$summary, out_t$summary, out_b$summary)
  rownames(tab) <- NULL

  list(
    functional = fun_obj$functional_label,
    diagnostic_class = fun_obj$diagnostic_class,
    reference = fun_obj$reference_label,
    gaussian_mp = out_g,
    t_mp = out_t,
    bayes_t = out_b,
    table = tab
  )
}

run_pbppc_suite_once <- function(y_obs, X_obs,
                                 fit_g, fit_t, fit_b,
                                 diagnostics,
                                 df = 5,
                                 functional_restarts = 10,
                                 parallel = FALSE,
                                 n_cores = 1L,
                                 seed0 = 2026) {
  res_list <- vector("list", length(diagnostics))
  tab_list <- vector("list", length(diagnostics))

  for (j in seq_along(diagnostics)) {
    spec <- diagnostics[[j]]
    spec_full <- modifyList(
      list(
        y_obs = y_obs,
        X_obs = X_obs,
        fit_g = fit_g,
        fit_t = fit_t,
        fit_b = fit_b,
        df = df,
        functional_restarts = functional_restarts,
        parallel = parallel,
        n_cores = n_cores,
        seed = seed0 + 100000 * j
      ),
      spec
    )

    res_j <- do.call(run_pbppc_three_engines, spec_full)
    res_list[[j]] <- res_j

    tab_j <- res_j$table
    tab_j$diagnostic_id <- j
    tab_list[[j]] <- tab_j
  }

  table_all <- do.call(rbind, tab_list)
  rownames(table_all) <- NULL

  list(results = res_list, table = table_all, diagnostics = diagnostics)
}

default_aids_ppc_diagnostics <- function(n_obs, q_tail = 0.995,
                                         n_mc_pvalue_chi2 = 100,
                                         n_mc_pvalue_tail = 100) {
  list(
    list(
      functional = "chi2",
      n_mc_pvalue = n_mc_pvalue_chi2,
      m_future_ppc = 2L * n_obs,
      p_value_type = "upper"
    ),
    list(
      functional = "tail",
      q_tail = q_tail,
      n_mc_pvalue = n_mc_pvalue_tail,
      m_future_ppc = 2L * n_obs,
      p_value_type = "upper"
    )
  )
}

paper_term_label <- function(term) {
  switch(
    term,
    "trt_2" = "Treatment arm (2)",
    "trt_3" = "Treatment arm (3)",
    term
  )
}

paper_diagnostic_label_text <- function(diagnostic_class) {
  switch(
    diagnostic_class,
    "chi_square_discrepancy" = "Chi-square discrepancy",
    "absolute_residual_tail" = "Absolute residual tail",
    "coefficient" = "Coefficient functional",
    diagnostic_class
  )
}

extract_coef_summary_table <- function(draws, coef_names, model_label,
                                       targets = c("trt_2", "trt_3"),
                                       level = 0.95) {
  idx <- match(targets, coef_names)
  if (any(is.na(idx))) {
    stop("Some target coefficients were not found in coef_names.")
  }
  a <- (1 - level) / 2
  out <- lapply(seq_along(targets), function(j) {
    z <- draws[, idx[j]]
    data.frame(
      model = model_label,
      term = targets[j],
      mean = mean(z),
      sd = sd(z),
      median = median(z),
      q_low = quantile(z, probs = a, names = FALSE),
      q_high = quantile(z, probs = 1 - a, names = FALSE),
      row.names = NULL
    )
  })
  do.call(rbind, out)
}

make_aids_coefficient_comparison_table <- function(fit_gaussian, fit_t_hybrid, fit_bayes,
                                                   coef_targets = c("trt_2", "trt_3"),
                                                   level = 0.95) {
  tabs <- list(
    extract_coef_summary_table(
      fit_gaussian$beta_draws,
      fit_gaussian$coef_names,
      fit_gaussian$display_label,
      targets = coef_targets,
      level = level
    ),
    extract_coef_summary_table(
      fit_t_hybrid$beta_draws,
      fit_t_hybrid$coef_names,
      fit_t_hybrid$display_label,
      targets = coef_targets,
      level = level
    ),
    extract_coef_summary_table(
      fit_bayes$beta_draws,
      fit_bayes$coef_names,
      fit_bayes$display_label,
      targets = coef_targets,
      level = level
    )
  )

  out <- do.call(rbind, tabs)
  rownames(out) <- NULL
  out
}

format_coefficient_table_for_paper <- function(tab) {
  out <- tab[, c("model", "term", "mean", "sd", "q_low", "q_high")]
  out$term <- vapply(out$term, paper_term_label, character(1))
  names(out) <- c("Model", "Coefficient", "Mean", "SD", "CI95_L", "CI95_U")
  out
}

format_ppc_table_for_paper <- function(tab) {
  out <- tab[, c(
    "diagnostic_class", "model", "S_obs", "S_rep_mean", "p_value",
    "delta_mean", "delta_sd", "pr_delta_gt_0", "n_mc_pvalue"
  )]
  out$diagnostic_class <- vapply(out$diagnostic_class, paper_diagnostic_label_text, character(1))
  names(out) <- c(
    "Diagnostic", "Model", "S_obs", "S_rep_mean", "p_value",
    "Delta_mean", "Delta_sd", "Pr_Delta_gt_0", "n_mc_pvalue"
  )
  out
}

make_ppc_summary_table <- function(tab) {
  keep <- tab[, c("diagnostic_class", "model", "p_value", "delta_mean", "delta_sd")]
  chi <- keep[keep$diagnostic_class == "chi_square_discrepancy", , drop = FALSE]
  tail <- keep[keep$diagnostic_class == "absolute_residual_tail", , drop = FALSE]

  models <- c("MGP-Gaussian-hybrid", "MGP-Student-t-hybrid", "Bayes-Student-t")
  out <- data.frame(
    Model = c("GPE", "TPE", "BTPE"),
    S_chi2_p_value = NA_real_,
    S_chi2_AvgDiff = NA_real_,
    S_chi2_StdDiff = NA_real_,
    S_tail_p_value = NA_real_,
    S_tail_AvgDiff = NA_real_,
    S_tail_StdDiff = NA_real_,
    row.names = NULL
  )

  for (i in seq_along(models)) {
    chi_i <- chi[chi$model == models[i], , drop = FALSE]
    tail_i <- tail[tail$model == models[i], , drop = FALSE]
    if (nrow(chi_i) != 1L || nrow(tail_i) != 1L) {
      stop("Could not construct the PPC summary row for model: ", models[i])
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

fit_aids_models <- function(filepath = "../data/AIDS.csv",
                            B_mp = 10000,
                            df_model = 5,
                            N_add_hybrid = 100,
                            bayes_iter_sampling = 10000,
                            bayes_iter_warmup = 2000,
                            bayes_chains = 1,
                            bayes_parallel_chains = 1,
                            seed_mp = 383,
                            seed_bayes = 381) {
  prep <- read_aids_data(filepath)

  y_obs <- prep$y
  X_obs <- prep$X

  g_hybrid_time <- time_with_value(
    gaussian_regression_mp(
      y_obs = y_obs,
      X_obs = X_obs,
      B = B_mp,
      N_add = N_add_hybrid,
      seed = seed_mp
    )
  )
  fit_g_hybrid <- g_hybrid_time$value

  t_hybrid_time <- time_with_value(
    student_t_regression_mp(
      y_obs = y_obs,
      X_obs = X_obs,
      B = B_mp,
      N_add = N_add_hybrid,
      df = df_model,
      n_restart = 10,
      seed = seed_mp
    )
  )
  fit_t_hybrid <- t_hybrid_time$value

  bayes_cmd_time <- time_with_value(
    fit_bayes_student_t_cmdstan(
      y_obs = y_obs,
      X_obs = X_obs,
      df_fixed = df_model,
      sigma0 = 10,
      iter_sampling = bayes_iter_sampling,
      iter_warmup = bayes_iter_warmup,
      chains = bayes_chains,
      parallel_chains = bayes_parallel_chains,
      seed = seed_bayes
    )
  )
  fit_b_cmd <- bayes_cmd_time$value
  fit_b <- extract_cmdstan_draws(fit_b_cmd, coef_names = prep$coef_names)

  runtime_table_application <- data.frame(
    method = c(
      fit_g_hybrid$display_label,
      fit_t_hybrid$display_label,
      fit_b$display_label
    ),
    elapsed_sec = c(
      g_hybrid_time$elapsed,
      t_hybrid_time$elapsed,
      bayes_cmd_time$elapsed
    ),
    row.names = NULL
  )

  list(
    prepared = prep,
    gaussian_hybrid = fit_g_hybrid,
    t_hybrid = fit_t_hybrid,
    bayes_cmdstan_fit = fit_b_cmd,
    bayes_t = fit_b,
    runtime_table_application = runtime_table_application
  )
}

run_aids_application <- function(filepath = "../data/AIDS.csv",
                                 q_tail = 0.995,
                                 B_mp = 10000,
                                 df_model = 5,
                                 N_add_hybrid = 100,
                                 bayes_iter_sampling = 10000,
                                 bayes_iter_warmup = 2000,
                                 bayes_chains = 1,
                                 bayes_parallel_chains = 1,
                                 diagnostics = NULL,
                                 functional_restarts = 10,
                                 ppc_parallel = FALSE,
                                 ppc_n_cores = recommended_ppc_cores(4L),
                                 seed_mp = 383,
                                 seed_bayes = 381,
                                 seed_ppc = 2026) {
  fits <- fit_aids_models(
    filepath = filepath,
    B_mp = B_mp,
    df_model = df_model,
    N_add_hybrid = N_add_hybrid,
    bayes_iter_sampling = bayes_iter_sampling,
    bayes_iter_warmup = bayes_iter_warmup,
    bayes_chains = bayes_chains,
    bayes_parallel_chains = bayes_parallel_chains,
    seed_mp = seed_mp,
    seed_bayes = seed_bayes
  )

  prep <- fits$prepared
  y_obs <- prep$y
  X_obs <- prep$X

  if (is.null(diagnostics)) {
    diagnostics <- default_aids_ppc_diagnostics(
      n_obs = length(y_obs),
      q_tail = q_tail
    )
  }

  ppc_single <- run_pbppc_suite_once(
    y_obs = y_obs,
    X_obs = X_obs,
    fit_g = fits$gaussian_hybrid,
    fit_t = fits$t_hybrid,
    fit_b = fits$bayes_t,
    diagnostics = diagnostics,
    df = df_model,
    functional_restarts = functional_restarts,
    parallel = ppc_parallel,
    n_cores = ppc_n_cores,
    seed0 = seed_ppc
  )

  coefficient_table <- make_aids_coefficient_comparison_table(
    fit_gaussian = fits$gaussian_hybrid,
    fit_t_hybrid = fits$t_hybrid,
    fit_bayes = fits$bayes_t,
    coef_targets = c("trt_2", "trt_3")
  )

  coefficient_table_paper <- format_coefficient_table_for_paper(coefficient_table)
  ppc_table_paper <- format_ppc_table_for_paper(ppc_single$table)
  ppc_summary <- make_ppc_summary_table(ppc_single$table)

  contour_spec <- list(
    draws1 = fits$gaussian_hybrid$beta_draws,
    draws2 = fits$t_hybrid$beta_draws,
    draws3 = fits$bayes_t$beta_draws,
    coef_names = prep$coef_names,
    coef_x = "trt_2",
    coef_y = "trt_3",
    labels = c(
      fits$gaussian_hybrid$display_label,
      fits$t_hybrid$display_label,
      fits$bayes_t$display_label
    ),
    xlab = expression(beta[2]),
    ylab = expression(beta[3])
  )

  caption_suggestions <- list(
    figure_contour = paste(
      "Posterior 95% probability contours for the coefficients of treatment arms (2) and (3)",
      "under Bayes-Student-t, MGP-Student-t-hybrid, and MGP-Gaussian-hybrid."
    ),
    table_coefficients = paste(
      "Posterior summaries for the treatment-arm coefficients under Bayes-Student-t,",
      "MGP-Student-t-hybrid, and MGP-Gaussian-hybrid. Intervals are 95% posterior intervals."
    ),
    table_ppc = paste(
      "PB-PPC summaries for the normalized chi-square discrepancy and absolute",
      "standardized-residual tail statistic. The two-panel PPC figure plots S_tail",
      "in the left panel and S_chi^2 in the right panel."
    )
  )

  formula_suggestions <- list(
    predictive_resampling = "Z_{n+1} ~ F(. | x_{1:n}), Z_{n+2} ~ F(. | x_{1:n}, Z_{n+1}), ...",
    finite_pbp = "For m future draws, P_{n,m} is the empirical measure of the observed n points and the m predictive draws.",
    observed_and_replicate = "S_obs = S(P_n), S_rep^{(b)} = S(P_rep,n^{(b)}).",
    p_value = "One-sided PB-PPC p-values are used for non-negative discrepancy diagnostics.",
    delta = "Delta_{S,n}^{(b)} = sqrt(n){S(P_{n,m}^{(b)}) - S(P_n)}.",
    chi2 = "S_chi2(D) = n^{-1} sum_i (y_i - x_i' beta_G_obs)^2 / sigma2_G_obs.",
    tail = "S_tail(D) = Q_0.995(|r_1|, ..., |r_n|), where r_i are standardized residuals under the observed-data Gaussian working fit.",
    robust_regression = "Y_i = X_i' beta + tau epsilon_i, epsilon_i ~ t_nu, with nu = 5 fixed."
  )

  diagnostic_lookup <- function(name, field) {
    hit <- vapply(diagnostics, function(z) identical(z$functional, name), logical(1))
    if (!any(hit)) {
      return(NA_character_)
    }
    value <- diagnostics[[which(hit)[1L]]][[field]]
    if (is.null(value)) NA_character_ else as.character(value)
  }

  application_metadata <- data.frame(
    field = c(
      "dataset",
      "n_obs",
      "outcome",
      "n_baseline_covariates",
      "n_treatment_dummies",
      "treatment_dummies",
      "n_regression_coefficients_including_intercept",
      "continuous_variables",
      "df_student_t",
      "mp_draws",
      "hybrid_future_steps",
      "bayes_iter_sampling",
      "bayes_iter_warmup",
      "bayes_chains",
      "ppc_replications_chi2",
      "ppc_replications_tail",
      "ppc_future_draws_chi2",
      "ppc_future_draws_tail",
      "tail_quantile",
      "seed_mp",
      "seed_bayes",
      "seed_ppc"
    ),
    value = c(
      "ACTG175 / AIDS Clinical Trials Group Study 175",
      as.character(prep$n_obs),
      prep$outcome_name,
      as.character(prep$p - 1L - 3L),
      "3",
      "trt_1, trt_2, trt_3; zidovudine reference group",
      as.character(prep$p),
      "Continuous covariates and outcome are assumed pre-standardized in AIDS.csv",
      as.character(df_model),
      as.character(B_mp),
      as.character(N_add_hybrid),
      as.character(bayes_iter_sampling),
      as.character(bayes_iter_warmup),
      as.character(bayes_chains),
      diagnostic_lookup("chi2", "n_mc_pvalue"),
      diagnostic_lookup("tail", "n_mc_pvalue"),
      diagnostic_lookup("chi2", "m_future_ppc"),
      diagnostic_lookup("tail", "m_future_ppc"),
      as.character(q_tail),
      as.character(seed_mp),
      as.character(seed_bayes),
      as.character(seed_ppc)
    ),
    stringsAsFactors = FALSE
  )

  list(
    prepared = prep,
    runtime_table_application = fits$runtime_table_application,
    gaussian_hybrid = fits$gaussian_hybrid,
    t_hybrid = fits$t_hybrid,
    bayes_t = fits$bayes_t,
    bayes_cmdstan_fit = fits$bayes_cmdstan_fit,
    coefficient_table = coefficient_table,
    coefficient_table_paper = coefficient_table_paper,
    contour_spec = contour_spec,
    ppc_single = ppc_single,
    ppc_table_paper = ppc_table_paper,
    ppc_summary = ppc_summary,
    application_metadata = application_metadata,
    caption_suggestions = caption_suggestions,
    formula_suggestions = formula_suggestions
  )
}

write_named_list_txt <- function(obj, filepath) {
  con <- file(filepath, open = "wt")
  on.exit(close(con), add = TRUE)
  for (nm in names(obj)) {
    writeLines(paste0("[", nm, "]"), con)
    val <- obj[[nm]]
    if (is.list(val) && !is.data.frame(val)) {
      for (sub_nm in names(val)) {
        writeLines(paste0(sub_nm, ":"), con)
        writeLines(as.character(val[[sub_nm]]), con)
        writeLines("", con)
      }
    } else {
      writeLines(as.character(val), con)
      writeLines("", con)
    }
  }
}

format_table_number <- function(x, digits) {
  ifelse(is.na(x), "", formatC(x, format = "f", digits = digits))
}

make_ppc_summary_latex <- function(tab,
                                  caption = NULL,
                                  label = "tab:ppc-pe-aids") {
  rows <- apply(tab, 1, function(z) {
    paste(
      z[["Model"]],
      format_table_number(as.numeric(z[["S_chi2_p_value"]]), 2),
      format_table_number(as.numeric(z[["S_chi2_AvgDiff"]]), 4),
      format_table_number(as.numeric(z[["S_chi2_StdDiff"]]), 4),
      format_table_number(as.numeric(z[["S_tail_p_value"]]), 2),
      format_table_number(as.numeric(z[["S_tail_AvgDiff"]]), 4),
      format_table_number(as.numeric(z[["S_tail_StdDiff"]]), 4),
      sep = " & "
    )
  })
  rows <- paste0(rows, " \\\\")

  lines <- c(
    "\\begin{table}[!ht]",
    "\\centering"
  )
  if (!is.null(caption)) {
    lines <- c(lines, paste0("\\caption{", caption, "}"))
  }
  if (!is.null(label)) {
    lines <- c(lines, paste0("\\label{", label, "}"))
  }
  c(
    lines,
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    " & \\multicolumn{3}{c}{Panel A: $S_{\\chi^2}$} & \\multicolumn{3}{c}{Panel B: $S_{\\mathrm{tail}}$} \\\\",
    "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
    "Model & $p$-value & AvgDiff & StdDiff & $p$-value & AvgDiff & StdDiff \\\\",
    "\\midrule",
    rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )
}

save_aids_application_outputs <- function(out,
                                          output_dir = "application_results") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  saveRDS(out, file = file.path(output_dir, "aids_application_results.rds"))

  write.csv(out$runtime_table_application,
            file = file.path(output_dir, "runtime_table_application.csv"),
            row.names = FALSE)
  write.csv(out$coefficient_table,
            file = file.path(output_dir, "coefficient_table_raw.csv"),
            row.names = FALSE)
  write.csv(out$coefficient_table_paper,
            file = file.path(output_dir, "coefficient_table_paper.csv"),
            row.names = FALSE)
  write.csv(out$ppc_single$table,
            file = file.path(output_dir, "ppc_single_table_raw.csv"),
            row.names = FALSE)
  write.csv(out$ppc_table_paper,
            file = file.path(output_dir, "ppc_single_table_paper.csv"),
            row.names = FALSE)
  write.csv(out$ppc_summary,
            file = file.path(output_dir, "ppc_pe_summary.csv"),
            row.names = FALSE)
  writeLines(
    make_ppc_summary_latex(
      out$ppc_summary,
      caption = paste(
        "PPC-PE results by model. GPE and TPE denote predictive engines based",
        "on Gaussian and Student-t regression models, respectively; BTPE denotes",
        "the Bayes posterior predictive engine based on the Student-t regression model."
      )
    ),
    con = file.path(output_dir, "ppc_pe_summary.tex")
  )
  write.csv(out$application_metadata,
            file = file.path(output_dir, "application_metadata.csv"),
            row.names = FALSE)
  write_named_list_txt(out$caption_suggestions,
                       file.path(output_dir, "caption_suggestions.txt"))
  write_named_list_txt(out$formula_suggestions,
                       file.path(output_dir, "formula_suggestions.txt"))

  invisible(out)
}

run_aids_application_and_save <- function(output_dir = "application_results", ...) {
  out <- run_aids_application(...)
  save_aids_application_outputs(out = out, output_dir = output_dir)
  invisible(out)
}

coef_axis_label <- function(coef_name) {
  switch(
    coef_name,
    "trt_2" = expression(beta[2]),
    "trt_3" = expression(beta[3]),
    coef_name
  )
}

pbppc_axis_label <- function() {
  expression(sqrt(n) * "{" * S(P[n, m]^{(b)}) - S(P[n]) * "}")
}

plot_joint_contours_three <- function(draws1, draws2, draws3,
                                      coef_names,
                                      coef_x, coef_y,
                                      labels = c("GPE", "TPE", "BTPE"),
                                      ltys = c(3, 2, 1),
                                      n_grid = 100,
                                      main = "",
                                      xlab = NULL,
                                      ylab = NULL) {
  ix <- match(coef_x, coef_names)
  iy <- match(coef_y, coef_names)
  if (any(is.na(c(ix, iy)))) {
    stop("coef_x / coef_y not found in coef_names.")
  }

  if (is.null(xlab)) xlab <- coef_axis_label(coef_x)
  if (is.null(ylab)) ylab <- coef_axis_label(coef_y)

  x_all <- c(draws1[, ix], draws2[, ix], draws3[, ix])
  y_all <- c(draws1[, iy], draws2[, iy], draws3[, iy])
  xlim <- range(x_all)
  ylim <- range(y_all)

  kd1 <- MASS::kde2d(draws1[, ix], draws1[, iy], n = n_grid, lims = c(xlim, ylim))
  kd2 <- MASS::kde2d(draws2[, ix], draws2[, iy], n = n_grid, lims = c(xlim, ylim))
  kd3 <- MASS::kde2d(draws3[, ix], draws3[, iy], n = n_grid, lims = c(xlim, ylim))

  contour_level <- function(kd, prob = 0.95) {
    z <- as.vector(kd$z)
    z_sorted <- z[order(z, decreasing = TRUE)]
    mass <- cumsum(z_sorted) / sum(z_sorted)
    z_sorted[min(which(mass >= prob))]
  }

  plot(
    NA,
    xlim = xlim,
    ylim = ylim,
    xlab = xlab,
    ylab = ylab,
    main = if (is.null(main)) "" else main
  )

  contour(kd1, levels = contour_level(kd1, 0.95), add = TRUE,
          drawlabels = FALSE, lty = ltys[1], lwd = 2)
  contour(kd2, levels = contour_level(kd2, 0.95), add = TRUE,
          drawlabels = FALSE, lty = ltys[2], lwd = 2)
  contour(kd3, levels = contour_level(kd3, 0.95), add = TRUE,
          drawlabels = FALSE, lty = ltys[3], lwd = 2)

  legend("topright", legend = labels, lty = ltys, lwd = 2, bty = "n")
}

plot_ppc_hist_three <- function(delta1, delta2, delta3,
                                labels = c("GPE", "TPE", "BTPE"),
                                main = "") {
  rng <- range(c(delta1, delta2, delta3))

  d1 <- density(delta1, from = rng[1], to = rng[2])
  d2 <- density(delta2, from = rng[1], to = rng[2])
  d3 <- density(delta3, from = rng[1], to = rng[2])

  plot(
    d1,
    xlim = rng,
    ylim = c(0, max(d1$y, d2$y, d3$y)),
    lty = 1,
    lwd = 2,
    main = if (is.null(main)) "" else main,
    xlab = pbppc_axis_label(),
    ylab = "Density"
  )
  lines(d2, lty = 2, lwd = 2)
  lines(d3, lty = 3, lwd = 2)
  abline(v = 0, lty = 2, lwd = 2)
  legend("topright", legend = labels, lty = 1:3, lwd = 2, bty = "n")
}

plot_ppc_stat_density_three <- function(S_obs, S1, S2, S3,
                                        labels = c("GPE", "TPE", "BTPE"),
                                        main = "",
                                        xlab = "replicate statistic") {
  rng <- range(c(S_obs, S1, S2, S3))

  d1 <- density(S1, from = rng[1], to = rng[2])
  d2 <- density(S2, from = rng[1], to = rng[2])
  d3 <- density(S3, from = rng[1], to = rng[2])

  plot(
    d1,
    xlim = rng,
    ylim = c(0, max(d1$y, d2$y, d3$y)),
    lty = 1,
    lwd = 2,
    main = if (is.null(main)) "" else main,
    xlab = xlab,
    ylab = "Density"
  )
  lines(d2, lty = 2, lwd = 2)
  lines(d3, lty = 3, lwd = 2)
  abline(v = S_obs, lty = 2, lwd = 2)
  legend(
    "topright",
    legend = c(labels, "Observed"),
    lty = c(1, 2, 3, 2),
    lwd = c(2, 2, 2, 2),
    bty = "n"
  )
}

find_ppc_result_by_class <- function(ppc_single, diagnostic_class) {
  idx <- vapply(ppc_single$results, function(z) z$diagnostic_class, character(1))
  hit <- which(idx == diagnostic_class)
  if (length(hit) != 1L) {
    stop("Could not uniquely identify diagnostic_class = '", diagnostic_class, "'.")
  }
  ppc_single$results[[hit]]
}

save_plot_multi <- function(plot_fun, stem, figure_dir,
                            width = 7, height = 6, save_png = TRUE,
                            aliases = character()) {
  pdf_file <- file.path(figure_dir, paste0(stem, ".pdf"))
  grDevices::pdf(pdf_file, width = width, height = height)
  plot_fun()
  grDevices::dev.off()

  if (isTRUE(save_png)) {
    png_file <- file.path(figure_dir, paste0(stem, ".png"))
    grDevices::png(png_file, width = width, height = height,
                   units = "in", res = 300)
    plot_fun()
    grDevices::dev.off()
  }

  aliases <- setdiff(unique(aliases), stem)
  for (alias in aliases) {
    file.copy(pdf_file, file.path(figure_dir, paste0(alias, ".pdf")), overwrite = TRUE)
    if (isTRUE(save_png)) {
      file.copy(
        file.path(figure_dir, paste0(stem, ".png")),
        file.path(figure_dir, paste0(alias, ".png")),
        overwrite = TRUE
      )
    }
  }
}

plot_aids_application_from_saved <- function(results_file = file.path("application_results", "aids_application_results.rds"),
                                             figure_dir = file.path(dirname(results_file), "figures"),
                                             width = 7,
                                             height = 6,
                                             save_png = TRUE,
                                             make_delta_plots = TRUE) {
  out <- readRDS(results_file)
  dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
  engine_labels <- c("GPE", "TPE", "BTPE")

  save_plot_multi(
    plot_fun = function() {
      plot_joint_contours_three(
        draws1 = out$contour_spec$draws1,
        draws2 = out$contour_spec$draws2,
        draws3 = out$contour_spec$draws3,
        coef_names = out$contour_spec$coef_names,
        coef_x = out$contour_spec$coef_x,
        coef_y = out$contour_spec$coef_y,
        labels = engine_labels,
        main = "",
        xlab = out$contour_spec$xlab,
        ylab = out$contour_spec$ylab
      )
    },
    stem = "contour_bayes_t_vs_mgp_t_vs_mgp_gaussian",
    figure_dir = figure_dir,
    width = width,
    height = height,
    save_png = save_png
  )

  chi2_res <- find_ppc_result_by_class(out$ppc_single, "chi_square_discrepancy")
  tail_res <- find_ppc_result_by_class(out$ppc_single, "absolute_residual_tail")

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

      plot_ppc_stat_density_three(
        S_obs = tail_res$gaussian_mp$S_obs,
        S1 = tail_res$gaussian_mp$S_rep,
        S2 = tail_res$t_mp$S_rep,
        S3 = tail_res$bayes_t$S_rep,
        labels = engine_labels,
        main = "",
        xlab = expression(S[tail])
      )

      plot_ppc_stat_density_three(
        S_obs = chi2_res$gaussian_mp$S_obs,
        S1 = chi2_res$gaussian_mp$S_rep,
        S2 = chi2_res$t_mp$S_rep,
        S3 = chi2_res$bayes_t$S_rep,
        labels = engine_labels,
        main = "",
        xlab = expression(S[chi^2])
      )
    },
    stem = "ppc_replicate_density_two_panel",
    figure_dir = figure_dir,
    width = 12,
    height = 5,
    save_png = save_png,
    aliases = c("ppc_pe_density_two_panel")
  )

  if (isTRUE(make_delta_plots)) {
    save_plot_multi(
      plot_fun = function() {
        plot_ppc_hist_three(
          chi2_res$gaussian_mp$delta,
          chi2_res$t_mp$delta,
          chi2_res$bayes_t$delta,
          labels = engine_labels,
          main = ""
        )
      },
      stem = "ppc_chi2_delta_density",
      figure_dir = figure_dir,
      width = width,
      height = height,
      save_png = save_png
    )

    save_plot_multi(
      plot_fun = function() {
        plot_ppc_hist_three(
          tail_res$gaussian_mp$delta,
          tail_res$t_mp$delta,
          tail_res$bayes_t$delta,
          labels = engine_labels,
          main = ""
        )
      },
      stem = "ppc_tail_delta_density",
      figure_dir = figure_dir,
      width = width,
      height = height,
      save_png = save_png
    )
  }

  invisible(out)
}

main <- function() {
  out <- run_aids_application_and_save(
    output_dir = "application_results",
    filepath = "../data/AIDS.csv",
    q_tail = 0.995,
    ppc_parallel = TRUE,
    ppc_n_cores = 4
  )

  print(out$runtime_table_application)
  print(out$coefficient_table_paper)
  print(out$ppc_table_paper)
  print(out$ppc_summary)
  print(out$application_metadata)
  print(out$caption_suggestions)
  print(out$formula_suggestions)

  plot_aids_application_from_saved(
    results_file = file.path("application_results", "aids_application_results.rds")
  )

  invisible(out)
}

if (sys.nframe() == 0) {
  main()
}
