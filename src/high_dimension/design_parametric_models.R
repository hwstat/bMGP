## Model and predictive-engine definitions for the high-dimensional experiments.
## Sourced by scripts/Run_Tables23_HighDimension.R.
stable_log1pexp <- function(x) {
  ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
}

standardize_columns <- function(X) {
  X <- as.matrix(X)
  centers <- colMeans(X)
  scales <- apply(X, 2, stats::sd)
  scales[scales == 0] <- 1
  sweep(sweep(X, 2, centers, "-"), 2, scales, "/")
}

equicorr_matrix <- function(p, rho) {
  Sigma <- matrix(rho, p, p)
  diag(Sigma) <- 1
  Sigma
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

rmvnorm_precision <- function(mean, precision) {
  R <- chol_with_jitter(precision)
  as.vector(mean + backsolve(R, stats::rnorm(length(mean))))
}

thin_posterior <- function(samples, B) {
  if (B > nrow(samples)) {
    stop("B cannot exceed number of samples.", call. = FALSE)
  }
  idx <- round(seq(1, nrow(samples), length.out = B))
  samples[idx, , drop = FALSE]
}

posterior_ci <- function(samples, level = 0.95) {
  alpha <- 1 - level
  lower <- apply(samples, 2, stats::quantile, probs = alpha / 2, names = FALSE)
  upper <- apply(samples, 2, stats::quantile, probs = 1 - alpha / 2, names = FALSE)
  list(lower = lower, upper = upper)
}

coverage_indicator <- function(beta_true, lower, upper) {
  lower <= beta_true & beta_true <= upper
}

interval_length <- function(lower, upper) {
  upper - lower
}

generate_linear_data <- function(p = 400,
                                 n = 100,
                                 sigma2 = 1.0,
                                 active_ratio = 100 / 400,
                                 seed = 42,
                                 beta_scale = 3.0) {
  set.seed(seed)

  z <- sample(c(-1, 1), n, replace = TRUE)
  X <- matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  X <- X + z
  X <- standardize_columns(X)

  k <- max(1L, ceiling(p * active_ratio))
  active_indices <- sort(sample.int(p, k, replace = FALSE))

  beta <- numeric(p)
  beta[active_indices] <- beta_scale * stats::rnorm(k)
  y <- as.vector(X %*% beta + stats::rnorm(n, sd = sqrt(sigma2)))

  list(X = X, y = y, beta = beta, sigma2 = sigma2,
       active_indices = active_indices)
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

generate_gamma_loglink_data <- function(n = 10,
                                        p = 10,
                                        p_star = 5,
                                        rho = 0.0,
                                        lambda = 1.0,
                                        shape = 1.0,
                                        seed = 4994) {
  set.seed(seed)
  Sigma <- equicorr_matrix(p, rho)
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  X <- standardize_columns(X)

  beta <- numeric(p)
  active_indices <- sample.int(p, p_star, replace = FALSE)
  beta[active_indices] <- sample(c(-1, 1), p_star, replace = TRUE) *
    sqrt(lambda / p_star)

  eta <- as.vector(X %*% beta)
  mu <- exp(eta)
  y <- stats::rgamma(n, shape = shape, scale = mu / shape)

  list(X = X, y = y, beta = beta, shape = shape,
       active_indices = active_indices)
}

map_gamma_loglink_optim <- function(X,
                                    y,
                                    lambda_ridge,
                                    alpha,
                                    beta_init = NULL,
                                    maxit = 1000) {
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)
  beta0 <- if (is.null(beta_init)) numeric(p) else as.vector(beta_init)

  objective <- function(beta) {
    eta <- as.vector(X %*% beta)
    eta <- pmin(pmax(eta, -700), 700)
    mu <- exp(eta)
    sum(alpha * y / mu + alpha * eta) + 0.5 * lambda_ridge * sum(beta^2)
  }

  gradient <- function(beta) {
    eta <- as.vector(X %*% beta)
    eta <- pmin(pmax(eta, -700), 700)
    mu <- exp(eta)
    as.vector(crossprod(X, alpha * (1 - y / mu)) + lambda_ridge * beta)
  }

  fit <- stats::optim(beta0, objective, gradient, method = "BFGS",
                      control = list(maxit = maxit, reltol = 1e-10))
  fit$par
}

precompute_paths_gamma <- function(Sigma_init, T, alpha, X_samples) {
  X_samples <- as.matrix(X_samples)
  p <- nrow(X_samples)

  Sigma_x_samples <- matrix(NA_real_, p, T)
  quad_samples <- numeric(T)
  Sigma <- as.matrix(Sigma_init)

  for (t in seq_len(T)) {
    x <- X_samples[, t]
    v <- as.vector(Sigma %*% x)
    s <- sum(x * v)
    denom <- 1 + alpha * s

    Sigma <- Sigma - (alpha / denom) * tcrossprod(v)
    Sigma_x_samples[, t] <- v / denom
    quad_samples[t] <- s
  }

  list(Sigma_x_samples = Sigma_x_samples, quad_samples = quad_samples)
}

recursive_update_gamma <- function(beta_matrix,
                                   Sigma_x_samples,
                                   quad_samples,
                                   alpha,
                                   seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  beta_matrix <- as.matrix(beta_matrix)
  T <- ncol(Sigma_x_samples)
  B <- ncol(beta_matrix)

  G <- matrix(stats::rgamma(T * B, shape = alpha, scale = 1),
              nrow = T, ncol = B)
  B_mat <- G - alpha
  A_mat <- sweep(Sigma_x_samples, 2, sqrt(1 + alpha * quad_samples), "*")

  beta_matrix + A_mat %*% B_mat
}

generate_logistic_data <- function(n = 1000,
                                   p = 500,
                                   p_star = 5,
                                   rho = 0.0,
                                   lambda = 1.0,
                                   seed = 1234) {
  set.seed(seed)
  Sigma <- equicorr_matrix(p, rho)
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  X <- standardize_columns(X)

  beta <- numeric(p)
  active_indices <- seq_len(p_star)
  beta[active_indices] <- sample(c(-1, 1), p_star, replace = TRUE) *
    sqrt(lambda / p_star)

  eta <- as.vector(X %*% beta)
  prob <- stats::plogis(eta)
  y <- stats::rbinom(n, size = 1, prob = prob)
  while (length(unique(y)) < 2) {
    y <- stats::rbinom(n, size = 1, prob = prob)
  }

  list(X = X, y = y, beta = beta, active_indices = active_indices,
       eta = eta, true_prob = prob, lambda = lambda)
}

compute_stable_XtDX <- function(X, w) {
  X <- as.matrix(X)
  w <- as.vector(w)
  crossprod(sweep(X, 1, sqrt(w), "*"))
}

map_logistic_ridge_optim <- function(X,
                                     y,
                                     lambda_ridge,
                                     beta_init = NULL,
                                     maxit = 1000) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)
  beta0 <- if (is.null(beta_init)) numeric(p) else as.vector(beta_init)
  ridge_precision <- lambda_ridge * n

  objective <- function(beta) {
    eta <- as.vector(X %*% beta)
    sum(stable_log1pexp(eta) - y * eta) +
      0.5 * ridge_precision * sum(beta^2)
  }

  gradient <- function(beta) {
    eta <- as.vector(X %*% beta)
    prob <- stats::plogis(eta)
    as.vector(crossprod(X, prob - y) + ridge_precision * beta)
  }

  fit <- stats::optim(beta0, objective, gradient, method = "BFGS",
                      control = list(maxit = maxit, reltol = 1e-10))
  fit$par
}

rpg1_truncated <- function(c, trunc = 200L) {
  c <- as.vector(c)
  k <- seq_len(trunc) - 0.5
  denom_base <- k^2
  out <- numeric(length(c))

  for (i in seq_along(c)) {
    denom <- denom_base + c[i]^2 / (4 * pi^2)
    out[i] <- sum(stats::rgamma(trunc, shape = 1, scale = 1) / denom) /
      (2 * pi^2)
  }
  out
}

bayes_logistic_ridge_gibbs <- function(X,
                                       y,
                                       tau2,
                                       n_iter,
                                       burnin,
                                       beta_init = NULL,
                                       seed = NULL,
                                       pg_trunc = 200L) {
  if (!is.null(seed)) set.seed(seed)
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)

  beta_current <- if (is.null(beta_init)) rep(1, p) else as.vector(beta_init)
  kappa <- y - 0.5
  lambda <- 1 / tau2
  Xt_kappa <- as.vector(crossprod(X, kappa))

  n_keep <- n_iter - burnin
  beta_samples <- matrix(NA_real_, n_keep, p)
  omega_samples <- matrix(NA_real_, n_keep, n)
  keep <- 0L

  for (iter in seq_len(n_iter)) {
    psi <- as.vector(X %*% beta_current)
    omega <- rpg1_truncated(psi, trunc = pg_trunc)
    Q <- compute_stable_XtDX(X, omega) + diag(lambda, p)
    mu_beta <- solve_spd(Q, Xt_kappa)
    beta_current <- rmvnorm_precision(mu_beta, Q)

    if (iter > burnin) {
      keep <- keep + 1L
      beta_samples[keep, ] <- beta_current
      omega_samples[keep, ] <- omega
    }
  }

  list(beta_samples = beta_samples, omega_samples = omega_samples)
}

precompute_paths_logistic <- function(beta_n, X_samples, Sigma_n) {
  beta_n <- as.vector(beta_n)
  X_samples <- as.matrix(X_samples)
  p <- nrow(X_samples)
  T <- ncol(X_samples)

  psi_samples <- as.vector(crossprod(X_samples, beta_n))
  prob <- stats::plogis(psi_samples)
  w_samples <- prob * (1 - prob)
  log_w_samples <- log(w_samples)

  Sigma_x_samples <- matrix(NA_real_, p, T)
  xSigma_x_samples <- numeric(T)
  Sigma <- as.matrix(Sigma_n)

  for (t in seq_len(T)) {
    x <- X_samples[, t]
    v <- as.vector(Sigma %*% x)
    s <- sum(x * v)
    w <- w_samples[t]
    denom <- 1 + w * s

    Sigma_x_samples[, t] <- v / denom
    xSigma_x_samples[t] <- s
    Sigma <- Sigma - (w / denom) * tcrossprod(v)
  }

  list(Sigma_x_samples = Sigma_x_samples,
       xSigma_x_samples = xSigma_x_samples,
       w_samples = w_samples,
       log_w_samples = log_w_samples)
}

recursive_update_logistic <- function(beta_matrix,
                                      X_samples,
                                      Sigma_x_samples,
                                      xSigma_x_samples,
                                      w_samples,
                                      log_w_samples,
                                      seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  beta_matrix <- as.matrix(beta_matrix)
  X_samples <- as.matrix(X_samples)
  T <- ncol(X_samples)
  B <- ncol(beta_matrix)

  half_log_w <- 0.5 * log_w_samples
  log1p_wxSigma_x <- log1p(w_samples * xSigma_x_samples)

  for (t in seq_len(T)) {
    x <- X_samples[, t]
    Sigma_x <- Sigma_x_samples[, t]
    psi <- as.vector(crossprod(beta_matrix, x))
    p_N <- stats::plogis(psi)
    y_N <- stats::runif(B) < p_N

    const_exp <- half_log_w[t] + 0.5 * log1p_wxSigma_x[t]
    exp_arg <- ifelse(y_N, -0.5 * psi, 0.5 * psi) + const_exp
    exp_arg <- pmin(pmax(exp_arg, -700), 700)
    scale <- ifelse(y_N, 1, -1) * exp(exp_arg)

    beta_matrix <- beta_matrix + tcrossprod(Sigma_x, scale)
  }

  beta_matrix
}

generate_robust_data <- function(n = 250,
                                 p = 500,
                                 p_star = 5,
                                 rho = 0.0,
                                 lambda = 20.0,
                                 nu = 4.0,
                                 sigma = 1.0,
                                 seed = 123) {
  set.seed(seed)
  Sigma <- equicorr_matrix(p, rho)
  X <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  X <- standardize_columns(X)

  beta <- numeric(p)
  active_indices <- seq_len(p_star)
  beta[active_indices] <- sample(c(-1, 1), p_star, replace = TRUE) *
    sqrt(lambda / p_star)

  location <- as.vector(X %*% beta)
  y <- location + sigma * stats::rt(n, df = nu)

  list(X = X, y = y, beta = beta, active_indices = active_indices,
       nu = nu, sigma = sigma)
}

map_em_studentt_spikeslab <- function(X,
                                      y,
                                      v0,
                                      v1,
                                      w,
                                      nu,
                                      sigma,
                                      beta_init = NULL,
                                      max_iter = 1000,
                                      tol = 1e-6,
                                      verbose = FALSE) {
  X <- as.matrix(X)
  y <- as.vector(y)
  p <- ncol(X)
  beta <- if (is.null(beta_init)) numeric(p) else as.vector(beta_init)

  for (iter in seq_len(max_iter)) {
    beta_old <- beta
    resid <- y - as.vector(X %*% beta)

    lambda_exp <- (nu + 1) / (nu + resid^2 / sigma^2)

    slab_log <- log(w) + stats::dnorm(beta, 0, sqrt(v1), log = TRUE)
    spike_log <- log1p(-w) + stats::dnorm(beta, 0, sqrt(v0), log = TRUE)
    z_exp <- 1 / (1 + exp(spike_log - slab_log))

    row_weight <- lambda_exp / sigma^2
    Xw <- sweep(X, 1, sqrt(row_weight), "*")
    Dinv <- z_exp / v1 + (1 - z_exp) / v0
    A <- crossprod(Xw) + diag(Dinv, p)
    b <- as.vector(crossprod(X, row_weight * y))

    beta <- tryCatch(solve_spd(A, b), error = function(e) as.vector(qr.solve(A, b)))

    if (sqrt(sum((beta - beta_old)^2)) < tol) {
      if (verbose) message("Converged after ", iter, " iterations")
      return(beta)
    }
  }

  if (verbose) message("Warning: reached maximum iterations without convergence")
  beta
}

continuation_em_studentt_spikeslab <- function(X,
                                               y,
                                               v0_grid,
                                               v1,
                                               w,
                                               nu,
                                               sigma,
                                               max_iter = 500,
                                               tol = 1e-6,
                                               verbose = FALSE) {
  beta_current <- numeric(ncol(X))

  for (i in seq_along(v0_grid)) {
    if (verbose) {
      message("Running EM for v0 = ", v0_grid[i],
              " (step ", i, "/", length(v0_grid), ")")
    }
    beta_current <- map_em_studentt_spikeslab(
      X, y, v0_grid[i], v1, w, nu, sigma,
      beta_init = beta_current, max_iter = max_iter, tol = tol,
      verbose = verbose
    )
  }

  beta_current
}

normal0_density <- function(x, v) {
  stats::dnorm(x, mean = 0, sd = sqrt(v))
}

spike_posterior_prob <- function(theta, alpha, v0, v1) {
  f0 <- normal0_density(theta, v0)
  f1 <- normal0_density(theta, v1)
  alpha * f0 / (alpha * f0 + (1 - alpha) * f1)
}

prior_hessian_diag <- function(theta, alpha, v0, v1) {
  r <- spike_posterior_prob(theta, alpha, v0, v1)
  term1 <- r / v0 + (1 - r) / v1
  term2 <- theta^2 * (1 / v0 - 1 / v1)^2 * r * (1 - r)
  term1 - term2
}

regularized_fisher <- function(theta, alpha, v0, v1) {
  diag(prior_hessian_diag(theta, alpha, v0, v1), length(theta))
}

gibbs_studentt_spikeslab <- function(X,
                                     y,
                                     v0,
                                     v1,
                                     w,
                                     nu,
                                     sigma,
                                     burnin = 10000,
                                     n_keep = 25000,
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)
  total_iter <- burnin + n_keep

  beta <- numeric(p)
  z <- stats::runif(p) < w
  lambda <- rep(1, n)

  beta_samples <- matrix(NA_real_, n_keep, p)
  z_samples <- matrix(0L, n_keep, p)
  z_counts <- integer(p)

  inv_sigma2 <- 1 / sigma^2
  shape <- 0.5 * (nu + 1)
  invv0 <- 1 / v0
  invv1 <- 1 / v1
  logw <- log(w)
  log1mw <- log1p(-w)
  logv0 <- log(v0)
  logv1 <- log(v1)
  keep <- 0L

  for (iter in seq_len(total_iter)) {
    resid <- y - as.vector(X %*% beta)
    rate <- 0.5 * (nu + resid^2 * inv_sigma2)
    lambda <- stats::rgamma(n, shape = shape, scale = 1 / rate)

    row_weight <- lambda * inv_sigma2
    Xw <- sweep(X, 1, sqrt(row_weight), "*")
    A <- crossprod(Xw) + diag(ifelse(z, invv1, invv0), p)
    b <- as.vector(crossprod(X, row_weight * y))
    mu <- solve_spd(A, b)
    beta <- rmvnorm_precision(mu, A)

    slab_log <- logw - 0.5 * logv1 - 0.5 * beta^2 * invv1
    spike_log <- log1mw - 0.5 * logv0 - 0.5 * beta^2 * invv0
    prob <- 1 / (1 + exp(spike_log - slab_log))
    z <- stats::runif(p) < prob

    if (iter > burnin) {
      keep <- keep + 1L
      beta_samples[keep, ] <- beta
      z_samples[keep, ] <- as.integer(z)
      z_counts <- z_counts + as.integer(z)
    }
  }

  list(beta_samples = beta_samples,
       z_samples = z_samples,
       inclusion_prob = z_counts / n_keep,
       iterations = total_iter,
       burnin = burnin,
       n_keep = n_keep)
}

sample_X_empirical <- function(X, T, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- as.matrix(X)
  idx <- sample.int(nrow(X), T, replace = TRUE)
  t(X[idx, , drop = FALSE])
}

precompute_paths_studentt <- function(Sigma_init, T, nu, sigma, X_samples) {
  X_samples <- as.matrix(X_samples)
  p <- nrow(X_samples)
  cnu <- (nu + 1) / ((nu + 3) * sigma^2)

  Sigma_x_samples <- matrix(NA_real_, p, T)
  quad_samples <- numeric(T)
  Sigma <- as.matrix(Sigma_init)

  for (t in seq_len(T)) {
    x <- X_samples[, t]
    v <- as.vector(Sigma %*% x)
    s <- sum(x * v)
    denom <- 1 + cnu * s

    Sigma <- Sigma - (cnu / denom) * tcrossprod(v)
    Sigma_x_samples[, t] <- v / denom
    quad_samples[t] <- s
  }

  list(Sigma_x_samples = Sigma_x_samples, quad_samples = quad_samples)
}

recursive_update_studentt <- function(theta_matrix,
                                      Sigma_x_samples,
                                      quad_samples,
                                      nu,
                                      sigma,
                                      seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  theta_matrix <- as.matrix(theta_matrix)
  T <- ncol(Sigma_x_samples)
  B <- ncol(theta_matrix)
  cnu <- (nu + 1) / ((nu + 3) * sigma^2)

  T_rvs <- matrix(stats::rt(T * B, df = nu), nrow = T, ncol = B)
  B_mat <- ((nu + 1) / sigma) * T_rvs / (nu + T_rvs^2)
  A_mat <- sweep(Sigma_x_samples, 2, sqrt(1 + cnu * quad_samples), "*")

  theta_matrix + A_mat %*% B_mat
}

recursive_update_studentt_without_correction <- function(theta_matrix,
                                                         Sigma_x_samples,
                                                         nu,
                                                         sigma,
                                                         seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  theta_matrix <- as.matrix(theta_matrix)
  T <- ncol(Sigma_x_samples)
  B <- ncol(theta_matrix)

  T_rvs <- matrix(stats::rt(T * B, df = nu), nrow = T, ncol = B)
  B_mat <- ((nu + 1) / sigma) * T_rvs / (nu + T_rvs^2)

  theta_matrix + Sigma_x_samples %*% B_mat
}

recursive_update_studentt_trace <- function(beta0,
                                            Sigma_x_samples,
                                            quad_samples,
                                            nu,
                                            sigma,
                                            seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  beta0 <- as.vector(beta0)
  p <- length(beta0)
  T <- ncol(Sigma_x_samples)
  beta_path <- matrix(NA_real_, p, T + 1)
  beta_path[, 1] <- beta0

  cnu <- (nu + 1) / ((nu + 3) * sigma^2)
  for (t in seq_len(T)) {
    z <- stats::rt(1, df = nu)
    b <- ((nu + 1) / sigma) * z / (nu + z^2)
    alpha <- sqrt(1 + cnu * quad_samples[t])
    beta_path[, t + 1] <- beta_path[, t] + alpha * b * Sigma_x_samples[, t]
  }

  beta_path
}

one_dataset_gibbs_mp <- function(n = 250,
                                 p = 500,
                                 p_star = 5,
                                 rho = 0.0,
                                 lambda = 20.0,
                                 nu = 4.0,
                                 sigma = 1.0,
                                 v0_grid = c(0.5, 0.1, 0.05, 0.01,
                                             0.005, 0.001, 0.0005),
                                 v1 = 1.0,
                                 w = 0.1,
                                 T = p + 100,
                                 B_mp = 5000,
                                 level = 0.95,
                                 seed = 123,
                                 gibbs_burnin = 10000,
                                 gibbs_n_keep = 25000,
                                 gibbs_thin = 5000,
                                 em_max_iter = 500) {
  data <- generate_robust_data(n = n, p = p, p_star = p_star, rho = rho,
                               lambda = lambda, nu = nu, sigma = sigma,
                               seed = seed)
  X <- data$X
  y <- data$y
  beta_true <- data$beta
  v0 <- tail(v0_grid, 1)

  result_gibbs <- gibbs_studentt_spikeslab(
    X, y, v0, v1, w, nu, sigma,
    burnin = gibbs_burnin, n_keep = gibbs_n_keep, seed = seed + 1
  )
  beta_samples_gibbs <- thin_posterior(result_gibbs$beta_samples, gibbs_thin)
  ci_gibbs <- posterior_ci(beta_samples_gibbs, level = level)

  beta_n <- continuation_em_studentt_spikeslab(
    X, y, v0_grid, v1, w, nu, sigma, max_iter = em_max_iter
  )
  cnu <- (nu + 1) / ((nu + 3) * sigma^2)
  D_n <- regularized_fisher(beta_n, 1 - w, v0, v1)
  Sigma_n <- solve_spd(cnu * crossprod(X) + D_n)

  theta0 <- matrix(beta_n, nrow = p, ncol = B_mp)
  set.seed(seed + 2)
  X_samples <- matrix(stats::rnorm(p * T, mean = 0, sd = 10), nrow = p)
  paths <- precompute_paths_studentt(Sigma_n, T, nu, sigma, X_samples)
  beta_samples_pmp <- recursive_update_studentt(
    theta0, paths$Sigma_x_samples, paths$quad_samples, nu, sigma, seed = seed + 3
  )
  ci_mp <- posterior_ci(t(beta_samples_pmp), level = level)

  list(beta_true = beta_true,
       CI_gibbs = ci_gibbs,
       CI_mp = ci_mp,
       len_gibbs = interval_length(ci_gibbs$lower, ci_gibbs$upper),
       len_mp = interval_length(ci_mp$lower, ci_mp$upper))
}

coverage_length_simulation <- function(B_data = 200,
                                       n = 250,
                                       p = 500,
                                       p_star = 5,
                                       seed = 123,
                                       ...) {
  cover_gibbs <- integer(p)
  cover_mp <- integer(p)
  len_gibbs_sum <- numeric(p)
  len_mp_sum <- numeric(p)
  len_gibbs_sq_sum <- numeric(p)
  len_mp_sq_sum <- numeric(p)

  for (b in seq_len(B_data)) {
    message("Dataset ", b, " / ", B_data)
    res <- one_dataset_gibbs_mp(n = n, p = p, p_star = p_star,
                                seed = seed + b, ...)

    beta_true <- res$beta_true
    cover_gibbs <- cover_gibbs +
      coverage_indicator(beta_true, res$CI_gibbs$lower, res$CI_gibbs$upper)
    cover_mp <- cover_mp +
      coverage_indicator(beta_true, res$CI_mp$lower, res$CI_mp$upper)

    len_gibbs_sum <- len_gibbs_sum + res$len_gibbs
    len_mp_sum <- len_mp_sum + res$len_mp
    len_gibbs_sq_sum <- len_gibbs_sq_sum + res$len_gibbs^2
    len_mp_sq_sum <- len_mp_sq_sum + res$len_mp^2
  }

  coverage_gibbs <- cover_gibbs / B_data
  coverage_mp <- cover_mp / B_data
  mean_len_gibbs <- len_gibbs_sum / B_data
  mean_len_mp <- len_mp_sum / B_data

  var_len_gibbs <- (len_gibbs_sq_sum / B_data - mean_len_gibbs^2) *
    (B_data / max(1, B_data - 1))
  var_len_mp <- (len_mp_sq_sum / B_data - mean_len_mp^2) *
    (B_data / max(1, B_data - 1))

  list(coverage_gibbs = coverage_gibbs,
       coverage_mp = coverage_mp,
       se_coverage_gibbs = sqrt(coverage_gibbs * (1 - coverage_gibbs) / B_data),
       se_coverage_mp = sqrt(coverage_mp * (1 - coverage_mp) / B_data),
       length_gibbs = mean_len_gibbs,
       length_mp = mean_len_mp,
       se_length_gibbs = sqrt(var_len_gibbs / B_data),
       se_length_mp = sqrt(var_len_mp / B_data))
}

summarize_set <- function(mean_vec, se_vec, idx) {
  list(mean_est = mean(mean_vec[idx]), max_se = max(se_vec[idx]))
}

report_simulation <- function(res, p_star = 5) {
  p <- length(res$coverage_gibbs)
  active <- seq_len(p_star)
  inactive <- seq.int(p_star + 1, p)

  data.frame(
    set = c("Active", "Inactive"),
    gibbs_cov = c(mean(res$coverage_gibbs[active]),
                  mean(res$coverage_gibbs[inactive])),
    gibbs_len = c(mean(res$length_gibbs[active]),
                  mean(res$length_gibbs[inactive])),
    mp_cov = c(mean(res$coverage_mp[active]),
               mean(res$coverage_mp[inactive])),
    mp_len = c(mean(res$length_mp[active]),
               mean(res$length_mp[inactive]))
  )
}
