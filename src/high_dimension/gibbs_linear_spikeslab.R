## Gibbs sampler for the George and McCulloch continuous spike-and-slab prior
## under a Gaussian likelihood.
## Sourced by scripts/Run_Table8_Timing_SpikeSlab.R.
##
## This is the lambda = 1 special case of gibbs_studentt_spikeslab() in
## design_parametric_models.R.  In the Student-t sampler each observation
## carries a latent scale lambda_i with
##
##   lambda_i | beta ~ Gamma((nu + 1) / 2, (nu + (y_i - x_i' beta)^2 / sigma^2) / 2)
##
## so the working precision of row i is lambda_i / sigma^2.  A Gaussian
## likelihood is the same model with lambda_i identically 1, so the latent
## scale update disappears, the row weights are the constant 1 / sigma^2, and
## the cross-product crossprod(X) / sigma^2 and the score crossprod(X, y) /
## sigma^2 no longer change between iterations.  They are formed once here.
## Nothing else differs: the beta block is drawn from the same Gaussian full
## conditional through solve_spd() and rmvnorm_precision(), the inclusion
## indicators z are drawn from the same Bernoulli full conditional, and the
## random number stream is consumed in the same order, so the sampler visits
## the same states the Student-t sampler would visit if every lambda_i were
## pinned at 1.  test_gibbs_linear_matches_studentt() below checks that claim
## against the shipped Student-t sampler.
##
## The model is
##   y | beta ~ N(X beta, sigma^2 I)
##   beta_j | z_j ~ (1 - z_j) N(0, v0) + z_j N(0, v1)
##   z_j ~ Bernoulli(w)
## with sigma fixed, as in Section 4 of Sun and Fong (2026).
##
## Dependencies: solve_spd() and rmvnorm_precision() from
## design_parametric_models.R, which has to be sourced first.  Nothing else is
## needed.

gibbs_linear_spikeslab <- function(X,
                                   y,
                                   v0,
                                   v1,
                                   w,
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

  beta_samples <- matrix(NA_real_, n_keep, p)
  z_samples <- matrix(0L, n_keep, p)
  z_counts <- integer(p)

  inv_sigma2 <- 1 / sigma^2
  invv0 <- 1 / v0
  invv1 <- 1 / v1
  logw <- log(w)
  log1mw <- log1p(-w)
  logv0 <- log(v0)
  logv1 <- log(v1)
  keep <- 0L

  ## lambda = 1, so both of these are constant across iterations.
  XtX_scaled <- crossprod(X) * inv_sigma2
  Xty_scaled <- as.vector(crossprod(X, y)) * inv_sigma2

  for (iter in seq_len(total_iter)) {
    A <- XtX_scaled + diag(ifelse(z, invv1, invv0), p)
    mu <- solve_spd(A, Xty_scaled)
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

## Checks the per-iteration structure against the shipped Student-t sampler.
##
## gibbs_studentt_spikeslab() is run with its Gamma draw for the latent scales
## replaced by the constant vector of ones, which is what that draw converges
## to when the likelihood is Gaussian.  The replacement consumes no random
## numbers, matching gibbs_linear_spikeslab(), so with the same seed the two
## samplers must return bit-identical draws.  The substitution is made by
## rebinding `::` in a child of the function's own environment, so the shipped
## function itself is read only.
test_gibbs_linear_matches_studentt <- function(n = 40L, p = 15L, v0 = 0.0005,
                                               v1 = 9, w = 0.25, sigma = 1,
                                               burnin = 20L, n_keep = 30L,
                                               seed = 7L, verbose = TRUE) {
  set.seed(1L)
  X <- matrix(stats::rnorm(n * p), n, p)
  beta_true <- c(rep(2, 3), numeric(p - 3))
  y <- as.vector(X %*% beta_true) + sigma * stats::rnorm(n)

  pinned_env <- new.env(parent = environment(gibbs_studentt_spikeslab))
  pinned_env$`::` <- function(pkg, name) {
    pkg_name <- as.character(substitute(pkg))
    fun_name <- as.character(substitute(name))
    if (identical(pkg_name, "stats") && identical(fun_name, "rgamma")) {
      return(function(n, shape, scale) rep(1, n))
    }
    get(fun_name, envir = asNamespace(pkg_name))
  }
  pinned <- gibbs_studentt_spikeslab
  environment(pinned) <- pinned_env

  reference <- pinned(X, y, v0, v1, w, nu = 4, sigma = sigma,
                      burnin = burnin, n_keep = n_keep, seed = seed)
  mine <- gibbs_linear_spikeslab(X, y, v0, v1, w, sigma,
                                 burnin = burnin, n_keep = n_keep, seed = seed)

  checks <- c(
    beta_samples = identical(reference$beta_samples, mine$beta_samples),
    z_samples = identical(reference$z_samples, mine$z_samples),
    inclusion_prob = identical(reference$inclusion_prob, mine$inclusion_prob),
    iterations = identical(reference$iterations, mine$iterations),
    burnin = identical(reference$burnin, mine$burnin),
    n_keep = identical(reference$n_keep, mine$n_keep)
  )
  max_abs_diff <- max(abs(reference$beta_samples - mine$beta_samples))
  if (verbose) {
    for (name in names(checks)) {
      message(sprintf("  %-15s identical: %s", name, checks[[name]]))
    }
    message(sprintf("  max abs difference in beta draws: %.3g", max_abs_diff))
  }
  list(passed = all(checks), checks = checks, max_abs_diff = max_abs_diff)
}
