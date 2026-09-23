# Wall-clock timing for the dependent quantile pyramid (An and MacEachern) on
# the cyclone data.  Part of Table 11.
#
# The body is run_scripts/7.2_cyclone_dqp/7.2_cyclone_dqp_mcmc.R of the qmp
# repository (https://github.com/edfong/qmp), which is itself Hyoin An's code
# as modified by Edwin Fong. The model, the priors and the proposals are
# unchanged. What was added: the number of iterations is read from the command
# line so a short pilot chain can be timed, the compile step and the sampling
# step are timed separately, the result is written as JSON instead of being
# printed, and the initial quantile pyramid is drawn from a seed instead of
# from whatever state the session happens to be in (see init_seed below).
#
# The C++ sources it compiles are in dqp/ next to this file, copied from the
# ver0/ folder of that same upstream directory; see dqp/LICENSE.
#
# Usage:
#   Rscript time_dqp.R <niter> <rep> <out.json> [dqp_dir] [data_path] \
#       [save_rdata] [init_seed] [nburn]
#
#   niter       MCMC iterations (the upstream driver: 20000)
#   rep         label written into the JSON
#   out.json    where the timing record goes; resolved before setwd()
#   dqp_dir     folder holding MCMC_binary.cpp, default dqp/ next to this file
#   data_path   default data/globalTCmax4.txt in the repository
#   save_rdata  optional path for save(res, ...); resolved before setwd()
#   init_seed   seed for the initial pyramid, default 804
#   nburn       burn-in, default niter %/% 2 (10000 of 20000, which is what
#               7.2_cyclone_dqp_process.R discards)
# Driven by scripts/Run_Table10_Cyclone_Timing.sh.

script_dir_of_this_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
}

SCRIPT_DIR <- script_dir_of_this_file()
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))

args <- commandArgs(trailingOnly = TRUE)
niter <- as.integer(args[1])
rep_id <- as.integer(args[2])

# Resolve every output path while the working directory is still the caller's:
# the script setwd()s into the DQP source tree a few lines below, and a
# relative path written after that would land in the clone.  normalizePath()
# needs the file to exist, so the directory is normalized and the basename
# appended.
abs_path <- function(path) {
  if (!nzchar(path)) return(path)
  file.path(normalizePath(dirname(path), mustWork = TRUE), basename(path))
}

out_path <- abs_path(args[3])
dqp_dir <- normalizePath(if (length(args) >= 4 && nzchar(args[4])) args[4] else
                           file.path(SCRIPT_DIR, "dqp"))
data_path <- normalizePath(if (length(args) >= 5 && nzchar(args[5])) args[5] else
                             file.path(REPO_ROOT, "data", "globalTCmax4.txt"))
save_rdata <- abs_path(if (length(args) >= 6) args[6] else "")
init_seed <- if (length(args) >= 7) as.integer(args[7]) else 804L
nburn <- if (length(args) >= 8) as.integer(args[8]) else niter %/% 2L
nthin <- 1L
stopifnot(niter > nburn, nburn >= 0L, nthin >= 1L)

# One-minute load average, without assuming macOS.  Diagnostic only, so every
# failure path returns NA rather than stopping a chain that is about to run for
# a quarter of an hour.
load_1min <- function() {
  out <- try({
    if (file.exists("/proc/loadavg")) {
      scan("/proc/loadavg", what = numeric(), n = 1L, quiet = TRUE)
    } else {
      txt <- suppressWarnings(system("uptime", intern = TRUE))[1]
      nums <- regmatches(txt, regexpr("load averages?:.*$", txt))
      as.numeric(strsplit(trimws(sub("load averages?:", "", nums)),
                          "[ ,]+")[[1]][1])
    }
  }, silent = TRUE)
  if (inherits(out, "try-error") || length(out) != 1L || !is.finite(out)) {
    return(NA_real_)
  }
  as.numeric(out)
}

setwd(dqp_dir)

t_script0 <- Sys.time()
load_before <- load_1min()

t0 <- Sys.time()
Rcpp::sourceCpp("MCMC_binary.cpp")
t_compile <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

### Data set up ###
dat <- read.table(data_path) # 2098 obs, over the globe, 1981-2006
dat <- dat[is.na(dat$Basin), ] # NA = North Atlantic

y <- dat$WmaxST
x <- dat$Year

x_centered <- x - mean(x)
x_prime <- unique(x)

##### set up #####

# mu and Sigma for Gaussian process
mu <- rep(0, length(x_prime))
tausq <- 0  # zero nugget effect for smooth covariance function
sigsq <- 1  # we want a correlation matrix
r <- 1      # exponential
phi <- 5    # for discrete chose the distance parameter so the nearby points would have meaningful dependence
Sigma <- sigsq * exp(-abs(as.matrix(dist(x_prime)))^r / phi) + diag(rep(tausq), length(x_prime))

# beta prior & proposal parameters
mu_0 <- c(71, 0.5)
Sigma_0 <- matrix(c(15, 0, 0, 2), ncol = 2)
ols_res <- lm(y ~ x_centered)
Sigma_beta <- solve((t(cbind(1, x_centered))) %*% as.matrix(cbind(1, x_centered))) * summary(ols_res)$sigma^2

# sigma_x
sigma_x <- aggregate(y, by = list(x), FUN = sd)[, 2] # empirical sd vector
sigma_x <- lm(sigma_x ~ x_prime)$fit

# Quantile pyramid structure specification
quantile_levels <- list(c(0.5), c(0.25, 0.75), c(0.1, 0.35, 0.65, 0.9),
                        c(0.05, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.95))

# initial values
beta_initial <- ols_res$coefficients
mu_x_initial <- cbind(1, unique(x_centered)) %*% beta_initial
# The upstream script calls DQPbinarySampling() with no seed set, and that
# function draws through R's RNG (Rcpp::rnorm in ver0/utilities.h), so upstream
# the chain starts wherever the session's RNG state happens to be and two runs
# of the same script start in different places.  Seeding it here makes the run
# reproducible.  init_seed defaults to 804, one past the chain's own seed 803,
# so the starting pyramid and the chain do not consume the same stream.
set.seed(init_seed)
DQP_res <- DQPbinarySampling(mu, Sigma, quantile_levels, alpha_scale = 5)
DQP_initial_unit <- DQP_res$Q
DQP_Z_initial <- DQP_res$Z
DQP_Q_initial <- DQP_initial_unit
for (r in 1:nrow(DQP_Q_initial)) {
  DQP_Q_initial[r, ] <- qnorm(DQP_initial_unit[r, ], mu_x_initial, sigma_x)
}

##### Run MCMC #####
set.seed(803)
t0 <- Sys.time()
p0 <- proc.time()
res <- MCMC_param2(niter, y, as.matrix(x_centered), sigma_x, delta = 1, Sigma_beta,
                   DQP_Q_initial, DQP_Z_initial, beta_initial, quantile_levels,
                   mu, Sigma, mu_0, Sigma_0, alpha_scale = 5, n_chunk_Q = 13, n_chunk_beta = 1)
t_mcmc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
p1 <- proc.time()

t_script <- as.numeric(difftime(Sys.time(), t_script0, units = "secs"))
cpu_total <- unname((p1 - p0)["user.self"] + (p1 - p0)["sys.self"])

if (nzchar(save_rdata)) save(res, file = save_rdata)

quantiles_of_interest <- sort(unlist(quantile_levels))

rec <- list(
  method = "DQP",
  variant = "MCMC_param2 (Rcpp/RcppArmadillo)",
  rep = rep_id,
  n = length(y),
  n_unique_x = length(x_prime),
  n_quantiles = length(quantiles_of_interest),
  pyramid_levels = length(quantile_levels),
  niter = niter,
  nburn = nburn,
  nthin = nthin,
  init_seed = init_seed,
  mcmc_seed = 803L,
  quantile_levels = quantile_levels,
  quantiles_of_interest = quantiles_of_interest,
  n_chunk_Q = 13,
  n_chunk_beta = 1,
  alpha_scale = 5,
  delta = 1,
  t_compile_s = round(t_compile, 4),
  t_mcmc_s = round(t_mcmc, 4),
  t_posterior_pipeline_s = round(t_mcmc, 4),
  t_script_total_s = round(t_script, 4),
  cpu_time_s = round(cpu_total, 4),
  cpu_over_wall = round(cpu_total / t_script, 3),
  load_1min_before = round(load_before, 2),
  load_1min_after = round(load_1min(), 2),
  r_version = R.version.string,
  blas = unname(extSoftVersion()[["BLAS"]]),
  env_threads = list(
    OMP_NUM_THREADS = Sys.getenv("OMP_NUM_THREADS"),
    OPENBLAS_NUM_THREADS = Sys.getenv("OPENBLAS_NUM_THREADS"),
    VECLIB_MAXIMUM_THREADS = Sys.getenv("VECLIB_MAXIMUM_THREADS")
  ),
  n_stored_draws = length(res$Q)
)

writeLines(jsonlite::toJSON(rec, auto_unbox = TRUE, pretty = TRUE), out_path)
cat("MCMC niter =", niter, "took", round(t_mcmc, 2), "seconds\n")
