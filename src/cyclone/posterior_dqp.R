# Save the DQP posterior draws for the cyclone data and reduce them to the
# per-level slope in Year.
#
# The chain itself is not re-implemented here. This script calls
# `time_dqp.R` next to it with exactly the command line
# Run_Table10_Cyclone_Timing.sh uses, plus the argument that script already
# accepts for `save(res, ...)` after the chain. The settings of the chain live
# in `time_dqp.R` alone; this script reads them back out of the JSON that run
# writes.
#
# The reduction then follows run_scripts/7.2_cyclone_dqp/
# 7.2_cyclone_dqp_process.R of the qmp repository:
#
#   * quantiles_of_interest = sort(unlist(quantile_levels)), 15 pinned levels;
#   * DQP_post[[j]][i, ] = res$Q[[i]][j + 1, ], so row j+1 of the stored matrix
#     is level j and the columns are the 26 distinct years;
#   * ind = seq(nburn + 1, niter, nthin);
#   * beta1_linearized[i, t] = coef(lm(Q_draw ~ unique(x)))[2].
#
# The burn-in, the thinning and the quantile levels are not restated here.
# They are read out of the JSON `time_dqp.R` writes, so the chain and its
# reduction cannot drift apart.
#
# The last step is done in closed form, slope = cov(Q, x) / var(x) over the 26
# years, which is the same number `lm` returns. The script checks this against
# `coef(lm(...))` on a sample of draws and stops if the two disagree by more
# than 1e-10.
#
# y stays on the WmaxST scale throughout the DQP code and the regression is on
# the raw years, so these slopes are already in WmaxST units per year and need
# no rescaling.
#
# Outputs, all under scripts/output/cyclone/posterior/:
#   dqp_cyclone_result.RData   the raw `res` object from the chain
#   dqp_slope_draws.rds        matrix, n_kept x 15, slope draws per level
#   dqp_slope_draws.csv        the same as CSV, for summarize_posteriors.py
#   dqp_Q_draws.rds            array, n_kept x 26 x 15, the quantile draws
#   dqp_slope_summary.csv      mean, sd and 95% equal-tailed interval per level
#   dqp_posterior_run.json     the timing record time_dqp.R writes
#
# Usage:
#   Rscript posterior_dqp.R [niter] [outdir] [init_seed]
# Driven by scripts/Run_Table11_Cyclone_Posterior.sh.

args <- commandArgs(trailingOnly = TRUE)
niter <- if (length(args) >= 1) as.integer(args[1]) else 20000L
init_seed <- if (length(args) >= 3) as.integer(args[3]) else NA_integer_
script_dir <- normalizePath(dirname(sub("^--file=", "", grep("^--file=",
                commandArgs(trailingOnly = FALSE), value = TRUE)[1])))
root <- normalizePath(file.path(script_dir, "..", ".."))
outdir <- if (length(args) >= 2) args[2] else
  file.path(root, "scripts", "output", "cyclone", "posterior")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
outdir <- normalizePath(outdir)

dqp_dir <- file.path(script_dir, "dqp")
data_path <- file.path(root, "data", "globalTCmax4.txt")
rdata_path <- file.path(outdir, "dqp_cyclone_result.RData")
json_path <- file.path(outdir, "dqp_posterior_run.json")

## 1. Run the chain through the existing timing script ------------------------

cat("running time_dqp.R with niter =", niter,
    "and posterior draws saved to", rdata_path, "\n")
call_args <- c(shQuote(file.path(script_dir, "time_dqp.R")),
               niter, 1L, shQuote(json_path),
               shQuote(dqp_dir), shQuote(data_path),
               shQuote(rdata_path))
if (!is.na(init_seed)) call_args <- c(call_args, init_seed)
status <- system2("Rscript", call_args)
if (status != 0) stop("time_dqp.R exited with status ", status)

## 2. Reduce to the per-level slope in Year -----------------------------------

load(rdata_path)  # brings in `res`

## Burn-in, thinning and the pinned levels come from the run record, not from a
## second copy of the constants.
run <- jsonlite::fromJSON(json_path)
quantiles_of_interest <- sort(as.numeric(unlist(run$quantile_levels)))
n_quantiles <- length(quantiles_of_interest)
nburn <- as.integer(run$nburn)
nthin <- as.integer(run$nthin)
stopifnot(identical(quantiles_of_interest,
                    sort(as.numeric(run$quantiles_of_interest))))

niter_stored <- length(res$Q)
stopifnot(niter_stored > nburn, nthin >= 1L)
ind <- seq(from = nburn + 1, to = niter_stored, by = nthin)
cat("from", basename(json_path), ": nburn =", nburn, " nthin =", nthin,
    " levels =", n_quantiles, " init_seed =", run$init_seed, "\n")

dat <- read.table(data_path)
dat <- dat[is.na(dat$Basin), ]
y <- dat$WmaxST
x <- dat$Year
x_prime <- unique(x)

stopifnot(nrow(res$Q[[1]]) == n_quantiles + 2,
          ncol(res$Q[[1]]) == length(x_prime))

# Q_draws[i, , t] is the posterior draw of the level-t conditional quantile
# curve over the 26 years, for kept iteration i.
Q_draws <- array(NA_real_, dim = c(length(ind), length(x_prime), n_quantiles))
for (t in seq_len(n_quantiles)) {
  for (i in seq_along(ind)) {
    Q_draws[i, , t] <- res$Q[[ind[i]]][t + 1, ]
  }
}
dimnames(Q_draws) <- list(NULL, as.character(x_prime),
                          as.character(quantiles_of_interest))

# Slope of the fitted line through the 26 year-specific quantiles.
xc <- x_prime - mean(x_prime)
denom <- sum(xc^2)
slope_draws <- matrix(NA_real_, nrow = length(ind), ncol = n_quantiles)
for (t in seq_len(n_quantiles)) {
  slope_draws[, t] <- as.vector(Q_draws[, , t] %*% xc) / denom
}
colnames(slope_draws) <- as.character(quantiles_of_interest)

# Check the closed form against lm on a sample of draws.
set.seed(1)
chk_i <- sample(seq_along(ind), min(50, length(ind)))
max_gap <- 0
for (t in seq_len(n_quantiles)) {
  for (i in chk_i) {
    b <- unname(coef(lm(Q_draws[i, , t] ~ x_prime))[2])
    max_gap <- max(max_gap, abs(b - slope_draws[i, t]))
  }
}
cat("max |closed form - lm| over", length(chk_i) * n_quantiles,
    "checks:", format(max_gap, scientific = TRUE), "\n")
if (max_gap > 1e-10) stop("closed-form slope does not match lm")

## 3. Also carry the MCMC draws of the parametric centre beta ------------------

beta_draws <- t(sapply(ind, function(i) res$beta[[i]]))
colnames(beta_draws) <- c("beta0", "beta1_centre")

## 4. Write -------------------------------------------------------------------

saveRDS(slope_draws, file.path(outdir, "dqp_slope_draws.rds"))
saveRDS(Q_draws, file.path(outdir, "dqp_Q_draws.rds"))
saveRDS(beta_draws, file.path(outdir, "dqp_beta_centre_draws.rds"))
write.csv(slope_draws, file.path(outdir, "dqp_slope_draws.csv"), row.names = FALSE)

summ <- data.frame(
  level = quantiles_of_interest,
  mean = colMeans(slope_draws),
  sd = apply(slope_draws, 2, sd),
  q025 = apply(slope_draws, 2, quantile, probs = 0.025),
  q975 = apply(slope_draws, 2, quantile, probs = 0.975),
  n_draws = nrow(slope_draws)
)
write.csv(summ, file.path(outdir, "dqp_slope_summary.csv"), row.names = FALSE)

cat("niter stored =", niter_stored, " kept =", length(ind), "\n")
print(summ, row.names = FALSE)
cat("wrote", file.path(outdir, "dqp_slope_draws.rds"), "\n")
