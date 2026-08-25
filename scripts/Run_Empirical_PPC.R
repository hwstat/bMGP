## ACTG175 PPC-PE with replicate sample size m = n (Appendix E.2), reusing the
## saved bagged-calibration results. Usage: Rscript Run_Empirical_PPC.R
src <- readLines("Run_Empirical_ACTG175.R")
stop_at <- grep("^if \\(sys\\.nframe\\(\\) == 0\\)", src)
stopifnot(length(stop_at) == 1L)
eval(parse(text = paste(src[seq_len(stop_at - 1L)], collapse = "\n")), envir = globalenv())

res <- readRDS(file.path("double_bootstrap_results",
                         "aids_double_bootstrap_calibration.rds"))
app  <- res$standard_app
prep <- app$prepared
y_obs <- prep$y
X_obs <- prep$X
n_obs <- length(y_obs)

diagnostics <- default_aids_ppc_diagnostics(n_obs = n_obs, q_tail = 0.995)
for (j in seq_along(diagnostics)) diagnostics[[j]]$m_future_ppc <- as.integer(n_obs)

cat("n = ", n_obs, ";  m_future_ppc = ",
    diagnostics[[1]]$m_future_ppc, " (was ", 2L * n_obs, ")\n", sep = "")

ppc_single <- run_pbppc_suite_once(
  y_obs       = y_obs,
  X_obs       = X_obs,
  fit_g       = app$gaussian_hybrid,
  fit_t       = app$t_hybrid,
  fit_b       = app$bayes_t,
  diagnostics = diagnostics,
  df          = 5,
  parallel    = FALSE,
  n_cores     = 1L,
  seed0       = 2026
)

dir.create("application_results", showWarnings = FALSE)
saveRDS(list(ppc_single = ppc_single),
        file = file.path("application_results", "aids_application_results.rds"))

print(ppc_single$table[, intersect(
  c("diagnostic_class", "engine", "S_obs", "S_rep_mean", "p_value", "m_future_ppc"),
  names(ppc_single$table))])
