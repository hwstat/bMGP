## Shared helpers for the spike-and-slab timing scripts.
## Sourced by scripts/Run_Table8_Timing_SpikeSlab.R and
## scripts/Make_Table8_Timing.R.
##
## It holds the command line parser, the engine loader, the scenario
## configuration builder, the machine and load-average probes, the CSV
## appender, the timing helper and the effective-sample-size estimator.
## Like the other files under src/, it resolves no paths of its own: the entry
## script passes the directory of the engine files to load_timing_engine().

## --- engine -------------------------------------------------------------

TIMING_ENGINE_FILES <- c("design_parametric_models.R", "gibbs_linear_spikeslab.R",
                         "simulation_runner.R", "double_bootstrap_runner.R",
                         "well_miss_comparison.R")

load_timing_engine <- function(engine_dir) {
  suppressPackageStartupMessages({
    library(MASS)
    library(Matrix)
  })
  if (!dir.exists(engine_dir)) {
    stop("Engine folder not found: ", engine_dir, call. = FALSE)
  }
  for (f in TIMING_ENGINE_FILES) source(file.path(engine_dir, f))
  invisible(TRUE)
}

## --- command line -------------------------------------------------------

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
  value <- tolower(as.character(cli_value(cli, key, NA_character_)))
  if (is.na(value)) default else value %in% c("1", "true", "yes", "y")
}

## A bare file name is taken relative to out_dir; an absolute or ~ path is used
## as it stands.
out_path_of <- function(name, out_dir) {
  if (grepl("^(/|~)", name)) name else file.path(out_dir, name)
}

## --- scenario configuration ---------------------------------------------
## Exactly what Run_Tables23_HighDimension.R builds: the driver forces
## model = all and profile = medium, keeps design = normal and seed = 123 from
## parse_simulation_args(), and turns draw saving off for the sweep.

scenario_config <- function(model_id, n_value = NA, p_value = NA,
                            out_dir = file.path(tempdir(), "timing_scratch")) {
  params <- list(save_draws = "false")
  if (!is.na(n_value)) params$n <- as.integer(n_value)
  if (!is.na(p_value)) params$p <- as.integer(p_value)
  base <- simulation_config(
    model = "all",
    profile = "medium",
    design = "normal",
    out_dir = out_dir,
    seed = 123L,
    run_gibbs = FALSE,
    params = params
  )
  comparison_variant_config(base, model_id)
}

## m is the heteroskedasticity multiplier of the paper: m = 0 under the
## homoskedastic DGP, m = hetero_multiplier (4) under the misspecified one.
scenario_m <- function(config) {
  if (identical(config_value(config, "scenario", "well"), "miss")) {
    config_num(config, "hetero_multiplier", 4)
  } else {
    0
  }
}

repeat_seed_of <- function(config, rep_index) {
  config$seed + 100000L * (as.integer(rep_index) - 1L)
}

## --- machine probes ------------------------------------------------------

machine_row <- function() {
  info <- Sys.info()
  data.frame(
    r_version = R.version.string,
    platform = R.version$platform,
    sysname = paste(info[["sysname"]], info[["release"]]),
    logical_cores = detect_logical_cores(),
    openblas_num_threads = Sys.getenv("OPENBLAS_NUM_THREADS", "unset"),
    omp_num_threads = Sys.getenv("OMP_NUM_THREADS", "unset"),
    veclib_maximum_threads = Sys.getenv("VECLIB_MAXIMUM_THREADS", "unset"),
    mkl_num_threads = Sys.getenv("MKL_NUM_THREADS", "unset"),
    stringsAsFactors = FALSE
  )
}

cpu_of <- function(timing) {
  as.numeric(timing[["user.self"]]) + as.numeric(timing[["sys.self"]])
}

## One-minute load average, recorded before and after every timed run so a
## contended run is recognisable afterwards.  /proc/loadavg where it exists,
## uptime otherwise, and NA on any failure: the load is a diagnostic and must
## not stop a run.
load_average <- function() {
  out <- try({
    if (file.exists("/proc/loadavg")) {
      scan("/proc/loadavg", what = numeric(), n = 1L, quiet = TRUE)
    } else {
      txt <- suppressWarnings(system2("uptime", stdout = TRUE,
                                      stderr = FALSE))[1]
      hit <- regmatches(txt, regexpr("load averages?:.*$", txt))
      as.numeric(strsplit(trimws(sub("load averages?:", "", hit)),
                          "[ ,]+")[[1]][1])
    }
  }, silent = TRUE)
  if (inherits(out, "try-error") || length(out) != 1L || !is.finite(out)) {
    return(NA_real_)
  }
  as.numeric(out)
}

## --- timing --------------------------------------------------------------
## proc.time() deltas rather than system.time(), whose default gcFirst = TRUE
## runs a full garbage collection before the expression and charges several
## seconds of collector time to whatever it wraps.  Never nest these: a timer
## inside a timed block double counts.

timed <- function(expr) {
  start <- proc.time()
  value <- expr
  spent <- proc.time() - start
  list(value = value,
       seconds = as.numeric(spent[["elapsed"]]),
       cpu_seconds = as.numeric(spent[["user.self"]] + spent[["sys.self"]]))
}

## --- output --------------------------------------------------------------

append_csv <- function(rows, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  append <- file.exists(path)
  utils::write.table(rows, path, sep = ",", row.names = FALSE,
                     col.names = !append, append = append, qmethod = "double")
  message("Wrote ", path)
  invisible(path)
}

## --- effective sample size ----------------------------------------------
## Geyer's initial positive sequence estimator, the same algorithm as
## ess_geyer() in src/cyclone/summarize_posteriors.py: autocovariance by FFT
## normalised by N, pairs (0,1), (2,3), ..., truncation at the first
## non-positive pair, then the initial monotone step.

autocovariance_fft <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  x <- x - mean(x)
  nfft <- 2^ceiling(log2(2 * n))
  f <- stats::fft(c(x, rep(0, nfft - n)))
  out <- Re(stats::fft(f * Conj(f), inverse = TRUE)) / nfft
  out[seq_len(n)] / n
}

ess_geyer <- function(x) {
  x <- as.numeric(x)
  n <- length(x)
  if (n < 4L) return(as.numeric(n))
  g <- autocovariance_fft(x)
  if (g[1] <= 0) return(as.numeric(n))
  k <- n %/% 2L
  gam <- g[seq(1L, 2L * k - 1L, by = 2L)] + g[seq(2L, 2L * k, by = 2L)]
  neg <- which(gam <= 0)
  m <- if (length(neg) > 0L) neg[1L] - 1L else length(gam)
  if (m == 0L) return(as.numeric(n))
  gam <- cummin(gam[seq_len(m)])
  sigma2 <- -g[1] + 2 * sum(gam)
  if (sigma2 <= 0) return(as.numeric(n))
  as.numeric(n * g[1] / sigma2)
}

## ESS over every coefficient, split by the active set.  No subsetting: all p
## columns are passed through the estimator.
ess_summary <- function(beta_samples, active_indices) {
  p <- ncol(beta_samples)
  active <- sort(unique(active_indices[active_indices >= 1L &
                                         active_indices <= p]))
  inactive <- setdiff(seq_len(p), active)
  ess <- vapply(seq_len(p), function(j) ess_geyer(beta_samples[, j]),
                numeric(1))
  data.frame(
    n_draws = nrow(beta_samples),
    p = p,
    n_active = length(active),
    n_inactive = length(inactive),
    ess_active_min = if (length(active)) min(ess[active]) else NA_real_,
    ess_active_median = if (length(active)) stats::median(ess[active]) else NA_real_,
    ess_active_max = if (length(active)) max(ess[active]) else NA_real_,
    ess_inactive_min = if (length(inactive)) min(ess[inactive]) else NA_real_,
    ess_inactive_median = if (length(inactive)) stats::median(ess[inactive]) else NA_real_,
    ess_inactive_max = if (length(inactive)) max(ess[inactive]) else NA_real_,
    ess_all_min = min(ess),
    ess_all_median = stats::median(ess),
    stringsAsFactors = FALSE
  )
}
