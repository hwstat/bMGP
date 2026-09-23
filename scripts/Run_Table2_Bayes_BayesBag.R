#!/usr/bin/env Rscript
## =========================================================================
## Table 2, Bayes and BayesBag rows of the linear ridge block (Section 4.1.1);
## with --p=300 the same rows of the appendix table, Table 7.
## Sources src/high_dimension/.
##
## The data generator, the replication seeds, the posterior-summary code and
## the aggregation code are the ones in src/high_dimension/ that
## Run_Tables23_HighDimension.R uses, so these rows are computed on exactly the
## data sets that produced the MGP and bMGP rows of the same tables.
##
## Methods
##   Bayes     conjugate Gaussian ridge posterior N(mu, Sigma) from
##             analytical_ridge_posterior(X, y, sigma2, tau2) on the full
##             data; n_draws = 1000 samples; 2.5% / 97.5% sample quantiles.
##   BayesBag  n_boot = 50 row-wise nonparametric bootstrap resamples, the
##             same closed-form posterior on each resample, draws_per_boot =
##             20 samples per resample, 1000 pooled draws (Huggins and
##             Miller, 2024).  The resampling and the seed bookkeeping are
##             the engine's: BayesBag is pool_bpbp_draws() called with a
##             fit function that draws from the conjugate posterior, so it
##             sees exactly the resamples bMGP sees.
##   MGP/bMGP  the published pipeline re-run on the same data.  These rows
##             are already in the paper; they are off by default and come
##             back with --methods=bayes,bayesbag,mgp,bmgp.
##
## This script writes the CSVs only.  The Markdown tables, the paper-style CSV
## and ridge_bayesbag_table.tex are all rendered by
## Make_Table2_Bayes_BayesBag.R from ridge_bayesbag_summary.csv, so the
## formatting lives in one place.
##
## Usage: Rscript Run_Table2_Bayes_BayesBag.R [--mode=all --p=200,300 --repeats=80]
##   Rscript Run_Table2_Bayes_BayesBag.R                # everything, defaults
##   Rscript Run_Table2_Bayes_BayesBag.R --mode=metrics # coverage CSVs only
##   Rscript Run_Table2_Bayes_BayesBag.R --mode=timing  # timing CSV only
##   Rscript Run_Table2_Bayes_BayesBag.R --p=300        # the appendix table
##   Rscript Make_Table2_Bayes_BayesBag.R               # then the tables
##
## Options (defaults in brackets)
##   --mode=metrics|timing|all   [all]
##   --p=200,300                 [200,300]   dimensions to run
##   --n=250                     [250]       sample size
##   --repeats=80                [80]        replicated data sets R
##   --n_draws=1000              [1000]      posterior draws, Bayes
##   --n_boot=50                 [50]        bootstrap resamples, BayesBag
##   --draws_per_boot=20         [20]        draws per resample, BayesBag
##   --seed=123                  [123]       base seed of the pipeline
##   --scenarios=well,miss       [well,miss] m = 0 and m = 4
##   --methods=bayes,bayesbag    [bayes,bayesbag]  add mgp,bmgp to re-run
##                                                 the published rows
##   --workers=auto              [auto]      set to 1 for a single core
##   --parallel=true             [true]
##   --timing_keep_existing=true [true]      in timing mode, carry over the
##                                           rows of methods not run now
##   --out=<script dir>/output/table2_bayes_bayesbag
##
## For a single-core run, start R with the BLAS thread count pinned:
##   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
##     MKL_NUM_THREADS=1 Rscript Run_Table2_Bayes_BayesBag.R --mode=timing \
##     --workers=1 --parallel=false
##
## Runtime: the metrics sweep at the defaults is about two minutes on one core;
## adding mgp,bmgp to --methods re-runs the published rows and costs an hour.
## =========================================================================

script_dir_of_this_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 1L) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
}

SCRIPT_DIR <- script_dir_of_this_file()
ENGINE_DIR <- file.path(SCRIPT_DIR, "..", "src", "high_dimension")

## Read before this script or the engine touches the environment, so the
## timing rows record the thread count the shell actually set rather than the
## "1" that configure_parallel_environment() puts there later.
BLAS_THREADS_AT_STARTUP <- Sys.getenv("OPENBLAS_NUM_THREADS", unset = "unset")

METHOD_LABELS <- c(bayes = "Bayes", bayesbag = "BayesBag",
                   mgp = "MGP", bmgp = "bMGP")
SCENARIO_M <- c(well = 0, miss = 4)

load_engine <- function() {
  suppressPackageStartupMessages({
    library(MASS)
    library(Matrix)
  })
  for (f in c("design_parametric_models.R", "simulation_runner.R",
              "double_bootstrap_runner.R", "well_miss_comparison.R")) {
    source(file.path(ENGINE_DIR, f))
  }
  invisible(TRUE)
}

## --- configuration ------------------------------------------------------
## Rebuilds the config the published driver hands to the ridge scenarios:
## profile "medium" gives repeats = 80, n_boot = 50, paths_per_boot = 20,
## B = 1000, T = 5000, n = 250, p = 200, p_star = 50, sigma2 = 1, tau2 = 1.

ridge_variant_config <- function(scenario, n, p, seed, out_dir,
                                 extra_params = list()) {
  params <- utils::modifyList(
    list(n = as.integer(n), p = as.integer(p), save_draws = "false"),
    extra_params
  )
  base <- simulation_config(
    model = "all",
    profile = "medium",
    design = "normal",
    out_dir = out_dir,
    seed = as.integer(seed),
    run_gibbs = FALSE,
    params = params
  )
  comparison_variant_config(base, paste0("linear_ridge_", scenario))
}

repeat_seed_of <- function(config, r) config$seed + 100000L * (as.integer(r) - 1L)

## --- the two Bayesian baselines ----------------------------------------

sample_mvnorm_cov <- function(mu, Sigma, n_draws, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  p <- length(mu)
  R <- chol_with_jitter(Sigma)
  Z <- matrix(stats::rnorm(p * n_draws), nrow = p, ncol = n_draws)
  mu + crossprod(R, Z)
}

bayes_ridge_draws <- function(data, config, n_draws, seed) {
  sigma2 <- data$sigma2
  tau2 <- config_num(config, "tau2", 1)
  post <- analytical_ridge_posterior(data$X, data$y, sigma2, tau2)
  list(
    draws = sample_mvnorm_cov(post$mu, post$Sigma, n_draws, seed = seed),
    beta_n = post$mu,
    Sigma_n = post$Sigma
  )
}

## Adapter with the signature pool_bpbp_draws() and run_bpbp_repeat() expect,
## so that the Bayesian baselines use the engine's resampling and the engine's
## seed bookkeeping instead of a copy of them.
##
## The offset matters.  pool_bpbp_draws() hands bag b the fit seed
## seed + 2000 b and draws that bag's rows with seed + 1000 b, and the engine's
## own fit functions never call set.seed() on the seed they are given: they
## offset it, set.seed(seed + 1) for the design draw and seed + 2 for the
## recursion (double_bootstrap_runner.R lines 477 and 492).  That offset is
## what keeps a fit seed off every bootstrap seed, since seed + 2000 b + 1 and
## seed + 2000 b + 2 are never equal to seed + 1000 b' for whole b, b'.  An
## adapter that passed the seed straight to set.seed() would give bag b the
## RNG state that drew the rows of bag 2 b.  T_steps is ignored: the conjugate
## posterior has no forward horizon.
bayes_fit_draws <- function(data, config, n_paths, T_steps = NULL, seed) {
  bayes_ridge_draws(data, config, n_draws = n_paths, seed = seed + 1L)
}

bayesbag_ridge_draws <- function(data, config, n_boot, draws_per_boot, seed) {
  pool_bpbp_draws(
    data, config, bayes_fit_draws,
    n_boot = n_boot,
    paths_per_boot = draws_per_boot,
    T_steps = NULL,
    seed = seed
  )
}

## --- one replicated data set -------------------------------------------

draws_for_method <- function(method, data, config, settings, repeat_seed) {
  if (identical(method, "bayes")) {
    ## repeat_seed + 10 is the seed run_bpbp_repeat() gives the unbagged fit.
    return(bayes_fit_draws(
      data, config,
      n_paths = settings$n_draws,
      seed = repeat_seed + 10L
    )$draws)
  }
  if (identical(method, "bayesbag")) {
    return(bayesbag_ridge_draws(
      data, config,
      n_boot = settings$n_boot,
      draws_per_boot = settings$draws_per_boot,
      seed = repeat_seed + 20L
    )$draws)
  }
  if (identical(method, "mgp")) {
    return(draws_linear_from_data(
      data, config,
      n_paths = settings$total_paths,
      T_steps = settings$T_steps,
      seed = repeat_seed + 10L
    )$draws)
  }
  if (identical(method, "bmgp")) {
    return(pool_bpbp_draws(
      data, config, draws_linear_from_data,
      n_boot = settings$n_boot,
      paths_per_boot = settings$draws_per_boot,
      T_steps = settings$T_steps,
      seed = repeat_seed + 20L
    )$draws)
  }
  stop("Unknown method: ", method, call. = FALSE)
}

run_one_repeat <- function(r, config, model_id, methods, settings,
                           level = 0.95) {
  repeat_seed <- repeat_seed_of(config, r)
  data <- linear_comparison_data_from_config(config, repeat_seed)

  rows <- lapply(methods, function(method) {
    draws <- draws_for_method(method, data, config, settings, repeat_seed)
    summary <- posterior_summary_from_draws(draws, data$beta, level = level)
    summarize_parameter_sets(
      summary, model_id, METHOD_LABELS[[method]], r, data$active_indices
    )
  })

  list(repeat_id = r, raw_rows = do.call(rbind, rows))
}

## --- one scenario -------------------------------------------------------

run_scenario <- function(scenario, n, p, settings, out_dir) {
  config <- ridge_variant_config(
    scenario, n, p, settings$seed, out_dir,
    extra_params = list(
      workers = settings$workers,
      parallel = if (settings$parallel) "true" else "false"
    )
  )
  model_id <- paste0("linear_ridge_", scenario)
  repeats <- settings$repeats

  local_settings <- settings
  local_settings$total_paths <- config_int(config, "B", 1000L)
  local_settings$T_steps <- config_int(config, "T", 5000L)

  workers <- bpbp_worker_count(config, repeats)
  chunk_size <- bpbp_chunk_size(config, workers, repeats)
  message("scenario=", model_id, "; p=", p, "; n=", n,
          "; repeats=", repeats, "; methods=",
          paste(settings$methods, collapse = ","),
          "; workers=", workers)

  raw_rows <- list()
  repeat_ids <- seq_len(repeats)
  chunks <- split(repeat_ids, ceiling(seq_along(repeat_ids) / chunk_size))
  done <- 0L

  for (chunk in chunks) {
    results <- parallel_lapply_lb(
      chunk,
      function(r) {
        run_one_repeat(r, config, model_id, settings$methods, local_settings)
      },
      workers = min(workers, length(chunk)),
      config = config
    )
    for (res in results) raw_rows[[length(raw_rows) + 1L]] <- res$raw_rows
    done <- done + length(chunk)
    message("[", model_id, " p=", p, "] repeats ", done, " / ", repeats,
            " at ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
  }

  raw <- do.call(rbind, raw_rows)
  agg <- aggregate_comparison_rows(raw)

  label <- function(df) {
    df$scenario <- scenario
    df$m <- SCENARIO_M[[scenario]]
    df$n <- as.integer(n)
    df$p <- as.integer(p)
    df$p_star <- config_int(config, "p_star", NA_integer_)
    df$sigma2 <- config_num(config, "sigma2", NA_real_)
    df$tau2 <- config_num(config, "tau2", NA_real_)
    df$base_seed <- config$seed
    first <- c("scenario", "m", "n", "p")
    df[c(first, setdiff(names(df), first))]
  }

  list(raw = label(raw), aggregate = label(agg))
}


## --- timing -------------------------------------------------------------

time_one_replication <- function(scenario, n, p, settings, out_dir, r = 1L) {
  config <- ridge_variant_config(
    scenario, n, p, settings$seed, out_dir,
    extra_params = list(parallel = "false", workers = "1")
  )
  repeat_seed <- repeat_seed_of(config, r)
  data <- linear_comparison_data_from_config(config, repeat_seed)

  local_settings <- settings
  local_settings$total_paths <- config_int(config, "B", 1000L)
  local_settings$T_steps <- config_int(config, "T", 5000L)

  rows <- lapply(settings$methods, function(method) {
    invisible(gc(FALSE))
    timing <- system.time({
      draws <- draws_for_method(method, data, config, local_settings,
                                repeat_seed)
    })
    rm(draws)
    data.frame(
      scenario = scenario,
      m = SCENARIO_M[[scenario]],
      n = as.integer(n),
      p = as.integer(p),
      repeat_id = as.integer(r),
      method = METHOD_LABELS[[method]],
      elapsed_sec = as.numeric(timing[["elapsed"]]),
      user_sec = as.numeric(timing[["user.self"]]),
      sys_sec = as.numeric(timing[["sys.self"]]),
      cores_used = 1L,
      blas_threads = BLAS_THREADS_AT_STARTUP,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

## Timing runs are often restricted to a few methods, so the rows of the
## methods that were not run are carried over from the previous file instead
## of being dropped.  --timing_keep_existing=false starts a fresh table.
merge_timing_rows <- function(new_rows, path) {
  if (!file.exists(path)) return(new_rows)
  old <- utils::read.csv(path, stringsAsFactors = FALSE)
  if (!identical(sort(names(old)), sort(names(new_rows)))) {
    stop("Existing ", basename(path), " has different columns; rerun with ",
         "--timing_keep_existing=false", call. = FALSE)
  }
  key <- function(df) paste(df$p, df$scenario, df$method, sep = "|")
  carried <- old[!(key(old) %in% key(new_rows)), names(new_rows), drop = FALSE]
  if (nrow(carried) > 0L) {
    message("Carried over ", nrow(carried), " timing row(s) for ",
            paste(sort(unique(carried$method)), collapse = ", "))
  }
  out <- rbind(new_rows, carried)
  out[order(out$p, match(out$scenario, c("well", "miss")),
            match(out$method, METHOD_LABELS)), , drop = FALSE]
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
cli_int <- function(cli, key, default) as.integer(cli_value(cli, key, default))
cli_list <- function(cli, key, default) {
  strsplit(cli_value(cli, key, default), ",", fixed = TRUE)[[1]]
}
cli_flag <- function(cli, key, default) {
  tolower(cli_value(cli, key, if (default) "true" else "false")) %in%
    c("1", "true", "yes", "y")
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cli <- parse_cli(args)
  load_engine()

  out_dir <- cli_value(cli, "out",
                       file.path(SCRIPT_DIR, "output",
                                 "table2_bayes_bayesbag"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  mode <- tolower(cli_value(cli, "mode", "all"))
  p_grid <- as.integer(cli_list(cli, "p", "200,300"))
  n_value <- cli_int(cli, "n", 250L)
  scenarios <- tolower(cli_list(cli, "scenarios", "well,miss"))
  methods <- tolower(cli_list(cli, "methods", "bayes,bayesbag"))
  unknown <- setdiff(methods, names(METHOD_LABELS))
  if (length(unknown) > 0L) {
    stop("Unknown --methods entries: ", paste(unknown, collapse = ", "),
         call. = FALSE)
  }

  settings <- list(
    seed = cli_int(cli, "seed", 123L),
    repeats = cli_int(cli, "repeats", 80L),
    n_draws = cli_int(cli, "n_draws", 1000L),
    n_boot = cli_int(cli, "n_boot", 50L),
    draws_per_boot = cli_int(cli, "draws_per_boot", 20L),
    methods = methods,
    workers = cli_value(cli, "workers", "auto"),
    parallel = cli_flag(cli, "parallel", TRUE),
    timing_keep_existing = cli_flag(cli, "timing_keep_existing", TRUE)
  )

  configure_parallel_environment(list(params = list(single_thread_blas = "true")))

  if (mode %in% c("all", "metrics")) {
    raw_all <- list()
    agg_all <- list()
    for (p_value in p_grid) {
      for (scenario in scenarios) {
        res <- run_scenario(scenario, n_value, p_value, settings, out_dir)
        raw_all[[length(raw_all) + 1L]] <- res$raw
        agg_all[[length(agg_all) + 1L]] <- res$aggregate
      }
    }
    raw <- do.call(rbind, raw_all)
    agg <- do.call(rbind, agg_all)
    rownames(raw) <- NULL
    rownames(agg) <- NULL

    utils::write.csv(raw, file.path(out_dir, "ridge_bayesbag_raw.csv"),
                     row.names = FALSE)
    utils::write.csv(agg, file.path(out_dir, "ridge_bayesbag_summary.csv"),
                     row.names = FALSE)
    message("Wrote metric outputs to ", out_dir)
    message("Run: Rscript Make_Table2_Bayes_BayesBag.R   for the tables and the .tex")
  }

  if (mode %in% c("all", "timing")) {
    timing_rows <- list()
    for (p_value in p_grid) {
      for (scenario in scenarios) {
        timing_rows[[length(timing_rows) + 1L]] <-
          time_one_replication(scenario, n_value, p_value, settings, out_dir)
      }
    }
    timing <- do.call(rbind, timing_rows)
    timing_path <- file.path(out_dir, "ridge_bayesbag_timing.csv")
    if (settings$timing_keep_existing) {
      timing <- merge_timing_rows(timing, timing_path)
    }
    rownames(timing) <- NULL
    utils::write.csv(timing, timing_path, row.names = FALSE)
    message("Wrote timing output to ", out_dir)
  }

  capture.output(utils::sessionInfo(),
                 file = file.path(out_dir, "session_info.txt"))
  invisible(NULL)
}

## Set TABLE2_BAYES_BAYESBAG_NO_MAIN=1 to source this file for its functions
## only.
if (!nzchar(Sys.getenv("TABLE2_BAYES_BAYESBAG_NO_MAIN")) &&
      (sys.nframe() == 0L || identical(environment(), globalenv()))) {
  main()
}
