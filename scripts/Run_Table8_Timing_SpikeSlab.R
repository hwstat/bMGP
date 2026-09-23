#!/usr/bin/env Rscript
## Table 9: wall-clock time of one replication of the spike-and-slab experiment
## of Section 4.1, for the MGP, the bMGP, one Gibbs posterior fit and the 50
## bagged Gibbs fits that BayesBag costs. Sources src/high_dimension/.
## Writes the raw CSVs only; Make_Table8_Timing.R builds the table from them.
## Usage: Rscript Run_Table8_Timing_SpikeSlab.R [--model=linear_sas_well --mode=all --p=200]
##
## The SHIPPED functions are timed, not copies of them:
##   MGP   draws_linear_sas_from_data() or draws_studentt_from_data()
##   bMGP  pool_bpbp_draws()
##   Bayes gibbs_linear_spikeslab() for the linear scenarios (the lambda = 1
##         special case) and gibbs_studentt_spikeslab() for the Student-t ones
## Each total comes from one timer that wraps one call and contains no other
## timer, and every timer is a proc.time() difference rather than
## system.time(), whose default gcFirst = TRUE charges a full garbage
## collection to whatever it wraps.
##
## The split of a replication into initialization and predictive paths is
## obtained afterwards by timing the EM continuation and the p by p covariance
## solve on their own, on the same data sets and with the same arguments, and
## subtracting from the total. Nothing inside the shipped functions is
## instrumented.
##
## The bagged data sets are drawn with the seeds pool_bpbp_draws() uses,
##   bootstrap_observed_data(data, seed = repeat_seed + 20L + 1000L * b),
## so BayesBag sees exactly the resamples the bMGP sees. --verify_bags checks
## that against the shipped pool_bpbp_draws() by recording the indices it draws
## and stops the run if they disagree.
##
## Modes
##   --mode=mgp     MGP and bMGP, one call and one timer each
##   --mode=gibbs   one Gibbs fit, repeated --reps times, plus effective
##                  sample sizes of the draws
##   --mode=bags    one Gibbs fit on each of --bags bootstrap data sets
##   --mode=all     all three, in that order (the default)
##
## Options (defaults in brackets)
##   --model=linear_sas_well  [linear_sas_well]  also linear_sas_miss,
##                            studentt_sas_well, studentt_sas_miss
##   --rep=1                  [1]        replication index in 1..80
##   --n=250 --p=200          [250, 200] the paper's design
##   --burnin=10000           [10000]    burn-in, following Sun and Fong (2026)
##   --n_keep=25000           [25000]    kept iterations, same source
##   --thin=5000              [5000]     thinned chain used for intervals
##   --reps=3                 [3]        timing repetitions of the single fit
##   --bags=50                [50]       bootstrap data sets; the bMGP uses 50
##   --split=true             [true]     also time the EM and the solve alone
##   --verify_bags=true       [true]
##   --tag=gibbs              [gibbs]    label of the single-fit and MGP rows
##   --bags_tag=bags50        [bags50]   label of the bagged-fit rows
##   --out=<script dir>/output/table8
##
## The scripts append to their CSVs, so delete the target files before rerunning
## for a clean set.
##
## Runtime. The paper's table needs eight commands, four scenarios for
## --mode=mgp and --mode=gibbs and three for --mode=bags, and takes about two
## hours in total on one core; the bagged runs are almost all of it, at roughly
## 25 minutes each. A single --mode=mgp run is about half a minute. Pin the BLAS
## threads before R starts:
##   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
##     MKL_NUM_THREADS=1 Rscript Run_Table8_Timing_SpikeSlab.R --model=linear_sas_well

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

source(file.path(ENGINE_DIR, "timing_common.R"))
load_timing_engine(ENGINE_DIR)

SUPPORTED <- c("linear_sas_well", "linear_sas_miss",
               "studentt_sas_well", "studentt_sas_miss")

## --- MGP and bMGP -------------------------------------------------------

## The initialization that the shipped fit function performs before it starts
## the predictive paths: the warm-started EM continuation down the v0 ladder,
## then the p by p covariance solve. Same arguments as the shipped code.
initialization_timer <- function(family, config) {
  if (identical(family, "linear")) {
    v0_grid <- config_num_vector(
      config, "v0_grid", c(1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    )
    v1 <- config_num(config, "v1", 9)
    w <- config_num(config, "w", config_num(config, "active_ratio", 0.25))
    function(data) {
      em <- timed(continuation_linear_sas_em(
        data$X, data$y, data$sigma2, v0_grid, v1, w,
        max_iter = config_int(config, "em_max_iter", 500L),
        tol = config_num(config, "em_tol", 1e-6)
      ))
      beta_n <- em$value
      solve_step <- timed({
        D_n <- regularized_fisher(beta_n, 1 - w, tail(v0_grid, 1L), v1)
        solve_spd(crossprod(data$X) / data$sigma2 + D_n)
      })
      list(em_seconds = em$seconds, solve_seconds = solve_step$seconds)
    }
  } else {
    v0_grid <- config_num_vector(
      config, "v0_grid", c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    )
    v1 <- config_num(config, "v1", 1)
    w <- config_num(config, "w", 0.1)
    function(data) {
      em <- timed(continuation_em_studentt_spikeslab(
        data$X, data$y, v0_grid, v1, w, data$nu, data$sigma,
        max_iter = config_int(config, "em_max_iter", 500L)
      ))
      beta_n <- em$value
      solve_step <- timed({
        cnu <- (data$nu + 1) / ((data$nu + 3) * data$sigma^2)
        D_n <- regularized_fisher(beta_n, 1 - w, tail(v0_grid, 1L), v1)
        solve_spd(cnu * crossprod(data$X) + D_n)
      })
      list(em_seconds = em$seconds, solve_seconds = solve_step$seconds)
    }
  }
}

run_mgp_bmgp <- function(spec, config, data, settings, meta_base, out_dir) {
  total_paths <- settings$total_paths
  T_steps <- settings$T_steps
  n_boot <- settings$n_boot
  paths_per_boot <- settings$paths_per_boot
  fit_fun <- comparison_fit_function(spec)
  repeat_seed <- settings$repeat_seed

  load_before_mgp <- load_average()
  mgp <- timed(fit_fun(data, config, n_paths = total_paths,
                       T_steps = T_steps, seed = repeat_seed + 10L))
  load_after_mgp <- load_average()
  message(sprintf("  MGP  total %8.2f s (cpu %.2f)", mgp$seconds,
                  mgp$cpu_seconds))

  load_before_bmgp <- load_average()
  bmgp <- timed(pool_bpbp_draws(data, config, fit_fun, n_boot = n_boot,
                                paths_per_boot = paths_per_boot,
                                T_steps = T_steps,
                                seed = repeat_seed + 20L))
  load_after_bmgp <- load_average()
  message(sprintf("  bMGP total %8.2f s (cpu %.2f)", bmgp$seconds,
                  bmgp$cpu_seconds))

  mgp_em <- NA_real_; mgp_solve <- NA_real_
  bmgp_boot <- NA_real_; bmgp_em <- NA_real_; bmgp_solve <- NA_real_
  if (settings$split) {
    time_init <- initialization_timer(spec$family, config)
    one <- time_init(data)
    mgp_em <- one$em_seconds
    mgp_solve <- one$solve_seconds

    bmgp_boot <- 0; bmgp_em <- 0; bmgp_solve <- 0
    for (b in seq_len(n_boot)) {
      boot_step <- timed(bootstrap_observed_data(
        data, seed = repeat_seed + 20L + 1000L * b
      ))
      bmgp_boot <- bmgp_boot + boot_step$seconds
      parts <- time_init(boot_step$value)
      bmgp_em <- bmgp_em + parts$em_seconds
      bmgp_solve <- bmgp_solve + parts$solve_seconds
    }
    message(sprintf(
      "  split: MGP EM %.3f s, solve %.3f s; bMGP resample %.3f s, EM %.3f s, solve %.3f s",
      mgp_em, mgp_solve, bmgp_boot, bmgp_em, bmgp_solve))
  }

  level <- config_num(config, "level", 0.95)
  mgp_summary <- posterior_summary_from_draws(mgp$value$draws, data$beta,
                                              level = level)
  bmgp_summary <- posterior_summary_from_draws(bmgp$value$draws, data$beta,
                                               level = level)

  meta <- cbind(meta_base, data.frame(
    v0_grid = as.character(config_value(config, "v0_grid", "")),
    T_steps = T_steps,
    n_boot_used = n_boot,
    paths_per_boot = paths_per_boot,
    total_paths_mgp = total_paths,
    load_average_before = load_before_mgp,
    load_average_after = load_after_bmgp,
    stringsAsFactors = FALSE
  ))

  row_of <- function(method, component, seconds, cpu_seconds = NA_real_,
                     n_paths = NA_integer_, load_before = NA_real_,
                     load_after = NA_real_) {
    cbind(meta, data.frame(
      method = method, component = component,
      seconds = as.numeric(seconds), cpu_seconds = as.numeric(cpu_seconds),
      n_paths = as.integer(n_paths),
      component_load_before = load_before, component_load_after = load_after,
      stringsAsFactors = FALSE
    ))
  }

  rows <- rbind(
    row_of("MGP", "total_wall", mgp$seconds, mgp$cpu_seconds, total_paths,
           load_before_mgp, load_after_mgp),
    row_of("MGP", "em_initialization", mgp_em, NA_real_, total_paths),
    row_of("MGP", "covariance_solve", mgp_solve, NA_real_, total_paths),
    row_of("MGP", "predictive_paths_by_difference",
           mgp$seconds - mgp_em - mgp_solve, NA_real_, total_paths),
    row_of("bMGP", "total_wall", bmgp$seconds, bmgp$cpu_seconds,
           n_boot * paths_per_boot, load_before_bmgp, load_after_bmgp),
    row_of("bMGP", "bootstrap_resample", bmgp_boot, NA_real_,
           n_boot * paths_per_boot),
    row_of("bMGP", "em_initialization", bmgp_em, NA_real_,
           n_boot * paths_per_boot),
    row_of("bMGP", "covariance_solve", bmgp_solve, NA_real_,
           n_boot * paths_per_boot),
    row_of("bMGP", "predictive_paths_by_difference",
           bmgp$seconds - bmgp_boot - bmgp_em - bmgp_solve, NA_real_,
           n_boot * paths_per_boot)
  )
  append_csv(rows, out_path_of(settings$mgp_csv, out_dir))

  message(sprintf("  mean 95%% interval length: MGP %.3f, bMGP %.3f",
                  mean(mgp_summary$interval_length),
                  mean(bmgp_summary$interval_length)))
  invisible(rows)
}

## --- the Bayesian posterior ---------------------------------------------

## The prior of the paper's own working model, read off the scenario config:
## varpi = w, slab variance v1, spike variance v0 = the last rung of the v0
## ladder, sigma fixed. For the linear family sigma^2 = sigma2 = 1; for the
## Student-t family sigma = 1 and nu = 4, the setting of Sun and Fong (2026).
gibbs_settings <- function(family, config, data) {
  if (identical(family, "linear")) {
    v0_grid <- config_num_vector(
      config, "v0_grid", c(1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    )
    list(v0 = tail(v0_grid, 1L),
         v1 = config_num(config, "v1", 9),
         w = config_num(config, "w", 0.25),
         sigma = sqrt(data$sigma2),
         nu = NA_real_)
  } else {
    v0_grid <- config_num_vector(
      config, "v0_grid", c(0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005)
    )
    list(v0 = tail(v0_grid, 1L),
         v1 = config_num(config, "v1", 1),
         w = config_num(config, "w", 0.1),
         sigma = data$sigma,
         nu = data$nu)
  }
}

gibbs_runner <- function(family, settings, burnin, n_keep) {
  if (identical(family, "linear")) {
    function(X, y, seed) {
      gibbs_linear_spikeslab(X, y, settings$v0, settings$v1, settings$w,
                             settings$sigma, burnin = burnin,
                             n_keep = n_keep, seed = seed)
    }
  } else {
    function(X, y, seed) {
      gibbs_studentt_spikeslab(X, y, settings$v0, settings$v1, settings$w,
                               settings$nu, settings$sigma, burnin = burnin,
                               n_keep = n_keep, seed = seed)
    }
  }
}

## Runs the shipped pool_bpbp_draws() at a throwaway horizon with a recording
## wrapper around bootstrap_observed_data(), and compares the indices of bag 1
## with the formula this script uses. The wrapper is installed in the global
## environment, which is where pool_bpbp_draws() resolves the name, and is
## removed again on exit.
verify_bag_seeds <- function(data, config, fit_fun, repeat_seed) {
  recorded <- list()
  original <- bootstrap_observed_data
  assign("bootstrap_observed_data", function(data, seed = NULL) {
    out <- original(data, seed = seed)
    recorded[[length(recorded) + 1L]] <<- list(
      seed = seed, indices = out$bootstrap_indices
    )
    out
  }, envir = globalenv())
  on.exit(assign("bootstrap_observed_data", original, envir = globalenv()),
          add = TRUE)

  invisible(pool_bpbp_draws(data, config, fit_fun, n_boot = 2L,
                            paths_per_boot = 2L, T_steps = 5L,
                            seed = repeat_seed + 20L))

  shipped_bag1 <- recorded[[1L]]$indices
  mine_bag1 <- original(data, seed = repeat_seed + 20L + 1000L)$bootstrap_indices
  old_bag1 <- original(data, seed = repeat_seed + 1000L)$bootstrap_indices

  list(
    shipped_seed = recorded[[1L]]$seed,
    expected_seed = repeat_seed + 20L + 1000L,
    matches_shipped = identical(shipped_bag1, mine_bag1),
    old_seed_matches = identical(shipped_bag1, old_bag1),
    n_distinct_rows = length(unique(shipped_bag1))
  )
}

run_gibbs_timing <- function(spec, config, data, settings, meta_base, out_dir,
                             do_fit, do_bags) {
  gibbs <- gibbs_settings(spec$family, config, data)
  run_gibbs <- gibbs_runner(spec$family, gibbs, settings$burnin,
                            settings$n_keep)
  repeat_seed <- settings$repeat_seed

  meta <- cbind(meta_base, data.frame(
    varpi = gibbs$w,
    v1 = gibbs$v1,
    v0 = gibbs$v0,
    sigma = gibbs$sigma,
    nu = gibbs$nu,
    burnin = settings$burnin,
    n_keep = settings$n_keep,
    thin = settings$thin,
    sampler = if (identical(spec$family, "linear")) {
      "gibbs_linear_spikeslab (lambda = 1 special case)"
    } else {
      "gibbs_studentt_spikeslab (shipped)"
    },
    bag_seed_matches_bmgp = settings$bag_seed_matches_bmgp,
    stringsAsFactors = FALSE
  ))

  rows <- list()
  ess_rows <- list()

  if (do_fit) {
    ## The same chain is run --reps times, so the spread is timing noise on
    ## this machine and not variation across chains or data sets.
    fit_seed <- repeat_seed + 1L
    for (i in seq_len(settings$reps)) {
      load_before <- load_average()
      run <- timed(run_gibbs(data$X, data$y, fit_seed))
      load_after <- load_average()
      message(sprintf("  single fit %d/%d: %.2f s (cpu %.2f), load %.2f -> %.2f",
                      i, settings$reps, run$seconds, run$cpu_seconds,
                      load_before, load_after))
      rows[[length(rows) + 1L]] <- cbind(meta, data.frame(
        kind = "single_fit", index = i, gibbs_seed = fit_seed,
        seconds = run$seconds, cpu_seconds = run$cpu_seconds,
        load_average_before = load_before, load_average_after = load_after,
        stringsAsFactors = FALSE
      ))
      if (i == settings$reps) {
        kept <- run$value$beta_samples
        thinned <- thin_posterior(kept, settings$thin)
        inclusion <- data.frame(
          mean_inclusion_active =
            mean(run$value$inclusion_prob[data$active_indices]),
          mean_inclusion_inactive =
            mean(run$value$inclusion_prob[-data$active_indices]),
          stringsAsFactors = FALSE
        )
        ess_rows[[length(ess_rows) + 1L]] <- cbind(
          meta, data.frame(chain = "kept", stringsAsFactors = FALSE),
          ess_summary(kept, data$active_indices), inclusion)
        ess_rows[[length(ess_rows) + 1L]] <- cbind(
          meta, data.frame(chain = "thinned", stringsAsFactors = FALSE),
          ess_summary(thinned, data$active_indices), inclusion)
      }
      rm(run); invisible(gc(verbose = FALSE))
    }
  }

  bag_rows <- list()
  if (do_bags && settings$bags > 0L) {
    bag_meta <- meta
    bag_meta$tag <- settings$bags_tag
    for (b in seq_len(settings$bags)) {
      boot_data <- bootstrap_observed_data(
        data, seed = repeat_seed + 20L + 1000L * b
      )
      bag_seed <- repeat_seed + 20L + 2000L * b + 1L
      load_before <- load_average()
      run <- timed(run_gibbs(boot_data$X, boot_data$y, bag_seed))
      load_after <- load_average()
      message(sprintf("  bag %d/%d: %.2f s (cpu %.2f), distinct rows %d, load %.2f -> %.2f",
                      b, settings$bags, run$seconds, run$cpu_seconds,
                      length(unique(boot_data$bootstrap_indices)),
                      load_before, load_after))
      bag_rows[[length(bag_rows) + 1L]] <- cbind(bag_meta, data.frame(
        kind = "bagged_fit", index = b, gibbs_seed = bag_seed,
        seconds = run$seconds, cpu_seconds = run$cpu_seconds,
        load_average_before = load_before, load_average_after = load_after,
        stringsAsFactors = FALSE
      ))
      rm(run, boot_data); invisible(gc(verbose = FALSE))
    }
  }

  if (length(rows) > 0L) {
    rows <- do.call(rbind, rows)
    append_csv(rows, out_path_of(settings$gibbs_csv, out_dir))
    single <- rows$seconds[rows$kind == "single_fit"]
    message(sprintf(
      "\n  one Gibbs fit (%d + %d iterations): mean %.1f s, range %.1f to %.1f s over %d runs",
      settings$burnin, settings$n_keep, mean(single), min(single), max(single),
      length(single)))
  }
  if (length(ess_rows) > 0L) {
    append_csv(do.call(rbind, ess_rows),
               out_path_of(sub("\\.csv$", "_ess.csv", settings$gibbs_csv),
                           out_dir))
  }
  if (length(bag_rows) > 0L) {
    bag_rows <- do.call(rbind, bag_rows)
    append_csv(bag_rows, out_path_of(settings$bags_csv, out_dir))
    bagged <- bag_rows$seconds
    message(sprintf(
      "  %d bagged Gibbs fits: total %.1f s, mean %.1f s, range %.1f to %.1f s",
      length(bagged), sum(bagged), mean(bagged), min(bagged), max(bagged)))
  }
  invisible(NULL)
}

## --- driver --------------------------------------------------------------

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))

  model_id <- cli_value(cli, "model", "linear_sas_well")
  if (!model_id %in% SUPPORTED) {
    stop("--model must be one of: ", paste(SUPPORTED, collapse = ", "),
         call. = FALSE)
  }
  mode <- tolower(cli_value(cli, "mode", "all"))
  if (!mode %in% c("all", "mgp", "gibbs", "bags")) {
    stop("--mode must be one of: all, mgp, gibbs, bags", call. = FALSE)
  }

  out_dir <- cli_value(cli, "out", file.path(SCRIPT_DIR, "output", "table8"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  spec <- comparison_model_spec(model_id)
  rep_index <- as.integer(cli_value(cli, "rep", "1"))
  n_value <- as.integer(cli_value(cli, "n", "250"))
  p_value <- as.integer(cli_value(cli, "p", "200"))
  tag <- cli_value(cli, "tag", "gibbs")

  ## The config's own out_dir is a scratch path the engine never writes to
  ## with save_draws = false; the CSVs of this script go to --out.
  config <- scenario_config(model_id, n_value, p_value)
  m_value <- scenario_m(config)
  repeat_seed <- repeat_seed_of(config, rep_index)
  data <- comparison_data_function(spec)(config, repeat_seed)

  ## The overrides exist only for smoke tests; the reported runs leave them
  ## alone and take the paper's values from the scenario config.
  settings <- list(
    repeat_seed = repeat_seed,
    n_boot = as.integer(cli_value(cli, "n_boot",
                                  as.character(config_int(config, "n_boot", 50L)))),
    paths_per_boot = as.integer(cli_value(
      cli, "paths_per_boot",
      as.character(config_int(config, "paths_per_boot", 20L)))),
    total_paths = as.integer(cli_value(cli, "B",
                                       as.character(config_int(config, "B", 1000L)))),
    T_steps = as.integer(cli_value(cli, "T",
                                   as.character(config_int(config, "T", 5000L)))),
    burnin = as.integer(cli_value(cli, "burnin", "10000")),
    n_keep = as.integer(cli_value(cli, "n_keep", "25000")),
    thin = as.integer(cli_value(cli, "thin", "5000")),
    reps = as.integer(cli_value(cli, "reps", "3")),
    bags = as.integer(cli_value(cli, "bags", "50")),
    split = cli_flag(cli, "split", TRUE),
    tag = tag,
    bags_tag = cli_value(cli, "bags_tag", "bags50"),
    mgp_csv = cli_value(cli, "mgp_csv", "mgp_bmgp_timings.csv"),
    gibbs_csv = cli_value(cli, "gibbs_csv", "gibbs_timings.csv"),
    bags_csv = cli_value(cli, "bags_csv", "gibbs_bags.csv"),
    bag_seed_matches_bmgp = NA
  )

  message("scenario=", model_id, "; family=", spec$family,
          "; m=", m_value, "; rep=", rep_index,
          "; repeat_seed=", repeat_seed,
          "; n=", nrow(data$X), "; p=", ncol(data$X),
          "; p_star=", length(data$active_indices),
          "; v1=", config_num(config, "v1", NA),
          "; w=", config_num(config, "w", NA),
          "; v0_grid=", config_value(config, "v0_grid", ""),
          "; T=", settings$T_steps, "; B=", settings$total_paths,
          "; n_boot=", settings$n_boot,
          "; paths_per_boot=", settings$paths_per_boot,
          "; mode=", mode, "; out=", out_dir)

  if (cli_flag(cli, "verify_bags", TRUE)) {
    verification <- verify_bag_seeds(data, config,
                                     comparison_fit_function(spec),
                                     repeat_seed)
    message("bag-seed check: shipped seed ", verification$shipped_seed,
            ", expected ", verification$expected_seed,
            "; bag 1 indices match: ", verification$matches_shipped,
            "; the previous seed (no + 20L) matches: ",
            verification$old_seed_matches,
            "; distinct rows in bag 1: ", verification$n_distinct_rows)
    if (!isTRUE(verification$matches_shipped)) {
      stop("Bag seeds do not reproduce the bMGP resamples.", call. = FALSE)
    }
    settings$bag_seed_matches_bmgp <- verification$matches_shipped
  }

  meta_base <- cbind(
    data.frame(
      tag = tag,
      timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
      scenario = model_id,
      family = spec$family,
      m = m_value,
      repeat_id = rep_index,
      repeat_seed = repeat_seed,
      n = nrow(data$X),
      p = ncol(data$X),
      p_star = length(data$active_indices),
      stringsAsFactors = FALSE
    ),
    machine_row()
  )

  if (mode %in% c("all", "mgp")) {
    run_mgp_bmgp(spec, config, data, settings, meta_base, out_dir)
  }
  if (mode %in% c("all", "gibbs", "bags")) {
    run_gibbs_timing(spec, config, data, settings, meta_base, out_dir,
                     do_fit = mode %in% c("all", "gibbs"),
                     do_bags = mode %in% c("all", "bags"))
  }
  message("Run: Rscript Make_Table8_Timing.R   for the table and the .tex")
  invisible(NULL)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
