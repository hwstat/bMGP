#!/usr/bin/env Rscript
## Table 1: Monte Carlo coverage for the population mean, MGP vs bMGP, under the
## fixed-variance Gaussian predictive engine with varkappa = 1/2 (Section 3.2 of the paper).
## Usage: Rscript Run_Table1_Mean_Coverage.R [--R=200 --ncores=10 --n_grid=100,200,500]
## The engine draws with the fixed variance var(x)/2 (the paper design);
## --fixed_var=false updates the variance along the path instead.

round_numeric_df <- function(df, digits = 5) {
  out <- df
  num_cols <- vapply(out, is.numeric, logical(1))
  out[num_cols] <- lapply(out[num_cols], round, digits)
  out
}

validate_choice <- function(x, choices, name) {
  if (!all(x %in% choices)) {
    stop(name, " must be one of: ", paste(choices, collapse = ", "), call. = FALSE)
  }
  x
}

simulate_x_by_case <- function(case, n, shape_g = 2, rate_g = 2) {
  if (case == "Well_Normal") return(stats::rnorm(n, mean = 0, sd = 1))
  if (case == "Miss_Gamma") return(stats::rgamma(n, shape = shape_g, rate = rate_g))
  stop("Unknown case: ", case, call. = FALSE)
}

mean_targets_by_case <- function(case, shape_g = 2, rate_g = 2) {
  if (case == "Well_Normal") {
    theta_true <- 0
  } else if (case == "Miss_Gamma") {
    theta_true <- shape_g / rate_g
  } else {
    stop("Unknown case: ", case, call. = FALSE)
  }
  theta_engine <- theta_true
  c(true = theta_true, engine = theta_engine)
}

ordinary_seed_by_case_n <- function(case, n) {
  if (case == "Well_Normal") return(1000L + as.integer(n))
  if (case == "Miss_Gamma") return(2000L + as.integer(n))
  stop("Unknown case: ", case, call. = FALSE)
}

gaussian_predictive_engine_original <- function(x_obs, B, N, var_floor = 1e-6,
                                                var_mult = 0.5) {
  n_obs <- length(x_obs)
  n_vec <- rep(n_obs, B)
  sum_vec <- rep(sum(x_obs), B)
  sumsq_vec <- rep(sum(x_obs^2), B)
  var0 <- max(stats::var(x_obs), var_floor)
  fixed_var <- identical(Sys.getenv("GPE_FIXED_VAR"), "1")
  Z <- matrix(stats::rnorm(B * N), nrow = B, ncol = N)
  paths <- matrix(NA_real_, nrow = B, ncol = N)

  for (m in seq_len(N)) {
    mu_hat <- sum_vec / n_vec
    numerator <- sumsq_vec - (sum_vec^2) / n_vec
    denominator <- n_vec - 1
    var_raw <- ifelse(denominator > 0, numerator / denominator, 0)
    var_hat <- pmax(var_raw, var_floor)
    if (fixed_var) var_hat <- rep(var0, B)
    x_new <- mu_hat + Z[, m] * sqrt(var_mult * var_hat)
    paths[, m] <- x_new
    n_vec <- n_vec + 1
    sum_vec <- sum_vec + x_new
    sumsq_vec <- sumsq_vec + x_new^2
  }
  paths
}

ordinary_mean_ci_from_x <- function(x0, B, N, level = 0.95,
                                    var_floor = 1e-6, var_mult = 0.5) {
  z_paths <- gaussian_predictive_engine_original(
    x_obs = x0,
    B = B,
    N = N,
    var_floor = var_floor,
    var_mult = var_mult
  )
  theta_draws <- (sum(x0) + rowSums(z_paths)) / (length(x0) + N)
  a <- (1 - level) / 2
  ci <- stats::quantile(theta_draws, probs = c(a, 1 - a), type = 7, names = FALSE)
  c(ci_lo = ci[1], ci_hi = ci[2], center = mean(theta_draws))
}

ordinary_gpe_with_xlist_original_style <- function(case, R, n, S_ord, N_future,
                                                   level = 0.95, var_floor = 1e-6,
                                                   var_mult = 0.5, shape_g = 2,
                                                   rate_g = 2, ncores = NULL,
                                                   seed = ordinary_seed_by_case_n(case, n)) {
  if (is.null(ncores)) ncores <- max(1L, parallel::detectCores() - 1L)
  ncores <- min(ncores, R)

  cl <- parallel::makeCluster(ncores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterSetRNGStream(cl, seed)

  parallel::clusterExport(
    cl,
    varlist = c(
      "simulate_x_by_case", "gaussian_predictive_engine_original",
      "ordinary_mean_ci_from_x", "case", "n", "S_ord", "N_future",
      "level", "var_floor", "var_mult", "shape_g", "rate_g"
    ),
    envir = environment()
  )

  out <- parallel::parLapply(cl, seq_len(R), function(r) {
    x0 <- simulate_x_by_case(case, n, shape_g = shape_g, rate_g = rate_g)
    s <- ordinary_mean_ci_from_x(
      x0 = x0,
      B = S_ord,
      N = N_future,
      level = level,
      var_floor = var_floor,
      var_mult = var_mult
    )
    list(
      x0 = x0,
      row = data.frame(
        rep = r,
        ci_lo = as.numeric(s["ci_lo"]),
        ci_hi = as.numeric(s["ci_hi"]),
        center = as.numeric(s["center"]),
        stringsAsFactors = FALSE
      )
    )
  })

  list(
    x_list = lapply(out, `[[`, "x0"),
    ordinary_df = do.call(rbind, lapply(out, `[[`, "row"))
  )
}

gpe_mean_draws_fast <- function(x_obs, S, N_future, var_mult = 0.5,
                                variance_type = c("unbiased", "mle"),
                                var_floor = 1e-6) {
  variance_type <- match.arg(variance_type)
  n0 <- length(x_obs)
  if (n0 < 2L) stop("x_obs must contain at least two observations.", call. = FALSE)

  n_vec <- rep(as.numeric(n0), S)
  sum_vec <- rep(sum(x_obs), S)
  sumsq_vec <- rep(sum(x_obs^2), S)
  future_sum <- numeric(S)
  var0 <- max(if (variance_type == "unbiased") stats::var(x_obs) else mean((x_obs - mean(x_obs))^2), var_floor)
  fixed_var <- identical(Sys.getenv("GPE_FIXED_VAR"), "1")

  for (m in seq_len(N_future)) {
    mu_hat <- sum_vec / n_vec
    numerator <- sumsq_vec - (sum_vec^2) / n_vec
    denominator <- if (variance_type == "unbiased") n_vec - 1 else n_vec
    var_raw <- ifelse(denominator > 0, numerator / denominator, 0)
    var_hat <- pmax(var_raw, var_floor)
    if (fixed_var) var_hat <- rep(var0, S)
    x_new <- mu_hat + stats::rnorm(S) * sqrt(var_mult * var_hat)

    future_sum <- future_sum + x_new
    n_vec <- n_vec + 1
    sum_vec <- sum_vec + x_new
    sumsq_vec <- sumsq_vec + x_new^2
  }

  (sum(x_obs) + future_sum) / (n0 + N_future)
}

summarise_draws <- function(draws, level = 0.95) {
  a <- (1 - level) / 2
  ci <- stats::quantile(draws, probs = c(a, 1 - a), type = 7, names = FALSE)
  c(ci_lo = ci[1], ci_hi = ci[2], center = mean(draws))
}

nested_db_mean_draw_matrix <- function(x_obs, B_max, S_max, boot_m, N_original,
                                       future_horizon_rule = c("bootstrap_m", "original_n"),
                                       var_mult = 0.5,
                                       variance_type = c("unbiased", "mle"),
                                       var_floor = 1e-6) {
  future_horizon_rule <- match.arg(future_horizon_rule)
  variance_type <- match.arg(variance_type)
  N_future_b <- if (future_horizon_rule == "bootstrap_m") {
    ceiling(boot_m^(3 / 2))
  } else {
    N_original
  }

  out <- matrix(NA_real_, nrow = B_max, ncol = S_max)
  for (b in seq_len(B_max)) {
    x_star <- sample(x_obs, size = boot_m, replace = TRUE)
    out[b, ] <- gpe_mean_draws_fast(
      x_obs = x_star,
      S = S_max,
      N_future = N_future_b,
      var_mult = var_mult,
      variance_type = variance_type,
      var_floor = var_floor
    )
  }
  out
}

run_db_for_xlist <- function(case, n, x_list, config_df, level = 0.95,
                             var_mult = 0.5, variance_type = "unbiased",
                             var_floor = 1e-6,
                             future_horizon_rule = "bootstrap_m",
                             ncores = NULL, seed = 900000L + n) {
  R <- length(x_list)
  if (is.null(ncores)) ncores <- max(1L, parallel::detectCores() - 1L)
  ncores <- min(ncores, R)
  N_original <- ceiling(n^(3 / 2))

  B_max <- max(as.integer(config_df$B_boot))
  S_max <- max(as.integer(config_df$S_each))
  boot_m_values <- sort(unique(as.integer(config_df$boot_m)))

  cl <- parallel::makeCluster(ncores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterSetRNGStream(cl, seed)

  parallel::clusterExport(
    cl,
    varlist = c(
      "nested_db_mean_draw_matrix", "gpe_mean_draws_fast", "summarise_draws",
      "config_df", "level", "var_mult", "variance_type", "var_floor",
      "future_horizon_rule", "N_original", "B_max", "S_max", "boot_m_values",
      "case", "n"
    ),
    envir = environment()
  )

  tmp <- parallel::parLapply(cl, seq_along(x_list), function(r) {
    x0 <- x_list[[r]]

    db_cache <- vector("list", length(boot_m_values))
    names(db_cache) <- as.character(boot_m_values)
    for (boot_m in boot_m_values) {
      db_cache[[as.character(boot_m)]] <- nested_db_mean_draw_matrix(
        x_obs = x0,
        B_max = B_max,
        S_max = S_max,
        boot_m = boot_m,
        N_original = N_original,
        future_horizon_rule = future_horizon_rule,
        var_mult = var_mult,
        variance_type = variance_type,
        var_floor = var_floor
      )
    }

    out <- vector("list", nrow(config_df))
    for (j in seq_len(nrow(config_df))) {
      B_boot <- as.integer(config_df$B_boot[j])
      S_each <- as.integer(config_df$S_each[j])
      boot_m <- as.integer(config_df$boot_m[j])
      db_mat <- db_cache[[as.character(boot_m)]]
      draws_db <- as.vector(db_mat[seq_len(B_boot), seq_len(S_each), drop = FALSE])
      s <- summarise_draws(draws_db, level = level)
      out[[j]] <- data.frame(
        case = case,
        n = n,
        rep = r,
        boot_m_ratio = config_df$boot_m_ratio[j],
        boot_m = boot_m,
        B_boot = B_boot,
        S_each = S_each,
        S_db_total = B_boot * S_each,
        method = "Double-bootstrap GPE-PBI",
        ci_lo = as.numeric(s["ci_lo"]),
        ci_hi = as.numeric(s["ci_hi"]),
        center = as.numeric(s["center"]),
        stringsAsFactors = FALSE
      )
    }
    do.call(rbind, out)
  })

  do.call(rbind, tmp)
}

summarise_mean_results <- function(raw_df, target_source_grid = "engine", level = 0.95) {
  validate_choice(target_source_grid, c("true", "engine"), "target_source_grid")
  out <- list()
  k <- 1L

  for (target_source in target_source_grid) {
    target_col <- if (target_source == "true") "theta_true" else "theta_engine"
    sub <- raw_df
    target_value <- sub[[target_col]]
    sub$covered <- as.numeric(target_value >= sub$ci_lo & target_value <= sub$ci_hi)
    sub$signed_bias <- sub$center - target_value
    sub$target_source <- target_source
    sub$target_value <- target_value

    agg <- stats::aggregate(
      cbind(coverage = covered, signed_bias = signed_bias) ~
        case + n + target_source + target_value + boot_m_ratio + boot_m +
        B_boot + S_each + S_db_total + S_ord + future_horizon_rule + method,
      data = sub,
      FUN = mean
    )
    agg$R <- length(unique(sub$rep))
    agg$nominal_level <- level
    out[[k]] <- agg
    k <- k + 1L
  }

  ans <- do.call(rbind, out)
  row.names(ans) <- NULL
  ans[order(ans$target_source, ans$case, ans$n,
            ans$boot_m_ratio, ans$B_boot, ans$method), ]
}

make_comparison <- function(summary_df) {
  ord <- summary_df[summary_df$method == "Ordinary GPE-PBI", , drop = FALSE]
  db <- summary_df[summary_df$method == "Double-bootstrap GPE-PBI", , drop = FALSE]
  by_cols <- c("case", "n", "target_source", "target_value", "boot_m_ratio",
               "boot_m", "B_boot", "S_each", "S_db_total", "S_ord",
               "future_horizon_rule", "R", "nominal_level")
  merged <- merge(ord, db, by = by_cols, suffixes = c("_ord", "_db"))
  out <- data.frame(
    case = merged$case,
    n = merged$n,
    target_source = merged$target_source,
    R = merged$R,
    boot_m_ratio = merged$boot_m_ratio,
    boot_m = merged$boot_m,
    B_boot = merged$B_boot,
    S_each = merged$S_each,
    S_db_total = merged$S_db_total,
    S_ord = merged$S_ord,
    coverage_ord = merged$coverage_ord,
    signed_bias_ord = merged$signed_bias_ord,
    coverage_db = merged$coverage_db,
    signed_bias_db = merged$signed_bias_db,
    coverage_gain_db_minus_ord = merged$coverage_db - merged$coverage_ord,
    bias_change_db_minus_ord = merged$signed_bias_db - merged$signed_bias_ord,
    row.names = NULL
  )
  out[order(out$target_source, out$case, out$n, out$boot_m_ratio, out$B_boot), ]
}

compact_comparison <- function(comparison_df) {
  comparison_df[, c(
    "case", "n", "target_source", "R", "boot_m_ratio", "boot_m",
    "B_boot", "S_each", "S_db_total", "S_ord",
    "coverage_ord", "signed_bias_ord", "coverage_db", "signed_bias_db",
    "coverage_gain_db_minus_ord", "bias_change_db_minus_ord"
  )]
}

check_ordinary_invariance <- function(comparison_df, tolerance = 1e-12) {
  key_cols <- c("case", "n", "target_source", "R", "S_ord")
  split_key <- do.call(interaction, c(comparison_df[key_cols], list(drop = TRUE, lex.order = TRUE)))
  out <- lapply(split(comparison_df, split_key), function(d) {
    data.frame(
      case = d$case[1],
      n = d$n[1],
      target_source = d$target_source[1],
      R = d$R[1],
      S_ord = d$S_ord[1],
      max_diff_coverage_ord = max(d$coverage_ord) - min(d$coverage_ord),
      max_diff_signed_bias_ord = max(d$signed_bias_ord) - min(d$signed_bias_ord),
      ordinary_invariant =
        (max(d$coverage_ord) - min(d$coverage_ord) <= tolerance) &&
        (max(d$signed_bias_ord) - min(d$signed_bias_ord) <= tolerance),
      row.names = NULL
    )
  })
  ans <- do.call(rbind, out)
  row.names(ans) <- NULL
  ans[order(ans$target_source, ans$case, ans$n), ]
}

run_mean_pbi_dbgrid_matched_original <- function(
  n_grid = c(100, 200, 500),
  cases = c("Well_Normal", "Miss_Gamma"),
  R = 200,
  boot_m_ratio_grid = c(0.5, 1.0),
  B_boot_grid = c(20, 50),
  S_each_grid = 20,
  S_ord = 1000,
  target_source_grid = "engine",
  level = 0.95,
  var_mult = 0.5,
  var_floor = 1e-6,
  shape_g = 2,
  rate_g = 2,
  future_horizon_rule = c("bootstrap_m", "original_n"),
  variance_type_db = c("unbiased", "mle"),
  ncores = NULL,
  db_seed_base = 900000L
) {
  future_horizon_rule <- match.arg(future_horizon_rule)
  variance_type_db <- match.arg(variance_type_db)
  validate_choice(cases, c("Well_Normal", "Miss_Gamma"), "cases")
  validate_choice(target_source_grid, c("true", "engine"), "target_source_grid")

  if (is.null(ncores)) ncores <- max(1L, parallel::detectCores() - 1L)

  config_base <- expand.grid(
    boot_m_ratio = boot_m_ratio_grid,
    B_boot = B_boot_grid,
    S_each = S_each_grid,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  raw_all <- list()
  idx <- 1L

  for (case in cases) {
    for (n in n_grid) {
      N_future <- ceiling(n^(3 / 2))
      config_df <- config_base
      config_df$boot_m <- pmax(2L, as.integer(floor(config_df$boot_m_ratio * n)))
      config_df$boot_m <- pmin(config_df$boot_m, n)

      message("Ordinary baseline: case=", case, ", n=", n,
              ", seed=", ordinary_seed_by_case_n(case, n))
      ord <- ordinary_gpe_with_xlist_original_style(
        case = case,
        R = R,
        n = n,
        S_ord = S_ord,
        N_future = N_future,
        level = level,
        var_floor = var_floor,
        var_mult = var_mult,
        shape_g = shape_g,
        rate_g = rate_g,
        ncores = ncores,
        seed = ordinary_seed_by_case_n(case, n)
      )

      message("DB step: case=", case, ", n=", n,
              ", configs=", nrow(config_df))
      db <- run_db_for_xlist(
        case = case,
        n = n,
        x_list = ord$x_list,
        config_df = config_df,
        level = level,
        var_mult = var_mult,
        variance_type = variance_type_db,
        var_floor = var_floor,
        future_horizon_rule = future_horizon_rule,
        ncores = ncores,
        seed = db_seed_base + ifelse(case == "Well_Normal", 1000L, 2000L) + as.integer(n)
      )

      targets <- mean_targets_by_case(case, shape_g = shape_g, rate_g = rate_g)

      ord_rows <- vector("list", nrow(config_df))
      for (j in seq_len(nrow(config_df))) {
        tmp <- ord$ordinary_df
        tmp$case <- case
        tmp$n <- n
        tmp$boot_m_ratio <- config_df$boot_m_ratio[j]
        tmp$boot_m <- config_df$boot_m[j]
        tmp$B_boot <- config_df$B_boot[j]
        tmp$S_each <- config_df$S_each[j]
        tmp$S_db_total <- config_df$B_boot[j] * config_df$S_each[j]
        tmp$method <- "Ordinary GPE-PBI"
        ord_rows[[j]] <- tmp
      }
      ord_df <- do.call(rbind, ord_rows)

      common_cols <- c("case", "n", "rep", "boot_m_ratio", "boot_m", "B_boot",
                       "S_each", "S_db_total", "method", "ci_lo", "ci_hi", "center")
      combined <- rbind(ord_df[, common_cols], db[, common_cols])
      combined$S_ord <- S_ord
      combined$future_horizon_rule <- future_horizon_rule
      combined$theta_true <- as.numeric(targets["true"])
      combined$theta_engine <- as.numeric(targets["engine"])

      raw_all[[idx]] <- combined
      idx <- idx + 1L
    }
  }

  raw <- do.call(rbind, raw_all)
  row.names(raw) <- NULL
  summary <- summarise_mean_results(raw, target_source_grid = target_source_grid, level = level)
  comparison <- make_comparison(summary)

  list(
    raw = raw,
    summary = summary,
    comparison = comparison,
    ordinary_invariance_check = check_ordinary_invariance(comparison),
    settings = list(
      n_grid = n_grid,
      cases = cases,
      R = R,
      boot_m_ratio_grid = boot_m_ratio_grid,
      B_boot_grid = B_boot_grid,
      S_each_grid = S_each_grid,
      S_ord = S_ord,
      target_source_grid = target_source_grid,
      level = level,
      var_mult = var_mult,
      future_horizon_rule = future_horizon_rule,
      variance_type_db = variance_type_db
    )
  )
}

CASE_LABELS <- c(
  Well_Normal = "DGP: $N(0,1)$",
  Miss_Gamma  = "DGP: $\\mathrm{Ga}(2,2)$"
)
CASE_LABELS_PLAIN <- c(
  Well_Normal = "DGP: N(0,1)",
  Miss_Gamma  = "DGP: Ga(2,2)"
)
METHOD_LABELS <- c(ord = "MGP", db = "bMGP")

select_table_config <- function(comparison_df, boot_m_ratio, B_boot, S_each) {
  keep <- comparison_df$boot_m_ratio == boot_m_ratio &
    comparison_df$B_boot == B_boot &
    comparison_df$S_each == S_each
  sub <- comparison_df[keep, , drop = FALSE]
  if (nrow(sub) == 0L) {
    stop("No results for boot_m_ratio=", boot_m_ratio, ", B_boot=", B_boot,
         ", S_each=", S_each, ".\n  Available: ",
         paste(unique(sprintf("(%.2f, %d, %d)", comparison_df$boot_m_ratio,
                              comparison_df$B_boot, comparison_df$S_each)),
               collapse = " "), call. = FALSE)
  }
  sub
}

build_table_cells <- function(sub, cases, n_grid, cov_digits = 3, bias_digits = 5) {
  cell <- function(coverage, bias) {
    sprintf(paste0("%.", cov_digits, "f (%.", bias_digits, "f)"), coverage, bias)
  }
  out <- data.frame(n = n_grid)
  for (cs in cases) {
    for (meth in c("ord", "db")) {
      values <- vapply(n_grid, function(nn) {
        row <- sub[sub$case == cs & sub$n == nn, , drop = FALSE]
        if (nrow(row) != 1L) {
          stop("Expected exactly one row for case=", cs, ", n=", nn,
               " but found ", nrow(row), call. = FALSE)
        }
        cell(row[[paste0("coverage_", meth)]], row[[paste0("signed_bias_", meth)]])
      }, character(1))
      out[[paste(cs, meth, sep = "_")]] <- values
    }
  }
  out
}

print_console_table <- function(cells, cases, level, R, config) {
  header1 <- sprintf("%5s", "")
  header2 <- sprintf("%5s", "n")
  for (cs in cases) {
    header1 <- paste0(header1, sprintf("  %-36s", CASE_LABELS_PLAIN[[cs]]))
    header2 <- paste0(header2, sprintf("  %-17s %-17s", "MGP", "bMGP"))
  }
  bar <- strrep("-", max(nchar(header1), nchar(header2)))
  cat("\n", bar, "\n", header1, "\n", header2, "\n", bar, "\n", sep = "")
  for (i in seq_len(nrow(cells))) {
    line <- sprintf("%5s", cells$n[i])
    for (cs in cases) {
      line <- paste0(line, sprintf("  %-17s %-17s",
                                   cells[[paste0(cs, "_ord")]][i],
                                   cells[[paste0(cs, "_db")]][i]))
    }
    cat(line, "\n", sep = "")
  }
  cat(bar, "\n", sep = "")
  cat(sprintf("cell = coverage (signed bias);  nominal level %.2f;  R = %d\n",
              level, R))
  cat(sprintf("bMGP config: boot_m_ratio = %.2f, B_boot = %d, S_each = %d",
              config$boot_m_ratio, config$B_boot, config$S_each),
      sprintf(" (%d draws vs S_ord = %d)\n", config$B_boot * config$S_each,
              config$S_ord), sep = "")
}

write_latex_table <- function(cells, cases, level, R, config, path) {
  n_cases <- length(cases)
  col_spec <- paste0("l", strrep("cc", n_cases))
  group <- character(0)
  cmid <- character(0)
  for (k in seq_along(cases)) {
    lo <- 2L + 2L * (k - 1L)
    group <- c(group, sprintf("\\multicolumn{2}{c}{%s}", CASE_LABELS[[cases[k]]]))
    cmid <- c(cmid, sprintf("\\cmidrule(lr){%d-%d}", lo, lo + 1L))
  }

  L <- c(
    "% =========================================================================",
    "%  Coverage (signed bias) of the mean functional, MGP vs bMGP.",
    "%  Generated by Run_Table1_Mean_Coverage.R -- do not edit by hand.",
    "%  Requires \\usepackage{booktabs} in the preamble.",
    "% =========================================================================",
    sprintf("\\begin{tabular}{%s}", col_spec),
    "\\toprule",
    paste0(" & ", paste(group, collapse = " & "), " \\\\"),
    paste0(paste(cmid, collapse = " "), ""),
    paste0("$n$ & ", paste(rep(c("MGP", "bMGP"), n_cases), collapse = " & "),
           " \\\\"),
    "\\midrule"
  )
  for (i in seq_len(nrow(cells))) {
    entries <- character(0)
    for (cs in cases) {
      for (meth in c("ord", "db")) {
        txt <- cells[[paste0(cs, "_", meth)]][i]
        txt <- sub("^([-0-9.]+) \\((.*)\\)$", "$\\1\\\\;(\\2)$", txt)
        entries <- c(entries, txt)
      }
    }
    L <- c(L, paste0(cells$n[i], " & ", paste(entries, collapse = " & "), " \\\\"))
  }
  L <- c(L, "\\bottomrule", "\\end{tabular}")
  writeLines(L, path)
  path
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
cli_num <- function(cli, key, default) {
  as.numeric(strsplit(cli_value(cli, key, paste(default, collapse = ",")),
                      ",", fixed = TRUE)[[1]])
}
cli_flag <- function(cli, key, default = FALSE) {
  v <- cli_value(cli, key, NA_character_)
  if (is.na(v)) default else tolower(v) %in% c("true", "t", "yes", "1")
}
script_dir_of_this_file <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) == 1L) dirname(normalizePath(sub("^--file=", "", fa))) else getwd()
}

main <- function() {
  cli <- parse_cli(commandArgs(trailingOnly = TRUE))

  R <- as.integer(cli_value(cli, "R", 200))
  if (identical(tolower(cli_value(cli, "fixed_var", "true")), "true")) Sys.setenv(GPE_FIXED_VAR = "1")
  n_grid <- cli_num(cli, "n_grid", c(100, 200, 500))
  cases <- strsplit(cli_value(cli, "cases", "Well_Normal,Miss_Gamma"), ",")[[1]]
  S_ord <- as.integer(cli_value(cli, "S_ord", 1000))
  boot_m_ratio <- as.numeric(cli_value(cli, "boot_m_ratio", 1.0))
  B_boot <- as.integer(cli_value(cli, "B_boot", 50))
  S_each <- as.integer(cli_value(cli, "S_each", 20))
  level <- as.numeric(cli_value(cli, "level", 0.95))
  ncores <- as.integer(cli_value(cli, "ncores",
                                 max(1L, parallel::detectCores() - 1L)))
  all_configs <- cli_flag(cli, "all_configs", FALSE)
  save_raw <- cli_flag(cli, "save_raw", FALSE)
  out_dir <- cli_value(cli, "out", file.path(script_dir_of_this_file(), "output"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  ratio_grid <- if (all_configs) c(0.5, 1.0) else boot_m_ratio
  B_grid <- if (all_configs) c(20, 50) else B_boot

  message("R = ", R, ", n = ", paste(n_grid, collapse = ", "),
          ", cases = ", paste(cases, collapse = ", "), ", cores = ", ncores)
  started <- Sys.time()

  res <- run_mean_pbi_dbgrid_matched_original(
    n_grid = n_grid,
    cases = cases,
    R = R,
    boot_m_ratio_grid = ratio_grid,
    B_boot_grid = B_grid,
    S_each_grid = S_each,
    S_ord = S_ord,
    target_source_grid = "engine",
    level = level,
    var_mult = 0.5,
    future_horizon_rule = "bootstrap_m",
    variance_type_db = "unbiased",
    ncores = ncores
  )
  message("elapsed: ",
          format(round(difftime(Sys.time(), started, units = "mins"), 2)))

  comparison <- compact_comparison(res$comparison)
  full_path <- file.path(out_dir, "mean_gpe_comparison_full.csv")
  utils::write.csv(round_numeric_df(comparison, 6), full_path, row.names = FALSE)

  inv <- res$ordinary_invariance_check
  if (!all(inv$ordinary_invariant)) {
    warning("Ordinary baseline is not invariant across DB configurations.")
  } else if (nrow(inv) > 0L) {
    message("ordinary baseline invariant across DB configs: OK")
  }

  sub <- select_table_config(comparison, boot_m_ratio, B_boot, S_each)
  cells <- build_table_cells(sub, cases, n_grid)
  config <- list(boot_m_ratio = boot_m_ratio, B_boot = B_boot,
                 S_each = S_each, S_ord = S_ord)
  print_console_table(cells, cases, level, R, config)

  tex_path <- write_latex_table(cells, cases, level, R, config,
                                file.path(out_dir, "mean_gpe_coverage_table.tex"))
  csv_path <- file.path(out_dir, "mean_gpe_coverage_table.csv")
  utils::write.csv(cells, csv_path, row.names = FALSE)

  written <- c(tex_path, csv_path, full_path)
  if (save_raw) {
    raw_path <- file.path(out_dir, "mean_gpe_raw.rds")
    saveRDS(res$raw, raw_path, compress = "xz")
    written <- c(written, raw_path)
  }
  if (all_configs) {
    cat("\nAll configurations:\n")
    print(round_numeric_df(comparison, 5))
  }
  message("\nWrote:\n  ", paste(written, collapse = "\n  "))
  invisible(res)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  main()
}
