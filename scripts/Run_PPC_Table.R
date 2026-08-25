#!/usr/bin/env Rscript
## PPC-PE simulation table (Appendix E.1): posterior predictive checks of the
## predictive engine, two DGPs x three sample sizes.
## Usage: Rscript Run_PPC_Table.R [--R=2000 --B=1000 --ncores=10]; --R=200 for a quick run.

default_ncores <- function() {
  k1 <- parallel::detectCores(logical = FALSE)
  k2 <- parallel::detectCores(logical = TRUE)
  k <- if (!is.na(k1)) k1 else if (!is.na(k2)) k2 else 2L
  max(1L, as.integer(k) - 1L)
}

empirical_variance_measure <- function(x) {
  m <- mean(x)
  mean((x - m)^2)
}

empirical_skewness_measure <- function(x, eps = 1e-12) {
  x <- as.numeric(x)
  m1 <- mean(x)
  xc <- x - m1
  m2 <- mean(xc^2)

  if (!is.finite(m2) || m2 <= eps) {
    return(0)
  }

  m3 <- mean(xc^3)
  m3 / (m2^(3/2))
}

raw_moments_to_variance <- function(mean1, mean2) {
  pmax(as.numeric(mean2) - as.numeric(mean1)^2, 0)
}

raw_moments_to_skewness <- function(mean1, mean2, mean3, eps = 1e-12) {
  mean1 <- as.numeric(mean1)
  mean2 <- as.numeric(mean2)
  mean3 <- as.numeric(mean3)

  var_val <- pmax(mean2 - mean1^2, 0)
  mu3 <- mean3 - 3 * mean1 * mean2 + 2 * mean1^3

  out <- rep(0, length(mean1))
  ok <- is.finite(var_val) & (var_val > eps)

  out[ok] <- mu3[ok] / (var_val[ok]^(3/2))
  out
}

compute_ppc_pvalue <- function(S_rep,
                               S_obs,
                               pval_style = c("two_sided", "upper", "lower")) {
  pval_style <- match.arg(pval_style)

  if (pval_style == "two_sided") {
    u_hat <- mean(S_rep >= S_obs)
    p_value <- min(1, 2 * min(u_hat, 1 - u_hat))
  } else if (pval_style == "upper") {
    u_hat <- mean(S_rep >= S_obs)
    p_value <- u_hat
  } else {
    u_hat <- mean(S_rep <= S_obs)
    p_value <- u_hat
  }

  c(u_hat = u_hat, p_value = p_value)
}

make_default_scalar_diagnostics <- function() {
  list(
    list(
      name = "skewness",
      kind = "skewness",
      panel = "Sample Skewness",
      pval_style = "two_sided"
    ),
    list(
      name = "variance",
      kind = "variance",
      panel = "Sample Variance",
      pval_style = "two_sided"
    )
  )
}

gaussian_predictive_engine_fast <- function(x_obs,
                                            B,
                                            N_future,
                                            var_floor = 1e-6,
                                            keep_paths = FALSE,
                                            n_keep = length(x_obs)) {
  x_obs <- as.numeric(x_obs)

  n_obs <- length(x_obs)
  n_keep <- min(as.integer(n_keep), as.integer(N_future))

  n_vec <- rep.int(n_obs, B)
  sum_vec <- rep.int(sum(x_obs), B)
  sumsq_vec <- rep.int(sum(x_obs^2), B)

  if (isTRUE(keep_paths)) {
    paths <- matrix(NA_real_, nrow = B, ncol = N_future)
  } else {
    paths <- NULL
  }

  future_sum   <- numeric(B)
  future_sumsq <- numeric(B)
  future_sum3  <- numeric(B)

  keep_sum   <- numeric(B)
  keep_sumsq <- numeric(B)
  keep_sum3  <- numeric(B)

  for (m in seq_len(N_future)) {
    mu_hat <- sum_vec / n_vec

    numerator <- sumsq_vec - (sum_vec^2) / n_vec
    denominator <- n_vec - 1L
    var_raw <- ifelse(denominator > 0L, numerator / denominator, 0)
    var_hat <- pmax(var_raw, var_floor)

    x_new <- mu_hat + stats::rnorm(B) * sqrt(var_hat)

    if (isTRUE(keep_paths)) {
      paths[, m] <- x_new
    }

    future_sum   <- future_sum + x_new
    future_sumsq <- future_sumsq + x_new^2
    future_sum3  <- future_sum3 + x_new^3

    if (m <= n_keep) {
      keep_sum   <- keep_sum + x_new
      keep_sumsq <- keep_sumsq + x_new^2
      keep_sum3  <- keep_sum3 + x_new^3
    }

    n_vec     <- n_vec + 1L
    sum_vec   <- sum_vec + x_new
    sumsq_vec <- sumsq_vec + x_new^2
  }

  list(
    paths = paths,
    future_sum = future_sum,
    future_sumsq = future_sumsq,
    future_sum3 = future_sum3,
    keep_sum = keep_sum,
    keep_sumsq = keep_sumsq,
    keep_sum3 = keep_sum3
  )
}

pbppc_one_dataset_fast <- function(x_obs,
                                   diagnostics,
                                   B = 1000L,
                                   N_future = ceiling(length(x_obs)^1.5),
                                   alpha = 0.05,
                                   scale_fun = function(n) sqrt(n),
                                   var_floor = 1e-6,
                                   store_draws = FALSE) {
  x_obs <- as.numeric(x_obs)
  n <- length(x_obs)

  if (N_future < n) {
    stop("N_future must be at least n because replicate statistics use the first n pseudo-observations.")
  }

  engine_out <- gaussian_predictive_engine_fast(
    x_obs = x_obs,
    B = B,
    N_future = N_future,
    var_floor = var_floor,
    keep_paths = FALSE,
    n_keep = n
  )

  sum_x <- sum(x_obs)
  sumsq_x <- sum(x_obs^2)
  sum3_x <- sum(x_obs^3)
  total_m <- n + N_future

  rep_m1 <- engine_out$keep_sum / n
  rep_m2 <- engine_out$keep_sumsq / n
  rep_m3 <- engine_out$keep_sum3 / n

  post_m1 <- (sum_x + engine_out$future_sum) / total_m
  post_m2 <- (sumsq_x + engine_out$future_sumsq) / total_m
  post_m3 <- (sum3_x + engine_out$future_sum3) / total_m

  out_rows <- vector("list", length(diagnostics))
  draw_store <- list()

  for (j in seq_along(diagnostics)) {
    diagnostic <- diagnostics[[j]]

    if (diagnostic$kind == "skewness") {
      S_obs <- empirical_skewness_measure(x_obs)
      S_rep <- raw_moments_to_skewness(rep_m1, rep_m2, rep_m3)
      S_post <- raw_moments_to_skewness(post_m1, post_m2, post_m3)

    } else if (diagnostic$kind == "variance") {
      S_obs <- empirical_variance_measure(x_obs)
      S_rep <- raw_moments_to_variance(rep_m1, rep_m2)
      S_post <- raw_moments_to_variance(post_m1, post_m2)

    } else {
      stop("Unknown diagnostic kind: ", diagnostic$kind)
    }

    pv <- compute_ppc_pvalue(
      S_rep = S_rep,
      S_obs = S_obs,
      pval_style = diagnostic$pval_style
    )

    delta_draws <- scale_fun(n) * (S_post - S_obs)

    out_rows[[j]] <- data.frame(
      diagnostic = diagnostic$name,
      pval_style = diagnostic$pval_style,
      S_obs = S_obs,
      u_hat = unname(pv["u_hat"]),
      p_value = unname(pv["p_value"]),
      reject = as.integer(unname(pv["p_value"]) < alpha),
      delta_mean = mean(delta_draws),
      delta_sd = stats::sd(delta_draws),
      mc_pr_delta_pos = mean(delta_draws > 0),
      stringsAsFactors = FALSE
    )

    if (isTRUE(store_draws)) {
      draw_store[[diagnostic$name]] <- list(
        S_rep = S_rep,
        S_post = S_post,
        delta_draws = delta_draws
      )
    }
  }

  list(
    summary = do.call(rbind, out_rows),
    draws = draw_store
  )
}

summarize_pbppc_results <- function(dataset_results) {
  split_key <- interaction(
    dataset_results$case,
    dataset_results$n,
    dataset_results$diagnostic,
    drop = TRUE,
    lex.order = TRUE
  )

  out <- lapply(split(dataset_results, split_key), function(df) {
    rej_rate <- mean(df$reject)

    data.frame(
      case = df$case[1],
      n = df$n[1],
      diagnostic = df$diagnostic[1],
      pval_style = df$pval_style[1],
      mean_S_obs = mean(df$S_obs),
      mean_p_value = mean(df$p_value),
      median_p_value = stats::median(df$p_value),
      rejection_rate = rej_rate,
      rejection_se = sqrt(rej_rate * (1 - rej_rate) / nrow(df)),
      mean_delta_mean = mean(df$delta_mean),
      mean_delta_sd = mean(df$delta_sd),
      mean_mc_pr_delta_pos = mean(df$mc_pr_delta_pos),
      ndatasets = nrow(df),
      stringsAsFactors = FALSE
    )
  })

  out_df <- do.call(rbind, out)
  rownames(out_df) <- NULL
  out_df[order(out_df$diagnostic, out_df$case, out_df$n), ]
}

pbppc_job_worker <- function(job_row,
                             dgp_funs,
                             diagnostics,
                             B,
                             N_future_fun,
                             alpha,
                             scale_fun,
                             var_floor) {
  case_name <- as.character(job_row$case)
  n <- as.integer(job_row$n)
  r <- as.integer(job_row$replicate)

  x_obs <- dgp_funs[[case_name]](n)

  res <- pbppc_one_dataset_fast(
    x_obs = x_obs,
    diagnostics = diagnostics,
    B = B,
    N_future = N_future_fun(n),
    alpha = alpha,
    scale_fun = scale_fun,
    var_floor = var_floor,
    store_draws = FALSE
  )

  out <- res$summary
  out$case <- case_name
  out$n <- n
  out$replicate <- r
  out
}

simulate_pbppc_study_parallel <- function(dgp_funs,
                                          diagnostics,
                                          n_grid = c(100L, 200L, 500L),
                                          R = 200L,
                                          B = 1000L,
                                          N_future_fun = function(n) ceiling(n^1.5),
                                          alpha = 0.05,
                                          scale_fun = function(n) sqrt(n),
                                          var_floor = 1e-6,
                                          ncores = NULL,
                                          seed = 2026L) {
  if (is.null(names(dgp_funs)) || any(names(dgp_funs) == "")) {
    stop("dgp_funs must be a named list.")
  }

  jobs_df <- expand.grid(
    case = names(dgp_funs),
    n = n_grid,
    replicate = seq_len(R),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  job_list <- split(jobs_df, seq_len(nrow(jobs_df)))

  if (is.null(ncores)) {
    ncores <- min(default_ncores(), 6L)
  }
  ncores <- max(1L, min(as.integer(ncores), length(job_list)))

  if (ncores == 1L) {
    set.seed(seed)

    out_list <- lapply(
      job_list,
      pbppc_job_worker,
      dgp_funs = dgp_funs,
      diagnostics = diagnostics,
      B = B,
      N_future_fun = N_future_fun,
      alpha = alpha,
      scale_fun = scale_fun,
      var_floor = var_floor
    )

  } else {
    cl <- parallel::makeCluster(ncores)
    on.exit(parallel::stopCluster(cl), add = TRUE)

    parallel::clusterSetRNGStream(cl, iseed = seed)

    parallel::clusterExport(
      cl,
      varlist = c(
        "empirical_variance_measure",
        "empirical_skewness_measure",
        "raw_moments_to_variance",
        "raw_moments_to_skewness",
        "compute_ppc_pvalue",
        "gaussian_predictive_engine_fast",
        "pbppc_one_dataset_fast",
        "pbppc_job_worker"
      ),
      envir = .GlobalEnv
    )

    out_list <- parallel::parLapplyLB(
      cl,
      job_list,
      pbppc_job_worker,
      dgp_funs = dgp_funs,
      diagnostics = diagnostics,
      B = B,
      N_future_fun = N_future_fun,
      alpha = alpha,
      scale_fun = scale_fun,
      var_floor = var_floor
    )
  }

  dataset_results <- do.call(rbind, out_list)
  rownames(dataset_results) <- NULL

  list(
    dataset_results = dataset_results,
    summary = summarize_pbppc_results(dataset_results)
  )
}

TABLE_CASES <- list(
  list(case = "misspec",   tex = "$\\mathrm{Ga}(2,2)$", plain = "Ga(2,2)"),
  list(case = "well_spec", tex = "$N(0,1)$",            plain = "N(0,1)")
)

prepare_main_table <- function(summary_df,
                               n_grid,
                               diagnostics,
                               pvalue_stat = c("median", "mean"),
                               digits = 3L) {
  pvalue_stat <- match.arg(pvalue_stat)
  p_col <- if (pvalue_stat == "mean") "mean_p_value" else "median_p_value"

  fmt <- function(v) {
    sub("^-(0\\.0*)$", "\\1", sprintf(paste0("%.", digits, "f"), v))
  }

  diag_names <- vapply(diagnostics, function(d) d$name, character(1))

  out <- NULL
  for (blk in TABLE_CASES) {
    for (nn in n_grid) {
      row <- data.frame(dgp = blk$plain, n = as.integer(nn),
                        stringsAsFactors = FALSE)
      for (dn in diag_names) {
        hit <- summary_df[summary_df$case == blk$case &
                            summary_df$n == nn &
                            summary_df$diagnostic == dn, , drop = FALSE]
        if (nrow(hit) != 1L) {
          stop("Expected exactly one summary row for case=", blk$case,
               ", n=", nn, ", diagnostic=", dn, " but found ", nrow(hit))
        }
        row[[paste0(dn, "_p")]]    <- fmt(hit[[p_col]])
        row[[paste0(dn, "_rate")]] <- fmt(hit$rejection_rate)
        row[[paste0(dn, "_diff")]] <- fmt(hit$mean_delta_mean)
      }
      out <- rbind(out, row)
    }
  }
  rownames(out) <- NULL
  out
}

print_main_table <- function(tab_df, diagnostics, alpha, R, B, pvalue_stat) {
  panels <- vapply(diagnostics, function(d) d$panel, character(1))
  names_d <- vapply(diagnostics, function(d) d$name, character(1))
  labels <- LETTERS[seq_along(panels)]

  h1 <- sprintf("%-9s %5s", "", "")
  h2 <- sprintf("%-9s %5s", "DGP", "n")
  for (k in seq_along(panels)) {
    h1 <- paste0(h1, sprintf("  %-25s",
                             sprintf("Panel %s: %s", labels[k], panels[k])))
    h2 <- paste0(h2, sprintf("  %8s %7s %8s", "p-value", "Rate", "AvgDiff"))
  }
  bar <- strrep("-", max(nchar(h1), nchar(h2)))

  cat("\n", bar, "\n", h1, "\n", h2, "\n", bar, "\n", sep = "")
  last_dgp <- ""
  for (i in seq_len(nrow(tab_df))) {
    tag <- if (identical(tab_df$dgp[i], last_dgp)) "" else tab_df$dgp[i]
    if (nzchar(tag) && i > 1L) cat(bar, "\n", sep = "")
    last_dgp <- tab_df$dgp[i]
    line <- sprintf("%-9s %5d", tag, tab_df$n[i])
    for (dn in names_d) {
      line <- paste0(line, sprintf("  %8s %7s %8s",
                                   tab_df[[paste0(dn, "_p")]][i],
                                   tab_df[[paste0(dn, "_rate")]][i],
                                   tab_df[[paste0(dn, "_diff")]][i]))
    }
    cat(line, "\n", sep = "")
  }
  cat(bar, "\n", sep = "")
  cat(sprintf("p-value = %s over datasets;  Rate = fraction rejected at alpha = %.2f;  ",
              pvalue_stat, alpha),
      "AvgDiff = mean of sqrt(n){S(P_nN) - S(P_n)}\n", sep = "")
  cat(sprintf("R = %d datasets per cell, B = %d predictive paths each\n", R, B))
}

write_main_table_tex <- function(tab_df,
                                 diagnostics,
                                 alpha,
                                 R,
                                 pvalue_stat,
                                 file = "pbppc_main_table.tex") {
  n_diag <- length(diagnostics)
  panels <- vapply(diagnostics, function(d) d$panel, character(1))
  names_d <- vapply(diagnostics, function(d) d$name, character(1))
  labels <- LETTERS[seq_len(n_diag)]

  col_spec <- paste0("ll", strrep("ccc", n_diag))

  group <- character(0)
  cmid  <- character(0)
  for (k in seq_len(n_diag)) {
    lo <- 3L + 3L * (k - 1L)
    group <- c(group, sprintf("\\multicolumn{3}{c}{\\textbf{Panel %s: %s}}",
                              labels[k], panels[k]))
    cmid <- c(cmid, sprintf("\\cmidrule(lr){%d-%d}", lo, lo + 2L))
  }

  lines <- c(
    "% =========================================================================",
    "%  PB-PPC for the Gaussian predictive engine: sample skewness and sample",
    "%  variance, two DGPs x three sample sizes.",
    "%  Generated by run_ppc_pe_table.R -- do not edit by hand.",
    "%  Requires \\usepackage{booktabs} in the preamble.",
    sprintf("%%  p-value = %s over R = %d datasets; Rate = fraction with p < %.2f;",
            pvalue_stat, R, alpha),
    "%  AvgDiff = mean of sqrt(n){S(P_{n,N}) - S(P_n)}.",
    "% =========================================================================",
    sprintf("\\begin{tabular}{%s}", col_spec),
    "\\toprule",
    paste0(" & & ", paste(group, collapse = " & "), " \\\\"),
    paste(cmid, collapse = " "),
    paste0("DGP & $n$ & ",
           paste(rep(c("$p$-value", "Rate", "AvgDiff"), n_diag),
                 collapse = " & "),
           " \\\\"),
    "\\midrule"
  )

  plains <- vapply(TABLE_CASES, function(b) b$plain, character(1))
  texs   <- vapply(TABLE_CASES, function(b) b$tex, character(1))

  for (bi in seq_along(plains)) {
    idx <- which(tab_df$dgp == plains[bi])
    if (length(idx) == 0L) next
    mid <- idx[ceiling(length(idx) / 2)]

    for (i in idx) {
      tag <- if (i == mid) texs[bi] else ""
      entries <- character(0)
      for (dn in names_d) {
        entries <- c(entries,
                     sprintf("$%s$", tab_df[[paste0(dn, "_p")]][i]),
                     sprintf("$%s$", tab_df[[paste0(dn, "_rate")]][i]),
                     sprintf("$%s$", tab_df[[paste0(dn, "_diff")]][i]))
      }
      eol <- if (i == idx[length(idx)]) " \\\\" else " \\\\[2pt]"
      lines <- c(lines, paste0(tag, " & ", tab_df$n[i], " & ",
                               paste(entries, collapse = " & "), eol))
    }
    if (bi < length(plains)) lines <- c(lines, "\\midrule")
  }

  lines <- c(lines, "\\bottomrule", "\\end{tabular}")
  writeLines(lines, con = file)
  file
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
script_dir_of_this_file <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) == 1L) dirname(normalizePath(sub("^--file=", "", fa))) else getwd()
}

cli <- parse_cli(commandArgs(trailingOnly = TRUE))

dgp_list <- list(
  well_spec = function(n) stats::rnorm(n, mean = 0, sd = 1),
  misspec   = function(n) stats::rgamma(n, shape = 2, rate = 2)
)

alpha_main <- 0.05
n_grid_main <- c(100L, 200L, 500L)

R_main <- as.integer(cli_value(cli, "R", 2000))
B_main <- as.integer(cli_value(cli, "B", 1000))

N_future_fun_main <- function(n) ceiling(n^1.5)

ncores_main <- as.integer(cli_value(cli, "ncores", min(default_ncores(), 6L)))

pvalue_stat_main <- cli_value(cli, "pvalue", "median")

diagnostics_main <- make_default_scalar_diagnostics()

out_dir <- cli_value(cli, "out", file.path(script_dir_of_this_file(), "output"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Running repeated-sample PB-PPC study ...")
message("R = ", R_main, ", B = ", B_main,
        ", n = ", paste(n_grid_main, collapse = ", "),
        ", horizon = ", paste(N_future_fun_main(n_grid_main), collapse = ", "),
        ", cores = ", ncores_main)

pbppc_main <- simulate_pbppc_study_parallel(
  dgp_funs = dgp_list,
  diagnostics = diagnostics_main,
  n_grid = n_grid_main,
  R = R_main,
  B = B_main,
  N_future_fun = N_future_fun_main,
  alpha = alpha_main,
  scale_fun = function(n) sqrt(n),
  var_floor = 1e-6,
  ncores = ncores_main,
  seed = 2026L
)

print(pbppc_main$summary)

write.csv(
  pbppc_main$summary,
  file = file.path(out_dir, "pbppc_repeated_summary_full.csv"),
  row.names = FALSE
)

write.csv(
  pbppc_main$dataset_results,
  file = file.path(out_dir, "pbppc_dataset_level_results.csv"),
  row.names = FALSE
)

main_table <- prepare_main_table(
  pbppc_main$summary,
  n_grid = n_grid_main,
  diagnostics = diagnostics_main,
  pvalue_stat = pvalue_stat_main
)

print_main_table(main_table, diagnostics_main, alpha_main, R_main, B_main,
                 pvalue_stat_main)

write.csv(
  main_table,
  file = file.path(out_dir, "pbppc_main_table.csv"),
  row.names = FALSE
)

write_main_table_tex(
  tab_df = main_table,
  diagnostics = diagnostics_main,
  alpha = alpha_main,
  R = R_main,
  pvalue_stat = pvalue_stat_main,
  file = file.path(out_dir, "pbppc_main_table.tex")
)

saveRDS(
  pbppc_main,
  file = file.path(out_dir, "pbppc_main_full_results.rds")
)

message("Done.")
message("Files written to ", out_dir, ":")
message("  pbppc_main_table.tex")
message("  pbppc_main_table.csv")
message("  pbppc_repeated_summary_full.csv")
message("  pbppc_dataset_level_results.csv")
message("  pbppc_main_full_results.rds")
