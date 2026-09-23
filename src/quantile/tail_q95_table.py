# =========================================================
# Appendix B.3, Table 10: coverage of the population quantile Q_{0.95}(P_0)
# by the QMP, the QMP-GP and their bagged versions, n in {100, 200, 500},
# R = 200 replicated data sets, B = 1000 paths for the ordinary QMP and
# B_boot = 50 bagged data sets with S = 20 paths each for the bagged versions.
# This is the script that produced the published Table 10; its random-number
# streams (one default_rng(2026) stream for the data, with the fit, posterior
# and bootstrap seeds offset from seed_base) differ from quantile_sweep.py,
# which produces Tables 4 and 5. QMP internals come from qmp_base.py.
# Run through scripts/Run_Quantile_Tail_Table.py.
# =========================================================

import gc
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

import qmp_base as base_qmp


TARGET_Q = 0.95
INTERVAL_LEVEL = base_qmp.INTERVAL_LEVEL

# The paper uses 50 bagged data sets with 20 paths each, matching the
# 1000 paths of the ordinary QMP.
DEFAULT_BOOTSTRAP_REPS = 50
DEFAULT_PATHS_PER_BOOT = 20
DEFAULT_TOTAL_PATHS = DEFAULT_BOOTSTRAP_REPS * DEFAULT_PATHS_PER_BOOT

# R = 200 replicated data sets, as in the paper (coverage resolution 0.005).
# PILOT_MONTE_CARLO_REPS is for quick local checks only.
PAPER_MONTE_CARLO_REPS = 200
PILOT_MONTE_CARLO_REPS = 100
DEFAULT_MONTE_CARLO_REPS = PAPER_MONTE_CARLO_REPS

# For bagged exact QMP, each bootstrap dataset re-estimates a and c.
# qmp.sample_qmp_functions.PR_loop_B treats a/c/k/n/T as JAX static args,
# so vectorizing exact sampling across paths can trigger many recompilations
# across bootstrap datasets. The single-draw loop is slower but much more
# memory-stable for double-bootstrap experiments.
BAGGED_EXACT_USE_VECTORIZED = False

METHOD_LABELS = {
    ("original", "exact"): "QMP-exact",
    ("original", "gp"): "QMP-GP",
    ("bagged", "exact"): "DB-QMP-exact",
    ("bagged", "gp"): "DB-QMP-GP",
}


ORIGINAL_EXACT_USE_VECTORIZED = True
VARIANT_ORDER = ("original", "bagged")
METHOD_ORDER = ("exact", "gp")
SCENARIOS = (
    ("well", "DGP:N(0,1)"),
    ("miss", "DGP:Ga(2,2)"),
)


def combined_output_prefix(q, R, original_B, B_boot, S_per_boot):
    q_tag = base_qmp.quantile_file_tag(q)
    return (
        f"case1_qmp_original_db_exact_gp_{q_tag}"
        f"_R{R}_B{original_B}_Bboot{B_boot}_S{S_per_boot}"
    )


def theta_draws_original_qmp(
    y,
    q=TARGET_Q,
    method="gp",
    B=DEFAULT_TOTAL_PATHS,
    seed_fit=123,
    seed_post=456,
    T_exact=base_qmp.DEFAULT_EXACT_T,
    use_vectorized_exact=ORIGINAL_EXACT_USE_VECTORIZED,
):
    """Draw from the ordinary QMP posterior for theta = Q(q)."""
    method = method.lower()
    if method == "gp":
        return base_qmp.theta_draws_gp_qmp(
            y,
            q=q,
            B=B,
            seed_fit=seed_fit,
            seed_post=seed_post,
        )
    if method == "exact":
        return base_qmp.theta_draws_exact_qmp(
            y,
            q=q,
            B=B,
            T_exact=T_exact,
            seed_fit=seed_fit,
            seed_post=seed_post,
            use_vectorized=use_vectorized_exact,
        )
    raise ValueError("method must be 'gp' or 'exact'.")


def theta_draws_bagged_qmp(
    y,
    q=TARGET_Q,
    method="gp",
    B_boot=DEFAULT_BOOTSTRAP_REPS,
    S_per_boot=DEFAULT_PATHS_PER_BOOT,
    M_boot=None,
    seed=12345,
    T_exact=base_qmp.DEFAULT_EXACT_T,
    use_vectorized_exact=BAGGED_EXACT_USE_VECTORIZED,
    verbose=False,
    collect_boot_details=False,
):
    """Draw from the bagged QMP posterior of theta = Q(q).

    This implements the bagged QMP of the paper:
      1. draw bootstrap datasets y_boot^(b) from the empirical distribution;
      2. run QMP posterior sampling conditional on each y_boot^(b);
      3. pool all B_boot * S_per_boot terminal theta draws.

    M_boot defaults to n, as recommended for BayesBag.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = len(y)
    if n == 0:
        raise ValueError("y must contain at least one observation.")

    if M_boot is None:
        M_boot = n
    M_boot = int(M_boot)
    B_boot = int(B_boot)
    S_per_boot = int(S_per_boot)

    if B_boot <= 0 or S_per_boot <= 0 or M_boot <= 0:
        raise ValueError("B_boot, S_per_boot, and M_boot must all be positive.")

    rng = np.random.default_rng(seed)
    theta_draws = np.empty(B_boot * S_per_boot, dtype=np.float32)
    boot_rows = [] if collect_boot_details else None

    method = method.lower()
    if method not in {"gp", "exact"}:
        raise ValueError("method must be 'gp' or 'exact'.")

    for b in range(B_boot):
        if verbose and ((b + 1) % 10 == 0 or b == 0):
            print(f"    bootstrap {b + 1}/{B_boot}")

        idx = rng.integers(0, n, size=M_boot)
        y_boot = y[idx]
        seed_fit = int(rng.integers(1, 2**31 - 1))
        seed_post = int(rng.integers(1, 2**31 - 1))

        t_start = time.time()
        if method == "gp":
            theta_b, fit_obj = base_qmp.theta_draws_gp_qmp(
                y_boot,
                q=q,
                B=S_per_boot,
                seed_fit=seed_fit,
                seed_post=seed_post,
            )
        else:
            theta_b, fit_obj = base_qmp.theta_draws_exact_qmp(
                y_boot,
                q=q,
                B=S_per_boot,
                T_exact=T_exact,
                seed_fit=seed_fit,
                seed_post=seed_post,
                use_vectorized=use_vectorized_exact,
            )
        elapsed = time.time() - t_start

        start = b * S_per_boot
        stop = start + S_per_boot
        theta_draws[start:stop] = theta_b

        if collect_boot_details:
            boot_rows.append({
                "bootstrap": b + 1,
                "method": method,
                "n": n,
                "M_boot": M_boot,
                "S_per_boot": S_per_boot,
                "a": float(fit_obj["a"]),
                "c": float(fit_obj["c"]),
                "k": float(fit_obj["k"]),
                "preq_score": float(fit_obj["preq_score"]),
                "n_rearr": float(fit_obj["n_rearr"]),
                "theta_mean_boot": float(np.mean(theta_b)),
                "theta_sd_boot": float(np.std(theta_b, ddof=1)) if S_per_boot > 1 else 0.0,
                "elapsed_seconds": float(elapsed),
            })

        del y_boot, theta_b, fit_obj
        if (b + 1) % 10 == 0:
            gc.collect()

    boot_detail = pd.DataFrame(boot_rows) if collect_boot_details else pd.DataFrame()
    return theta_draws, boot_detail


def draw_theta_for_variant(
    y,
    variant,
    method,
    q=TARGET_Q,
    original_B=DEFAULT_TOTAL_PATHS,
    B_boot=DEFAULT_BOOTSTRAP_REPS,
    S_per_boot=DEFAULT_PATHS_PER_BOOT,
    M_boot=None,
    seed_fit=123,
    seed_post=456,
    seed_boot=789,
    T_exact=base_qmp.DEFAULT_EXACT_T,
    original_exact_use_vectorized=ORIGINAL_EXACT_USE_VECTORIZED,
    bagged_exact_use_vectorized=BAGGED_EXACT_USE_VECTORIZED,
):
    variant = variant.lower()
    method = method.lower()

    if variant == "original":
        theta_draws, _ = theta_draws_original_qmp(
            y,
            q=q,
            method=method,
            B=original_B,
            seed_fit=seed_fit,
            seed_post=seed_post,
            T_exact=T_exact,
            use_vectorized_exact=original_exact_use_vectorized,
        )
        return theta_draws

    if variant == "bagged":
        theta_draws, _ = theta_draws_bagged_qmp(
            y,
            q=q,
            method=method,
            B_boot=B_boot,
            S_per_boot=S_per_boot,
            M_boot=M_boot,
            seed=seed_boot,
            T_exact=T_exact,
            use_vectorized_exact=bagged_exact_use_vectorized,
            verbose=False,
        )
        return theta_draws

    raise ValueError("variant must be 'original' or 'bagged'.")


def make_case1_data(rng, n, shape_g=2.0, rate_g=2.0):
    return {
        "well": rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32),
        "miss": rng.gamma(shape=shape_g, scale=1.0 / rate_g, size=n).astype(np.float32),
    }


def true_case1_quantiles(q, shape_g=2.0, rate_g=2.0):
    return {
        "well": float(norm.ppf(q)),
        "miss": float(gamma.ppf(q, a=shape_g, scale=1.0 / rate_g)),
    }


def run_case1_predbayes_qmp_table(
    q=TARGET_Q,
    level=INTERVAL_LEVEL,
    n_grid=(100, 200, 500),
    R=DEFAULT_MONTE_CARLO_REPS,
    original_B=DEFAULT_TOTAL_PATHS,
    B_boot=DEFAULT_BOOTSTRAP_REPS,
    S_per_boot=DEFAULT_PATHS_PER_BOOT,
    M_boot=None,
    seed=2026,
    T_exact=base_qmp.DEFAULT_EXACT_T,
    original_exact_use_vectorized=ORIGINAL_EXACT_USE_VECTORIZED,
    bagged_exact_use_vectorized=BAGGED_EXACT_USE_VECTORIZED,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    detail_rows = []
    theta0 = true_case1_quantiles(q)

    for n in n_grid:
        if verbose:
            print(f"\n[combined QMP] n = {n}, q = {q}")

        t_n_start = time.time()

        for r in range(R):
            if verbose and ((r + 1) % 10 == 0 or r == 0):
                print(f"  replication {r + 1}/{R}")

            y_by_scenario = make_case1_data(rng, n)

            for variant_idx, variant in enumerate(VARIANT_ORDER):
                for method_idx, method in enumerate(METHOD_ORDER):
                    for scenario_idx, (scenario, dgp) in enumerate(SCENARIOS):
                        seed_base = (
                            10000000 * variant_idx
                            + 1000000 * method_idx
                            + 10000 * n
                            + 10 * r
                            + scenario_idx
                        )
                        theta_draws = draw_theta_for_variant(
                            y_by_scenario[scenario],
                            variant=variant,
                            method=method,
                            q=q,
                            original_B=original_B,
                            B_boot=B_boot,
                            S_per_boot=S_per_boot,
                            M_boot=M_boot,
                            seed_fit=100000 + seed_base,
                            seed_post=200000 + seed_base,
                            seed_boot=300000 + seed_base,
                            T_exact=T_exact,
                            original_exact_use_vectorized=original_exact_use_vectorized,
                            bagged_exact_use_vectorized=bagged_exact_use_vectorized,
                        )
                        out = base_qmp.posterior_summary(theta_draws, level=level)
                        target = theta0[scenario]
                        detail_rows.append({
                            "q": float(q),
                            "variant": variant,
                            "method": method,
                            "method_label": METHOD_LABELS[(variant, method)],
                            "n": int(n),
                            "replication": int(r + 1),
                            "scenario": scenario,
                            "dgp": dgp,
                            "R": int(R),
                            "original_B": int(original_B),
                            "B_boot": int(B_boot) if variant == "bagged" else np.nan,
                            "S_per_boot": int(S_per_boot) if variant == "bagged" else np.nan,
                            "total_draws": int(B_boot * S_per_boot)
                            if variant == "bagged"
                            else int(original_B),
                            "posterior_mean": out["center"],
                            "true_quantile": target,
                            "signed_bias": out["center"] - target,
                            "abs_bias": abs(out["center"] - target),
                            "ci_lo": out["ci_lo"],
                            "ci_hi": out["ci_hi"],
                            "ci_width": out["ci_hi"] - out["ci_lo"],
                            "covered": float(out["ci_lo"] <= target <= out["ci_hi"]),
                        })

                        del theta_draws
                        gc.collect()

        t_n_end = time.time()

        if verbose:
            print(f"  finished n={n} in {round(t_n_end - t_n_start, 2)} seconds")

    detail_df = pd.DataFrame(detail_rows)
    group_cols = ["q", "variant", "method", "method_label", "n", "scenario", "dgp"]
    summary_df = (
        detail_df
        .groupby(group_cols, sort=False)
        .agg(
            coverage=("covered", "mean"),
            signed_bias=("signed_bias", "mean"),
            abs_bias=("abs_bias", "mean"),
            mean_ci_width=("ci_width", "mean"),
            R=("R", "first"),
            original_B=("original_B", "first"),
            B_boot=("B_boot", "first"),
            S_per_boot=("S_per_boot", "first"),
            total_draws=("total_draws", "first"),
        )
        .reset_index()
    )
    summary_df["variant_order"] = summary_df["variant"].map({
        value: idx for idx, value in enumerate(VARIANT_ORDER)
    })
    summary_df["method_order"] = summary_df["method"].map({
        value: idx for idx, value in enumerate(METHOD_ORDER)
    })
    summary_df["scenario_order"] = summary_df["scenario"].map({
        value: idx for idx, (value, _) in enumerate(SCENARIOS)
    })
    summary_df = (
        summary_df
        .sort_values(["variant_order", "n", "scenario_order", "method_order"])
        .reset_index(drop=True)
    )
    return summary_df, detail_df


def table_entry(coverage, signed_bias):
    return f"{coverage:.3f} ({signed_bias:.3f})"


def make_predbayes_table(summary_df, q=TARGET_Q):
    """Create Table 10 (q = 0.95), ordinary rows first, then bagged rows."""
    rows = []
    for variant in VARIANT_ORDER:
        variant_df = summary_df[summary_df["variant"] == variant]
        for n in sorted(variant_df["n"].unique()):
            row = {
                "panel": "Original PBP" if variant == "original" else "Double-bootstrap PBP",
                "q": float(q),
                "n": int(n),
            }
            for scenario, dgp in SCENARIOS:
                for method in METHOD_ORDER:
                    match = variant_df[
                        (variant_df["n"] == n)
                        & (variant_df["scenario"] == scenario)
                        & (variant_df["method"] == method)
                    ]
                    col = f"{dgp} {method}"
                    if match.empty:
                        row[col] = ""
                        continue
                    record = match.iloc[0]
                    row[col] = table_entry(
                        float(record["coverage"]),
                        float(record["signed_bias"]),
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def run_and_save_predbayes_qmp_table(
    q=TARGET_Q,
    level=INTERVAL_LEVEL,
    n_grid=(100, 200, 500),
    R=DEFAULT_MONTE_CARLO_REPS,
    original_B=DEFAULT_TOTAL_PATHS,
    B_boot=DEFAULT_BOOTSTRAP_REPS,
    S_per_boot=DEFAULT_PATHS_PER_BOOT,
    M_boot=None,
    seed=2026,
    T_exact=base_qmp.DEFAULT_EXACT_T,
    original_exact_use_vectorized=ORIGINAL_EXACT_USE_VECTORIZED,
    bagged_exact_use_vectorized=BAGGED_EXACT_USE_VECTORIZED,
    verbose=True,
    save_table=True,
    save_summary=False,
    save_detail=False,
):
    summary, detail = run_case1_predbayes_qmp_table(
        q=q,
        level=level,
        n_grid=n_grid,
        R=R,
        original_B=original_B,
        B_boot=B_boot,
        S_per_boot=S_per_boot,
        M_boot=M_boot,
        seed=seed,
        T_exact=T_exact,
        original_exact_use_vectorized=original_exact_use_vectorized,
        bagged_exact_use_vectorized=bagged_exact_use_vectorized,
        verbose=verbose,
    )

    table = make_predbayes_table(summary, q=q)
    prefix = combined_output_prefix(q, R, original_B, B_boot, S_per_boot)
    if save_table:
        table.to_csv(f"{prefix}_predbayes_table.csv", index=False)

    if save_summary:
        summary.to_csv(f"{prefix}_summary_debug.csv", index=False)

    if save_detail:
        detail.to_csv(f"{prefix}_detail_debug.csv", index=False)

    return summary, detail, table


if __name__ == "__main__":
    # Table 10 of the paper: ordinary QMP rows first, then bagged rows.
    # B_boot = 50 and S = 20 keep the bagged total at the ordinary QMP's
    # 1000 paths. Prefer scripts/Run_Quantile_Tail_Table.py.
    summary_df, detail_df, table_df = run_and_save_predbayes_qmp_table(
        q=TARGET_Q,
        level=INTERVAL_LEVEL,
        n_grid=(100, 200, 500),
        R=DEFAULT_MONTE_CARLO_REPS,
        original_B=DEFAULT_TOTAL_PATHS,
        B_boot=DEFAULT_BOOTSTRAP_REPS,
        S_per_boot=DEFAULT_PATHS_PER_BOOT,
        M_boot=None,
        seed=2026,
        T_exact=base_qmp.DEFAULT_EXACT_T,
        original_exact_use_vectorized=ORIGINAL_EXACT_USE_VECTORIZED,
        bagged_exact_use_vectorized=BAGGED_EXACT_USE_VECTORIZED,
        verbose=True,
        save_summary=False,
        save_detail=False,
    )

    print("\n================ QMP original + double-bootstrap table ================\n")
    print(table_df.to_string(index=False))
