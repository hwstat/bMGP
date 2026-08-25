# Core QMP recursions and samplers; imported by quantile_sweep.py.
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "false"

import gc
import time
import warnings
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import norm, gamma

import qmp.qmp_functions as qmp
import qmp.sample_qmp_functions as sqmp

TARGET_Q = 0.95         # Table 2 / bPBP comparison focuses on the upper quantile.
INTERVAL_LEVEL = 0.95   # 后验区间仍然用 95%

DU = 0.005
U_GRID = np.arange(DU, 1.0, DU, dtype=np.float32)

C_VALS = np.arange(0.05, 1.0, 0.05, dtype=np.float32)
K_BAND = np.float32(0.5)
N_PERM = 10
DEFAULT_POSTERIOR_B = 1000
OFFICIAL_EXACT_T = 5000
DEFAULT_EXACT_T = "n_power"

def quantile_file_tag(q):
    q_percent = 100.0 * float(q)
    if np.isclose(q_percent, round(q_percent)):
        return f"q{int(round(q_percent)):02d}"
    return "q" + f"{q_percent:.3f}".rstrip("0").rstrip(".").replace(".", "p")

def fit_qmp_init(y, seed_fit=123, du=DU, n_perm=N_PERM):
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    a = np.float32(np.sqrt(12.0) * np.std(y, ddof=0))

    preq_score = np.zeros(len(C_VALS), dtype=np.float32)
    n_rearr = np.zeros(len(C_VALS), dtype=np.float32)

    for i, c in enumerate(C_VALS):
        Q_plot_i, preq_score_i, n_rearr_i = qmp.fit_Q_perm(
            np.array([a, c, K_BAND], dtype=np.float32),
            y,
            du=du,
            n_perm=n_perm,
            seed=seed_fit
        )
        preq_score[i] = np.float32(np.asarray(jax.device_get(preq_score_i)))
        n_rearr[i] = np.float32(np.asarray(jax.device_get(n_rearr_i)))

    finite_mask = np.isfinite(preq_score)
    if not np.any(finite_mask):
        raise RuntimeError("All preq_score values are non-finite; cannot choose c.")

    idx_best = np.where(finite_mask)[0][np.argmax(preq_score[finite_mask])]
    c_opt = np.float32(C_VALS[idx_best])

    Q_init, preq_score_opt, n_rearr_opt = qmp.fit_Q_perm(
        np.array([a, c_opt, K_BAND], dtype=np.float32),
        y,
        du=du,
        n_perm=n_perm,
        seed=seed_fit
    )

    Q_init = np.asarray(jax.device_get(Q_init), dtype=np.float32)

    return {
        "Q_init": Q_init,
        "a": np.float32(a),
        "c": np.float32(c_opt),
        "k": np.float32(K_BAND),
        "preq_score": float(np.asarray(jax.device_get(preq_score_opt))),
        "n_rearr": float(np.asarray(jax.device_get(n_rearr_opt))),
        "preq_score_grid": preq_score,
        "n_rearr_grid": n_rearr,
    }

def extract_theta_draws(Q_draws_rearr, q=TARGET_Q, du=DU):
    u_grid = np.arange(du, 1.0, du, dtype=np.float32)
    Q_draws_rearr = np.asarray(Q_draws_rearr, dtype=np.float32)

    q = np.float32(np.clip(q, du, 1.0 - du))

    theta_draws = np.array(
        [np.interp(q, u_grid, row) for row in Q_draws_rearr],
        dtype=np.float32
    )
    return theta_draws

def theta_draws_exact_qmp(
    y,
    q=TARGET_Q,
    B=DEFAULT_POSTERIOR_B,
    T_exact=DEFAULT_EXACT_T,
    seed_fit=123,
    seed_post=456,
    use_vectorized=True
):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = len(y)

    N_sim = int(np.ceil(n ** 1.5)) if T_exact == "n_power" else int(T_exact)
    fit_obj = fit_qmp_init(y, seed_fit=seed_fit, du=DU, n_perm=N_PERM)

    keys = jax.random.split(jax.random.PRNGKey(seed_post), B)

    if use_vectorized:
        try:
            Q_pr = sqmp.PR_loop_B(
                keys,
                jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32),
                float(fit_obj["a"]),
                float(fit_obj["c"]),
                float(fit_obj["k"]),
                int(n),
                int(N_sim)
            )
            Q_pr_rearr = qmp.rearrange_Q_B(Q_pr)
            Q_pr_rearr = np.asarray(jax.device_get(Q_pr_rearr), dtype=np.float32)
            theta_draws = extract_theta_draws(Q_pr_rearr, q=q, du=DU)
            return theta_draws, fit_obj
        except (RuntimeError, MemoryError) as exc:
            warnings.warn(
                "Vectorized PR_loop_B failed; falling back to single-draw PR_loop. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2
            )

    theta_draws = np.empty(B, dtype=np.float32)

    for b in range(B):
        Q_b = sqmp.PR_loop(
            keys[b],
            jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32),
            fit_obj["a"],
            fit_obj["c"],
            fit_obj["k"],
            n,
            N_sim
        )
        Q_b = qmp.rearrange_Q(Q_b)
        Q_b = np.asarray(jax.device_get(Q_b), dtype=np.float32)

        theta_draws[b] = np.float32(
            np.interp(np.float32(q), U_GRID, Q_b)
        )

        del Q_b
        if (b + 1) % 20 == 0:
            gc.collect()

    return theta_draws, fit_obj

def theta_draws_gp_qmp(
    y,
    q=TARGET_Q,
    B=DEFAULT_POSTERIOR_B,
    seed_fit=123,
    seed_post=456
):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = len(y)

    fit_obj = fit_qmp_init(y, seed_fit=seed_fit, du=DU, n_perm=N_PERM)

    Q_pr_gp = sqmp.approx_PR_B(
        int(seed_post),
        np.asarray(fit_obj["Q_init"], dtype=np.float32),
        float(fit_obj["a"]),
        float(fit_obj["c"]),
        float(fit_obj["k"]),
        int(n),
        int(B)
    )

    Q_pr_gp_rearr = qmp.rearrange_Q_B(jnp.asarray(Q_pr_gp, dtype=jnp.float32))
    Q_pr_gp_rearr = np.asarray(jax.device_get(Q_pr_gp_rearr), dtype=np.float32)

    theta_draws = extract_theta_draws(Q_pr_gp_rearr, q=q, du=DU)
    return theta_draws, fit_obj

def posterior_summary(theta_draws, level=INTERVAL_LEVEL):
    alpha = (1.0 - level) / 2.0
    ci_lo, ci_hi = np.quantile(theta_draws, [alpha, 1.0 - alpha])
    center = float(np.mean(theta_draws))
    return {
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "center": center
    }

def run_case1_qmp_table(
    method="gp",
    q=TARGET_Q,
    level=INTERVAL_LEVEL,
    n_grid=(100, 200, 500),
    R=50,
    B_cov=DEFAULT_POSTERIOR_B,
    seed=2026,
    verbose=True
):
    rng = np.random.default_rng(seed)

    shape_g = 2.0
    rate_g = 2.0

    theta0_well = float(norm.ppf(q))
    theta0_miss = float(gamma.ppf(q, a=shape_g, scale=1.0 / rate_g))

    summary_rows = []
    detail_rows = []

    for n in n_grid:
        if verbose:
            print(f"\n[{method}] n = {n}, q = {q}")

        t_n_start = time.time()

        for r in range(R):
            if verbose and ((r + 1) % 10 == 0 or r == 0):
                print(f"  replication {r+1}/{R}")

            y_w = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float32)

            if method == "exact":
                theta_w, _ = theta_draws_exact_qmp(
                    y_w,
                    q=q,
                    B=B_cov,
                    seed_fit=100000 + 1000 * n + r,
                    seed_post=200000 + 1000 * n + r
                )
            elif method == "gp":
                theta_w, _ = theta_draws_gp_qmp(
                    y_w,
                    q=q,
                    B=B_cov,
                    seed_fit=100000 + 1000 * n + r,
                    seed_post=200000 + 1000 * n + r
                )
            else:
                raise ValueError("method must be 'exact' or 'gp'.")

            out_w = posterior_summary(theta_w, level=level)
            detail_rows.append({
                "method": method,
                "n": n,
                "replication": r + 1,
                "scenario": "well",
                "posterior_mean": out_w["center"],
                "true_quantile": theta0_well,
                "signed_bias": out_w["center"] - theta0_well,
                "abs_bias": abs(out_w["center"] - theta0_well),
                "ci_lo": out_w["ci_lo"],
                "ci_hi": out_w["ci_hi"],
                "ci_width": out_w["ci_hi"] - out_w["ci_lo"],
                "covered": float(out_w["ci_lo"] <= theta0_well <= out_w["ci_hi"]),
            })

            y_m = rng.gamma(shape=shape_g, scale=1.0 / rate_g, size=n).astype(np.float32)

            if method == "exact":
                theta_m, _ = theta_draws_exact_qmp(
                    y_m,
                    q=q,
                    B=B_cov,
                    seed_fit=300000 + 1000 * n + r,
                    seed_post=400000 + 1000 * n + r
                )
            else:
                theta_m, _ = theta_draws_gp_qmp(
                    y_m,
                    q=q,
                    B=B_cov,
                    seed_fit=300000 + 1000 * n + r,
                    seed_post=400000 + 1000 * n + r
                )

            out_m = posterior_summary(theta_m, level=level)
            detail_rows.append({
                "method": method,
                "n": n,
                "replication": r + 1,
                "scenario": "miss",
                "posterior_mean": out_m["center"],
                "true_quantile": theta0_miss,
                "signed_bias": out_m["center"] - theta0_miss,
                "abs_bias": abs(out_m["center"] - theta0_miss),
                "ci_lo": out_m["ci_lo"],
                "ci_hi": out_m["ci_hi"],
                "ci_width": out_m["ci_hi"] - out_m["ci_lo"],
                "covered": float(out_m["ci_lo"] <= theta0_miss <= out_m["ci_hi"]),
            })

            gc.collect()

        t_n_end = time.time()

        df_n = pd.DataFrame([row for row in detail_rows if row["method"] == method and row["n"] == n])

        df_w = df_n[df_n["scenario"] == "well"]
        df_m = df_n[df_n["scenario"] == "miss"]

        summary_rows.append({
            "n": n,
            "cover_well_theta0": float(df_w["covered"].mean()),
            "cover_miss_theta0": float(df_m["covered"].mean()),
            "bhat_well": float(df_w["abs_bias"].mean()),
            "bhat_miss": float(df_m["abs_bias"].mean()),
            "mean_signed_bias_well": float(df_w["signed_bias"].mean()),
            "mean_signed_bias_miss": float(df_m["signed_bias"].mean()),
            "mean_ci_width_well": float(df_w["ci_width"].mean()),
            "mean_ci_width_miss": float(df_m["ci_width"].mean()),
        })

        if verbose:
            print(f"  finished n={n} in {round(t_n_end - t_n_start, 2)} seconds")

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    return summary_df, detail_df

def run_and_save_both(
    q=TARGET_Q,
    level=INTERVAL_LEVEL,
    n_grid=(100, 200, 500),
    R_exact=50,
    B_exact=DEFAULT_POSTERIOR_B,
    R_gp=200,
    B_gp=DEFAULT_POSTERIOR_B,
    seed=2026,
    verbose=True
):
    if verbose:
        print("\n================ Running QMP exact (Algorithm 4) ================\n")
    exact_summary, exact_detail = run_case1_qmp_table(
        method="exact",
        q=q,
        level=level,
        n_grid=n_grid,
        R=R_exact,
        B_cov=B_exact,
        seed=seed,
        verbose=verbose
    )

    if verbose:
        print("\n============= Running QMP GP approx (Algorithm 5) ===============\n")
    gp_summary, gp_detail = run_case1_qmp_table(
        method="gp",
        q=q,
        level=level,
        n_grid=n_grid,
        R=R_gp,
        B_cov=B_gp,
        seed=seed,
        verbose=verbose
    )

    q_tag = quantile_file_tag(q)

    exact_summary.to_csv(f"case1_qmp_exact_{q_tag}_summary.csv", index=False)
    gp_summary.to_csv(f"case1_qmp_gp_{q_tag}_summary.csv", index=False)

    exact_detail.to_csv(f"case1_qmp_exact_{q_tag}_detail.csv", index=False)
    gp_detail.to_csv(f"case1_qmp_gp_{q_tag}_detail.csv", index=False)

    exact_bias = exact_summary[[
        "n",
        "mean_signed_bias_well",
        "mean_signed_bias_miss",
        "bhat_well",
        "bhat_miss",
        "mean_ci_width_well",
        "mean_ci_width_miss"
    ]].copy()

    gp_bias = gp_summary[[
        "n",
        "mean_signed_bias_well",
        "mean_signed_bias_miss",
        "bhat_well",
        "bhat_miss",
        "mean_ci_width_well",
        "mean_ci_width_miss"
    ]].copy()

    exact_bias.to_csv(f"case1_qmp_exact_{q_tag}_bias_only.csv", index=False)
    gp_bias.to_csv(f"case1_qmp_gp_{q_tag}_bias_only.csv", index=False)

    return exact_summary, exact_detail, gp_summary, gp_detail

if __name__ == "__main__":
    exact_summary, exact_detail, gp_summary, gp_detail = run_and_save_both(
        q=TARGET_Q,
        level=INTERVAL_LEVEL,
        n_grid=(100, 200, 500),
        R_exact=50,     # exact is heavy; increase only if your machine can handle it
        B_exact=DEFAULT_POSTERIOR_B,
        R_gp=200,
        B_gp=DEFAULT_POSTERIOR_B,
        seed=2026,
        verbose=True
    )

    print("\n================ QMP exact summary ================\n")
    print(exact_summary)

    print("\n============= QMP GP approx summary ===============\n")
    print(gp_summary)
