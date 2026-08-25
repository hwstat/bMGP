# Coverage-sweep driver for the quantile functional; imported by
# scripts/Run_Quantile_Tables.py.
import argparse
import gc
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "false")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qmp_base as base_qmp
import qmp.qmp_functions as qmp_functions
import qmp.sample_qmp_functions as sqmp

Q_GRID = (0.25, 0.50, 0.75, 0.95)
INTERVAL_LEVEL = 0.95
R_REPS = 200
ORIGINAL_B = 1000
B_BOOT = 50
S_PER_BOOT = 20
QMP_ORACLE_N = 20000
VAR_FLOOR = 1e-6
DEFAULT_N_JOBS = max(1, os.cpu_count() or 1)
DEFAULT_REP_CHUNK_SIZE = 10

VARIANTS = ("original", "double_bootstrap")
METHODS = ("gpe", "qmp_exact", "qmp_gp")
SCENARIOS = ("well", "miss")

VARIANT_LABELS = {
    "original": "Original PBP",
    "double_bootstrap": "Double-bootstrap PBP",
}

METHOD_LABELS = {
    "gpe": "GPE",
    "qmp_exact": "QMP",
    "qmp_gp": "QMP-GP",
}

DGP_LABELS = {
    "well": "DGP:N(0,1)",
    "miss": "DGP:Ga(2,2)",
}

def q_tag(q):
    q_percent = 100.0 * float(q)
    if np.isclose(q_percent, round(q_percent)):
        return f"q{int(round(q_percent)):02d}"
    return "q" + f"{q_percent:.3f}".rstrip("0").rstrip(".").replace(".", "p")

def q_grid_tag(q_grid):
    return "q" + "_".join(q_tag(q).replace("q", "") for q in q_grid)

def available_cores():
    return max(1, os.cpu_count() or 1)

def resolve_n_jobs(n_jobs, n_tasks):
    if n_jobs is None or int(n_jobs) <= 0:
        n_jobs = available_cores()
    return max(1, min(int(n_jobs), int(n_tasks)))

def n_sim_from_n(n):
    return int(math.ceil(float(n) ** 1.5))

def data_seed(base_seed, scenario, n, r):
    scenario_offset = 0 if scenario == "well" else 500_000
    return int(base_seed + 10_000_000 + scenario_offset + 10_000 * int(n) + int(r))

def engine_seed(base_seed, variant, method, scenario, n, r):
    variant_offset = {"original": 0, "double_bootstrap": 20_000_000}[variant]
    method_offset = {"gpe": 0, "qmp_exact": 2_000_000, "qmp_gp": 4_000_000}[method]
    scenario_offset = 0 if scenario == "well" else 500_000
    return int(base_seed + variant_offset + method_offset + scenario_offset + 10_000 * int(n) + int(r))

def sample_dgp(scenario, n, rng):
    if scenario == "well":
        return rng.normal(loc=0.0, scale=1.0, size=int(n)).astype(np.float32)
    if scenario == "miss":
        return rng.gamma(shape=2.0, scale=0.5, size=int(n)).astype(np.float32)
    raise ValueError("scenario must be 'well' or 'miss'.")

def validate_q_grid(q_grid):
    q_values = tuple(float(q) for q in q_grid)
    if any(q <= 0.0 or q >= 1.0 for q in q_values):
        raise ValueError("All q values must lie strictly between 0 and 1.")
    return q_values

def true_p0_targets(q_grid):
    out = {}
    for q in q_grid:
        out[("well", q)] = float(norm.ppf(q))
        out[("miss", q)] = float(gamma.ppf(q, a=2.0, scale=0.5))
    return out

def gpe_fstar_targets(q_grid):
    out = {}
    for q in q_grid:
        out[("well", q)] = float(norm.ppf(q, loc=0.0, scale=1.0))
        out[("miss", q)] = float(norm.ppf(q, loc=1.0, scale=math.sqrt(0.5)))
    return out

def qmp_oracle_proxy_targets(q_grid, oracle_n=QMP_ORACLE_N, seed=9173):
    """Diagnostic proxy for QMP engine target, not a formal closed-form F-star.

    The QMP paper and official edfong/qmp scripts do not provide a closed-form
    QMP F-star calculation for this experiment. This large-sample initialization
    is saved only as a diagnostic for path-dependence and tail behavior.
    """
    out = {}
    meta = {}
    for scenario in SCENARIOS:
        rng = np.random.default_rng(seed + (0 if scenario == "well" else 100_000))
        y_oracle = sample_dgp(scenario, int(oracle_n), rng)
        fit_obj = base_qmp.fit_qmp_init(
            y_oracle,
            seed_fit=seed + (11 if scenario == "well" else 22),
            du=base_qmp.DU,
            n_perm=base_qmp.N_PERM,
        )
        q_init = np.asarray(jax.device_get(
            qmp_functions.rearrange_Q(jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32))
        ), dtype=np.float32)
        for q in q_grid:
            out[(scenario, q)] = float(np.interp(float(q), base_qmp.U_GRID, q_init))
        meta[scenario] = {
            "qmp_oracle_n": int(oracle_n),
            "qmp_oracle_a": float(fit_obj["a"]),
            "qmp_oracle_c": float(fit_obj["c"]),
            "qmp_oracle_preq_score": float(fit_obj["preq_score"]),
            "qmp_oracle_n_rearr": float(fit_obj["n_rearr"]),
        }
        del y_oracle, fit_obj, q_init
        gc.collect()
    return out, meta

def build_targets(q_grid, qmp_oracle_n, seed):
    p0 = true_p0_targets(q_grid)
    gpe_engine = gpe_fstar_targets(q_grid)
    qmp_proxy, qmp_proxy_meta = qmp_oracle_proxy_targets(
        q_grid,
        oracle_n=qmp_oracle_n,
        seed=seed + 50_000,
    )
    targets = {}
    target_rows = []
    for scenario in SCENARIOS:
        for q in q_grid:
            for method in METHODS:
                if method == "gpe":
                    engine_target = gpe_engine[(scenario, q)]
                    engine_type = "analytic_gpe_fstar"
                    oracle_meta = {}
                else:
                    engine_target = qmp_proxy[(scenario, q)]
                    engine_type = "qmp_oracle_init_proxy_not_formal_fstar"
                    oracle_meta = qmp_proxy_meta[scenario]
                targets[(method, scenario, q)] = {
                    "p0": float(p0[(scenario, q)]),
                    "engine": float(engine_target),
                    "engine_type": engine_type,
                }
                target_rows.append({
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "scenario": scenario,
                    "dgp": DGP_LABELS[scenario],
                    "q": float(q),
                    "target_p0": float(p0[(scenario, q)]),
                    "target_engine": float(engine_target),
                    "engine_target_type": engine_type,
                    **oracle_meta,
                })
    return targets, pd.DataFrame(target_rows)

def extract_from_curves(curves, q_grid):
    curves = np.asarray(curves, dtype=np.float64)
    out = np.empty((curves.shape[0], len(q_grid)), dtype=np.float64)
    for j, q in enumerate(q_grid):
        out[:, j] = np.array(
            [np.interp(float(q), base_qmp.U_GRID, row) for row in curves],
            dtype=np.float64,
        )
    return out

def extract_order_quantiles(values, q_grid):
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n_values = len(sorted_values)
    out = np.empty(len(q_grid), dtype=np.float64)
    for j, q in enumerate(q_grid):
        k = int(math.ceil(float(q) * n_values))
        k = min(max(k, 1), n_values)
        out[j] = sorted_values[k - 1]
    return out

def posterior_summary_multi(theta_draws, q_grid, level=INTERVAL_LEVEL):
    alpha = (1.0 - float(level)) / 2.0
    theta_draws = np.asarray(theta_draws, dtype=np.float64)
    rows = []
    for j, q in enumerate(q_grid):
        draws_q = theta_draws[:, j]
        ci_lo, ci_hi = np.quantile(draws_q, [alpha, 1.0 - alpha])
        rows.append({
            "q": float(q),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "center": float(np.mean(draws_q)),
            "ci_width": float(ci_hi - ci_lo),
        })
    return rows

def gaussian_predictive_engine_paths(x_obs, B, N, rng, var_floor=VAR_FLOOR):
    x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
    B = int(B)
    N = int(N)

    n_vec = np.full(B, len(x_obs), dtype=np.float64)
    sum_vec = np.full(B, np.sum(x_obs), dtype=np.float64)
    sumsq_vec = np.full(B, np.sum(x_obs ** 2), dtype=np.float64)
    z = rng.standard_normal(size=(B, N))
    paths = np.empty((B, N), dtype=np.float64)

    for m in range(N):
        mu_hat = sum_vec / n_vec
        numerator = sumsq_vec - (sum_vec ** 2) / n_vec
        denominator = n_vec - 1.0
        var_raw = np.where(denominator > 0.0, numerator / denominator, 0.0)
        var_hat = np.maximum(var_raw, var_floor)
        x_new = mu_hat + z[:, m] * np.sqrt(var_hat)
        paths[:, m] = x_new
        n_vec += 1.0
        sum_vec += x_new
        sumsq_vec += x_new ** 2

    return paths

def theta_draws_gpe_original_multi(x_obs, q_grid, B, N, rng, var_floor=VAR_FLOOR):
    paths = gaussian_predictive_engine_paths(x_obs, B=B, N=N, rng=rng, var_floor=var_floor)
    x_obs = np.asarray(x_obs, dtype=np.float64)
    theta = np.empty((int(B), len(q_grid)), dtype=np.float64)
    for b in range(int(B)):
        theta[b, :] = extract_order_quantiles(np.concatenate([x_obs, paths[b]]), q_grid)
    return theta, {}

def theta_draws_gpe_double_bootstrap_multi(
    x_obs,
    q_grid,
    B_boot,
    S_per_boot,
    N,
    rng,
    M_boot=None,
    var_floor=VAR_FLOOR,
):
    x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
    n = len(x_obs)
    if M_boot is None:
        M_boot = n
    M_boot = int(M_boot)

    theta = np.empty((int(B_boot) * int(S_per_boot), len(q_grid)), dtype=np.float64)
    for b in range(int(B_boot)):
        idx = rng.integers(0, n, size=M_boot)
        x_boot = x_obs[idx]
        paths = gaussian_predictive_engine_paths(
            x_boot,
            B=S_per_boot,
            N=N,
            rng=rng,
            var_floor=var_floor,
        )
        start = b * int(S_per_boot)
        for s in range(int(S_per_boot)):
            theta[start + s, :] = extract_order_quantiles(np.concatenate([x_boot, paths[s]]), q_grid)
    return theta, {}

def qmp_init_diagnostics(fit_obj, q_grid, target_p0_by_q):
    q_init = np.asarray(jax.device_get(
        qmp_functions.rearrange_Q(jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32))
    ), dtype=np.float32)
    init_quantiles = np.array([
        float(np.interp(float(q), base_qmp.U_GRID, q_init))
        for q in q_grid
    ], dtype=np.float64)
    init_bias = np.array([
        init_quantiles[j] - float(target_p0_by_q[float(q)])
        for j, q in enumerate(q_grid)
    ], dtype=np.float64)
    return {
        "init_quantile_mean": init_quantiles,
        "init_quantile_sd": np.zeros(len(q_grid), dtype=np.float64),
        "init_bias_p0_mean": init_bias,
        "fit_a_mean": float(fit_obj["a"]),
        "fit_c_mean": float(fit_obj["c"]),
        "fit_preq_score_mean": float(fit_obj["preq_score"]),
        "fit_n_rearr_mean": float(fit_obj["n_rearr"]),
    }

def qmp_draw_curves_from_fit(fit_obj, method, B, N, n_obs, seed_post, exact_use_vectorized):
    B = int(B)
    if method == "qmp_gp":
        Q_pr_gp = sqmp.approx_PR_B(
            int(seed_post),
            np.asarray(fit_obj["Q_init"], dtype=np.float32),
            float(fit_obj["a"]),
            float(fit_obj["c"]),
            float(fit_obj["k"]),
            int(n_obs),
            B,
        )
        Q_rearr = qmp_functions.rearrange_Q_B(jnp.asarray(Q_pr_gp, dtype=jnp.float32))
        return np.asarray(jax.device_get(Q_rearr), dtype=np.float32)

    if method != "qmp_exact":
        raise ValueError("method must be 'qmp_exact' or 'qmp_gp'.")

    keys = jax.random.split(jax.random.PRNGKey(int(seed_post)), B)
    if exact_use_vectorized:
        try:
            Q_pr = sqmp.PR_loop_B(
                keys,
                jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32),
                float(fit_obj["a"]),
                float(fit_obj["c"]),
                float(fit_obj["k"]),
                int(n_obs),
                int(N),
            )
            Q_rearr = qmp_functions.rearrange_Q_B(Q_pr)
            return np.asarray(jax.device_get(Q_rearr), dtype=np.float32)
        except (RuntimeError, MemoryError) as exc:
            print(f"Vectorized exact QMP failed; falling back to loop. Error: {exc}", flush=True)

    curves = np.empty((B, len(base_qmp.U_GRID)), dtype=np.float32)
    for b in range(B):
        Q_b = sqmp.PR_loop(
            keys[b],
            jnp.asarray(fit_obj["Q_init"], dtype=jnp.float32),
            float(fit_obj["a"]),
            float(fit_obj["c"]),
            float(fit_obj["k"]),
            int(n_obs),
            int(N),
        )
        Q_b = qmp_functions.rearrange_Q(Q_b)
        curves[b, :] = np.asarray(jax.device_get(Q_b), dtype=np.float32)
    return curves

def theta_draws_qmp_original_multi(
    x_obs,
    q_grid,
    target_p0_by_q,
    method,
    B,
    N,
    seed_fit,
    seed_post,
    exact_use_vectorized=False,
):
    x_obs = np.asarray(x_obs, dtype=np.float32).reshape(-1)
    fit_obj = base_qmp.fit_qmp_init(
        x_obs,
        seed_fit=int(seed_fit),
        du=base_qmp.DU,
        n_perm=base_qmp.N_PERM,
    )
    curves = qmp_draw_curves_from_fit(
        fit_obj=fit_obj,
        method=method,
        B=B,
        N=N,
        n_obs=len(x_obs),
        seed_post=seed_post,
        exact_use_vectorized=exact_use_vectorized,
    )
    theta = extract_from_curves(curves, q_grid)
    diag = qmp_init_diagnostics(fit_obj, q_grid, target_p0_by_q)
    return theta, diag

def theta_draws_qmp_double_bootstrap_multi(
    x_obs,
    q_grid,
    target_p0_by_q,
    method,
    B_boot,
    S_per_boot,
    N,
    rng,
    M_boot=None,
    exact_use_vectorized=False,
):
    x_obs = np.asarray(x_obs, dtype=np.float32).reshape(-1)
    n = len(x_obs)
    if M_boot is None:
        M_boot = n
    M_boot = int(M_boot)

    theta = np.empty((int(B_boot) * int(S_per_boot), len(q_grid)), dtype=np.float64)
    boot_init = np.empty((int(B_boot), len(q_grid)), dtype=np.float64)
    a_vals = np.empty(int(B_boot), dtype=np.float64)
    c_vals = np.empty(int(B_boot), dtype=np.float64)
    preq_vals = np.empty(int(B_boot), dtype=np.float64)
    rearr_vals = np.empty(int(B_boot), dtype=np.float64)

    for b in range(int(B_boot)):
        idx = rng.integers(0, n, size=M_boot)
        x_boot = x_obs[idx]
        seed_fit = int(rng.integers(1, 2**31 - 1))
        seed_post = int(rng.integers(1, 2**31 - 1))
        fit_obj = base_qmp.fit_qmp_init(
            x_boot,
            seed_fit=seed_fit,
            du=base_qmp.DU,
            n_perm=base_qmp.N_PERM,
        )
        diag_b = qmp_init_diagnostics(fit_obj, q_grid, target_p0_by_q)
        boot_init[b, :] = diag_b["init_quantile_mean"]
        a_vals[b] = diag_b["fit_a_mean"]
        c_vals[b] = diag_b["fit_c_mean"]
        preq_vals[b] = diag_b["fit_preq_score_mean"]
        rearr_vals[b] = diag_b["fit_n_rearr_mean"]

        curves = qmp_draw_curves_from_fit(
            fit_obj=fit_obj,
            method=method,
            B=int(S_per_boot),
            N=N,
            n_obs=len(x_boot),
            seed_post=seed_post,
            exact_use_vectorized=exact_use_vectorized,
        )
        start = b * int(S_per_boot)
        theta[start:start + int(S_per_boot), :] = extract_from_curves(curves, q_grid)
        del x_boot, fit_obj, curves
        if (b + 1) % 10 == 0:
            gc.collect()

    init_mean = np.mean(boot_init, axis=0)
    diag = {
        "init_quantile_mean": init_mean,
        "init_quantile_sd": np.std(boot_init, axis=0, ddof=1) if int(B_boot) > 1 else np.zeros(len(q_grid)),
        "init_bias_p0_mean": np.array([
            init_mean[j] - float(target_p0_by_q[float(q)])
            for j, q in enumerate(q_grid)
        ], dtype=np.float64),
        "fit_a_mean": float(np.mean(a_vals)),
        "fit_c_mean": float(np.mean(c_vals)),
        "fit_preq_score_mean": float(np.mean(preq_vals)),
        "fit_n_rearr_mean": float(np.mean(rearr_vals)),
    }
    return theta, diag

def theta_draws_for_case(
    x_obs,
    q_grid,
    target_p0_by_q,
    variant,
    method,
    original_B,
    B_boot,
    S_per_boot,
    N,
    seed,
    M_boot=None,
    var_floor=VAR_FLOOR,
    original_exact_use_vectorized=True,
    bagged_exact_use_vectorized=False,
):
    rng = np.random.default_rng(int(seed))

    if method == "gpe":
        if variant == "original":
            return theta_draws_gpe_original_multi(
                x_obs,
                q_grid=q_grid,
                B=original_B,
                N=N,
                rng=rng,
                var_floor=var_floor,
            )
        return theta_draws_gpe_double_bootstrap_multi(
            x_obs,
            q_grid=q_grid,
            B_boot=B_boot,
            S_per_boot=S_per_boot,
            N=N,
            rng=rng,
            M_boot=M_boot,
            var_floor=var_floor,
        )

    if method in {"qmp_exact", "qmp_gp"}:
        if variant == "original":
            seed_fit = int(rng.integers(1, 2**31 - 1))
            seed_post = int(rng.integers(1, 2**31 - 1))
            return theta_draws_qmp_original_multi(
                x_obs,
                q_grid=q_grid,
                target_p0_by_q=target_p0_by_q,
                method=method,
                B=original_B,
                N=N,
                seed_fit=seed_fit,
                seed_post=seed_post,
                exact_use_vectorized=original_exact_use_vectorized,
            )
        return theta_draws_qmp_double_bootstrap_multi(
            x_obs,
            q_grid=q_grid,
            target_p0_by_q=target_p0_by_q,
            method=method,
            B_boot=B_boot,
            S_per_boot=S_per_boot,
            N=N,
            rng=rng,
            M_boot=M_boot,
            exact_use_vectorized=bagged_exact_use_vectorized,
        )

    raise ValueError("Unknown method.")

def run_chunk_task(task):
    variant = task["variant"]
    method = task["method"]
    scenario = task["scenario"]
    n = int(task["n"])
    R_total = int(task["R"])
    r_start = int(task["r_start"])
    r_stop = int(task["r_stop"])
    q_grid = tuple(float(q) for q in task["q_grid"])
    level = float(task["level"])
    original_B = int(task["original_B"])
    B_boot = int(task["B_boot"])
    S_per_boot = int(task["S_per_boot"])
    M_boot = task["M_boot"]
    seed = int(task["seed"])
    targets = task["targets"]
    var_floor = float(task["var_floor"])
    original_exact_use_vectorized = bool(task["original_exact_use_vectorized"])
    bagged_exact_use_vectorized = bool(task["bagged_exact_use_vectorized"])

    N = n_sim_from_n(n)
    rows = []
    t0 = time.time()
    for r in range(r_start, r_stop):
        rng_data = np.random.default_rng(data_seed(seed, scenario, n, r))
        x_obs = sample_dgp(scenario, n, rng_data)
        target_p0_by_q = {
            float(q): float(targets[(method, scenario, float(q))]["p0"])
            for q in q_grid
        }
        draws, diag = theta_draws_for_case(
            x_obs,
            q_grid=q_grid,
            target_p0_by_q=target_p0_by_q,
            variant=variant,
            method=method,
            original_B=original_B,
            B_boot=B_boot,
            S_per_boot=S_per_boot,
            N=N,
            seed=engine_seed(seed, variant, method, scenario, n, r),
            M_boot=M_boot,
            var_floor=var_floor,
            original_exact_use_vectorized=original_exact_use_vectorized,
            bagged_exact_use_vectorized=bagged_exact_use_vectorized,
        )
        summ_rows = posterior_summary_multi(draws, q_grid=q_grid, level=level)
        for j, summ in enumerate(summ_rows):
            q = float(summ["q"])
            target = targets[(method, scenario, q)]
            target_p0 = float(target["p0"])
            target_engine = float(target["engine"])
            row = {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "scenario": scenario,
                "dgp": DGP_LABELS[scenario],
                "q": q,
                "n": n,
                "N": N,
                "R": R_total,
                "replication": int(r),
                "original_B": original_B,
                "B_boot": B_boot if variant == "double_bootstrap" else np.nan,
                "S_per_boot": S_per_boot if variant == "double_bootstrap" else np.nan,
                "total_draws": B_boot * S_per_boot if variant == "double_bootstrap" else original_B,
                "M_boot": M_boot if M_boot is not None else n,
                "target_p0": target_p0,
                "target_engine": target_engine,
                "engine_target_type": target["engine_type"],
                "ci_lo": summ["ci_lo"],
                "ci_hi": summ["ci_hi"],
                "center": summ["center"],
                "ci_width": summ["ci_width"],
                "covered_p0": float(summ["ci_lo"] <= target_p0 <= summ["ci_hi"]),
                "signed_bias_p0": float(summ["center"] - target_p0),
                "abs_bias_p0": float(abs(summ["center"] - target_p0)),
                "covered_engine": float(summ["ci_lo"] <= target_engine <= summ["ci_hi"]),
                "signed_bias_engine": float(summ["center"] - target_engine),
                "abs_bias_engine": float(abs(summ["center"] - target_engine)),
                "init_quantile_mean": np.nan,
                "init_quantile_sd": np.nan,
                "init_bias_p0_mean": np.nan,
                "fit_a_mean": np.nan,
                "fit_c_mean": np.nan,
                "fit_preq_score_mean": np.nan,
                "fit_n_rearr_mean": np.nan,
            }
            if method != "gpe":
                row["init_quantile_mean"] = float(diag["init_quantile_mean"][j])
                row["init_quantile_sd"] = float(diag["init_quantile_sd"][j])
                row["init_bias_p0_mean"] = float(diag["init_bias_p0_mean"][j])
                row["fit_a_mean"] = float(diag["fit_a_mean"])
                row["fit_c_mean"] = float(diag["fit_c_mean"])
                row["fit_preq_score_mean"] = float(diag["fit_preq_score_mean"])
                row["fit_n_rearr_mean"] = float(diag["fit_n_rearr_mean"])
            rows.append(row)
        del x_obs, draws
        if (r + 1) % 10 == 0:
            gc.collect()

    elapsed = time.time() - t0
    for row in rows:
        row["chunk_r_start"] = int(r_start)
        row["chunk_r_stop"] = int(r_stop)
        row["chunk_elapsed_seconds"] = float(elapsed)
    return rows

def make_tasks(
    q_grid,
    level,
    n_grid,
    R,
    original_B,
    B_boot,
    S_per_boot,
    M_boot,
    seed,
    targets,
    var_floor,
    original_exact_use_vectorized,
    bagged_exact_use_vectorized,
    rep_chunk_size,
):
    tasks = []
    if rep_chunk_size is None or int(rep_chunk_size) <= 0:
        rep_chunk_size = R
    rep_chunk_size = max(1, int(rep_chunk_size))
    for n in n_grid:
        for variant in VARIANTS:
            for method in METHODS:
                for scenario in SCENARIOS:
                    for r_start in range(0, int(R), rep_chunk_size):
                        r_stop = min(r_start + rep_chunk_size, int(R))
                        tasks.append({
                            "variant": variant,
                            "method": method,
                            "scenario": scenario,
                            "n": int(n),
                            "R": int(R),
                            "r_start": int(r_start),
                            "r_stop": int(r_stop),
                            "q_grid": tuple(float(q) for q in q_grid),
                            "level": float(level),
                            "original_B": int(original_B),
                            "B_boot": int(B_boot),
                            "S_per_boot": int(S_per_boot),
                            "M_boot": None if M_boot is None else int(M_boot),
                            "seed": int(seed),
                            "targets": targets,
                            "var_floor": float(var_floor),
                            "original_exact_use_vectorized": bool(original_exact_use_vectorized),
                            "bagged_exact_use_vectorized": bool(bagged_exact_use_vectorized),
                        })
    return tasks

def aggregate_replications(rep_df):
    group_cols = [
        "variant",
        "variant_label",
        "method",
        "method_label",
        "scenario",
        "dgp",
        "q",
        "n",
        "N",
        "R",
        "original_B",
        "B_boot",
        "S_per_boot",
        "total_draws",
        "M_boot",
        "target_p0",
        "target_engine",
        "engine_target_type",
    ]
    mean_cols = [
        "covered_p0",
        "signed_bias_p0",
        "abs_bias_p0",
        "covered_engine",
        "signed_bias_engine",
        "abs_bias_engine",
        "center",
        "ci_width",
        "init_quantile_mean",
        "init_quantile_sd",
        "init_bias_p0_mean",
        "fit_a_mean",
        "fit_c_mean",
        "fit_preq_score_mean",
        "fit_n_rearr_mean",
    ]
    summary = (
        rep_df
        .groupby(group_cols, dropna=False, sort=False)[mean_cols]
        .mean()
        .reset_index()
    )
    counts = (
        rep_df
        .groupby(group_cols, dropna=False, sort=False)["replication"]
        .count()
        .reset_index(name="R_completed")
    )
    summary = summary.merge(counts, on=group_cols, how="left")
    summary = summary.rename(columns={
        "covered_p0": "coverage_p0",
        "covered_engine": "coverage_engine",
        "center": "mean_center",
        "ci_width": "mean_ci_width",
    })
    return sort_summary(summary)

def sort_summary(summary_df):
    variant_order = {value: i for i, value in enumerate(VARIANTS)}
    method_order = {value: i for i, value in enumerate(METHODS)}
    scenario_order = {value: i for i, value in enumerate(SCENARIOS)}
    df = summary_df.copy()
    df["variant_order"] = df["variant"].map(variant_order)
    df["method_order"] = df["method"].map(method_order)
    df["scenario_order"] = df["scenario"].map(scenario_order)
    return (
        df.sort_values(["q", "variant_order", "n", "scenario_order", "method_order"])
        .drop(columns=["variant_order", "method_order", "scenario_order"])
        .reset_index(drop=True)
    )

def table_entry(coverage, signed_bias):
    return f"{float(coverage):.3f} ({float(signed_bias):.3f})"

def make_table(summary_df, target_kind="p0"):
    if target_kind == "p0":
        coverage_col = "coverage_p0"
        bias_col = "signed_bias_p0"
        target_label = "P0"
    elif target_kind == "engine":
        coverage_col = "coverage_engine"
        bias_col = "signed_bias_engine"
        target_label = "Fstar"
    else:
        raise ValueError("target_kind must be 'p0' or 'engine'.")

    rows = []
    for q in sorted(summary_df["q"].unique()):
        df_q = summary_df[np.isclose(summary_df["q"], q)]
        for variant in VARIANTS:
            df_v = df_q[df_q["variant"] == variant]
            for n in sorted(df_v["n"].unique()):
                row = {
                    "panel": VARIANT_LABELS[variant],
                    "target": target_label,
                    "q": float(q),
                    "n": int(n),
                }
                for scenario in SCENARIOS:
                    for method in METHODS:
                        match = df_v[
                            (df_v["n"] == n)
                            & (df_v["scenario"] == scenario)
                            & (df_v["method"] == method)
                        ]
                        col = f"{DGP_LABELS[scenario]} {METHOD_LABELS[method]}"
                        if match.empty:
                            row[col] = ""
                        else:
                            rec = match.iloc[0]
                            row[col] = table_entry(rec[coverage_col], rec[bias_col])
                rows.append(row)
    return pd.DataFrame(rows)

def make_combined_table(summary_df):
    return pd.concat(
        [
            make_table(summary_df, target_kind="p0"),
            make_table(summary_df, target_kind="engine"),
        ],
        ignore_index=True,
    )

def run_experiment(
    q_grid=Q_GRID,
    n_grid=(100, 200),
    level=INTERVAL_LEVEL,
    R=R_REPS,
    original_B=ORIGINAL_B,
    B_boot=B_BOOT,
    S_per_boot=S_PER_BOOT,
    M_boot=None,
    qmp_oracle_n=QMP_ORACLE_N,
    seed=2026,
    n_jobs=DEFAULT_N_JOBS,
    var_floor=VAR_FLOOR,
    original_exact_use_vectorized=True,
    bagged_exact_use_vectorized=False,
    rep_chunk_size=DEFAULT_REP_CHUNK_SIZE,
    verbose=True,
):
    q_grid = validate_q_grid(q_grid)
    if verbose:
        print("Computing P0 targets and engine-target diagnostics...")
    targets, target_df = build_targets(q_grid, qmp_oracle_n=qmp_oracle_n, seed=seed)
    if verbose:
        print(target_df.to_string(index=False))

    tasks = make_tasks(
        q_grid=q_grid,
        level=level,
        n_grid=n_grid,
        R=R,
        original_B=original_B,
        B_boot=B_boot,
        S_per_boot=S_per_boot,
        M_boot=M_boot,
        seed=seed,
        targets=targets,
        var_floor=var_floor,
        original_exact_use_vectorized=original_exact_use_vectorized,
        bagged_exact_use_vectorized=bagged_exact_use_vectorized,
        rep_chunk_size=rep_chunk_size,
    )
    n_jobs = resolve_n_jobs(n_jobs, len(tasks))
    if verbose:
        print(f"Using {n_jobs} worker process(es) out of {available_cores()} visible logical core(s).")
        print(f"Total chunks: {len(tasks)}")

    all_rows = []
    if n_jobs == 1:
        for i, task in enumerate(tasks, start=1):
            if verbose:
                print(
                    f"[{i}/{len(tasks)}] {task['variant']} {task['method']} "
                    f"{task['scenario']} n={task['n']} reps={task['r_start']}:{task['r_stop']}",
                    flush=True,
                )
            all_rows.extend(run_chunk_task(task))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as pool:
            futures = {pool.submit(run_chunk_task, task): task for task in tasks}
            for i, fut in enumerate(as_completed(futures), start=1):
                task = futures[fut]
                if verbose:
                    print(
                        f"[{i}/{len(tasks)}] done {task['variant']} {task['method']} "
                        f"{task['scenario']} n={task['n']} reps={task['r_start']}:{task['r_stop']}",
                        flush=True,
                    )
                all_rows.extend(fut.result())

    rep_df = pd.DataFrame(all_rows)
    summary_df = aggregate_replications(rep_df)
    table_p0_df = make_table(summary_df, target_kind="p0")
    table_engine_df = make_table(summary_df, target_kind="engine")
    table_combined_df = make_combined_table(summary_df)
    return rep_df, summary_df, table_p0_df, table_engine_df, table_combined_df, target_df

def default_output_prefix(label, q_grid, R, original_B, B_boot, S_per_boot):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(
        out_dir
        / (
            f"case1_quantile_sweep_{label}_{q_grid_tag(q_grid)}"
            f"_R{int(R)}_B{int(original_B)}_Bboot{int(B_boot)}_S{int(S_per_boot)}"
        )
    )

def build_parser(description, default_rep_chunk_size):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--q-grid", type=float, nargs="+", default=list(Q_GRID))
    parser.add_argument("--level", type=float, default=INTERVAL_LEVEL)
    parser.add_argument("--R", type=int, default=R_REPS)
    parser.add_argument("--B", type=int, default=ORIGINAL_B)
    parser.add_argument("--B-boot", type=int, default=B_BOOT)
    parser.add_argument("--S-per-boot", type=int, default=S_PER_BOOT)
    parser.add_argument("--M-boot", type=int, default=None)
    parser.add_argument("--qmp-oracle-n", type=int, default=QMP_ORACLE_N)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--rep-chunk-size", type=int, default=default_rep_chunk_size)
    parser.add_argument("--var-floor", type=float, default=VAR_FLOOR)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--no-vectorized-original-exact", action="store_true")
    parser.add_argument("--vectorized-bagged-exact", action="store_true")
    parser.add_argument("--vectorized-exact", action="store_true")
    parser.add_argument("--no-save-replications", action="store_true")
    return parser

def run_from_args(args, n_grid, label):
    q_grid = validate_q_grid(args.q_grid)
    original_exact_use_vectorized = (
        args.vectorized_exact or not args.no_vectorized_original_exact
    )
    bagged_exact_use_vectorized = (
        args.vectorized_exact or args.vectorized_bagged_exact
    )

    rep_df, summary_df, table_p0_df, table_engine_df, table_combined_df, target_df = run_experiment(
        q_grid=q_grid,
        n_grid=tuple(int(n) for n in n_grid),
        level=args.level,
        R=args.R,
        original_B=args.B,
        B_boot=args.B_boot,
        S_per_boot=args.S_per_boot,
        M_boot=args.M_boot,
        qmp_oracle_n=args.qmp_oracle_n,
        seed=args.seed,
        n_jobs=args.n_jobs,
        var_floor=args.var_floor,
        original_exact_use_vectorized=original_exact_use_vectorized,
        bagged_exact_use_vectorized=bagged_exact_use_vectorized,
        rep_chunk_size=args.rep_chunk_size,
        verbose=True,
    )

    prefix = args.output_prefix
    if prefix is None:
        prefix = default_output_prefix(label, q_grid, args.R, args.B, args.B_boot, args.S_per_boot)

    target_path = f"{prefix}_targets.csv"
    summary_path = f"{prefix}_summary.csv"
    table_p0_path = f"{prefix}_table_p0.csv"
    table_engine_path = f"{prefix}_table_engine_diagnostic.csv"
    table_combined_path = f"{prefix}_table_p0_and_fstar.csv"
    target_df.to_csv(target_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    table_p0_df.to_csv(table_p0_path, index=False)
    table_engine_df.to_csv(table_engine_path, index=False)
    table_combined_df.to_csv(table_combined_path, index=False)

    rep_path = None
    if not args.no_save_replications:
        rep_path = f"{prefix}_replications.csv"
        rep_df.to_csv(rep_path, index=False)

    print("\n================ P0 target table ================\n")
    print(table_p0_df.to_string(index=False))
    print("\n================ Engine-target diagnostic table ================\n")
    print(table_engine_df.to_string(index=False))
    print("\n================ Combined P0 and Fstar table ================\n")
    print(table_combined_df.to_string(index=False))
    print(f"\nSaved: {target_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {table_p0_path}")
    print(f"Saved: {table_engine_path}")
    print(f"Saved: {table_combined_path}")
    if rep_path is not None:
        print(f"Saved: {rep_path}")
