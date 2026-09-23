#!/usr/bin/env python3
"""
Wall-clock timing for the bagged quantile martingale posterior on the cyclone
data.

The bagging loop is a thin wrapper around the same upstream functions that
time_qmp.py calls (fit_beta_perm and PR_reg_loop_B). It follows
src/quantile/quantile_sweep.py::theta_draws_qmp_double_bootstrap_multi step for
step, with the one change the regression setting forces: a bootstrap resample
draws row indices and carries the covariate row along with the response, so
(y_i, x_i) pairs stay together.

  for b = 1 .. B_boot:
      resample n row indices with replacement
      refit the hyperparameters on the resample   (full c grid search + refit)
      draw M_B predictive paths from the resampled fit, T steps each
  pool the B_boot * M_B paths

Paper settings: B_boot = 50, M_B = 20, so 1000 pooled draws. T is passed in and
is set to the same number of predictive steps as the QMP run being compared
against.

Note that the hyperparameters a and c are re-estimated inside every bootstrap
iteration. That is what src/quantile/quantile_sweep.py does (fit_qmp_init sits
inside the b loop) and it is where most of the bagged cost goes.

--sampler picks which QMP sampler each resample draws from:

  exact (default)  PR_reg_loop_B, method "bagged QMP".
  gp               approx_PR_reg_B, method "bagged QMP-GP". Same bootstrap
                   loop, same per-resample prequential refit, M_B draws per
                   resample, 1000 pooled draws.

PR_reg_loop_B declares a, c, k, n and T as static arguments, so each resample's
fresh a and c force an XLA recompile. That is left as it is: the reference
implementation recompiles per resample and the manuscript says so. The cost of
that recompilation was measured once with a cache-hit second call per resample;
the flag that produced it has been removed from this script so that no timed
run can double its sampling work.

Usage
  python time_bqmp.py --data data/globalTCmax4.txt --B-boot 50 --M-B 20 \
      --T 5000 --rep 1 [--sampler exact|gp] --out <run.json>
Driven by scripts/Run_Table10_Cyclone_Timing.sh.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent
# src/cyclone for the modules next to this file, src/quantile for the vendored
# qmp package (src/quantile/qmp/), whose regression modules these scripts import.
for _extra in (HERE, HERE.parent / "quantile"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


import numpy as np

import jax
import jax.numpy as jnp
from qmp.qmp_reg_functions import fit_beta_perm
from qmp.sample_qmp_reg_functions import PR_reg_loop_B, approx_PR_reg_B

from cyclone_common import load_cyclone_small, fit_hyperparameters, DU, N_PERM, C_VALS, K_BAND

BaggedRun = namedtuple(
    "BaggedRun",
    "pooled a_vals c_vals t_fit_total t_sample_total n_nonfinite_preq sampler")


def bagged_qmp(x, y, B_boot, M_B, T, seed, sampler="exact"):
    """Bagged QMP. Returns a BaggedRun with the pooled predictive-path array
    and the per-resample diagnostics.

    sampler="exact" draws each resample's M_B paths with PR_reg_loop_B, the
    exact predictive resampling sampler. sampler="gp" draws them with
    approx_PR_reg_B, the Gaussian process approximation; everything else about
    the loop, including the full prequential refit on every resample, is the
    same."""
    if sampler not in ("exact", "gp"):
        raise ValueError("sampler must be 'exact' or 'gp', got " + repr(sampler))
    rng = np.random.default_rng(int(seed))
    n = np.shape(x)[0]

    pooled = []
    a_vals = np.empty(B_boot)
    c_vals = np.empty(B_boot)
    t_fit_total = 0.0
    t_sample_total = 0.0
    n_nonfinite_preq = 0

    for b in range(B_boot):
        idx = rng.integers(0, n, size=n)
        x_b = x[idx, :]
        y_b = y[idx]
        seed_fit = int(rng.integers(1, 2 ** 31 - 1))
        seed_post = int(rng.integers(1, 2 ** 31 - 1))

        t0 = time.perf_counter()
        fit_b = fit_hyperparameters(x_b, y_b, fit_beta_perm, seed=seed_fit)
        t_fit_total += time.perf_counter() - t0
        beta_init_b, a_b, c_b, k_b = fit_b.beta_init, fit_b.a, fit_b.c_opt, fit_b.k
        a_vals[b] = a_b
        c_vals[b] = c_b
        n_nonfinite_preq += fit_b.n_nonfinite

        t0 = time.perf_counter()
        if sampler == "exact":
            key = jax.random.split(jax.random.key(seed_post), M_B)
            paths_b = PR_reg_loop_B(
                key, jnp.array(beta_init_b), jnp.array(x_b), a_b, c_b, k_b, n, T
            )
            paths_b.block_until_ready()
        else:
            paths_b = approx_PR_reg_B(
                seed_post, beta_init_b, x_b, a_b, c_b, k_b, n, M_B
            )
            paths_b = np.asarray(paths_b)
        t_sample_total += time.perf_counter() - t0

        pooled.append(np.asarray(paths_b))

    pooled = np.concatenate(pooled, axis=0)
    return BaggedRun(pooled=pooled, a_vals=a_vals, c_vals=c_vals,
                     t_fit_total=t_fit_total, t_sample_total=t_sample_total,
                     n_nonfinite_preq=n_nonfinite_preq, sampler=sampler)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--B-boot", type=int, default=50, help="bootstrap resamples (paper: 50)")
    p.add_argument("--M-B", type=int, default=20, help="paths per resample (paper: 20)")
    p.add_argument("--T", type=int, default=5000, help="predictive steps")
    p.add_argument("--seed", type=int, default=1706)
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--sampler", choices=["exact", "gp"], default="exact",
                   help="exact: PR_reg_loop_B (default, method 'bagged QMP'). "
                        "gp: approx_PR_reg_B (method 'bagged QMP-GP').")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t_wall0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    load_before = os.getloadavg()

    dat = load_cyclone_small(args.data)
    x, y, n = dat["x"], dat["y"], dat["n"]

    t0 = time.perf_counter()
    run = bagged_qmp(x, y, args.B_boot, args.M_B, args.T, args.seed,
                     sampler=args.sampler)
    t_bag = time.perf_counter() - t0
    pooled, a_vals, c_vals = run.pooled, run.a_vals, run.c_vals
    t_fit, t_sample = run.t_fit_total, run.t_sample_total

    t_total_script = time.perf_counter() - t_wall0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    load_after = os.getloadavg()

    rec = {
        "method": "bagged QMP" if args.sampler == "exact" else "bagged QMP-GP",
        "variant": ("exact (PR_reg_loop_B) inside bootstrap loop"
                    if args.sampler == "exact"
                    else "GP approximation (approx_PR_reg_B) inside bootstrap loop"),
        "rep": args.rep,
        "n": int(n),
        "d": int(dat["d"]),
        "B_boot": int(args.B_boot),
        "M_B": int(args.M_B),
        "total_draws": int(args.B_boot * args.M_B),
        "predictive_steps_T": int(args.T),
        "n_perm": N_PERM,
        "c_grid_size": int(len(C_VALS)),
        "k": K_BAND,
        "du": DU,
        "a_mean": float(np.mean(a_vals)),
        "c_mean": float(np.mean(c_vals)),
        "n_nonfinite_preq": int(run.n_nonfinite_preq),
        "sampler": args.sampler,
        "t_refit_total_s": round(t_fit, 4),
        "t_sample_total_s": round(t_sample, 4),
        "t_posterior_pipeline_s": round(t_bag, 4),
        "t_script_total_s": round(t_total_script, 4),
        "cpu_time_s": round(cpu, 4),
        "cpu_over_wall": round(cpu / t_total_script, 3),
        "load_1min_before": round(load_before[0], 2),
        "load_1min_after": round(load_after[0], 2),
        "jax_version": jax.__version__,
        "jax_devices": str(jax.devices()),
        "python": platform.python_version(),
        "env_threads": {v: os.environ.get(v) for v in
                        ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS"]},
        "posterior_shape": list(np.shape(pooled)),
    }
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
