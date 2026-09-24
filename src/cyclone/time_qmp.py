#!/usr/bin/env python3
"""
Wall-clock timing for the quantile martingale posterior on the cyclone data,
with the settings of run_scripts/7.2_cyclone_qmp_small.py of the qmp
repository (https://github.com/edfong/qmp).

Upstream settings kept here:
  n_perm  = 10        permutations per prequential fit
  c grid  = 0.05 to 0.95 step 0.05   (19 values)
  k       = 0.5
  du      = 0.005     so the quantile curve sits on 199 grid points
  B       = 10000     predictive paths
  T       = 5000      predictive resampling steps
  seed    = 1706 (exact sampler), 5124 (GP sampler)

--B lets the same pipeline be run at a smaller number of paths, which is used
only for the matched-draw reference row in the report. Everything else is
upstream.

Two methods, chosen with --method:

  exact (default)  method "QMP". Times the fit, then PR_reg_loop_B, then
                   approx_PR_reg_B as a side measurement. The reported
                   pipeline is fit + exact.
  gp               method "QMP-GP". Times the fit, then approx_PR_reg_B
                   only, and reports fit + gp as the pipeline. This is the
                   Gaussian process approximation of Algorithm 7 costed as
                   a method in its own right rather than as a field on the
                   exact row.

Timed phases:
  fit    the c grid search plus the refit at c_opt
  exact  PR_reg_loop_B, the exact predictive resampling sampler
  gp     approx_PR_reg_B, the Gaussian process approximation

The exact and gp phases are blocked with block_until_ready so the JAX
asynchronous dispatch cannot hide work outside the timed region.

Usage
  python time_qmp.py --data data/globalTCmax4.txt --B 10000 --T 5000 --rep 1 \
      [--method exact|gp] --out <run.json>
Driven by scripts/Run_Table10_Cyclone_Timing.sh.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--B", type=int, default=10000, help="predictive paths (upstream: 10000)")
    p.add_argument("--T", type=int, default=5000, help="predictive steps (upstream: 5000)")
    p.add_argument("--seed-exact", type=int, default=1706)
    p.add_argument("--seed-gp", type=int, default=5124)
    p.add_argument("--rep", type=int, default=1, help="label for this repeat")
    p.add_argument("--method", choices=["exact", "gp"], default="exact",
                   help="exact: QMP via PR_reg_loop_B (default). "
                        "gp: QMP-GP via approx_PR_reg_B, timed as its own "
                        "method row.")
    p.add_argument("--skip-gp", action="store_true",
                   help="with --method exact, skip the side measurement of "
                        "the GP sampler")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t_wall0 = time.perf_counter()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    load_before = os.getloadavg()

    dat = load_cyclone_small(args.data)
    x, y, n = dat["x"], dat["y"], dat["n"]

    t0 = time.perf_counter()
    fit = fit_hyperparameters(x, y, fit_beta_perm)
    beta_init, a, c_opt, k = fit.beta_init, fit.a, fit.c_opt, fit.k
    t_fit = time.perf_counter() - t0

    t_exact = float("nan")
    beta_pr_exact = None
    if args.method == "exact":
        key = jax.random.split(jax.random.key(args.seed_exact), args.B)
        t0 = time.perf_counter()
        beta_pr_exact = PR_reg_loop_B(
            key, jnp.array(beta_init), jnp.array(x), a, c_opt, k, n, args.T
        )
        beta_pr_exact.block_until_ready()
        t_exact = time.perf_counter() - t0

    t_gp = float("nan")
    beta_pr_gp = None
    if args.method == "gp" or not args.skip_gp:
        t0 = time.perf_counter()
        beta_pr_gp = approx_PR_reg_B(args.seed_gp, beta_init, x, a, c_opt, k, n, args.B)
        beta_pr_gp = np.asarray(beta_pr_gp)
        t_gp = time.perf_counter() - t0

    if args.method == "exact":
        t_pipeline = t_fit + t_exact
        shape_of = beta_pr_exact
    else:
        t_pipeline = t_fit + t_gp
        shape_of = beta_pr_gp

    t_total_script = time.perf_counter() - t_wall0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    load_after = os.getloadavg()

    rec = {
        "method": "QMP" if args.method == "exact" else "QMP-GP",
        "variant": ("exact (PR_reg_loop_B)" if args.method == "exact"
                    else "GP approximation (approx_PR_reg_B)"),
        "rep": args.rep,
        "n": int(n),
        "d": int(dat["d"]),
        "paths_B": int(args.B),
        "predictive_steps_T": int(args.T),
        "n_perm": N_PERM,
        "c_grid_size": int(len(C_VALS)),
        "k": K_BAND,
        "du": DU,
        "total_draws": int(args.B),
        "a": float(a),
        "c_opt": float(c_opt),
        "n_nonfinite_preq": int(fit.n_nonfinite),
        "t_fit_s": round(t_fit, 4),
        "t_sample_exact_s": (round(t_exact, 4) if t_exact == t_exact else None),
        "t_sample_gp_s": (round(t_gp, 4) if t_gp == t_gp else None),
        "t_posterior_pipeline_s": round(t_pipeline, 4),
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
        "posterior_shape": list(np.shape(shape_of)),
    }
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
