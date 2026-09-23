#!/usr/bin/env python3
"""
Save the bagged QMP posterior draws for the cyclone data.

Same device as `posterior_qmp.py`: this script imports `time_bqmp`, replaces
`time_bqmp.bagged_qmp` by a pass-through that keeps the pooled draws, and then
calls `time_bqmp.main()` with the command line
`scripts/Run_Table10_Cyclone_Timing.sh` uses. The bootstrap
loop, the per-resample refits, the seeds and the sampler are the ones in
`time_bqmp.py`, which is left untouched.

The timing JSON that `time_bqmp.main()` writes lands next to the draws, in
`bqmp_posterior_run.json`, and is a by-product; the reported timings stay the
ones in `cyclone_timings.csv`.

Output: bqmp_beta_draws.npz (or bqmp_gp_beta_draws.npz) under
scripts/output/cyclone/posterior/

  beta_pooled  (B_boot * M_B, 199, 2)  pooled coefficient curve draws
  a_vals c_vals (B_boot,)              hyperparameters refitted per resample
  u_plot, mean_y, sd_y, mean_x, sd_x, ratio   as in posterior_qmp.py

The bootstrap resamples rows of the already standardised (y, x), so the same
ratio = sd_y / sd_x returns beta_1(u) to WmaxST units per year.

--sampler is passed straight through to time_bqmp.py: "exact" saves the bagged
QMP draws, "gp" saves the bagged QMP-GP draws.

Usage:
  python src/cyclone/posterior_bqmp.py --data data/globalTCmax4.txt \
      --B-boot 50 --M-B 20 --T 5000 --out <bqmp_beta_draws.npz>
  python src/cyclone/posterior_bqmp.py --data data/globalTCmax4.txt \
      --B-boot 50 --M-B 20 --T 5000 --sampler gp \
      --out <bqmp_gp_beta_draws.npz>
Driven by scripts/Run_Table11_Cyclone_Posterior.sh.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# src/cyclone for the modules next to this file, src/quantile for the vendored
# qmp package (src/quantile/qmp/), whose regression modules these scripts import.
for _extra in (HERE, HERE.parent / "quantile"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))


import numpy as np

import time_bqmp
from cyclone_common import load_cyclone_small

CAPTURED = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--B-boot", type=int, default=50)
    p.add_argument("--M-B", type=int, default=20)
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1706)
    p.add_argument("--sampler", choices=["exact", "gp"], default="exact")
    p.add_argument("--out", required=True)
    p.add_argument("--run-json", default=None)
    args = p.parse_args()

    default_json = ("bqmp_posterior_run.json" if args.sampler == "exact"
                    else "bqmp_gp_posterior_run.json")
    run_json = args.run_json or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), default_json)

    original = time_bqmp.bagged_qmp

    def wrapper(*a, **kw):
        out = original(*a, **kw)
        CAPTURED["pooled"] = out.pooled
        CAPTURED["a_vals"] = out.a_vals
        CAPTURED["c_vals"] = out.c_vals
        return out

    time_bqmp.bagged_qmp = wrapper

    argv_saved = sys.argv
    sys.argv = [
        "time_bqmp.py",
        "--data", args.data,
        "--B-boot", str(args.B_boot),
        "--M-B", str(args.M_B),
        "--T", str(args.T),
        "--seed", str(args.seed),
        "--rep", "1",
        "--sampler", args.sampler,
        "--out", run_json,
    ]
    try:
        time_bqmp.main()
    finally:
        sys.argv = argv_saved

    with open(run_json) as f:
        rec = json.load(f)

    dat = load_cyclone_small(args.data)

    pooled = np.asarray(CAPTURED["pooled"])
    u_plot = np.arange(0.005, 1.0, 0.005)
    assert pooled.shape == (args.B_boot * args.M_B, len(u_plot), dat["d"]), pooled.shape

    ratio = float(np.asarray(dat["sd_y"]).ravel()[0] /
                  np.asarray(dat["sd_x"]).ravel()[0])

    np.savez_compressed(
        args.out,
        beta_pooled=pooled,
        a_vals=np.asarray(CAPTURED["a_vals"]),
        c_vals=np.asarray(CAPTURED["c_vals"]),
        u_plot=u_plot,
        mean_y=float(dat["mean_y"]), sd_y=float(dat["sd_y"]),
        mean_x=float(np.asarray(dat["mean_x"]).ravel()[0]),
        sd_x=float(np.asarray(dat["sd_x"]).ravel()[0]),
        ratio=ratio,
        n=dat["n"], d=dat["d"],
        B_boot=args.B_boot, M_B=args.M_B,
        predictive_steps_T=args.T, seed=args.seed,
        sampler=args.sampler,
    )
    print("wrote {} sampler={} beta_pooled={} a_mean={:.4f} c_mean={:.4f} "
          "ratio={:.6f}".format(args.out, args.sampler, pooled.shape,
                                rec["a_mean"], rec["c_mean"], ratio))


if __name__ == "__main__":
    main()
