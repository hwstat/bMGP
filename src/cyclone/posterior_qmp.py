#!/usr/bin/env python3
"""
Save the QMP posterior draws for the cyclone data.

This does not re-implement the QMP run. It imports `time_qmp` and calls
`time_qmp.main()` with the same command line that
`scripts/Run_Table10_Cyclone_Timing.sh` uses, after
replacing the two sampler names in `time_qmp`'s module namespace by thin
wrappers that hand the returned arrays back to this script on their way
through. The settings, the seeds, the hyperparameter search and the order of
operations are therefore the ones in `time_qmp.py`, which is left untouched.

The timing JSON that `time_qmp.main()` writes lands next to the draws, in
`qmp_posterior_run.json`. It is a by-product. The reported timings are the ones
in `cyclone_timings.csv`; this run carries the extra cost of moving
10000 x 199 x 2 floats to host memory and writing them out.

Output: scripts/output/cyclone/posterior/qmp_beta_draws.npz

  beta_exact   (B, 199, 2)  coefficient curve draws, PR_reg_loop_B, seed 1706
  beta_gp      (B, 199, 2)  the same from approx_PR_reg_B, seed 5124
  u_plot       (199,)       quantile levels, arange(0.005, 1, 0.005)
  mean_y sd_y mean_x sd_x   the standardisation the QMP script applies
  ratio        sd_y / sd_x, the factor that returns beta_1(u) to WmaxST units
                            per year (plots/plot_reg.ipynb of the qmp
                            repository, cell 4)

Usage:
  python src/cyclone/posterior_qmp.py --data data/globalTCmax4.txt \
      --B 10000 --T 5000 --out <qmp_beta_draws.npz>
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

import time_qmp
from cyclone_common import load_cyclone_small

CAPTURED = {}


def _wrap(name):
    """Replace time_qmp.<name> by a pass-through that keeps the result."""
    original = getattr(time_qmp, name)

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        CAPTURED[name] = out
        return out

    setattr(time_qmp, name, wrapper)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--B", type=int, default=10000)
    p.add_argument("--T", type=int, default=5000)
    p.add_argument("--seed-exact", type=int, default=1706)
    p.add_argument("--seed-gp", type=int, default=5124)
    p.add_argument("--out", required=True)
    p.add_argument("--run-json", default=None,
                   help="where time_qmp.main writes its timing record "
                        "(default: alongside --out)")
    args = p.parse_args()

    run_json = args.run_json or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "qmp_posterior_run.json")

    _wrap("PR_reg_loop_B")
    _wrap("approx_PR_reg_B")

    argv_saved = sys.argv
    sys.argv = [
        "time_qmp.py",
        "--data", args.data,
        "--B", str(args.B),
        "--T", str(args.T),
        "--seed-exact", str(args.seed_exact),
        "--seed-gp", str(args.seed_gp),
        "--rep", "1",
        "--out", run_json,
    ]
    try:
        time_qmp.main()
    finally:
        sys.argv = argv_saved

    with open(run_json) as f:
        rec = json.load(f)

    dat = load_cyclone_small(args.data)

    beta_exact = np.asarray(CAPTURED["PR_reg_loop_B"])
    beta_gp = np.asarray(CAPTURED["approx_PR_reg_B"])
    u_plot = np.arange(0.005, 1.0, 0.005)
    assert beta_exact.shape == (args.B, len(u_plot), dat["d"]), beta_exact.shape

    ratio = float(np.asarray(dat["sd_y"]).ravel()[0] /
                  np.asarray(dat["sd_x"]).ravel()[0])

    np.savez_compressed(
        args.out,
        beta_exact=beta_exact,
        beta_gp=beta_gp,
        u_plot=u_plot,
        mean_y=float(dat["mean_y"]), sd_y=float(dat["sd_y"]),
        mean_x=float(np.asarray(dat["mean_x"]).ravel()[0]),
        sd_x=float(np.asarray(dat["sd_x"]).ravel()[0]),
        ratio=ratio,
        a=rec["a"], c_opt=rec["c_opt"], k=rec["k"],
        n=dat["n"], d=dat["d"],
        paths_B=args.B, predictive_steps_T=args.T,
        seed_exact=args.seed_exact, seed_gp=args.seed_gp,
    )
    print("wrote {} beta_exact={} beta_gp={} a={:.4f} c_opt={:.2f} ratio={:.6f}".format(
        args.out, beta_exact.shape, beta_gp.shape,
        rec["a"], rec["c_opt"], ratio))


if __name__ == "__main__":
    main()
