#!/usr/bin/env python3
"""Gathers the per-run JSON files into one CSV.

Usage
  python collect_timings.py --raw <raw json dir> --out <cyclone_timings.csv>
Driven by scripts/Run_Table10_Cyclone_Timing.sh.

Every run records the one-minute load average before and after itself. Those
two columns are carried through to the CSV, and make_tables.py warns about the
rows whose load was high, since a wall-clock number measured on a busy machine
is not comparable with one measured on a quiet machine.
"""

import argparse
import glob
import json
import os

import pandas as pd

COLS = [
    "run_id", "method", "variant", "rep", "n", "d", "total_draws",
    "paths_B", "B_boot", "M_B", "predictive_steps_T",
    "niter", "nburn", "nthin", "n_quantiles",
    "t_fit_s", "t_refit_total_s", "t_sample_exact_s", "t_sample_total_s",
    "t_sample_gp_s", "t_compile_s", "t_mcmc_s",
    "t_posterior_pipeline_s", "t_script_total_s",
    "cpu_time_s", "cpu_over_wall",
    "load_1min_before", "load_1min_after",
    "n_nonfinite_preq", "init_seed",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.raw, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        rec = {k: (v[0] if isinstance(v, list) and len(v) == 1 and not
                   isinstance(v[0], (list, dict)) else v)
               for k, v in rec.items()}
        rec["run_id"] = os.path.splitext(os.path.basename(path))[0]
        rows.append(rec)

    df = pd.DataFrame(rows)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLS].sort_values(["method", "run_id", "rep"])
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
