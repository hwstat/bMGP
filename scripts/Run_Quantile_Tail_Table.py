#!/usr/bin/env python3
# Appendix B.3, Table 10: tail experiment at q = 0.95 against the population
# quantile, QMP / QMP-GP and their bagged versions, n = 100, 200, 500, R = 200.
# Long runtime (several hours on one machine). Writes the table CSV to --out.
# Usage: python Run_Quantile_Tail_Table.py [--out DIR]

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "quantile"))

import tail_q95_table as tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=str(HERE / "output"),
                        help="directory for the CSV output")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    _, _, table = tail.run_and_save_predbayes_qmp_table(
        q=tail.TARGET_Q,
        level=tail.INTERVAL_LEVEL,
        n_grid=(100, 200, 500),
        R=tail.DEFAULT_MONTE_CARLO_REPS,
        original_B=tail.DEFAULT_TOTAL_PATHS,
        B_boot=tail.DEFAULT_BOOTSTRAP_REPS,
        S_per_boot=tail.DEFAULT_PATHS_PER_BOOT,
        M_boot=None,
        seed=2026,
        T_exact=tail.base_qmp.DEFAULT_EXACT_T,
        original_exact_use_vectorized=tail.ORIGINAL_EXACT_USE_VECTORIZED,
        bagged_exact_use_vectorized=tail.BAGGED_EXACT_USE_VECTORIZED,
        verbose=True,
        save_summary=False,
        save_detail=False,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
