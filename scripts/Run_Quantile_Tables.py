#!/usr/bin/env python3
# Quantile-functional coverage tables (Section 4.2): QMP vs bagged QMP over
# one sweep produces the q = 0.5 and 0.75 tables (Tables 4 and 5); the q = 0.75 rows at
# n = 1000, 2000 are the same script with --n-grid 1000 2000 (long runtimes).
# Usage: python Run_Quantile_Tables.py [--n-grid 100 200 500 --out DIR]

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "quantile"))

import quantile_sweep as sweep

ALLOWED_N = (100, 200, 500, 1000, 2000)
DEFAULT_CHUNK = {100: 10, 200: 10, 500: 5, 1000: 2, 2000: 1}

def parse_args():
    parser = sweep.build_parser(
        description=("Run the bQMP quantile sweep for n in {100, 200, 500, 1000, 2000}. "
                     "Only these sample sizes are supported."),
        default_rep_chunk_size=sweep.DEFAULT_REP_CHUNK_SIZE,
    )
    parser.add_argument(
        "--n-grid", type=int, nargs="+", default=[100, 200, 500],
        help="sample sizes to run",
    )
    parser.add_argument(
        "--out", type=str, default=str(HERE / "output"),
        help="directory for the CSV outputs",
    )
    args = parser.parse_args()

    bad = sorted(set(args.n_grid) - set(ALLOWED_N))
    if bad:
        parser.error(
            "unsupported --n-grid value(s): %s. Supported values: %s."
            % (", ".join(str(b) for b in bad),
               ", ".join(str(a) for a in ALLOWED_N))
        )
    args.n_grid = sorted(set(args.n_grid))
    return args

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_given = any(a.startswith("--rep-chunk-size") for a in sys.argv[1:])
    prefix_given = args.output_prefix is not None

    print("bQMP sweep: n =", ", ".join(str(n) for n in args.n_grid),
          "| q =", ", ".join(str(q) for q in args.q_grid),
          "| R =", args.R, flush=True)

    q_grid = sweep.validate_q_grid(args.q_grid)
    for n_value in args.n_grid:
        label = "n%d" % n_value
        if not chunk_given:
            args.rep_chunk_size = DEFAULT_CHUNK[n_value]
        if not prefix_given:
            args.output_prefix = str(out_dir / (
                "case1_quantile_sweep_%s_%s_R%d_B%d_Bboot%d_S%d"
                % (label, sweep.q_grid_tag(q_grid), args.R, args.B,
                   args.B_boot, args.S_per_boot)))
        print("\n=================== n = %d ===================" % n_value,
              flush=True)
        sweep.run_from_args(args, n_grid=(n_value,), label=label)
        if not prefix_given:
            args.output_prefix = None

if __name__ == "__main__":
    main()
