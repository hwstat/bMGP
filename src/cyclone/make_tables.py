#!/usr/bin/env python3
"""Turns cyclone_timings.csv into the markdown and LaTeX tables of Table 11.
Every number printed here is read from the CSV, so nothing in the table is
transcribed by hand.

Usage
  python make_tables.py --csv <cyclone_timings.csv> \
      --md-out <timing_table.md> --tex-out <timing_table.tex> \
      [--load-warn 6.0]
Driven by scripts/Run_Table10_Cyclone_Timing.sh.

A wall-clock number is only worth as much as the machine it was measured on, so
any row whose one-minute load average before or after the run exceeded
--load-warn is listed under a warning line beneath the table.
"""

import argparse

import pandas as pd

LABELS = {
    "qmp_B10000": "QMP (Fong and Yiu settings)",
    "bqmp_50x20": "bagged QMP (paper settings)",
    "qmp_B1000": "QMP at 1000 paths (reference)",
    "qmpgp_B10000": "QMP-GP (Algorithm 7 approximation)",
    "bqmpgp_50x20": "bagged QMP-GP (paper settings)",
    "dqp_20000": "DQP (Edwin's MCMC settings)",
}
ORDER = ["qmp_B10000", "bqmp_50x20", "qmp_B1000",
         "qmpgp_B10000", "bqmpgp_50x20", "dqp_20000"]


def fmt(x, nd=1):
    return f"{x:,.{nd}f}"


def mmss(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--md-out", required=True)
    p.add_argument("--tex-out", required=True)
    p.add_argument("--load-warn", type=float, default=6.0,
                   help="warn about rows whose one-minute load average before "
                        "or after the run exceeded this (default 6.0, which on "
                        "a 14-core machine leaves about eight cores free)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["family"] = df["run_id"].str.replace(r"_rep\d+$", "", regex=True)

    rows = []
    for fam in ORDER:
        sub = df[df["family"] == fam].sort_values("rep")
        if sub.empty:
            continue
        t = sub["t_posterior_pipeline_s"].to_list()
        draws = sub["total_draws"].iloc[0]
        if pd.isna(draws):
            # DQP stores no total_draws; the posterior sample it keeps is
            # niter minus the burn-in, thinned.
            draws = ((sub["niter"].iloc[0] - sub["nburn"].iloc[0])
                     / sub["nthin"].iloc[0])
        rows.append({
            "family": fam,
            "label": LABELS[fam],
            "run1": t[0],
            "run2": t[1] if len(t) > 1 else float("nan"),
            "mean": sum(t) / len(t),
            "cpu": sub["cpu_time_s"].mean(),
            "cpu_over_wall": sub["cpu_over_wall"].mean(),
            "draws": int(draws),
        })
    out = pd.DataFrame(rows)

    md = ["| method | posterior draws | run 1 (s) | run 2 (s) | mean (s) | mean (m:ss) | CPU/wall |",
          "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        md.append(
            f"| {r['label']} | {r['draws']:,} | {fmt(r['run1'])} | {fmt(r['run2'])} | "
            f"{fmt(r['mean'])} | {mmss(r['mean'])} | {r['cpu_over_wall']:.1f} |"
        )
    md_txt = "\n".join(md)

    base = out.loc[out["family"] == "bqmp_50x20", "mean"]
    base = float(base.iloc[0]) if len(base) else float("nan")
    ratios = ["", "Ratio to bagged QMP:"]
    for _, r in out.iterrows():
        ratios.append(f"  {r['label']}: {r['mean'] / base:.2f}x")
    md_txt += "\n" + "\n".join(ratios)

    # A wall-clock number measured on a busy machine is not comparable with one
    # measured on a quiet machine, so say which rows were busy instead of
    # leaving it to the reader to open the CSV.
    load = df[["run_id", "load_1min_before", "load_1min_after"]].copy()
    busy = load[(load["load_1min_before"] > args.load_warn) |
                (load["load_1min_after"] > args.load_warn)]
    if len(busy):
        lines = ["", f"WARNING: {len(busy)} of {len(load)} runs saw a "
                     f"one-minute load average above {args.load_warn:g}:"]
        for _, r in busy.iterrows():
            lines.append(f"  {r['run_id']}: load {r['load_1min_before']} "
                         f"before, {r['load_1min_after']} after")
        lines.append("  Wall-clock times from those runs may be inflated.")
        md_txt += "\n" + "\n".join(lines)
    else:
        md_txt += (f"\n\nNo run saw a one-minute load average above "
                   f"{args.load_warn:g}.")

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wall-clock time to produce one posterior for the conditional",
        r"quantiles of lifetime maximum wind speed on the $n=291$ North Atlantic",
        r"cyclone records of \citet{elsner2008increasing}, on a single",
        r"Apple M4 Pro (14 cores, 48\,GB), with BLAS threads pinned to one. Each",
        r"method was run twice; both runs and their mean are reported. QMP-GP is",
        r"the Gaussian process approximation of \citet{fong2025bayesian}",
        r"(their Algorithm 7) in place of the exact predictive resampler. DQP is",
        r"the dependent quantile pyramid of \citet{an2024process}, run with the",
        r"MCMC settings of the scripts distributed with \citet{fong2025bayesian}.}",
        r"\label{tab:cyclone_runtime}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Draws & Run 1 (s) & Run 2 (s) & CPU (s) & Cores \\",
        r"\midrule",
    ]
    tex_labels = {
        "qmp_B10000": r"QMP, $M=10{,}000$ paths",
        "bqmp_50x20": r"bagged QMP, $B=50$, $M_B=20$",
        "qmp_B1000": r"QMP, $M=1000$ paths",
        "qmpgp_B10000": r"QMP-GP, $M=10{,}000$ draws",
        "bqmpgp_50x20": r"bagged QMP-GP, $B=50$, $M_B=20$",
        "dqp_20000": r"DQP, $20{,}000$ MCMC iterations",
    }
    for _, r in out.iterrows():
        tex.append(
            f"{tex_labels[r['family']]} & {r['draws']:,} & {fmt(r['run1'])} & "
            f"{fmt(r['run2'])} & {fmt(r['cpu'], 0 if r['cpu'] >= 10 else 1)} & {r['cpu_over_wall']:.1f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_txt = "\n".join(tex)

    with open(args.md_out, "w") as f:
        f.write(md_txt + "\n")
    with open(args.tex_out, "w") as f:
        f.write(tex_txt + "\n")
    print(md_txt)
    print()
    print(tex_txt)


if __name__ == "__main__":
    main()
