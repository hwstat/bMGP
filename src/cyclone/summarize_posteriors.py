#!/usr/bin/env python3
"""
Reduce the saved posterior draws to one slope summary per method and quantile
level, and write the CSV and the LaTeX table.

Target
------
For each quantile level q, the slope in Year of the conditional q-quantile of
WmaxST, in the original units of both variables.

  QMP, bagged QMP, QMP-GP, bagged QMP-GP
      The sampler returns beta(u) on the 199-point grid u = 0.005, ..., 0.995
      for the standardised y and x with an intercept, so Q(u | x) = beta(u)'x.
      The year coefficient is beta_1(u). The QMP script standardises both
      variables, so the slope on the original scale is

          beta_1(u) * sd_y / sd_x.

      This is the mapping in plots/plot_reg.ipynb of the qmp repository
      (https://github.com/edfong/qmp), which sets
      ratio = normalize['sd_y'] / normalize['sd_x'] and plots
      ratio * beta_mean_exact[u_ind, 1] on the same axes as the DQP slopes.

  DQP
      posterior_dqp.R follows run_scripts/7.2_cyclone_dqp/
      7.2_cyclone_dqp_process.R of the qmp repository: for each
      retained MCMC draw, fit a line to the 26 year-specific posterior
      quantiles and keep its slope. y is never standardised in the DQP code and
      the line is fitted against the raw years, so those slopes are already in
      WmaxST units per year.

The two GP rows are the Gaussian process approximation of Algorithm 5 in place
of the exact predictive resampler: QMP-GP is the beta_gp array that
posterior_qmp.py already saves (seed 5124), and bagged QMP-GP is the pooled
array from posterior_bqmp.py --sampler gp. Either is skipped if its file is not
there.

Effective sample size
---------------------
Reported for DQP, whose draws are an autocorrelated chain. Two estimators:
Geyer's initial positive sequence (the one in the table) and non-overlapping
batch means with batch size floor(sqrt(N)) (a cross-check in the CSV). The QMP
variants draw independent paths, so their effective sample size is the number
of draws.

Outputs, all under scripts/output/cyclone/posterior/
----------------------------------------------------
cyclone_slope_summary.csv
cyclone_slope_table.tex        the Table 12 body
cyclone_slope_checks.csv
cyclone_slope_draws_long.csv   the slope draws themselves

Usage
-----
  python src/cyclone/summarize_posteriors.py [--postdir DIR]
Driven by scripts/Run_Table11_Cyclone_Posterior.sh.
"""

import argparse
import os

import numpy as np
import pandas as pd

LEVELS = [0.50, 0.75, 0.90, 0.95]
DU = 0.005


# ----------------------------------------------------------------- estimators

def acov(x):
    """Autocovariance at lags 0 .. N-1, by FFT, normalised by N."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    out = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    return out / n


def ess_geyer(x):
    """Geyer's initial positive sequence effective sample size."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = acov(x)
    if g[0] <= 0:
        return float(n)
    k = n // 2
    gam = g[0:2 * k:2] + g[1:2 * k:2]          # Gamma_m = gamma_2m + gamma_{2m+1}
    neg = np.nonzero(gam <= 0)[0]
    m = int(neg[0]) if neg.size else len(gam)  # keep Gamma_0 .. Gamma_{m-1}
    if m == 0:
        return float(n)
    gam = np.minimum.accumulate(gam[:m])       # initial monotone sequence
    sigma2 = -g[0] + 2.0 * gam.sum()
    if sigma2 <= 0:
        return float(n)
    return float(n * g[0] / sigma2)


def ess_batch_means(x, b=None):
    """Non-overlapping batch means. Default batch size floor(sqrt(N)).

    Batch means needs a batch long relative to the correlation time. On this
    DQP chain floor(sqrt(N)) = 100 is much shorter than that, so the default
    reads high; the checks file also reports b = N/20 and b = N/10.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if b is None:
        b = max(1, int(np.floor(np.sqrt(n))))
    b = int(b)
    nb = n // b
    if nb < 2:
        return float(n)
    bm = x[:nb * b].reshape(nb, b).mean(axis=1)
    var_bm = b * bm.var(ddof=1)
    var_x = x.var(ddof=1)
    if var_bm <= 0:
        return float(n)
    return float(n * var_x / var_bm)


def summarise(draws, method, level, level_used, ess=None, ess_bm=None):
    d = np.asarray(draws, dtype=float)
    q025, q975 = np.quantile(d, [0.025, 0.975])
    return {
        "method": method,
        "level_requested": level,
        "level_used": level_used,
        "post_mean": float(d.mean()),
        "post_sd": float(d.std(ddof=1)),
        "ci95_lo": float(q025),
        "ci95_hi": float(q975),
        "ci95_width": float(q975 - q025),
        "n_draws": int(d.size),
        "ess_geyer": (float(d.size) if ess is None else ess),
        "ess_batch_means": (float(d.size) if ess_bm is None else ess_bm),
    }


# ----------------------------------------------------------------------- main

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))   # the repository root
    p = argparse.ArgumentParser()
    p.add_argument("--postdir",
                   default=os.path.join(root, "scripts", "output", "cyclone",
                                        "posterior"))
    p.add_argument("--dqp-reference",
                   default=os.path.join(root, "data", "DQP_beta1.csv"),
                   help="plots/plot_data/DQP_beta1.csv of the qmp repository, "
                        "vendored as data/DQP_beta1.csv. The comparison against "
                        "it is skipped when the file is absent.")
    args = p.parse_args()
    pd_ = args.postdir

    qmp = np.load(os.path.join(pd_, "qmp_beta_draws.npz"))
    bq = np.load(os.path.join(pd_, "bqmp_beta_draws.npz"))
    dqp = pd.read_csv(os.path.join(pd_, "dqp_slope_draws.csv"))

    bq_gp_path = os.path.join(pd_, "bqmp_gp_beta_draws.npz")
    bq_gp = np.load(bq_gp_path) if os.path.exists(bq_gp_path) else None
    if bq_gp is not None:
        assert abs(float(bq_gp["ratio"]) - float(bq["ratio"])) < 1e-12

    u_plot = np.asarray(qmp["u_plot"], dtype=float)
    ratio_qmp = float(qmp["ratio"])
    ratio_bq = float(bq["ratio"])
    assert abs(ratio_qmp - ratio_bq) < 1e-12, (ratio_qmp, ratio_bq)

    dqp_levels = np.array([float(c) for c in dqp.columns])

    rows = []
    draw_cols = {}
    for q in LEVELS:
        iu = int(np.argmin(np.abs(u_plot - q)))
        u_used = float(u_plot[iu])

        s_qmp = ratio_qmp * np.asarray(qmp["beta_exact"][:, iu, 1], dtype=float)
        rows.append(summarise(s_qmp, "QMP", q, u_used))
        draw_cols[("QMP", q)] = s_qmp

        s_bq = ratio_bq * np.asarray(bq["beta_pooled"][:, iu, 1], dtype=float)
        rows.append(summarise(s_bq, "bagged QMP", q, u_used))
        draw_cols[("bagged QMP", q)] = s_bq

        if "beta_gp" in qmp.files:
            s_qmp_gp = ratio_qmp * np.asarray(qmp["beta_gp"][:, iu, 1],
                                              dtype=float)
            rows.append(summarise(s_qmp_gp, "QMP-GP", q, u_used))
            draw_cols[("QMP-GP", q)] = s_qmp_gp

        if bq_gp is not None:
            s_bq_gp = ratio_bq * np.asarray(bq_gp["beta_pooled"][:, iu, 1],
                                            dtype=float)
            rows.append(summarise(s_bq_gp, "bagged QMP-GP", q, u_used))
            draw_cols[("bagged QMP-GP", q)] = s_bq_gp

        jd = int(np.argmin(np.abs(dqp_levels - q)))
        s_dqp = np.asarray(dqp.iloc[:, jd], dtype=float)
        rows.append(summarise(s_dqp, "DQP", q, float(dqp_levels[jd]),
                              ess=ess_geyer(s_dqp), ess_bm=ess_batch_means(s_dqp)))
        draw_cols[("DQP", q)] = s_dqp

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(pd_, "cyclone_slope_summary.csv")
    summary.to_csv(out_csv, index=False, float_format="%.6f")

    # keep the draws that produced the table; the three methods have different
    # numbers of draws, so this is long format
    long = pd.concat([
        pd.DataFrame({"method": m, "level": q, "draw": np.arange(v.size),
                      "slope": v})
        for (m, q), v in draw_cols.items()
    ], ignore_index=True)
    long.to_csv(os.path.join(pd_, "cyclone_slope_draws_long.csv"),
                index=False, float_format="%.6f")

    # ------------------------------------------------------------------ checks
    checks = []

    # (a) the OLS slope hard-coded in plots/plot_reg.ipynb of the qmp
    #     repository as the reference line for the integrated coefficient,
    #     1.790786e-01 on the standardised scale.
    nb_ols_std = 1.790786e-01
    checks.append({"check": "notebook OLS slope, standardised scale",
                   "value": nb_ols_std, "reference": np.nan})
    checks.append({"check": "notebook OLS slope, WmaxST per year",
                   "value": nb_ols_std * ratio_qmp, "reference": np.nan})

    # integrated coefficient bar-beta = int beta_1(u) du, the second panel of
    # that notebook cell
    bbar_qmp = np.sum(np.asarray(qmp["beta_exact"][:, :, 1], dtype=float),
                      axis=1) * DU * ratio_qmp
    bbar_bq = np.sum(np.asarray(bq["beta_pooled"][:, :, 1], dtype=float),
                     axis=1) * DU * ratio_bq
    checks.append({"check": "QMP posterior mean of integrated slope",
                   "value": float(bbar_qmp.mean()),
                   "reference": nb_ols_std * ratio_qmp})
    checks.append({"check": "bagged QMP posterior mean of integrated slope",
                   "value": float(bbar_bq.mean()),
                   "reference": nb_ols_std * ratio_qmp})

    # (b) our DQP re-run against the DQP_beta1.csv that ships with the qmp repo
    if os.path.exists(args.dqp_reference):
        ref = pd.read_csv(args.dqp_reference)
        for q in LEVELS:
            jr = int(np.argmin(np.abs(np.asarray(ref["Level"]) - q)))
            mine = summary[(summary.method == "DQP") &
                           (summary.level_requested == q)].iloc[0]
            checks.append({
                "check": "DQP mean slope at q={:.2f}, rerun vs repo DQP_beta1.csv".format(q),
                "value": float(mine["post_mean"]),
                "reference": float(ref["Slope"].iloc[jr])})
            checks.append({
                "check": "DQP 95% width at q={:.2f}, rerun vs repo DQP_beta1.csv".format(q),
                "value": float(mine["ci95_width"]),
                "reference": float(ref["q975"].iloc[jr] - ref["q025"].iloc[jr])})

    # (c) interval width ratios
    present = list(summary["method"].unique())
    for q in LEVELS:
        w = {m: float(summary[(summary.method == m) &
                              (summary.level_requested == q)]["ci95_width"].iloc[0])
             for m in present}
        pairs = [("bagged QMP", "QMP"), ("DQP", "QMP"), ("DQP", "bagged QMP"),
                 ("QMP-GP", "QMP"), ("bagged QMP-GP", "bagged QMP"),
                 ("bagged QMP-GP", "QMP-GP")]
        for num, den in pairs:
            if num in w and den in w:
                checks.append({
                    "check": "width ratio {} / {} at q={:.2f}".format(num, den, q),
                    "value": w[num] / w[den], "reference": np.nan})

    # (d) DQP mixing diagnostics. The Geyer and batch-means estimators disagree
    #     by an order of magnitude at the default batch size, so record the
    #     autocorrelations and the half-chain means that explain why.
    for q in LEVELS:
        s = draw_cols[("DQP", q)]
        n = s.size
        g = acov(s)
        rho = g / g[0]
        for lag in (1, 50, 200, 1000):
            if lag < n:
                checks.append({"check": "DQP autocorrelation at lag {} , q={:.2f}".format(lag, q),
                               "value": float(rho[lag]), "reference": np.nan})
        for b in (n // 20, n // 10):
            checks.append({"check": "DQP ESS batch means b={}, q={:.2f}".format(b, q),
                           "value": ess_batch_means(s, b=b), "reference": np.nan})
        checks.append({"check": "DQP first-half mean, q={:.2f}".format(q),
                       "value": float(s[: n // 2].mean()), "reference": np.nan})
        checks.append({"check": "DQP second-half mean, q={:.2f}".format(q),
                       "value": float(s[n // 2:].mean()), "reference": np.nan})

    checks = pd.DataFrame(checks)
    checks.to_csv(os.path.join(pd_, "cyclone_slope_checks.csv"),
                  index=False, float_format="%.6f")

    # -------------------------------------------------------------- LaTeX table
    tex = tex_table(summary)
    with open(os.path.join(pd_, "cyclone_slope_table.tex"), "w") as f:
        f.write(tex)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False,
                            float_format=lambda v: "{:.4f}".format(v)))
    print()
    print(checks.to_string(index=False,
                           float_format=lambda v: "{:.4f}".format(v)))
    print()
    print("wrote", out_csv)


def tex_table(summary):
    lines = [
        r"% Generated by src/cyclone/summarize_posteriors.py."
        r" Needs \usepackage{booktabs}.",
        r"\begin{tabular}{llrrcrrr}",
        r"\toprule",
        r"$q$ & Method & Mean & SD & $95\%$ interval & Width & Draws & ESS \\",
        r"\midrule",
    ]
    order = [m for m in ["QMP", "bagged QMP", "QMP-GP", "bagged QMP-GP", "DQP"]
             if m in set(summary["method"])]
    label = {"QMP": "QMP", "bagged QMP": "Bagged QMP", "QMP-GP": "QMP-GP",
             "bagged QMP-GP": "Bagged QMP-GP", "DQP": "DQP"}
    levels = sorted(summary["level_requested"].unique())
    for li, q in enumerate(levels):
        for mi, m in enumerate(order):
            r = summary[(summary.method == m) &
                        (summary.level_requested == q)].iloc[0]
            first = "{:.2f}".format(q) if mi == 0 else ""
            lines.append(
                "{} & {} & {:.3f} & {:.3f} & $[{:.3f},\\, {:.3f}]$ & {:.3f} & {:.0f} & {:.0f} \\\\".format(
                    first, label[m], r["post_mean"], r["post_sd"],
                    r["ci95_lo"], r["ci95_hi"], r["ci95_width"],
                    r["n_draws"], r["ess_geyer"]))
        if li < len(levels) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
