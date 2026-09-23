#!/usr/bin/env python3
# Figure 2: bMGP predictive paths for the mean functional and the terminal
# posterior, under the Table 1 design (Section 3.2 of the paper).
# Usage: python Run_Figure2_Mean_Paths.py  (the engine holds the variance at var(x)/2;
# --update-variance updates it along the path instead)

import argparse
from pathlib import Path

import numpy as np

FIXED_VARIANCE = False

SCENARIOS = ("well", "miss")
SCENARIO_TITLES = {
    "well": r"DGP: $N(0,1)$",
    "miss": r"DGP: $\mathrm{Ga}(2,2)$",
}

TURBO = ("#30123B", "#4454C4", "#4490FE", "#1FC8DE", "#29EFA2", "#7DFF56",
         "#C1F334", "#F1CA3A", "#FE922A", "#EA4F0D", "#BE2102", "#7A0403")

OPTS = dict(
    ribbon_levels=((0.025, 0.975), (0.10, 0.90), (0.25, 0.75)),
    ribbon_greys=("#E4E4E4", "#D2D2D2", "#BABABA"),
    path_opacity=0.34,
    path_width_pt=0.30,
    median_width_pt=0.9,
    density_width_pt=1.1,
    frame_colour="#4D4D4D",
    grid_colour="#E8E8E8",
    panel_w_cm=6.44,   # -> ~15.6 cm total, the paper's common figure width
    panel_h_cm=4.05,
    hsep_cm=0.75,   # the right column carries no y descriptions
    vsep_cm=0.55,   # row 1 carries no x descriptions
    prefix="bmp",
)

def sample_dgp(scenario, n, rng):
    if scenario == "well":
        return rng.normal(loc=0.0, scale=1.0, size=int(n))
    if scenario == "miss":
        return rng.gamma(shape=2.0, scale=0.5, size=int(n))
    raise ValueError(f"unknown scenario: {scenario}")

def population_mean(scenario):
    if scenario == "well":
        return 0.0
    if scenario == "miss":
        return 1.0
    raise ValueError(f"unknown scenario: {scenario}")

def bmgp_mean_paths_for_bootstrap(x_boot, N, S_per_boot, rng,
                                  var_bias_multiplier=-0.5, var_floor=1e-10):
    """bMGP paths for the mean functional under the Gaussian recursion.

    By default (FIXED_VARIANCE is set to True in main()) the predictive variance
    is held at var_scale * var(x_boot) along the whole path, where
    var_scale = 1 + var_bias_multiplier equals 1/2 at the default multiplier.
    This is the fixed-variance engine of the paper with varkappa = 1/2, and it
    matches the `var_mult * var_hat` used by the Table 1 script. With
    --update-variance the variance state sigma is instead updated along the
    path by the recursion of Fong and Yiu (2026), and var_scale multiplies the
    current state.
    """
    x_boot = np.asarray(x_boot, dtype=np.float64).reshape(-1)
    n0 = x_boot.size
    S, N = int(S_per_boot), int(N)

    mu = np.full(S, float(np.mean(x_boot)), dtype=np.float64)
    sigma0 = float(np.var(x_boot, ddof=0))
    sigma = np.full(S, max(sigma0, var_floor), dtype=np.float64)
    var_scale = 1.0 + float(var_bias_multiplier)
    paths = np.empty((S, N + 1), dtype=np.float64)

    for m in range(N + 1):
        paths[:, m] = mu
        if m == N:
            break
        t = float(n0 + m)
        pred_var = np.maximum(var_scale * sigma, var_floor)
        z_new = mu + rng.standard_normal(S) * np.sqrt(pred_var)
        delta = z_new - mu
        mu = mu + delta / (t + 1.0)
        if not FIXED_VARIANCE:
            sigma = (t / (t + 1.0)) * sigma + (t / ((t + 1.0) ** 2.0)) * (delta ** 2)
            sigma = np.maximum(sigma, var_floor)
    return paths

def compute_groups_for_scenario(scenario, args):
    obs_rng = np.random.default_rng(
        int(args.seed) + (0 if scenario == "well" else 100_000))
    boot_rng = np.random.default_rng(
        int(args.seed) + (11_000 if scenario == "well" else 111_000))
    x_obs = sample_dgp(scenario, n=args.n, rng=obs_rng)

    m_boot = int(x_obs.size if args.M_boot is None else args.M_boot)
    groups = []
    for _ in range(int(args.n_boot)):
        idx = boot_rng.integers(0, x_obs.size, size=m_boot)
        groups.append(bmgp_mean_paths_for_bootstrap(
            x_boot=x_obs[idx], N=args.N, S_per_boot=args.S_per_boot,
            rng=boot_rng, var_bias_multiplier=args.var_bias_multiplier,
            var_floor=args.var_floor))
    return groups

def kde_density(values, grid, bandwidth_scale=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid)
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if sd <= 0.0:
        out = np.zeros_like(grid)
        out[int(np.argmin(np.abs(grid - values[0])))] = 1.0
        return out
    bw = max(bandwidth_scale * 1.06 * sd * values.size ** (-0.2), 1e-8)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bw * np.sqrt(2.0 * np.pi))

def display_steps(N, k):
    """Geometric step grid: every early step, sparse late ones."""
    if k >= N + 1:
        return np.arange(N + 1)
    geo = np.geomspace(1.0, float(N), num=int(k) - 1)
    steps = np.unique(np.concatenate(([0.0], np.round(geo))).astype(int))
    return steps[steps <= N]

def build_panel(scenario, groups, args, opts):
    steps = display_steps(args.N, args.steps)
    stacked = np.vstack(groups)                       # every path, for summaries

    target = population_mean(scenario)
    ribbons = [(np.quantile(stacked[:, steps], lo, axis=0),
                np.quantile(stacked[:, steps], hi, axis=0))
               for lo, hi in opts["ribbon_levels"]]
    median = np.quantile(stacked[:, steps], 0.50, axis=0)

    per_boot = args.paths_per_boot
    shown = []
    for b, g in enumerate(groups):
        take = (np.linspace(0, g.shape[0] - 1, int(per_boot), dtype=int)
                if per_boot is not None and g.shape[0] > per_boot
                else np.arange(g.shape[0]))
        for s in take:
            shown.append((b, g[s, steps]))

    vals = np.concatenate([stacked.reshape(-1), [target]])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [args.ylim_lower_quantile, args.ylim_upper_quantile])
    lo, hi = min(float(lo), target), max(float(hi), target)
    pad = args.ylim_pad_frac * max(hi - lo, 1e-8)
    path_ylim = (lo - pad, hi + pad)

    terminal = stacked[:, -1]
    dlo, dhi = min(float(terminal.min()), target), max(float(terminal.max()), target)
    dpad = args.density_ylim_pad_frac * max(dhi - dlo, 1e-8)
    density_ylim = (dlo - dpad, dhi + dpad)

    ylim = (min(path_ylim[0], density_ylim[0]), max(path_ylim[1], density_ylim[1]))

    grid = np.linspace(ylim[0], ylim[1], args.density_grid_size)
    density = kde_density(terminal, grid, args.density_bandwidth_scale)

    return dict(scenario=scenario, steps=steps, ribbons=ribbons, median=median,
                shown=shown, target=target, ylim=ylim, grid=grid,
                density=density, n_boot=len(groups))

def ramp_colours(anchors, k):
    rgb = np.array([[int(a[i:i + 2], 16) for i in (1, 3, 5)] for a in anchors],
                   dtype=np.float64)
    t = np.linspace(0.0, 1.0, len(anchors))
    u = np.linspace(0.0, 1.0, int(k)) if k > 1 else np.array([0.5])
    out = np.column_stack([np.interp(u, t, rgb[:, c]) for c in range(3)])
    return ["%02X%02X%02X" % tuple(int(round(v)) for v in row) for row in out]

def fmt(x, d=5):
    return f"{x:.{d}f}"

def coord_block(xs, ys, indent="      ", per_line=6):
    pairs = [f"({fmt(x, 3)},{fmt(y)})" for x, y in zip(xs, ys)]
    return [indent + " ".join(pairs[i:i + per_line])
            for i in range(0, len(pairs), per_line)]

def write_tikz(panels, args, opts, path):
    px = opts["prefix"]
    cols = ramp_colours(TURBO, panels[0]["n_boot"])
    n_row = len(panels)
    matplotlib_labels = args.labels == "matplotlib"
    vsep = opts["vsep_cm"] + (1.35 if matplotlib_labels else 0.0)
    hsep = opts["hsep_cm"] + (0.60 if matplotlib_labels else 0.0)

    L = [
        "% =========================================================================",
        "%  bMGP predictive paths for the mean functional (the Table 1 design,",
        "%  fixed variance var(x)/2, varkappa = 1/2). Left: paths coloured by bootstrap sample, over",
        "%  pooled 2.5-97.5 / 10-90 / 25-75 quantile ribbons, with the pooled",
        "%  median in black. Right: the terminal posterior. Dashed line = the",
        "%  population mean.",
        "%  Generated by Run_Figure2_Mean_Paths.py -- do not edit by hand.",
        "%",
        "%  Preamble requirements (main.tex):",
        "%      \\usepackage{pgfplots}",
        "%      \\pgfplotsset{compat=1.18}",
        "%      \\usepgfplotslibrary{groupplots}",
        "% =========================================================================",
        "\\begin{tikzpicture}",
        "  % ---- colours --------------------------------------------------------",
    ]
    for i, c in enumerate(cols):
        L.append(f"  \\definecolor{{{px}B{i:02d}}}{{HTML}}{{{c}}}")
    for k, g in enumerate(opts["ribbon_greys"]):
        L.append(f"  \\definecolor{{{px}R{k}}}{{HTML}}{{{g.lstrip('#').upper()}}}")
    L.append(f"  \\definecolor{{{px}Frame}}{{HTML}}"
             f"{{{opts['frame_colour'].lstrip('#').upper()}}}")
    L.append(f"  \\definecolor{{{px}Grid}}{{HTML}}"
             f"{{{opts['grid_colour'].lstrip('#').upper()}}}")

    L += [
        "  % ---- styles ---------------------------------------------------------",
        "  \\pgfplotsset{",
        f"    {px}Panel/.style={{",
        f"      width={fmt(opts['panel_w_cm'], 2)}cm,"
        f" height={fmt(opts['panel_h_cm'], 2)}cm, scale only axis,",
        f"      axis line style={{draw={px}Frame, line width=0.45pt}},",
        f"      tick style={{draw={px}Frame, line width=0.45pt}},",
        f"      grid=major, grid style={{draw={px}Grid, line width=0.3pt}},",
        "      label style={font=\\small}, title style={font=\\small, yshift=-2pt},",
        "      scaled ticks=false,",
        "      tick label style={font=\\small, /pgf/number format/1000 sep={}},",
        "      enlargelimits=false, clip mode=individual,",
        "      every axis plot/.append style={line join=round},",
        "    },",
        f"    {px}Path/.style={{line width={fmt(opts['path_width_pt'], 2)}pt,"
        f" draw opacity={fmt(opts['path_opacity'], 2)}}},",
        f"    {px}Median/.style={{draw=black,"
        f" line width={fmt(opts['median_width_pt'], 2)}pt}},",
        f"    {px}Truth/.style={{draw=black, dash pattern=on 3pt off 3pt,"
        " line width=0.7pt},",
        f"    {px}Dens/.style={{draw=black,"
        f" line width={fmt(opts['density_width_pt'], 2)}pt}},",
        "  }",
        f"  \\tikzset{{{px}Tag/.style={{anchor=north west, font=\\scriptsize,",
        "    fill=white, fill opacity=0.92, text opacity=1, inner sep=1.5pt}}",
        "  % ---- panels ---------------------------------------------------------",
        "  \\begin{groupplot}[",
        f"    {px}Panel,",
        f"    group style={{group name={px}grp, group size=2 by {n_row},",
        f"      horizontal sep={fmt(hsep, 2)}cm,"
        f" vertical sep={fmt(vsep, 2)}cm,",
    ]
    if not matplotlib_labels:
        L += ["      x descriptions at=edge bottom,",
              "      y descriptions at=edge left},"]
    else:
        L += ["      },"]
    L += ["  ]"]

    # one density limit for the whole right column, which shares its x labels
    dmax = max(float(q["density"].max()) for q in panels) * 1.06
    for r, p in enumerate(panels):
        steps, ylo, yhi = p["steps"], p["ylim"][0], p["ylim"][1]
        last_row = r == n_row - 1

        head = [f"xmin=0, xmax={args.N}",
                f"ymin={fmt(ylo)}, ymax={fmt(yhi)}"]
        if last_row or matplotlib_labels:
            head.append("xlabel={Predictive step}")
        if matplotlib_labels:
            head.append("ylabel={Mean functional}")
            head.append(f"title={{{SCENARIO_TITLES[p['scenario']]}}}")
        elif r == 0:
            head.append("title={Predictive paths}")
        L += ["", f"  % ---------- {p['scenario']}: paths ----------",
              f"  \\nextgroupplot[{', '.join(head)}]"]

        for k, (qlo, qhi) in enumerate(p["ribbons"]):
            xs = np.concatenate([steps, steps[::-1]])
            ys = np.concatenate([qhi, qlo[::-1]])
            L.append(f"  \\addplot[draw=none, fill={px}R{k}] coordinates {{")
            L += coord_block(xs, ys)
            L.append("  };")
        for b, ys in p["shown"]:
            L.append(f"  \\addplot[{px}Path, draw={px}B{b:02d}] coordinates {{")
            L += coord_block(steps, ys)
            L.append("  };")
        L.append(f"  \\addplot[{px}Median] coordinates {{")
        L += coord_block(steps, p["median"])
        L.append("  };")
        L.append(f"  \\addplot[{px}Truth] coordinates "
                 f"{{(0,{fmt(p['target'])}) ({args.N},{fmt(p['target'])})}};")

        if not matplotlib_labels:
            L.append(f"  \\node[{px}Tag] at (rel axis cs:0.025,0.965) "
                     f"{{{SCENARIO_TITLES[p['scenario']]}}};")

        head = [f"xmin=0, xmax={fmt(dmax, 4)}",
                f"ymin={fmt(ylo)}, ymax={fmt(yhi)}"]
        if last_row or matplotlib_labels:
            head.append("xlabel={Density}")
        if matplotlib_labels:
            head.append("ylabel={Mean functional}")
            if r == 0:
                head.append("title={Posterior distribution}")
        elif r == 0:
            head.append("title={Posterior distribution}")
        L += ["", f"  % ---------- {p['scenario']}: posterior ----------",
              f"  \\nextgroupplot[{', '.join(head)}]"]
        L.append(f"  \\addplot[{px}Dens] coordinates {{")
        L += coord_block(p["density"], p["grid"])
        L.append("  };")
        L.append(f"  \\addplot[{px}Truth] coordinates "
                 f"{{(0,{fmt(p['target'])}) ({fmt(dmax, 4)},{fmt(p['target'])})}};")

    L.append("  \\end{groupplot}")

    if not matplotlib_labels:
        y_drop = (opts["panel_h_cm"] + vsep) * (n_row - 1) / 2.0
        L += [
            "  % ---- shared y label --------------------------------------------------",
            f"  \\node[anchor=south, rotate=90, font=\\small] at "
            f"([xshift=-11mm, yshift=-{fmt(y_drop, 3)}cm] {px}grp c1r1.west) "
            "{Mean functional};",
        ]
    L.append("\\end{tikzpicture}")

    Path(path).write_text("\n".join(L) + "\n")
    return path

def write_standalone(figure_path, path):
    Path(path).write_text("\n".join([
        "% Minimal wrapper: compile this to preview the figure on its own.",
        "\\documentclass[border=4pt]{standalone}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{amsmath}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\usepgfplotslibrary{groupplots}",
        "\\begin{document}",
        f"\\input{{{Path(figure_path).stem}}}",
        "\\end{document}",
    ]) + "\n")
    return path

def parse_args():
    p = argparse.ArgumentParser(
        description="bMGP mean-functional paths as a native pgfplots figure.")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--n-boot", dest="n_boot", type=int, default=50)
    p.add_argument("--S-per-boot", dest="S_per_boot", type=int, default=20)
    p.add_argument("--M-boot", dest="M_boot", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--var-bias-multiplier", dest="var_bias_multiplier",
                   type=float, default=-0.5)
    p.add_argument("--var-floor", dest="var_floor", type=float, default=1e-10)
    p.add_argument("--update-variance", dest="update_variance", action="store_true",
                   help="update the variance along the path instead of holding it at var(x)/2")
    p.add_argument("--paths-per-boot", dest="paths_per_boot", type=int, default=3,
                   help="paths drawn per bootstrap sample (summaries always "
                        "use all of them)")
    p.add_argument("--steps", type=int, default=130,
                   help="vertices kept per curve, geometric spacing")
    p.add_argument("--ylim-lower-quantile", dest="ylim_lower_quantile",
                   type=float, default=0.0)
    p.add_argument("--ylim-upper-quantile", dest="ylim_upper_quantile",
                   type=float, default=1.0)
    p.add_argument("--ylim-pad-frac", dest="ylim_pad_frac", type=float, default=0.06)
    p.add_argument("--density-ylim-pad-frac", dest="density_ylim_pad_frac",
                   type=float, default=0.15)
    p.add_argument("--density-grid-size", dest="density_grid_size",
                   type=int, default=300)
    p.add_argument("--density-bandwidth-scale", dest="density_bandwidth_scale",
                   type=float, default=1.0)
    p.add_argument("--labels", choices=("headers", "matplotlib"), default="headers",
                   help="'headers': column headers + rotated row labels; "
                        "'matplotlib': the original title scheme")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--cache", type=str, default=None)
    return p.parse_args()

def main():
    global FIXED_VARIANCE
    args = parse_args()
    opts = dict(OPTS)
    here = Path(__file__).resolve().parent
    FIXED_VARIANCE = not args.update_variance
    out_dir = Path(args.out) if args.out else here / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        print(f"Reusing cached paths: {cache}")
        blob = np.load(cache)
        groups_by_scenario = {
            sc: [blob[f"{sc}_{b}"] for b in range(int(blob["n_boot"]))]
            for sc in SCENARIOS}
    else:
        groups_by_scenario = {}
        for sc in SCENARIOS:
            print(f"Simulating {sc}: n={args.n}, N={args.N}, "
                  f"n_boot={args.n_boot}, S={args.S_per_boot}, "
                  f"c_n={args.var_bias_multiplier}*sigma_n", flush=True)
            groups_by_scenario[sc] = compute_groups_for_scenario(sc, args)
        if cache is not None:
            blob = {"n_boot": np.array(args.n_boot)}
            for sc in SCENARIOS:
                for b, g in enumerate(groups_by_scenario[sc]):
                    blob[f"{sc}_{b}"] = g
            np.savez_compressed(cache, **blob)
            print(f"Cached paths: {cache}")

    panels = [build_panel(sc, groups_by_scenario[sc], args, opts)
              for sc in SCENARIOS]

    raw = sum(g.size for gs in groups_by_scenario.values() for g in gs)
    kept = sum(len(p["steps"]) * (len(p["shown"]) + 1 + 2 * len(p["ribbons"]))
               + 2 * len(p["grid"]) for p in panels)
    lines = sum(len(p["shown"]) for p in panels) // len(panels)
    print(f"simulated {raw:,} path values; drawing {lines} paths per panel "
          f"-> {kept:,} coordinates ({100.0 * kept / raw:.2f}%)")

    fig = write_tikz(panels, args, opts, out_dir / "bmgp_mean_paths_pgfplots.tex")
    wrap = write_standalone(fig, out_dir / "bmgp_mean_paths_standalone.tex")
    print(f"Wrote:\n  {fig}\n  {wrap}")

if __name__ == "__main__":
    main()
