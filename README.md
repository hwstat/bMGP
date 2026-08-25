# bMGP-code

Code for the paper **Bagged Martingale Posteriors: Calibrated Uncertainty Quantification for Predictive Resampling**.

- `scripts/` — entry scripts; run everything from this directory
- `src/` — function libraries sourced by the scripts
- `data/AIDS.csv` — ACTG175 data for the empirical application

## Reproducing the paper

```bash
cd scripts
```

| Paper | Command |
|---|---|
| Figure 1 | `Rscript Run_Figure1_Ridge_Misspecification.R` |
| Table 1 | `Rscript Run_Table1_Mean_Coverage.R` |
| Figure 2 | `python Run_Figure2_Mean_Paths.py` |
| Tables 2–3 (appendix: add `--p=300`) | `Rscript Run_Tables23_HighDimension.R` |
| Quantile tables (q = 0.5, 0.75, 0.95; for the q = 0.75 rows at n = 1000, 2000 add `--n-grid 1000 2000`) | `python Run_Quantile_Tables.py` |
| PPC simulation table | `Rscript Run_PPC_Table.R` |
| Figures 3–4, ACTG175 application | `Rscript Run_Empirical_ACTG175.R`, then `Run_Empirical_Bagged.R`, then `Plot_Empirical_Contours.R` (Figure 3) and `Run_Empirical_PPC.R` + `Plot_PPC_Density.R` (Figure 4) |

Scripts are independent of one another except the empirical pipeline (last row), which runs in the order shown. Outputs (CSV tables, `.rds` results, and pgfplots figure sources) are written under `scripts/`.

## Dependencies

- **R**: `MASS`, `Matrix`; plus `cmdstanr` + `posterior` (empirical application) and `ggplot2`, `ggdist`, `scales` (Figure 1)
- **Python** ≥ 3.9: `numpy`; plus `scipy`, `pandas`, `jax` (quantile sweep)
