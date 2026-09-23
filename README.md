# bMGP-code

Code for the paper **Bagged Martingale Posteriors: Calibrated Uncertainty Quantification for Predictive Resampling**.

- `scripts/` — entry scripts; run everything from this directory
- `src/` — function libraries sourced by the scripts
- `data/AIDS.csv` — ACTG175 data for the empirical application (Hammer et al. 1996), the 2139-record version distributed by the UCI Machine Learning Repository (dataset 890, <https://doi.org/10.24432/C5ZG8F>), with the outcome `cd420` and the continuous covariates already standardized and the treatment arms coded as the dummies `trt_1` (zidovudine plus didanosine), `trt_2` (zidovudine plus zalcitabine) and `trt_3` (didanosine), zidovudine alone being the reference
- `data/globalTCmax4.txt` — lifetime maximum wind speed records (Elsner, Kossin and Jagger 2008), distributed with the `qmp` repository, which names <https://myweb.fsu.edu/jelsner/temp/Data.html> as the source; `data/DQP_beta1.csv` is the DQP reference output shipped with the same repository, used only for the optional cross-check in `summarize_posteriors.py`

## Reproducing the paper

```bash
cd scripts
```

| Paper | Command |
|---|---|
| Figure 1 | `Rscript Run_Figure1_Ridge_Misspecification.R --simulate=true` (without the flag the script only re-plots saved draws) |
| Table 1 | `Rscript Run_Table1_Mean_Coverage.R --ncores=8` (the default `--fixed_var=true` holds the predictive variance fixed, as in equation (8) of the paper; the random-number streams are assigned per worker, so the paper's numbers are reproduced with 8 workers) |
| Figure 2 | `python Run_Figure2_Mean_Paths.py` |
| Tables 2 and 3, MGP and bMGP rows; Tables 7 and 8 with `--p=300` (the inclusion probability stays at 0.25 and the Student-t horizon becomes p + 100 = 400, as in the paper) | `Rscript Run_Tables23_HighDimension.R` |
| Tables 2 and 7, Bayes and BayesBag rows (both dimensions by default; `--p=200` for Table 2 alone, `--p=300` for Table 7 alone) | `Rscript Run_Table2_Bayes_BayesBag.R`, then `Rscript Make_Table2_Bayes_BayesBag.R` |
| Footnote of Table 2 (50 draws per bagged data set) | `Rscript Run_Table2_Bayes_BayesBag.R --draws_per_boot=50 --out=output/table2_draws50` |
| Ridge timings quoted in Appendix A.3.1 | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 Rscript Run_Table2_Bayes_BayesBag.R --mode=timing --methods=bayes,bayesbag,mgp,bmgp --workers=1 --parallel=false` (the thread variables must be set before R starts) |
| Tables 4 and 5 (q = 0.5, 0.75 at n = 100, 200, 500; for the q = 0.75 rows at n = 1000, 2000 add `--n-grid 1000 2000`) | `python Run_Quantile_Tables.py` |
| Table 10 (q = 0.95 against the population quantile, n = 100, 200, 500; several hours) | `python Run_Quantile_Tail_Table.py` |
| Table 9 and the spike-and-slab timings quoted in Appendix A.3.1 (four scenarios `linear_sas_{well,miss}`, `studentt_sas_{well,miss}`; `--mode=mgp` and `--mode=gibbs` for all four and `--mode=bags` for the three whose bagged timings the paper reports, `linear_sas_well`, `linear_sas_miss` and `studentt_sas_well`; about 2 hours in total) | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 Rscript Run_Table8_Timing_SpikeSlab.R --model=<scenario> --mode=<mode>`, then `Rscript Make_Table8_Timing.R` (Table 9 is the linear panel; the Student-t panel supplies the numbers quoted in the text) |
| Table 11 | `bash Run_Table10_Cyclone_Timing.sh` (two repetitions by default, `REPS=2`; the Python interpreter is taken from `PYTHON`, default `python3`; about 35 minutes, most of it the two DQP chains) |
| Tables 6 and 12 | `bash Run_Table11_Cyclone_Posterior.sh` (same `PYTHON` convention) |
| Figure 3, ACTG175 application | `Rscript Run_Empirical_ACTG175.R`, then `Rscript Run_Empirical_Bagged.R`, then `Rscript Plot_Empirical_Contours.R` |

The script names `Run_Table8_*`, `Run_Table10_*` and `Run_Table11_*` predate the final table numbering; the rows above give the current numbers. The quantile sweep also evaluates a Gaussian predictive engine and the levels q = 0.25 and 0.95; the paper does not report these.

Scripts are independent of one another except the empirical pipeline (last row), which runs in the order shown. Notation: the paper's $B$ (bagged data sets), $M_B$ (paths per bagged data set) and $N-n$ (predictive horizon) are `n_boot`, `paths_per_boot` and `T` in the high-dimension code and `B_boot`, `M_B` and `T` in the cyclone code; the symbol `B` in `well_miss_comparison.R` is the total number of predictive paths, $B\times M_B=1000$. `Run_Table1_Mean_Coverage.R` calls them `B_boot`, `S_each` and `S_ord` (the last is the number of paths of the unbagged posterior), `Run_Figure2_Mean_Paths.py` uses `--n-boot`, `--S-per-boot` and `--N` (the horizon), and `Run_Quantile_Tables.py` uses `--B-boot`, `--S-per-boot` and `--B` (unbagged paths). Outputs (CSV tables, `.rds` results, and pgfplots figure sources) are written under `scripts/`.

## Dependencies

- **R**: `MASS`, `Matrix`; plus `cmdstanr` + `posterior` (empirical application), `ggplot2` (Figure 1 and the empirical application), `ggdist`, `scales` (Figure 1), and `Rcpp`, `RcppArmadillo`, `jsonlite` (the DQP chain of Tables 11–12, whose C++ sources are compiled at run time)
- **Python** ≥ 3.9: `numpy`; plus `scipy`, `pandas`, `jax` (quantile sweep); the cyclone scripts of Tables 11–12 also need `scikit-learn` (for `LinearRegression` only) and the vendored `qmp` package in `src/quantile/qmp/`, and were run with the versions pinned in that package's upstream `setup.py`, `jax` 0.4.30 with the CPU `jaxlib` 0.4.30, `numpy` 1.26.4 and `scipy` 1.12.0

`src/quantile/qmp/` and `src/cyclone/dqp/` are vendored third-party code, under the MIT licences in `src/quantile/qmp/LICENSE` (Edwin Fong) and `src/cyclone/dqp/LICENSE` (Hyoin An).
