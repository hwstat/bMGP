"""
Shared data preparation for the cyclone timing runs.

The steps here reproduce, line for line, what
run_scripts/7.2_cyclone_qmp_small.py of the qmp repository
(https://github.com/edfong/qmp) does to the Elsner et al. (2008) lifetime
maximum wind speed data before any posterior is drawn:

  * read globalTCmax4.txt with sep=" " and the pandas default NA handling, so
    the Basin code "NA" (North Atlantic) is read as a missing value;
  * keep the rows whose Basin is missing, i.e. the North Atlantic basin;
  * response y = WmaxST, single covariate x = Year;
  * standardise y and x, then prepend an intercept column.

The data preparation is unchanged relative to the upstream script. The one
departure is in fit_hyperparameters: non-finite prequential scores are masked
before the argmax, as in src/quantile/qmp_base.py::fit_qmp_init. See that
function's docstring.

The data file is data/globalTCmax4.txt.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DU = 0.005
K_BAND = 0.5
N_PERM = 10
C_VALS = np.arange(0.05, 1.0, 0.05)

HyperFit = namedtuple(
    "HyperFit", "beta_init a c_opt k n_nonfinite preq_score")


def load_cyclone_small(data_path):
    """North Atlantic subset used by both Edwin's small cyclone script and the
    DQP MCMC script. Returns the standardised design matrix and response."""
    df = pd.read_csv(data_path, sep=" ")
    df = df.loc[df["Basin"].isnull(), :]
    n = np.shape(df)[0]

    x = np.array(df[["Year"]])
    y = np.array(df["WmaxST"])

    mean_y = np.mean(y)
    sd_y = np.std(y)
    y = (y - mean_y) / sd_y

    mean_x = np.mean(x, axis=0)
    sd_x = np.std(x, axis=0)
    x = (x - mean_x) / sd_x

    x = np.hstack((np.ones(n).reshape(-1, 1), x))
    d = np.shape(x)[1]

    return {
        "x": x,
        "y": y,
        "n": n,
        "d": d,
        "mean_y": mean_y,
        "sd_y": sd_y,
        "mean_x": mean_x,
        "sd_x": sd_x,
        "years": np.array(df["Year"]),
    }


def estimate_a(x, y):
    """Bandwidth a, exactly as in 7.2_cyclone_qmp_small.py."""
    n, d = np.shape(x)
    model = LinearRegression()
    model.fit(x, y)
    sigma = np.sqrt(np.mean((y - model.predict(x)) ** 2))
    a = np.sqrt(12) * sigma / (
        (np.linalg.det(np.dot(np.transpose(x), x) / n)) ** (1 / (d - 1))
    )
    return float(a)


def fit_hyperparameters(x, y, fit_beta_perm, n_perm=N_PERM, c_vals=C_VALS,
                        k=K_BAND, seed=100):
    """The c grid search from the upstream script, followed by the refit at
    c_opt.

    Non-finite prequential scores are masked out before the argmax, the way
    src/quantile/qmp_base.py::fit_qmp_init does it. np.argmax returns the index
    of the first NaN if one is present, so an unmasked search can select a c
    whose score did not exist rather than the best one. The upstream script
    does not mask, but on this data set no score is non-finite, so the mask
    changes nothing here; n_nonfinite records how many there were, so a run
    that did hit one says so in its JSON instead of failing silently.

    Returns a HyperFit named tuple.
    """
    a = estimate_a(x, y)
    preq_score = np.zeros(len(c_vals))
    for i, c in enumerate(c_vals):
        _, preq_score_vec = fit_beta_perm(
            np.array([a, c, k]), y, x, n_perm=n_perm, seed=seed
        )
        preq_score[i] = np.mean(preq_score_vec)

    finite_mask = np.isfinite(preq_score)
    n_nonfinite = int(np.sum(~finite_mask))
    if not np.any(finite_mask):
        raise RuntimeError(
            "All preq_score values are non-finite; cannot choose c.")
    idx_best = int(np.where(finite_mask)[0][np.argmax(preq_score[finite_mask])])
    c_opt = float(c_vals[idx_best])

    beta_init, _ = fit_beta_perm(
        np.array([a, c_opt, k]), y, x, n_perm=n_perm, seed=seed
    )
    beta_init = beta_init.block_until_ready()
    return HyperFit(beta_init=beta_init, a=a, c_opt=c_opt, k=float(k),
                    n_nonfinite=n_nonfinite, preq_score=preq_score)
