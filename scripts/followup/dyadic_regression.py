"""Dyadic-robust inference for the atlas regression tau ~ delta + same_obj + same_fam.

scripts/mixed_effects_analysis.py gives each pair a random intercept for both models by
duplicating every pair row (190 -> 380 rows). The duplicated rows share one response, so that
fit treats 190 observations as 380 and understates standard errors. This script refits on the
190 pairs only and reports three dependence-aware inferences:
  1. OLS with two-way cluster-robust standard errors (clusters: model_a and model_b),
     the Cameron-Gelbach-Miller multiway estimator commonly used for dyadic data;
  2. dyadic-robust (Fafchamps-Gubert) standard errors: residual products are kept for every
     pair of dyads that share a model;
  3. a model-level bootstrap: resample the 20 models with replacement, keep every pair among
     the drawn models (with multiplicity), refit OLS, 2,000 replicates, percentile intervals.
Output: data/experiments/dyadic_regression.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import mixed_effects_analysis as mea  # noqa: E402

COLS = ["delta", "same_obj", "same_fam"]
OUT = ROOT / "data/experiments/dyadic_regression.json"


def fafchamps_gubert(X: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dyadic-robust covariance: sum x_i e_i e_j x_j' over dyad pairs (i,j) sharing a node."""
    n = len(e)
    share = ((a[:, None] == a[None, :]) | (a[:, None] == b[None, :]) |
             (b[:, None] == a[None, :]) | (b[:, None] == b[None, :])).astype(float)
    xe = X * e[:, None]
    meat = xe.T @ share @ xe
    bread = np.linalg.inv(X.T @ X)
    k = X.shape[1]
    return bread @ meat @ bread * n / (n - k)


def main() -> None:
    df = mea.load_atlas()
    X = sm.add_constant(df[COLS].astype(float)).to_numpy()
    y = df["tau"].to_numpy()
    names = ["const", *COLS]
    ols = sm.OLS(y, X).fit()
    groups = np.column_stack([df["model_a"].astype("category").cat.codes,
                              df["model_b"].astype("category").cat.codes])
    cl2 = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    a = df["model_a"].to_numpy()
    b = df["model_b"].to_numpy()
    V = fafchamps_gubert(X, ols.resid, a, b)
    se_fg = np.sqrt(np.diag(V))
    p_fg = 2 * stats.norm.sf(np.abs(ols.params / se_fg))

    rng = np.random.default_rng(0)
    models = sorted(set(a) | set(b))
    boot = []
    for _ in range(2000):
        draw = rng.choice(models, len(models), replace=True)
        cnt = {m: int((draw == m).sum()) for m in set(draw)}
        idx = [i for i in range(len(df)) if a[i] in cnt and b[i] in cnt for _ in range(cnt[a[i]] * cnt[b[i]])]
        sub = df.iloc[idx]
        Xb = sm.add_constant(sub[COLS].astype(float), has_constant="add").to_numpy()
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        boot.append(np.linalg.lstsq(Xb, sub["tau"].to_numpy(), rcond=None)[0])
    boot = np.array(boot)

    res = {"n_pairs": len(df), "n_models": len(models), "coef": {}, "r2_ols": round(float(ols.rsquared), 4),
           "n_boot_valid": len(boot)}
    for j, n in enumerate(names):
        res["coef"][n] = {
            "beta": round(float(ols.params[j]), 4),
            "naive_ols": {"se": round(float(ols.bse[j]), 4), "p": float(ols.pvalues[j])},
            "twoway_cluster": {"se": round(float(cl2.bse[j]), 4), "p": float(cl2.pvalues[j])},
            "dyadic_fg": {"se": round(float(se_fg[j]), 4), "p": float(p_fg[j])},
            "model_bootstrap": {"ci95": [round(float(np.percentile(boot[:, j], 2.5)), 4),
                                         round(float(np.percentile(boot[:, j], 97.5)), 4)],
                                "share_same_sign": round(float(np.mean(np.sign(boot[:, j]) == np.sign(ols.params[j]))), 4)},
        }
        c = res["coef"][n]
        print(f"{n:9s} beta={c['beta']:+.4f}  naive p={c['naive_ols']['p']:.2g}  "
              f"2way p={c['twoway_cluster']['p']:.2g}  FG p={c['dyadic_fg']['p']:.2g}  "
              f"boot CI={c['model_bootstrap']['ci95']}")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"R2={res['r2_ols']}  saved -> {OUT}")


if __name__ == "__main__":
    main()
