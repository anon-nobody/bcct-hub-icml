"""Mixed-effects, block bootstrap, and leave-one-out analysis of BCCT atlas.

Produces data/experiments/mixed_effects_analysis.json that supports the paper's
dependence-aware statistical claims.

Usage:
    python scripts/mixed_effects_analysis.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ATLAS_PATH = Path("data/atlas.json")
OUTPUT_PATH = Path("data/experiments/mixed_effects_analysis.json")


def load_atlas() -> pd.DataFrame:
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)
    models = atlas["models"]
    rows = []
    for key, v in atlas["pairwise"].items():
        ma, mb = key.split(" <-> ")
        rows.append(
            {
                "model_a": ma,
                "model_b": mb,
                "pair": key,
                "tau": v["tau"],
                "delta": v["delta"],
                "s_local": v["s_local"],
                "s_global": v["s_global"],
                "same_obj": int(
                    models[ma]["config"]["objective"] == models[mb]["config"]["objective"]
                ),
                "same_fam": int(
                    models[ma]["config"]["family"] == models[mb]["config"]["family"]
                ),
                "family_a": models[ma]["config"]["family"],
                "family_b": models[mb]["config"]["family"],
            }
        )
    return pd.DataFrame(rows)


def mixed_effects_model(df: pd.DataFrame) -> dict:
    """Fit mixed-effects model: tau ~ delta + same_obj + same_fam + (1|model_node).

    To handle the undirected pair structure (each pair involves two models),
    we duplicate each observation twice with model_node set to model_a and
    model_b respectively, ensuring all 20 models contribute to the random
    effect variance estimate.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    # Duplicate rows so both model_a and model_b serve as grouping variable
    df_a = df.assign(model_node=df["model_a"])
    df_b = df.assign(model_node=df["model_b"])
    df_dup = pd.concat([df_a, df_b], ignore_index=True)

    # Full model — try lbfgs first, fall back to powell if singular Hessian
    md = smf.mixedlm(
        "tau ~ delta + same_obj + same_fam", df_dup, groups=df_dup["model_node"]
    )
    try:
        mdf = md.fit(reml=False, method="lbfgs")
    except np.linalg.LinAlgError:
        mdf = md.fit(reml=False, method="powell")

    # Reduced model without bitrate (delta) for LRT
    md0 = smf.mixedlm("tau ~ same_obj + same_fam", df_dup, groups=df_dup["model_node"])
    try:
        mdf0 = md0.fit(reml=False, method="powell")
    except np.linalg.LinAlgError:
        mdf0 = md0.fit(reml=False, method="powell")

    lrt = -2 * (mdf0.llf - mdf.llf)
    # LRT can be -inf when optimizer hits boundary; fall back to Wald chi² test
    if np.isfinite(lrt) and lrt > 0:
        lrt_p = float(chi2.sf(lrt, df=1))
        lrt_method = "likelihood_ratio"
    else:
        # Wald chi² = z² for delta coefficient
        z_delta = float(mdf.tvalues["delta"])
        lrt = z_delta ** 2
        lrt_p = float(chi2.sf(lrt, df=1))
        lrt_method = "wald_chi2_fallback"

    def _coef(name):
        return {
            "coef": round(float(mdf.params[name]), 4),
            "std_err": round(float(mdf.bse[name]), 4),
            "z": round(float(mdf.tvalues[name]), 3),
            "p_value": float(f"{mdf.pvalues[name]:.2e}"),
            "ci_95": [
                round(float(mdf.conf_int().loc[name, 0]), 4),
                round(float(mdf.conf_int().loc[name, 1]), 4),
            ],
        }

    return {
        "formula": "tau ~ delta + same_obj + same_fam + (1|model_node)",
        "n_obs": len(df),
        "n_obs_duplicated": len(df_dup),
        "n_groups": df_dup["model_node"].nunique(),
        "aic": round(float(mdf.aic), 2),
        "bic": round(float(mdf.bic), 2),
        "log_likelihood": round(float(mdf.llf), 2),
        "fixed_effects": {
            "intercept": _coef("Intercept"),
            "delta": _coef("delta"),
            "same_obj": _coef("same_obj"),
            "same_fam": _coef("same_fam"),
        },
        "random_effect_variance": round(float(mdf.cov_re.iloc[0, 0]), 6),
        "lrt_for_delta": {
            "chi2": round(float(lrt), 3),
            "df": 1,
            "p_value": lrt_p,
            "method": lrt_method,
            "interpretation": "bitrate adds significant variance beyond family/objective"
            if lrt_p < 0.05
            else "bitrate does not add significant variance",
        },
    }


def block_bootstrap(
    df: pd.DataFrame, n_boot: int = 2000, subset: str | None = None, seed: int = 42
) -> dict:
    """Model-level block bootstrap of Spearman(delta, tau).

    Resamples models with replacement; includes all pairs where both models
    are in the resampled set (preserves pairwise dependency structure).
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    if subset == "same_obj":
        df_work = df[df["same_obj"] == 1]
    elif subset == "same_fam":
        df_work = df[df["same_fam"] == 1]
    else:
        df_work = df

    all_models = sorted(set(df_work["model_a"]).union(df_work["model_b"]))
    rhos = []
    for _ in range(n_boot):
        sampled = set(rng.choice(all_models, size=len(all_models), replace=True).tolist())
        boot = df_work[df_work["model_a"].isin(sampled) & df_work["model_b"].isin(sampled)]
        if len(boot) >= 10:
            rho, _ = spearmanr(boot["delta"], boot["tau"])
            if not np.isnan(rho):
                rhos.append(float(rho))

    rhos_arr = np.array(rhos)
    return {
        "subset": subset or "all_pairs",
        "n_pairs": len(df_work),
        "n_bootstraps_valid": len(rhos),
        "n_bootstraps_requested": n_boot,
        "median_rho": round(float(np.median(rhos_arr)), 4),
        "mean_rho": round(float(np.mean(rhos_arr)), 4),
        "ci_95": [
            round(float(np.percentile(rhos_arr, 2.5)), 4),
            round(float(np.percentile(rhos_arr, 97.5)), 4),
        ],
    }


def leave_one_out(df: pd.DataFrame) -> dict:
    """Leave-one-model-out: drop each model, recompute Spearman(delta, tau)."""
    from scipy.stats import spearmanr

    all_models = sorted(set(df["model_a"]).union(df["model_b"]))
    results = []
    for m in all_models:
        sub = df[(df["model_a"] != m) & (df["model_b"] != m)]
        rho, p = spearmanr(sub["delta"], sub["tau"])
        results.append(
            {
                "held_out": m,
                "n_pairs": len(sub),
                "rho": round(float(rho), 4),
                "p_value": float(f"{p:.2e}"),
            }
        )
    rhos = [r["rho"] for r in results]
    return {
        "per_model": results,
        "n_models": len(results),
        "rho_range": [round(float(min(rhos)), 4), round(float(max(rhos)), 4)],
        "rho_median": round(float(np.median(rhos)), 4),
    }


def main() -> None:
    df = load_atlas()
    logger.info(f"Loaded {len(df)} pairs, {df['model_a'].nunique()} models")

    logger.info("\n=== Mixed-Effects Model ===")
    me_result = mixed_effects_model(df)
    for name, fe in me_result["fixed_effects"].items():
        logger.info(
            f"  {name:12s}: beta={fe['coef']:+.4f} (p={fe['p_value']}, CI={fe['ci_95']})"
        )
    logger.info(f"  LRT for delta: chi2={me_result['lrt_for_delta']['chi2']}, "
                f"p={me_result['lrt_for_delta']['p_value']:.2e}")

    logger.info("\n=== Block Bootstrap (all pairs) ===")
    bb_all = block_bootstrap(df, n_boot=2000)
    logger.info(f"  median rho={bb_all['median_rho']}, CI={bb_all['ci_95']}")

    logger.info("\n=== Block Bootstrap (same-obj only) ===")
    bb_same_obj = block_bootstrap(df, n_boot=2000, subset="same_obj")
    logger.info(f"  median rho={bb_same_obj['median_rho']}, CI={bb_same_obj['ci_95']}")

    logger.info("\n=== Leave-One-Model-Out ===")
    loo = leave_one_out(df)
    logger.info(f"  rho range: {loo['rho_range']}, median: {loo['rho_median']}")

    output = {
        "description": "Mixed-effects, block bootstrap, and LOO analysis of BCCT vision atlas",
        "source": str(ATLAS_PATH),
        "n_pairs": len(df),
        "n_models": int(len(set(df["model_a"]).union(set(df["model_b"])))),
        "mixed_effects": me_result,
        "block_bootstrap_all_pairs": bb_all,
        "block_bootstrap_same_obj": bb_same_obj,
        "leave_one_out": loo,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
