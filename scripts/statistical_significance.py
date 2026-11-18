"""Compute bootstrap CIs, effect sizes, and corrected p-values for BCCT claims.

Usage:
    python scripts/statistical_significance.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

logger = logging.getLogger(__name__)


def bootstrap_ci(data: np.ndarray, stat_fn=np.mean, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        estimates.append(stat_fn(sample))
    estimates = np.array(estimates)
    alpha = (1 - ci) / 2
    return float(stat_fn(data)), float(np.percentile(estimates, 100 * alpha)), float(np.percentile(estimates, 100 * (1 - alpha)))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.std(ddof=1)**2 + (n2 - 1) * group2.std(ddof=1)**2) / (n1 + n2 - 2))
    return float((group1.mean() - group2.mean()) / max(pooled_std, 1e-10))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas = json.loads(Path("data/atlas.json").read_text())

    # Classify pairs
    objectives = {
        "contrastive": ["clip_vitb16", "clip_vitb32", "clip_vitl14"],
        "self_supervised": ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
                            "vit_base_patch16_224.mae", "vit_large_patch16_224.mae"],
        "supervised": ["resnet50", "resnet101", "vit_base_patch16_224", "vit_large_patch16_224",
                        "efficientnet_b0", "convnext_base", "convnext_large",
                        "swin_base_patch4_window7_224", "mixer_b16_224"],
        "masked_image": ["beit_base_patch16_224", "beit_large_patch16_224"],
    }
    families = {
        "clip": ["clip_vitb16", "clip_vitb32", "clip_vitl14"],
        "dinov2": ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        "resnet": ["resnet50", "resnet101"],
        "vit_sup": ["vit_base_patch16_224", "vit_large_patch16_224"],
        "vit_mae": ["vit_base_patch16_224.mae", "vit_large_patch16_224.mae"],
        "beit": ["beit_base_patch16_224", "beit_large_patch16_224"],
        "convnext": ["convnext_base", "convnext_large"],
    }

    def get_obj(name):
        for o, models in objectives.items():
            if name in models:
                return o
        return "unknown"

    def get_fam(name):
        for f, models in families.items():
            if name in models:
                return f
        return "singleton"

    records = []
    for pair_key, m in atlas["pairwise"].items():
        names = pair_key.split(" <-> ")
        oa, ob = get_obj(names[0]), get_obj(names[1])
        fa, fb = get_fam(names[0]), get_fam(names[1])
        records.append({
            "tau": m["tau"], "s_local": m["s_local"], "delta": m["delta"],
            "same_obj": oa == ob, "same_fam": fa == fb,
        })

    taus = np.array([r["tau"] for r in records])
    deltas = np.array([r["delta"] for r in records])
    s_locals = np.array([r["s_local"] for r in records])

    results = {}

    # 1. Bootstrap CIs for global metrics
    logger.info("=== Bootstrap CIs (95%%) ===")
    for name, data in [("tau", taus), ("s_local", s_locals), ("delta", deltas)]:
        mean_est, ci_lo, ci_hi = bootstrap_ci(data, np.mean)
        med_est, med_lo, med_hi = bootstrap_ci(data, np.median)
        logger.info(f"{name}: mean={mean_est:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], "
                     f"median={med_est:.4f} [{med_lo:.4f}, {med_hi:.4f}]")
        results[f"{name}_mean_ci"] = [round(mean_est, 4), round(ci_lo, 4), round(ci_hi, 4)]
        results[f"{name}_median_ci"] = [round(med_est, 4), round(med_lo, 4), round(med_hi, 4)]

    # 2. Same-obj vs diff-obj: effect size + CI
    same_obj_tau = np.array([r["tau"] for r in records if r["same_obj"]])
    diff_obj_tau = np.array([r["tau"] for r in records if not r["same_obj"]])
    same_fam_tau = np.array([r["tau"] for r in records if r["same_fam"]])
    diff_fam_tau = np.array([r["tau"] for r in records if not r["same_fam"]])

    logger.info("\n=== Objective Effect ===")
    d_obj = cohens_d(same_obj_tau, diff_obj_tau)
    u_obj, p_obj = mannwhitneyu(same_obj_tau, diff_obj_tau, alternative="greater")
    mean_same, ci_lo_s, ci_hi_s = bootstrap_ci(same_obj_tau)
    mean_diff, ci_lo_d, ci_hi_d = bootstrap_ci(diff_obj_tau)
    logger.info(f"Same obj: mean={mean_same:.4f} [{ci_lo_s:.4f}, {ci_hi_s:.4f}] (n={len(same_obj_tau)})")
    logger.info(f"Diff obj: mean={mean_diff:.4f} [{ci_lo_d:.4f}, {ci_hi_d:.4f}] (n={len(diff_obj_tau)})")
    logger.info(f"Cohen's d = {d_obj:.3f}, Mann-Whitney p = {p_obj:.2e}")
    results["objective_effect"] = {
        "cohens_d": round(d_obj, 3),
        "mann_whitney_p": float(f"{p_obj:.2e}"),
        "same_obj_mean_ci": [round(mean_same, 4), round(ci_lo_s, 4), round(ci_hi_s, 4)],
        "diff_obj_mean_ci": [round(mean_diff, 4), round(ci_lo_d, 4), round(ci_hi_d, 4)],
        "n_same": len(same_obj_tau),
        "n_diff": len(diff_obj_tau),
    }

    logger.info("\n=== Family Effect ===")
    d_fam = cohens_d(same_fam_tau, diff_fam_tau)
    u_fam, p_fam = mannwhitneyu(same_fam_tau, diff_fam_tau, alternative="greater")
    mean_sf, ci_lo_sf, ci_hi_sf = bootstrap_ci(same_fam_tau)
    mean_df, ci_lo_df, ci_hi_df = bootstrap_ci(diff_fam_tau)
    logger.info(f"Same fam: mean={mean_sf:.4f} [{ci_lo_sf:.4f}, {ci_hi_sf:.4f}] (n={len(same_fam_tau)})")
    logger.info(f"Diff fam: mean={mean_df:.4f} [{ci_lo_df:.4f}, {ci_hi_df:.4f}] (n={len(diff_fam_tau)})")
    logger.info(f"Cohen's d = {d_fam:.3f}, Mann-Whitney p = {p_fam:.2e}")
    results["family_effect"] = {
        "cohens_d": round(d_fam, 3),
        "mann_whitney_p": float(f"{p_fam:.2e}"),
        "same_fam_mean_ci": [round(mean_sf, 4), round(ci_lo_sf, 4), round(ci_hi_sf, 4)],
        "diff_fam_mean_ci": [round(mean_df, 4), round(ci_lo_df, 4), round(ci_hi_df, 4)],
    }

    # 3. Spearman correlation with bootstrap CI
    logger.info("\n=== Spearman(Δ, τ) with bootstrap CI ===")
    rho, p_spear = spearmanr(deltas, taus)
    rng = np.random.default_rng(42)
    rho_boots = []
    for _ in range(10000):
        idx = rng.choice(len(deltas), size=len(deltas), replace=True)
        r, _ = spearmanr(deltas[idx], taus[idx])
        rho_boots.append(r)
    rho_boots = np.array(rho_boots)
    rho_lo, rho_hi = np.percentile(rho_boots, [2.5, 97.5])
    logger.info(f"Spearman rho = {rho:.4f} [{rho_lo:.4f}, {rho_hi:.4f}], p = {p_spear:.2e}")
    results["spearman_delta_tau"] = {
        "rho": round(rho, 4), "ci_95": [round(rho_lo, 4), round(rho_hi, 4)],
        "p_value": float(f"{p_spear:.2e}"),
    }

    # 4. Bonferroni correction for 5 predictions
    logger.info("\n=== Bonferroni correction (5 tests) ===")
    p_values = {
        "P1_spearman": p_spear,
        "P4_objective": p_obj,
        "P4_family": p_fam,
    }
    n_tests = 5  # P1-P5
    for name, p in p_values.items():
        corrected = min(p * n_tests, 1.0)
        sig = "***" if corrected < 0.001 else "**" if corrected < 0.01 else "*" if corrected < 0.05 else "ns"
        logger.info(f"{name}: raw p={p:.2e}, Bonferroni={corrected:.2e} {sig}")
        results[f"bonferroni_{name}"] = {"raw_p": float(f"{p:.2e}"), "corrected_p": float(f"{corrected:.2e}")}

    # 5. OP1 sigmoid fit bootstrap CI
    logger.info("\n=== OP1 sigmoid fit R² CI ===")
    causal = json.loads(Path("data/experiments/causal_bitrate_verification.json").read_text())
    r2s = []
    for r in causal["results"]:
        fit = r["sigmoid_fit_alignment"]
        if fit.get("r2_fit") is not None:
            r2s.append(fit["r2_fit"])
    r2s = np.array(r2s)
    r2_mean, r2_lo, r2_hi = bootstrap_ci(r2s)
    logger.info(f"OP1 R² = {r2_mean:.4f} [{r2_lo:.4f}, {r2_hi:.4f}] (n={len(r2s)})")
    results["op1_r2_ci"] = [round(r2_mean, 4), round(r2_lo, 4), round(r2_hi, 4)]

    # Save
    output_path = Path("data/experiments/statistical_significance.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nAll results saved to %s", output_path)


if __name__ == "__main__":
    main()
