"""OP1: Causal verification of the bitrate-alignment relationship.

For each model pair, systematically compresses model A via rank reduction
at 7 levels and measures R (bitrate), S_local (alignment), and τ (transport)
against the uncompressed model B.

Fits a sigmoid S_local ~ S_max / (1 + exp(-κ(R - R0))) to each pair's curve
and extracts R_crit (50% of S_max) and R_sat (90% of S_max).

Usage:
    python scripts/causal_bitrate_verification.py [--device cuda:0] [--output data/experiments/causal_bitrate_verification.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

from latent.evaluation.protocols import _compress_rank_reduction
from latent.metrics.alignment import null_calibrated_mutual_knn
from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
from latent.metrics.transport import transport_linearity_score

logger = logging.getLogger(__name__)

# 10 pairs covering all regime types
PAIRS = [
    # Convergent (same family, same objective)
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("clip_vitb16", "clip_vitl14"),
    ("resnet50", "resnet101"),
    # Local-Only (different family/objective)
    ("dinov2_vitl14", "beit_large_patch16_224"),
    ("vit_base_patch16_224", "resnet50"),
    # Divergent (very different objectives)
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    # Cross-scale
    ("dinov2_vits14", "dinov2_vitg14"),
    ("clip_vitb32", "clip_vitl14"),
    ("convnext_base", "swin_base_patch4_window7_224"),
]

COMPRESSION_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]


def sigmoid(r: np.ndarray, s_max: float, kappa: float, r0: float) -> np.ndarray:
    """Sigmoid model: S_local ~ S_max / (1 + exp(-kappa * (R - R0)))."""
    return s_max / (1.0 + np.exp(-kappa * (r - r0)))


def fit_sigmoid(
    r_values: list[float], s_values: list[float]
) -> dict:
    """Fit sigmoid to (R, S_local) data and extract critical points."""
    r_arr = np.array(r_values)
    s_arr = np.array(s_values)

    try:
        # Initial guesses: S_max=max(s), kappa=1, R0=median(R)
        p0 = [max(s_arr) * 1.1, 1.0, np.median(r_arr)]
        bounds = ([0, 0, 0], [max(s_arr) * 5, 50, max(r_arr) * 2])
        popt, pcov = curve_fit(sigmoid, r_arr, s_arr, p0=p0, bounds=bounds, maxfev=10000)
        s_max, kappa, r0 = popt

        # Goodness of fit
        s_pred = sigmoid(r_arr, *popt)
        ss_res = np.sum((s_arr - s_pred) ** 2)
        ss_tot = np.sum((s_arr - np.mean(s_arr)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

        # Critical points
        # R_crit: where sigmoid reaches 50% of S_max
        r_crit = r0  # sigmoid midpoint is at R0
        # R_sat: where sigmoid reaches 90% of S_max → solve: 0.9 = 1/(1+exp(-k*(R-R0)))
        # → exp(-k*(R-R0)) = 1/9 → R = R0 + ln(9)/k
        r_sat = r0 + np.log(9) / max(kappa, 1e-6)

        return {
            "s_max": round(float(s_max), 4),
            "kappa": round(float(kappa), 4),
            "r0": round(float(r0), 4),
            "r2_fit": round(float(r2), 4),
            "r_crit": round(float(r_crit), 4),
            "r_sat": round(float(r_sat), 4),
        }
    except (RuntimeError, ValueError) as e:
        logger.warning("Sigmoid fit failed: %s", e)
        return {
            "s_max": None,
            "kappa": None,
            "r0": None,
            "r2_fit": None,
            "r_crit": None,
            "r_sat": None,
            "error": str(e),
        }


def _fit_linear(r: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * r + b

def _fit_log(r: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.log(np.maximum(r, 1e-10)) + b

def _fit_power(r: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(np.maximum(r, 1e-10), b) + c


def model_comparison(r_values: list[float], s_values: list[float]) -> dict:
    """Compare sigmoid vs linear vs log vs power law fits using BIC."""
    r_arr = np.array(r_values)
    s_arr = np.array(s_values)
    n = len(r_arr)
    results = {}

    models = {
        "linear": (_fit_linear, [0.01, 0], 2),
        "log": (_fit_log, [0.1, 0], 2),
        "sigmoid": (sigmoid, [max(s_arr) * 1.1, 1.0, np.median(r_arr)], 3),
    }

    for name, (fn, p0, k) in models.items():
        try:
            if name == "sigmoid":
                bounds = ([0, 0, 0], [max(s_arr) * 5, 50, max(r_arr) * 2])
                popt, _ = curve_fit(fn, r_arr, s_arr, p0=p0, bounds=bounds, maxfev=10000)
            else:
                popt, _ = curve_fit(fn, r_arr, s_arr, p0=p0, maxfev=10000)

            pred = fn(r_arr, *popt)
            ss_res = np.sum((s_arr - pred) ** 2)
            ss_tot = np.sum((s_arr - np.mean(s_arr)) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

            # BIC = n * ln(ss_res/n) + k * ln(n)
            bic = n * np.log(ss_res / n + 1e-20) + k * np.log(n)
            results[name] = {"r2": round(float(r2), 4), "bic": round(float(bic), 2), "n_params": k}
        except Exception:
            results[name] = {"r2": None, "bic": None, "n_params": k}

    # Power law has 3 params
    try:
        popt, _ = curve_fit(_fit_power, r_arr, s_arr, p0=[0.1, 0.5, 0], maxfev=10000)
        pred = _fit_power(r_arr, *popt)
        ss_res = np.sum((s_arr - pred) ** 2)
        ss_tot = np.sum((s_arr - np.mean(s_arr)) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        bic = n * np.log(ss_res / n + 1e-20) + 3 * np.log(n)
        results["power"] = {"r2": round(float(r2), 4), "bic": round(float(bic), 2), "n_params": 3}
    except Exception:
        results["power"] = {"r2": None, "bic": None, "n_params": 3}

    return results


def permutation_null(r_values: list[float], s_values: list[float],
                     n_perms: int = 1000, seed: int = 42) -> dict:
    """Compute permutation null distribution for sigmoid R²."""
    rng = np.random.default_rng(seed)
    r_arr = np.array(r_values)
    np.array(s_values)

    # Observed R²
    obs_fit = fit_sigmoid(r_values, s_values)
    obs_r2 = obs_fit.get("r2_fit")
    if obs_r2 is None:
        return {"obs_r2": None, "null_mean": None, "null_std": None, "p_value": None}

    # Permutation null: shuffle R values, fit sigmoid
    null_r2s = []
    for _ in range(n_perms):
        perm_r = rng.permutation(r_arr)
        null_fit = fit_sigmoid(perm_r.tolist(), s_values)
        if null_fit.get("r2_fit") is not None:
            null_r2s.append(null_fit["r2_fit"])

    null_r2s = np.array(null_r2s)
    p_value = (null_r2s >= obs_r2).mean() if len(null_r2s) > 0 else 1.0

    return {
        "obs_r2": round(obs_r2, 4),
        "null_mean": round(float(null_r2s.mean()), 4) if len(null_r2s) > 0 else None,
        "null_std": round(float(null_r2s.std()), 4) if len(null_r2s) > 0 else None,
        "p_value": round(float(p_value), 4),
        "n_perms": len(null_r2s),
    }


def load_features(features_dir: Path, model_name: str, device: str) -> torch.Tensor:
    """Load features for a model."""
    pt_file = features_dir / f"{model_name}.pt"
    data = torch.load(pt_file, weights_only=True)
    return data["features"].to(device)


def run_pair(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    pair_name: str,
    k: int = 10,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """Run causal verification for one model pair."""
    # Ensure same N
    n = min(features_a.shape[0], features_b.shape[0])
    fa = features_a[:n]
    fb = features_b[:n]

    points = []
    for level in COMPRESSION_LEVELS:
        logger.info("  %s: compression level=%.4f", pair_name, level)

        # Compress model A
        compressed_a = _compress_rank_reduction(fa, level)

        # Measure bitrate
        r = representation_bitrate(compressed_a).bitrate
        r_b = representation_bitrate(fb).bitrate
        delta = bottleneck_mismatch(compressed_a, fb)

        # Measure alignment
        s_local, raw_overlap, _ = null_calibrated_mutual_knn(
            compressed_a, fb, k=k, n_permutations=n_permutations, seed=seed
        )

        # Measure transport
        transport = transport_linearity_score(compressed_a, fb, mlp_steps=3000, seed=seed)

        points.append({
            "level": level,
            "bitrate_a": round(r, 4),
            "bitrate_b": round(r_b, 4),
            "delta": round(delta, 4),
            "s_local": round(s_local, 4),
            "raw_overlap": round(raw_overlap, 4),
            "tau": round(transport.tau, 4),
            "perf_linear": round(transport.perf_linear, 4),
            "perf_best": round(transport.perf_best, 4),
            "converged": transport.converged,
        })

    # Fit sigmoid to (R, S_local)
    r_values = [p["bitrate_a"] for p in points]
    s_values = [p["s_local"] for p in points]
    sigmoid_fit = fit_sigmoid(r_values, s_values)

    # Also fit sigmoid to (R, tau)
    tau_values = [p["tau"] for p in points]
    sigmoid_fit_tau = fit_sigmoid(r_values, tau_values)

    # Model comparison (sigmoid vs linear vs log vs power)
    models = model_comparison(r_values, s_values)

    # Permutation null for sigmoid R²
    null_test = permutation_null(r_values, s_values, n_perms=1000)

    return {
        "pair": pair_name,
        "points": points,
        "sigmoid_fit_alignment": sigmoid_fit,
        "sigmoid_fit_transport": sigmoid_fit_tau,
        "model_comparison": models,
        "permutation_null": null_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OP1: Causal bitrate verification")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--output", type=str, default="data/experiments/causal_bitrate_verification.json")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path(args.features_dir)
    results = []

    for model_a, model_b in PAIRS:
        pair_name = f"{model_a} <-> {model_b}"
        logger.info("Processing pair: %s", pair_name)

        try:
            fa = load_features(features_dir, model_a, args.device)
            fb = load_features(features_dir, model_b, args.device)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pair_name, e)
            continue

        pair_result = run_pair(fa, fb, pair_name, k=args.k, n_permutations=args.permutations, seed=args.seed)
        results.append(pair_result)

    # Summary statistics
    successful_fits = [r for r in results if r["sigmoid_fit_alignment"].get("r2_fit") is not None]
    good_fits = [r for r in successful_fits if r["sigmoid_fit_alignment"]["r2_fit"] > 0.5]

    summary = {
        "n_pairs": len(results),
        "n_successful_fits": len(successful_fits),
        "n_good_fits_r2_gt_0.5": len(good_fits),
        "mean_r2_alignment": round(
            np.mean([r["sigmoid_fit_alignment"]["r2_fit"] for r in successful_fits]), 4
        ) if successful_fits else None,
        "mean_r_crit": round(
            np.mean([r["sigmoid_fit_alignment"]["r_crit"] for r in good_fits]), 4
        ) if good_fits else None,
        "mean_r_sat": round(
            np.mean([r["sigmoid_fit_alignment"]["r_sat"] for r in good_fits]), 4
        ) if good_fits else None,
    }

    output = {
        "metadata": {
            "compression_levels": COMPRESSION_LEVELS,
            "k": args.k,
            "n_permutations": args.permutations,
            "seed": args.seed,
        },
        "results": results,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
