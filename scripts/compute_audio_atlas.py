"""Compute pairwise BCCT metrics for audio models.

Reads extracted audio features from data/audio_features/*.pt and computes
all 6 BCCT metrics for each pair. Outputs data/audio_atlas.json.

Usage:
    python scripts/compute_audio_atlas.py
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import torch

from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
from latent.metrics.bitrate import representation_bitrate
from latent.metrics.regime import classify_regime
from latent.metrics.transport import transport_asymmetry_index

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    features_dir = Path("data/audio_features")
    pt_files = sorted(features_dir.glob("*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {features_dir}")

    # Load all features
    models: dict[str, dict] = {}
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=True)
        name = data["model_name"]
        features = data["features"].to(device)
        config = data["model_config"]
        logger.info(f"Loaded {name}: shape={features.shape}, family={config['family']}")
        models[name] = {"features": features, "config": config}

    logger.info(
        f"Loaded {len(models)} models, computing {len(models) * (len(models) - 1) // 2} pairs"
    )

    # Compute pairwise metrics
    results = []
    model_names = list(models.keys())

    for name_a, name_b in combinations(model_names, 2):
        logger.info(f"Computing: {name_a} × {name_b}")
        feat_a = models[name_a]["features"]
        feat_b = models[name_b]["features"]

        # Ensure same number of samples
        n = min(feat_a.shape[0], feat_b.shape[0])
        feat_a = feat_a[:n]
        feat_b = feat_b[:n]

        # Alignment metrics
        s_local_calibrated, s_local_raw, null_mean = null_calibrated_mutual_knn(
            feat_a, feat_b, k=10
        )
        s_global = centered_kernel_alignment(feat_a, feat_b)

        lambda_val = (
            s_local_calibrated / s_global
            if s_global > 1e-10
            else float("inf")
            if s_local_calibrated > 0
            else 0.0
        )

        transport_result = transport_asymmetry_index(feat_a, feat_b)
        tau = min(transport_result.a_to_b.tau, transport_result.b_to_a.tau)
        # Bitrate
        r_a = representation_bitrate(feat_a)
        r_b = representation_bitrate(feat_b)
        delta = abs(r_a.bitrate - r_b.bitrate)

        # Regime classification
        regime = classify_regime(
            bitrate=(r_a.bitrate + r_b.bitrate) / 2,
            s_local=s_local_calibrated,
            tau=tau,
            s_global=s_global,
            lambda_val=lambda_val,
        )

        pair_result = {
            "pair": f"{name_a} × {name_b}",
            "model_a": name_a,
            "model_b": name_b,
            "family_a": models[name_a]["config"]["family"],
            "family_b": models[name_b]["config"]["family"],
            "objective_a": models[name_a]["config"]["objective"],
            "objective_b": models[name_b]["config"]["objective"],
            "s_local": round(float(s_local_calibrated), 4),
            "s_local_raw": round(float(s_local_raw), 4),
            "null_mean": round(float(null_mean), 4),
            "s_global": round(float(s_global), 4),
            "lambda": round(float(lambda_val), 4),
            "tau": round(float(tau), 4),
            "tau_a_to_b": round(float(transport_result.a_to_b.tau), 4),
            "tau_b_to_a": round(float(transport_result.b_to_a.tau), 4),
            "tai": round(float(transport_result.tai), 4),
            "bitrate_a": round(float(r_a.bitrate), 2),
            "bitrate_b": round(float(r_b.bitrate), 2),
            "delta": round(float(delta), 2),
            "regime": regime.regime.value,
        }
        results.append(pair_result)
        s_local = pair_result["s_local"]
        tau = pair_result["tau"]
        regime = pair_result["regime"]
        logger.info(f"  s_local={s_local:.4f}, tau={tau:.4f}, regime={regime}")

    # Summary statistics
    regime_counts: dict[str, int] = {}
    for r in results:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

    summary = {
        "n_models": len(models),
        "n_pairs": len(results),
        "models": [
            {
                "name": name,
                "family": models[name]["config"]["family"],
                "objective": models[name]["config"]["objective"],
                "dim": models[name]["config"]["dim"],
            }
            for name in model_names
        ],
        "regime_counts": regime_counts,
        "pairs": results,
    }

    output_path = Path("data/audio_atlas.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nAudio Atlas Summary:")
    logger.info(f"  Models: {len(models)}")
    logger.info(f"  Pairs: {len(results)}")
    logger.info(f"  Regime distribution: {regime_counts}")
    logger.info(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
