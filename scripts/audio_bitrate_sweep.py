"""Audio bitrate sweep: vary PCA bottleneck dimension and measure alignment+transport.

Analogous to the vision bitrate sweep. Takes pre-extracted audio features,
applies PCA truncation at various ranks to modulate effective bitrate,
then measures s_local and τ for each rank.

Usage:
    python scripts/audio_bitrate_sweep.py --device cuda:0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import Tensor

from latent.metrics.alignment import null_calibrated_mutual_knn
from latent.metrics.bitrate import representation_bitrate
from latent.metrics.transport import transport_asymmetry_index

logger = logging.getLogger(__name__)


def pca_truncate(features: Tensor, rank: int) -> Tensor:
    """Apply PCA truncation to reduce effective bitrate.

    Args:
        features: (N, D) feature matrix.
        rank: Number of principal components to keep.

    Returns:
        Truncated features projected back to original space, shape (N, D).
    """
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # Truncate
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    # Reconstruct in original space
    reconstructed = U_k @ torch.diag(S_k) @ Vh_k + mean
    return reconstructed


def run_sweep(
    feat_a: Tensor,
    feat_b: Tensor,
    name_a: str,
    name_b: str,
    ranks: list[int],
) -> list[dict]:
    """Run bitrate sweep for one pair at multiple PCA ranks."""
    results = []
    for rank in ranks:
        logger.info(f"  rank={rank}")
        fa = pca_truncate(feat_a, rank)
        fb = pca_truncate(feat_b, rank)

        r_a = representation_bitrate(fa)
        r_b = representation_bitrate(fb)
        s_cal, s_raw, null_mean = null_calibrated_mutual_knn(fa, fb, k=10)
        transport_result = transport_asymmetry_index(fa, fb)
        tau_pair = min(transport_result.a_to_b.tau, transport_result.b_to_a.tau)

        results.append(
            {
                "pair": f"{name_a} × {name_b}",
                "rank": rank,
                "bitrate_a": round(float(r_a.bitrate), 2),
                "bitrate_b": round(float(r_b.bitrate), 2),
                "bitrate_mean": round((float(r_a.bitrate) + float(r_b.bitrate)) / 2, 2),
                "s_local": round(float(s_cal), 4),
                "tau": round(float(tau_pair), 4),
                "tau_a_to_b": round(float(transport_result.a_to_b.tau), 4),
                "tau_b_to_a": round(float(transport_result.b_to_a.tau), 4),
            }
        )
        logger.info(
            f"    R_mean={results[-1]['bitrate_mean']:.2f}, "
            f"s_local={results[-1]['s_local']:.4f}, "
            f"tau={results[-1]['tau']:.4f}"
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audio bitrate sweep")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    features_dir = Path("data/audio_features")
    pt_files = sorted(features_dir.glob("*.pt"))

    # Load features
    models: dict[str, Tensor] = {}
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=True)
        name = data["model_name"]
        models[name] = data["features"].to(device)
        logger.info(f"Loaded {name}: shape={models[name].shape}")

    # Select representative pairs for sweep:
    # 1. Same-family convergent: hubert-base × hubert-large
    # 2. Same-family broken: wav2vec2-base × wav2vec2-large
    # 3. Cross-family: hubert-base × whisper-base
    sweep_pairs = [
        ("hubert-base", "hubert-large"),
        ("wav2vec2-base", "wav2vec2-large"),
        ("hubert-base", "whisper-base"),
        ("wav2vec2-large", "whisper-small"),
    ]

    # Min dim across all models is 512 (whisper-base)
    # Use ranks from 4 to min_dim
    ranks = [4, 8, 16, 32, 64, 128, 256, 512]

    all_results = []
    for name_a, name_b in sweep_pairs:
        if name_a not in models or name_b not in models:
            logger.warning(f"Skipping {name_a} × {name_b}: model not found")
            continue
        logger.info(f"Sweep: {name_a} × {name_b}")
        min_dim = min(models[name_a].shape[1], models[name_b].shape[1])
        pair_ranks = [r for r in ranks if r <= min_dim]
        results = run_sweep(models[name_a], models[name_b], name_a, name_b, pair_ranks)
        all_results.extend(results)

    output_path = Path("data/audio_bitrate_sweep.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved {len(all_results)} sweep points to {output_path}")


if __name__ == "__main__":
    main()
