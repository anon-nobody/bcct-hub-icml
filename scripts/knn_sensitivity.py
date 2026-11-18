"""k-NN sensitivity: sweep k={3,5,10,20,50,100} on 15 representative pairs.

Usage:
    python scripts/knn_sensitivity.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from latent.metrics.alignment import null_calibrated_mutual_knn
from latent.metrics.bitrate import representation_bitrate
from latent.metrics.regime import classify_regime

logger = logging.getLogger(__name__)

PAIRS = [
    ("clip_vitb16", "clip_vitl14"),
    ("clip_vitb32", "clip_vitb16"),
    ("resnet50", "resnet101"),
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("dinov2_vits14", "dinov2_vitg14"),
    ("vit_base_patch16_224", "vit_large_patch16_224"),
    ("convnext_base", "convnext_large"),
    ("beit_base_patch16_224", "beit_large_patch16_224"),
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    ("dinov2_vitb14", "resnet50"),
    ("clip_vitl14", "dinov2_vitl14"),
    ("convnext_base", "resnet101"),
    ("mixer_b16_224", "vit_base_patch16_224"),
    ("efficientnet_b0", "resnet50"),
]

K_VALUES = [3, 5, 10, 20, 50, 100]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path("data/features")
    atlas = json.loads(Path("data/atlas.json").read_text())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results = []
    for model_a, model_b in PAIRS:
        pair_key = f"{model_a} <-> {model_b}"
        alt_key = f"{model_b} <-> {model_a}"
        atlas_entry = atlas["pairwise"].get(pair_key) or atlas["pairwise"].get(alt_key)
        if not atlas_entry:
            continue

        try:
            fa = torch.load(features_dir / f"{model_a}.pt", weights_only=True)["features"].to(device)
            fb = torch.load(features_dir / f"{model_b}.pt", weights_only=True)["features"].to(device)
        except FileNotFoundError:
            continue

        n = min(fa.shape[0], fb.shape[0])
        fa, fb = fa[:n], fb[:n]

        pair_result = {"pair": pair_key, "tau": atlas_entry["tau"], "regimes": {}}

        for k in K_VALUES:
            if k >= n:
                continue
            s_local, _, _ = null_calibrated_mutual_knn(fa, fb, k=k, n_permutations=100, seed=42)

            r_a = representation_bitrate(fa).bitrate
            r_b = representation_bitrate(fb).bitrate
            regime = classify_regime(
                bitrate=(r_a + r_b) / 2, s_local=s_local,
                tau=atlas_entry["tau"],  # tau doesn't change with k
            )
            pair_result["regimes"][str(k)] = {
                "s_local": round(s_local, 4),
                "regime": regime.regime.value,
            }
            logger.info("%s k=%d: s_local=%.4f regime=%s", pair_key, k, s_local, regime.regime.value)

        results.append(pair_result)

    # Regime stability: fraction of pairs with same regime across all k
    stable = 0
    for r in results:
        regimes = [v["regime"] for v in r["regimes"].values()]
        if len(set(regimes)) == 1:
            stable += 1
    stability = stable / len(results) if results else 0

    output = {
        "k_values": K_VALUES,
        "n_pairs": len(results),
        "regime_stability": round(stability, 4),
        "results": results,
    }

    output_path = Path("data/experiments/knn_sensitivity.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Regime stability: %.1f%% (%d/%d pairs same regime across all k)",
                stability * 100, stable, len(results))


if __name__ == "__main__":
    main()
