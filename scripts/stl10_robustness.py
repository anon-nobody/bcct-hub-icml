"""STL-10 Dataset Robustness Validation.

Extracts features from 8 representative vision models on STL-10 (96x96),
computes 28 pairwise BCCT metrics, and compares regime classifications
with the CIFAR-100 atlas.

Usage:
    python scripts/stl10_robustness.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
from latent.metrics.bitrate import representation_bitrate
from latent.metrics.regime import classify_regime
from latent.metrics.transport import transport_asymmetry_index

logger = logging.getLogger(__name__)

REPRESENTATIVE_MODELS = [
    ("dinov2_vitb14", "dinov2", "torch.hub"),
    ("clip_vitl14", "clip", "transformers"),
    ("resnet50", "supervised", "timm"),
    ("vit_base_patch16_224", "supervised", "timm"),
    ("convnext_base", "supervised", "timm"),
    ("vit_base_patch16_224.mae", "mae", "timm"),
    ("beit_base_patch16_224", "beit", "timm"),
    ("mixer_b16_224", "mlpmixer", "timm"),
]


def extract_stl10_features(output_dir: Path, device: str, n_samples: int = 5000) -> None:
    """Extract features from 8 models on STL-10."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.STL10(root="data/stl10", split="test", download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    logger.info("STL-10 test set: %d images", len(dataset))

    for model_name, family, source in REPRESENTATIVE_MODELS:
        out_path = output_dir / f"{model_name}.pt"
        if out_path.exists():
            logger.info("Skipping %s (cached)", model_name)
            continue

        logger.info("Loading %s (%s)...", model_name, source)
        try:
            if source == "torch.hub":
                model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
            elif source == "transformers":
                from transformers import CLIPVisionModelWithProjection

                model = CLIPVisionModelWithProjection.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
            else:
                model = timm.create_model(model_name, pretrained=True, num_classes=0)

            model = model.to(device).eval()
            features_list = []
            collected = 0
            with torch.no_grad():
                for batch_images, _ in dataloader:
                    if collected >= n_samples:
                        break
                    batch_images = batch_images.to(device)
                    if source == "transformers":
                        out = model(pixel_values=batch_images).image_embeds
                    else:
                        out = model(batch_images)
                    out = F.normalize(out, dim=-1)
                    features_list.append(out.cpu())
                    collected += out.shape[0]

            features = torch.cat(features_list, dim=0)[:n_samples]
            torch.save(
                {
                    "features": features,
                    "model_name": model_name,
                    "model_config": {"family": family},
                },
                out_path,
            )
            logger.info("  Saved %s: shape=%s", model_name, features.shape)

            del model
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("  Failed %s", model_name)


def compute_stl10_atlas(features_dir: Path, device: str) -> dict:
    """Compute 28 pairwise BCCT metrics for STL-10."""
    model_names = [m[0] for m in REPRESENTATIVE_MODELS]
    features = {}
    bitrates = {}

    for name in model_names:
        data = torch.load(features_dir / f"{name}.pt", map_location="cpu", weights_only=True)
        feat = data["features"]
        features[name] = feat
        r = representation_bitrate(feat)
        bitrates[name] = r.bitrate
        logger.info("Loaded %s: shape=%s, R=%.3f", name, feat.shape, r.bitrate)

    results = []
    pairs = list(combinations(model_names, 2))
    logger.info("Computing %d pairs...", len(pairs))

    for i, (m1, m2) in enumerate(pairs):
        f1 = features[m1].to(device)
        f2 = features[m2].to(device)
        n = min(f1.shape[0], f2.shape[0])
        f1, f2 = f1[:n], f2[:n]

        s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
            f1, f2, k=10, n_permutations=100
        )
        s_global = centered_kernel_alignment(f1, f2)
        transport_result = transport_asymmetry_index(f1, f2, mlp_steps=500)
        tau_pair = min(transport_result.a_to_b.tau, transport_result.b_to_a.tau)
        lambda_val = s_local / s_global if s_global > 1e-10 else float("inf")
        regime_result = classify_regime(
            bitrate=(bitrates[m1] + bitrates[m2]) / 2,
            s_local=s_local,
            tau=tau_pair,
            s_global=s_global,
            lambda_val=lambda_val,
        )

        results.append(
            {
                "model_a": m1,
                "model_b": m2,
                "s_local": round(s_local, 4),
                "s_global": round(s_global, 4),
                "tau": round(tau_pair, 4),
                "tau_a_to_b": round(transport_result.a_to_b.tau, 4),
                "tau_b_to_a": round(transport_result.b_to_a.tau, 4),
                "tai": round(transport_result.tai, 4),
                "perf_linear": round(
                    min(transport_result.a_to_b.perf_linear, transport_result.b_to_a.perf_linear), 4
                ),
                "perf_best": round(
                    min(transport_result.a_to_b.perf_best, transport_result.b_to_a.perf_best), 4
                ),
                "R_a": round(bitrates[m1], 3),
                "R_b": round(bitrates[m2], 3),
                "regime": regime_result.regime.value,
            }
        )
        logger.info(
            "  [%d/%d] %s x %s: s_local=%.3f, tau=%.3f -> %s",
            i + 1,
            len(pairs),
            m1,
            m2,
            s_local,
            tau_pair,
            regime_result.regime.value,
        )
        f1, f2 = f1.cpu(), f2.cpu()
        torch.cuda.empty_cache()

    return {
        "dataset": "STL-10",
        "n_models": len(model_names),
        "n_pairs": len(results),
        "models": {n: {"bitrate": bitrates[n], "dim": features[n].shape[1]} for n in model_names},
        "pairs": results,
    }


def compare_with_cifar100(stl10_atlas: dict, cifar_path: str = "data/atlas.json") -> dict:
    """Compare STL-10 regime classifications with CIFAR-100."""
    with open(cifar_path) as f:
        cifar = json.load(f)
    cifar_pairs: dict = {}
    for key, val in cifar["pairwise"].items():
        parts = key.split(" <-> ")
        if len(parts) == 2:
            cifar_pairs[(parts[0], parts[1])] = val
            cifar_pairs[(parts[1], parts[0])] = val

    matches, mismatches = 0, 0
    comparison = []
    for r in stl10_atlas["pairs"]:
        cifar_r = cifar_pairs.get((r["model_a"], r["model_b"]))
        if cifar_r:
            match = r["regime"] == cifar_r["regime"]
            matches += int(match)
            mismatches += int(not match)
            comparison.append(
                {
                    "pair": f"{r['model_a']} x {r['model_b']}",
                    "stl10_regime": r["regime"],
                    "cifar100_regime": cifar_r["regime"],
                    "match": match,
                }
            )

    total = matches + mismatches
    agreement = matches / total * 100 if total > 0 else 0
    logger.info("Agreement: %d/%d (%.1f%%)", matches, total, agreement)
    return {"agreement_rate": agreement, "matches": matches, "total": total, "pairs": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description="STL-10 Robustness Validation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path("data/features_stl10")
    extract_stl10_features(features_dir, args.device, args.n_samples)

    atlas = compute_stl10_atlas(features_dir, args.device)
    with open("data/stl10_atlas.json", "w") as f:
        json.dump(atlas, f, indent=2)

    comparison = compare_with_cifar100(atlas)
    with open("data/stl10_vs_cifar100_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    agreement_rate = comparison["agreement_rate"]
    matches = comparison["matches"]
    total = comparison["total"]
    print(f"Agreement: {agreement_rate:.1f}% ({matches}/{total})")


if __name__ == "__main__":
    main()
