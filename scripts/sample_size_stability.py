# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, "src")

from latent.metrics.alignment import centered_kernel_alignment, mutual_knn_overlap  # noqa: E402
from latent.metrics.regime import classify_regime  # noqa: E402

LOGGER = logging.getLogger(__name__)

REFERENCE_N = 5000
REQUESTED_SAMPLE_SIZES = [1000, 2000, 5000, 10000, 25000]
K_NEIGHBORS = 10
N_PERMUTATIONS = 100
SEED = 42


def _load_atlas(atlas_path: Path) -> dict[str, Any]:
    with atlas_path.open() as f:
        return json.load(f)


def _load_feature_tensor(feature_path: Path) -> torch.Tensor:
    data = torch.load(feature_path, weights_only=True)
    features: Any
    if isinstance(data, dict):
        if "features" not in data:
            raise KeyError(f"Missing 'features' in {feature_path}")
        features = data["features"]  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(data, torch.Tensor):
        features = data
    else:
        raise TypeError(f"Unsupported feature format in {feature_path}: {type(data)}")

    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Loaded object is not a Tensor for {feature_path}")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {feature_path}, got shape {tuple(features.shape)}")
    if features.shape[0] != REFERENCE_N:
        raise ValueError(
            f"Expected {REFERENCE_N} samples in {feature_path}, got {features.shape[0]}"
        )

    return features.float()


def _load_features(features_dir: Path, model_names: list[str]) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for model_name in model_names:
        feature_path = features_dir / f"{model_name}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        features[model_name] = _load_feature_tensor(feature_path)
    return features


def _build_indices(sample_sizes: list[int]) -> dict[int, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(SEED)

    indices: dict[int, torch.Tensor] = {}
    for sample_size in sample_sizes:
        if sample_size > REFERENCE_N:
            LOGGER.warning(
                "Skipping N=%d (requires feature re-extraction beyond %d samples)",
                sample_size,
                REFERENCE_N,
            )
            continue
        if sample_size == REFERENCE_N:
            indices[sample_size] = torch.arange(REFERENCE_N)
        else:
            indices[sample_size] = torch.randperm(REFERENCE_N, generator=generator)[:sample_size]
    return indices


def _knn_mask(features: torch.Tensor, k: int) -> torch.Tensor:
    n = features.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than N={n}")

    dist = torch.cdist(features, features, p=2)
    dist.fill_diagonal_(float("inf"))
    neighbors = dist.topk(k, dim=1, largest=False).indices

    rows = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k)
    mask = torch.zeros(n, n, dtype=torch.bool, device=features.device)
    mask[rows, neighbors] = True
    return mask


def _overlap_from_masks(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int) -> float:
    overlap_per_row = (mask_a & mask_b).sum(dim=1).float() / k
    return float(overlap_per_row.mean().item())


def _calibrated_s_local(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    k: int,
    n_permutations: int,
    seed: int,
) -> float:
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Feature matrices must have same N: {features_a.shape[0]} vs {features_b.shape[0]}"
        )

    n = features_a.shape[0]
    mask_a = _knn_mask(features_a, k)
    mask_b = _knn_mask(features_b, k)
    raw_overlap = _overlap_from_masks(mask_a, mask_b, k)

    if n <= 2000:
        ref_overlap = mutual_knn_overlap(features_a, features_b, k=k)
        if abs(ref_overlap - raw_overlap) > 1e-6:
            LOGGER.warning(
                "Raw overlap mismatch between optimized and reference implementations: %.6f vs %.6f",
                raw_overlap,
                ref_overlap,
            )

    generator = torch.Generator(device=features_b.device)
    generator.manual_seed(seed)

    null_scores: list[float] = []
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=generator, device=features_b.device)
        mask_b_perm = mask_b[perm][:, perm]
        null_scores.append(_overlap_from_masks(mask_a, mask_b_perm, k))

    null_mean = float(torch.tensor(null_scores).mean().item())
    if (1.0 - null_mean) < 1e-10:
        return 0.0 if abs(raw_overlap - null_mean) < 1e-10 else 1.0
    return max(0.0, (raw_overlap - null_mean) / (1.0 - null_mean))


def compute_stability(
    atlas: dict[str, Any],
    features: dict[str, torch.Tensor],
    sample_sizes: list[int],
) -> dict[str, Any]:
    model_metrics = atlas["models"]
    pairwise_ref = atlas["pairwise"]
    pair_keys = sorted(pairwise_ref.keys())

    indices_by_size = _build_indices(sample_sizes)
    tested_sizes = sorted(indices_by_size.keys())
    agreement_rates: dict[str, Any] = {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    for sample_size in tested_sizes:
        idx = indices_by_size[sample_size]
        flipped_pairs: list[str] = []
        LOGGER.info("Evaluating sample size N=%d", sample_size)

        for pair_key in pair_keys:
            name_a, name_b = pair_key.split(" <-> ")
            ref = pairwise_ref[pair_key]
            bitrate = (
                float(model_metrics[name_a]["metrics"]["bitrate"])
                + float(model_metrics[name_b]["metrics"]["bitrate"])
            ) / 2.0

            feat_a = features[name_a][idx].to(device)
            feat_b = features[name_b][idx].to(device)

            s_local = _calibrated_s_local(
                feat_a,
                feat_b,
                k=K_NEIGHBORS,
                n_permutations=N_PERMUTATIONS,
                seed=SEED,
            )
            s_global = centered_kernel_alignment(feat_a, feat_b)

            pred = classify_regime(
                bitrate=bitrate,
                s_local=float(s_local),
                tau=float(ref["tau"]),
                s_global=float(s_global),
            ).regime.value

            if pred != ref["regime"]:
                flipped_pairs.append(pair_key)

        total_pairs = len(pair_keys)
        agreement_rates[str(sample_size)] = {
            "agreement": round((total_pairs - len(flipped_pairs)) / total_pairs, 3),
            "flipped_pairs": flipped_pairs,
            "total": total_pairs,
        }

    return {
        "reference_n": REFERENCE_N,
        "sample_sizes_tested": tested_sizes,
        "agreement_rates": agreement_rates,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas_path = Path("data/atlas.json")
    output_path = Path("data/sample_size_stability.json")
    features_dir = Path("data/features")

    atlas = _load_atlas(atlas_path)
    model_names = sorted(atlas["models"].keys())

    LOGGER.info("Loading %d model feature tensors", len(model_names))
    features = _load_features(features_dir, model_names)

    results = compute_stability(atlas, features, REQUESTED_SAMPLE_SIZES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Saved stability report to %s", output_path)


if __name__ == "__main__":
    main()
