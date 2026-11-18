"""Compute the BCCT Alignment Atlas: pairwise metrics for all extracted models.

Usage:
    python scripts/compute_alignment_atlas.py --features-dir data/features --output data/atlas.json

Produces a JSON atlas with pairwise BCCT metrics for all model pairs,
plus individual model scorecards.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
from latent.metrics.regime import classify_regime
from latent.metrics.transport import transport_asymmetry_index

logger = logging.getLogger(__name__)


def load_features(
    features_dir: Path,
    include_modalities: set[str] | None = None,
    exclude_suffixes: tuple[str, ...] = ("_text", "_labels"),
) -> dict[str, dict]:
    """Load extracted feature files, filtering by modality.

    Args:
        features_dir: Directory containing ``*.pt`` feature files.
        include_modalities: If provided, only load files whose
            ``model_config.modality`` is in this set. Defaults to vision-only
            when ``None`` is passed by callers that want strict filtering;
            kept ``None`` here means "no modality filter, but still drop
            files matching ``exclude_suffixes``" so the default vision atlas
            command produces 20 vision models / 190 pairs even when text or
            label tensors live in the same directory.
        exclude_suffixes: Filename stems ending in any of these are skipped.
            Defaults catch sibling files like ``clip_vitb16_text.pt`` and
            label tensors that accidentally share ``data/features/``.

    Returns:
        Dict mapping model_name → {"features": Tensor, "config": dict}.
    """
    models: dict[str, dict] = {}
    skipped: list[str] = []
    for pt_file in sorted(features_dir.glob("*.pt")):
        stem = pt_file.stem
        if any(stem.endswith(sfx) for sfx in exclude_suffixes):
            skipped.append(pt_file.name)
            continue
        data = torch.load(pt_file, weights_only=True)
        # Skip non-feature tensors (e.g., label files saved as raw Tensors)
        if not isinstance(data, dict) or "features" not in data:
            skipped.append(pt_file.name)
            continue
        config = data.get("model_config", {}) or {}
        if include_modalities is not None:
            modality = config.get("modality", "vision")  # default = vision
            if modality not in include_modalities:
                skipped.append(pt_file.name)
                continue
        name = data["model_name"]
        models[name] = {
            "features": data["features"],
            "config": config,
        }
        logger.info("Loaded %s: shape=%s", name, data["features"].shape)
    if skipped:
        logger.info("Skipped %d non-target files: %s", len(skipped), skipped)
    return models


def compute_pairwise_metrics(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    k: int = 10,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """Compute all BCCT pairwise metrics between two feature sets.

    Uses reduced permutations for speed (100 instead of 1000).
    """
    # Ensure same number of samples
    n = min(features_a.shape[0], features_b.shape[0])
    fa = features_a[:n]
    fb = features_b[:n]

    # Move to GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fa = fa.to(device)
    fb = fb.to(device)

    # Alignment
    s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
        fa, fb, k=k, n_permutations=n_permutations, seed=seed
    )
    s_global = centered_kernel_alignment(fa, fb)

    # Bitrate
    delta = bottleneck_mismatch(fa, fb)

    transport = transport_asymmetry_index(fa, fb, mlp_steps=3000, seed=seed)
    tau_pair = min(transport.a_to_b.tau, transport.b_to_a.tau)

    # Lambda
    lambda_val = s_local / s_global if s_global > 1e-10 else float("inf")

    # Regime
    r_a = representation_bitrate(fa).bitrate
    r_b = representation_bitrate(fb).bitrate
    regime = classify_regime(
        bitrate=(r_a + r_b) / 2,
        s_local=s_local,
        tau=tau_pair,
        s_global=s_global,
        lambda_val=lambda_val,
    )

    return {
        "s_local": round(s_local, 3),
        "s_global": round(s_global, 4),
        "lambda": round(lambda_val, 4),
        "tau": round(tau_pair, 4),
        "tau_a_to_b": round(transport.a_to_b.tau, 4),
        "tau_b_to_a": round(transport.b_to_a.tau, 4),
        "tai": round(transport.tai, 4),
        "perf_linear": round(min(transport.a_to_b.perf_linear, transport.b_to_a.perf_linear), 4),
        "perf_best": round(min(transport.a_to_b.perf_best, transport.b_to_a.perf_best), 4),
        "delta": round(delta, 3),
        "regime": regime.regime.value,
        "raw_overlap": round(raw_overlap, 4),
    }


def compute_individual_metrics(features: torch.Tensor) -> dict:
    """Compute individual model metrics (bitrate, effective rank)."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    f = features.to(device)
    br = representation_bitrate(f)
    return {
        "bitrate": round(br.bitrate, 3),
        "effective_rank": round(br.effective_rank_value, 2),
        "dim": features.shape[1],
        "n_samples": features.shape[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BCCT Alignment Atlas")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--output", type=str, default="data/atlas.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        logger.error("Features directory not found: %s", features_dir)
        logger.info("Run scripts/extract_features.py first.")
        return

    models = load_features(features_dir)
    model_names = sorted(models.keys())
    n_models = len(model_names)

    logger.info(
        "Computing atlas for %d models (%d pairs)", n_models, n_models * (n_models - 1) // 2
    )

    atlas = {
        "models": {},
        "pairwise": {},
        "metadata": {
            "k": args.k,
            "n_permutations": args.permutations,
            "seed": args.seed,
            "n_models": n_models,
        },
    }

    # Individual metrics
    for name in model_names:
        logger.info("Computing individual metrics for %s", name)
        atlas["models"][name] = {
            "metrics": compute_individual_metrics(models[name]["features"]),
            "config": models[name]["config"],
        }

    # Pairwise metrics
    total_pairs = n_models * (n_models - 1) // 2
    with tqdm(total=total_pairs, desc="Pairwise metrics") as pbar:
        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_a = model_names[i]
                name_b = model_names[j]

                logger.info("Computing %s ↔ %s", name_a, name_b)

                metrics = compute_pairwise_metrics(
                    models[name_a]["features"],
                    models[name_b]["features"],
                    k=args.k,
                    n_permutations=args.permutations,
                    seed=args.seed,
                )

                pair_key = f"{name_a} <-> {name_b}"
                atlas["pairwise"][pair_key] = metrics
                pbar.update(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(atlas, f, indent=2)

    logger.info("Atlas saved to %s", output_path)

    # Print summary
    regimes = {}
    for _pair_key, metrics in atlas["pairwise"].items():
        r = metrics["regime"]
        regimes[r] = regimes.get(r, 0) + 1

    logger.info("Regime distribution:")
    for regime, count in sorted(regimes.items()):
        logger.info("  %s: %d pairs", regime, count)


if __name__ == "__main__":
    main()
