"""Track B: Bitrate Sweep Experiment.

Runs all three BCCT evaluation protocols on selected model pairs from the atlas,
then saves structured results for paper figures and analysis.

Usage:
    python scripts/run_bitrate_experiment.py --features-dir data/features --output data/experiments/
    python scripts/run_bitrate_experiment.py --device cuda:2 --pairs "dinov2_vitb14,clip_vitl14"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from latent.evaluation.protocols import (
    BitrateSweepResult,
    NeighborhoodAlignmentResult,
    StressTestResult,
    run_bitrate_sweep,
    run_neighborhood_alignment,
    run_transport_stress_test,
)

logger = logging.getLogger(__name__)

# Representative model pairs for the experiment
# Chosen to cover all 4 regimes and key comparisons
DEFAULT_PAIRS: list[tuple[str, str]] = [
    # Same-family, same-objective (expected: Convergent)
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("resnet50", "resnet101"),
    ("clip_vitb16", "clip_vitl14"),
    # Cross-family, same-objective (expected: Local-Only)
    ("dinov2_vitl14", "beit_large_patch16_224"),
    ("vit_base_patch16_224", "resnet50"),
    # Cross-objective (expected: Divergent or Local-Only)
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    # Extreme bitrate mismatch
    ("vit_base_patch16_224.mae", "resnet101"),
]


def load_features(features_dir: Path) -> dict[str, torch.Tensor]:
    """Load all extracted feature files."""
    features = {}
    for f in sorted(features_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "features" in data:
            features[f.stem] = data["features"]
        elif isinstance(data, torch.Tensor):
            features[f.stem] = data
        else:
            logger.warning("Unknown format in %s, skipping", f.name)
    return features


def serialize_sweep_result(result: BitrateSweepResult) -> dict:
    """Serialize BitrateSweepResult to JSON-safe dict."""
    return {
        "model_name": result.model_name,
        "reference_name": result.reference_name,
        "compression_method": result.compression_method,
        "points": [
            {
                "level": p.level,
                "bitrate": round(p.bitrate, 4),
                "s_local": round(p.s_local, 4),
                "tau": round(p.tau, 4),
                "downstream_acc": p.downstream_acc,
            }
            for p in result.points
        ],
    }


def serialize_alignment_result(result: NeighborhoodAlignmentResult) -> dict:
    """Serialize NeighborhoodAlignmentResult to JSON-safe dict."""
    return {
        "s_local": round(result.s_local, 4),
        "s_global": round(result.s_global, 4),
        "lambda_val": round(result.lambda_val, 4) if result.lambda_val < 1e6 else "inf",
        "raw_overlap": round(result.raw_overlap, 4),
        "null_mean": round(result.null_mean, 4),
        "interpretation": result.interpretation,
    }


def serialize_stress_result(result: StressTestResult) -> dict:
    """Serialize StressTestResult to JSON-safe dict."""
    return {
        "model_a_name": result.model_a_name,
        "model_b_name": result.model_b_name,
        "points": [
            {
                "delta": round(p.delta, 4),
                "tau_a_to_b": round(p.tau_a_to_b, 4),
                "tau_b_to_a": round(p.tau_b_to_a, 4),
                "tai": round(p.tai, 4),
                "compression_level": p.compression_level,
                "regime": p.regime,
            }
            for p in result.points
        ],
    }


def run_experiment_for_pair(
    name_a: str,
    name_b: str,
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    k: int = 10,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict:
    """Run all three protocols for a single model pair."""
    pair_key = f"{name_a} <-> {name_b}"
    logger.info("=" * 60)
    logger.info("Running experiments for: %s", pair_key)
    logger.info("=" * 60)

    results: dict = {"pair": pair_key, "model_a": name_a, "model_b": name_b}

    # Protocol 1: Bitrate Sweep (A compressed, B as reference)
    t0 = time.time()
    logger.info("[Protocol 1] Bitrate Sweep: compressing %s vs reference %s", name_a, name_b)
    sweep = run_bitrate_sweep(
        features_a,
        features_b,
        compression="rank_reduction",
        k=k,
        n_permutations=n_permutations,
        seed=seed,
    )
    sweep.model_name = name_a
    sweep.reference_name = name_b
    results["bitrate_sweep"] = serialize_sweep_result(sweep)
    logger.info("[Protocol 1] Done in %.1fs", time.time() - t0)

    # Also sweep B compressed vs A as reference
    t0 = time.time()
    logger.info("[Protocol 1b] Bitrate Sweep: compressing %s vs reference %s", name_b, name_a)
    sweep_reverse = run_bitrate_sweep(
        features_b,
        features_a,
        compression="rank_reduction",
        k=k,
        n_permutations=n_permutations,
        seed=seed,
    )
    sweep_reverse.model_name = name_b
    sweep_reverse.reference_name = name_a
    results["bitrate_sweep_reverse"] = serialize_sweep_result(sweep_reverse)
    logger.info("[Protocol 1b] Done in %.1fs", time.time() - t0)

    # Protocol 2: Neighborhood-First Alignment
    t0 = time.time()
    logger.info("[Protocol 2] Neighborhood-First Alignment")
    alignment = run_neighborhood_alignment(
        features_a,
        features_b,
        k=k,
        n_permutations=n_permutations,
        seed=seed,
    )
    results["neighborhood_alignment"] = serialize_alignment_result(alignment)
    logger.info("[Protocol 2] Done in %.1fs — %s", time.time() - t0, alignment.interpretation)

    # Protocol 3: Transport Stress Test
    t0 = time.time()
    logger.info("[Protocol 3] Transport Stress Test")
    stress = run_transport_stress_test(
        features_a,
        features_b,
        compression="rank_reduction",
        k=k,
        n_permutations=n_permutations,
        seed=seed,
    )
    stress.model_a_name = name_a
    stress.model_b_name = name_b
    results["transport_stress_test"] = serialize_stress_result(stress)
    logger.info("[Protocol 3] Done in %.1fs", time.time() - t0)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B: Bitrate Sweep Experiment")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--output", type=str, default="data/experiments")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated pairs, e.g. 'modelA,modelB;modelC,modelD'. "
        "If empty, uses DEFAULT_PAIRS.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load features
    features_dir = Path(args.features_dir)
    logger.info("Loading features from %s", features_dir)
    all_features = load_features(features_dir)
    logger.info("Loaded %d models: %s", len(all_features), list(all_features.keys()))

    # Move to device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    for name in all_features:
        all_features[name] = all_features[name].to(device)

    # Determine pairs
    if args.pairs:
        pairs = []
        for pair_str in args.pairs.split(";"):
            a, b = pair_str.strip().split(",")
            pairs.append((a.strip(), b.strip()))
    else:
        pairs = DEFAULT_PAIRS

    # Filter to available features
    available_pairs = []
    for a, b in pairs:
        if a in all_features and b in all_features:
            available_pairs.append((a, b))
        else:
            missing = []
            if a not in all_features:
                missing.append(a)
            if b not in all_features:
                missing.append(b)
            logger.warning("Skipping pair (%s, %s) — missing: %s", a, b, missing)

    logger.info("Running %d pairs", len(available_pairs))

    # Run experiments
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_start = time.time()

    for i, (name_a, name_b) in enumerate(available_pairs):
        logger.info("\n[Pair %d/%d] %s <-> %s", i + 1, len(available_pairs), name_a, name_b)

        result = run_experiment_for_pair(
            name_a,
            name_b,
            all_features[name_a],
            all_features[name_b],
            k=args.k,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        all_results.append(result)

        # Save incrementally
        output_path = output_dir / "bitrate_experiment.json"
        with open(output_path, "w") as f:
            json.dump(
                {
                    "experiment": "Track B: Bitrate Sweep",
                    "config": {
                        "k": args.k,
                        "n_permutations": args.permutations,
                        "seed": args.seed,
                        "device": device,
                        "n_pairs": len(available_pairs),
                    },
                    "results": all_results,
                },
                f,
                indent=2,
            )
        logger.info("Saved progress to %s", output_path)

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("All experiments complete in %.1f minutes", total_time / 60)
    logger.info("Results saved to %s", output_dir / "bitrate_experiment.json")

    # Print summary
    logger.info("\n--- SUMMARY ---")
    for result in all_results:
        pair = result["pair"]
        align = result["neighborhood_alignment"]
        sweep_pts = result["bitrate_sweep"]["points"]
        logger.info(
            "  %s: s_local=%.3f, s_global=%.3f, tau@full=%.3f, tau@0.25=%.3f | %s",
            pair,
            align["s_local"],
            align["s_global"],
            sweep_pts[0]["tau"] if sweep_pts else 0.0,
            next((p["tau"] for p in sweep_pts if p["level"] == 0.25), 0.0),
            align["interpretation"],
        )


if __name__ == "__main__":
    main()
