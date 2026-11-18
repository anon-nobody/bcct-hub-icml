from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnnecessaryIsInstance=false
import json
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import torch

from latent.metrics.bitrate import BitrateMethod
from latent.metrics.regime import classify_regime

logger = logging.getLogger(__name__)

ATLAS_PATH = Path("data/atlas.json")
FEATURES_DIR = Path("data/features")
OUTPUT_PATH = Path("data/bitrate_estimator_robustness.json")
MAX_SAMPLES = 1024

ESTIMATORS = [
    BitrateMethod.EFFECTIVE_RANK,
    BitrateMethod.PARTICIPATION_RATIO,
    BitrateMethod.GINI_SPECTRUM,
    BitrateMethod.SPECTRAL_ENTROPY,
]


def _bitrates_from_spectrum(features: torch.Tensor) -> dict[str, float]:
    centered = features - features.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(centered)

    sigma = sigma[sigma > 1e-12]
    if sigma.numel() == 0:
        return {method.value: 0.0 for method in ESTIMATORS}

    eigvals = sigma.square()
    eig_total = float(eigvals.sum().item())
    if eig_total <= 0.0:
        return {method.value: 0.0 for method in ESTIMATORS}

    p = eigvals / eig_total
    p = p[p > 1e-12]

    entropy = float((-(p * torch.log(p)).sum()).item())
    effective_rank = max(1.0, math.exp(entropy))
    effective_rank_bitrate = max(0.0, math.log2(effective_rank))

    sigma_norm = sigma / sigma.max().clamp(min=1e-12)
    pr_num = float(sigma_norm.sum().square().item())
    pr_den = float(sigma_norm.square().sum().clamp(min=1e-12).item())
    participation_rank = max(1.0, pr_num / pr_den)
    participation_bitrate = max(0.0, math.log2(participation_rank))

    sigma_sorted, _ = torch.sort(sigma)
    n = int(sigma_sorted.shape[0])
    idx = torch.arange(1, n + 1, device=sigma_sorted.device, dtype=sigma_sorted.dtype)
    weighted_sum = float((idx * sigma_sorted).sum().item())
    total = float(sigma_sorted.sum().clamp(min=1e-12).item())
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
    gini = min(1.0, max(0.0, gini))
    gini_rank = max(1.0, 1.0 / (1.0 - gini + 1e-10))
    gini_bitrate = max(0.0, math.log2(gini_rank))

    renyi2 = float((-torch.log((p.square()).sum().clamp(min=1e-12))).item())
    spectral_rank = max(1.0, math.exp(renyi2))
    spectral_bitrate = max(0.0, math.log2(spectral_rank))

    return {
        BitrateMethod.EFFECTIVE_RANK.value: effective_rank_bitrate,
        BitrateMethod.PARTICIPATION_RATIO.value: participation_bitrate,
        BitrateMethod.GINI_SPECTRUM.value: gini_bitrate,
        BitrateMethod.SPECTRAL_ENTROPY.value: spectral_bitrate,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data)!r}")
    return data


def _compute_model_bitrates(model_names: list[str]) -> dict[str, dict[str, float]]:
    model_bitrates: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        feature_path = FEATURES_DIR / f"{model_name}.pt"
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found for model '{model_name}': {feature_path}"
            )

        payload = cast(
            "dict[str, Any]", torch.load(feature_path, map_location="cpu", weights_only=True)
        )
        if not isinstance(payload, dict) or "features" not in payload:
            raise ValueError(f"Invalid feature payload in {feature_path}")

        features = payload["features"]
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected Tensor in {feature_path}['features'], got {type(features)!r}"
            )
        if features.shape[0] > MAX_SAMPLES:
            features = features[:MAX_SAMPLES]

        model_bitrates[model_name] = _bitrates_from_spectrum(features)

        logger.info("Computed bitrates for %s", model_name)

    return model_bitrates


def _parse_pair_key(pair_key: str) -> tuple[str, str]:
    parts = pair_key.split(" <-> ")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair key format: {pair_key}")
    return parts[0], parts[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    atlas = _load_json(ATLAS_PATH)
    models_raw = atlas.get("models")
    pairwise_raw = atlas.get("pairwise")
    if not isinstance(models_raw, dict) or not isinstance(pairwise_raw, dict):
        raise ValueError("atlas.json must contain 'models' and 'pairwise' dictionaries")

    model_names = sorted(cast("dict[str, Any]", models_raw).keys())
    model_bitrates = _compute_model_bitrates(model_names)

    regimes_by_method: dict[str, dict[str, str]] = {method.value: {} for method in ESTIMATORS}
    regime_counts: dict[str, dict[str, int]] = {method.value: {} for method in ESTIMATORS}

    for pair_key, metrics_raw in cast("dict[str, Any]", pairwise_raw).items():
        if not isinstance(pair_key, str) or not isinstance(metrics_raw, dict):
            continue

        model_a, model_b = _parse_pair_key(pair_key)
        if model_a not in model_bitrates or model_b not in model_bitrates:
            raise KeyError(f"Missing bitrate entry for pair {pair_key}")

        s_local = float(metrics_raw["s_local"])
        tau = float(metrics_raw["tau"])
        s_global = float(metrics_raw.get("s_global", 0.0))
        lambda_val = float(metrics_raw["lambda"]) if "lambda" in metrics_raw else None

        for method in ESTIMATORS:
            method_key = method.value
            bitrate_avg = (
                model_bitrates[model_a][method_key] + model_bitrates[model_b][method_key]
            ) / 2.0
            regime = classify_regime(
                bitrate=bitrate_avg,
                s_local=s_local,
                tau=tau,
                s_global=s_global,
                lambda_val=lambda_val,
            ).regime.value
            regimes_by_method[method_key][pair_key] = regime
            regime_counts[method_key][regime] = regime_counts[method_key].get(regime, 0) + 1

    pairwise_agreement: dict[str, dict[str, object]] = {}
    method_keys = [method.value for method in ESTIMATORS]
    for left, right in combinations(method_keys, 2):
        flips: list[dict[str, str]] = []
        same = 0
        total = 0

        for pair_key in sorted(regimes_by_method[left].keys()):
            left_regime = regimes_by_method[left][pair_key]
            right_regime = regimes_by_method[right][pair_key]
            total += 1
            if left_regime == right_regime:
                same += 1
            else:
                flips.append(
                    {
                        "pair": pair_key,
                        left: left_regime,
                        right: right_regime,
                    }
                )

        agreement_rate = 100.0 * same / total if total > 0 else 0.0
        key = f"{left} vs {right}"
        pairwise_agreement[key] = {
            "agreement_rate_pct": round(agreement_rate, 2),
            "agree_count": same,
            "total_pairs": total,
            "flip_count": total - same,
            "flips": flips,
        }

    results = {
        "estimators": method_keys,
        "n_models": len(model_names),
        "n_pairs": len(pairwise_raw),
        "model_bitrates": model_bitrates,
        "regime_counts": regime_counts,
        "pairwise_agreement": pairwise_agreement,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as handle:
        json.dump(results, handle, indent=2)

    logger.info("Saved robustness analysis to %s", OUTPUT_PATH)
    for method in method_keys:
        logger.info("Regime counts (%s): %s", method, regime_counts[method])
    for key, stats in pairwise_agreement.items():
        agreement = stats["agreement_rate_pct"]
        logger.info("%s agreement: %.2f%%", key, agreement)


if __name__ == "__main__":
    main()
