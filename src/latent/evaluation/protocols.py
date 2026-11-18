"""BCCT evaluation protocols: Bitrate Sweep, Neighborhood-First Alignment, Transport Stress Test.

Implements the three standardized evaluation protocols defined in the BCCT
framework (see paper §3, Evaluation Protocols and Empirical Results).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from latent.metrics.alignment import (
    centered_kernel_alignment,
    null_calibrated_mutual_knn,
)
from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
from latent.metrics.regime import RegimeThresholds, classify_regime
from latent.metrics.transport import transport_asymmetry_index, transport_linearity_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol 1: Bitrate Sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BitrateSweepPoint:
    """One point on the bitrate sweep curve.

    Attributes:
        level: Compression level identifier (e.g., rank fraction or noise sigma).
        bitrate: Representation Bitrate at this level.
        s_local: Null-calibrated mutual k-NN overlap vs reference.
        tau: Transport Linearity Score vs reference.
        downstream_acc: Downstream accuracy (if evaluated).
    """

    level: float
    bitrate: float
    s_local: float
    tau: float
    downstream_acc: float | None = None


@dataclass
class BitrateSweepResult:
    """Full result of Protocol 1: Bitrate Sweep.

    Attributes:
        points: List of measurement points at each compression level.
        model_name: Name of the model being swept.
        reference_name: Name of the reference encoder.
        compression_method: Method used for compression.
    """

    points: list[BitrateSweepPoint] = field(default_factory=list)  # type: ignore[arg-type]
    model_name: str = ""
    reference_name: str = ""
    compression_method: str = ""


@torch.no_grad()
def _compress_rank_reduction(features: Tensor, fraction: float) -> Tensor:
    """Compress features by projecting onto top-k principal components.

    Args:
        features: Feature matrix, shape (N, D).
        fraction: Fraction of components to retain, in (0, 1].

    Returns:
        Compressed features projected back to original space, shape (N, D).
    """
    centered = features - features.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)  # type: ignore[reportUnknownMemberType]
    k = max(1, int(features.shape[1] * fraction))
    # Reconstruct from top-k components
    reconstructed = u[:, :k] @ torch.diag(s[:k]) @ vh[:k, :]  # type: ignore[reportUnknownVariableType]
    return reconstructed + features.mean(dim=0, keepdim=True)  # type: ignore[reportUnknownVariableType]


@torch.no_grad()
def _compress_noise_injection(features: Tensor, sigma: float, seed: int = 42) -> Tensor:
    """Compress features by adding isotropic Gaussian noise.

    Args:
        features: Feature matrix, shape (N, D).
        sigma: Standard deviation of noise.
        seed: Random seed.

    Returns:
        Noisy features, shape (N, D).
    """
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    noise = torch.randn_like(features, generator=generator) * sigma
    return features + noise


def run_bitrate_sweep(
    features: Tensor,
    reference_features: Tensor,
    compression: str = "rank_reduction",
    levels: list[float] | None = None,
    k: int = 10,
    n_permutations: int = 100,
    seed: int = 42,
) -> BitrateSweepResult:
    """Run Protocol 1: Bitrate Sweep.

    Systematically varies the information bottleneck and measures alignment
    and transport at each level.

    Args:
        features: Model features to compress, shape (N, D).
        reference_features: Reference encoder features, shape (N, D_ref).
        compression: Compression method — "rank_reduction" or "noise_injection".
        levels: Compression levels to sweep. For rank_reduction: fractions in (0, 1].
            For noise_injection: sigma values. If None, uses defaults.
        k: Number of nearest neighbors for alignment.
        n_permutations: Permutations for null calibration (lower = faster).
        seed: Random seed.

    Returns:
        BitrateSweepResult with measurements at each compression level.
    """
    if levels is None:
        if compression == "rank_reduction":
            levels = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]
        elif compression == "noise_injection":
            levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
        else:
            raise ValueError(f"Unknown compression method: {compression}")

    result = BitrateSweepResult(compression_method=compression)

    for level in levels:
        logger.info("Bitrate sweep: %s level=%.4f", compression, level)

        # Compress
        if compression == "rank_reduction":
            compressed = _compress_rank_reduction(features, level)
        elif compression == "noise_injection":
            compressed = _compress_noise_injection(features, level, seed=seed)
        else:
            raise ValueError(f"Unknown compression: {compression}")

        # Measure bitrate
        r = representation_bitrate(compressed).bitrate

        # Measure alignment (use reduced permutations for speed)
        s_local, _, _ = null_calibrated_mutual_knn(
            compressed, reference_features, k=k, n_permutations=n_permutations, seed=seed
        )

        # Measure transport
        transport = transport_linearity_score(compressed, reference_features, seed=seed)

        point = BitrateSweepPoint(
            level=level,
            bitrate=r,
            s_local=s_local,
            tau=transport.tau,
        )
        result.points.append(point)

    return result


# ---------------------------------------------------------------------------
# Protocol 2: Neighborhood-First Alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeighborhoodAlignmentResult:
    """Result of Protocol 2: Neighborhood-First Alignment.

    Attributes:
        s_local: Null-calibrated mutual k-NN overlap.
        s_global: Linear CKA score.
        lambda_val: Alignment Locality = S_local / S_global.
        raw_overlap: Raw mutual k-NN overlap (before null calibration).
        null_mean: Mean of the null distribution.
        interpretation: Human-readable regime interpretation.
    """

    s_local: float
    s_global: float
    lambda_val: float
    raw_overlap: float
    null_mean: float
    interpretation: str


def run_neighborhood_alignment(
    features_a: Tensor,
    features_b: Tensor,
    k: int = 10,
    n_permutations: int = 1000,
    seed: int = 42,
) -> NeighborhoodAlignmentResult:
    """Run Protocol 2: Neighborhood-First Alignment.

    Computes local (null-calibrated mutual k-NN) and global (CKA) alignment,
    then reports their ratio λ to distinguish Aristotelian from Platonic convergence.

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        k: Number of nearest neighbors.
        n_permutations: Permutations for null calibration.
        seed: Random seed.

    Returns:
        NeighborhoodAlignmentResult with S_local, S_global, λ, and interpretation.
    """
    # Step 1: Local alignment
    s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
        features_a, features_b, k=k, n_permutations=n_permutations, seed=seed
    )

    # Step 2: Global alignment
    s_global = centered_kernel_alignment(features_a, features_b)

    # Step 3: Alignment Locality
    lambda_val = (float("inf") if s_local > 0 else 0.0) if s_global < 1e-10 else s_local / s_global

    # Interpretation guide for human-readable summary.
    # Thresholds are on the calibrated overlap [0, 1] scale, consistent
    # with the regime classification in regime.py (s_local_thresh=0.05).
    if s_local < 0.05 and s_global > 0.7:
        interpretation = "Inflated: CKA confounded by scale; local alignment not significant."
    elif s_local < 0.05:
        interpretation = "No meaningful alignment (Collapsed or Divergent regime)."
    elif s_local >= 0.2 and s_global < 0.5:
        interpretation = "Local-Only (Aristotelian): neighborhoods align, global geometry diverges."
    elif s_local >= 0.2 and s_global >= 0.7:
        interpretation = "Convergent (Platonic): full alignment; linear transport likely succeeds."
    else:
        interpretation = "Moderate alignment; further analysis recommended."

    return NeighborhoodAlignmentResult(
        s_local=s_local,
        s_global=s_global,
        lambda_val=lambda_val,
        raw_overlap=raw_overlap,
        null_mean=null_mean,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Protocol 3: Transport Stress Test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StressTestPoint:
    """One point in the Transport Stress Test.

    Attributes:
        delta: Bottleneck Mismatch at this compression level.
        tau_a_to_b: Transport Linearity Score A→B.
        tau_b_to_a: Transport Linearity Score B→A.
        tai: Transport Asymmetry Index.
        compression_level: The compression level applied to model A.
        regime: Classified convergence regime.
    """

    delta: float
    tau_a_to_b: float
    tau_b_to_a: float
    tai: float
    compression_level: float
    regime: str


@dataclass
class StressTestResult:
    """Full result of Protocol 3: Transport Stress Test.

    Attributes:
        points: Measurements at each compression level.
        model_a_name: Name of model A (compressed).
        model_b_name: Name of model B (reference, uncompressed).
    """

    points: list[StressTestPoint] = field(default_factory=list)  # type: ignore[arg-type]
    model_a_name: str = ""
    model_b_name: str = ""


def run_transport_stress_test(
    features_a: Tensor,
    features_b: Tensor,
    compression: str = "rank_reduction",
    levels: list[float] | None = None,
    k: int = 10,
    n_permutations: int = 100,
    seed: int = 42,
    thresholds: RegimeThresholds | None = None,
) -> StressTestResult:
    """Run Protocol 3: Transport Stress Test.

    Compresses model A at multiple levels while keeping model B at native bitrate,
    then evaluates bidirectional transport and TAI at each level.

    Args:
        features_a: Model A features (will be compressed), shape (N, D_a).
        features_b: Model B features (kept intact), shape (N, D_b).
        compression: Compression method.
        levels: Compression levels for model A.
        k: Number of nearest neighbors for alignment.
        n_permutations: Permutations for null calibration.
        seed: Random seed.
        thresholds: Custom regime thresholds.

    Returns:
        StressTestResult with measurements at each compression level.
    """
    if levels is None:
        if compression == "rank_reduction":
            levels = [1.0, 0.75, 0.5, 0.25, 0.125]
        else:
            levels = [0.0, 0.1, 0.3, 0.5, 1.0]

    result = StressTestResult()

    for level in levels:
        logger.info("Stress test: %s level=%.4f", compression, level)

        # Compress model A
        if compression == "rank_reduction":
            compressed_a = _compress_rank_reduction(features_a, level)
        elif compression == "noise_injection":
            compressed_a = _compress_noise_injection(features_a, level, seed=seed)
        else:
            raise ValueError(f"Unknown compression: {compression}")

        # Compute Δ
        delta = bottleneck_mismatch(compressed_a, features_b)

        # Bidirectional transport
        bi_result = transport_asymmetry_index(compressed_a, features_b, seed=seed)

        # Alignment for regime classification
        s_local, _, _ = null_calibrated_mutual_knn(
            compressed_a, features_b, k=k, n_permutations=n_permutations, seed=seed
        )

        # Bitrate for regime classification
        r = representation_bitrate(compressed_a).bitrate

        # Classify regime using conservative bidirectional tau (min of directions),
        # consistent with the atlas/scorecard definition.
        tau_bidirectional = min(bi_result.a_to_b.tau, bi_result.b_to_a.tau)
        regime = classify_regime(
            bitrate=r,
            s_local=s_local,
            tau=tau_bidirectional,
            thresholds=thresholds,
        )

        point = StressTestPoint(
            delta=delta,
            tau_a_to_b=bi_result.a_to_b.tau,
            tau_b_to_a=bi_result.b_to_a.tau,
            tai=bi_result.tai,
            compression_level=level,
            regime=regime.regime.value,
        )
        result.points.append(point)

    return result
