"""Bitrate metrics: Representation Bitrate (R), Effective Rank, and Bottleneck Mismatch (Δ).

Implements §2.1 and §2.6 of the BCCT framework. Provides multiple proxy
measurements for representation information density.
"""

from __future__ import annotations

# pyright: reportUnreachable=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import logging
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class BitrateMethod(Enum):
    """Available methods for estimating Representation Bitrate.

    Notes:
        ``RATE_DISTORTION`` is currently a *placeholder*: it logs a warning
        and falls back to ``EFFECTIVE_RANK``. A full empirical R(D) sweep
        is listed as open problem OP1 (training-time bitrate intervention)
        and is not exposed by the released scorecard. All paper numbers
        use ``EFFECTIVE_RANK``; the other four entries are validated
        proxies (see ``data/bitrate_estimator_robustness.json``).
    """

    EFFECTIVE_RANK = "effective_rank"
    PARTICIPATION_RATIO = "participation_ratio"
    GINI_SPECTRUM = "gini_spectrum"
    SPECTRAL_ENTROPY = "spectral_entropy"
    RATE_DISTORTION = "rate_distortion"  # placeholder; falls back to EFFECTIVE_RANK


@dataclass(frozen=True)
class BitrateResult:
    """Result of bitrate estimation.

    Attributes:
        bitrate: Estimated Representation Bitrate R (bits/dim).
        effective_rank_value: Effective rank of the latent covariance.
        dimensionality: Original latent dimensionality d.
        method: Method used for estimation.
    """

    bitrate: float
    effective_rank_value: float
    dimensionality: int
    method: BitrateMethod


@torch.no_grad()
def effective_rank(features: Tensor) -> float:
    """Compute effective rank of the feature covariance matrix.

    Effective rank = exp(H(σ₁, ..., σ_d)) where H is Shannon entropy
    of the normalized singular values, following Roy & Vetterli (2007).

    A low effective rank relative to d indicates information concentration
    in few dimensions — lower effective bitrate.

    Args:
        features: Feature matrix, shape (N, D).

    Returns:
        Effective rank in [1, D].
    """
    # Center features
    centered = features - features.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(centered)  # type: ignore[reportUnknownMemberType]
    eigvals = s.square()

    # Normalize to a probability distribution
    eig_total = eigvals.sum()  # type: ignore[reportUnknownMemberType]
    s_norm = eigvals / eig_total  # type: ignore[reportUnknownVariableType]

    # Remove zeros to avoid log(0)
    s_norm = s_norm[s_norm > 1e-12]  # type: ignore[reportUnknownVariableType]

    # Shannon entropy
    entropy = -(s_norm * torch.log(s_norm)).sum()  # type: ignore[reportUnknownMemberType]

    return float(torch.exp(entropy).item())  # type: ignore[reportUnknownArgumentType]


@torch.no_grad()
def _singular_values(features: Tensor) -> Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(centered)  # type: ignore[reportUnknownMemberType]


@torch.no_grad()
def participation_ratio(features: Tensor) -> float:
    sigma = _singular_values(features)
    sigma = sigma[sigma > 1e-12]
    if sigma.numel() == 0:
        return 1.0

    sigma = sigma / sigma.max().clamp(min=1e-12)
    numerator = sigma.sum().square()
    denominator = sigma.square().sum().clamp(min=1e-12)
    return float((numerator / denominator).item())


@torch.no_grad()
def gini_spectrum(features: Tensor) -> float:
    sigma = _singular_values(features)
    sigma = sigma[sigma > 1e-12]
    if sigma.numel() == 0:
        return 0.0

    sigma_sorted, _ = torch.sort(sigma)
    n = sigma_sorted.shape[0]
    idx = torch.arange(1, n + 1, device=sigma_sorted.device, dtype=sigma_sorted.dtype)
    weighted_sum = (idx * sigma_sorted).sum()
    total = sigma_sorted.sum().clamp(min=1e-12)
    gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
    return float(torch.clamp(gini, min=0.0, max=1.0).item())


@torch.no_grad()
def spectral_entropy_rank(features: Tensor) -> float:
    sigma = _singular_values(features)
    eigvals = sigma.square()
    eig_total = eigvals.sum().clamp(min=1e-12)
    p = eigvals / eig_total
    p = p[p > 1e-12]
    if p.numel() == 0:
        return 1.0

    renyi2 = -torch.log((p.square()).sum().clamp(min=1e-12))
    return float(torch.exp(renyi2).item())


@torch.no_grad()
def representation_bitrate(
    features: Tensor,
    method: BitrateMethod = BitrateMethod.EFFECTIVE_RANK,
    quantization_bits: list[int] | None = None,
) -> BitrateResult:
    """Estimate Representation Bitrate R of a feature set.

    R = I(X; Z) / d, estimated via effective rank proxy:
    R_proxy = log2(erank) — the log2 of effective rank provides an
    estimate of the number of bits per dimension needed to represent
    the information content.

    Args:
        features: Feature matrix, shape (N, D).
        method: Estimation method to use.
        quantization_bits: For RATE_DISTORTION method, list of bit levels
            to sweep (e.g., [32, 16, 8, 4, 2]).

    Returns:
        BitrateResult with estimated R and auxiliary information.
    """
    d = features.shape[1]
    erank = effective_rank(features)

    if method == BitrateMethod.EFFECTIVE_RANK:
        # R_proxy ≈ log2(erank) / d * d = log2(erank) bits spread over d dims
        # Normalized: bits per dimension ≈ log2(erank)
        bitrate = max(0.0, torch.tensor(erank).log2().item())

        return BitrateResult(
            bitrate=bitrate,
            effective_rank_value=erank,
            dimensionality=d,
            method=method,
        )

    elif method == BitrateMethod.PARTICIPATION_RATIO:
        pr = participation_ratio(features)
        bitrate = max(0.0, torch.tensor(pr).log2().item())

        return BitrateResult(
            bitrate=bitrate,
            effective_rank_value=pr,
            dimensionality=d,
            method=method,
        )

    elif method == BitrateMethod.GINI_SPECTRUM:
        gini = gini_spectrum(features)
        bitrate = max(0.0, torch.log2(torch.tensor(max(1.0, 1.0 / (1.0 - gini + 1e-10)))).item())

        return BitrateResult(
            bitrate=bitrate,
            effective_rank_value=max(1.0, 1.0 / (1.0 - gini + 1e-10)),
            dimensionality=d,
            method=method,
        )

    elif method == BitrateMethod.SPECTRAL_ENTROPY:
        renyi_rank = spectral_entropy_rank(features)
        bitrate = max(0.0, torch.tensor(renyi_rank).log2().item())

        return BitrateResult(
            bitrate=bitrate,
            effective_rank_value=renyi_rank,
            dimensionality=d,
            method=method,
        )

    elif method == BitrateMethod.RATE_DISTORTION:
        # NOTE: Full rate-distortion sweep is not yet implemented.
        # Falls back to effective-rank bitrate as a proxy.
        # A proper implementation would quantize features at multiple bit levels
        # and trace an empirical R(D) curve.
        logger.warning(
            "RATE_DISTORTION method is not fully implemented; "
            "falling back to EFFECTIVE_RANK proxy."
        )
        bitrate = max(0.0, torch.tensor(erank).log2().item())

        return BitrateResult(
            bitrate=bitrate,
            effective_rank_value=erank,
            dimensionality=d,
            method=method,
        )

    raise ValueError(f"Unknown bitrate method: {method}")  # pyright: ignore[reportUnreachable]


@torch.no_grad()
def bottleneck_mismatch(
    features_a: Tensor,
    features_b: Tensor,
    method: BitrateMethod = BitrateMethod.EFFECTIVE_RANK,
) -> float:
    """Compute Bottleneck Mismatch Δ(A, B) = |R(A) - R(B)|.

    Measures the difference in information capacity between two encoders.
    Higher Δ predicts harder transport (Hypothesis P1).

    Args:
        features_a: First feature matrix, shape (N, D_a).
        features_b: Second feature matrix, shape (N, D_b).
        method: Bitrate estimation method.

    Returns:
        Δ in bits/dim (non-negative).
    """
    r_a = representation_bitrate(features_a, method=method)
    r_b = representation_bitrate(features_b, method=method)

    return abs(r_a.bitrate - r_b.bitrate)
