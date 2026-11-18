"""Tests for bitrate metrics: Representation Bitrate (R), effective rank, and Δ."""

from __future__ import annotations

import pytest
import torch

from latent.metrics.bitrate import (
    BitrateMethod,
    BitrateResult,
    bottleneck_mismatch,
    effective_rank,
    representation_bitrate,
)


@pytest.fixture()
def full_rank_features() -> torch.Tensor:
    """Features with high effective rank (all dimensions informative)."""
    torch.manual_seed(0)
    return torch.randn(500, 32)


@pytest.fixture()
def low_rank_features() -> torch.Tensor:
    """Features with low effective rank (most variance in few dims)."""
    torch.manual_seed(0)
    # Generate data that lives on a 4-dimensional subspace of R^32
    basis = torch.randn(500, 4)
    projection = torch.randn(4, 32)
    return basis @ projection + 0.01 * torch.randn(500, 32)


class TestEffectiveRank:
    """Tests for effective rank computation."""

    def test_full_rank_high(self, full_rank_features: torch.Tensor) -> None:
        erank = effective_rank(full_rank_features)
        d = full_rank_features.shape[1]
        assert erank > d * 0.5, f"Full-rank data should have high erank, got {erank}/{d}"

    def test_low_rank_low(self, low_rank_features: torch.Tensor) -> None:
        erank = effective_rank(low_rank_features)
        assert erank < 10, f"Low-rank (4D subspace) should have low erank, got {erank}"

    def test_range(self, full_rank_features: torch.Tensor) -> None:
        erank = effective_rank(full_rank_features)
        d = full_rank_features.shape[1]
        assert 1.0 <= erank <= d, f"Effective rank should be in [1, {d}], got {erank}"

    def test_single_dim_erank_one(self) -> None:
        # All data along a single direction
        x = torch.randn(100, 1).expand(-1, 32)  # all columns identical
        erank = effective_rank(x)
        assert erank == pytest.approx(1.0, abs=0.1), (
            f"Single-dim data should have erank~1, got {erank}"
        )


class TestRepresentationBitrate:
    """Tests for Representation Bitrate estimation."""

    def test_returns_result(self, full_rank_features: torch.Tensor) -> None:
        result = representation_bitrate(full_rank_features)
        assert isinstance(result, BitrateResult)
        assert result.bitrate > 0
        assert result.dimensionality == 32
        assert result.method == BitrateMethod.EFFECTIVE_RANK

    def test_full_rank_higher_bitrate(
        self,
        full_rank_features: torch.Tensor,
        low_rank_features: torch.Tensor,
    ) -> None:
        r_full = representation_bitrate(full_rank_features)
        r_low = representation_bitrate(low_rank_features)
        assert r_full.bitrate > r_low.bitrate, (
            f"Full-rank should have higher R: {r_full.bitrate} vs {r_low.bitrate}"
        )

    def test_rate_distortion_method(self, full_rank_features: torch.Tensor) -> None:
        result = representation_bitrate(full_rank_features, method=BitrateMethod.RATE_DISTORTION)
        assert result.method == BitrateMethod.RATE_DISTORTION
        assert result.bitrate > 0


class TestBottleneckMismatch:
    """Tests for Bottleneck Mismatch (Δ)."""

    def test_same_features_zero_mismatch(self, full_rank_features: torch.Tensor) -> None:
        delta = bottleneck_mismatch(full_rank_features, full_rank_features)
        assert delta == pytest.approx(0.0, abs=1e-6), f"Same features should have Δ=0, got {delta}"

    def test_different_rank_positive_mismatch(
        self,
        full_rank_features: torch.Tensor,
        low_rank_features: torch.Tensor,
    ) -> None:
        delta = bottleneck_mismatch(full_rank_features, low_rank_features)
        assert delta > 0, f"Different-rank features should have Δ>0, got {delta}"

    def test_symmetry(
        self,
        full_rank_features: torch.Tensor,
        low_rank_features: torch.Tensor,
    ) -> None:
        delta_ab = bottleneck_mismatch(full_rank_features, low_rank_features)
        delta_ba = bottleneck_mismatch(low_rank_features, full_rank_features)
        assert delta_ab == pytest.approx(delta_ba, abs=1e-6), "Δ should be symmetric"

    def test_non_negative(
        self,
        full_rank_features: torch.Tensor,
        low_rank_features: torch.Tensor,
    ) -> None:
        delta = bottleneck_mismatch(full_rank_features, low_rank_features)
        assert delta >= 0, f"Δ should be non-negative, got {delta}"
