"""Tests for transport metrics: Transport Linearity Score (τ) and TAI."""

from __future__ import annotations

import pytest
import torch

from latent.metrics.transport import (
    BidirectionalTransportResult,
    TransportResult,
    transport_asymmetry_index,
    transport_linearity_score,
)

# Use small dimensions and sample counts for fast CPU tests.
_N = 150
_D = 16
_MLP_STEPS = 200


@pytest.fixture()
def linearly_related() -> tuple[torch.Tensor, torch.Tensor]:
    """Features related by a known linear map — τ should be ~1."""
    torch.manual_seed(0)
    x = torch.randn(_N, _D)
    q, _ = torch.linalg.qr(torch.randn(_D, _D))
    y = x @ q
    return x, y


@pytest.fixture()
def nonlinearly_related() -> tuple[torch.Tensor, torch.Tensor]:
    """Features related by a nonlinear map — τ should be < 1."""
    torch.manual_seed(0)
    x = torch.randn(_N, _D)
    torch.manual_seed(1)
    w1 = torch.randn(_D, _D)
    w2 = torch.randn(_D, _D)
    y = torch.tanh(x @ w1) + 0.5 * x @ w2
    return x, y


@pytest.fixture()
def asymmetric_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Features with information asymmetry — high TAI expected."""
    torch.manual_seed(0)
    a = torch.randn(_N, _D)
    b_core = a[:, :4]
    b_noise = torch.randn(_N, _D - 4) * 0.01
    b = torch.cat([b_core, b_noise], dim=1)
    return a, b


class TestTransportLinearityScore:
    """Tests for τ computation."""

    def test_linear_relation_high_tau(
        self, linearly_related: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = linearly_related
        result = transport_linearity_score(x, y, mlp_steps=_MLP_STEPS, seed=42)
        assert isinstance(result, TransportResult)
        assert result.tau > 0.8, f"Linearly related features should have high τ, got {result.tau}"

    def test_nonlinear_relation_lower_tau(
        self, nonlinearly_related: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = nonlinearly_related
        result = transport_linearity_score(x, y, mlp_steps=_MLP_STEPS, seed=42)
        assert result.tau < 0.99, f"Nonlinearly related should have lower τ, got {result.tau}"

    def test_tau_range(self, linearly_related: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = linearly_related
        result = transport_linearity_score(x, y, mlp_steps=100, seed=42)
        assert 0.0 <= result.tau <= 1.0, f"τ should be in [0, 1], got {result.tau}"

    def test_perf_best_geq_perf_linear(
        self, nonlinearly_related: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = nonlinearly_related
        result = transport_linearity_score(x, y, mlp_steps=_MLP_STEPS, seed=42)
        # Allow small tolerance for noise
        assert result.perf_best >= result.perf_linear - 0.05, (
            f"MLP should be at least as good as linear: "
            f"linear={result.perf_linear}, best={result.perf_best}"
        )

    def test_different_n_raises(self) -> None:
        x = torch.randn(100, _D)
        y = torch.randn(50, _D)
        with pytest.raises(ValueError, match="same N"):
            transport_linearity_score(x, y)

    def test_reproducible_with_seed(
        self, linearly_related: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = linearly_related
        r1 = transport_linearity_score(x, y, mlp_steps=100, seed=42)
        r2 = transport_linearity_score(x, y, mlp_steps=100, seed=42)
        assert r1.tau == pytest.approx(r2.tau, abs=1e-4), "Same seed should give same τ"


    def test_high_dim_transport(self) -> None:
        """Per-dim R² should yield meaningful perf_linear even for high-dim mismatched pairs."""
        torch.manual_seed(42)
        n = 500
        d_src, d_tgt = 768, 1024
        x = torch.randn(n, d_src)
        # Linear map from d_src to d_tgt
        w_true = torch.randn(d_src, d_tgt) * 0.1
        y = x @ w_true + torch.randn(n, d_tgt) * 0.01
        result = transport_linearity_score(x, y, mlp_steps=500, seed=42)
        # With per-dim R², Procrustes should capture the linear relationship
        assert result.perf_linear > 0.01, (
            f"High-dim linear pair should have non-trivial perf_linear, got {result.perf_linear}"
        )

    def test_degenerate_random_low_tau(self) -> None:
        """Completely unrelated random features should yield τ ≈ 0."""
        torch.manual_seed(0)
        x = torch.randn(200, 64)
        torch.manual_seed(999)
        y = torch.randn(200, 64)
        result = transport_linearity_score(x, y, mlp_steps=300, seed=42)
        assert result.perf_linear < 0.15, (
            f"Random perf_linear should be low, got {result.perf_linear}"
        )
        assert result.perf_best < 0.15, (
            f"Random perf_best should be low, got {result.perf_best}"
        )

    def test_converged_field(self, linearly_related: tuple[torch.Tensor, torch.Tensor]) -> None:
        """TransportResult should have a converged field."""
        x, y = linearly_related
        result = transport_linearity_score(x, y, mlp_steps=_MLP_STEPS, seed=42)
        assert isinstance(result.converged, bool)


class TestTransportAsymmetryIndex:
    """Tests for TAI computation."""

    def test_symmetric_pair_low_tai(
        self, linearly_related: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = linearly_related
        result = transport_asymmetry_index(x, y, mlp_steps=_MLP_STEPS, seed=42)
        assert isinstance(result, BidirectionalTransportResult)
        assert result.tai < 0.3, f"Symmetric (rotated) pair should have low TAI, got {result.tai}"

    def test_tai_range(self, asymmetric_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        a, b = asymmetric_pair
        result = transport_asymmetry_index(a, b, mlp_steps=_MLP_STEPS, seed=42)
        assert 0.0 <= result.tai <= 1.0, f"TAI should be in [0, 1], got {result.tai}"

    def test_tai_direction_exists(self, asymmetric_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        a, b = asymmetric_pair
        result = transport_asymmetry_index(a, b, mlp_steps=_MLP_STEPS, seed=42)
        assert result.asymmetry_direction in {"A→B", "B→A"}
