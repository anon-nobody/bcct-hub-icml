"""Tests for alignment metrics: mutual k-NN, CKA, and Alignment Locality."""

from __future__ import annotations

import pytest
import torch

from latent.metrics.alignment import (
    alignment_locality,
    centered_kernel_alignment,
    mutual_knn_overlap,
    mutual_knn_overlap_fast,
    null_calibrated_mutual_knn,
)


@pytest.fixture()
def identical_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two identical feature matrices — perfect alignment."""
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    return x, x.clone()


@pytest.fixture()
def similar_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two similar feature matrices — high but not perfect alignment."""
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    # Apply a rotation + small noise
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    y = x @ q + 0.05 * torch.randn(200, 32)
    return x, y


@pytest.fixture()
def random_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two unrelated feature matrices — low alignment."""
    torch.manual_seed(0)
    x = torch.randn(200, 32)
    torch.manual_seed(999)
    y = torch.randn(200, 64)
    return x, y


class TestMutualKNNOverlap:
    """Tests for mutual k-NN overlap."""

    def test_identical_features_perfect_overlap(
        self, identical_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = identical_features
        score = mutual_knn_overlap(x, y, k=10)
        assert score == pytest.approx(1.0), "Identical features should have perfect overlap"

    def test_random_features_low_overlap(
        self, random_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = random_features
        score = mutual_knn_overlap(x, y, k=10)
        assert score < 0.3, f"Random features should have low overlap, got {score}"

    def test_symmetry(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        score_xy = mutual_knn_overlap(x, y, k=10)
        score_yx = mutual_knn_overlap(y, x, k=10)
        assert score_xy == pytest.approx(score_yx, abs=1e-6), "Mutual k-NN should be symmetric"

    def test_range_bounds(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        score = mutual_knn_overlap(x, y, k=10)
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"

    def test_different_n_raises(self) -> None:
        x = torch.randn(100, 32)
        y = torch.randn(50, 32)
        with pytest.raises(ValueError, match="same N"):
            mutual_knn_overlap(x, y, k=10)

    def test_k_too_large_raises(self) -> None:
        x = torch.randn(10, 32)
        with pytest.raises(ValueError, match="k="):
            mutual_knn_overlap(x, x, k=10)

    def test_fast_matches_slow(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        slow = mutual_knn_overlap(x, y, k=5)
        fast = mutual_knn_overlap_fast(x, y, k=5)
        assert slow == pytest.approx(fast, abs=1e-6), "Fast and slow should match"


class TestNullCalibratedMutualKNN:
    """Tests for null-calibrated mutual k-NN (S_local)."""

    def test_identical_features_high_zscore(
        self, identical_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = identical_features
        z_score, raw, null_mean = null_calibrated_mutual_knn(x, y, k=10, n_permutations=50, seed=42)
        assert 0.9 <= z_score <= 1.0, (
            f"Identical features should have high calibrated overlap, got {z_score}"
        )
        assert raw > null_mean, "Raw should exceed null mean"

    def test_random_features_low_zscore(
        self, random_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = random_features
        z_score, raw, null_mean = null_calibrated_mutual_knn(x, y, k=10, n_permutations=50, seed=42)
        assert 0.0 <= z_score <= 0.5, (
            f"Random features should have low calibrated overlap, got {z_score}"
        )

    def test_reproducible_with_seed(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = similar_features
        z1, _, _ = null_calibrated_mutual_knn(x, y, k=10, n_permutations=50, seed=42)
        z2, _, _ = null_calibrated_mutual_knn(x, y, k=10, n_permutations=50, seed=42)
        assert z1 == pytest.approx(z2), "Same seed should give same result"


class TestCKA:
    """Tests for Centered Kernel Alignment."""

    def test_identical_features_cka_one(
        self, identical_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        x, y = identical_features
        score = centered_kernel_alignment(x, y)
        assert score == pytest.approx(1.0, abs=1e-5), f"CKA of identical should be 1, got {score}"

    def test_cka_range(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        score = centered_kernel_alignment(x, y)
        assert 0.0 <= score <= 1.0, f"CKA should be in [0, 1], got {score}"

    def test_cka_symmetry(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        score_xy = centered_kernel_alignment(x, y)
        score_yx = centered_kernel_alignment(y, x)
        assert score_xy == pytest.approx(score_yx, abs=1e-6), "CKA should be symmetric"

    def test_different_n_raises(self) -> None:
        x = torch.randn(100, 32)
        y = torch.randn(50, 32)
        with pytest.raises(ValueError, match="same N"):
            centered_kernel_alignment(x, y)


class TestAlignmentLocality:
    """Tests for Alignment Locality (λ)."""

    def test_lambda_computable(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = similar_features
        lambda_val, s_local, s_global = alignment_locality(x, y, k=10, n_permutations=50, seed=42)
        assert isinstance(lambda_val, float)
        assert isinstance(s_local, float)
        assert isinstance(s_global, float)
        assert s_global > 0, "CKA should be positive for similar features"
