"""Integration tests for the full BCCT evaluation pipeline.

Tests end-to-end flow: features → metrics → regime → protocols → scorecard.
Uses synthetic features to ensure tests run without GPU or pretrained models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from latent.evaluation.protocols import (
    run_bitrate_sweep,
    run_neighborhood_alignment,
    run_transport_stress_test,
)
from latent.evaluation.scorecard import ScorecardEntry, scorecard_to_markdown
from latent.metrics.alignment import (
    centered_kernel_alignment,
    mutual_knn_overlap,
    null_calibrated_mutual_knn,
)
from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
from latent.metrics.regime import classify_regime
from latent.metrics.transport import transport_asymmetry_index, transport_linearity_score

# ---------------------------------------------------------------------------
# Fixtures: synthetic features mimicking real model outputs
# ---------------------------------------------------------------------------


@pytest.fixture()
def similar_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two feature sets that are similar (shared structure + noise)."""
    torch.manual_seed(42)
    n, d = 200, 64
    shared = torch.randn(n, d)
    f_a = shared + 0.1 * torch.randn(n, d)
    f_b = shared @ torch.randn(d, d) * 0.3 + 0.1 * torch.randn(n, d)
    return f_a, f_b


@pytest.fixture()
def dissimilar_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two completely independent feature sets."""
    torch.manual_seed(123)
    n = 200
    f_a = torch.randn(n, 64)
    f_b = torch.randn(n, 128)  # different dimensionality
    return f_a, f_b


@pytest.fixture()
def identical_features() -> tuple[torch.Tensor, torch.Tensor]:
    """Two identical feature sets (linear transform of each other)."""
    torch.manual_seed(99)
    n, d = 1000, 8
    f_a = torch.randn(n, d)
    # Well-conditioned scaling + small rotation (easy for linear map)
    w = torch.eye(d) * 2.0 + 0.1 * torch.randn(d, d)
    f_b = f_a @ w + 0.01 * torch.randn(n, d)
    return f_a, f_b


# ---------------------------------------------------------------------------
# Test 1: Full metric pipeline for a single pair
# ---------------------------------------------------------------------------


class TestFullMetricPipeline:
    """Test computing all BCCT metrics on a single pair."""

    def test_all_metrics_computable(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """All BCCT metrics should be computable on valid features."""
        f_a, f_b = similar_features

        # Alignment metrics
        s_local_raw = mutual_knn_overlap(f_a, f_b, k=5)
        s_global = centered_kernel_alignment(f_a, f_b)
        s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
            f_a, f_b, k=5, n_permutations=20, seed=42
        )

        # Transport metrics
        transport = transport_linearity_score(f_a, f_b, seed=42)
        tai_result = transport_asymmetry_index(f_a, f_b, seed=42)

        # Bitrate metrics
        r_a = representation_bitrate(f_a)
        r_b = representation_bitrate(f_b)
        delta = bottleneck_mismatch(f_a, f_b)

        # Regime classification
        regime = classify_regime(
            bitrate=r_a.bitrate,
            s_local=s_local,
            tau=transport.tau,
        )

        # All values should be finite
        assert 0 <= s_local_raw <= 1
        assert 0 <= s_global <= 1
        assert isinstance(s_local, float)
        assert isinstance(raw_overlap, float)
        assert isinstance(null_mean, float)
        assert 0 <= transport.tau <= 1
        assert 0 <= tai_result.tai <= 1
        assert r_a.bitrate > 0
        assert r_b.bitrate > 0
        assert delta >= 0
        assert regime.regime is not None

    def test_similar_features_higher_alignment(
        self,
        similar_features: tuple[torch.Tensor, torch.Tensor],
        dissimilar_features: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Similar features should have higher alignment than dissimilar ones."""
        f_sim_a, f_sim_b = similar_features
        f_dis_a, f_dis_b = dissimilar_features

        s_sim = mutual_knn_overlap(f_sim_a, f_sim_b, k=5)
        s_dis = mutual_knn_overlap(f_dis_a, f_dis_b, k=5)

        assert s_sim > s_dis, f"Similar: {s_sim:.3f}, Dissimilar: {s_dis:.3f}"

    def test_identical_features_perfect_transport(
        self, identical_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Near-linearly related features should have tau > 0.5."""
        f_a, f_b = identical_features
        transport = transport_linearity_score(f_a, f_b, seed=42)
        assert transport.tau > 0.5, (
            f"Expected tau > 0.5 for near-linear relation, got {transport.tau:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 2: Protocol 1 — Bitrate Sweep
# ---------------------------------------------------------------------------


class TestBitrateSweepProtocol:
    """Test the full Bitrate Sweep protocol."""

    def test_bitrate_sweep_runs(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Protocol 1 should complete without errors."""
        f_a, f_b = similar_features
        result = run_bitrate_sweep(
            f_a,
            f_b,
            compression="rank_reduction",
            levels=[1.0, 0.5, 0.25],
            k=5,
            n_permutations=10,
            seed=42,
        )
        assert len(result.points) == 3
        # Bitrate should decrease with compression
        bitrates = [p.bitrate for p in result.points]
        assert bitrates[0] > bitrates[-1], f"Expected decreasing bitrate, got {bitrates}"

    def test_bitrate_sweep_noise_injection(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Noise injection compression should also work."""
        f_a, f_b = similar_features
        result = run_bitrate_sweep(
            f_a,
            f_b,
            compression="noise_injection",
            levels=[0.0, 0.5, 1.0],
            k=5,
            n_permutations=10,
            seed=42,
        )
        assert len(result.points) == 3

    def test_alignment_degrades_with_compression(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Alignment should generally degrade as we compress more aggressively."""
        f_a, f_b = similar_features
        result = run_bitrate_sweep(
            f_a,
            f_b,
            compression="rank_reduction",
            levels=[1.0, 0.25, 0.0625],
            k=5,
            n_permutations=10,
            seed=42,
        )
        s_locals = [p.s_local for p in result.points]
        # Full rank should have >= alignment than extreme compression
        assert s_locals[0] >= s_locals[-1] - 0.05, f"Expected alignment to degrade, got {s_locals}"


# ---------------------------------------------------------------------------
# Test 3: Protocol 2 — Neighborhood-First Alignment
# ---------------------------------------------------------------------------


class TestNeighborhoodAlignmentProtocol:
    """Test the Neighborhood-First Alignment protocol."""

    def test_neighborhood_alignment_runs(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Protocol 2 should complete and return all fields."""
        f_a, f_b = similar_features
        result = run_neighborhood_alignment(f_a, f_b, k=5, n_permutations=10, seed=42)
        assert isinstance(result.s_local, float)
        assert isinstance(result.s_global, float)
        assert isinstance(result.lambda_val, float)
        assert isinstance(result.raw_overlap, float)
        assert isinstance(result.null_mean, float)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_dissimilar_features_low_alignment(
        self, dissimilar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Dissimilar features should have low alignment."""
        f_a, f_b = dissimilar_features
        result = run_neighborhood_alignment(f_a, f_b, k=5, n_permutations=10, seed=42)
        assert result.s_local < 0.5, f"Expected low s_local, got {result.s_local:.3f}"


# ---------------------------------------------------------------------------
# Test 4: Protocol 3 — Transport Stress Test
# ---------------------------------------------------------------------------


class TestTransportStressTestProtocol:
    """Test the Transport Stress Test protocol."""

    def test_stress_test_runs(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Protocol 3 should complete without errors."""
        f_a, f_b = similar_features
        result = run_transport_stress_test(
            f_a,
            f_b,
            compression="rank_reduction",
            levels=[1.0, 0.5, 0.25],
            k=5,
            n_permutations=10,
            seed=42,
        )
        assert len(result.points) == 3
        for p in result.points:
            assert 0 <= p.tai <= 1
            assert 0 <= p.tau_a_to_b <= 1
            assert 0 <= p.tau_b_to_a <= 1
            assert p.delta >= 0
            assert p.regime in {"Collapsed", "Local-Only", "Convergent", "Divergent"}

    def test_delta_increases_with_compression(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Bottleneck mismatch should increase as one model is compressed more."""
        f_a, f_b = similar_features
        result = run_transport_stress_test(
            f_a,
            f_b,
            compression="rank_reduction",
            levels=[1.0, 0.5, 0.125],
            k=5,
            n_permutations=10,
            seed=42,
        )
        deltas = [p.delta for p in result.points]
        # Delta at highest compression should be >= delta at lowest
        assert deltas[-1] >= deltas[0] - 0.1, f"Expected increasing delta, got {deltas}"


# ---------------------------------------------------------------------------
# Test 5: Scorecard generation
# ---------------------------------------------------------------------------


class TestScorecardGeneration:
    """Test BCCT Scorecard output."""

    def test_scorecard_markdown(self, similar_features: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Scorecard should generate valid Markdown output."""
        f_a, f_b = similar_features

        # Compute all metrics
        s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
            f_a, f_b, k=5, n_permutations=10, seed=42
        )
        s_global = centered_kernel_alignment(f_a, f_b)
        transport = transport_linearity_score(f_a, f_b, seed=42)
        tai_result = transport_asymmetry_index(f_a, f_b, seed=42)
        r_a = representation_bitrate(f_a)
        _ = representation_bitrate(f_b)  # verify it runs on both
        delta = bottleneck_mismatch(f_a, f_b)
        regime = classify_regime(bitrate=r_a.bitrate, s_local=s_local, tau=transport.tau)

        scorecard = ScorecardEntry(
            model_name="TestModelA",
            reference_name="TestModelB",
            s_local=s_local,
            s_global=s_global,
            lambda_val=s_local / max(s_global, 1e-10),
            tau=transport.tau,
            tai=tai_result.tai,
            tai_direction="A→B" if tai_result.a_to_b.tau >= tai_result.b_to_a.tau else "B→A",
            bitrate=r_a.bitrate,
            delta=delta,
            regime=regime.regime.value,
        )

        md = scorecard_to_markdown(scorecard)
        assert "TestModelA" in md
        assert "TestModelB" in md
        assert "BCCT Scorecard" in md
        assert regime.regime.value in md


# ---------------------------------------------------------------------------
# Test 6: End-to-end with JSON serialization
# ---------------------------------------------------------------------------


class TestEndToEndSerialization:
    """Test that results can be serialized to JSON (like atlas.json)."""

    def test_full_pipeline_json_serializable(
        self, similar_features: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """All metrics should be JSON-serializable."""
        f_a, f_b = similar_features

        s_local, raw_overlap, null_mean = null_calibrated_mutual_knn(
            f_a, f_b, k=5, n_permutations=10, seed=42
        )
        s_global = centered_kernel_alignment(f_a, f_b)
        transport = transport_linearity_score(f_a, f_b, seed=42)
        r_a = representation_bitrate(f_a)
        delta = bottleneck_mismatch(f_a, f_b)
        regime = classify_regime(bitrate=r_a.bitrate, s_local=s_local, tau=transport.tau)

        result = {
            "s_local": round(float(s_local), 4),
            "s_global": round(float(s_global), 4),
            "raw_overlap": round(float(raw_overlap), 4),
            "null_mean": round(float(null_mean), 4),
            "tau": round(float(transport.tau), 4),
            "bitrate": round(float(r_a.bitrate), 4),
            "delta": round(float(delta), 4),
            "regime": regime.regime.value,
        }

        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["regime"] in {"Collapsed", "Local-Only", "Convergent", "Divergent"}
        assert all(isinstance(v, (int, float, str)) for v in parsed.values())


# ---------------------------------------------------------------------------
# Test 7: Atlas data loading (if available)
# ---------------------------------------------------------------------------


class TestAtlasDataIntegrity:
    """Test that atlas.json data is well-formed (if it exists)."""

    @pytest.fixture()
    def atlas_path(self) -> Path:
        return Path("data/atlas.json")

    def test_atlas_structure(self, atlas_path: Path) -> None:
        """Atlas JSON should have expected structure."""
        if not atlas_path.exists():
            pytest.skip("Atlas data not available")

        with open(atlas_path) as f:
            atlas = json.load(f)

        assert "models" in atlas
        assert "pairwise" in atlas
        assert "metadata" in atlas
        assert len(atlas["models"]) > 0
        assert len(atlas["pairwise"]) > 0

    def test_atlas_metric_ranges(self, atlas_path: Path) -> None:
        """All metrics in atlas should be within valid ranges."""
        if not atlas_path.exists():
            pytest.skip("Atlas data not available")

        with open(atlas_path) as f:
            atlas = json.load(f)

        for pair_key, metrics in atlas["pairwise"].items():
            assert 0 <= metrics["s_local"] <= 1, f"{pair_key}: s_local={metrics['s_local']}"
            assert 0 <= metrics["tau"] <= 1, f"{pair_key}: tau={metrics['tau']}"
            assert metrics["regime"] in {"Collapsed", "Local-Only", "Convergent", "Divergent"}, (
                f"{pair_key}: regime={metrics['regime']}"
            )

    def test_atlas_model_count(self, atlas_path: Path) -> None:
        """Atlas should have the expected number of models."""
        if not atlas_path.exists():
            pytest.skip("Atlas data not available")

        with open(atlas_path) as f:
            atlas = json.load(f)

        n_models = len(atlas["models"])
        expected_pairs = n_models * (n_models - 1) // 2
        assert len(atlas["pairwise"]) == expected_pairs, (
            f"Expected {expected_pairs} pairs for {n_models} models, got {len(atlas['pairwise'])}"
        )
