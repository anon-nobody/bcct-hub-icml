"""Tests for Convergence Regime classification."""

from __future__ import annotations

from latent.metrics.regime import (
    ConvergenceRegime,
    RegimeThresholds,
    classify_regime,
)


class TestRegimeClassification:
    """Tests for the four BCCT convergence regimes."""

    def test_collapsed_low_bitrate(self) -> None:
        result = classify_regime(bitrate=1.0, s_local=0.2, tau=0.3)
        assert result.regime == ConvergenceRegime.COLLAPSED

    def test_convergent_high_everything(self) -> None:
        result = classify_regime(bitrate=8.0, s_local=0.8, tau=0.95)
        assert result.regime == ConvergenceRegime.CONVERGENT

    def test_local_only_high_lambda_low_tau(self) -> None:
        result = classify_regime(bitrate=5.0, s_local=0.3, tau=0.4, s_global=0.3, lambda_val=2.5)
        assert result.regime == ConvergenceRegime.LOCAL_ONLY

    def test_divergent_high_bitrate_low_alignment(self) -> None:
        result = classify_regime(bitrate=10.0, s_local=0.01, tau=0.2)
        assert result.regime == ConvergenceRegime.DIVERGENT

    def test_custom_thresholds(self) -> None:
        thresholds = RegimeThresholds(r_min=1.0, s_local_thresh=0.1, tau_thresh=0.5)
        result = classify_regime(bitrate=2.0, s_local=0.4, tau=0.6, thresholds=thresholds)
        assert result.regime == ConvergenceRegime.CONVERGENT

    def test_result_contains_all_inputs(self) -> None:
        result = classify_regime(bitrate=5.0, s_local=0.6, tau=0.85, s_global=0.7, lambda_val=1.2)
        assert result.bitrate == 5.0
        assert result.s_local == 0.6
        assert result.tau == 0.85
        assert result.s_global == 0.7
        assert result.lambda_val == 1.2

    def test_boundary_collapsed_to_convergent(self) -> None:
        """Test exactly at R_min boundary."""
        thresholds = RegimeThresholds(r_min=2.0)
        # Just below threshold
        below = classify_regime(bitrate=1.99, s_local=0.8, tau=0.95, thresholds=thresholds)
        assert below.regime == ConvergenceRegime.COLLAPSED
        # At threshold
        at = classify_regime(bitrate=2.0, s_local=0.8, tau=0.95, thresholds=thresholds)
        assert at.regime == ConvergenceRegime.CONVERGENT
