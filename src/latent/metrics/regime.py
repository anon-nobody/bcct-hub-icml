"""Convergence Regime classification.

Implements §2.5 of the BCCT framework. Classifies model pairs into one of
four regimes based on their BCCT metrics: Collapsed, Local-Only, Convergent, Divergent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConvergenceRegime(Enum):
    """The four BCCT convergence regimes."""

    COLLAPSED = "Collapsed"
    LOCAL_ONLY = "Local-Only"
    CONVERGENT = "Convergent"
    DIVERGENT = "Divergent"


@dataclass(frozen=True)
class RegimeThresholds:
    """Decision boundaries for regime classification.

    These defaults are initial estimates; the meta-analysis (§7.5) will
    calibrate them empirically. Users should run sensitivity analysis
    by varying thresholds ±20%.

    Attributes:
        r_min: Minimum bitrate to escape Collapsed regime (bits/dim).
        s_local_thresh: Minimum S_local (calibrated overlap) for Convergent regime.
        tau_thresh: Minimum τ for Convergent regime.
        lambda_thresh: Reserved for future threshold-based refinements of λ.
    """

    r_min: float = 2.0
    s_local_thresh: float = 0.05
    tau_thresh: float = 0.5
    lambda_thresh: float = 1.5


@dataclass(frozen=True)
class RegimeClassification:
    """Result of regime classification.

    Attributes:
        regime: Classified convergence regime.
        bitrate: Representation Bitrate used for classification.
        s_local: Null-calibrated mutual k-NN overlap.
        s_global: Linear CKA score.
        tau: Transport Linearity Score.
        lambda_val: Alignment Locality ratio.
        thresholds: Thresholds used for classification.
    """

    regime: ConvergenceRegime
    bitrate: float
    s_local: float
    s_global: float
    tau: float
    lambda_val: float
    thresholds: RegimeThresholds


def classify_regime(
    bitrate: float,
    s_local: float,
    tau: float,
    s_global: float = 0.0,
    lambda_val: float | None = None,
    thresholds: RegimeThresholds | None = None,
) -> RegimeClassification:
    """Classify a model pair into a BCCT Convergence Regime.

    Decision logic (from §2.5):
        1. R < R_min → Collapsed
        2. R ≥ R_min AND S_local ≥ S_thresh AND τ ≥ τ_thresh → Convergent
        3. R ≥ R_min AND S_local ≥ S_thresh AND τ < τ_thresh → Local-Only
        4. R ≥ R_min AND S_local < S_thresh → Divergent

    Args:
        bitrate: Representation Bitrate R (bits/dim).
        s_local: Null-calibrated mutual k-NN overlap.
        tau: Transport Linearity Score.
        s_global: Linear CKA score (used for λ computation if lambda_val not given).
        lambda_val: Pre-computed Alignment Locality. If None, computed from s_local/s_global.
        thresholds: Custom thresholds. If None, uses defaults.

    Returns:
        RegimeClassification with regime label and all input metrics.
    """
    if thresholds is None:
        thresholds = RegimeThresholds()

    if lambda_val is None:
        if s_global > 1e-10:
            lambda_val = s_local / s_global
        else:
            lambda_val = float("inf") if s_local > 0 else 0.0

    # Decision logic
    # 1. Low bitrate → Collapsed (information lost)
    # 2. High local alignment + good transport → Convergent
    # 3. High local alignment + poor transport → Local-Only (aligned but not linearly transportable)
    # 4. Low local alignment → Divergent (representations do not share structure)
    if bitrate < thresholds.r_min:
        regime = ConvergenceRegime.COLLAPSED
    elif s_local >= thresholds.s_local_thresh and tau >= thresholds.tau_thresh:
        regime = ConvergenceRegime.CONVERGENT
    elif s_local >= thresholds.s_local_thresh and tau < thresholds.tau_thresh:
        regime = ConvergenceRegime.LOCAL_ONLY
    else:
        regime = ConvergenceRegime.DIVERGENT

    return RegimeClassification(
        regime=regime,
        bitrate=bitrate,
        s_local=s_local,
        s_global=s_global,
        tau=tau,
        lambda_val=lambda_val,
        thresholds=thresholds,
    )
