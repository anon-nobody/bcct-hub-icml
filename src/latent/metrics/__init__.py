"""BCCT metrics: the six core quantities for representation analysis.

Quantities:
    R  — Representation Bitrate (bits/dim)
    τ  — Transport Linearity Score [0, 1]
    λ  — Alignment Locality (0, ∞)
    TAI — Transport Asymmetry Index [0, 1]
    Convergence Regime — {Collapsed, Local-Only, Convergent, Divergent}
    Δ  — Bottleneck Mismatch (bits/dim)
"""

from latent.metrics.alignment import (
    alignment_locality,
    centered_kernel_alignment,
    mutual_knn_overlap,
    null_calibrated_mutual_knn,
)
from latent.metrics.bitrate import (
    bottleneck_mismatch,
    effective_rank,
    representation_bitrate,
)
from latent.metrics.regime import (
    ConvergenceRegime,
    classify_regime,
)
from latent.metrics.transport import (
    transport_asymmetry_index,
    transport_linearity_score,
)

__all__ = [
    "ConvergenceRegime",
    "alignment_locality",
    "bottleneck_mismatch",
    "centered_kernel_alignment",
    "classify_regime",
    "effective_rank",
    "mutual_knn_overlap",
    "null_calibrated_mutual_knn",
    "representation_bitrate",
    "transport_asymmetry_index",
    "transport_linearity_score",
]
