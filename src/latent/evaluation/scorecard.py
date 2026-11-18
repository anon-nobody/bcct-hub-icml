"""BCCT Scorecard: standardized one-page report for representation evaluation.

Generates a scorecard that can be included in papers as a supplementary table
(see paper §3, Evaluation Protocols and Empirical Results).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
from latent.metrics.bitrate import representation_bitrate
from latent.metrics.regime import RegimeThresholds, classify_regime
from latent.metrics.transport import transport_linearity_score

logger = logging.getLogger(__name__)


@dataclass
class ScorecardEntry:
    """A single BCCT Scorecard entry.

    All fields correspond to the required fields in §7.4 Table.
    """

    model_name: str
    reference_name: str
    bitrate: float  # R (bits/dim)
    s_local: float
    s_global: float  # linear CKA
    lambda_val: float  # alignment locality
    tau: float  # transport linearity score
    tai: float  # transport asymmetry index
    tai_direction: str  # which direction is better
    delta: float  # bottleneck mismatch vs reference
    regime: str  # convergence regime label


def compute_scorecard(
    model_features: Tensor,
    reference_features: Tensor,
    model_name: str = "model",
    reference_name: str = "CLIP-ViT-L/14",
    k: int = 10,
    n_permutations: int = 1000,
    mlp_steps: int = 3000,
    seed: int = 42,
    thresholds: RegimeThresholds | None = None,
) -> ScorecardEntry:
    """Compute full BCCT Scorecard for a model against a reference.

    This is the primary user-facing function: given features from a model
    and a reference encoder, it computes all BCCT metrics and returns a
    complete Scorecard entry.

    Args:
        model_features: Model representations, shape (N, D_model).
        reference_features: Reference encoder representations, shape (N, D_ref).
        model_name: Human-readable model identifier.
        reference_name: Human-readable reference identifier.
        k: Nearest neighbors for alignment metrics.
        n_permutations: Permutations for null calibration.
        seed: Random seed.
        thresholds: Custom regime thresholds.

    Returns:
        ScorecardEntry with all BCCT metrics filled in.
    """
    logger.info("Computing BCCT Scorecard for '%s' vs '%s'", model_name, reference_name)

    # Bitrate
    model_r = representation_bitrate(model_features)
    ref_r = representation_bitrate(reference_features)
    delta = abs(model_r.bitrate - ref_r.bitrate)

    logger.info("  R(model)=%.2f, R(ref)=%.2f, Δ=%.2f", model_r.bitrate, ref_r.bitrate, delta)

    # Alignment (Protocol 2)
    s_local, _, _ = null_calibrated_mutual_knn(
        model_features, reference_features, k=k, n_permutations=n_permutations, seed=seed
    )
    s_global = centered_kernel_alignment(model_features, reference_features)

    lambda_val = s_local / s_global if s_global > 1e-10 else float("inf") if s_local > 0 else 0.0

    logger.info("  S_local=%.2f, S_global=%.3f, λ=%.2f", s_local, s_global, lambda_val)

    # Transport — A→B direction
    transport_ab = transport_linearity_score(
        model_features, reference_features, mlp_steps=mlp_steps, seed=seed
    )
    # Transport — B→A direction
    transport_ba = transport_linearity_score(
        reference_features, model_features, mlp_steps=mlp_steps, seed=seed
    )

    # Use conservative bidirectional definition: min of both directions
    tau = min(transport_ab.tau, transport_ba.tau)
    perf_ab = transport_ab.perf_best
    perf_ba = transport_ba.perf_best
    max_perf = max(perf_ab, perf_ba)

    tai = 0.0 if max_perf < 1e-10 else abs(perf_ab - perf_ba) / max_perf
    tai_direction = "model→ref" if perf_ab >= perf_ba else "ref→model"

    logger.info("  τ=%.3f, TAI=%.3f (%s)", tau, tai, tai_direction)

    # Regime classification
    regime_result = classify_regime(
        bitrate=model_r.bitrate,
        s_local=s_local,
        tau=tau,
        s_global=s_global,
        lambda_val=lambda_val,
        thresholds=thresholds,
    )

    logger.info("  Regime: %s", regime_result.regime.value)

    return ScorecardEntry(
        model_name=model_name,
        reference_name=reference_name,
        bitrate=round(model_r.bitrate, 3),
        s_local=round(s_local, 2),
        s_global=round(s_global, 4),
        lambda_val=round(lambda_val, 2),
        tau=round(tau, 4),
        tai=round(tai, 4),
        tai_direction=tai_direction,
        delta=round(delta, 3),
        regime=regime_result.regime.value,
    )


def scorecard_to_latex(entry: ScorecardEntry) -> str:
    """Format a ScorecardEntry as a LaTeX table.

    Args:
        entry: Computed scorecard entry.

    Returns:
        LaTeX string for inclusion in a paper.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{BCCT Scorecard: " + entry.model_name + r"}",
        r"\label{tab:scorecard_" + entry.model_name.replace("/", "_").replace(" ", "_") + r"}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"\textbf{Field} & \textbf{Value} \\",
        r"\midrule",
        rf"Model & {entry.model_name} \\",
        rf"Reference & {entry.reference_name} \\",
        rf"$\mathcal{{R}}$ (bits/dim) & {entry.bitrate} \\",
        rf"$S_{{\text{{local}}}}$ (calibrated overlap) & {entry.s_local} \\",
        rf"$S_{{\text{{global}}}}$ (CKA) & {entry.s_global} \\",
        rf"$\lambda$ & {entry.lambda_val} \\",
        rf"$\tau$ & {entry.tau} \\",
        rf"TAI & {entry.tai} ({entry.tai_direction}) \\",
        rf"$\Delta$ vs. reference & {entry.delta} \\",
        rf"Regime & \textbf{{{entry.regime}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def scorecard_to_markdown(entry: ScorecardEntry) -> str:
    """Format a ScorecardEntry as a Markdown table.

    Args:
        entry: Computed scorecard entry.

    Returns:
        Markdown string.
    """
    lines = [
        f"## BCCT Scorecard: {entry.model_name}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Model | {entry.model_name} |",
        f"| Reference | {entry.reference_name} |",
        f"| R (bits/dim) | {entry.bitrate} |",
        f"| S_local (calibrated overlap) | {entry.s_local} |",
        f"| S_global (CKA) | {entry.s_global} |",
        f"| λ | {entry.lambda_val} |",
        f"| τ | {entry.tau} |",
        f"| TAI | {entry.tai} ({entry.tai_direction}) |",
        f"| Δ vs. reference | {entry.delta} |",
        f"| **Regime** | **{entry.regime}** |",
    ]
    return "\n".join(lines)


def scorecard_to_json(entry: ScorecardEntry) -> str:
    """Serialize a ScorecardEntry to JSON.

    Args:
        entry: Computed scorecard entry.

    Returns:
        JSON string.
    """
    return json.dumps(asdict(entry), indent=2)


def main() -> None:
    """CLI entry point for bcct-scorecard command.

    Usage: bcct-scorecard --model-features model.pt --ref-features ref.pt
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compute BCCT Scorecard")
    parser.add_argument(
        "--model-features", type=str, required=True, help="Path to model features .pt file"
    )
    parser.add_argument(
        "--ref-features", type=str, required=True, help="Path to reference features .pt file"
    )
    parser.add_argument("--model-name", type=str, default="model", help="Model identifier")
    parser.add_argument(
        "--ref-name", type=str, default="CLIP-ViT-L/14", help="Reference identifier"
    )
    parser.add_argument(
        "--format", choices=["latex", "markdown", "json"], default="markdown", help="Output format"
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("-k", type=int, default=10, help="k for mutual k-NN")
    parser.add_argument(
        "--permutations", type=int, default=1000, help="Null calibration permutations"
    )
    parser.add_argument(
        "--mlp-steps", type=int, default=3000, help="MLP probe training steps"
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Subsample N samples from features (default: all)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast smoke-test mode: 500 samples, 100 permutations, 300 MLP steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.fast:
        args.n_samples = args.n_samples or 500
        args.permutations = min(args.permutations, 100)
        args.mlp_steps = min(args.mlp_steps, 300)
        logging.info(
            "FAST mode: n_samples=%d, permutations=%d, mlp_steps=%d",
            args.n_samples, args.permutations, args.mlp_steps,
        )

    def _load_feature_tensor(path: str) -> Tensor:
        obj = torch.load(path, weights_only=True)
        if isinstance(obj, dict) and "features" in obj:
            return obj["features"]
        if isinstance(obj, Tensor):
            return obj
        raise TypeError(f"{path} must contain a Tensor or a dict with 'features' key")

    model_feat = _load_feature_tensor(args.model_features)
    ref_feat = _load_feature_tensor(args.ref_features)

    if args.n_samples is not None:
        n = min(args.n_samples, model_feat.shape[0], ref_feat.shape[0])
        # Deterministic subsample for reproducibility
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(min(model_feat.shape[0], ref_feat.shape[0]), generator=g)[:n]
        model_feat = model_feat[idx]
        ref_feat = ref_feat[idx]
        logging.info("Subsampled %d features (seed=%d)", n, args.seed)

    entry = compute_scorecard(
        model_features=model_feat,
        reference_features=ref_feat,
        model_name=args.model_name,
        reference_name=args.ref_name,
        k=args.k,
        n_permutations=args.permutations,
        mlp_steps=args.mlp_steps,
        seed=args.seed,
    )

    if args.format == "latex":
        output = scorecard_to_latex(entry)
    elif args.format == "json":
        output = scorecard_to_json(entry)
    else:
        output = scorecard_to_markdown(entry)

    if args.output:
        Path(args.output).write_text(output)
        logger.info("Scorecard written to %s", args.output)
    else:
        print(output)
