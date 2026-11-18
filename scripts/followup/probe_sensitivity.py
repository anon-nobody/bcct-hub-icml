"""Regenerate the transport-probe sensitivity analysis for 9 model pairs.

data/experiments/probe_sensitivity.json was produced inline in May 2026 with no
script checked into the repo. This script recomputes the same analysis directly
from data/features/<model>.pt using the production Procrustes implementation in
latent.metrics.transport, a closed-form ridge baseline, and a scaled copy of the
production 2-layer MLP.

Definitions (reverse-engineered from the original file's per-pair numbers --
see the task report for the full derivation and the checks below):

  - tau(A, B) is single-directional: A is the first model named in "A <-> B",
    source = features_a, target = features_b (NOT the bidirectional min used
    by data/atlas.json -- the original file's numbers for these same pairs do
    not match atlas's bidirectional-conservative tau, but do match a single
    A->B computation).
  - The train/test split is generated with a plain, device-unspecified
    ``torch.randperm(n)`` under ``torch.manual_seed(42)`` -- i.e. on CPU, not
    on whatever device the features happen to live on. This matters because
    torch's CPU and CUDA RNGs are different algorithms, so the *same* seed
    produces a *different* permutation on CPU vs GPU. Confirmed empirically:
    running the split + Procrustes + ridge computation entirely on CPU
    reproduces the original file's ridge_alpha=1.0 R^2 for
    "clip_vitb32 <-> clip_vitb16" bit-for-bit (0.5195095539093018) and all
    three ridge alphas to >=15 significant figures, and procrustes_r2 to 7
    significant figures (0.327512264 vs 0.327512145, i.e. float32 SVD noise,
    not a different split). Generating the permutation on GPU instead gives a
    visibly different procrustes_r2 (off by ~2e-3, i.e. a different split).
  - "procrustes_r2" is the Procrustes (orthogonal linear) R^2 on the held-out
    20% split: exactly latent.metrics.transport._evaluate_linear_transport's
    per-dimension R^2, computed on CPU.
  - "ridge": closed-form L2-regularized linear regression (ridge(alpha) =
    argmin_W ||Yc - Xc @ W||^2 + alpha ||W||^2 on the centered train split),
    evaluated with the same per-dimension R^2 convention, for alpha in
    {0.1, 1.0, 10.0}, computed on CPU.
  - "mlp" grid: for steps in {1000, 3000, 10000} x hidden_multiplier in
    {1, 2, 4}, the 2-layer MLP from _TransportMLP (Linear -> ReLU ->
    Dropout(0.3) -> Linear, AdamW lr=1e-3 wd=1e-4, CosineAnnealingLR,
    normalized inputs/targets) but with hidden width = hidden_multiplier *
    d_out (the output dimension), NOT transport.py's own default hidden
    formula of max(2*d_out, 256). This was reverse-engineered, not assumed:
    for all 9 pairs, tau_baseline (and, where it isn't just capped at 1.0,
    tau_ridge_best) in the original file exactly equals
    procrustes_r2 / mlp_r2[steps=3000, hidden_multiplier=2] -- e.g. for
    "dinov2_vits14 <-> dinov2_vitg14": 0.019612/0.266452 = 0.073604, which
    matches the stored tau_baseline (0.07360391829935198) to 6 significant
    figures, and the stored tau_ridge_best (0.9552823908895103) exactly
    equals ridge_best_r2/mlp_r2[3000,2] (0.955282...). hidden_multiplier=2
    therefore reproduces transport.py's *current* default hidden width
    (2*d_out) for every one of these 9 pairs (all have d_out >= 384, so the
    max(., 256) floor never binds) -- i.e. the original grid's multiplier
    axis is {1,2,4}*d_out, offset by one "slot" from a literal reading of
    "hidden_multiplier * transport.py's default hidden". This directly
    contradicts a literal reading of the task's "hidden width = multiplier x
    default" instruction; this file follows the data, not that literal
    reading -- see the task report for the full reconciliation.
  - tau at a given (linear_r2, nonlinear_r2) pair is
    clamp(linear_r2 / nonlinear_r2, 0, 1), 0 if nonlinear_r2 < 1e-10 --
    transport.py's own tau formula.
  - tau_baseline = tau(procrustes_r2, mlp_r2[steps=3000, hidden_multiplier=2])
    -- the "standard" MLP cell, per the derivation above.
  - tau_min/tau_max/tau_std are computed over the 9 tau(procrustes_r2,
    mlp_r2[setting]) values across the full MLP grid.
  - ridge_best_r2 = max over the 3 ridge alphas.
  - tau_ridge_best = tau(ridge_best_r2, mlp_r2[steps=3000, hidden_multiplier=2])
    -- substituting ridge for Procrustes as the "linear" baseline against the
    same standard MLP denominator used for tau_baseline.

Device: the split, Procrustes, and ridge computations run on CPU (see above --
this is required to reproduce the original split). The MLP grid runs on GPU 0
by default (CUDA_VISIBLE_DEVICES=0) purely for speed: 9 pairs x 9 settings of
up to 10000 AdamW steps is impractically slow on CPU on this machine (~0.15s
per MLP step measured here at 32 CPU threads, i.e. the full 9x9 grid across all
9 pairs would be single-digit hours), and GPU training is not expected to
reproduce the original file bit-for-bit regardless (transport.py never calls
torch.use_deterministic_algorithms, so GPU MLP training is only
approximately reproducible even run-to-run on the same GPU). Set
CUDA_VISIBLE_DEVICES="" to force the MLP grid onto CPU as well (much slower
but maximally faithful).

Input:  data/features/<model>.pt for the 9 pairs below
Output: data/experiments/probe_sensitivity.json (the May 2026 inline result is kept as
        data/experiments/probe_sensitivity_may2026.json)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/followup/probe_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.transport import _evaluate_linear_transport  # noqa: E402

FEATURES_DIR = ROOT / "data/features"
OUTPUT_PATH = ROOT / "data/experiments/probe_sensitivity.json"

PAIRS: list[tuple[str, str]] = [
    ("clip_vitb16", "clip_vitl14"),
    ("clip_vitb32", "clip_vitb16"),
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("dinov2_vits14", "dinov2_vitg14"),
    ("resnet50", "resnet101"),
    ("vit_base_patch16_224", "resnet50"),
    ("beit_base_patch16_224", "beit_large_patch16_224"),
    ("dinov2_vitl14", "beit_large_patch16_224"),
    ("clip_vitb16", "dinov2_vitb14"),
]

MLP_STEPS = (1000, 3000, 10000)
HIDDEN_MULTIPLIERS = (1, 2, 4)
STANDARD_MLP_KEY = "steps=3000,h_mult=2"
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
TRAIN_RATIO = 0.8
MLP_LR = 1e-3
SEED = 42
MLP_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class _ScaledTransportMLP(nn.Module):
    """Same 2-layer MLP as transport._TransportMLP, with scalable hidden width.

    hidden = hidden_multiplier * d_out (see module docstring for why this,
    rather than transport._TransportMLP's own max(2*d_out, 256) formula,
    reproduces the original file's grid).
    """

    def __init__(self, d_in: int, d_out: int, hidden_multiplier: int) -> None:
        super().__init__()
        hidden = hidden_multiplier * d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _r2(target_test: torch.Tensor, pred_test: torch.Tensor) -> float:
    """Per-dimension R^2, averaged, clamped at 0 -- transport.py's convention."""
    ss_res_d = ((target_test - pred_test) ** 2).sum(dim=0)
    ss_tot_d = ((target_test - target_test.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    if ss_tot_d.max() < 1e-10:
        return 1.0
    r2_d = 1.0 - ss_res_d / ss_tot_d.clamp(min=1e-10)
    return r2_d.clamp(min=0.0).mean().item()


def _mlp_r2(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_test: torch.Tensor,
    target_test: torch.Tensor,
    n_steps: int,
    hidden_multiplier: int,
    device: torch.device | str,
    lr: float = MLP_LR,
    seed: int = SEED,
) -> float:
    """Train _ScaledTransportMLP exactly as transport._evaluate_mlp_transport does."""
    source_train = source_train.to(device)
    target_train = target_train.to(device)
    source_test = source_test.to(device)
    target_test = target_test.to(device)

    torch.manual_seed(seed)
    d_in = source_train.shape[1]
    d_out = target_train.shape[1]
    model = _ScaledTransportMLP(d_in, d_out, hidden_multiplier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

    src_mean = source_train.mean(dim=0, keepdim=True)
    src_std = source_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    tgt_mean = target_train.mean(dim=0, keepdim=True)
    tgt_std = target_train.std(dim=0, keepdim=True).clamp(min=1e-6)

    src_train_n = (source_train - src_mean) / src_std
    tgt_train_n = (target_train - tgt_mean) / tgt_std
    src_test_n = (source_test - src_mean) / src_std

    model.train()
    for _ in range(n_steps):
        pred = model(src_train_n)
        loss = nn.functional.mse_loss(pred, tgt_train_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        pred_test_n = model(src_test_n)
        pred_test = pred_test_n * tgt_std + tgt_mean
        return _r2(target_test, pred_test)


def _ridge_r2(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_test: torch.Tensor,
    target_test: torch.Tensor,
    alpha: float,
) -> float:
    """Closed-form ridge regression on the centered train split."""
    src_mean = source_train.mean(dim=0, keepdim=True)
    tgt_mean = target_train.mean(dim=0, keepdim=True)
    xc = source_train - src_mean
    yc = target_train - tgt_mean
    d_in = xc.shape[1]
    eye = torch.eye(d_in, dtype=xc.dtype)
    w = torch.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ yc)
    pred = (source_test - src_mean) @ w + tgt_mean
    return _r2(target_test, pred)


def _tau(linear_r2: float, nonlinear_r2: float) -> float:
    """transport.py's own tau formula: clamp(linear/nonlinear, 0, 1)."""
    if nonlinear_r2 < 1e-10:
        return 0.0
    return min(1.0, max(0.0, linear_r2 / nonlinear_r2))


def _load_features(name: str) -> torch.Tensor:
    """Load features on CPU (the split must happen on CPU -- see module docstring)."""
    data = torch.load(FEATURES_DIR / f"{name}.pt", weights_only=True)
    return data["features"]


def _split(n: int, seed: int, train_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU train/test split, matching the original file's apparent computation.

    Deliberately does NOT pass device= to torch.randperm (see module
    docstring): the original file's numbers only reproduce with a
    device-unspecified (CPU) permutation, even though transport.py's own
    transport_linearity_score generates the permutation on the features'
    device.
    """
    torch.manual_seed(seed)
    n_train = int(n * train_ratio)
    perm = torch.randperm(n)
    return perm[:n_train], perm[n_train:]


def run_pair(model_a: str, model_b: str) -> dict:
    features_a = _load_features(model_a)
    features_b = _load_features(model_b)
    n = features_a.shape[0]
    train_idx, test_idx = _split(n, SEED, TRAIN_RATIO)
    src_train, src_test = features_a[train_idx], features_a[test_idx]
    tgt_train, tgt_test = features_b[train_idx], features_b[test_idx]

    procrustes_r2, _ = _evaluate_linear_transport(src_train, tgt_train, src_test, tgt_test)

    mlp_grid: dict[str, float] = {}
    for steps in MLP_STEPS:
        for h_mult in HIDDEN_MULTIPLIERS:
            key = f"steps={steps},h_mult={h_mult}"
            mlp_grid[key] = _mlp_r2(
                src_train, tgt_train, src_test, tgt_test, steps, h_mult, MLP_DEVICE
            )

    ridge_grid: dict[str, float] = {}
    for alpha in RIDGE_ALPHAS:
        ridge_grid[f"ridge_alpha={alpha}"] = _ridge_r2(src_train, tgt_train, src_test, tgt_test, alpha)

    mlp_standard = mlp_grid[STANDARD_MLP_KEY]
    tau_grid = [_tau(procrustes_r2, r2) for r2 in mlp_grid.values()]
    ridge_best_r2 = max(ridge_grid.values())

    return {
        "procrustes_r2": procrustes_r2,
        "ridge": ridge_grid,
        "mlp": mlp_grid,
        "tau_baseline": _tau(procrustes_r2, mlp_standard),
        "tau_min": min(tau_grid),
        "tau_max": max(tau_grid),
        "tau_std": float(np.std(tau_grid)),
        "ridge_best_r2": ridge_best_r2,
        "tau_ridge_best": _tau(ridge_best_r2, mlp_standard),
    }


def main() -> None:
    print(f"mlp_device={MLP_DEVICE}")
    per_pair: dict[str, dict] = {}
    summary: list[dict] = []
    for model_a, model_b in PAIRS:
        pair_key = f"{model_a} <-> {model_b}"
        t0 = time.time()
        result = run_pair(model_a, model_b)
        elapsed = time.time() - t0
        print(
            f"{pair_key}: procrustes_r2={result['procrustes_r2']:.4f} "
            f"tau_baseline={result['tau_baseline']:.4f} ({elapsed:.1f}s)"
        )

        per_pair[pair_key] = {
            "procrustes_r2": result["procrustes_r2"],
            "ridge": result["ridge"],
            "mlp": result["mlp"],
        }
        summary.append(
            {
                "pair": pair_key,
                "procrustes_r2": result["procrustes_r2"],
                "tau_baseline": result["tau_baseline"],
                "tau_min": result["tau_min"],
                "tau_max": result["tau_max"],
                "tau_std": result["tau_std"],
                "ridge_best_r2": result["ridge_best_r2"],
                "tau_ridge_best": result["tau_ridge_best"],
            }
        )

    output = {
        "description": (
            "tau sensitivity to MLP probe (steps x capacity) and Procrustes vs ridge "
            "baseline. Regenerated from data/features/*.pt by "
            "scripts/followup/probe_sensitivity.py (see module docstring for exact "
            "definitions, including the tau/tau_baseline/tau_ridge_best formulas and "
            "the hidden-width convention)."
        ),
        "pairs": len(PAIRS),
        "mlp_grid": {"steps": list(MLP_STEPS), "hidden_multiplier": list(HIDDEN_MULTIPLIERS)},
        "ridge_alphas": list(RIDGE_ALPHAS),
        "per_pair": per_pair,
        "summary": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
