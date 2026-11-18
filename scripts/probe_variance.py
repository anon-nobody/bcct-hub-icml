"""S4: Seed / probe variance study.

For 10 representative pairs × 5 seeds × 4 probe architectures:
  - Compute τ (Procrustes + MLP probe ratio, per-dim R²)
  - Report std across seeds per architecture
  - Report regime classification stability

Success criteria:
  - τ std (within seed) < 0.05 per pair
  - Regime stability > 90%

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/probe_variance.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PAIRS = [
    ("clip_vitb16", "clip_vitl14"),
    ("clip_vitb32", "clip_vitb16"),
    ("resnet50", "resnet101"),
    ("convnext_base", "convnext_large"),
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("dinov2_vits14", "dinov2_vitb14"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    ("beit_base_patch16_224", "resnet50"),
    ("dinov2_vitb14", "clip_vitb16"),
    ("clip_vitb16", "resnet50"),
]

SEEDS = [42, 123, 456, 789, 1234]

# Probe configurations: (name, hidden_multiplier, n_layers, dropout)
PROBES = [
    ("linear", 0, 1, 0.0),
    ("mlp_2", 2, 2, 0.3),      # default
    ("mlp_2_wide", 4, 2, 0.3),
    ("mlp_3", 2, 3, 0.3),
]


def load_features(model: str, device: str) -> torch.Tensor:
    pt = Path("data/features") / f"{model}.pt"
    return torch.load(pt, weights_only=True)["features"].to(device)


def build_probe(d_in: int, d_out: int, hidden_mult: int, n_layers: int,
                dropout: float, device: str) -> nn.Module:
    if n_layers == 1:
        return nn.Linear(d_in, d_out).to(device)
    hidden = max(hidden_mult * d_out, 256)
    layers = [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden, d_out))
    return nn.Sequential(*layers).to(device)


def train_probe(src: torch.Tensor, tgt: torch.Tensor, probe: nn.Module,
                n_steps: int = 2000, seed: int = 42) -> float:
    """Train probe src -> tgt, return per-dim R² on held-out 20%."""
    torch.manual_seed(seed)
    n = src.shape[0]
    perm = torch.randperm(n, device=src.device, generator=torch.Generator(device=src.device).manual_seed(seed))
    n_tr = int(0.8 * n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    s_tr, t_tr = src[tr_idx], tgt[tr_idx]
    s_te, t_te = src[te_idx], tgt[te_idx]

    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    probe.train()
    for _ in range(n_steps):
        pred = probe(s_tr)
        loss = F.mse_loss(pred, t_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(s_te)
        # Per-dim R²
        ss_res = ((pred - t_te) ** 2).sum(dim=0)
        t_mean = t_te.mean(dim=0, keepdim=True)
        ss_tot = ((t_te - t_mean) ** 2).sum(dim=0) + 1e-10
        r2 = 1.0 - ss_res / ss_tot
        return float(r2.clamp(min=0).mean())


def procrustes_r2(src: torch.Tensor, tgt: torch.Tensor, seed: int = 42) -> float:
    torch.manual_seed(seed)
    n = src.shape[0]
    perm = torch.randperm(n, device=src.device, generator=torch.Generator(device=src.device).manual_seed(seed))
    n_tr = int(0.8 * n)
    tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    s_tr, t_tr = src[tr_idx], tgt[tr_idx]
    s_te, t_te = src[te_idx], tgt[te_idx]

    s_c = s_tr - s_tr.mean(0, keepdim=True)
    t_c = t_tr - t_tr.mean(0, keepdim=True)
    d_s, d_t = s_c.shape[1], t_c.shape[1]
    max_d = max(d_s, d_t)
    s_pad = F.pad(s_c, (0, max_d - d_s)) if d_s < max_d else s_c
    t_pad = F.pad(t_c, (0, max_d - d_t)) if d_t < max_d else t_c
    U, _, Vt = torch.linalg.svd(t_pad.T @ s_pad, full_matrices=False)
    W = U @ Vt
    s_te_full = F.pad(s_te - s_tr.mean(0, keepdim=True), (0, max_d - d_s)) if d_s < max_d else (s_te - s_tr.mean(0, keepdim=True))
    pred = (s_te_full @ W.T)[:, :d_t] + t_tr.mean(0, keepdim=True)

    ss_res = ((pred - t_te) ** 2).sum(dim=0)
    t_mean = t_te.mean(dim=0, keepdim=True)
    ss_tot = ((t_te - t_mean) ** 2).sum(dim=0) + 1e-10
    r2 = 1.0 - ss_res / ss_tot
    return float(r2.clamp(min=0).mean())


def classify_regime(tau: float, thresh: float = 0.5) -> str:
    return "Convergent" if tau > thresh else "NonConvergent"


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = []

    for model_a, model_b in PAIRS:
        try:
            fa = load_features(model_a, device)
            fb = load_features(model_b, device)
        except FileNotFoundError:
            continue

        n = min(fa.shape[0], fb.shape[0])
        fa, fb = fa[:n], fb[:n]

        pair_entry = {"pair": f"{model_a} <-> {model_b}", "probes": {}}

        for probe_name, mult, n_layers, dropout in PROBES:
            taus = []
            regimes = []
            for seed in SEEDS:
                r2_lin = procrustes_r2(fa, fb, seed=seed)
                if probe_name == "linear":
                    r2_probe = r2_lin  # linear ratio = 1 not informative; use simple mapping
                    # Use 1-layer learnable linear (not Procrustes) as "linear" probe
                    probe = build_probe(fa.shape[1], fb.shape[1], 0, 1, 0.0, device)
                    r2_probe = train_probe(fa, fb, probe, n_steps=2000, seed=seed)
                else:
                    probe = build_probe(fa.shape[1], fb.shape[1], mult, n_layers, dropout, device)
                    r2_probe = train_probe(fa, fb, probe, n_steps=2000, seed=seed)
                tau = r2_lin / max(r2_probe, 1e-10)
                tau = max(0.0, min(1.0, tau))
                taus.append(tau)
                regimes.append(classify_regime(tau))

            taus_arr = np.array(taus)
            # Regime stability: all seeds agree?
            stability = regimes.count(regimes[0]) / len(regimes)
            pair_entry["probes"][probe_name] = {
                "tau_mean": round(float(taus_arr.mean()), 4),
                "tau_std": round(float(taus_arr.std()), 4),
                "tau_range": [round(float(taus_arr.min()), 4), round(float(taus_arr.max()), 4)],
                "regime_stability": round(float(stability), 3),
                "per_seed": [round(t, 4) for t in taus],
            }
            logger.info(
                "%s x %s | %s: mean=%.3f, std=%.3f, stability=%.2f",
                model_a, model_b, probe_name,
                taus_arr.mean(), taus_arr.std(), stability,
            )

        # Cross-probe regime stability (at seed=42 only)
        cross_probe_taus = [pair_entry["probes"][p]["per_seed"][0] for p in ["linear", "mlp_2", "mlp_2_wide", "mlp_3"]]
        cross_probe_regimes = [classify_regime(t) for t in cross_probe_taus]
        pair_entry["cross_probe_regime_stability"] = (
            cross_probe_regimes.count(cross_probe_regimes[0]) / len(cross_probe_regimes)
        )

        results.append(pair_entry)

    # Aggregate
    all_stds = []
    all_stabilities = []
    for r in results:
        for p in ["linear", "mlp_2", "mlp_2_wide", "mlp_3"]:
            all_stds.append(r["probes"][p]["tau_std"])
            all_stabilities.append(r["probes"][p]["regime_stability"])
    cross_probe_ok = sum(1 for r in results if r["cross_probe_regime_stability"] >= 0.75)

    output = {
        "description": "Seed/probe variance study: 10 pairs × 5 seeds × 4 probe architectures",
        "n_pairs": len(results),
        "seeds": SEEDS,
        "probes": [p[0] for p in PROBES],
        "results": results,
        "summary": {
            "median_tau_std_across_seeds": round(float(np.median(all_stds)), 4),
            "max_tau_std": round(float(np.max(all_stds)), 4),
            "median_regime_stability_within_probe": round(float(np.median(all_stabilities)), 3),
            "pairs_with_cross_probe_agreement_ge_75pct": cross_probe_ok,
            "total_pairs": len(results),
        },
    }

    out = Path("data/experiments/probe_variance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    logger.info("\n=== Summary ===")
    logger.info(json.dumps(output["summary"], indent=2))
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    main()
