"""Probe Architecture Robustness Experiment.

Tests whether the bimodal τ distribution is a property of representations
or an artifact of probe architecture. Computes τ with 4 probe architectures
on 10 representative pairs.

Usage:
    python scripts/probe_robustness.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from latent.metrics.transport import _evaluate_linear_transport

logger = logging.getLogger(__name__)

TARGET_PAIRS = [
    ("clip_vitl14", "convnext_base"),
    ("clip_vitb32", "clip_vitb16"),
    ("resnet50", "vit_base_patch16_224"),
    ("dinov2_vitb14", "clip_vitl14"),
    ("dinov2_vitb14", "resnet50"),
    ("resnet50", "mixer_b16_224"),
    ("convnext_base", "mixer_b16_224"),
    ("dinov2_vitb14", "vit_base_patch16_224.mae"),
    ("clip_vitl14", "vit_base_patch16_224.mae"),
    ("dinov2_vitb14", "convnext_base"),
]


class MLP1Layer(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLP2Layer(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 2 * d_out), nn.ReLU(), nn.Linear(2 * d_out, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLP2LayerWide(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 4 * d_out), nn.ReLU(), nn.Linear(4 * d_out, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLP3Layer(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 4 * d_out),
            nn.ReLU(),
            nn.Linear(4 * d_out, 2 * d_out),
            nn.ReLU(),
            nn.Linear(2 * d_out, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


ARCHITECTURES = {
    "1-layer": MLP1Layer,
    "2-layer": MLP2Layer,
    "2-layer-wide": MLP2LayerWide,
    "3-layer": MLP3Layer,
}


def eval_mlp(
    cls: type,
    src_tr: Tensor,
    tgt_tr: Tensor,
    src_te: Tensor,
    tgt_te: Tensor,
    device: str,
    steps: int = 2000,
    lr: float = 1e-3,
    seed: int = 42,
) -> float:
    """Train and evaluate an MLP transport probe."""
    torch.manual_seed(seed)
    model = cls(src_tr.shape[1], tgt_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sm = src_tr.mean(0, keepdim=True)
    ss = src_tr.std(0, keepdim=True).clamp(min=1e-6)
    tm = tgt_tr.mean(0, keepdim=True)
    ts = tgt_tr.std(0, keepdim=True).clamp(min=1e-6)

    sn = (src_tr - sm) / ss
    tn = (tgt_tr - tm) / ts
    sn_te = (src_te - sm) / ss

    model.train()
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(sn), tn)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(sn_te) * ts + tm
        ss_res = ((tgt_te - pred) ** 2).sum()
        ss_tot = ((tgt_te - tgt_te.mean(0, keepdim=True)) ** 2).sum()
        return max(0.0, (1 - ss_res / ss_tot).item()) if ss_tot > 1e-10 else 1.0


def run_experiment(features_dir: Path, device: str) -> list[dict]:
    """Run probe robustness experiment on 10 pairs."""
    feats: dict[str, Tensor] = {}
    for p in features_dir.glob("*.pt"):
        d = torch.load(p, map_location="cpu", weights_only=True)
        feats[d.get("model_name", p.stem)] = d["features"]

    results = []
    for i, (m1, m2) in enumerate(TARGET_PAIRS):
        if m1 not in feats or m2 not in feats:
            logger.warning("Skipping %s x %s: not found", m1, m2)
            continue

        f1 = feats[m1].to(device)
        f2 = feats[m2].to(device)
        n = min(f1.shape[0], f2.shape[0])
        f1, f2 = f1[:n], f2[:n]

        torch.manual_seed(42)
        perm = torch.randperm(n, device=device)
        nt = int(n * 0.8)
        st, ste = f1[perm[:nt]], f1[perm[nt:]]
        tt, tte = f2[perm[:nt]], f2[perm[nt:]]

        pl, _ = _evaluate_linear_transport(st, tt, ste, tte)
        row: dict = {"pair": f"{m1} × {m2}", "perf_linear": round(pl, 4), "tau": {}}

        for aname, acls in ARCHITECTURES.items():
            pm = eval_mlp(acls, st, tt, ste, tte, device, steps=2000)
            tau = 0.0 if pm < 1e-10 else min(1.0, max(0.0, pl / pm))
            row["tau"][aname] = round(tau, 4)
            row[f"perf_{aname}"] = round(pm, 4)
            logger.info("  [%d/10] %s×%s | %s: R²=%.4f, τ=%.4f", i + 1, m1, m2, aname, pm, tau)

        results.append(row)
        f1, f2 = f1.cpu(), f2.cpu()
        torch.cuda.empty_cache()

    return results


def generate_figure(results: list[dict], output_path: str) -> None:
    """Generate probe robustness figure."""
    matplotlib.use("Agg")
    plt.style.use("seaborn-v0_8-paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    arch_names = list(ARCHITECTURES.keys())
    x = np.arange(len(results))
    w = 0.2
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for j, an in enumerate(arch_names):
        vals = [r["tau"][an] for r in results]
        ax1.bar(
            x + (j - 1.5) * w,
            vals,
            w,
            label=an,
            color=colors[j],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    labels = []
    for r in results:
        p = r["pair"]
        for old, new in [
            ("dinov2_vitb14", "DINOv2-B"),
            ("clip_vitl14", "CLIP-L"),
            ("clip_vitb32", "CLIP-B/32"),
            ("clip_vitb16", "CLIP-B/16"),
            ("resnet50", "RN-50"),
            ("convnext_base", "CNeXt-B"),
            ("mixer_b16_224", "Mixer-B"),
            ("vit_base_patch16_224.mae", "MAE-B"),
            ("vit_base_patch16_224", "ViT-B"),
        ]:
            p = p.replace(old, new)
        labels.append(p)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax1.set_ylabel(r"Transport Linearity Score ($\tau$)", fontsize=11)
    ax1.set_ylim(-0.05, 1.15)
    ax1.axhline(0.5, color="gray", ls="--", alpha=0.5, lw=1)
    ax1.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    ax1.set_title("(a) τ across probe architectures", fontsize=11)

    all_taus = [r["tau"][an] for r in results for an in arch_names]
    ax2.hist(all_taus, bins=20, range=(0, 1), color="#666", alpha=0.7, edgecolor="white")
    ax2.set_xlabel(r"$\tau$", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("(b) τ distribution\n(all probes pooled)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info("Saved figure to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Robustness Experiment")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="data/probe_robustness.json")
    parser.add_argument("--figure", type=str, default="paper/figures/fig18_probe_robustness.pdf")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = run_experiment(Path(args.features_dir), args.device)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    n_stable = sum(
        1
        for r in results
        if all(v < 0.5 for v in r["tau"].values()) or all(v >= 0.5 for v in r["tau"].values())
    )
    print(f"Regime stability: {n_stable}/{len(results)} ({100 * n_stable / len(results):.0f}%)")

    generate_figure(results, args.figure)


if __name__ == "__main__":
    main()
