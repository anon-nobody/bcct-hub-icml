"""Seed-replication CIs for the architecture and width contrasts (v2 open item).

Trains seeds 1,2 of ResNet (resnet18), w64, w32 (vit_b, supervised, d1, 100 ep) and reports
  C-arch  = tau(B0_i, ResNet_j)   all i,j            (architecture swap)
  C-w64   = tau(B0_i, w64_j)      all i,j            (width 512 -> 64)
  C-w32   = tau(B0_i, w32_j)      all i,j            (width 512 -> 32)
plus same-architecture controls:
  R-init  = tau(ResNet_i, ResNet_j) i<j              (is ResNet self-consistent across seeds?)
  w64-init= tau(w64_i, w64_j) i<j
If R-init is high while C-arch is low, the architecture swap genuinely breaks transport; if
R-init is also low, the C-arch drop reflects ResNet run-to-run variability, not architecture.
Reuses cached seed-0 features (min_tier_feats_strong) and B0 seeds (seed_ci_feats).
Usage: CUDA_VISIBLE_DEVICES=2 python run_seed_ci_arch.py [epochs=100]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "followup"))
sys.path.insert(0, str(ROOT / "src"))
import run_min_tier as base  # noqa: E402

from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STRONG = ROOT / "data/experiments/min_tier_feats_strong"
CI = ROOT / "data/experiments/seed_ci_feats"
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = ROOT / "data/experiments/seed_ci_arch_results.json"

TRAIN = {
    f"{k}_s{s}": dict(arch=a, objective="supervised", data="d1", seed=s, width=w)
    for k, a, w in [("ResNet", "resnet18", 512), ("w64", "vit_b", 64), ("w32", "vit_b", 32)]
    for s in (1, 2)
}


def load(p: Path) -> torch.Tensor:
    return torch.load(p, weights_only=True)


def tau(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.to(DEV), b.to(DEV)
    return min(tls(a, b, mlp_steps=3000).tau, tls(b, a, mlp_steps=3000).tau)


def boot_ci(vals: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    r = np.random.default_rng(seed)
    v = np.array(vals)
    m = [r.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return round(float(v.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)


def main() -> None:
    t0 = time.time()
    feats: dict[str, torch.Tensor] = {}
    for name, cfg in TRAIN.items():
        fp = CI / f"{name}.pt"
        if fp.exists():
            feats[name] = load(fp)
            continue
        enc = base.train_one(cfg, EP)
        feats[name] = base.extract(enc)
        torch.save(feats[name], fp)
        del enc
        torch.cuda.empty_cache()
        print(f"trained {name} ({time.time() - t0:.0f}s)", flush=True)

    b0 = {0: load(STRONG / "B0.pt"), 1: load(STRONG / "seed1.pt"), 2: load(CI / "B0_s2.pt")}
    grp = {k: {0: load(STRONG / f"{k}.pt"), 1: feats[f"{k}_s1"], 2: feats[f"{k}_s2"]}
           for k in ("ResNet", "w64", "w32")}

    rows = {
        "C-arch (B0 vs ResNet)": [tau(b0[i], grp["ResNet"][j]) for i in range(3) for j in range(3)],
        "C-w64 (B0 vs w64)": [tau(b0[i], grp["w64"][j]) for i in range(3) for j in range(3)],
        "C-w32 (B0 vs w32)": [tau(b0[i], grp["w32"][j]) for i in range(3) for j in range(3)],
        "R-init (ResNet seeds)": [tau(grp["ResNet"][i], grp["ResNet"][j]) for i in range(3) for j in range(i + 1, 3)],
        "w64-init (w64 seeds)": [tau(grp["w64"][i], grp["w64"][j]) for i in range(3) for j in range(i + 1, 3)],
        "w32-init (w32 seeds)": [tau(grp["w32"][i], grp["w32"][j]) for i in range(3) for j in range(i + 1, 3)],
    }
    out = {"epochs": EP, "contrasts": {}}
    for name, vals in rows.items():
        m, lo, hi = boot_ci(vals)
        out["contrasts"][name] = {"n": len(vals), "mean": m, "ci95": [lo, hi], "vals": [round(v, 3) for v in vals]}
        print(f"  {name:26s} n={len(vals)} mean={m:.3f} CI [{lo:.3f},{hi:.3f}] {[round(v, 3) for v in vals]}", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
