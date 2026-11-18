"""Seed-replication CIs for the headline causal contrasts.
Builds 3 seed instances (0,1,2) of B0, MAE, D2, then reports tau distributions:
  C-init = tau(B0_i, B0_j)  i<j        (same recipe, diff seed)  -> should stay HIGH
  C-obj  = tau(B0_i, MAE_j) all i,j    (objective swap)          -> should be ~0
  C-data = tau(B0_i, D2_j)  all i,j    (data swap)               -> should be ~0
Reuses strong cache for seed0 of each + B0 seed1; trains only the 5 missing instances.
95% CI via percentile bootstrap over the pairwise tau values (n noted; pseudo-replication
acknowledged since pairs share encoders).
Usage: python run_seed_ci.py [epochs=100]
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
import run_min_tier as base  # noqa
from latent.metrics.transport import transport_linearity_score as TLS  # noqa

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STRONG = ROOT / "data/experiments/min_tier_feats_strong"
CI = ROOT / "data/experiments/seed_ci_feats"
CI.mkdir(parents=True, exist_ok=True)
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# instances to train (missing from strong cache); strong cache already has seed0 of each + B0 s1
TRAIN = {
    "B0_s2": dict(arch="vit_b", objective="supervised", data="d1", seed=2, width=512),
    "MAE_s1": dict(arch="vit_b", objective="mae", data="d1", seed=1, width=512),
    "MAE_s2": dict(arch="vit_b", objective="mae", data="d1", seed=2, width=512),
    "D2_s1": dict(arch="vit_b", objective="supervised", data="d2", seed=1, width=512),
    "D2_s2": dict(arch="vit_b", objective="supervised", data="d2", seed=2, width=512),
}
t0 = time.time()
F = {}
for name, cfg in TRAIN.items():
    fp = CI / f"{name}.pt"
    if fp.exists():
        F[name] = torch.load(fp, weights_only=True)
    else:
        e = base.train_one(cfg, EP)
        F[name] = base.extract(e)
        torch.save(F[name], fp)
        del e
        torch.cuda.empty_cache()
        print(f"trained {name} ({time.time() - t0:.0f}s)", flush=True)

# assemble seed instances
B0 = {
    0: torch.load(STRONG / "B0.pt", weights_only=True),
    1: torch.load(STRONG / "seed1.pt", weights_only=True),
    2: F["B0_s2"],
}
MAE = {0: torch.load(STRONG / "MAE.pt", weights_only=True), 1: F["MAE_s1"], 2: F["MAE_s2"]}
D2 = {0: torch.load(STRONG / "D2.pt", weights_only=True), 1: F["D2_s1"], 2: F["D2_s2"]}


def tau(A, B):
    A, B = A.to(DEV), B.to(DEV)
    return min(TLS(A, B, mlp_steps=3000).tau, TLS(B, A, mlp_steps=3000).tau)


def boot_ci(vals, n=2000, seed=0):
    r = np.random.default_rng(seed)
    v = np.array(vals)
    m = [r.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return (
        round(float(v.mean()), 3),
        round(float(np.percentile(m, 2.5)), 3),
        round(float(np.percentile(m, 97.5)), 3),
    )


Cinit = [tau(B0[i], B0[j]) for i in range(3) for j in range(i + 1, 3)]  # n=3
Cobj = [tau(B0[i], MAE[j]) for i in range(3) for j in range(3)]  # n=9
Cdata = [tau(B0[i], D2[j]) for i in range(3) for j in range(3)]  # n=9

out = {}
for name, vals in [
    ("C-init (same recipe, diff seed)", Cinit),
    ("C-obj (objective swap)", Cobj),
    ("C-data (data swap)", Cdata),
]:
    m, lo, hi = boot_ci(vals)
    out[name] = {"n": len(vals), "mean": m, "ci95": [lo, hi], "vals": [round(v, 3) for v in vals]}
    print(
        f"  {name:34s} n={len(vals)}  mean={m:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  vals={[round(v, 3) for v in vals]}",
        flush=True,
    )

# separation test: is C-init CI above C-data CI? (gauge = data, not init)
init_lo = out["C-init (same recipe, diff seed)"]["ci95"][0]
data_hi = out["C-data (data swap)"]["ci95"][1]
print(
    f"\nSEPARATION: C-init CI-low ({init_lo}) vs C-data CI-high ({data_hi}) -> "
    f"{'DISJOINT (init >> data, gauge=data confirmed)' if init_lo > data_hi else 'OVERLAP'}"
)
with open(ROOT / "data/experiments/seed_ci_results.json", "w") as f:
    json.dump(
        {"epochs": EP, "contrasts": out, "init_low": init_lo, "data_high": data_hi}, f, indent=2
    )
print("saved -> data/experiments/seed_ci_results.json")
