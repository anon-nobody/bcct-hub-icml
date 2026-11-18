"""Overlap law under SimSiam (stable SSL: stop-grad, no negatives).
Same design as run_data_overlap.py but objective=simsiam. Tests whether the
'transport requires shared data' law generalizes to a self-supervised objective,
using a recipe that does not collapse at small batch (unlike the SimCLR attempt).
Usage: python run_overlap_simsiam.py [epochs=100]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "followup"))
sys.path.insert(0, str(ROOT / "src"))
from train_controlled import Encoder, build_objective, set_seed  # noqa
from run_overlap_simclr import ssl_window_loader  # reuse two-crop window loader (guarded import)  # noqa
import run_min_tier as base  # noqa
from latent.metrics.transport import transport_linearity_score as TLS  # noqa

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FEAT = ROOT / "data/experiments/overlap_simsiam_feats"
FEAT.mkdir(parents=True, exist_ok=True)
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def train_simsiam(lo, hi, seed):
    set_seed(seed)
    enc = Encoder("vit_b", 512).to(DEV)
    extra, step = build_objective("simsiam", enc, 512, DEV)
    ld = ssl_window_loader(lo, hi)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP * len(ld))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    for _ in range(EP):
        for v1, v2 in ld:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
                loss = step((v1.to(DEV), v2.to(DEV)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return enc.eval()


def tau(A, B):
    A, B = A.to(DEV), B.to(DEV)
    return round(min(TLS(A, B, mlp_steps=3000).tau, TLS(B, A, mlp_steps=3000).tau), 4)


def erank(X):
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    p = s / s.sum()
    p = p[p > 1e-12]
    return round(float(torch.exp(-(p * p.log()).sum())), 1)


JOBS = {
    "ref_s0": (0, 50, 0),
    "ref_s1": (0, 50, 1),
    "ov74": (13, 63, 0),
    "ov50": (25, 75, 0),
    "ov26": (37, 87, 0),
    "ov0": (50, 100, 0),
}


def main():
    t0 = time.time()
    F = {}
    for name, (lo, hi, sd) in JOBS.items():
        fp = FEAT / f"{name}.pt"
        if fp.exists():
            F[name] = torch.load(fp, weights_only=True)
        else:
            e = train_simsiam(lo, hi, sd)
            F[name] = base.extract(e)
            torch.save(F[name], fp)
            del e
            torch.cuda.empty_cache()
            print(f"trained {name} eff_rank={erank(F[name])} ({time.time() - t0:.0f}s)", flush=True)
    print("eff_ranks:", {k: erank(v) for k, v in F.items()}, flush=True)  # stability check
    ref = F["ref_s0"]
    ov = {"ref_s1": 100, "ov74": 74, "ov50": 50, "ov26": 26, "ov0": 0}
    curve = []
    for name in ["ref_s1", "ov74", "ov50", "ov26", "ov0"]:
        t = tau(ref, F[name])
        curve.append((ov[name], t))
        print(f"  SimSiam shared {ov[name]:3d}% -> tau = {t:.3f}", flush=True)
    from scipy.stats import spearmanr

    rho = spearmanr([o for o, _ in curve], [t for _, t in curve]).correlation
    print(f"\nSimSiam overlap law: Spearman(overlap%, tau) = {rho:.3f}")
    with open(ROOT / "data/experiments/overlap_simsiam_curve.json", "w") as f:
        json.dump(
            {
                "objective": "simsiam",
                "epochs": EP,
                "curve": curve,
                "spearman": rho,
                "eff_ranks": {k: erank(v) for k, v in F.items()},
            },
            f,
            indent=2,
        )
    print("saved -> data/experiments/overlap_simsiam_curve.json")


if __name__ == "__main__":
    main()
