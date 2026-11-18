"""Data-overlap gradient: quantify the lineage->transport LAW.
Fix recipe (vit_b/supervised/seed0/w512/100ep); vary the pretraining class window's
overlap with B0's window [0,50). Measure tau(B0, variant) vs % shared classes.
Reuses cached strong features for the endpoints: seed1 (100% data, diff run), D2 (0%).
Only the 3 intermediate windows are trained here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "followup"))
sys.path.insert(0, str(ROOT / "src"))
from train_controlled import Encoder, build_objective, set_seed  # noqa
import run_min_tier as base  # noqa
from latent.metrics.transport import transport_linearity_score as TLS  # noqa
from torchvision import datasets, transforms as T  # noqa

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STRONG = ROOT / "data/experiments/min_tier_feats_strong"
FEAT = ROOT / "data/experiments/overlap_feats"
FEAT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def window_loader(lo, hi):
    aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    ds = datasets.CIFAR100(str(ROOT / "data"), train=True, download=False, transform=aug)
    idx = [i for i, t in enumerate(ds.targets) if lo <= t < hi]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx), 256, shuffle=True, num_workers=4, drop_last=True
    )


def train_window(lo, hi, seed=0):
    set_seed(seed)
    enc = Encoder("vit_b", 512).to(DEV)
    extra, step = build_objective("supervised", enc, 512, DEV)
    ld = window_loader(lo, hi)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * len(ld))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    for _ in range(EPOCHS):
        for x, y in ld:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
                loss = step((x.to(DEV), y.to(DEV)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return enc.eval()


# windows [lo,hi) and their overlap with B0 [0,50): shared = max(0, 50-lo) classes
WINDOWS = {"ov74": (13, 63), "ov50": (25, 75), "ov26": (37, 87)}
t0 = time.time()
feats = {}
for name, (lo, hi) in WINDOWS.items():
    fp = FEAT / f"{name}.pt"
    if fp.exists():
        feats[name] = torch.load(fp, weights_only=True)
    else:
        e = train_window(lo, hi)
        feats[name] = base.extract(e)
        torch.save(feats[name], fp)
        del e
        torch.cuda.empty_cache()
        print(f"trained {name} [{lo},{hi}) ({time.time() - t0:.0f}s)", flush=True)
# endpoints from strong cache
B0 = torch.load(STRONG / "B0.pt", weights_only=True)
feats["ov100"] = torch.load(STRONG / "seed1.pt", weights_only=True)  # 100% data, different run
feats["ov0"] = torch.load(STRONG / "D2.pt", weights_only=True)  # 0% overlap

overlap_pct = {"ov100": 100, "ov74": 74, "ov50": 50, "ov26": 26, "ov0": 0}
curve = []
for name in ["ov100", "ov74", "ov50", "ov26", "ov0"]:
    tau = round(
        min(
            TLS(B0.to(DEV), feats[name].to(DEV), mlp_steps=3000).tau,
            TLS(feats[name].to(DEV), B0.to(DEV), mlp_steps=3000).tau,
        ),
        4,
    )
    curve.append((overlap_pct[name], tau))
    print(f"  shared-classes {overlap_pct[name]:3d}% -> tau(B0, {name}) = {tau:.3f}", flush=True)

rho = spearmanr([o for o, _ in curve], [t for _, t in curve]).correlation
print(f"\nLINEAGE->TRANSPORT LAW: Spearman(overlap%, tau) = {rho:.3f}")
with open(ROOT / "data/experiments/data_overlap_curve.json", "w") as f:
    json.dump({"curve": curve, "spearman": rho, "epochs": EPOCHS}, f, indent=2)
print("saved -> data/experiments/data_overlap_curve.json")
