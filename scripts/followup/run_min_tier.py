"""Minimal-tier controlled experiment (FIXED).
Train 7 encoders (one knob apart), extract features on a fixed probe, compute pairwise
transport tau with the REPO-VALIDATED metric, read the 5 causal contrasts.

Fixes vs first version:
  - tau uses latent.metrics.transport.transport_linearity_score (standardizes features,
    3000 MLP steps) instead of a hand-rolled estimator that gave spurious 0s.
  - module-level run block guarded under __main__ (importing this file no longer retrains).
  - each encoder's probe features are cached to data/experiments/min_tier_feats/ so metric
    changes never require retraining (delete the dir to force a fresh train).
Uses the existing CIFAR-100 in data/ (no download).
Usage: python run_min_tier.py [epochs=40] [mlp_steps=3000]
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
from train_controlled import Encoder, build_objective, set_seed  # noqa: E402

from latent.metrics.transport import transport_linearity_score as TLS  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FEATDIR = ROOT / "data/experiments/min_tier_feats"
FEATDIR.mkdir(parents=True, exist_ok=True)
BATCH, NPROBE = 256, 2000

CFGS = [
    ("B0", dict(arch="vit_s", objective="supervised", data="d1", seed=0, width=512)),
    ("MAE", dict(arch="vit_s", objective="mae", data="d1", seed=0, width=512)),
    ("ResNet", dict(arch="resnet18", objective="supervised", data="d1", seed=0, width=512)),
    ("seed1", dict(arch="vit_s", objective="supervised", data="d1", seed=1, width=512)),
    ("D2", dict(arch="vit_s", objective="supervised", data="d2", seed=0, width=512)),
    ("w256", dict(arch="vit_s", objective="supervised", data="d1", seed=0, width=256)),
    ("w128", dict(arch="vit_s", objective="supervised", data="d1", seed=0, width=128)),
]

from torchvision import datasets  # noqa: E402
from torchvision import transforms as T  # noqa: E402


def loader(data, train=True):
    aug = (
        T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        if train
        else T.Compose([T.ToTensor()])
    )
    ds = datasets.CIFAR100(str(ROOT / "data"), train=train, download=False, transform=aug)
    if data in ("d1", "d2") and train:
        lo, hi = (0, 50) if data == "d1" else (50, 100)
        ds = torch.utils.data.Subset(ds, [i for i, t in enumerate(ds.targets) if lo <= t < hi])
    return torch.utils.data.DataLoader(ds, BATCH, shuffle=train, num_workers=4, drop_last=train)


def train_one(cfg, epochs):
    set_seed(cfg["seed"])
    enc = Encoder(cfg["arch"], cfg["width"]).to(DEV)
    extra, step = build_objective(cfg["objective"], enc, cfg["width"], DEV)
    ld = loader(cfg["data"], True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(ld))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    for _ in range(epochs):
        for x, y in ld:
            batch = (x.to(DEV),) if cfg["objective"] == "mae" else (x.to(DEV), y.to(DEV))
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
                loss = step(batch)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return enc.eval()


@torch.no_grad()
def extract(enc):
    ld = loader("all", train=False)
    feats = []
    for x, _ in ld:
        with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
            feats.append(enc(x.to(DEV)).float().cpu())
        if sum(f.shape[0] for f in feats) >= NPROBE:
            break
    return torch.cat(feats)[:NPROBE]


def tau(A, B, mlp_steps):
    A, B = A.to(DEV), B.to(DEV)
    return round(min(TLS(A, B, mlp_steps=mlp_steps).tau, TLS(B, A, mlp_steps=mlp_steps).tau), 4)


def main(epochs, mlp_steps):
    t0 = time.time()
    feats = {}
    for name, cfg in CFGS:
        fp = FEATDIR / f"{name}.pt"
        if fp.exists():
            feats[name] = torch.load(fp, weights_only=True)
            print(f"loaded cached {name}", flush=True)
        else:
            e = train_one(cfg, epochs)
            feats[name] = extract(e)
            torch.save(feats[name], fp)
            del e
            torch.cuda.empty_cache()
            print(f"trained+extracted {name:7s} ({time.time() - t0:.0f}s)", flush=True)
    names = [n for n, _ in CFGS]
    tm = {
        a: {b: (1.0 if a == b else tau(feats[a], feats[b], mlp_steps)) for b in names}
        for a in names
    }
    print("\n===== pairwise tau (repo-validated metric) =====")
    print("        " + "".join(f"{n:>8s}" for n in names))
    for a in names:
        print(f"{a:7s} " + "".join(f"{tm[a][b]:8.3f}" for b in names))
    C = {
        "C-obj  (B0 vs MAE)": tm["B0"]["MAE"],
        "C-arch (B0 vs ResNet)": tm["B0"]["ResNet"],
        "C-init (B0 vs seed1)": tm["B0"]["seed1"],
        "C-data (B0 vs D2)": tm["B0"]["D2"],
        "C-bitr (B0 vs w256)": tm["B0"]["w256"],
        "C-bitr (B0 vs w128)": tm["B0"]["w128"],
    }
    print("\n===== CAUSAL CONTRASTS (tau) =====")
    for k, v in C.items():
        print(f"  {k:30s} tau = {v:.3f}")
    with open(ROOT / "data/experiments/min_tier_results.json", "w") as f:
        json.dump(
            {"epochs": epochs, "mlp_steps": mlp_steps, "tau_matrix": tm, "contrasts": C},
            f,
            indent=2,
        )
    print("\nsaved -> data/experiments/min_tier_results.json")


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    main(ep, ms)
