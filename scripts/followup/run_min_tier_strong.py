"""Strengthened minimal-tier run.
Fixes from v1 diagnosis: (a) larger/longer encoders (vit_b dim384/depth8, 100 epochs) to
raise same-recipe tau and test whether the C-arch contradiction with the atlas persists;
(b) bitrate arm uses NARROW widths (32,64) that bind below CIFAR-100's task dimension
(~50 classes) -- widths 128-512 did not bottleneck. Separate feature cache dir.
Usage: python run_min_tier_strong.py [epochs=100] [mlp_steps=3000]
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
import run_min_tier as base  # reuse loader/train_one/extract/tau  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
base.FEATDIR = ROOT / "data/experiments/min_tier_feats_strong"
base.FEATDIR.mkdir(parents=True, exist_ok=True)

CFGS = [
    ("B0", dict(arch="vit_b", objective="supervised", data="d1", seed=0, width=512)),
    ("MAE", dict(arch="vit_b", objective="mae", data="d1", seed=0, width=512)),
    ("ResNet", dict(arch="resnet18", objective="supervised", data="d1", seed=0, width=512)),
    ("seed1", dict(arch="vit_b", objective="supervised", data="d1", seed=1, width=512)),
    ("D2", dict(arch="vit_b", objective="supervised", data="d2", seed=0, width=512)),
    ("w64", dict(arch="vit_b", objective="supervised", data="d1", seed=0, width=64)),
    ("w32", dict(arch="vit_b", objective="supervised", data="d1", seed=0, width=32)),
]


def main(epochs, mlp_steps):
    t0 = time.time()
    feats = {}
    for name, cfg in CFGS:
        fp = base.FEATDIR / f"{name}.pt"
        if fp.exists():
            feats[name] = torch.load(fp, weights_only=True)
            print(f"loaded cached {name}", flush=True)
        else:
            e = base.train_one(cfg, epochs)
            feats[name] = base.extract(e)
            torch.save(feats[name], fp)
            del e
            torch.cuda.empty_cache()
            print(f"trained+extracted {name:7s} ({time.time() - t0:.0f}s)", flush=True)
    names = [n for n, _ in CFGS]
    tm = {
        a: {b: (1.0 if a == b else base.tau(feats[a], feats[b], mlp_steps)) for b in names}
        for a in names
    }
    print("\n===== pairwise tau (strong encoders) =====")
    print("        " + "".join(f"{n:>8s}" for n in names))
    for a in names:
        print(f"{a:7s} " + "".join(f"{tm[a][b]:8.3f}" for b in names))
    C = {
        "C-obj  (B0 vs MAE)": tm["B0"]["MAE"],
        "C-arch (B0 vs ResNet)": tm["B0"]["ResNet"],
        "C-init (B0 vs seed1)": tm["B0"]["seed1"],
        "C-data (B0 vs D2)": tm["B0"]["D2"],
        "C-bitr (B0 w512 vs w64)": tm["B0"]["w64"],
        "C-bitr (B0 w512 vs w32)": tm["B0"]["w32"],
        "same-recipe baseline (seed1<->w64, both vit_b/sup/d1)": tm["seed1"]["w64"],
    }
    print("\n===== CAUSAL CONTRASTS (tau, strong) =====")
    for k, v in C.items():
        print(f"  {k:52s} tau = {v:.3f}")
    with open(ROOT / "data/experiments/min_tier_strong_results.json", "w") as f:
        json.dump(
            {
                "epochs": epochs,
                "mlp_steps": mlp_steps,
                "arch": "vit_b",
                "tau_matrix": tm,
                "contrasts": C,
            },
            f,
            indent=2,
        )
    print("\nsaved -> data/experiments/min_tier_strong_results.json")


if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    main(ep, ms)
