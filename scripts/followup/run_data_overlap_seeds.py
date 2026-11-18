"""Seed replicates for the supervised-ViT CIFAR-100 data-overlap curve.

run_data_overlap.py trains one seed per intermediate window. Here seeds 1 and 2 of the 74/50/26%
windows are added (vit_b, supervised, W=512, 100 epochs, same recipe and probe as the original),
and tau with both reference seeds is computed for every window:
  100%: reference seeds B0_s0, B0_s1 (seed1), B0_s2 -> 3 pairs
  74/50/26%: {B0_s0, B0_s1} x {window seed 0, 1, 2} -> 6 pairs each
  0%: {B0_s0, B0_s1} x {D2 seed 0, 1, 2} -> 6 pairs (D2 = classes 50-99, disjoint)
Per-direction Procrustes / MLP R^2 is stored with every pair.
Output: data/experiments/data_overlap_seeds.json
Usage: CUDA_VISIBLE_DEVICES=3 python run_data_overlap_seeds.py
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
from torchvision import datasets  # noqa: E402
from torchvision import transforms as T  # noqa: E402
from train_controlled import Encoder, build_objective, set_seed  # noqa: E402

from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
STRONG = ROOT / "data/experiments/min_tier_feats_strong"
CI = ROOT / "data/experiments/seed_ci_feats"
OV = ROOT / "data/experiments/overlap_feats"
OUT = ROOT / "data/experiments/data_overlap_seeds.json"
WINDOWS = {74: 13, 50: 25, 26: 37}  # shared classes (%) -> window start; window = [lo, lo + 50)


def train_window(lo: int, hi: int, seed: int, epochs: int = 100) -> torch.nn.Module:
    """Identical to run_data_overlap.train_window (supervised vit_b, W=512) with a seed argument."""
    aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    ds = datasets.CIFAR100(str(ROOT / "data"), train=True, download=False, transform=aug)
    idx = [i for i, t in enumerate(ds.targets) if lo <= t < hi]
    ld = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), 256, shuffle=True, num_workers=4,
                                     drop_last=True)
    set_seed(seed)
    enc = Encoder("vit_b", 512).to(DEV)
    extra, step = build_objective("supervised", enc, 512, DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(ld))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    for _ in range(epochs):
        for x, y in ld:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
                loss = step((x.to(DEV), y.to(DEV)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return enc.eval()


def load(p: Path) -> torch.Tensor:
    return torch.load(p, weights_only=True)


def main() -> None:
    t0 = time.time()
    feats: dict[str, torch.Tensor] = {}
    for w, lo in WINDOWS.items():
        feats[f"ov{w}_s0"] = load(OV / f"ov{w}.pt")
        for sd in (1, 2):
            fp = OV / f"ov{w}_s{sd}.pt"
            if not fp.exists():
                enc = train_window(lo, lo + 50, sd)
                torch.save(base.extract(enc), fp)
                del enc
                torch.cuda.empty_cache()
                print(f"trained ov{w}_s{sd} ({time.time() - t0:.0f}s)", flush=True)
            feats[f"ov{w}_s{sd}"] = load(fp)
    refs = {"B0_s0": load(STRONG / "B0.pt"), "B0_s1": load(STRONG / "seed1.pt"), "B0_s2": load(CI / "B0_s2.pt")}
    zero = {"D2_s0": load(STRONG / "D2.pt"), "D2_s1": load(CI / "D2_s1.pt"), "D2_s2": load(CI / "D2_s2.pt")}

    def pair(a: torch.Tensor, b: torch.Tensor) -> dict:
        a, b = a.to(DEV), b.to(DEV)
        ab, ba = tls(a, b, mlp_steps=3000), tls(b, a, mlp_steps=3000)
        return {"tau": round(min(ab.tau, ba.tau), 4), "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)]}

    out = {"100": [{"a": a, "b": b, **pair(refs[a], refs[b])}
                   for a, b in (("B0_s0", "B0_s1"), ("B0_s0", "B0_s2"), ("B0_s1", "B0_s2"))]}
    for w in WINDOWS:
        out[str(w)] = [{"a": r, "b": f"ov{w}_s{sd}", **pair(refs[r], feats[f"ov{w}_s{sd}"])}
                       for r in ("B0_s0", "B0_s1") for sd in range(3)]
    out["0"] = [{"a": r, "b": z, **pair(refs[r], zero[z])} for r in ("B0_s0", "B0_s1") for z in zero]
    summary = {}
    for k, rows in out.items():
        t = [x["tau"] for x in rows]
        summary[k] = {"n": len(t), "tau_mean": round(float(np.mean(t)), 4), "tau_range": [min(t), max(t)]}
        print(f"overlap {k:>3s}%: n={len(t)} tau mean {np.mean(t):.4f} range [{min(t):.4f}, {max(t):.4f}]", flush=True)
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "pairs": out, "epochs": 100}, f, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
