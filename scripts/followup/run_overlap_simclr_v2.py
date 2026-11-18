"""Data-overlap law under a contrastive objective, second attempt (SimCLR, standard CIFAR recipe).

The first attempt (run_overlap_simclr.py) used a small ViT, weak augmentation, and a small batch,
and its encoders were unstable (effective rank 20-209). This version uses the standard CIFAR
SimCLR recipe: ResNet-18 with CIFAR stem, 2-layer BN projection head (512-2048-128), NT-Xent
(temperature 0.5), batch 512, SGD (lr 0.06*batch/256, momentum 0.9, wd 5e-4), cosine schedule,
RandomResizedCrop(0.08-1) + flip + color jitter (p=0.8) + grayscale (p=0.2).
Each encoder is trained on a 50-class window [lo, lo+50) of CIFAR-100 train; the reference window
is [0, 50). Representation = 512-d backbone output on the first 2,000 CIFAR-100 test images.
Quality gate: k-NN accuracy (k=20, cosine) on the window's own test classes, and effective rank.

Objective control: with OBJ=sup the same script trains the same ResNet-18 on the same windows
and seeds with cross-entropy (crop+flip augmentation, SGD lr 0.1, 200 epochs), so the supervised
and SimCLR curves differ only in the objective.

Augmentation control: with OBJ=sup and AUG=strong (4th argument) the supervised run uses the
SimCLR augmentation, so objective and augmentation strength can be separated.
Mode "mid": the same for the 74/50/26% windows (6 pairs per window).
Mode "zero": trains seeds 1 and 2 of the 0%-overlap window and reports tau and both R^2 values
for every reference-seed x 0%-seed pair (2 x 3 = 6) and for the two reference seeds (100%).

Usage:
  CUDA_VISIBLE_DEVICES=2 python run_overlap_simclr_v2.py gate [epochs] [simclr|sup] [std|strong]
  CUDA_VISIBLE_DEVICES=2 python run_overlap_simclr_v2.py full [epochs] [simclr|sup] [std|strong]
  CUDA_VISIBLE_DEVICES=2 python run_overlap_simclr_v2.py zero [epochs] [simclr|sup] [std|strong]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms as T
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.bitrate import effective_rank  # noqa: E402
from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
OBJ = sys.argv[3] if len(sys.argv) > 3 else "simclr"
AUG = sys.argv[4] if len(sys.argv) > 4 else "std"
TAG = "simclr_v2" if OBJ == "simclr" else ("sup_r18" if AUG == "std" else "sup_r18_strongaug")
FEAT = ROOT / f"data/experiments/overlap_{TAG}_feats"
OUT = ROOT / f"data/experiments/overlap_{TAG}_curve.json"
ZERO_OUT = ROOT / f"data/experiments/overlap_{TAG}_zero.json"
MID_OUT = ROOT / f"data/experiments/overlap_{TAG}_mid.json"
BATCH, NPROBE, TEMP = 512, 2000, 0.5
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
# name -> (window start, seed); shared classes with [0,50) = max(0, 50 - lo)
RUNS = {"ref_s0": (0, 0), "ref_s1": (0, 1), "ov74": (13, 0), "ov50": (25, 0), "ov26": (37, 0), "ov0": (50, 0),
        "ov0_s1": (50, 1), "ov0_s2": (50, 2),
        **{f"ov{w}_s{sd}": (lo, sd) for w, lo in ((74, 13), (50, 25), (26, 37)) for sd in (1, 2)}}
OVERLAP = {"ref_s1": 100, "ov74": 74, "ov50": 50, "ov26": 26, "ov0": 0}


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, 32, 32) -> (B, 512)
        return self.net(x)


def aug() -> T.Compose:
    return T.Compose([T.RandomResizedCrop(32, scale=(0.08, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(0.2),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class TwoView(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """SimCLR: two augmented views. Supervised (OBJ=sup): one view and the window-relative label."""

    def __init__(self, lo: int) -> None:
        self.ds = datasets.CIFAR100(str(ROOT / "data"), train=True, download=False)
        self.idx = [i for i, t in enumerate(self.ds.targets) if lo <= t < lo + 50]
        self.lo = lo
        self.t = aug() if (OBJ == "simclr" or AUG == "strong") else T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                                                          T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, y = self.ds[self.idx[k]]
        if OBJ == "simclr":
            return self.t(img), self.t(img)
        return self.t(img), torch.tensor(y - self.lo)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / TEMP
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def test_split(lo: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.CIFAR100(str(ROOT / "data"), train=False, download=False,
                           transform=T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)]))
    xs, ys = zip(*[ds[i] for i in range(len(ds)) if lo is None or lo <= ds.targets[i] < lo + 50], strict=True)
    return torch.stack(xs), torch.tensor(ys)


@torch.no_grad()
def embed(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    net.eval()
    out = torch.cat([net(x[i:i + 1000].to(DEV)) for i in range(0, len(x), 1000)])
    net.train()
    return out


@torch.no_grad()
def knn_acc(net: nn.Module, lo: int) -> float:
    """k-NN (k=20, cosine) accuracy on the window's test classes, 50/50 split of that test set."""
    x, y = test_split(lo)
    f = F.normalize(embed(net, x), dim=1)
    y = y.to(DEV)
    half = len(y) // 2
    sim = f[half:] @ f[:half].T
    pred = torch.mode(y[:half][sim.topk(20, dim=1).indices], dim=1).values
    return float((pred == y[half:]).float().mean())


def train(lo: int, seed: int, epochs: int) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Backbone().to(DEV)
    head = (nn.Sequential(nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                          nn.Linear(2048, 128)) if OBJ == "simclr" else nn.Linear(512, 50)).to(DEV)
    ld = torch.utils.data.DataLoader(TwoView(lo), BATCH, shuffle=True, num_workers=8, drop_last=True,
                                     persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=(0.06 if OBJ == "simclr" else 0.1) * BATCH / 256, momentum=0.9,
                          weight_decay=5e-4)
    total = epochs * len(ld)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    log = []
    for ep in range(epochs):
        tot = 0.0
        for v1, v2 in ld:
            v1, v2 = v1.to(DEV, non_blocking=True), v2.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
                loss = (nt_xent(head(net(v1)).float(), head(net(v2)).float()) if OBJ == "simclr"
                        else F.cross_entropy(head(net(v1)).float(), v2))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            acc = knn_acc(net, lo)
            log.append({"epoch": ep + 1, "loss": round(tot / len(ld), 4), "knn_acc": round(acc, 4)})
            print(f"  lo={lo} seed={seed} ep={ep + 1} loss={tot / len(ld):.3f} knn={acc:.3f}", flush=True)
    return net, log


def features(name: str, epochs: int, probe: torch.Tensor) -> tuple[torch.Tensor, dict]:
    fp = FEAT / f"{name}.pt"
    if fp.exists():
        d = torch.load(fp, weights_only=False)
        return d["features"], d["meta"]
    lo, seed = RUNS[name]
    t0 = time.time()
    net, log = train(lo, seed, epochs)
    f = embed(net, probe).float().cpu()
    meta = {"lo": lo, "seed": seed, "epochs": epochs, "log": log, "minutes": round((time.time() - t0) / 60, 1),
            "effective_rank": round(float(effective_rank(f)), 1)}
    torch.save({"features": f, "meta": meta}, fp)
    print(f"{name}: erank={meta['effective_rank']} final_knn={log[-1]['knn_acc']} ({meta['minutes']} min)", flush=True)
    return f, meta


def tau(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.to(DEV), b.to(DEV)
    return min(tls(a, b, mlp_steps=3000).tau, tls(b, a, mlp_steps=3000).tau)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    FEAT.mkdir(parents=True, exist_ok=True)
    probe = test_split()[0][:NPROBE]
    names = {"gate": ["ref_s0"], "full": [n for n in RUNS if "_s" not in n or n.startswith("ref")],
             "zero": ["ref_s0", "ref_s1", "ov0", "ov0_s1", "ov0_s2"],
             "mid": ["ref_s0", "ref_s1"] + [f"ov{w}{sfx}" for w in (74, 50, 26) for sfx in ("", "_s1", "_s2")]}[mode]
    feats, metas = {}, {}
    for n in names:
        feats[n], metas[n] = features(n, epochs, probe)
    if mode == "gate":
        return
    if mode == "mid":
        out = {}
        for w in (74, 50, 26):
            rows = []
            for r in ("ref_s0", "ref_s1"):
                for v in (f"ov{w}", f"ov{w}_s1", f"ov{w}_s2"):
                    fa, fb = feats[r].to(DEV), feats[v].to(DEV)
                    ab, ba = tls(fa, fb, mlp_steps=3000), tls(fb, fa, mlp_steps=3000)
                    rows.append({"a": r, "b": v, "tau": round(min(ab.tau, ba.tau), 4),
                                 "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                                 "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)]})
            t = [x["tau"] for x in rows]
            out[str(w)] = {"pairs": rows, "tau_mean": round(float(np.mean(t)), 4),
                           "tau_range": [round(min(t), 4), round(max(t), 4)]}
            print(f"overlap {w}%: tau mean {np.mean(t):.4f} range [{min(t):.4f}, {max(t):.4f}]", flush=True)
        with open(MID_OUT, "w") as f:
            json.dump({"objective": OBJ, "aug": AUG if OBJ == "sup" else "simclr", "epochs": epochs,
                       "windows": out, "encoders": metas}, f, indent=1)
        return
    if mode == "zero":
        rows = []
        pairs = [("ref_s0", "ref_s1")] + [(r, o) for r in ("ref_s0", "ref_s1") for o in ("ov0", "ov0_s1", "ov0_s2")]
        for a, b in pairs:
            fa, fb = feats[a].to(DEV), feats[b].to(DEV)
            ab, ba = tls(fa, fb, mlp_steps=3000), tls(fb, fa, mlp_steps=3000)
            rows.append({"a": a, "b": b, "overlap": 100 if b.startswith("ref") else 0,
                         "tau": round(min(ab.tau, ba.tau), 4),
                         "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                         "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)]})
            print(rows[-1], flush=True)
        zero = [r["tau"] for r in rows if r["overlap"] == 0]
        with open(ZERO_OUT, "w") as f:
            json.dump({"objective": OBJ, "aug": AUG if OBJ == "sup" else "simclr", "epochs": epochs, "pairs": rows,
                       "tau_0pct_mean": round(float(np.mean(zero)), 4),
                       "tau_0pct_range": [round(min(zero), 4), round(max(zero), 4)], "encoders": metas}, f, indent=1)
        print("zero-overlap tau", zero, "mean", round(float(np.mean(zero)), 4))
        return
    curve = sorted((OVERLAP[n], round(tau(feats["ref_s0"], feats[n]), 4)) for n in OVERLAP)
    from scipy.stats import spearmanr

    rho = float(spearmanr([c[0] for c in curve], [c[1] for c in curve]).correlation)
    print("curve", curve, "spearman", rho)
    with open(OUT, "w") as f:
        json.dump({"objective": OBJ, "arch": "resnet18-cifar", "epochs": epochs, "curve": curve,
                   "spearman": rho, "encoders": metas}, f, indent=1)


if __name__ == "__main__":
    main()
