"""Architecture contrast at 64x64 with one shared recipe (second attempt).

The first Tiny-ImageNet factorial (run_tin64_factorial.py) compared a ResNet-50 trained with SGD
and light augmentation against a ViT-S/8 trained with AdamW; the ViT-S was weak (20-NN accuracy
0.38) and its seeds barely predicted each other, so that contrast was inconclusive. Here every
architecture uses the same recipe, so the architecture is the only difference:
  ResNet-50 (max-pool removed), ConvNeXt-T (torchvision), ViT-S/4 (4x4 patches, 16x16 = 256 tokens)
  supervised on Tiny-ImageNet classes 0-99, AdamW (lr 1e-3, wd 0.05), batch 256, 150 epochs,
  10-epoch warm-up + cosine, label smoothing 0.1, RandomResizedCrop(0.35-1) + flip +
  RandAugment(2, 9) + RandomErasing(0.25), gradient clipping 5, bfloat16; three seeds each.
Analysis: worst-direction tau plus per-direction Procrustes / MLP R^2 for every pair of encoders
(3 within-architecture seed pairs per architecture, 9 pairs per architecture pair).

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_tin64_arch2.py train r50,cnx
  CUDA_VISIBLE_DEVICES=1 python run_tin64_arch2.py train vit
  CUDA_VISIBLE_DEVICES=0 python run_tin64_arch2.py analyze
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models import convnext_tiny, resnet50

sys.path.insert(0, str(Path(__file__).parent))
import run_tin64_factorial as base  # noqa: E402

DEV = base.DEV
FEAT = Path(os.environ.get("ARCH2_FEAT", base.ROOT / "data/experiments/tin64_arch2_feats"))
OUT = base.ROOT / "data/experiments/tin64_arch2_results.json"
EPOCHS = int(os.environ.get("EPOCHS", "150"))
ARCHS = ("r50", "cnx", "vit")


class ViTS4(base.ViTS):
    """ViT-S with 4x4 patches for 64x64 inputs (256 tokens + class token)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, 384, 4, 4)
        self.pos = nn.Parameter(torch.randn(1, 257, 384) * 0.02)


def build(arch: str) -> nn.Module:
    if arch == "vit":
        return ViTS4()
    if arch == "cnx":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Identity()  # keep LayerNorm2d + Flatten -> 768-d
        m.out_dim = 768
        return m
    m = resnet50(weights=None)
    m.maxpool = nn.Identity()
    m.fc = nn.Identity()
    m.out_dim = 2048
    return m


class StrongAug(torch.utils.data.Dataset):  # type: ignore[type-arg]
    def __init__(self, imgs: list, ys: torch.Tensor) -> None:
        self.imgs, self.ys = imgs, ys
        self.t = T.Compose([T.RandomResizedCrop(64, scale=(0.35, 1.0)), T.RandomHorizontalFlip(),
                            T.RandAugment(2, 9), T.ToTensor(), T.Normalize(base.MEAN, base.STD),
                            T.RandomErasing(0.25)])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.t(self.imgs[i]), self.ys[i]


def train(arch: str, seed: int, train_data: tuple, val: tuple) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build(arch).to(DEV)
    head = nn.Linear(net.out_dim, 100).to(DEV)
    ld = torch.utils.data.DataLoader(StrongAug(*train_data), 256, shuffle=True, num_workers=16, drop_last=True,
                                     persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    total, warm = EPOCHS * len(ld), 10 * len(ld)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    log = []
    for ep in range(EPOCHS):
        tot = 0.0
        for x, y in ld:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
                loss = F.cross_entropy(head(net(x)).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if (ep + 1) % 25 == 0 or ep == EPOCHS - 1:
            acc = base.knn_acc(net, val, 0)
            log.append({"epoch": ep + 1, "loss": round(tot / len(ld), 4), "knn_acc": round(acc, 4)})
            print(f"  {arch} s{seed} ep={ep + 1} loss={tot / len(ld):.3f} knn={acc:.3f}", flush=True)
    return net, log


def do_train(archs: list[str]) -> None:
    FEAT.mkdir(parents=True, exist_ok=True)
    val = base.load_val()
    probe = val[0][:base.NPROBE]
    data = base.load_train(0)
    for seed in range(3):
        for arch in archs:
            fp = FEAT / f"{arch}_s{seed}.pt"
            if fp.exists():
                continue
            t0 = time.time()
            net, log = train(arch, seed, data, val)
            f = base.embed(net, probe).cpu()
            meta = {"arch": arch, "seed": seed, "epochs": EPOCHS, "log": log,
                    "effective_rank": round(float(base.effective_rank(f)), 1),
                    "minutes": round((time.time() - t0) / 60, 1)}
            torch.save({"features": f, "meta": meta}, fp)
            print(f"{arch}_s{seed}: erank={meta['effective_rank']} knn={log[-1]['knn_acc']} ({meta['minutes']} min)",
                  flush=True)
            del net
            torch.cuda.empty_cache()


def analyze() -> None:
    f = {}
    metas = {}
    for a in ARCHS:
        for s in range(3):
            d = torch.load(FEAT / f"{a}_s{s}.pt", weights_only=False)
            f[(a, s)], metas[f"{a}_s{s}"] = d["features"].to(DEV), d["meta"]
    groups: dict[str, list[tuple]] = {f"{a} seeds": [((a, i), (a, j)) for i in range(3) for j in range(i + 1, 3)]
                                      for a in ARCHS}
    for a, b in itertools.combinations(ARCHS, 2):
        groups[f"{a} vs {b}"] = [((a, i), (b, j)) for i in range(3) for j in range(3)]
    res = {}
    rng = np.random.default_rng(0)
    for g, pairs in groups.items():
        rows = []
        for x, y in pairs:
            ab, ba = base.tls(f[x], f[y], mlp_steps=3000), base.tls(f[y], f[x], mlp_steps=3000)
            rows.append({"a": f"{x[0]}_s{x[1]}", "b": f"{y[0]}_s{y[1]}", "tau": round(min(ab.tau, ba.tau), 4),
                         "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                         "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)],
                         "underfit": bool(ab.perf_best < ab.perf_linear or ba.perf_best < ba.perf_linear)})
        taus = [r["tau"] for r in rows]
        boot = [rng.choice(taus, len(taus)).mean() for _ in range(2000)]
        res[g] = {"n": len(rows), "tau_mean": round(float(np.mean(taus)), 3),
                  "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                  "mean_lin_r2": round(float(np.mean([v for r in rows for v in r["lin"]])), 3),
                  "mean_mlp_r2": round(float(np.mean([v for r in rows for v in r["mlp"]])), 3),
                  "n_underfit": sum(r["underfit"] for r in rows), "pairs": rows}
        print(f"{g:14s} tau={res[g]['tau_mean']:.3f} CI {res[g]['ci95']} R2 lin/mlp "
              f"{res[g]['mean_lin_r2']:.2f}/{res[g]['mean_mlp_r2']:.2f} underfit {res[g]['n_underfit']}", flush=True)
    with open(OUT, "w") as fh:
        json.dump({"epochs": EPOCHS, "groups": res, "encoders": metas}, fh, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        do_train(sys.argv[2].split(","))
    else:
        analyze()
