"""Seed and shared-data contrasts at 128x128 on Food-101 (larger images and classes than CIFAR /
Tiny-ImageNet; ImageNet is not available on this machine).

ResNet-50 (torchvision, standard stem), supervised, 60 epochs, three seeds per group:
  ref  : classes [0, 50)
  half : classes [25, 75)   (50% of the reference classes shared)
  data : classes [51, 101)  (disjoint)
SGD (Nesterov, lr 0.1, wd 5e-4, batch 256), 5-epoch warm-up + cosine, label smoothing 0.1,
RandomResizedCrop(128, 0.35-1) + flip. Images are decoded once (Resize 146 + CenterCrop 146) into
one uint8 array. Food-101 lists samples class by class, so the probe is a class-stratified,
fixed-seed sample of 20 test images per class (2,020 images, indices saved), center-cropped to 128.
Quality gate: 20-NN accuracy on the test images of each encoder's own classes, with gallery and
queries alternating within each class (even / odd positions).
Analysis: worst-direction tau and per-direction Procrustes / MLP R^2 for
  100%: reference seed pairs (3); 50%: ref x half (9); 0%: ref x data (9);
  controls: half seeds (3), data seeds (3).
Usage: CUDA_VISIBLE_DEVICES=k python run_food128_factorial.py [train|analyze]
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision import transforms as T
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.bitrate import effective_rank  # noqa: E402
from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FEAT = ROOT / "data/experiments/food128_feats"
OUT = ROOT / "data/experiments/food128_results.json"
EPOCHS, PER_CLASS = 60, 20
MEAN, STD = (0.545, 0.444, 0.344), (0.271, 0.274, 0.278)
GROUPS = {"ref": 0, "half": 25, "data": 51}
SHRINK = T.Compose([T.Resize(146), T.CenterCrop(146)])


def _decode(path: str) -> np.ndarray:
    return np.asarray(SHRINK(Image.open(path).convert("RGB")), dtype=np.uint8)


def load(split: str) -> tuple[np.ndarray, torch.Tensor]:
    """Decode a split once into an (N, 146, 146, 3) uint8 array; labels from the dataset index."""
    cache = ROOT / f"data/experiments/food128_{split}_u8.npy"
    ds = datasets.Food101(str(ROOT / "data"), split=split, download=False)
    ys = torch.tensor(ds._labels)
    if cache.exists():
        return np.load(cache, mmap_mode="r"), ys
    with Pool(32) as pool:
        arr = np.stack(pool.map(_decode, [str(p) for p in ds._image_files], chunksize=256))
    np.save(cache, arr)
    return arr, ys


def probe_indices(ys: torch.Tensor) -> list[int]:
    """Class-stratified probe: PER_CLASS test images per class, fixed seed."""
    rng = np.random.default_rng(0)
    out: list[int] = []
    for c in range(101):
        idx = np.flatnonzero(ys.numpy() == c)
        out += sorted(rng.choice(idx, PER_CLASS, replace=False).tolist())
    return out


class Train(torch.utils.data.Dataset):  # type: ignore[type-arg]
    def __init__(self, imgs: list, ys: torch.Tensor, lo: int) -> None:
        self.idx = [i for i in range(len(ys)) if lo <= ys[i] < lo + 50]
        self.imgs, self.ys, self.lo = imgs, ys, lo
        self.t = T.Compose([T.RandomResizedCrop(128, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(),
                            T.Normalize(MEAN, STD)])

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.idx[k]
        return self.t(Image.fromarray(np.asarray(self.imgs[i]))), self.ys[i] - self.lo


EVAL = T.Compose([T.CenterCrop(128), T.ToTensor(), T.Normalize(MEAN, STD)])


@torch.no_grad()
def embed(net: nn.Module, imgs: np.ndarray, idx: list[int]) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(idx), 500):
        x = torch.stack([EVAL(Image.fromarray(np.asarray(imgs[j]))) for j in idx[i:i + 500]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
            out.append(net(x).float())
    net.train()
    return torch.cat(out)


def knn_acc(net: nn.Module, test: tuple, lo: int) -> float:
    """Gallery = even positions within each class, queries = odd positions (classes interleaved)."""
    imgs, ys = test
    gal, qry = [], []
    for c in range(lo, lo + 50):
        idx = np.flatnonzero(ys.numpy() == c)
        gal += idx[0::2].tolist()
        qry += idx[1::2].tolist()
    fg = F.normalize(embed(net, imgs, gal), dim=1)
    fq = F.normalize(embed(net, imgs, qry), dim=1)
    yg, yq = ys[gal].to(DEV), ys[qry].to(DEV)
    pred = torch.mode(yg[(fq @ fg.T).topk(20, dim=1).indices], dim=1).values
    return float((pred == yq).float().mean())


def train(lo: int, seed: int, tr: tuple, te: tuple) -> tuple[nn.Module, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = resnet50(weights=None)
    net.fc = nn.Identity()
    net = net.to(DEV)
    head = nn.Linear(2048, 50).to(DEV)
    ld = torch.utils.data.DataLoader(Train(*tr, lo), 256, shuffle=True, num_workers=16, drop_last=True,
                                     persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    params = list(net.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    total, warm = EPOCHS * len(ld), 5 * len(ld)
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
        if (ep + 1) % 15 == 0 or ep == EPOCHS - 1:
            acc = knn_acc(net, te, lo)
            log.append({"epoch": ep + 1, "loss": round(tot / len(ld), 4), "knn_acc": round(acc, 4)})
            print(f"  lo={lo} s{seed} ep={ep + 1} loss={tot / len(ld):.3f} knn={acc:.3f}", flush=True)
    return net, log


def do_train() -> None:
    FEAT.mkdir(parents=True, exist_ok=True)
    tr, te = load("train"), load("test")
    probe = probe_indices(te[1])
    for seed in range(3):
        for g, lo in GROUPS.items():
            fp = FEAT / f"{g}_s{seed}.pt"
            if fp.exists():
                continue
            t0 = time.time()
            net, log = train(lo, seed, tr, te)
            f = embed(net, te[0], probe).cpu()
            meta = {"group": g, "lo": lo, "seed": seed, "epochs": EPOCHS, "log": log,
                    "effective_rank": round(float(effective_rank(f)), 1), "minutes": round((time.time() - t0) / 60, 1)}
            torch.save({"features": f, "meta": meta}, fp)
            print(f"{g}_s{seed}: erank={meta['effective_rank']} knn={log[-1]['knn_acc']} ({meta['minutes']} min)",
                  flush=True)
            del net
            torch.cuda.empty_cache()


def analyze() -> None:
    f, metas = {}, {}
    for g in GROUPS:
        for s in range(3):
            d = torch.load(FEAT / f"{g}_s{s}.pt", weights_only=False)
            f[(g, s)], metas[f"{g}_s{s}"] = d["features"].to(DEV), d["meta"]
    sets = {"100 (ref seeds)": [(("ref", i), ("ref", j)) for i in range(3) for j in range(i + 1, 3)],
            "50 (ref vs half)": [(("ref", i), ("half", j)) for i in range(3) for j in range(3)],
            "0 (ref vs data)": [(("ref", i), ("data", j)) for i in range(3) for j in range(3)],
            "half seeds": [(("half", i), ("half", j)) for i in range(3) for j in range(i + 1, 3)],
            "data seeds": [(("data", i), ("data", j)) for i in range(3) for j in range(i + 1, 3)]}
    rng = np.random.default_rng(0)
    res = {}
    for k, pairs in sets.items():
        rows = []
        for x, y in pairs:
            ab, ba = tls(f[x], f[y], mlp_steps=3000), tls(f[y], f[x], mlp_steps=3000)
            rows.append({"a": f"{x[0]}_s{x[1]}", "b": f"{y[0]}_s{y[1]}", "tau": round(min(ab.tau, ba.tau), 4),
                         "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                         "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)],
                         "underfit": bool(ab.perf_best < ab.perf_linear or ba.perf_best < ba.perf_linear)})
        t = [r["tau"] for r in rows]
        boot = [rng.choice(t, len(t)).mean() for _ in range(2000)]
        res[k] = {"n": len(t), "tau_mean": round(float(np.mean(t)), 3),
                  "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                  "mean_lin_r2": round(float(np.mean([v for r in rows for v in r["lin"]])), 3),
                  "mean_mlp_r2": round(float(np.mean([v for r in rows for v in r["mlp"]])), 3),
                  "n_underfit": sum(r["underfit"] for r in rows), "pairs": rows}
        print(f"{k:18s} tau={res[k]['tau_mean']:.3f} CI {res[k]['ci95']} R2 lin/mlp "
              f"{res[k]['mean_lin_r2']:.2f}/{res[k]['mean_mlp_r2']:.2f} underfit {res[k]['n_underfit']}", flush=True)
    te_ys = torch.tensor(datasets.Food101(str(ROOT / "data"), split="test", download=False)._labels)
    with open(OUT, "w") as fh:
        json.dump({"epochs": EPOCHS, "contrasts": res, "encoders": metas,
                   "probe_test_indices": probe_indices(te_ys)}, fh, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    do_train() if (sys.argv[1] if len(sys.argv) > 1 else "train") == "train" else analyze()
