"""Mid-scale controlled factorial on Tiny-ImageNet at native 64x64 resolution.

Scales the CIFAR (32x32, small ViT) factorial up to ResNet-50 / ViT-S at 64x64 on 100-class
halves of Tiny-ImageNet. Every group has three seeds.
  ref : ResNet-50, supervised, classes [0,100)
  data: ResNet-50, supervised, classes [100,200)          (data swap, disjoint classes)
  arch: ViT-S/8, supervised, classes [0,100)               (architecture swap)
  obj : ResNet-50, SimCLR, classes [0,100)                 (objective swap)
Contrasts (worst-direction tau, atlas settings): C-init ref_i vs ref_j (3 pairs); C-data/C-arch/C-obj
ref_i vs variant_j (9 pairs each); same-variant controls variant_i vs variant_j (3 pairs each).
Representation = penultimate features (ResNet-50: 2048-d pooled; ViT-S: 384-d CLS) of the first
2,000 Tiny-ImageNet validation images (all 200 classes).
Quality gate per encoder: k-NN (k=20) accuracy on the validation images of its own classes, and
effective rank of the probe features.

Usage: CUDA_VISIBLE_DEVICES=2 python run_tin64_factorial.py [smoke|full] [sup_epochs] [ssl_epochs]
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
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.bitrate import effective_rank  # noqa: E402
from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
TIN = ROOT / "data/tiny-imagenet-200"
FEAT = ROOT / "data/experiments/tin64_factorial_feats"
OUT = ROOT / "data/experiments/tin64_factorial_results.json"
NPROBE = 2000
MEAN, STD = (0.480, 0.448, 0.398), (0.277, 0.269, 0.282)
GROUPS = {"ref": ("resnet50", "sup", 0), "data": ("resnet50", "sup", 100),
          "arch": ("vit_s", "sup", 0), "obj": ("resnet50", "simclr", 0)}

WNIDS = sorted(p.name for p in (TIN / "train").iterdir())


def load_train(lo: int) -> tuple[list[Image.Image], torch.Tensor]:
    imgs, ys = [], []
    for c in range(lo, lo + 100):
        for f in sorted((TIN / "train" / WNIDS[c] / "images").glob("*.JPEG")):
            imgs.append(Image.open(f).convert("RGB"))
            ys.append(c - lo)
    return imgs, torch.tensor(ys)


def load_val() -> tuple[list[Image.Image], torch.Tensor]:
    rows = [ln.split("\t")[:2] for ln in (TIN / "val/val_annotations.txt").read_text().splitlines()]
    rows.sort(key=lambda r: int(r[0].split("_")[1].split(".")[0]))
    idx = {w: i for i, w in enumerate(WNIDS)}
    imgs = [Image.open(TIN / "val/images" / f).convert("RGB") for f, _ in rows]
    return imgs, torch.tensor([idx[w] for _, w in rows])


class ViTS(nn.Module):
    """ViT-S with 8x8 patches for 64x64 inputs (64 tokens), CLS output."""

    def __init__(self, dim: int = 384, depth: int = 12, heads: int = 6) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 8, 8)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, 65, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,64,64) -> (B,384)
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.norm(self.blocks(t))[:, 0]


def backbone(arch: str) -> nn.Module:
    if arch == "vit_s":
        return ViTS()
    m = resnet50(weights=None)
    m.maxpool = nn.Identity()  # keep resolution for 64x64 inputs
    m.fc = nn.Identity()
    m.out_dim = 2048
    return m


class Aug(torch.utils.data.Dataset):  # type: ignore[type-arg]
    def __init__(self, imgs: list[Image.Image], ys: torch.Tensor, ssl: bool) -> None:
        self.imgs, self.ys, self.ssl = imgs, ys, ssl
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        if ssl:
            self.t = T.Compose([T.RandomResizedCrop(64, scale=(0.08, 1.0)), T.RandomHorizontalFlip(),
                                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                T.RandomGrayscale(0.2), *norm])
        else:
            self.t = T.Compose([T.RandomResizedCrop(64, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), *norm])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):  # noqa: ANN204
        if self.ssl:
            return self.t(self.imgs[i]), self.t(self.imgs[i])
        return self.t(self.imgs[i]), self.ys[i]


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temp
    n = len(z1)
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


PLAIN = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


@torch.no_grad()
def embed(net: nn.Module, imgs: list[Image.Image]) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(imgs), 500):
        x = torch.stack([PLAIN(im) for im in imgs[i:i + 500]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
            out.append(net(x).float())
    net.train()
    return torch.cat(out)


def knn_acc(net: nn.Module, val: tuple[list[Image.Image], torch.Tensor], lo: int) -> float:
    imgs, ys = val
    sel = [i for i in range(len(ys)) if lo <= ys[i] < lo + 100]
    f = F.normalize(embed(net, [imgs[i] for i in sel]), dim=1)
    y = ys[sel].to(DEV)
    h = len(y) // 2
    pred = torch.mode(y[:h][(f[h:] @ f[:h].T).topk(20, dim=1).indices], dim=1).values
    return float((pred == y[h:]).float().mean())


def train(group: str, seed: int, epochs: int, val: tuple, cache: dict) -> tuple[nn.Module, list[dict]]:
    arch, obj, lo = GROUPS[group]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if lo not in cache:
        cache[lo] = load_train(lo)
    imgs, ys = cache[lo]
    ssl = obj == "simclr"
    net = backbone(arch).to(DEV)
    head = (nn.Sequential(nn.Linear(net.out_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                          nn.Linear(2048, 128)) if ssl else nn.Linear(net.out_dim, 100)).to(DEV)
    bs = 512 if ssl else 256
    ld = torch.utils.data.DataLoader(Aug(imgs, ys, ssl), bs, shuffle=True, num_workers=16, drop_last=True,
                                     persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    params = list(net.parameters()) + list(head.parameters())
    if arch == "vit_s":
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    else:
        opt = torch.optim.SGD(params, lr=(0.06 if ssl else 0.1) * bs / 256, momentum=0.9, weight_decay=5e-4,
                              nesterov=not ssl)
    total, warm = epochs * len(ld), 5 * len(ld)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    log = []
    for ep in range(epochs):
        tot = 0.0
        for a, b in ld:
            a, b = a.to(DEV, non_blocking=True), b.to(DEV, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.startswith("cuda")):
                loss = (nt_xent(head(net(a)).float(), head(net(b)).float()) if ssl
                        else F.cross_entropy(head(net(a)).float(), b, label_smoothing=0.1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            acc = knn_acc(net, val, lo)
            log.append({"epoch": ep + 1, "loss": round(tot / len(ld), 4), "knn_acc": round(acc, 4)})
            print(f"  {group} s{seed} ep={ep + 1} loss={tot / len(ld):.3f} knn={acc:.3f}", flush=True)
    return net, log


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    sup_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ssl_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    feat_dir = FEAT if mode == "full" else FEAT / "smoke"
    feat_dir.mkdir(parents=True, exist_ok=True)
    val = load_val()
    probe = val[0][:NPROBE]
    cache: dict = {}
    order = [("ref", 0), ("obj", 0), ("arch", 0), ("data", 0)] + [(g, s) for s in (1, 2) for g in GROUPS]
    if mode == "smoke":
        order, sup_ep, ssl_ep = [("ref", 0), ("obj", 0), ("arch", 0)], 1, 1
    feats, metas = {}, {}
    for g, s in order:
        fp = feat_dir / f"{g}_s{s}.pt"
        if fp.exists():
            d = torch.load(fp, weights_only=False)
        else:
            t0 = time.time()
            net, log = train(g, s, ssl_ep if GROUPS[g][1] == "simclr" else sup_ep, val, cache)
            f = embed(net, probe).cpu()
            d = {"features": f, "meta": {"group": g, "seed": s, "log": log,
                                         "effective_rank": round(float(effective_rank(f)), 1),
                                         "minutes": round((time.time() - t0) / 60, 1)}}
            torch.save(d, fp)
            print(f"{g}_s{s}: erank={d['meta']['effective_rank']} knn={log[-1]['knn_acc']} "
                  f"({d['meta']['minutes']} min)", flush=True)
            del net
            torch.cuda.empty_cache()
        feats[(g, s)], metas[f"{g}_s{s}"] = d["features"], d["meta"]
    if mode == "smoke":
        return

    def tau(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.to(DEV), b.to(DEV)
        return min(tls(a, b, mlp_steps=3000).tau, tls(b, a, mlp_steps=3000).tau)

    def ci(v: list[float]) -> list[float]:
        r = np.random.default_rng(0)
        m = [r.choice(v, len(v)).mean() for _ in range(2000)]
        return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]

    res = {}
    rows = {"C-init (ref seeds)": [tau(feats[("ref", i)], feats[("ref", j)]) for i in range(3) for j in range(i + 1, 3)]}
    for g in ("data", "arch", "obj"):
        rows[f"C-{g} (ref vs {g})"] = [tau(feats[("ref", i)], feats[(g, j)]) for i in range(3) for j in range(3)]
        rows[f"{g}-init ({g} seeds)"] = [tau(feats[(g, i)], feats[(g, j)]) for i in range(3) for j in range(i + 1, 3)]
    for k, v in rows.items():
        res[k] = {"n": len(v), "mean": round(float(np.mean(v)), 3), "ci95": ci(v), "vals": [round(x, 3) for x in v]}
        print(f"  {k:24s} mean={res[k]['mean']:.3f} CI {res[k]['ci95']} {res[k]['vals']}", flush=True)
    with open(OUT, "w") as f:
        json.dump({"sup_epochs": sup_ep, "ssl_epochs": ssl_ep, "contrasts": res, "encoders": metas}, f, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
