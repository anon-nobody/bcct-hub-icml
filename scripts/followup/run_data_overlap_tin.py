"""Dataset-generalization of the overlap law: TinyImageNet (200 classes, 64x64).
Mirrors run_data_overlap.py (supervised, which trains stably) but on TinyImageNet
resized to 32x32, with 50-class windows [lo,hi) so the overlap structure matches the
CIFAR experiment. If tau again rises super-linearly with shared-class overlap, the
'transport requires shared data' law is not CIFAR-specific.
Usage: python run_data_overlap_tin.py [epochs=60]
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TIN = ROOT / "data/tiny-imagenet-200"
sys.path.insert(0, str(ROOT / "scripts" / "followup"))
sys.path.insert(0, str(ROOT / "src"))
from train_controlled import Encoder, build_objective, set_seed  # noqa
from latent.metrics.transport import transport_linearity_score as TLS  # noqa
from torchvision import datasets, transforms as T  # noqa

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FEAT = ROOT / "data/experiments/overlap_tin_feats"
FEAT.mkdir(parents=True, exist_ok=True)
EP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BATCH, NPROBE = 64, 2000

TRAIN_AUG = T.Compose(
    [
        T.Resize(32),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Grayscale(3) if False else T.Lambda(lambda im: im.convert("RGB")),
        T.ToTensor(),
    ]
)
PROBE_TF = T.Compose([T.Lambda(lambda im: im.convert("RGB")), T.Resize(32), T.ToTensor()])

_full = None


def train_loader(lo, hi):
    global _full
    if _full is None:
        _full = datasets.ImageFolder(
            str(TIN / "train"), transform=TRAIN_AUG
        )  # 200 classes (sorted wnid)
    idx = [i for i, (_, t) in enumerate(_full.samples) if lo <= t < hi]
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(_full, idx), BATCH, shuffle=True, num_workers=2, drop_last=True
    )


class ValProbe(torch.utils.data.Dataset):
    def __init__(self):
        self.files = sorted(glob.glob(str(TIN / "val/images/*.JPEG")))[:NPROBE]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return PROBE_TF(Image.open(self.files[i])), 0


def train_tin(lo, hi, seed):
    set_seed(seed)
    enc = Encoder("vit_b", 512).to(DEV)
    extra, step = build_objective("supervised", enc, 512, DEV)
    ld = train_loader(lo, hi)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP * len(ld))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.startswith("cuda"))
    for _ in range(EP):
        for x, y in ld:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
                loss = step((x.to(DEV), y.to(DEV)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return enc.eval()


@torch.no_grad()
def extract(enc):
    ld = torch.utils.data.DataLoader(ValProbe(), 128, num_workers=2)
    feats = []
    for x, _ in ld:
        with torch.amp.autocast("cuda", enabled=DEV.startswith("cuda")):
            feats.append(enc(x.to(DEV)).float().cpu())
    return torch.cat(feats)[:NPROBE]


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
            e = train_tin(lo, hi, sd)
            F[name] = extract(e)
            torch.save(F[name], fp)
            del e
            torch.cuda.empty_cache()
            print(f"trained {name} eff_rank={erank(F[name])} ({time.time() - t0:.0f}s)", flush=True)
    print("eff_ranks:", {k: erank(v) for k, v in F.items()}, flush=True)
    ref = F["ref_s0"]
    ov = {"ref_s1": 100, "ov74": 74, "ov50": 50, "ov26": 26, "ov0": 0}
    curve = []
    for name in ["ref_s1", "ov74", "ov50", "ov26", "ov0"]:
        t = tau(ref, F[name])
        curve.append((ov[name], t))
        print(f"  TinyImageNet shared {ov[name]:3d}% -> tau = {t:.3f}", flush=True)
    from scipy.stats import spearmanr

    rho = spearmanr([o for o, _ in curve], [t for _, t in curve]).correlation
    print(f"\nTinyImageNet overlap law: Spearman(overlap%, tau) = {rho:.3f}")
    with open(ROOT / "data/experiments/overlap_tin_curve.json", "w") as f:
        json.dump(
            {
                "dataset": "tiny-imagenet",
                "objective": "supervised",
                "epochs": EP,
                "curve": curve,
                "spearman": rho,
                "eff_ranks": {k: erank(v) for k, v in F.items()},
            },
            f,
            indent=2,
        )
    print("saved -> data/experiments/overlap_tin_curve.json")


if __name__ == "__main__":
    main()
