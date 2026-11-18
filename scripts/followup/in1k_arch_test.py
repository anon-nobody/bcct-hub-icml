"""Architecture effect at ImageNet scale with public checkpoints (no training).

All models are supervised and trained on ImageNet-1k only, so training data and objective are
held fixed. Within an architecture, checkpoints differ only in training recipe (augmentation,
schedule, optimizer); across architectures, the architecture differs as well. If architecture
matters, cross-architecture tau should be clearly below within-architecture tau.

Features: atlas pipeline (Resize 256, CenterCrop 224, ImageNet normalization, first 5,000
CIFAR-100 test images, timm num_classes=0 output, L2-normalized), cached per model.
tau: worst direction of transport_linearity_score (mlp_steps=3000), as in the atlas.
Output: data/experiments/in1k_arch_test.json
Usage: CUDA_VISIBLE_DEVICES=2 python in1k_arch_test.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.alignment import centered_kernel_alignment  # noqa: E402
from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FEAT = ROOT / "data/experiments/in1k_arch_feats"
OUT = ROOT / "data/experiments/in1k_arch_test.json"
N = 5000
MODELS = {
    "resnet50.a1_in1k": "resnet50", "resnet50.b1k_in1k": "resnet50", "resnet50.c1_in1k": "resnet50",
    "resnet50.d_in1k": "resnet50", "resnet50.gluon_in1k": "resnet50",
    "deit_base_patch16_224.fb_in1k": "vit_b16", "deit3_base_patch16_224.fb_in1k": "vit_b16",
    "vit_base_patch16_224.augreg_in1k": "vit_b16", "vit_base_patch16_224.sam_in1k": "vit_b16",
    "convnext_base.fb_in1k": "convnext", "convnext_small.fb_in1k": "convnext",
    "swin_base_patch4_window7_224.ms_in1k": "swin",
}


def loader() -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.CIFAR100(str(ROOT / "data/imagenet"), train=False, download=False, transform=tf)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(N)), batch_size=128,
                                       shuffle=False, num_workers=4)


@torch.no_grad()
def extract(name: str) -> torch.Tensor:
    fp = FEAT / f"{name}.pt"
    if fp.exists():
        return torch.load(fp, weights_only=True)
    import timm

    model = timm.create_model(name, pretrained=True, num_classes=0).to(DEV).eval()
    out = torch.cat([F.normalize(model(x.to(DEV)), dim=-1).cpu() for x, _ in loader()])
    torch.save(out, fp)
    del model
    torch.cuda.empty_cache()
    print(f"extracted {name} {tuple(out.shape)}", flush=True)
    return out


def main() -> None:
    FEAT.mkdir(parents=True, exist_ok=True)
    feats = {m: extract(m) for m in MODELS}
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        fa, fb = feats[a].to(DEV), feats[b].to(DEV)
        ab, ba = tls(fa, fb, mlp_steps=3000), tls(fb, fa, mlp_steps=3000)
        t = min(ab.tau, ba.tau)
        cka = float(centered_kernel_alignment(fa, fb))
        # MLP underfit: the nonlinear reference does worse than the linear map in some direction
        underfit = bool(ab.perf_best < ab.perf_linear or ba.perf_best < ba.perf_linear)
        rows.append({"a": a, "b": b, "arch_a": MODELS[a], "arch_b": MODELS[b],
                     "same_arch": MODELS[a] == MODELS[b], "tau": round(float(t), 4), "cka": round(cka, 4),
                     "r2_lin_ab": round(float(ab.perf_linear), 4), "r2_mlp_ab": round(float(ab.perf_best), 4),
                     "r2_lin_ba": round(float(ba.perf_linear), 4), "r2_mlp_ba": round(float(ba.perf_best), 4),
                     "mlp_underfit": underfit})
        print(f"{a:38s} {b:38s} same_arch={MODELS[a] == MODELS[b]!s:5s} tau={t:.3f} cka={cka:.3f}", flush=True)
    summaries = {}
    for tag, sel in [("all", rows), ("mlp_fit_ok", [r for r in rows if not r["mlp_underfit"]])]:
        summaries[tag] = summarize(sel)
    summary = summaries["all"] | {"fit_ok": summaries["mlp_fit_ok"],
                                  "n_underfit": sum(r["mlp_underfit"] for r in rows)}
    print(json.dumps(summary, indent=1))
    with open(OUT, "w") as f:
        json.dump({"models": MODELS, "pairs": rows, "summary": summary,
                   "note": "supervised, ImageNet-1k-only checkpoints; recipe varies within architecture"}, f, indent=1)


def summarize(rows: list[dict]) -> dict:
    same = [r["tau"] for r in rows if r["same_arch"]]
    cross = [r["tau"] for r in rows if not r["same_arch"]]
    by_arch = {k: float(np.mean(v)) for k in ("resnet50", "vit_b16", "convnext")
               if (v := [r["tau"] for r in rows if r["same_arch"] and r["arch_a"] == k])}
    return {
        "n_same": len(same), "n_cross": len(cross),
        "tau_same_mean": round(float(np.mean(same)), 4), "tau_cross_mean": round(float(np.mean(cross)), 4),
        "tau_same_median": round(float(np.median(same)), 4), "tau_cross_median": round(float(np.median(cross)), 4),
        "mannwhitney_p_same_gt_cross": float(mannwhitneyu(same, cross, alternative="greater").pvalue),
        "within_arch_mean_by_arch": {k: round(v, 4) for k, v in by_arch.items()},
        "cka_same_mean": round(float(np.mean([r["cka"] for r in rows if r["same_arch"]])), 4),
        "cka_cross_mean": round(float(np.mean([r["cka"] for r in rows if not r["same_arch"]])), 4),
    }


if __name__ == "__main__":
    main()
