"""Objective and data effects at ImageNet scale with public checkpoints (no training).

Objective contrast: ViT-B/16 trained on ImageNet-1k only, supervised (AugReg, DeiT, DeiT-3, SAM)
vs self-supervised (MAE, DINO); also ViT-S/16 AugReg vs DINO. Architecture and data fixed.
Data contrast: same architecture, same supervised objective and recipe family, ImageNet-1k vs
ImageNet-21k (ViT-B/16 AugReg, ViT-S/16 AugReg, ConvNeXt-B FB). ImageNet-21k contains the
ImageNet-1k classes, so the data sets overlap partially rather than being disjoint.
Recipe control: ViT-B/16 AugReg-in21k vs original-in21k (same data and objective, recipe differs).
Reference: the within-architecture supervised-recipe pairs from in1k_arch_test.json.

Features and tau exactly as in in1k_arch_test.py (shared feature cache).
Output: data/experiments/in1k_factor_test.json
Usage: CUDA_VISIBLE_DEVICES=2 python in1k_factor_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import in1k_arch_test as base  # noqa: E402

OUT = base.ROOT / "data/experiments/in1k_factor_test.json"
SUP_B = ["vit_base_patch16_224.augreg_in1k", "deit_base_patch16_224.fb_in1k",
         "deit3_base_patch16_224.fb_in1k", "vit_base_patch16_224.sam_in1k"]
SSL_B = ["vit_base_patch16_224.mae", "vit_base_patch16_224.dino"]
PAIRS: dict[str, list[tuple[str, str]]] = {
    "objective (sup vs SSL, ViT-B, IN-1k)": [(s, t) for s in SUP_B for t in SSL_B],
    "objective (MAE vs DINO, ViT-B, IN-1k)": [("vit_base_patch16_224.mae", "vit_base_patch16_224.dino")],
    "objective (sup vs DINO, ViT-S, IN-1k)": [("vit_small_patch16_224.augreg_in1k", "vit_small_patch16_224.dino")],
    "data (IN-1k vs IN-21k, sup, same recipe)": [
        ("vit_base_patch16_224.augreg_in1k", "vit_base_patch16_224.augreg_in21k"),
        ("vit_small_patch16_224.augreg_in1k", "vit_small_patch16_224.augreg_in21k"),
        ("convnext_base.fb_in1k", "convnext_base.fb_in22k"),
    ],
    "recipe (AugReg vs orig, ViT-B, IN-21k)": [("vit_base_patch16_224.augreg_in21k", "vit_base_patch16_224.orig_in21k")],
}


def main() -> None:
    base.FEAT.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, pairs in PAIRS.items():
        for a, b in pairs:
            fa, fb = base.extract(a).to(base.DEV), base.extract(b).to(base.DEV)
            ab = base.tls(fa, fb, mlp_steps=3000)
            ba = base.tls(fb, fa, mlp_steps=3000)
            t = min(ab.tau, ba.tau)
            cka = float(base.centered_kernel_alignment(fa, fb))
            uf = bool(ab.perf_best < ab.perf_linear or ba.perf_best < ba.perf_linear)
            rows.append({"group": group, "a": a, "b": b, "tau": round(float(t), 4), "cka": round(cka, 4),
                         "r2_lin_ab": round(float(ab.perf_linear), 4), "r2_mlp_ab": round(float(ab.perf_best), 4),
                         "r2_lin_ba": round(float(ba.perf_linear), 4), "r2_mlp_ba": round(float(ba.perf_best), 4),
                         "mlp_underfit": uf})
            print(f"[{group}] {a} vs {b}: tau={t:.3f} cka={cka:.3f} underfit={uf}", flush=True)
    with open(base.ROOT / "data/experiments/in1k_arch_test.json") as f:
        arch = json.load(f)
    ref = {k: [r["tau"] for r in arch["pairs"] if r["same_arch"] and r["arch_a"] == k] for k in ("resnet50", "vit_b16")}
    summary = {g: {"n": len(v), "tau_mean": round(float(np.mean(v)), 4), "tau_values": v}
               for g in PAIRS if (v := [r["tau"] for r in rows if r["group"] == g])}
    summary["reference: same-arch supervised recipes, IN-1k"] = {
        k: {"n": len(v), "tau_mean": round(float(np.mean(v)), 4)} for k, v in ref.items()}
    print(json.dumps(summary, indent=1))
    with open(OUT, "w") as f:
        json.dump({"pairs": rows, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    main()
