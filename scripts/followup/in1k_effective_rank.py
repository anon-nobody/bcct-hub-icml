"""Effective rank of the public ImageNet checkpoint features used in the objective contrast.

Reads the probe features saved by in1k_arch_test.py / in1k_factor_test.py
(data/experiments/in1k_arch_feats/<timm name>.pt) and reports the effective rank of each
(latent.metrics.bitrate.effective_rank), plus the range over the supervised ViT-B/16 checkpoints
and the MAE / DINO ViT-B/16 values quoted in the paper.
Output: data/experiments/in1k_effective_rank.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.bitrate import effective_rank  # noqa: E402

FEAT = ROOT / "data/experiments/in1k_arch_feats"
OUT = ROOT / "data/experiments/in1k_effective_rank.json"
SUP_VITB16 = [
    "deit3_base_patch16_224.fb_in1k",
    "deit_base_patch16_224.fb_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.sam_in1k",
]


def load(p: Path) -> torch.Tensor:
    d = torch.load(p, weights_only=False)
    return (d["features"] if isinstance(d, dict) else d).float()


def main() -> None:
    erank = {p.stem: round(float(effective_rank(load(p))), 1) for p in sorted(FEAT.glob("*.pt"))}
    sup = [erank[m] for m in SUP_VITB16]
    res = {
        "effective_rank": erank,
        "supervised_vitb16_range": [min(sup), max(sup)],
        "mae_vitb16": erank["vit_base_patch16_224.mae"],
        "dino_vitb16": erank["vit_base_patch16_224.dino"],
    }
    for k, v in erank.items():
        print(f"{k:45s} {v:7.1f}")
    print(f"supervised ViT-B/16 range {res['supervised_vitb16_range']}, MAE {res['mae_vitb16']}")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
