"""S1: Retrieval downstream sanity check.

For each of N pairs:
  1. Train Procrustes A->B on train features
  2. Apply to test features (disjoint)
  3. k-NN label transfer: query (stitched test) vs gallery (native B test)
  4. R@1, R@5, R@10 image-image retrieval

Correlates atlas metrics (tau, CKA, S_local, Delta) with downstream performance.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/retrieval_downstream.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ATLAS_PATH = Path("data/atlas.json")
OUT_PATH = Path("data/experiments/retrieval_downstream.json")

# Subset of 20 atlas pairs spanning tau range
PAIRS = [
    ("clip_vitb16", "clip_vitl14"),
    ("clip_vitb32", "clip_vitb16"),
    ("resnet50", "resnet101"),
    ("convnext_base", "convnext_large"),
    ("convnext_base", "swin_base_patch4_window7_224"),
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("dinov2_vitg14", "dinov2_vitl14"),
    ("dinov2_vits14", "dinov2_vitb14"),
    ("vit_base_patch16_224", "vit_large_patch16_224"),
    ("efficientnet_b0", "resnet101"),
    ("mixer_b16_224", "resnet101"),
    ("convnext_large", "vit_base_patch16_224"),
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    ("dinov2_vitb14", "resnet50"),
    ("beit_base_patch16_224", "resnet50"),
    ("clip_vitl14", "dinov2_vitl14"),
    ("beit_base_patch16_224", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "dinov2_vitb14"),
    ("clip_vitb16", "resnet50"),
]


def load_pair(model: str, device: str, split: str):
    if split == "train":
        pt = Path("data/features_train") / f"{model}.pt"
        lab = Path("data") / "cifar100_train_labels.pt"
    else:
        pt = Path("data/features") / f"{model}.pt"
        lab = Path("data") / "cifar100_test_labels.pt"
    feats = torch.load(pt, weights_only=True)["features"].to(device)
    labels = torch.load(lab, weights_only=True).to(device)
    n = min(feats.shape[0], labels.shape[0])
    return feats[:n], labels[:n]


def fit_procrustes(src_train: torch.Tensor, tgt_train: torch.Tensor) -> dict:
    """Fit Procrustes from src_train → tgt_train. Returns state dict for apply."""
    src_mean = src_train.mean(0, keepdim=True)
    tgt_mean = tgt_train.mean(0, keepdim=True)
    src_c = src_train - src_mean
    tgt_c = tgt_train - tgt_mean
    d_s, d_t = src_c.shape[1], tgt_c.shape[1]
    max_d = max(d_s, d_t)
    src_pad = F.pad(src_c, (0, max_d - d_s)) if d_s < max_d else src_c
    tgt_pad = F.pad(tgt_c, (0, max_d - d_t)) if d_t < max_d else tgt_c
    U, _, Vt = torch.linalg.svd(tgt_pad.T @ src_pad, full_matrices=False)
    return {"W": U @ Vt, "src_mean": src_mean, "tgt_mean": tgt_mean,
            "d_s": d_s, "d_t": d_t, "max_d": max_d}


def apply_procrustes(src: torch.Tensor, state: dict) -> torch.Tensor:
    """Apply a fitted Procrustes map to new source data."""
    d_s, d_t, max_d = state["d_s"], state["d_t"], state["max_d"]
    centered = src - state["src_mean"]
    padded = F.pad(centered, (0, max_d - d_s)) if d_s < max_d else centered
    return (padded @ state["W"].T)[:, :d_t] + state["tgt_mean"]


def knn_label_transfer(query: torch.Tensor, query_lab: torch.Tensor,
                        gallery: torch.Tensor, gallery_lab: torch.Tensor,
                        k: int = 10) -> float:
    """k-NN classifier: for each query, find k nearest in gallery, majority vote."""
    q = F.normalize(query, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.T  # (n_q, n_g)
    topk = sim.topk(k, dim=1).indices  # (n_q, k)
    top_labels = gallery_lab[topk]  # (n_q, k)
    preds = torch.mode(top_labels, dim=1).values
    return float((preds == query_lab).float().mean())


def retrieval_recall(query: torch.Tensor, query_lab: torch.Tensor,
                     gallery: torch.Tensor, gallery_lab: torch.Tensor,
                     k_vals=(1, 5, 10)) -> dict:
    """Image-image retrieval: same-class is "correct"."""
    q = F.normalize(query, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.T
    results = {}
    for k in k_vals:
        topk = sim.topk(k, dim=1).indices  # (n_q, k)
        # Check if any of top-k is same class
        same_class = (gallery_lab[topk] == query_lab[:, None]).any(dim=1)
        results[f"R@{k}"] = float(same_class.float().mean())
    return results


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    atlas = json.loads(ATLAS_PATH.read_text())
    pairwise = atlas["pairwise"]

    results = []
    for model_a, model_b in PAIRS:
        key = f"{model_a} <-> {model_b}"
        alt = f"{model_b} <-> {model_a}"
        entry = pairwise.get(key) or pairwise.get(alt)
        if not entry:
            continue

        try:
            fa_tr, la_tr = load_pair(model_a, device, "train")
            fb_tr, lb_tr = load_pair(model_b, device, "train")
            fa_te, la_te = load_pair(model_a, device, "test")
            fb_te, lb_te = load_pair(model_b, device, "test")
        except FileNotFoundError as e:
            logger.warning("skip %s: %s", key, e)
            continue

        n_tr = min(fa_tr.shape[0], fb_tr.shape[0])
        n_te = min(fa_te.shape[0], fb_te.shape[0])
        fa_tr, fb_tr, la_tr = fa_tr[:n_tr], fb_tr[:n_tr], la_tr[:n_tr]
        fa_te, fb_te, lb_te = fa_te[:n_te], fb_te[:n_te], lb_te[:n_te]

        # Fit Procrustes A->B on TRAIN only, apply to TEST (no data leakage)
        with torch.no_grad():
            pstate = fit_procrustes(fa_tr, fb_tr)
            mapped_te = apply_procrustes(fa_te, pstate)

        # Baseline: native B k-NN (gallery=train, query=test on B)
        knn_native = knn_label_transfer(fb_te, lb_te, fb_tr, la_tr, k=10)
        # Cross-encoder k-NN: query=mapped A test, gallery=native B train
        knn_stitched = knn_label_transfer(mapped_te, lb_te, fb_tr, la_tr, k=10)

        # Retrieval: split test into halves (query/gallery)
        half = n_te // 2
        # Native B
        ret_native = retrieval_recall(fb_te[:half], lb_te[:half],
                                       fb_te[half:], lb_te[half:])
        # Stitched A->B: query=mapped A, gallery=native B
        ret_stitched = retrieval_recall(mapped_te[:half], lb_te[:half],
                                         fb_te[half:], lb_te[half:])

        retention_knn = knn_stitched / max(knn_native, 1e-10)
        retention_r10 = ret_stitched["R@10"] / max(ret_native["R@10"], 1e-10)

        logger.info(
            "%s: tau=%.3f, knn_nat=%.3f, knn_stitched=%.3f, ret_nat=%.3f, ret_stitched=%.3f",
            key, entry["tau"], knn_native, knn_stitched,
            ret_native["R@10"], ret_stitched["R@10"],
        )

        results.append({
            "pair": key,
            "tau": entry["tau"],
            "s_global": entry["s_global"],
            "s_local": entry["s_local"],
            "delta": entry["delta"],
            "knn_native": round(knn_native, 4),
            "knn_stitched": round(knn_stitched, 4),
            "knn_retention": round(retention_knn, 4),
            "ret_native": {k: round(v, 4) for k, v in ret_native.items()},
            "ret_stitched": {k: round(v, 4) for k, v in ret_stitched.items()},
            "ret_r10_retention": round(retention_r10, 4),
        })

    # Correlations with atlas metrics
    metrics = {}
    for name in ["tau", "s_global", "s_local", "delta"]:
        vals = np.array([r[name] for r in results])
        sign = -1 if name == "delta" else 1
        for target in ["knn_stitched", "knn_retention", "ret_r10_retention"]:
            tgt_vals = np.array([
                r[target] if isinstance(r[target], (int, float))
                else r[target] for r in results
            ])
            rho, p = spearmanr(sign * vals, tgt_vals)
            metrics[f"{name}_vs_{target}"] = {
                "rho": round(float(rho), 3),
                "p": float(f"{p:.2e}"),
                "n": len(results),
            }

    # BCCT combined (standardize + sum) vs downstream
    taus = np.array([r["tau"] for r in results])
    ckas = np.array([r["s_global"] for r in results])
    slocs = np.array([r["s_local"] for r in results])
    deltas = -np.array([r["delta"] for r in results])  # invert: lower delta = better

    def std(x):
        return (x - x.mean()) / (x.std() + 1e-10)

    combined = std(taus) + std(ckas) + std(slocs) + std(deltas)
    for target in ["knn_stitched", "knn_retention", "ret_r10_retention"]:
        tgt_vals = np.array([r[target] for r in results])
        rho, p = spearmanr(combined, tgt_vals)
        metrics[f"bcct_combined_vs_{target}"] = {
            "rho": round(float(rho), 3),
            "p": float(f"{p:.2e}"),
        }

    output = {
        "description": "Retrieval downstream sanity check (k-NN + R@k) across 20 atlas pairs",
        "n_pairs": len(results),
        "results": results,
        "correlations": metrics,
        "summary": {
            "mean_knn_native": round(float(np.mean([r["knn_native"] for r in results])), 4),
            "mean_knn_stitched": round(float(np.mean([r["knn_stitched"] for r in results])), 4),
            "mean_knn_retention": round(float(np.mean([r["knn_retention"] for r in results])), 4),
            "mean_ret_r10_native": round(float(np.mean([r["ret_native"]["R@10"] for r in results])), 4),
            "mean_ret_r10_stitched": round(float(np.mean([r["ret_stitched"]["R@10"] for r in results])), 4),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))

    logger.info("\n=== Summary ===")
    logger.info("Native k-NN: %.3f, Stitched k-NN: %.3f, Retention: %.3f",
                output["summary"]["mean_knn_native"],
                output["summary"]["mean_knn_stitched"],
                output["summary"]["mean_knn_retention"])
    logger.info("\n=== Correlations with downstream ===")
    for name, m in metrics.items():
        logger.info("  %s: rho=%+.3f, p=%s", name, m["rho"], m["p"])
    logger.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
