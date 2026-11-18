"""Scorecard case study on external (out-of-atlas) models.

For each external model x anchor pair:
  1. Compute BCCT scorecard (bitrate, S_local, S_global, tau, Delta, regime)
  2. Compute downstream retrieval (k-NN label transfer with Procrustes fit on train, eval on test)
  3. Compare atlas-metric prediction with actual downstream accuracy

Anchors: DINOv2-ViT-B (SSL), CLIP-ViT-L/14 (contrastive), ResNet-50 (supervised)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/external_scorecard_study.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from latent.evaluation.scorecard import compute_scorecard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ANCHORS = [
    ("dinov2_vitb14", "DINOv2-ViT-B", "self-supervised"),
    ("clip_vitl14", "CLIP-ViT-L/14", "contrastive"),
    ("resnet50", "ResNet-50", "supervised"),
]

EXTERNAL = [
    ("siglip_base_p16_224", "SigLIP-B/16", "contrastive"),
    ("convnextv2_base", "ConvNeXt-V2-B", "supervised"),
    ("eva02_base_p14_224", "EVA02-B/14", "self-supervised"),
    ("deit3_base_p16_224", "DeiT3-B/16", "supervised"),
    ("dinov2_reg_base", "DINOv2+reg-B", "self-supervised"),
]

OUT_PATH = Path("data/experiments/external_scorecard_study.json")


def load_feat(path: Path, device: str) -> torch.Tensor:
    obj = torch.load(path, weights_only=True)
    return obj["features"].to(device) if isinstance(obj, dict) else obj.to(device)


def fit_procrustes(src_train: torch.Tensor, tgt_train: torch.Tensor) -> dict:
    src_mean = src_train.mean(0, keepdim=True)
    tgt_mean = tgt_train.mean(0, keepdim=True)
    src_c = src_train - src_mean
    tgt_c = tgt_train - tgt_mean
    d_s, d_t = src_c.shape[1], tgt_c.shape[1]
    max_d = max(d_s, d_t)
    src_pad = F.pad(src_c, (0, max_d - d_s)) if d_s < max_d else src_c
    tgt_pad = F.pad(tgt_c, (0, max_d - d_t)) if d_t < max_d else tgt_c
    U, _, Vt = torch.linalg.svd(tgt_pad.T @ src_pad, full_matrices=False)
    return {
        "W": U @ Vt,
        "src_mean": src_mean,
        "tgt_mean": tgt_mean,
        "d_s": d_s,
        "d_t": d_t,
        "max_d": max_d,
    }


def apply_procrustes(src: torch.Tensor, state: dict) -> torch.Tensor:
    d_s, d_t, max_d = state["d_s"], state["d_t"], state["max_d"]
    centered = src - state["src_mean"]
    padded = F.pad(centered, (0, max_d - d_s)) if d_s < max_d else centered
    return (padded @ state["W"].T)[:, :d_t] + state["tgt_mean"]


def knn_label_transfer(
    query: torch.Tensor,
    query_lab: torch.Tensor,
    gallery: torch.Tensor,
    gallery_lab: torch.Tensor,
    k: int = 10,
) -> float:
    q = F.normalize(query, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sim = q @ g.T
    topk = sim.topk(k, dim=1).indices
    preds = torch.mode(gallery_lab[topk], dim=1).values
    return float((preds == query_lab).float().mean())


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ext_test_dir = Path("data/features_external/test")
    ext_train_dir = Path("data/features_external/train")
    lab_test = torch.load("data/cifar100_test_labels.pt", weights_only=True).to(device)
    lab_train = torch.load("data/cifar100_train_labels.pt", weights_only=True).to(device)

    results = []
    for ext_key, ext_name, ext_obj in EXTERNAL:
        ext_test_path = ext_test_dir / f"{ext_key}.pt"
        ext_train_path = ext_train_dir / f"{ext_key}.pt"
        if not ext_test_path.exists() or not ext_train_path.exists():
            logger.warning("Skip %s: features missing", ext_key)
            continue

        ext_te = load_feat(ext_test_path, device)
        ext_tr = load_feat(ext_train_path, device)
        n_te = min(ext_te.shape[0], lab_test.shape[0])
        n_tr = min(ext_tr.shape[0], lab_train.shape[0])
        ext_te, ext_tr = ext_te[:n_te], ext_tr[:n_tr]

        for anc_key, anc_name, anc_obj in ANCHORS:
            anc_te = load_feat(Path(f"data/features/{anc_key}.pt"), device)[:n_te]
            anc_tr = load_feat(Path(f"data/features_train/{anc_key}.pt"), device)[:n_tr]

            # Atlas-style scorecard (fast mode: 300 MLP steps, 100 permutations)
            logger.info("Scorecard: %s vs %s", ext_name, anc_name)
            try:
                card = compute_scorecard(
                    model_features=ext_te,
                    reference_features=anc_te,
                    model_name=ext_name,
                    reference_name=anc_name,
                    k=10,
                    n_permutations=100,
                    mlp_steps=300,
                    seed=42,
                )
            except Exception as e:
                logger.error("Scorecard failed %s/%s: %s", ext_name, anc_name, e)
                continue

            # Downstream k-NN: fit Procrustes ext_tr -> anc_tr, apply to ext_te
            with torch.no_grad():
                pstate = fit_procrustes(ext_tr, anc_tr)
                mapped_te = apply_procrustes(ext_te, pstate)

            # Native anchor k-NN baseline (gallery=train anc, query=test anc)
            knn_anchor = knn_label_transfer(
                anc_te, lab_test[:n_te], anc_tr, lab_train[:n_tr], k=10
            )
            # Stitched external->anchor k-NN
            knn_stitched = knn_label_transfer(
                mapped_te, lab_test[:n_te], anc_tr, lab_train[:n_tr], k=10
            )

            pair_result = {
                "external": ext_name,
                "external_key": ext_key,
                "external_objective": ext_obj,
                "anchor": anc_name,
                "anchor_key": anc_key,
                "anchor_objective": anc_obj,
                "bitrate_ext": round(card.bitrate, 3),
                "delta": round(card.delta, 3),
                "s_local": round(card.s_local, 3),
                "s_global": round(card.s_global, 3),
                "tau": round(card.tau, 3),
                "regime": card.regime,
                "knn_anchor": round(knn_anchor, 4),
                "knn_stitched": round(knn_stitched, 4),
                "retention": round(knn_stitched / max(knn_anchor, 1e-10), 4),
            }
            results.append(pair_result)
            logger.info(
                "  tau=%.3f S_local=%.3f S_global=%.3f regime=%s | retention=%.3f",
                card.tau, card.s_local, card.s_global, card.regime,
                pair_result["retention"],
            )

    # Aggregate: regime prediction vs downstream retention
    if results:
        taus = np.array([r["tau"] for r in results])
        retentions = np.array([r["retention"] for r in results])
        s_locals = np.array([r["s_local"] for r in results])
        s_globals = np.array([r["s_global"] for r in results])
        rho_tau, p_tau = spearmanr(taus, retentions)
        rho_slocal, p_slocal = spearmanr(s_locals, retentions)
        rho_sglobal, p_sglobal = spearmanr(s_globals, retentions)

        regime_counts = {}
        for r in results:
            regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

        summary = {
            "n_pairs": len(results),
            "n_external_models": len({r["external_key"] for r in results}),
            "n_anchor_models": len({r["anchor_key"] for r in results}),
            "regime_distribution": regime_counts,
            "mean_retention": round(float(retentions.mean()), 4),
            "spearman_tau_retention": {
                "rho": round(float(rho_tau), 3),
                "p": float(f"{p_tau:.2e}"),
            },
            "spearman_slocal_retention": {
                "rho": round(float(rho_slocal), 3),
                "p": float(f"{p_slocal:.2e}"),
            },
            "spearman_sglobal_retention": {
                "rho": round(float(rho_sglobal), 3),
                "p": float(f"{p_sglobal:.2e}"),
            },
        }
    else:
        summary = {"n_pairs": 0}

    output = {
        "description": "Scorecard case study on 5 external models x 3 anchors",
        "anchors": [{"key": a[0], "name": a[1], "objective": a[2]} for a in ANCHORS],
        "external": [{"key": e[0], "name": e[1], "objective": e[2]} for e in EXTERNAL],
        "results": results,
        "summary": summary,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    logger.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()
