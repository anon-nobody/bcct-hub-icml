"""Downstream utility experiment: Does tau-guided model selection beat alternatives?

For each target model B, rank candidate source models A by different criteria
(tau, CKA, delta, random), then measure actual stitching accuracy.
Reports whether tau-guided selection picks better source models.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/model_selection_utility.py --device cuda:0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

ATLAS_PATH = Path("data/atlas.json")
FEATURES_DIR = Path("data/features")
OUTPUT_PATH = Path("data/experiments/model_selection_utility.json")

# Target models: pick diverse targets spanning objectives
TARGETS = [
    "resnet50",           # supervised
    "vit_base_patch16_224",  # supervised
    "clip_vitb16",        # contrastive
    "dinov2_vitb14",      # self-supervised
    "beit_base_patch16_224",  # self-supervised
]


def load_features(model_name: str, device: str, split: str = "test") -> torch.Tensor:
    """Load pre-extracted features for the specified CIFAR-100 split."""
    if split == "train":
        pt_file = FEATURES_DIR.parent / "features_train" / f"{model_name}.pt"
    else:
        pt_file = FEATURES_DIR / f"{model_name}.pt"
    data = torch.load(pt_file, weights_only=True)
    return data["features"].to(device)


def load_labels(device: str, split: str = "test") -> torch.Tensor:
    """Load CIFAR-100 labels for the specified split."""
    name = "cifar100_train_labels.pt" if split == "train" else "cifar100_test_labels.pt"
    labels_path = FEATURES_DIR.parent / name
    return torch.load(labels_path, weights_only=True).to(device)


def train_linear_probe(
    features: torch.Tensor, labels: torch.Tensor, n_epochs: int = 300
) -> nn.Linear:
    """Train linear probe with feature normalization."""
    d = features.shape[1]
    feat_mean = features.mean(dim=0, keepdim=True)
    feat_std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
    features_n = (features - feat_mean) / feat_std

    probe = nn.Linear(d, 100).to(features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.1, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    probe.train()
    for _ in range(n_epochs):
        logits = probe(features_n)
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    probe._feat_mean = feat_mean  # type: ignore[attr-defined]
    probe._feat_std = feat_std  # type: ignore[attr-defined]
    return probe


def evaluate_probe(
    probe: nn.Linear, features: torch.Tensor, labels: torch.Tensor
) -> float:
    """Evaluate accuracy with stored normalization."""
    probe.eval()
    with torch.no_grad():
        d = probe._feat_mean.shape[1]  # type: ignore[attr-defined]
        fn = (features[:, :d] - probe._feat_mean) / probe._feat_std  # type: ignore[attr-defined]
        return float((probe(fn).argmax(1) == labels).float().mean())


def fit_procrustes(src_train: torch.Tensor, tgt_train: torch.Tensor) -> dict:
    """Fit orthogonal Procrustes from src to tgt space on TRAIN data only.

    Returns a state dict that can be applied to held-out src features without
    refitting on tgt — eliminates test-set leakage in downstream evaluation.
    """
    src_mean = src_train.mean(0, keepdim=True)
    tgt_mean = tgt_train.mean(0, keepdim=True)
    src_c = src_train - src_mean
    tgt_c = tgt_train - tgt_mean
    d_s, d_t = src_c.shape[1], tgt_c.shape[1]
    max_d = max(d_s, d_t)
    src_pad = torch.nn.functional.pad(src_c, (0, max_d - d_s)) if d_s < max_d else src_c
    tgt_pad = torch.nn.functional.pad(tgt_c, (0, max_d - d_t)) if d_t < max_d else tgt_c
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
    """Apply a previously fit Procrustes map to src features."""
    d_s, d_t, max_d = state["d_s"], state["d_t"], state["max_d"]
    centered = src - state["src_mean"]
    padded = (
        torch.nn.functional.pad(centered, (0, max_d - d_s)) if d_s < max_d else centered
    )
    return (padded @ state["W"].T)[:, :d_t] + state["tgt_mean"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(ATLAS_PATH) as f:
        atlas = json.load(f)
    models_meta = atlas["models"]
    pairwise = atlas["pairwise"]

    all_models = sorted(models_meta.keys())
    lab_tr = load_labels(args.device, split="train")
    lab_te = load_labels(args.device, split="test")

    results_per_target = []

    for target in TARGETS:
        logger.info(f"\n=== Target: {target} ===")
        candidates = [m for m in all_models if m != target]

        # Load target TRAIN and TEST features separately (no overlap)
        feat_b_tr = load_features(target, args.device, split="train")
        feat_b_te = load_features(target, args.device, split="test")
        n_tr = min(feat_b_tr.shape[0], lab_tr.shape[0])
        n_te = min(feat_b_te.shape[0], lab_te.shape[0])
        feat_b_tr = feat_b_tr[:n_tr]
        feat_b_te = feat_b_te[:n_te]
        lab_tr_cut = lab_tr[:n_tr]
        lab_te_cut = lab_te[:n_te]

        # Native probe on target TRAIN, evaluate on TEST
        probe_b = train_linear_probe(feat_b_tr, lab_tr_cut)
        acc_native = evaluate_probe(probe_b, feat_b_te, lab_te_cut)
        logger.info(f"  Native acc: {acc_native:.3f}")

        # For each candidate, compute stitched accuracy
        candidate_results = []
        for cand in candidates:
            pair_key = f"{cand} <-> {target}"
            alt_key = f"{target} <-> {cand}"
            entry = pairwise.get(pair_key) or pairwise.get(alt_key)
            if not entry:
                continue

            try:
                feat_a_tr = load_features(cand, args.device, split="train")[:n_tr]
                feat_a_te = load_features(cand, args.device, split="test")[:n_te]
            except FileNotFoundError:
                continue

            # Fit Procrustes A→B on TRAIN only; apply the same rotation to TEST
            # (no test target leakage)
            with torch.no_grad():
                pstate = fit_procrustes(feat_a_tr, feat_b_tr)
                mapped_train = apply_procrustes(feat_a_tr, pstate)
                mapped_test = apply_procrustes(feat_a_te, pstate)

            probe_stitch = train_linear_probe(mapped_train, lab_tr_cut)
            acc_stitch = evaluate_probe(probe_stitch, mapped_test, lab_te_cut)

            candidate_results.append({
                "source": cand,
                "tau": entry["tau"],
                "s_global": entry["s_global"],
                "delta": entry["delta"],
                "s_local": entry["s_local"],
                "acc_stitched": round(acc_stitch, 4),
                "retention": round(acc_stitch / max(acc_native, 1e-10), 4),
            })

            logger.info(
                f"  {cand:30s} | tau={entry['tau']:.3f} | CKA={entry['s_global']:.3f} | "
                f"stitch_acc={acc_stitch:.3f} | retention={acc_stitch / max(acc_native, 1e-10):.3f}"
            )

        if not candidate_results:
            continue

        # Compute rank correlations: which metric best predicts stitching success?
        stitch_accs = np.array([c["acc_stitched"] for c in candidate_results])
        metrics = {}
        for metric_name in ["tau", "s_global", "s_local", "delta"]:
            vals = np.array([c[metric_name] for c in candidate_results])
            sign = -1 if metric_name == "delta" else 1  # delta: lower is better
            rho, p = spearmanr(sign * vals, stitch_accs)
            metrics[metric_name] = {"rho": round(float(rho), 3), "p": float(f"{p:.3e}")}

        # Top-1 selection accuracy: does best-by-metric = best-by-accuracy?
        best_by_acc = max(candidate_results, key=lambda c: c["acc_stitched"])
        for metric_name in ["tau", "s_global", "s_local"]:
            best_by_metric = max(candidate_results, key=lambda c: c[metric_name])
            metrics[f"{metric_name}_top1_match"] = best_by_metric["source"] == best_by_acc["source"]
            metrics[f"{metric_name}_top1_source"] = best_by_metric["source"]
            metrics[f"{metric_name}_top1_acc"] = best_by_metric["acc_stitched"]

        results_per_target.append({
            "target": target,
            "acc_native": round(acc_native, 4),
            "n_candidates": len(candidate_results),
            "best_source": best_by_acc["source"],
            "best_acc": best_by_acc["acc_stitched"],
            "metric_correlations": metrics,
            "candidates": candidate_results,
        })

        logger.info(f"  Best source: {best_by_acc['source']} (acc={best_by_acc['acc_stitched']:.3f})")
        logger.info("  Metric correlations with stitch acc:")
        for mn, mv in metrics.items():
            if isinstance(mv, dict):
                logger.info(f"    {mn}: rho={mv['rho']:.3f}, p={mv['p']}")

    # Aggregate across targets
    all_rhos = {m: [] for m in ["tau", "s_global", "s_local", "delta"]}
    for r in results_per_target:
        for m in all_rhos:
            rho = r["metric_correlations"][m]["rho"]
            if not np.isnan(rho):
                all_rhos[m].append(rho)

    logger.info("\n=== Aggregated metric → stitch-accuracy correlations ===")
    summary = {}
    for m, rhos in all_rhos.items():
        summary[m] = {
            "mean_rho": round(float(np.mean(rhos)), 3),
            "std_rho": round(float(np.std(rhos)), 3),
            "n_targets": len(rhos),
        }
        logger.info(f"  {m:12s} | mean_rho={summary[m]['mean_rho']:.3f}±{summary[m]['std_rho']:.3f}")

    output = {
        "description": "Model selection utility: which metric best predicts stitching success?",
        "per_target": results_per_target,
        "aggregated": summary,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
