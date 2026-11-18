"""Metric ablation: compare R², CCA, Procrustes distance, cosine as Perf functions.

Usage:
    python scripts/metric_ablation.py [--device cuda:0]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)

PAIRS = [
    ("clip_vitb16", "clip_vitl14"),
    ("resnet50", "resnet101"),
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("convnext_base", "convnext_large"),
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    ("beit_base_patch16_224", "resnet50"),
    ("clip_vitb32", "dinov2_vits14"),
    ("mixer_b16_224", "vit_base_patch16_224"),
    ("efficientnet_b0", "swin_base_patch4_window7_224"),
]


def procrustes_fit(src_train, tgt_train):
    """SVD-based orthogonal Procrustes."""
    src_c = src_train - src_train.mean(dim=0, keepdim=True)
    tgt_c = tgt_train - tgt_train.mean(dim=0, keepdim=True)
    d_s, d_t = src_c.shape[1], tgt_c.shape[1]
    if d_s != d_t:
        max_d = max(d_s, d_t)
        src_c = torch.nn.functional.pad(src_c, (0, max_d - d_s))
        tgt_c = torch.nn.functional.pad(tgt_c, (0, max_d - d_t))
    U, _, Vt = torch.linalg.svd(tgt_c.T @ src_c, full_matrices=False)
    W = (U @ Vt).T
    return W, src_train.mean(dim=0), tgt_train.mean(dim=0)


def predict_procrustes(features, W, src_mean, tgt_mean):
    """Apply Procrustes map."""
    d_s, d_t = features.shape[1], tgt_mean.shape[0]
    src_c = features - src_mean[:d_s]
    if src_c.shape[1] < W.shape[0]:
        src_c = torch.nn.functional.pad(src_c, (0, W.shape[0] - src_c.shape[1]))
    pred = src_c @ W
    return pred[:, :d_t] + tgt_mean


def perf_r2_perdim(pred, target):
    """Per-dimension R² (current default)."""
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-10)
    return r2.clamp(min=0.0).mean().item()


def perf_cca(source, target):
    """Mean canonical correlation."""
    # Center
    src = source - source.mean(dim=0, keepdim=True)
    tgt = target - target.mean(dim=0, keepdim=True)
    n = src.shape[0]
    d = min(src.shape[1], tgt.shape[1], n - 1)

    # Cross-covariance via SVD
    C = (src.T @ tgt) / (n - 1)
    Sxx = (src.T @ src) / (n - 1) + 1e-6 * torch.eye(src.shape[1], device=src.device)
    Syy = (tgt.T @ tgt) / (n - 1) + 1e-6 * torch.eye(tgt.shape[1], device=tgt.device)

    Sxx_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Sxx)).T
    Syy_inv_sqrt = torch.linalg.inv(torch.linalg.cholesky(Syy)).T

    T = Sxx_inv_sqrt @ C @ Syy_inv_sqrt
    _, s, _ = torch.linalg.svd(T, full_matrices=False)
    return s[:d].clamp(max=1.0).mean().item()


def perf_cosine(pred, target):
    """Mean cosine similarity."""
    pred_n = pred / pred.norm(dim=1, keepdim=True).clamp(min=1e-10)
    tgt_n = target / target.norm(dim=1, keepdim=True).clamp(min=1e-10)
    return (pred_n * tgt_n).sum(dim=1).mean().item()


def perf_procrustes_dist(pred, target):
    """1 - normalized Procrustes distance."""
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    tgt_c = target - target.mean(dim=0, keepdim=True)
    pred_n = pred_c / pred_c.norm().clamp(min=1e-10)
    tgt_n = tgt_c / tgt_c.norm().clamp(min=1e-10)
    dist = (pred_n - tgt_n).norm().item()
    return max(0.0, 1.0 - dist / 2.0)  # Normalized to [0,1]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path("data/features")
    atlas = json.loads(Path("data/atlas.json").read_text())

    perf_funcs = {
        "r2_perdim": perf_r2_perdim,
        "cosine": perf_cosine,
        "procrustes_dist": perf_procrustes_dist,
    }

    results = []
    for model_a, model_b in PAIRS:
        pair_key = f"{model_a} <-> {model_b}"
        alt_key = f"{model_b} <-> {model_a}"
        atlas_entry = atlas["pairwise"].get(pair_key) or atlas["pairwise"].get(alt_key)
        if not atlas_entry:
            continue

        try:
            fa = torch.load(features_dir / f"{model_a}.pt", weights_only=True)["features"].to(args.device)
            fb = torch.load(features_dir / f"{model_b}.pt", weights_only=True)["features"].to(args.device)
        except FileNotFoundError:
            continue

        n = min(fa.shape[0], fb.shape[0])
        fa, fb = fa[:n], fb[:n]

        torch.manual_seed(42)
        perm = torch.randperm(n, device=fa.device)
        n_train = int(0.8 * n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        # Fit Procrustes
        W, src_mean, tgt_mean = procrustes_fit(fa[train_idx], fb[train_idx])
        pred = predict_procrustes(fa[test_idx], W, src_mean, tgt_mean)

        pair_result = {"pair": pair_key, "tau_atlas": atlas_entry["tau"]}

        for name, fn in perf_funcs.items():
            score = fn(pred, fb[test_idx]) if name == "cosine" else fn(pred, fb[test_idx])
            pair_result[f"perf_linear_{name}"] = round(score, 4)

        # CCA doesn't use Procrustes prediction - it's computed directly
        try:
            cca_score = perf_cca(fa[test_idx], fb[test_idx])
            pair_result["cca_mean"] = round(cca_score, 4)
        except Exception as e:
            logger.warning("CCA failed for %s: %s", pair_key, e)
            pair_result["cca_mean"] = None

        logger.info("%s: %s", pair_key, {k: v for k, v in pair_result.items() if k != "pair"})
        results.append(pair_result)

    # Rank correlations between metrics
    metrics = ["perf_linear_r2_perdim", "perf_linear_cosine", "perf_linear_procrustes_dist", "cca_mean"]
    valid = [r for r in results if all(r.get(m) is not None for m in metrics)]

    correlations = {}
    for i, m1 in enumerate(metrics):
        for m2 in metrics[i + 1:]:
            vals1 = [r[m1] for r in valid]
            vals2 = [r[m2] for r in valid]
            kt, kp = kendalltau(vals1, vals2)
            sr, sp = spearmanr(vals1, vals2)
            correlations[f"{m1}_vs_{m2}"] = {
                "kendall_tau": round(float(kt), 4),
                "spearman_rho": round(float(sr), 4),
            }
            logger.info("%s vs %s: Kendall=%.3f, Spearman=%.3f", m1, m2, kt, sr)

    output = {
        "n_pairs": len(results),
        "results": results,
        "rank_correlations": correlations,
    }
    output_path = Path("data/experiments/metric_ablation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
