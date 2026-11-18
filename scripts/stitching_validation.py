"""Downstream stitching validation: correlate τ with actual classification accuracy.

For selected model pairs, trains a linear probe on encoder_B's features,
then evaluates it using stitched encoder_A→encoder_B features.
Reports Spearman correlation between τ and accuracy drop.

Usage:
    python scripts/stitching_validation.py [--device cuda:0]
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

# 20 pairs spanning full tau range (from atlas)
PAIRS = [
    # High tau (same family, same objective)
    ("clip_vitb16", "clip_vitl14"),
    ("clip_vitb32", "clip_vitb16"),
    ("resnet50", "resnet101"),
    ("convnext_base", "convnext_large"),
    ("convnext_base", "swin_base_patch4_window7_224"),
    # Moderate tau (same objective, cross family)
    ("dinov2_vitb14", "dinov2_vitl14"),
    ("dinov2_vitg14", "dinov2_vitl14"),
    ("dinov2_vits14", "dinov2_vitb14"),
    ("vit_base_patch16_224", "vit_large_patch16_224"),
    ("efficientnet_b0", "resnet101"),
    ("mixer_b16_224", "resnet101"),
    ("convnext_large", "vit_base_patch16_224"),
    # Low tau (cross objective)
    ("dinov2_vitb14", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "vit_base_patch16_224"),
    ("dinov2_vitb14", "resnet50"),
    ("beit_base_patch16_224", "resnet50"),
    ("clip_vitl14", "dinov2_vitl14"),
    ("beit_base_patch16_224", "clip_vitb16"),
    ("vit_base_patch16_224.mae", "dinov2_vitb14"),
    ("clip_vitb16", "resnet50"),
]


def load_features_and_labels(features_dir: Path, model_name: str, device: str,
                              split: str = "test"):
    """Load features and CIFAR-100 labels for the specified split.

    split: "train" loads from data/features_train/ + cifar100_train_labels.pt,
           "test" loads from data/features/ + cifar100_test_labels.pt.
    """
    if split == "train":
        pt_file = features_dir.parent / "features_train" / f"{model_name}.pt"
        labels_path = features_dir.parent / "cifar100_train_labels.pt"
    else:
        pt_file = features_dir / f"{model_name}.pt"
        labels_path = features_dir.parent / "cifar100_test_labels.pt"
    data = torch.load(pt_file, weights_only=True)
    features = data["features"].to(device)
    labels = torch.load(labels_path, weights_only=True).to(device)
    n = min(features.shape[0], labels.shape[0])
    return features[:n], labels[:n]


def train_linear_probe(features: torch.Tensor, labels: torch.Tensor,
                       n_classes: int = 100, lr: float = 0.1,
                       n_epochs: int = 500) -> nn.Linear:
    """Train a linear classifier on features."""
    d = features.shape[1]
    # Normalize features for stable training
    feat_mean = features.mean(dim=0, keepdim=True)
    feat_std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
    features_n = (features - feat_mean) / feat_std

    probe = nn.Linear(d, n_classes).to(features.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    probe.train()
    for _epoch in range(n_epochs):
        logits = probe(features_n)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Store normalization stats on the probe for inference
    probe._feat_mean = feat_mean  # type: ignore[attr-defined]
    probe._feat_std = feat_std  # type: ignore[attr-defined]
    return probe


def evaluate_accuracy(probe: nn.Linear, features: torch.Tensor,
                      labels: torch.Tensor) -> float:
    """Evaluate classification accuracy."""
    probe.eval()
    with torch.no_grad():
        # Apply same normalization as training
        feat_mean = probe._feat_mean  # type: ignore[attr-defined]
        feat_std = probe._feat_std  # type: ignore[attr-defined]
        d = feat_mean.shape[1]
        features_n = (features[:, :d] - feat_mean) / feat_std
        logits = probe(features_n)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc


def train_stitch_layer(source_features: torch.Tensor,
                       target_features: torch.Tensor) -> nn.Linear:
    """Train a linear layer mapping source → target feature space (Procrustes)."""
    # Use SVD-based Procrustes (same as in transport.py)
    src = source_features - source_features.mean(dim=0, keepdim=True)
    tgt = target_features - target_features.mean(dim=0, keepdim=True)

    # Pad to same dimension if needed
    d_src, d_tgt = src.shape[1], tgt.shape[1]
    if d_src != d_tgt:
        max_d = max(d_src, d_tgt)
        if d_src < max_d:
            src = torch.nn.functional.pad(src, (0, max_d - d_src))
        if d_tgt < max_d:
            tgt = torch.nn.functional.pad(tgt, (0, max_d - d_tgt))

    U, _, Vt = torch.linalg.svd(tgt.T @ src, full_matrices=False)
    W = (U @ Vt).T  # Orthogonal Procrustes solution

    # Create linear layer
    stitch = nn.Linear(d_src, d_tgt, bias=True).to(source_features.device)
    with torch.no_grad():
        stitch.weight.copy_(W[:d_tgt, :d_src])
        stitch.bias.copy_(target_features.mean(dim=0) - (source_features.mean(dim=0)[:d_src] @ W[:d_tgt, :d_src].T))
    return stitch


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--atlas", type=str, default="data/atlas.json")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="data/experiments/stitching_validation.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_dir = Path(args.features_dir)
    atlas = json.loads(Path(args.atlas).read_text())

    results = []
    for model_a, model_b in PAIRS:
        pair_key = f"{model_a} <-> {model_b}"
        alt_key = f"{model_b} <-> {model_a}"
        atlas_entry = atlas["pairwise"].get(pair_key) or atlas["pairwise"].get(alt_key)
        if not atlas_entry:
            logger.warning("Pair %s not found in atlas, skipping", pair_key)
            continue

        tau = atlas_entry["tau"]
        logger.info("Processing %s (tau=%.3f)", pair_key, tau)

        try:
            # TRAIN split for probe learning (CIFAR-100 train features)
            feat_a_tr, lab_tr = load_features_and_labels(
                features_dir, model_a, args.device, split="train"
            )
            feat_b_tr, _ = load_features_and_labels(
                features_dir, model_b, args.device, split="train"
            )
            # TEST split for evaluation only (CIFAR-100 test features, no overlap)
            feat_a_te, lab_te = load_features_and_labels(
                features_dir, model_a, args.device, split="test"
            )
            feat_b_te, _ = load_features_and_labels(
                features_dir, model_b, args.device, split="test"
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", pair_key, e)
            continue

        n_tr = min(feat_a_tr.shape[0], feat_b_tr.shape[0], lab_tr.shape[0])
        n_te = min(feat_a_te.shape[0], feat_b_te.shape[0], lab_te.shape[0])
        feat_a_tr, feat_b_tr, lab_tr = feat_a_tr[:n_tr], feat_b_tr[:n_tr], lab_tr[:n_tr]
        feat_a_te, feat_b_te, lab_te = feat_a_te[:n_te], feat_b_te[:n_te], lab_te[:n_te]

        # 1. Train linear probe on encoder_B TRAIN features; evaluate on TEST
        probe_b = train_linear_probe(feat_b_tr, lab_tr)
        acc_native = evaluate_accuracy(probe_b, feat_b_te, lab_te)

        # 2. Learn stitch A->B on TRAIN features only
        stitch = train_stitch_layer(feat_a_tr, feat_b_tr)

        # 3. Apply stitch to TEST features (held out during probe training)
        with torch.no_grad():
            stitched_train = stitch(feat_a_tr)
            stitched_test = stitch(feat_a_te)

        # 4a. Frozen-head: evaluate B's probe on stitched TEST features
        acc_frozen = evaluate_accuracy(probe_b, stitched_test, lab_te)

        # 4b. Retrained: train NEW probe on stitched TRAIN features, evaluate on stitched TEST
        probe_stitched = train_linear_probe(stitched_train, lab_tr)
        acc_retrained = evaluate_accuracy(probe_stitched, stitched_test, lab_te)

        # 5. Metrics
        retention_frozen = acc_frozen / max(acc_native, 1e-10)
        retention_retrained = acc_retrained / max(acc_native, 1e-10)
        acc_drop_frozen = acc_native - acc_frozen
        acc_drop_retrained = acc_native - acc_retrained

        logger.info("  native=%.3f, frozen=%.3f, retrained=%.3f, tau=%.3f",
                     acc_native, acc_frozen, acc_retrained, tau)

        results.append({
            "pair": pair_key,
            "model_a": model_a,
            "model_b": model_b,
            "tau": round(tau, 4),
            "acc_native": round(acc_native, 4),
            "acc_frozen_head": round(acc_frozen, 4),
            "acc_retrained_probe": round(acc_retrained, 4),
            "acc_drop_frozen": round(acc_drop_frozen, 4),
            "acc_drop_retrained": round(acc_drop_retrained, 4),
            "retention_frozen": round(retention_frozen, 4),
            "retention_retrained": round(retention_retrained, 4),
        })

    # Compute correlations for both protocols
    taus = np.array([r["tau"] for r in results])
    drops_frozen = np.array([r["acc_drop_frozen"] for r in results])
    drops_retrained = np.array([r["acc_drop_retrained"] for r in results])

    rho_frozen, p_frozen = spearmanr(taus, -drops_frozen)
    rho_retrained, p_retrained = spearmanr(taus, -drops_retrained)

    summary = {
        "n_pairs": len(results),
        "frozen_head": {
            "spearman_rho": round(float(rho_frozen), 4),
            "spearman_p": float(f"{p_frozen:.2e}"),
            "mean_acc": round(float(np.mean([r["acc_frozen_head"] for r in results])), 4),
        },
        "retrained_probe": {
            "spearman_rho": round(float(rho_retrained), 4),
            "spearman_p": float(f"{p_retrained:.2e}"),
            "mean_acc": round(float(np.mean([r["acc_retrained_probe"] for r in results])), 4),
        },
        "mean_native_acc": round(float(np.mean([r["acc_native"] for r in results])), 4),
    }

    logger.info("\n=== Frozen-head: rho=%.3f, p=%.2e", rho_frozen, p_frozen)
    logger.info("=== Retrained:   rho=%.3f, p=%.2e", rho_retrained, p_retrained)

    output = {"results": results, "summary": summary}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
