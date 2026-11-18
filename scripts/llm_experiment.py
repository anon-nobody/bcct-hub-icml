"""
Phase 3: LLM Representation Convergence Experiment for BCCT Survey.

Extracts hidden states from 6 open LLMs at multiple layers,
computes pairwise BCCT metrics, and generates figures for §9.

Models (chosen for family diversity + manageable size):
  - EleutherAI/pythia-410m      (Pythia family, small)
  - EleutherAI/pythia-1.4b      (Pythia family, medium)
  - EleutherAI/pythia-2.8b      (Pythia family, large — scaling analysis)
  - facebook/opt-1.3b            (OPT family)
  - microsoft/phi-2             (Phi family, 2.7B)
  - openai-community/gpt2-large (GPT-2 family, 774M)

Dataset: WikiText-103 (first 2000 samples, 128 tokens each)

Usage:
    python scripts/llm_experiment.py --device cuda:2 --n-samples 2000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Model Registry
# ============================================================================

LLM_MODELS = [
    {
        "name": "pythia-410m",
        "hf_id": "EleutherAI/pythia-410m",
        "family": "pythia",
        "params": "410M",
        "layers": 24,
        "dim": 1024,
    },
    {
        "name": "pythia-1.4b",
        "hf_id": "EleutherAI/pythia-1.4b",
        "family": "pythia",
        "params": "1.4B",
        "layers": 24,
        "dim": 2048,
    },
    {
        "name": "pythia-2.8b",
        "hf_id": "EleutherAI/pythia-2.8b",
        "family": "pythia",
        "params": "2.8B",
        "layers": 32,
        "dim": 2560,
    },
    {
        "name": "opt-1.3b",
        "hf_id": "facebook/opt-1.3b",
        "family": "opt",
        "params": "1.3B",
        "layers": 24,
        "dim": 2048,
    },
    {
        "name": "phi-2",
        "hf_id": "microsoft/phi-2",
        "family": "phi",
        "params": "2.7B",
        "layers": 32,
        "dim": 2560,
    },
    {
        "name": "gpt2-large",
        "hf_id": "openai-community/gpt2-large",
        "family": "gpt2",
        "params": "774M",
        "layers": 36,
        "dim": 1280,
    },
    {
        "name": "pythia-6.9b",
        "hf_id": "EleutherAI/pythia-6.9b",
        "family": "pythia",
        "params": "6.9B",
        "layers": 32,
        "dim": 4096,
    },
    {
        "name": "mistral-7b",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "family": "mistral",
        "params": "7.2B",
        "layers": 32,
        "dim": 4096,
    },
    {
        "name": "falcon-7b",
        "hf_id": "tiiuae/falcon-7b",
        "family": "falcon",
        "params": "7.2B",
        "layers": 32,
        "dim": 4544,
    },
]


# ============================================================================
# Feature Extraction
# ============================================================================


def get_layer_indices(n_layers: int, n_points: int = 5) -> list[int]:
    """Get evenly-spaced layer indices including first and last."""
    if n_layers <= n_points:
        return list(range(n_layers))
    indices = [int(i * (n_layers - 1) / (n_points - 1)) for i in range(n_points)]
    return sorted(set(indices))


def extract_llm_features(
    model_spec: dict,
    n_samples: int,
    max_length: int,
    device: str,
    output_dir: Path,
) -> dict[int, Tensor]:
    """Extract hidden states from an LLM at multiple layer positions.

    Returns:
        Dict mapping layer_index -> Tensor(n_samples, dim)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_file = output_dir / f"{model_spec['name']}.pt"
    if output_file.exists():
        logger.info("Loading cached features: %s", output_file)
        data = torch.load(output_file, weights_only=True)
        return data["layer_features"]

    name = model_spec["name"]
    hf_id = model_spec["hf_id"]
    logger.info("Loading model: %s (%s)", name, hf_id)

    # Phi-2 and Falcon use custom model code not yet merged into transformers.
    _needs_remote_code = model_spec.get("family") in {"phi", "falcon"}
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=_needs_remote_code,  # Required: model uses custom code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=_needs_remote_code,  # Required: model uses custom code
        output_hidden_states=True,
    )
    model.eval()

    # Load WikiText-103
    logger.info("Loading WikiText-103 dataset...")
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][: n_samples * 2]

    # Tokenize
    logger.info("Tokenizing %d texts (max_length=%d)...", len(texts), max_length)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Select layer indices
    n_layers = model_spec["layers"]
    layer_indices = get_layer_indices(n_layers, n_points=5)
    logger.info("Extracting layers %s from %d-layer model", layer_indices, n_layers)

    # Extract hidden states
    layer_features: dict[int, list[Tensor]] = {li: [] for li in layer_indices}
    batch_size = 16
    collected = 0

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {name}"):
            if collected >= n_samples:
                break
            end = min(start + batch_size, len(texts))
            batch_ids = encodings["input_ids"][start:end].to(device)
            batch_mask = encodings["attention_mask"][start:end].to(device)

            try:
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            except Exception as e:
                logger.warning("Batch %d failed: %s", start, e)
                continue

            hidden_states = outputs.hidden_states  # tuple of (batch, seq, dim)

            for li in layer_indices:
                # Use mean pooling over non-padding tokens
                hs = hidden_states[li].float()  # (batch, seq, dim)
                mask = batch_mask.unsqueeze(-1).float()  # (batch, seq, 1)
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                layer_features[li].append(pooled.cpu())

            collected += end - start

    # Concatenate and trim to n_samples
    result: dict[int, Tensor] = {}
    for li in layer_indices:
        if layer_features[li]:
            feat = torch.cat(layer_features[li], dim=0)[:n_samples]
            result[li] = feat
            logger.info("  Layer %d: shape %s", li, feat.shape)

    # Save
    torch.save(
        {
            "layer_features": result,
            "model_spec": model_spec,
            "n_samples": n_samples,
            "max_length": max_length,
            "layer_indices": layer_indices,
        },
        output_file,
    )
    logger.info("Saved features to %s", output_file)

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================================
# BCCT Metrics Computation
# ============================================================================


def compute_llm_bcct_metrics(
    features_dict: dict[str, dict[int, Tensor]],
    model_specs: dict[str, dict],
    device: str,
) -> dict:
    """Compute pairwise BCCT metrics for LLM representations.

    Uses the 3/4-depth layer as the "representative" layer for each model.
    """
    from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
    from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
    from latent.metrics.regime import classify_regime
    from latent.metrics.transport import transport_asymmetry_index

    results = {
        "models": {},
        "pairwise": {},
        "layerwise": {},
    }

    # Per-model metrics at representative layer (3/4 depth)
    for name, layer_feats in features_dict.items():
        spec = model_specs[name]
        n_layers = spec["layers"]
        rep_layer = get_layer_indices(n_layers, 5)[3]  # 3/4 depth

        if rep_layer not in layer_feats:
            rep_layer = max(layer_feats.keys())

        feats = layer_feats[rep_layer].to(device)
        bitrate_result = representation_bitrate(feats)
        r = bitrate_result.bitrate
        results["models"][name] = {
            "bitrate": round(r, 3),
            "effective_rank": round(bitrate_result.effective_rank_value, 1),
            "dim": feats.shape[1],
            "n_samples": feats.shape[0],
            "rep_layer": rep_layer,
            "family": spec["family"],
            "params": spec["params"],
        }
        logger.info("Model %s: R=%.3f, dim=%d, rep_layer=%d", name, r, feats.shape[1], rep_layer)

    # Pairwise metrics at representative layer
    model_names = sorted(features_dict.keys())
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if j <= i:
                continue

            spec_a = model_specs[name_a]
            spec_b = model_specs[name_b]
            rep_a = get_layer_indices(spec_a["layers"], 5)[3]
            rep_b = get_layer_indices(spec_b["layers"], 5)[3]

            if rep_a not in features_dict[name_a]:
                rep_a = max(features_dict[name_a].keys())
            if rep_b not in features_dict[name_b]:
                rep_b = max(features_dict[name_b].keys())

            fa = features_dict[name_a][rep_a].to(device)
            fb = features_dict[name_b][rep_b].to(device)
            n = min(fa.shape[0], fb.shape[0])
            fa, fb = fa[:n], fb[:n]

            logger.info("Computing %s <-> %s (layers %d, %d)...", name_a, name_b, rep_a, rep_b)

            s_local, raw_overlap, _ = null_calibrated_mutual_knn(
                fa, fb, k=10, n_permutations=100, seed=42
            )
            s_global = centered_kernel_alignment(fa, fb)
            delta = bottleneck_mismatch(fa, fb)
            transport = transport_asymmetry_index(fa, fb, mlp_steps=500, seed=42)

            tau = min(transport.a_to_b.tau, transport.b_to_a.tau)
            lambda_val = s_local / s_global if s_global > 1e-10 else float("inf")

            r_a = representation_bitrate(fa).bitrate
            r_b = representation_bitrate(fb).bitrate
            regime = classify_regime(
                bitrate=(r_a + r_b) / 2,
                s_local=s_local,
                tau=tau,
                s_global=s_global,
                lambda_val=lambda_val,
            ).regime.value

            pair_key = f"{name_a} <-> {name_b}"
            results["pairwise"][pair_key] = {
                "s_local": round(float(s_local), 4),
                "s_global": round(float(s_global), 4),
                "lambda": round(float(lambda_val), 3),
                "tau": round(float(tau), 4),
                "tau_a_to_b": round(float(transport.a_to_b.tau), 4),
                "tau_b_to_a": round(float(transport.b_to_a.tau), 4),
                "tai": round(float(transport.tai), 4),
                "delta": round(float(delta), 3),
                "regime": regime,
                "family_a": spec_a["family"],
                "family_b": spec_b["family"],
            }
            logger.info(
                "  s_local=%.4f, s_global=%.4f, tau=%.4f, delta=%.3f -> %s",
                s_local,
                s_global,
                tau,
                delta,
                regime,
            )

    # Layer-wise alignment for selected interesting pairs
    # Pythia scaling: 410m vs 2.8b (same family, different scale)
    # Cross-family: pythia-1.4b vs opt-1.3b (similar size, different family)
    interesting_pairs = [
        ("pythia-410m", "pythia-2.8b"),
        ("pythia-1.4b", "opt-1.3b"),
        ("phi-2", "gpt2-large"),
        ("mistral-7b", "pythia-6.9b"),
        ("falcon-7b", "mistral-7b"),
    ]

    for name_a, name_b in interesting_pairs:
        if name_a not in features_dict or name_b not in features_dict:
            continue

        spec_a = model_specs[name_a]
        spec_b = model_specs[name_b]
        layers_a = get_layer_indices(spec_a["layers"], 5)
        layers_b = get_layer_indices(spec_b["layers"], 5)

        pair_key = f"{name_a} <-> {name_b}"
        results["layerwise"][pair_key] = []

        # Compare at matched fractional depth
        for frac_idx in range(min(len(layers_a), len(layers_b))):
            la = layers_a[frac_idx]
            lb = layers_b[frac_idx]
            if la not in features_dict[name_a] or lb not in features_dict[name_b]:
                continue

            fa = features_dict[name_a][la].to(device)
            fb = features_dict[name_b][lb].to(device)
            n = min(fa.shape[0], fb.shape[0])
            fa, fb = fa[:n], fb[:n]

            s_local, _, _ = null_calibrated_mutual_knn(fa, fb, k=10, n_permutations=50, seed=42)
            s_global = centered_kernel_alignment(fa, fb)

            results["layerwise"][pair_key].append(
                {
                    "frac_depth": round(frac_idx / (min(len(layers_a), len(layers_b)) - 1), 2),
                    "layer_a": la,
                    "layer_b": lb,
                    "s_local": round(float(s_local), 4),
                    "s_global": round(float(s_global), 4),
                }
            )

    return results


# ============================================================================
# Figure Generation
# ============================================================================


def generate_figures(results: dict, fig_dir: Path) -> None:
    """Generate Phase 3 figures for §9."""

    # --- Figure 12: LLM Pairwise BCCT Heatmap ---
    model_names = sorted(results["models"].keys())
    n = len(model_names)
    s_local_matrix = np.zeros((n, n))
    tau_matrix = np.zeros((n, n))

    for pair_key, metrics in results["pairwise"].items():
        parts = pair_key.split(" <-> ")
        i = model_names.index(parts[0])
        j = model_names.index(parts[1])
        s_local_matrix[i, j] = metrics["s_local"]
        s_local_matrix[j, i] = metrics["s_local"]
        tau_matrix[i, j] = metrics["tau"]
        tau_matrix[j, i] = metrics["tau"]

    np.fill_diagonal(s_local_matrix, 1.0)
    np.fill_diagonal(tau_matrix, 1.0)

    short_names = []
    for m in model_names:
        spec = results["models"][m]
        short_names.append(f"{m}\n({spec['params']})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(s_local_matrix, cmap="YlGnBu", vmin=0, vmax=max(0.5, s_local_matrix.max()))
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(short_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            val = s_local_matrix[i, j]
            color = "white" if val > 0.3 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax1.set_title(r"$S_{\mathrm{local}}$ (null-calibrated $k$-NN)", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(tau_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(short_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            val = tau_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax2.set_title(r"$\tau$ (Transport Linearity)", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(
        "LLM Representation Convergence: BCCT Pairwise Metrics", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "fig12_llm_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig12_llm_heatmap.pdf")

    # --- Figure 13: Layer-wise alignment curves ---
    if results["layerwise"]:
        fig, axes = plt.subplots(
            1, len(results["layerwise"]), figsize=(4.5 * len(results["layerwise"]), 4)
        )
        if len(results["layerwise"]) == 1:
            axes = [axes]

        colors = {"s_local": "#2196F3", "s_global": "#FF9800"}

        for idx, (pair_key, layer_data) in enumerate(results["layerwise"].items()):
            ax = axes[idx]
            depths = [d["frac_depth"] for d in layer_data]
            s_locals = [d["s_local"] for d in layer_data]
            s_globals = [d["s_global"] for d in layer_data]

            ax.plot(
                depths,
                s_locals,
                "o-",
                color=colors["s_local"],
                label=r"$S_{\mathrm{local}}$",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                depths,
                s_globals,
                "s--",
                color=colors["s_global"],
                label=r"$S_{\mathrm{global}}$ (CKA)",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel("Fractional Depth", fontsize=10)
            ax.set_ylabel("Alignment Score", fontsize=10)
            ax.set_title(pair_key, fontsize=9, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_ylim(-0.05, max(max(s_locals + s_globals) * 1.2, 0.5))
            ax.grid(True, alpha=0.3)

            for j, (xv, yv) in enumerate(zip(depths, s_locals, strict=True)):
                yoff = 8 if j % 2 == 0 else -10
                ax.annotate(
                    f"{yv:.2f}",
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(0, yoff),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colors["s_local"],
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
                )
            for j, (xv, yv) in enumerate(zip(depths, s_globals, strict=True)):
                yoff = -14 if j % 2 == 0 else 10
                ax.annotate(
                    f"{yv:.2f}",
                    (xv, yv),
                    textcoords="offset points",
                    xytext=(0, yoff),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=colors["s_global"],
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
                )

        fig.suptitle("Layer-Wise Alignment in LLMs", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(fig_dir / "fig13_llm_layerwise.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fig13_llm_layerwise.pdf")

    # --- Figure 14: LLM Phase Diagram (s_local vs tau, colored by family) ---
    family_colors = {
        "pythia": "#2196F3",
        "opt": "#4CAF50",
        "phi": "#FF9800",
        "gpt2": "#9C27B0",
        "mistral": "#E91E63",
        "falcon": "#00BCD4",
    }

    fig, ax = plt.subplots(figsize=(11.8, 6.4))

    # Compute axis limits from actual data
    all_s_local = [m["s_local"] for m in results["pairwise"].values()]
    [m["tau"] for m in results["pairwise"].values()]
    x_min = -0.02  # Start from 0 to show all four regime regions
    x_max = max(all_s_local) + 0.08
    y_min, y_max = -0.02, 1.02

    # Draw four regime regions
    s_thresh = 0.05  # s_local threshold
    tau_thresh = 0.5  # tau threshold

    # Collapsed (high tau, low s_local)
    ax.axvspan(
        x_min,
        s_thresh,
        ymin=(tau_thresh - y_min) / (y_max - y_min),
        ymax=1.0,
        color="#90CAF9",
        alpha=0.18,
        zorder=0,
    )
    # Divergent (low tau, low s_local)
    ax.axvspan(
        x_min,
        s_thresh,
        ymin=0.0,
        ymax=(tau_thresh - y_min) / (y_max - y_min),
        color="#EF9A9A",
        alpha=0.18,
        zorder=0,
    )
    # Convergent (high tau, high s_local)
    ax.axvspan(
        s_thresh,
        x_max,
        ymin=(tau_thresh - y_min) / (y_max - y_min),
        ymax=1.0,
        color="#C8E6C9",
        alpha=0.18,
        zorder=0,
    )
    # Local-Only (low tau, high s_local)
    ax.axvspan(
        s_thresh,
        x_max,
        ymin=0.0,
        ymax=(tau_thresh - y_min) / (y_max - y_min),
        color="#FFF9C4",
        alpha=0.18,
        zorder=0,
    )

    # Regime labels
    ax.text(
        0.025,
        0.78,
        "Collapsed",
        fontsize=9,
        fontweight="bold",
        color="#1565C0",
        ha="center",
        va="center",
        alpha=0.7,
    )
    ax.text(
        0.025,
        0.22,
        "Divergent",
        fontsize=9,
        fontweight="bold",
        color="#B71C1C",
        ha="center",
        va="center",
        alpha=0.7,
    )
    mid_x = (s_thresh + x_max) / 2
    ax.text(
        mid_x,
        0.85,
        "Convergent",
        fontsize=11,
        fontweight="bold",
        color="#2E7D32",
        ha="center",
        va="center",
        alpha=0.5,
    )
    ax.text(
        mid_x,
        0.15,
        "Local-Only",
        fontsize=11,
        fontweight="bold",
        color="#F57F17",
        ha="center",
        va="center",
        alpha=0.5,
    )

    # Threshold lines
    ax.axhline(y=tau_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=s_thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Plot data points
    points_data: list[tuple[str, float, float]] = []
    for pair_key, metrics in results["pairwise"].items():
        parts = pair_key.split(" <-> ")
        fa = metrics["family_a"]
        fb = metrics["family_b"]
        color = family_colors.get(fa, "#999999") if fa == fb else "#999999"
        marker = "o" if fa == fb else "^"

        ax.scatter(
            metrics["s_local"],
            metrics["tau"],
            c=color,
            marker=marker,
            s=100,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        short_label = pair_key.replace(" <-> ", "\n\u2194 ")
        points_data.append((short_label, metrics["s_local"], metrics["tau"]))

    # Place labels using left/right margin groups split by tau threshold
    if points_data:
        convergent_pts = [p for p in points_data if p[2] >= tau_thresh]
        local_only_pts = [p for p in points_data if p[2] < tau_thresh]

        def place_group(
            group: list[tuple[str, float, float]],
            x_frac: float,
            ha: str,
            y_start: float,
            y_end: float,
        ) -> None:
            if not group:
                return
            ordered = sorted(group, key=lambda item: item[2], reverse=True)
            y_slots = np.linspace(y_start, y_end, len(ordered))
            for (label, x_val, y_val), y_frac in zip(ordered, y_slots, strict=False):
                ax.annotate(
                    label,
                    xy=(x_val, y_val),
                    xycoords="data",
                    xytext=(x_frac, float(y_frac)),
                    textcoords="axes fraction",
                    ha=ha,
                    va="center",
                    fontsize=6.5,
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "gray", "alpha": 0.9},
                    arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.6, "alpha": 0.9},
                    clip_on=False,
                )

        # Convergent labels on left side, Local-Only on right side
        place_group(convergent_pts, 0.04, "left", 0.95, 0.55)
        place_group(local_only_pts, 0.96, "right", 0.45, 0.05)

    # Legend
    handles = [mpatches.Patch(color=c, label=f) for f, c in family_colors.items()]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Same family",
        )
    )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Cross family",
        )
    )
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3)

    ax.set_xlabel(r"$S_{\mathrm{local}}$ (null-calibrated)", fontsize=11)
    ax.set_ylabel(r"$\tau$ (Transport Linearity)", fontsize=11)
    ax.set_title("LLM BCCT Phase Diagram", fontsize=12, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.11, top=0.82)
    fig.savefig(fig_dir / "fig14_llm_phase_diagram.pdf", dpi=300)
    plt.close(fig)
    logger.info("Saved fig14_llm_phase_diagram.pdf")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM BCCT Experiment")
    parser.add_argument("--device", default="cuda:2", help="GPU device")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default="data/llm_features")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Subset of model names to run (default: all)"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "paper" / "figures"
    results_path = root / "data" / "llm_atlas.json"

    # Select models
    models_to_run = LLM_MODELS
    if args.models:
        models_to_run = [m for m in LLM_MODELS if m["name"] in args.models]

    model_specs = {m["name"]: m for m in models_to_run}

    # Step 1: Extract features
    if not args.skip_extraction:
        features_dict: dict[str, dict[int, Tensor]] = {}
        for spec in models_to_run:
            try:
                feats = extract_llm_features(
                    spec, args.n_samples, args.max_length, args.device, output_dir
                )
                features_dict[spec["name"]] = feats
            except Exception as e:
                logger.error("Failed to extract %s: %s", spec["name"], e)
                continue
    else:
        # Load cached features
        features_dict = {}
        for spec in models_to_run:
            pt_file = output_dir / f"{spec['name']}.pt"
            if pt_file.exists():
                data = torch.load(pt_file, weights_only=True)
                features_dict[spec["name"]] = data["layer_features"]
                logger.info("Loaded cached: %s", spec["name"])

    if len(features_dict) < 2:
        logger.error("Need at least 2 models, got %d", len(features_dict))
        return

    # Step 2: Compute BCCT metrics
    logger.info("Computing BCCT metrics for %d models...", len(features_dict))
    results = compute_llm_bcct_metrics(features_dict, model_specs, args.device)

    # Save results
    # Convert any remaining tensors to Python types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_results = json.loads(json.dumps(results, default=to_serializable))
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info("Saved LLM atlas to %s", results_path)

    # Step 3: Generate figures
    generate_figures(serializable_results, fig_dir)

    # Step 4: Print summary
    print("\n" + "=" * 60)
    print("LLM BCCT EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"\nModels: {len(results['models'])}")
    for name, m in results["models"].items():
        print(f"  {name}: R={m['bitrate']:.3f}, dim={m['dim']}, erank={m['effective_rank']:.1f}")

    print(f"\nPairwise comparisons: {len(results['pairwise'])}")
    regimes = {}
    for pair_key, p in results["pairwise"].items():
        regime = p["regime"]
        regimes[regime] = regimes.get(regime, 0) + 1
        print(
            f"  {pair_key}: s_local={p['s_local']:.4f}, tau={p['tau']:.4f}, "
            f"delta={p['delta']:.3f} -> {regime}"
        )

    print("\nRegime distribution:")
    for r, c in sorted(regimes.items(), key=lambda x: -x[1]):
        total = len(results["pairwise"])
        print(f"  {r}: {c}/{total} ({100 * c / total:.0f}%)")


if __name__ == "__main__":
    main()
