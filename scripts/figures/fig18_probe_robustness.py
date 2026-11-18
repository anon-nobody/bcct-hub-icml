"""Generate Figure 18: Probe architecture robustness.

Shows transport linearity (τ) across 4 probe architectures for 10 model pairs,
demonstrating that regime classification is stable across probe choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data/probe_robustness.json") as f:
    data = json.load(f)

# Shorter pair labels for readability
PAIR_LABELS = {
    "clip_vitl14 × convnext_base": "CLIP-L\n× CNeXt-B",
    "clip_vitb32 × clip_vitb16": "CLIP-B/32\n× CLIP-B/16",
    "resnet50 × vit_base_patch16_224": "RN50\n× ViT-B",
    "dinov2_vitb14 × clip_vitl14": "DINOv2-B\n× CLIP-L",
    "dinov2_vitb14 × resnet50": "DINOv2-B\n× RN50",
    "resnet50 × mixer_b16_224": "RN50\n× Mixer-B",
    "convnext_base × mixer_b16_224": "CNeXt-B\n× Mixer-B",
    "dinov2_vitb14 × vit_base_patch16_224.mae": "DINOv2-B\n× MAE-B",
    "clip_vitl14 × vit_base_patch16_224.mae": "CLIP-L\n× MAE-B",
    "dinov2_vitb14 × convnext_base": "DINOv2-B\n× CNeXt-B",
}

# Extract data
pairs = [d["pair"] for d in data]
labels = [PAIR_LABELS.get(p, p.replace(" × ", "\n× ")) for p in pairs]

arch_keys = ["1-layer", "2-layer", "2-layer-wide", "3-layer"]
arch_labels = ["1-layer\nMLP", "2-layer\nMLP", "2-layer\nwide MLP", "3-layer\nMLP"]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# Build τ matrix: (n_pairs, n_archs)
tau_matrix = np.array([[d["tau"][k] for k in arch_keys] for d in data])

# Plot
plt.style.use("seaborn-v0_8-paper")
fig, ax = plt.subplots(figsize=(12, 5))

n_pairs = len(pairs)
n_archs = len(arch_keys)
x = np.arange(n_pairs)
width = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

for i, (arch_label, color) in enumerate(zip(arch_labels, colors, strict=False)):
    bars = ax.bar(
        x + offsets[i],
        tau_matrix[:, i],
        width,
        label=arch_label,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

# Threshold line
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, label="τ threshold")

# Formatting
ax.set_xlabel("Model Pair", fontsize=11, labelpad=8)
ax.set_ylabel("Transport Linearity (τ)", fontsize=11)
ax.set_title("Probe Architecture Robustness: τ Across 4 Probe Types", fontsize=12, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, ha="center", linespacing=1.1)
ax.set_ylim(-0.05, 1.15)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.legend(
    fontsize=8.5,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=True,
    framealpha=0.9,
    edgecolor="lightgray",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

plt.tight_layout()

output_dir = Path("paper/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "fig18_probe_robustness.pdf", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "fig18_probe_robustness.png", dpi=300, bbox_inches="tight")
print("Saved fig18_probe_robustness.pdf and .png")
plt.close()
