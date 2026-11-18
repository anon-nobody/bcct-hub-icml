"""Generate Figure 21: Video domain BCCT atlas.

Heatmap showing pairwise s_local and tau for 6 video models (15 pairs).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data/video_atlas.json") as f:
    data = json.load(f)
models = [m["name"] for m in data["models"]]
n = len(models)

# Build matrices
s_local_matrix = np.full((n, n), np.nan)
tau_matrix = np.full((n, n), np.nan)

name_to_idx = {m: i for i, m in enumerate(models)}
for p in data["pairs"]:
    i = name_to_idx[p["model_a"]]
    j = name_to_idx[p["model_b"]]
    s_local_matrix[i, j] = p["s_local"]
    s_local_matrix[j, i] = p["s_local"]
    tau_matrix[i, j] = p["tau"]
    tau_matrix[j, i] = p["tau"]

# Short labels
SHORT_LABELS = {
    "videomae-base": "VideoMAE-B",
    "timesformer-base": "TimeSfmr-B",
    "vivit-base": "ViViT-B",
    "xclip-base": "X-CLIP-B",
    "r3d-18": "R3D-18",
    "mvit-v2-s": "MViT-v2-S",
}
labels = [SHORT_LABELS.get(m, m) for m in models]

# Family colors for labels
FAMILY_COLORS = {
    "videomae": "#4CAF50",
    "timesformer": "#2196F3",
    "vivit": "#FF9800",
    "xclip": "#9C27B0",
    "resnet3d": "#F44336",
    "mvit": "#795548",
}
family_list = [data["models"][i]["family"] for i in range(n)]

# Plot
plt.style.use("seaborn-v0_8-paper")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: s_local heatmap
im1 = ax1.imshow(s_local_matrix, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="equal")
ax1.set_xticks(range(n))
ax1.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
ax1.set_yticks(range(n))
ax1.set_yticklabels(labels, fontsize=7.5)
ax1.set_title("Null-Calibrated $s_{\\mathrm{local}}$", fontsize=11, pad=8)
for i in range(n):
    for j in range(n):
        if not np.isnan(s_local_matrix[i, j]):
            val = s_local_matrix[i, j]
            color = "white" if val > 0.35 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
plt.colorbar(im1, ax=ax1, shrink=0.8, label="$s_{\\mathrm{local}}$")

# Color labels by family
for i, label in enumerate(ax1.get_yticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")
for i, label in enumerate(ax1.get_xticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")

# Right: tau heatmap
im2 = ax2.imshow(tau_matrix, cmap="YlGnBu", vmin=0, vmax=1.0, aspect="equal")
ax2.set_xticks(range(n))
ax2.set_xticklabels(labels, fontsize=7.5, rotation=45, ha="right")
ax2.set_yticks(range(n))
ax2.set_yticklabels(labels, fontsize=7.5)
ax2.set_title("Transport Linearity $\\tau$", fontsize=11, pad=8)
for i in range(n):
    for j in range(n):
        if not np.isnan(tau_matrix[i, j]):
            val = tau_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
plt.colorbar(im2, ax=ax2, shrink=0.8, label="$\\tau$")

for i, label in enumerate(ax2.get_yticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")
for i, label in enumerate(ax2.get_xticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")

fig.suptitle(
    "Video Domain Atlas: 6 Models, 15 Pairs (STL-10 pseudo-video, 500 clips)",
    fontsize=11,
    y=1.02,
)
plt.tight_layout()

output_dir = Path("paper/figures")
fig.savefig(output_dir / "fig21_video_atlas.pdf", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "fig21_video_atlas.png", dpi=300, bbox_inches="tight")
print("Saved fig21_video_atlas.pdf and .png")

# Print summary
print(f"\nRegime distribution: {data['regime_counts']}")
for p in data["pairs"]:
    print(f"  {p['pair']}: s_local={p['s_local']:.4f}, tau={p['tau']:.4f}, regime={p['regime']}")

plt.close()
