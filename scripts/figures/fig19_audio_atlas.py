"""Generate Figure 19: Audio domain BCCT atlas.

Heatmap showing pairwise s_local and τ for 6 audio models (15 pairs).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load data
with open("data/audio_atlas.json") as f:
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
    "hubert-base": "HuBERT-B",
    "hubert-large": "HuBERT-L",
    "wav2vec2-base": "W2V2-B",
    "wav2vec2-large": "W2V2-L",
    "whisper-base": "Whisper-B",
    "whisper-small": "Whisper-S",
}
labels = [SHORT_LABELS.get(m, m) for m in models]

# Family colors for y-axis labels
FAMILY_COLORS = {
    "hubert": "#4CAF50",
    "wav2vec2": "#2196F3",
    "whisper": "#FF9800",
}
family_list = [data["models"][i]["family"] for i in range(n)]

# Plot
plt.style.use("seaborn-v0_8-paper")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: s_local heatmap
im1 = ax1.imshow(s_local_matrix, cmap="YlOrRd", vmin=0, vmax=0.5, aspect="equal")
ax1.set_xticks(range(n))
ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax1.set_yticks(range(n))
ax1.set_yticklabels(labels, fontsize=8)
ax1.set_title("Null-Calibrated $s_{\\mathrm{local}}$", fontsize=11, pad=8)
for i in range(n):
    for j in range(n):
        if not np.isnan(s_local_matrix[i, j]):
            val = s_local_matrix[i, j]
            color = "white" if val > 0.35 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im1, ax=ax1, shrink=0.8, label="$s_{\\mathrm{local}}$")

# Color y-axis labels by family
for i, label in enumerate(ax1.get_yticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")
for i, label in enumerate(ax1.get_xticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")

# Right: τ heatmap
im2 = ax2.imshow(tau_matrix, cmap="YlGnBu", vmin=0, vmax=1.0, aspect="equal")
ax2.set_xticks(range(n))
ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
ax2.set_yticks(range(n))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_title("Transport Linearity $\\tau$", fontsize=11, pad=8)
for i in range(n):
    for j in range(n):
        if not np.isnan(tau_matrix[i, j]):
            val = tau_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im2, ax=ax2, shrink=0.8, label="$\\tau$")

# Color y-axis labels
for i, label in enumerate(ax2.get_yticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")
for i, label in enumerate(ax2.get_xticklabels()):
    label.set_color(FAMILY_COLORS[family_list[i]])
    label.set_fontweight("bold")

fig.suptitle(
    "Audio Domain Atlas: 6 Models, 15 Pairs (LibriSpeech test-clean, 2000 samples)",
    fontsize=11,
    y=1.02,
)
plt.tight_layout()

output_dir = Path("paper/figures")
fig.savefig(output_dir / "fig19_audio_atlas.pdf", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "fig19_audio_atlas.png", dpi=300, bbox_inches="tight")
print("Saved fig19_audio_atlas.pdf and .png")
plt.close()
