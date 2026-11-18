"""Generate Figure 20: Audio Bitrate Sweep.

Shows how s_local and τ change as bitrate is modulated via PCA truncation
for 4 representative audio model pairs. Demonstrates that the BCCT
framework's bitrate dependence holds in the audio domain.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# Load sweep data
with open("data/audio_bitrate_sweep.json") as f:
    data = json.load(f)

# Group by pair
pairs: dict[str, list[dict]] = {}
for point in data:
    pair_name = point["pair"]
    if pair_name not in pairs:
        pairs[pair_name] = []
    pairs[pair_name].append(point)

# Sort each pair by bitrate
for pair_name in pairs:
    pairs[pair_name].sort(key=lambda x: x["bitrate_mean"])

# Short labels for legend
SHORT_NAMES = {
    "hubert-base × hubert-large": "HuBERT-B × HuBERT-L",
    "wav2vec2-base × wav2vec2-large": "W2V2-B × W2V2-L",
    "hubert-base × whisper-base": "HuBERT-B × Whisper-B",
    "wav2vec2-large × whisper-small": "W2V2-L × Whisper-S",
}

# Colors and markers
PAIR_STYLES = {
    "hubert-base × hubert-large": {"color": "#4CAF50", "marker": "o"},
    "wav2vec2-base × wav2vec2-large": {"color": "#2196F3", "marker": "s"},
    "hubert-base × whisper-base": {"color": "#FF9800", "marker": "^"},
    "wav2vec2-large × whisper-small": {"color": "#9C27B0", "marker": "D"},
}

plt.style.use("seaborn-v0_8-paper")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for pair_name, points in pairs.items():
    r = [p["bitrate_mean"] for p in points]
    s = [p["s_local"] for p in points]
    t = [p["tau"] for p in points]
    style = PAIR_STYLES[pair_name]
    label = SHORT_NAMES[pair_name]

    ax1.plot(
        r,
        s,
        color=style["color"],
        marker=style["marker"],
        markersize=5,
        linewidth=1.8,
        label=label,
        alpha=0.9,
    )
    ax2.plot(
        r,
        t,
        color=style["color"],
        marker=style["marker"],
        markersize=5,
        linewidth=1.8,
        label=label,
        alpha=0.9,
    )

# Left panel: R vs s_local
ax1.set_xlabel("Mean Representation Bitrate $\\mathcal{R}$ (bits/dim)", fontsize=10)
ax1.set_ylabel("Null-Calibrated $s_{\\mathrm{local}}$", fontsize=10)
ax1.set_title("(a) Alignment vs. Bitrate", fontsize=11, pad=8)
ax1.axhline(
    y=0.05,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=0.8,
    label="$s_{\\mathrm{local}}$ threshold",
)
ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax1.set_ylim(-0.02, 0.55)
ax1.grid(True, alpha=0.3)

# Right panel: R vs τ
ax2.set_xlabel("Mean Representation Bitrate $\\mathcal{R}$ (bits/dim)", fontsize=10)
ax2.set_ylabel("Transport Linearity $\\tau$", fontsize=10)
ax2.set_title("(b) Transport Linearity vs. Bitrate", fontsize=11, pad=8)
ax2.axhline(
    y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8, label="$\\tau$ threshold"
)
ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

fig.suptitle(
    "Audio Bitrate Sweep: PCA Truncation Modulates Alignment and Transport",
    fontsize=11,
    y=1.02,
)
plt.tight_layout()

output_dir = Path("paper/figures")
fig.savefig(output_dir / "fig20_audio_bitrate_sweep.pdf", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "fig20_audio_bitrate_sweep.png", dpi=300, bbox_inches="tight")
print("Saved fig20_audio_bitrate_sweep.pdf and .png")

# Print summary statistics
print("\nSweep Summary:")
for pair_name, points in pairs.items():
    r_range = (points[0]["bitrate_mean"], points[-1]["bitrate_mean"])
    s_range = (points[0]["s_local"], points[-1]["s_local"])
    t_values = [p["tau"] for p in points]
    t_range = (min(t_values), max(t_values))
    print(f"  {SHORT_NAMES[pair_name]}:")
    print(f"    R: {r_range[0]:.2f} → {r_range[1]:.2f}")
    print(f"    s_local: {s_range[0]:.4f} → {s_range[1]:.4f}")
    print(f"    τ: {t_range[0]:.4f} → {t_range[1]:.4f}")

plt.close()
