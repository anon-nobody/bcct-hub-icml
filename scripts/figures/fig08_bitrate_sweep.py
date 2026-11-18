"""Figure 8: Track B — Bitrate Sweep Experiment Results.

Shows how alignment (s_local) and transport (tau) degrade under
progressive compression, broken out by pair type.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Pair categories for color coding
CONVERGENT_PAIRS = {"clip_vitb16 <-> clip_vitl14", "resnet50 <-> resnet101"}
LOCAL_ONLY_PAIRS = {
    "dinov2_vitb14 <-> dinov2_vitl14",
    "dinov2_vitl14 <-> beit_large_patch16_224",
    "vit_base_patch16_224 <-> resnet50",
    "dinov2_vitb14 <-> clip_vitb16",
}
DIVERGENT_PAIRS = {
    "vit_base_patch16_224.mae <-> vit_base_patch16_224",
    "vit_base_patch16_224.mae <-> resnet101",
}

PAIR_COLORS = {
    # Convergent: greens
    "clip_vitb16 <-> clip_vitl14": "#2ca02c",
    "resnet50 <-> resnet101": "#98df8a",
    # Local-Only: blues
    "dinov2_vitb14 <-> dinov2_vitl14": "#1f77b4",
    "dinov2_vitl14 <-> beit_large_patch16_224": "#aec7e8",
    "vit_base_patch16_224 <-> resnet50": "#6baed6",
    "dinov2_vitb14 <-> clip_vitb16": "#3182bd",
    # Divergent: reds
    "vit_base_patch16_224.mae <-> vit_base_patch16_224": "#d62728",
    "vit_base_patch16_224.mae <-> resnet101": "#ff9896",
}

SHORT_NAMES = {
    "clip_vitb16 <-> clip_vitl14": "CLIP-B/16↔L/14",
    "resnet50 <-> resnet101": "RN50↔RN101",
    "dinov2_vitb14 <-> dinov2_vitl14": "DINOv2-B↔L",
    "dinov2_vitl14 <-> beit_large_patch16_224": "DINOv2-L↔BEiT-L",
    "vit_base_patch16_224 <-> resnet50": "ViT-B↔RN50",
    "dinov2_vitb14 <-> clip_vitb16": "DINOv2-B↔CLIP-B",
    "vit_base_patch16_224.mae <-> vit_base_patch16_224": "MAE↔ViT-B(sup)",
    "vit_base_patch16_224.mae <-> resnet101": "MAE↔RN101",
}


def load_results() -> list[dict]:
    """Load merged experiment results."""
    full_path = Path("data/experiments/bitrate_experiment_full.json")
    if full_path.exists():
        with open(full_path) as f:
            return json.load(f)["results"]

    # Try loading both parts
    results = []
    for p in [
        "data/experiments/bitrate_experiment.json",
        "data/experiments_part2/bitrate_experiment.json",
    ]:
        if Path(p).exists():
            with open(p) as f:
                results.extend(json.load(f)["results"])
    return results


def main() -> None:
    results = load_results()
    if not results:
        print("No experiment results found. Run scripts/run_bitrate_experiment.py first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel A: S_local vs bitrate under compression
    ax = axes[0, 0]
    for r in results:
        pair = r["pair"]
        sweep = r["bitrate_sweep"]["points"]
        bitrates = [p["bitrate"] for p in sweep]
        s_locals = [p["s_local"] for p in sweep]
        ax.plot(
            bitrates,
            s_locals,
            "o-",
            color=PAIR_COLORS.get(pair, "gray"),
            label=SHORT_NAMES.get(pair, pair),
            markersize=4,
            linewidth=1.5,
        )
    ax.set_xlabel("Representation Bitrate (bits/dim)")
    ax.set_ylabel("$S_{\\mathrm{local}}$ (calibrated overlap)")
    ax.set_title("(a) Alignment under compression")
    # Place legend in upper-left corner (low-bitrate / low-S_local region is sparse)
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85, borderaxespad=0.4)
    ax.grid(True, alpha=0.3)

    # Panel B: Tau vs bitrate under compression
    ax = axes[0, 1]
    for r in results:
        pair = r["pair"]
        sweep = r["bitrate_sweep"]["points"]
        bitrates = [p["bitrate"] for p in sweep]
        taus = [p["tau"] for p in sweep]
        ax.plot(
            bitrates,
            taus,
            "s-",
            color=PAIR_COLORS.get(pair, "gray"),
            label=SHORT_NAMES.get(pair, pair),
            markersize=4,
            linewidth=1.5,
        )
    ax.set_xlabel("Representation Bitrate (bits/dim)")
    ax.set_ylabel("$\\tau$ (Transport Linearity)")
    ax.set_title("(b) Transport under compression")
    ax.axhline(y=0.5, color="k", linestyle="--", alpha=0.3, label="$\\tau=0.5$ threshold")
    # Place legend in upper-left corner; τ rises toward upper-right where data is dense
    ax.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.85, borderaxespad=0.4)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.title.set_size(13)
    ax.grid(True, alpha=0.3)

    # Panel C: Transport Stress Test — TAI vs Delta
    ax = axes[1, 0]
    for r in results:
        pair = r["pair"]
        stress = r["transport_stress_test"]["points"]
        deltas = [p["delta"] for p in stress]
        tais = [p["tai"] for p in stress]
        ax.plot(
            deltas,
            tais,
            "^-",
            color=PAIR_COLORS.get(pair, "gray"),
            label=SHORT_NAMES.get(pair, pair),
            markersize=5,
            linewidth=1.5,
        )
    ax.set_xlabel("$\\Delta$ (Bottleneck Mismatch)")
    ax.set_ylabel("TAI (Transport Asymmetry)")
    ax.set_title("(c) Asymmetry vs mismatch")
    # Place legend inside the lower-right corner (TAI-vs-Δ data clusters in upper region)
    ax.legend(fontsize=7, ncol=2, loc="lower right", framealpha=0.85, borderaxespad=0.4)
    ax.grid(True, alpha=0.3)

    # Panel D: Summary bar chart — tau at full vs compressed
    ax = axes[1, 1]
    pair_names = []
    tau_full = []
    tau_compressed = []
    for r in results:
        pair = r["pair"]
        sweep = r["bitrate_sweep"]["points"]
        pair_names.append(SHORT_NAMES.get(pair, pair))
        tau_full.append(sweep[0]["tau"])
        # Get tau at 12.5% rank
        tau_c = next((p["tau"] for p in sweep if p["level"] == 0.125), sweep[-1]["tau"])
        tau_compressed.append(tau_c)

    x = np.arange(len(pair_names))
    width = 0.35
    ax.bar(x - width / 2, tau_full, width, label="Full bitrate", color="#1f77b4", alpha=0.8)
    ax.bar(x + width / 2, tau_compressed, width, label="12.5% rank", color="#ff7f0e", alpha=0.8)
    ax.set_ylabel("$\\tau$ (Transport Linearity)")
    ax.set_title("(d) Transport robustness to compression")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="k", linestyle="--", alpha=0.3)

    # Title removed; the figure caption in the paper provides full context.
    plt.tight_layout()

    # Save
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig08_bitrate_sweep.pdf")
    fig.savefig(out_dir / "fig08_bitrate_sweep.png")
    print(f"Saved to {out_dir}/fig08_bitrate_sweep.{{pdf,png}}")
    plt.close()


if __name__ == "__main__":
    main()
