"""Figure 4: R-Alignment and R-Transport curves from Bitrate Sweep (Protocol 1)."""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"

with open("data/atlas.json") as f:
    atlas = json.load(f)


def plot_bitrate_sweep() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    pairwise = atlas["pairwise"]
    regime_colors = {"Convergent": "green", "Local-Only": "orange", "Divergent": "red"}

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for regime, color in regime_colors.items():
        subset = [metrics for metrics in pairwise.values() if metrics["regime"] == regime]
        x = [float(metrics["delta"]) for metrics in subset]
        y = [float(metrics["tau"]) for metrics in subset]
        ax.scatter(
            x, y, c=color, alpha=0.72, edgecolors="white", linewidth=0.45, s=45, label=regime
        )

    all_delta = np.array([float(metrics["delta"]) for metrics in pairwise.values()])
    all_tau = np.array([float(metrics["tau"]) for metrics in pairwise.values()])
    slope, intercept = np.polyfit(all_delta, all_tau, deg=1)
    trend_x = np.linspace(all_delta.min(), all_delta.max(), 200)
    trend_y = slope * trend_x + intercept
    ax.plot(trend_x, trend_y, "k--", linewidth=1.8, label=f"Linear trend (slope={slope:.2f})")

    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.65, label="$\\Delta=2$")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.65, label="$\\tau=0.5$")
    ax.set_xlim(float(all_delta.min()) - 0.05, float(all_delta.max()) + 0.1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Bottleneck Mismatch $\\Delta$")
    ax.set_ylabel("Transport Linearity Score $\\tau$")
    ax.set_title("Bitrate Sweep: Larger Mismatch Lowers Linear Transport")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Protocol 1 (Atlas): Mismatch vs. Transport", fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig04_bitrate_sweep.pdf")
    plt.savefig(OUTPUT_DIR / "fig04_bitrate_sweep.png")
    print("Figure 4 saved.")


if __name__ == "__main__":
    plot_bitrate_sweep()
