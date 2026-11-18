from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS_PATH = ROOT / "data" / "atlas.json"
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig11_regime_distribution.pdf"

COLORS = {
    "Convergent": "#2ca02c",
    "Local-Only": "#ff7f0e",
    "Mixed": "#7f7f7f",
    "Divergent": "#d62728",
    "Collapsed": "#1f77b4",
}


def _load_pairwise_regime_counts() -> dict[str, int]:
    with ATLAS_PATH.open() as f:
        atlas = json.load(f)

    pairwise = atlas["pairwise"]
    counts = Counter(str(metrics["regime"]) for metrics in pairwise.values())
    return {
        "Convergent": counts.get("Convergent", 0),
        "Local-Only": counts.get("Local-Only", 0),
        "Divergent": counts.get("Divergent", 0),
        "Collapsed": counts.get("Collapsed", 0),
    }


def _add_bar_labels(ax: Axes, bars: list[Rectangle], total: int) -> None:
    for bar in bars:
        height = int(bar.get_height())
        pct = 100.0 * height / total if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{height}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300})

    literature_total = 49
    literature_counts = {
        "Convergent": int(0.59 * literature_total),
        "Local-Only": int(0.31 * literature_total),
        "Mixed": 4,
        "Divergent": 1,
    }

    atlas_counts = _load_pairwise_regime_counts()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    left_labels = ["Convergent", "Local-Only", "Mixed", "Divergent"]
    left_values = [literature_counts[label] for label in left_labels]
    left_colors = [COLORS[label] for label in left_labels]
    left_bars = axes[0].bar(left_labels, left_values, color=left_colors, edgecolor="white")
    _add_bar_labels(axes[0], list(left_bars), literature_total)
    axes[0].set_title("Literature Meta-Analysis (n=49)")
    axes[0].set_ylabel("Number of papers")
    axes[0].set_ylim(0, max(left_values) + 8)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    right_labels = ["Convergent", "Local-Only", "Divergent", "Collapsed"]
    right_values = [atlas_counts[label] for label in right_labels]
    right_colors = [COLORS[label] for label in right_labels]
    right_total = sum(right_values)
    right_bars = axes[1].bar(right_labels, right_values, color=right_colors, edgecolor="white")
    _add_bar_labels(axes[1], list(right_bars), right_total)
    axes[1].set_title(f"Vision Atlas (n={right_total})")
    axes[1].set_ylabel("Number of model pairs")
    axes[1].set_ylim(0, max(right_values) + 16)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Regime Distribution: Literature vs Vision Atlas", fontsize=12, fontweight="bold")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
