"""Figure 7: Practitioner's Decision Tree for embedding migration."""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def _box(ax, x, y, w, h, text, color, fontsize=9, style="round,pad=0.2", text_color="black"):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=style,
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, label="", ha="center"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.text(
            mx + 0.15, my, label, fontsize=8, ha=ha, va="center", color="gray", fontstyle="italic"
        )


def plot_decision_tree() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    # Start node
    _box(ax, 5, 7.5, 4, 0.8, "Encoder A deprecated\nEncoder B available", "#e0e0e0", fontsize=10)

    # Compute τ
    _arrow(ax, 5, 7.1, 5, 6.3)
    _box(
        ax,
        5,
        5.8,
        4.5,
        0.8,
        "Compute τ(A, B)\non 5K sample embeddings",
        "#bbdefb",
        style="round,pad=0.2",
        fontsize=9,
    )

    # τ > 0.9 branch
    _arrow(ax, 3.0, 5.4, 1.5, 4.3, "τ > 0.9", ha="right")
    _box(ax, 1.5, 3.8, 2.5, 1.0, "LINEAR\nADAPTER\n~minutes", "#a5d6a7", fontsize=9)
    ax.text(
        1.5,
        3.0,
        "Fit Procrustes on 5K pairs\nApply to all vectors",
        ha="center",
        fontsize=7,
        color="gray",
    )

    # 0.5 < τ < 0.9 branch
    _arrow(ax, 5, 5.4, 5, 4.3, "0.5 < τ < 0.9")
    _box(ax, 5, 3.8, 2.5, 1.0, "MLP\nADAPTER\n~hours", "#fff9c4", fontsize=9)
    ax.text(
        5,
        3.0,
        "Train 2-layer MLP on 50K\nA/B test 1% traffic",
        ha="center",
        fontsize=7,
        color="gray",
    )

    # τ < 0.5 branch
    _arrow(ax, 7.0, 5.4, 8.5, 4.3, "τ < 0.5", ha="left")
    _box(ax, 8.5, 3.8, 2.2, 0.8, "τ < 0.5", "#ffcdd2", style="round,pad=0.15", fontsize=9)

    # Sub-branches for τ < 0.5
    _arrow(ax, 7.5, 3.4, 6.5, 2.0, "Budget\nallows", ha="right")
    _box(ax, 6.5, 1.3, 2.5, 1.0, "FULL\nRE-EMBEDDING\n$X, Y days", "#ef9a9a", fontsize=9)
    ax.text(
        6.5,
        0.5,
        "Dual-index → gradual\nmigration → retire old",
        ha="center",
        fontsize=7,
        color="gray",
    )

    _arrow(ax, 9.5, 3.4, 10, 2.0, "Budget\ntight", ha="left")
    _box(ax, 10, 1.3, 2.0, 1.0, "SHADOW +\nGRADUAL", "#ffcc80", fontsize=9)
    ax.text(
        10,
        0.5,
        "New docs → B, old → A\nMerge results, backfill",
        ha="center",
        fontsize=7,
        color="gray",
    )

    # Cost legend
    legend_items = [
        mpatches.Patch(facecolor="#a5d6a7", edgecolor="black", label="Cheap (minutes)"),
        mpatches.Patch(facecolor="#fff9c4", edgecolor="black", label="Moderate (hours)"),
        mpatches.Patch(facecolor="#ffcc80", edgecolor="black", label="Expensive (days)"),
        mpatches.Patch(facecolor="#ef9a9a", edgecolor="black", label="Very expensive ($$$)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, title="Migration Cost")

    ax.set_title("Embedding Migration Decision Tree", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig07_decision_tree.pdf")
    plt.savefig(OUTPUT_DIR / "fig07_decision_tree.png")
    print("Figure 7 saved.")


if __name__ == "__main__":
    plot_decision_tree()
