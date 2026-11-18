"""Figure 4 (paper): Levels of Representational Equivalence.

Clean redesign with no overlapping objects:
- 4 horizontal level boxes stacked top-to-bottom (Level 3 strongest at top)
- Each box has 3 inline columns: Header / Metric / Example
- A single vertical 'Stronger' arrow on the right spans the full height
- No per-row arrows or duplicated implication symbols
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def plot_equivalence_ladder() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Strongest at top → weakest at bottom (matches Level 3 ⇒ Level 0)
    levels = [
        {
            "label": "Level 3: Semantic",
            "metric": r"Concept probes, SAE atoms",
            "example": "Individual latent directions correspond",
            "color": "#33a02c",
            "y": 7.0,
        },
        {
            "label": "Level 2: Functional",
            "metric": r"$\tau$ (stitching quality)",
            "example": "Components are interchangeable",
            "color": "#b2df8a",
            "y": 5.0,
        },
        {
            "label": "Level 1: Geometric",
            "metric": r"Procrustes distance",
            "example": "Distances and angles preserved",
            "color": "#1f78b4",
            "y": 3.0,
        },
        {
            "label": "Level 0: Statistical",
            "metric": r"CKA, kernel overlap",
            "example": "Shared variance structure",
            "color": "#a6cee3",
            "y": 1.0,
        },
    ]

    box_x = 0.6
    box_w = 9.6
    box_h = 1.5

    for lvl in levels:
        y0 = lvl["y"]
        rect = mpatches.FancyBboxPatch(
            (box_x, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.10",
            facecolor=lvl["color"],
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        ax.add_patch(rect)

        # Three-column layout inside the box. Anchor each column at a fixed x
        # so labels never collide.
        ax.text(
            box_x + 0.30,
            y0 + box_h * 0.66,
            lvl["label"],
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.text(
            box_x + 0.30,
            y0 + box_h * 0.26,
            "Metric: " + lvl["metric"],
            fontsize=11,
            va="center",
            ha="left",
            style="italic",
        )
        ax.text(
            box_x + 4.6,
            y0 + box_h * 0.5,
            lvl["example"],
            fontsize=11,
            va="center",
            ha="left",
            color="#333333",
        )

    # Right-side single 'Stronger' arrow spanning the full ladder height,
    # placed outside the box column so it never crosses any box.
    arrow_x = 11.2
    ax.annotate(
        "",
        xy=(arrow_x, levels[0]["y"] + box_h - 0.05),
        xytext=(arrow_x, levels[-1]["y"] + 0.05),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#444444", mutation_scale=18),
    )
    ax.text(
        arrow_x + 0.30,
        (levels[0]["y"] + box_h + levels[-1]["y"]) / 2,
        "stronger\nequivalence",
        fontsize=11,
        ha="left",
        va="center",
        rotation=90,
        color="#444444",
    )

    # Footer: implication chain in plain text (replaces inter-box arrows)
    ax.text(
        6.0,
        0.25,
        r"Level 3 $\Rightarrow$ Level 2 $\Rightarrow$ Level 1 $\Rightarrow$ Level 0",
        ha="center",
        va="center",
        fontsize=12,
        fontstyle="italic",
        color="#333333",
    )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig02_equivalence_ladder.pdf")
    plt.savefig(OUTPUT_DIR / "fig02_equivalence_ladder.png")
    print("Figure 2 saved.")


if __name__ == "__main__":
    plot_equivalence_ladder()
