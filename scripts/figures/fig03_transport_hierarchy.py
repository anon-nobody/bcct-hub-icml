"""Figure 5 (paper): Transport Map Family — nested complexity hierarchy.

Clean redesign with no overlapping objects:
- Concentric ellipses preserved (each strict superset of the inner one)
- Each layer's label and parameter cost are placed at the top arc of its
  visible band, with distinct y-coordinates so no label sits on top of
  another
- Representative method names are placed in a separate column on the right
  with leader lines connecting to the ellipse outlines
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def plot_transport_hierarchy() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1.0, 4.5)
    ax.axis("off")

    # Outer to inner: each layer is a strict superset of the next.
    layers = [
        {"w": 10.0, "h": 3.8, "label": "Diffusion Bridge", "params": r"$O(\infty)$",
         "method": "Score Bridge", "color": "#08306b", "text_color": "white"},
        {"w": 8.5,  "h": 3.2, "label": "MLP",             "params": r"$O(4d^{2})$",
         "method": "FuLA",         "color": "#2171b5", "text_color": "white"},
        {"w": 7.0,  "h": 2.6, "label": "Affine",          "params": r"$O(d^{2}+d)$",
         "method": "Ridge Regression", "color": "#4292c6", "text_color": "white"},
        {"w": 5.5,  "h": 2.0, "label": "Orthogonal",      "params": r"$O(d^{2}/2)$",
         "method": "Procrustes",   "color": "#6baed6", "text_color": "black"},
        {"w": 4.0,  "h": 1.4, "label": "Permutation",     "params": r"$O(d)$",
         "method": "Re-Basin",     "color": "#9ecae1", "text_color": "black"},
        {"w": 2.5,  "h": 0.85, "label": "Identity",        "params": r"$O(1)$",
         "method": "Relative Repr.", "color": "#c6dbef", "text_color": "black"},
    ]

    cx, cy = 5.0, 1.5

    # Draw ellipses (largest first so smaller ones render on top).
    for layer in layers:
        ell = Ellipse(
            (cx, cy),
            layer["w"],
            layer["h"],
            facecolor=layer["color"],
            edgecolor="white",
            linewidth=1.8,
            alpha=0.95,
        )
        ax.add_patch(ell)

    # Place the layer label inside the top arc of *its own* visible band.
    # Visible band of layer i is the strip between the top of ellipse i and
    # the top of ellipse i+1 (or the centre if i is the innermost layer).
    n = len(layers)
    label_ys: list[float] = []
    for i, layer in enumerate(layers):
        top_i = cy + layer["h"] / 2
        if i + 1 < n:
            top_inner = cy + layers[i + 1]["h"] / 2
            label_y = (top_i + top_inner) / 2
        else:
            # innermost: centre vertically
            label_y = cy
        label_ys.append(label_y)
        ax.text(
            cx,
            label_y,
            f"{layer['label']}  [{layer['params']}]",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=layer["text_color"],
        )

    # Methods column on the right, each row aligned with that layer's label
    # band, connected by a dotted leader line.
    method_x = 13.0
    for i, layer in enumerate(layers):
        y = label_ys[i]
        # Leader line from the right edge of *this layer's ellipse* at the
        # label's y-height, out to the methods column.
        # Right-edge x of ellipse at height y:
        h2 = (layer["h"] / 2) ** 2
        dy = (y - cy)
        if h2 > dy ** 2:
            x_edge = cx + (layer["w"] / 2) * (1 - dy ** 2 / h2) ** 0.5
        else:
            x_edge = cx + layer["w"] / 2
        ax.plot(
            [x_edge + 0.1, method_x - 0.2],
            [y, y],
            ls=":",
            color="#777777",
            lw=0.9,
        )
        ax.text(
            method_x,
            y,
            layer["method"],
            ha="left",
            va="center",
            fontsize=9,
            fontstyle="italic",
            color="#333333",
        )

    # Header for the methods column.
    ax.text(
        method_x,
        cy + layers[0]["h"] / 2 + 0.25,
        "Representative method",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )

    # Footer: explicit subset chain.
    ax.text(
        cx,
        -0.55,
        r"Identity $\subset$ Permutation $\subset$ Orthogonal "
        r"$\subset$ Affine $\subset$ MLP $\subset$ Diffusion Bridge",
        ha="center",
        va="center",
        fontsize=10,
        fontstyle="italic",
        color="#333333",
    )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig03_transport_hierarchy.pdf")
    plt.savefig(OUTPUT_DIR / "fig03_transport_hierarchy.png")
    print("Figure 3 saved.")


if __name__ == "__main__":
    plot_transport_hierarchy()
