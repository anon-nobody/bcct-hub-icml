# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedVariable=false
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data/atlas.json") as f:
    atlas = json.load(f)


def plot_phase_diagram() -> None:
    # Style setup
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {"font.size": 10, "axes.labelsize": 12, "figure.figsize": (8, 6), "savefig.dpi": 300}
    )

    models = atlas["models"]
    pairwise = atlas["pairwise"]

    regime_colors = {"Convergent": "green", "Local-Only": "orange", "Divergent": "red"}
    points: list[dict[str, float | str]] = []
    for pair_name, metrics in pairwise.items():
        model_a, model_b = pair_name.split(" <-> ")
        bitrate_a = float(models[model_a]["metrics"]["bitrate"])
        bitrate_b = float(models[model_b]["metrics"]["bitrate"])
        regime = str(metrics["regime"])
        if regime not in regime_colors:
            continue

        points.append(
            {
                "pair": pair_name,
                "model_a": model_a,
                "model_b": model_b,
                "family_a": str(models[model_a]["config"]["family"]),
                "family_b": str(models[model_b]["config"]["family"]),
                "avg_bitrate": (bitrate_a + bitrate_b) / 2,
                "s_local": float(metrics["s_local"]),
                "tau": float(metrics["tau"]),
                "regime": regime,
            }
        )

    _, ax = plt.subplots()

    ax.fill_betweenx([0.22, 0.42], 2.1, 9.4, color="orange", alpha=0.08)
    ax.fill_betweenx([0.0, 0.5], 1.45, 2.1, color="red", alpha=0.08)
    ax.fill_betweenx([0.25, 0.5], 8.6, 9.4, color="red", alpha=0.08)

    ax.annotate(
        "Low-bitrate edge",
        xy=(1.8, 0.10),
        xycoords="data",
        xytext=(1.8, 0.20),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=7,
        color="firebrick",
        arrowprops={"arrowstyle": "->", "color": "firebrick", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "firebrick", "alpha": 0.9},
    )
    ax.annotate(
        "Local-Only concentration",
        xy=(5.2, 0.285),
        xycoords="data",
        xytext=(5.2, 0.43),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=8,
        color="darkorange",
        arrowprops={"arrowstyle": "->", "color": "darkorange", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "darkorange", "alpha": 0.9},
    )
    ax.annotate(
        "High-rate divergent edge",
        xy=(8.95, 0.34),
        xycoords="data",
        xytext=(8.95, 0.44),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=7,
        color="firebrick",
        arrowprops={"arrowstyle": "->", "color": "firebrick", "lw": 0.7},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "firebrick", "alpha": 0.9},
    )

    for regime, color in regime_colors.items():
        subset = [point for point in points if point["regime"] == regime]
        x = [float(point["avg_bitrate"]) for point in subset]
        y = [float(point["s_local"]) for point in subset]
        sizes = [35 + 70 * float(point["tau"]) for point in subset]
        ax.scatter(x, y, c=color, s=sizes, alpha=0.72, edgecolors="w", linewidth=0.45)

    regime_counter = Counter(str(point["regime"]) for point in points)

    dinov2_mae = [
        point
        for point in points
        if {str(point["family_a"]), str(point["family_b"])} == {"dinov2", "mae"}
    ]
    if dinov2_mae:
        exemplar = min(
            dinov2_mae,
            key=lambda point: (
                abs(float(point["s_local"]) - 0.3) + abs(float(point["avg_bitrate"]) - 8.8)
            ),
        )
        ax.annotate(
            "DINOv2 <-> MAE exemplar",
            (float(exemplar["avg_bitrate"]), float(exemplar["s_local"])),
            xytext=(2.2, 0.08),
            textcoords="data",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "gray", "alpha": 0.9},
        )

    # Labels and Title
    ax.set_xscale("log")
    ax.set_xlabel("Representation Bitrate $\\mathcal{R}$ (bits/dim)")
    ax.set_ylabel("Calibrated $S_{local}$")
    ax.set_title("BCCT Phase Diagram: Convergence vs. Bitrate")
    ax.set_xlim(1.4, 10.0)
    ax.set_ylim(0.0, 0.7)

    # Custom Legend for Regimes
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Local-Only (n={regime_counter.get('Local-Only', 0)})",
            markerfacecolor="orange",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Convergent (n={regime_counter.get('Convergent', 0)})",
            markerfacecolor="green",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Divergent (n={regime_counter.get('Divergent', 0)})",
            markerfacecolor="red",
            markersize=10,
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", title="Regime")

    # Save
    output_dir = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "fig01_phase_diagram.pdf")
    plt.savefig(output_dir / "fig01_phase_diagram.png")
    print("Figure 1 saved.")


if __name__ == "__main__":
    plot_phase_diagram()
