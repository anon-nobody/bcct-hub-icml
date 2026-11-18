"""Figure 5: Meta-Analysis scatter — papers re-plotted in BCCT coordinates."""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"

with open("data/atlas.json") as f:
    atlas = json.load(f)


def _objective_pair_key(obj_a: str, obj_b: str) -> tuple[str, str]:
    if obj_a <= obj_b:
        return (obj_a, obj_b)
    return (obj_b, obj_a)


def plot_meta_analysis() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    models = atlas["models"]
    pairwise = atlas["pairwise"]

    same_family: list[dict[str, float]] = []
    cross_family: list[dict[str, float]] = []
    objective_stats: defaultdict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for pair_name, metrics in pairwise.items():
        model_a, model_b = pair_name.split(" <-> ")
        family_a = str(models[model_a]["config"]["family"])
        family_b = str(models[model_b]["config"]["family"])
        objective_a = str(models[model_a]["config"]["objective"])
        objective_b = str(models[model_b]["config"]["objective"])
        row = {"s_local": float(metrics["s_local"]), "tau": float(metrics["tau"])}

        if family_a == family_b:
            same_family.append(row)
        else:
            cross_family.append(row)

        objective_stats[_objective_pair_key(objective_a, objective_b)].append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5.5))

    same_avg_s = float(np.mean([item["s_local"] for item in same_family]))
    same_avg_tau = float(np.mean([item["tau"] for item in same_family]))
    cross_avg_s = float(np.mean([item["s_local"] for item in cross_family]))
    cross_avg_tau = float(np.mean([item["tau"] for item in cross_family]))

    x = np.arange(2)
    width = 0.36
    ax1.bar(
        x - width / 2,
        [same_avg_s, cross_avg_s],
        width,
        color="#009E73",
        label="avg $s_{local}$",
        edgecolor="white",
    )
    ax1.bar(
        x + width / 2,
        [same_avg_tau, cross_avg_tau],
        width,
        color="#0072B2",
        label="avg $\\tau$",
        edgecolor="white",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Same-family", "Cross-family"], rotation=12, ha="right")
    ax1.set_ylim(0, 0.5)
    ax1.set_title("(A) Family Effect")
    ax1.set_ylabel("Average score")
    ax1.legend(fontsize=10, loc="upper right")

    ax1.text(0 - width / 2, same_avg_s + 0.015, f"{same_avg_s:.3f}", fontsize=9, ha="center")
    ax1.text(0 + width / 2, same_avg_tau + 0.015, f"{same_avg_tau:.3f}", fontsize=9, ha="center")
    ax1.text(1 - width / 2, cross_avg_s + 0.015, f"{cross_avg_s:.3f}", fontsize=9, ha="center")
    ax1.text(1 + width / 2, cross_avg_tau + 0.015, f"{cross_avg_tau:.3f}", fontsize=9, ha="center")

    objectives = ["contrastive", "self-supervised", "supervised"]
    matrix = np.zeros((len(objectives), len(objectives)), dtype=float)
    count_matrix = np.zeros((len(objectives), len(objectives)), dtype=int)
    for i, obj_a in enumerate(objectives):
        for j, obj_b in enumerate(objectives):
            key = _objective_pair_key(obj_a, obj_b)
            rows = objective_stats.get(key, [])
            if rows:
                matrix[i, j] = float(np.mean([row["s_local"] for row in rows]))
                count_matrix[i, j] = len(rows)

    im = ax2.imshow(matrix, cmap="YlGnBu", vmin=0.1, vmax=0.35)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            ax2.text(
                j,
                i,
                f"{matrix[i, j]:.3f}\n(n={count_matrix[i, j]})",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    ax2.set_xticks(np.arange(len(objectives)))
    ax2.set_yticks(np.arange(len(objectives)))
    ax2.set_xticklabels(objectives, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(objectives, fontsize=9)
    ax2.set_title("(B) Objective-Pair Heatmap ($s_{local}$)")
    ax2.set_xlabel("Objective B")
    ax2.set_ylabel("Objective A")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Meta-Analysis on 190 Atlas Pairs: Family and Objective Structure",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig05_meta_analysis.pdf")
    plt.savefig(OUTPUT_DIR / "fig05_meta_analysis.png")
    print("Figure 5 saved.")


if __name__ == "__main__":
    plot_meta_analysis()
