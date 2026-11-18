"""Figure 6: Transport Asymmetry Analysis — 2x2 panel."""

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


def _short_objective_pair(label: str) -> str:
    replacements = {
        "self-supervised": "SSL",
        "contrastive": "CTR",
        "supervised": "SUP",
    }
    short = label
    for src, dst in replacements.items():
        short = short.replace(src, dst)
    short = short.replace(" x ", "x")
    return short


def plot_asymmetry() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
        }
    )

    models = atlas["models"]
    pairwise = atlas["pairwise"]

    same_tau: list[float] = []
    cross_tau: list[float] = []
    same_delta: list[float] = []
    cross_delta: list[float] = []
    objective_to_tau: defaultdict[str, list[float]] = defaultdict(list)

    for pair_name, metrics in pairwise.items():
        model_a, model_b = pair_name.split(" <-> ")
        family_a = str(models[model_a]["config"]["family"])
        family_b = str(models[model_b]["config"]["family"])
        objective_a = str(models[model_a]["config"]["objective"])
        objective_b = str(models[model_b]["config"]["objective"])
        objective_pair = " x ".join(sorted([objective_a, objective_b]))

        tau = float(metrics["tau"])
        delta = float(metrics["delta"])
        objective_to_tau[objective_pair].append(tau)

        if family_a == family_b:
            same_tau.append(tau)
            same_delta.append(delta)
        else:
            cross_tau.append(tau)
            cross_delta.append(delta)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- (a) tau by family match ---
    ax = axes[0, 0]
    box = ax.boxplot(
        [same_tau, cross_tau], tick_labels=["Same-family", "Cross-family"], patch_artist=True
    )
    for patch, color in zip(box["boxes"], ["#BFE8DA", "#F6C8B4"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Transport Linearity Score $\\tau$")
    ax.set_title("(a) $\\tau$ Distribution by Family Match")

    # --- (b) tau by objective-pair type ---
    ax = axes[0, 1]
    objective_labels = sorted(objective_to_tau)
    objective_labels_short = [_short_objective_pair(label) for label in objective_labels]
    objective_data = [objective_to_tau[label] for label in objective_labels]
    parts = ax.violinplot(objective_data, positions=range(len(objective_data)), showmeans=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#BFDDF2")
        pc.set_alpha(0.9)
    ax.set_xticks(range(len(objective_labels)))
    ax.set_xticklabels(objective_labels_short, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Transport Linearity Score $\\tau$")
    ax.set_title("(b) $\\tau$ by Objective-Pair Type")

    # --- (c) tau vs delta (A->B transport only) ---
    ax = axes[1, 0]
    ax.scatter(same_delta, same_tau, c="#5BAE93", alpha=0.8, s=34, edgecolors="white", label="Same")
    ax.scatter(
        cross_delta,
        cross_tau,
        c="#D98A62",
        alpha=0.8,
        s=34,
        edgecolors="white",
        label="Cross",
    )
    all_delta = np.array(same_delta + cross_delta)
    all_tau = np.array(same_tau + cross_tau)
    slope, intercept = np.polyfit(all_delta, all_tau, 1)
    x_line = np.linspace(all_delta.min(), all_delta.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.4, label=f"slope={slope:.2f}")
    ax.set_xlabel("Bottleneck Mismatch $\\Delta$")
    ax.set_ylabel("Transport Linearity Score $\\tau$")
    ax.set_title("(c) $\\tau$ vs. $\\Delta$ (Atlas, A$\\to$B)")
    ax.legend(fontsize=9)

    # --- (d) objective means ---
    ax = axes[1, 1]
    mean_tau = [float(np.mean(objective_to_tau[label])) for label in objective_labels]
    bars = ax.barh(
        objective_labels_short, mean_tau, color="#E8C7DC", edgecolor="white", height=0.62
    )
    ax.set_xlabel("Mean $\\tau$")
    ax.set_title("(d) Mean Transport by Objective Pair")
    ax.set_xlim(0, 1.06)
    for bar, value in zip(bars, mean_tau, strict=False):
        x_text = min(value + 0.015, 1.02)
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            ha="left",
        )

    fig.suptitle("Transport Asymmetry Proxy (A$\\to$B Only)", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.90, hspace=0.40, wspace=0.30)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig06_asymmetry.pdf")
    plt.savefig(OUTPUT_DIR / "fig06_asymmetry.png")
    print("Figure 6 saved.")


if __name__ == "__main__":
    plot_asymmetry()
