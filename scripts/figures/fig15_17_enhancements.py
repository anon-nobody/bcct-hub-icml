from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "paper" / "figures"
ATLAS_PATH = ROOT / "data" / "atlas.json"


def _load_atlas() -> dict[str, object]:
    with open(ATLAS_PATH) as f:
        return cast("dict[str, object]", json.load(f))


def _infer_equivalence_level(s_local: float, tau: float) -> str:
    if s_local >= 0.2 and tau >= 0.8:
        return "E2: Functional"
    if s_local >= 0.1:
        return "E1: Geometric"
    return "E0: Statistical"


def _infer_asymmetry_source(
    objective_a: str,
    objective_b: str,
    family_a: str,
    family_b: str,
    delta: float,
) -> str:
    if objective_a != objective_b:
        return "A_obj"
    if family_a != family_b:
        return "A_arch"
    if delta > 1.0:
        return "A_cap"
    return "A_cap"


def plot_taxonomy_sankey(atlas: dict[str, object]) -> None:
    models = cast("dict[str, dict[str, object]]", atlas["models"])
    pairs = cast("dict[str, dict[str, object]]", atlas["pairwise"])

    flow_e_to_a: Counter[tuple[str, str]] = Counter()
    flow_a_to_r: Counter[tuple[str, str]] = Counter()

    for pair_name, m in pairs.items():
        a, b = pair_name.split(" <-> ")
        a_conf = cast("dict[str, object]", models[a]["config"])
        b_conf = cast("dict[str, object]", models[b]["config"])
        eq = _infer_equivalence_level(float(m["s_local"]), float(m["tau"]))
        asym = _infer_asymmetry_source(
            str(a_conf["objective"]),
            str(b_conf["objective"]),
            str(a_conf["family"]),
            str(b_conf["family"]),
            float(m["delta"]),
        )
        reg = str(m["regime"])
        flow_e_to_a[(eq, asym)] += 1
        flow_a_to_r[(asym, reg)] += 1

    left_nodes = ["E0: Statistical", "E1: Geometric", "E2: Functional"]
    mid_nodes = ["A_obj", "A_arch", "A_cap"]
    right_nodes = ["Divergent", "Local-Only", "Convergent"]

    left_colors = {
        "E0: Statistical": "#a6cee3",
        "E1: Geometric": "#1f78b4",
        "E2: Functional": "#08306b",
    }
    mid_colors = {"A_obj": "#fb9a99", "A_arch": "#fdbf6f", "A_cap": "#cab2d6"}
    right_colors = {
        "Divergent": "#e31a1c",
        "Local-Only": "#ff7f00",
        "Convergent": "#33a02c",
    }

    total = len(pairs)

    def _stack_positions(
        nodes: list[str], totals: dict[str, int]
    ) -> dict[str, tuple[float, float]]:
        y0 = 0.0
        out: dict[str, tuple[float, float]] = {}
        for node in nodes:
            h = totals.get(node, 0) / total
            out[node] = (y0, y0 + h)
            y0 += h
        return out

    left_totals = defaultdict(int)
    for (left_node, _), c in flow_e_to_a.items():
        left_totals[left_node] += c
    mid_in_totals = defaultdict(int)
    for (_, m), c in flow_e_to_a.items():
        mid_in_totals[m] += c
    right_totals = defaultdict(int)
    for (_, r), c in flow_a_to_r.items():
        right_totals[r] += c

    left_pos = _stack_positions(left_nodes, left_totals)
    mid_pos = _stack_positions(mid_nodes, mid_in_totals)
    right_pos = _stack_positions(right_nodes, right_totals)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x_left, x_mid, x_right = 0.05, 0.5, 0.95

    def _draw_node(x: float, node: str, pos: tuple[float, float], color: str, count: int) -> None:
        y0, y1 = pos
        ax.add_patch(Rectangle((x - 0.02, y0), 0.04, y1 - y0, color=color, alpha=0.9))
        pct = 100.0 * count / total if total > 0 else 0.0
        ax.text(
            x,
            (y0 + y1) / 2,
            f"{node}\n{count} ({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=8.5,
            color="black",
        )

    for n in left_nodes:
        _draw_node(x_left, n, left_pos[n], left_colors[n], left_totals.get(n, 0))
    for n in mid_nodes:
        _draw_node(x_mid, n, mid_pos[n], mid_colors[n], mid_in_totals.get(n, 0))
    for n in right_nodes:
        _draw_node(x_right, n, right_pos[n], right_colors[n], right_totals.get(n, 0))

    left_cursor = {n: left_pos[n][0] for n in left_nodes}
    mid_in_cursor = {n: mid_pos[n][0] for n in mid_nodes}

    for left_node in left_nodes:
        for m in mid_nodes:
            c = flow_e_to_a.get((left_node, m), 0)
            if c == 0:
                continue
            h = c / total
            y0_l, y1_l = left_cursor[left_node], left_cursor[left_node] + h
            y0_m, y1_m = mid_in_cursor[m], mid_in_cursor[m] + h
            left_cursor[left_node] += h
            mid_in_cursor[m] += h
            y_l = (y0_l + y1_l) / 2
            y_m = (y0_m + y1_m) / 2
            rad = 0.18 if y_m >= y_l else -0.18
            ax.annotate(
                "",
                xy=(x_mid - 0.02, y_m),
                xytext=(x_left + 0.02, y_l),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": left_colors[left_node],
                    "alpha": 0.55,
                    "linewidth": max(1.0, 24 * h),
                    "connectionstyle": f"arc3,rad={rad}",
                },
            )
            if c >= 10:
                ax.text(
                    0.28,
                    (y_l + y_m) / 2,
                    f"{c}",
                    fontsize=8,
                    color="#37474f",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8},
                )

    mid_out_cursor = {n: mid_pos[n][0] for n in mid_nodes}
    right_cursor = {n: right_pos[n][0] for n in right_nodes}

    for m in mid_nodes:
        for r in right_nodes:
            c = flow_a_to_r.get((m, r), 0)
            if c == 0:
                continue
            h = c / total
            y0_m, y1_m = mid_out_cursor[m], mid_out_cursor[m] + h
            y0_r, y1_r = right_cursor[r], right_cursor[r] + h
            mid_out_cursor[m] += h
            right_cursor[r] += h
            y_m = (y0_m + y1_m) / 2
            y_r = (y0_r + y1_r) / 2
            rad = 0.18 if y_r >= y_m else -0.18
            ax.annotate(
                "",
                xy=(x_right - 0.02, y_r),
                xytext=(x_mid + 0.02, y_m),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": mid_colors[m],
                    "alpha": 0.55,
                    "linewidth": max(1.0, 24 * h),
                    "connectionstyle": f"arc3,rad={rad}",
                },
            )
            if c >= 10:
                ax.text(
                    0.72,
                    (y_m + y_r) / 2,
                    f"{c}",
                    fontsize=8,
                    color="#37474f",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8},
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Taxonomy-to-Regime Flow on 190 Vision Pairs", fontsize=12, fontweight="bold")
    ax.text(0.05, 1.02, "Equivalence", ha="center", va="bottom", fontsize=10)
    ax.text(0.50, 1.02, "Dominant Asymmetry", ha="center", va="bottom", fontsize=10)
    ax.text(0.95, 1.02, "Regime", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.50,
        -0.06,
        "Arrow direction shows causal flow (left→middle→right);\n"
        "thickness and labels indicate pair counts.",
        ha="center",
        va="top",
        fontsize=9,
        color="#455a64",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig15_taxonomy_sankey.pdf")
    fig.savefig(FIG_DIR / "fig15_taxonomy_sankey.png")
    plt.close(fig)


def plot_threshold_heatmap(atlas: dict[str, object]) -> None:
    pairs = cast("dict[str, dict[str, object]]", atlas["pairwise"])
    models = cast("dict[str, dict[str, object]]", atlas["models"])

    s_vals = [0.03, 0.05, 0.07]
    t_vals = [0.4, 0.5, 0.6]

    conv = np.zeros((len(s_vals), len(t_vals)), dtype=int)
    local = np.zeros_like(conv)
    div = np.zeros_like(conv)

    for i, s_thr in enumerate(s_vals):
        for j, t_thr in enumerate(t_vals):
            c = local_count = d = 0
            for pair_name, m in pairs.items():
                a, b = pair_name.split(" <-> ")
                a_metrics = cast("dict[str, object]", models[a]["metrics"])
                b_metrics = cast("dict[str, object]", models[b]["metrics"])
                s = float(m["s_local"])
                t = float(m["tau"])
                r_a = float(a_metrics["bitrate"])
                r_b = float(b_metrics["bitrate"])
                if s > s_thr and t > t_thr:
                    c += 1
                elif s > s_thr:
                    local_count += 1
                elif min(r_a, r_b) < 5.0:
                    pass
                else:
                    d += 1
            conv[i, j], local[i, j], div[i, j] = c, local_count, d

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    mats = [
        (conv, "Convergent count", "YlGn"),
        (local, "Local-Only count", "YlOrBr"),
        (div, "Divergent count", "PuRd"),
    ]

    for ax, (mat, title, cmap) in zip(axes, mats, strict=True):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(1, int(np.max(mat))))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                color = "white" if mat[i, j] >= max(1, int(np.max(mat)) * 0.55) else "black"
                ax.text(
                    j,
                    i,
                    str(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xticks(np.arange(len(t_vals)))
        ax.set_xticklabels([f"{x:.1f}" for x in t_vals], fontsize=10)
        ax.set_yticks(np.arange(len(s_vals)))
        ax.set_yticklabels([f"{x:.2f}" for x in s_vals], fontsize=10)
        ax.set_xlabel(r"$\tau$ threshold", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel(r"$S_{local}$ threshold")
    fig.suptitle(
        "Threshold Robustness of Regime Counts (190 Pairs)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    fig.savefig(FIG_DIR / "fig16_threshold_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig16_threshold_heatmap.png")
    plt.close(fig)


def plot_tau_bimodality(atlas: dict[str, object]) -> None:
    pairs = cast("dict[str, dict[str, object]]", atlas["pairwise"])
    taus = np.array([float(v["tau"]) for v in pairs.values()])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    bins = np.linspace(0, 1, 21)
    ax1.hist(taus, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.9)
    ax1.set_xlabel(r"$\tau$ (Transport Linearity)")
    ax1.set_ylabel("Number of pairs")
    ax1.set_title("Histogram")
    ax1.grid(alpha=0.2, axis="y")

    n_zero = int(np.sum(np.isclose(taus, 0.0)))
    n_one = int(np.sum(np.isclose(taus, 1.0)))
    n_mid = int(len(taus) - n_zero - n_one)
    vals = np.array([n_zero, n_mid, n_one], dtype=float)
    labels = [r"$\tau=0$", r"$0<\tau<1$", r"$\tau=1$"]
    colors = ["#d62728", "#ffbf00", "#2ca02c"]
    ax2.bar(labels, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax2.text(
            i,
            v + 2,
            f"{int(v)}\n({100 * v / len(taus):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Number of pairs")
    ax2.set_title("Mass at extremes")
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle(
        "Bimodality of Transport Linearity in Vision Atlas", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    fig.savefig(FIG_DIR / "fig17_tau_bimodality.pdf")
    fig.savefig(FIG_DIR / "fig17_tau_bimodality.png")
    plt.close(fig)


def main() -> None:
    atlas = _load_atlas()
    plot_taxonomy_sankey(atlas)
    plot_threshold_heatmap(atlas)
    plot_tau_bimodality(atlas)
    print("Saved fig15, fig16, fig17 to paper/figures")


if __name__ == "__main__":
    main()
