# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Figure: P1 Δ-τ threshold verification scatter plot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def plot_p1_threshold() -> None:
    with open("data/experiments/p1_threshold.json") as f:
        data = json.load(f)

    points = data["points"]
    analysis = data["analysis"]
    logistic = analysis["logistic_regression"]

    if not points:
        print("No data points to plot.")
        return

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): Δ vs τ colored by source
    atlas_pts = [p for p in points if p["source"] == "atlas"]
    causal_pts = [p for p in points if p["source"] == "causal"]

    if atlas_pts:
        ax1.scatter(
            [p["delta"] for p in atlas_pts],
            [p["tau"] for p in atlas_pts],
            c="steelblue", s=30, alpha=0.6, edgecolors="w", linewidth=0.3, label="Atlas (native)",
        )
    if causal_pts:
        ax1.scatter(
            [p["delta"] for p in causal_pts],
            [p["tau"] for p in causal_pts],
            c="coral", s=30, alpha=0.6, edgecolors="w", linewidth=0.3, marker="D", label="Causal (compressed)",
        )

    # τ = 0.5 threshold line
    ax1.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.text(0.02, 0.52, "τ = 0.5", transform=ax1.get_yaxis_transform(), fontsize=8, color="gray")

    # Critical Δ from logistic regression
    if "delta_critical" in logistic and logistic["delta_critical"] is not None:
        dc = logistic["delta_critical"]
        ax1.axvline(x=dc, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.text(dc + 0.05, 0.95, f"Δ_crit={dc:.2f}", fontsize=8, color="red",
                 transform=ax1.get_xaxis_transform())

    ax1.set_xlabel("Bottleneck Mismatch $\\Delta$")
    ax1.set_ylabel("Transport Linearity Score $\\tau$")
    ax1.set_title("(a) Δ vs τ: Transport Failure Boundary")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Δ/R_min vs τ (normalized)
    ratios_atlas = [p["delta_over_r_min"] for p in atlas_pts] if atlas_pts else []
    ratios_causal = [p["delta_over_r_min"] for p in causal_pts] if causal_pts else []

    if ratios_atlas:
        ax2.scatter(
            ratios_atlas,
            [p["tau"] for p in atlas_pts],
            c="steelblue", s=30, alpha=0.6, edgecolors="w", linewidth=0.3, label="Atlas",
        )
    if ratios_causal:
        ax2.scatter(
            ratios_causal,
            [p["tau"] for p in causal_pts],
            c="coral", s=30, alpha=0.6, edgecolors="w", linewidth=0.3, marker="D", label="Causal",
        )

    # P1 prediction boundary: Δ/R_min = 2
    ax2.axvline(x=2.0, color="darkgreen", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Shade the prediction region
    xlim = ax2.get_xlim()
    ax2.fill_between([2.0, max(xlim[1], 5.0)], 0, 0.5, color="green", alpha=0.06)
    ax2.text(2.6, 0.15, "P1 prediction:\nΔ/R_min > 2 → τ < 0.5", fontsize=8, color="darkgreen")

    # Add accuracy annotation
    p1_result = analysis["thresholds"].get("delta_gt_2Rmin", {})
    if p1_result:
        acc_text = f"Acc={p1_result.get('accuracy', 0):.1%}, Prec={p1_result.get('precision', 0):.1%}"
        ax2.text(0.98, 0.02, acc_text, transform=ax2.transAxes, fontsize=9,
                 ha="right", va="bottom", bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "gray"})

    ax2.set_xlabel("$\\Delta / \\mathcal{R}_{min}$")
    ax2.set_ylabel("Transport Linearity Score $\\tau$")
    ax2.set_title("(b) Normalized Mismatch: P1 Prediction Test")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("P1 Verification: High Mismatch Predicts Transport Failure", fontsize=13, y=1.01)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig_p1_threshold.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_p1_threshold.png", bbox_inches="tight")
    print("P1 threshold figure saved.")


if __name__ == "__main__":
    plot_p1_threshold()
