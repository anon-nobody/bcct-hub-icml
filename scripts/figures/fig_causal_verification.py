# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Figure: OP1 Causal Verification — R vs S_local and R vs τ curves with sigmoid fits."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures"


def plot_causal_verification() -> None:
    # Use v2 data (15 compression levels + permutation null)
    import os
    v2 = "data/experiments/causal_bitrate_verification_v2.json"
    v1 = "data/experiments/causal_bitrate_verification.json"
    path = v2 if os.path.exists(v2) else v1
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    if not results:
        print("No results to plot.")
        return

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, pair_result in enumerate(results):
        pair_name = pair_result["pair"]
        points = pair_result["points"]
        # Short label
        label = pair_name.replace(" <-> ", " ↔ ").replace("_patch16_224", "").replace("_patch4_window7_224", "")

        rs = [p["bitrate_a"] for p in points]
        s_locals = [p["s_local"] for p in points]
        taus = [p["tau"] for p in points]
        color = colors[i]

        # Panel (a): R vs S_local
        ax1.plot(rs, s_locals, "o-", color=color, label=label, markersize=5, linewidth=1.5, alpha=0.8)

        # Overlay sigmoid fit if available
        fit = pair_result["sigmoid_fit_alignment"]
        if fit.get("r2_fit") is not None and fit["r2_fit"] > 0.3:
            r_smooth = np.linspace(min(rs) * 0.9, max(rs) * 1.1, 100)
            s_pred = fit["s_max"] / (1.0 + np.exp(-fit["kappa"] * (r_smooth - fit["r0"])))
            ax1.plot(r_smooth, s_pred, "--", color=color, alpha=0.4, linewidth=1.0)

        # Panel (b): R vs τ
        ax2.plot(rs, taus, "s-", color=color, label=label, markersize=5, linewidth=1.5, alpha=0.8)

    # Panel (a) formatting
    ax1.set_xlabel("Representation Bitrate $\\mathcal{R}$ (bits/dim)")
    ax1.set_ylabel("Calibrated $S_{local}$")
    ax1.set_title("(a) Alignment under bitrate intervention")
    ax1.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel (b) formatting
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="τ = 0.5 threshold")
    ax2.set_xlabel("Representation Bitrate $\\mathcal{R}$ (bits/dim)")
    ax2.set_ylabel("Transport Linearity Score $\\tau$")
    ax2.set_title("(b) Transport under bitrate intervention")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("OP1: Causal Verification — Bitrate Intervention Effects", fontsize=13, y=1.01)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "fig_op1_causal_verification.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "fig_op1_causal_verification.png", bbox_inches="tight")
    print("OP1 causal verification figure saved.")


if __name__ == "__main__":
    plot_causal_verification()
