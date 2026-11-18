from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ATLAS_PATH = ROOT / "data" / "atlas.json"
OUTPUT_PATH = ROOT / "paper" / "figures" / "fig16_threshold_heatmap.pdf"

THRESHOLD_SETTINGS = [
    ("Baseline\n$s_{local}^*=0.05, \\tau^*=0.5$", 0.05, 0.5),
    ("Loose $s_{local}^*$\n$s_{local}^*=0.03, \\tau^*=0.5$", 0.03, 0.5),
    ("Tight $s_{local}^*$\n$s_{local}^*=0.07, \\tau^*=0.5$", 0.07, 0.5),
]
REGIME_ORDER = ["Convergent", "Local-Only", "Divergent", "Collapsed"]


def _load_pairwise_metrics() -> list[dict[str, float]]:
    with ATLAS_PATH.open() as f:
        atlas = json.load(f)

    pairwise = atlas["pairwise"]
    model_bitrates = {
        name: float(data["metrics"]["bitrate"]) for name, data in atlas["models"].items()
    }
    rows = []
    for pair_name, metrics in pairwise.items():
        a, b = pair_name.split(" <-> ")
        # Use the minimum bitrate of the two models (pair-level bottleneck)
        pair_bitrate = (model_bitrates.get(a, 0.0) + model_bitrates.get(b, 0.0)) / 2.0
        rows.append(
            {
                "s_local": float(metrics["s_local"]),
                "tau": float(metrics["tau"]),
                "bitrate": pair_bitrate,
            }
        )
    return rows


def _recompute_counts(
    rows: list[dict[str, float]],
    s_local_threshold: float,
    tau_threshold: float,
    r_min: float = 2.0,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        s_local = row["s_local"]
        tau = row["tau"]
        bitrate = row["bitrate"]
        if bitrate < r_min:
            counts["Collapsed"] += 1
        elif s_local >= s_local_threshold and tau >= tau_threshold:
            counts["Convergent"] += 1
        elif s_local >= s_local_threshold:
            counts["Local-Only"] += 1
        else:
            counts["Divergent"] += 1

    return {name: counts.get(name, 0) for name in REGIME_ORDER}


def main() -> None:
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300})

    rows = _load_pairwise_metrics()

    matrix = np.zeros((len(REGIME_ORDER), len(THRESHOLD_SETTINGS)), dtype=int)
    for col_idx, (_, s_local_thr, tau_thr) in enumerate(THRESHOLD_SETTINGS):
        counts = _recompute_counts(rows, s_local_thr, tau_thr)
        for row_idx, regime in enumerate(REGIME_ORDER):
            matrix[row_idx, col_idx] = counts[regime]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu")

    max_value = int(matrix.max()) if matrix.size > 0 else 0
    threshold_for_light_text = max_value * 0.55

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            color = "white" if value >= threshold_for_light_text and value > 0 else "black"
            ax.text(
                col_idx, row_idx, str(value), ha="center", va="center", fontsize=11, color=color
            )

    ax.set_xticks(np.arange(len(THRESHOLD_SETTINGS)))
    ax.set_xticklabels([label for label, _, _ in THRESHOLD_SETTINGS], fontsize=9)
    ax.set_yticks(np.arange(len(REGIME_ORDER)))
    ax.set_yticklabels(REGIME_ORDER, fontsize=10)
    ax.set_title("Regime Counts Under Threshold Settings")
    ax.set_xlabel("Threshold combination")
    ax.set_ylabel("Regime")

    colorbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("Count")

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
