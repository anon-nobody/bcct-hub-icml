"""Validate BCCT regime thresholds via unsupervised clustering.

Fits Gaussian Mixture Models (GMM) on (R, S_local, tau) and shows
that data-driven clusters align with the manual 4-regime classification.

Usage:
    python scripts/regime_clustering.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/experiments")
FIG_DIR = Path("paper/figures")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    atlas = json.loads(Path("data/atlas.json").read_text())

    # Build feature matrix
    records = []
    for pair_key, m in atlas["pairwise"].items():
        names = pair_key.split(" <-> ")
        r_a = atlas["models"][names[0]]["metrics"]["bitrate"]
        r_b = atlas["models"][names[1]]["metrics"]["bitrate"]
        records.append({
            "pair": pair_key,
            "bitrate_avg": (r_a + r_b) / 2,
            "s_local": m["s_local"],
            "tau": m["tau"],
            "manual_regime": m["regime"],
        })

    X_raw = np.array([[r["bitrate_avg"], r["s_local"], r["tau"]] for r in records])
    manual_labels = [r["manual_regime"] for r in records]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Fit GMM with 2-6 components, select by BIC
    results = {}
    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=10, random_state=42)
        gmm.fit(X)
        results[k] = {
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "labels": gmm.predict(X),
            "model": gmm,
        }
        logger.info("k=%d: BIC=%.1f, AIC=%.1f", k, gmm.bic(X), gmm.aic(X))

    # Best k by BIC
    best_k = min(results, key=lambda k: results[k]["bic"])
    logger.info("Best k by BIC: %d", best_k)

    # Agreement with manual labels
    # Encode manual labels as integers
    regime_map = {"Collapsed": 0, "Local-Only": 1, "Convergent": 2, "Divergent": 3}
    manual_encoded = np.array([regime_map.get(r, -1) for r in manual_labels])

    for k in range(2, 7):
        gmm_labels = results[k]["labels"]
        ari = adjusted_rand_score(manual_encoded, gmm_labels)
        nmi = normalized_mutual_info_score(manual_encoded, gmm_labels)
        logger.info("k=%d: ARI=%.4f, NMI=%.4f", k, ari, nmi)
        results[k]["ari"] = ari
        results[k]["nmi"] = nmi

    # Detailed analysis for best k and k=3 (since we have 3 regimes in data)
    for k_check in [best_k, 3, 4]:
        if k_check not in results:
            continue
        gmm_labels = results[k_check]["labels"]
        logger.info("\n=== k=%d detailed analysis ===", k_check)

        # Cross-tabulation
        cross_tab = {}
        for ml, gl in zip(manual_labels, gmm_labels, strict=False):
            cross_tab[(ml, gl)] = cross_tab.get((ml, gl), 0) + 1

        regimes_present = sorted(set(manual_labels))
        clusters = sorted(set(gmm_labels))
        logger.info("Cross-tabulation (manual regime × GMM cluster):")
        header = "  " + "".join(f"  C{c}" for c in clusters)
        logger.info(header)
        for reg in regimes_present:
            row = f"  {reg:12s}"
            for c in clusters:
                row += f"  {cross_tab.get((reg, c), 0):3d}"
            logger.info(row)

    # Generate figure
    _plot_clustering(X_raw, manual_labels, results, best_k)

    # Save results
    output = {
        "best_k_bic": best_k,
        "model_selection": {
            str(k): {"bic": results[k]["bic"], "aic": results[k]["aic"],
                      "ari": results[k]["ari"], "nmi": results[k]["nmi"]}
            for k in range(2, 7)
        },
        "agreement_best_k": {
            "ari": results[best_k]["ari"],
            "nmi": results[best_k]["nmi"],
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "regime_clustering.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_DIR / "regime_clustering.json")


def _plot_clustering(X_raw, manual_labels, results, best_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 12, "savefig.dpi": 300})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel (a): Manual regime classification
    regime_colors = {"Convergent": "#2ecc71", "Local-Only": "#f39c12", "Divergent": "#e74c3c", "Collapsed": "#95a5a6"}
    for reg in sorted(set(manual_labels)):
        mask = np.array([r == reg for r in manual_labels])
        axes[0].scatter(X_raw[mask, 2], X_raw[mask, 1], c=regime_colors.get(reg, "gray"),
                        label=reg, s=25, alpha=0.7, edgecolors="w", linewidth=0.3)
    axes[0].set_xlabel("Transport Linearity $\\tau$")
    axes[0].set_ylabel("Calibrated $S_{local}$")
    axes[0].set_title("(a) Manual Regime Classification")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel (b): GMM clustering (best k)
    gmm_labels = results[best_k]["labels"]
    colors = plt.cm.Set1(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = gmm_labels == c
        axes[1].scatter(X_raw[mask, 2], X_raw[mask, 1], c=[colors[c]], label=f"Cluster {c}",
                        s=25, alpha=0.7, edgecolors="w", linewidth=0.3)
    axes[1].set_xlabel("Transport Linearity $\\tau$")
    axes[1].set_ylabel("Calibrated $S_{local}$")
    axes[1].set_title(f"(b) GMM Clustering (k={best_k}, BIC-optimal)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel (c): BIC curve
    ks = list(range(2, 7))
    bics = [results[k]["bic"] for k in ks]
    aris = [results[k]["ari"] for k in ks]
    ax_bic = axes[2]
    ax_bic.plot(ks, bics, "o-", color="steelblue", label="BIC", linewidth=2)
    ax_bic.set_xlabel("Number of Components k")
    ax_bic.set_ylabel("BIC", color="steelblue")
    ax_bic.tick_params(axis="y", labelcolor="steelblue")

    ax_ari = ax_bic.twinx()
    ax_ari.plot(ks, aris, "s--", color="coral", label="ARI", linewidth=2)
    ax_ari.set_ylabel("Adjusted Rand Index", color="coral")
    ax_ari.tick_params(axis="y", labelcolor="coral")

    axes[2].set_title("(c) Model Selection")
    lines1, labels1 = ax_bic.get_legend_handles_labels()
    lines2, labels2 = ax_ari.get_legend_handles_labels()
    ax_bic.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "fig_regime_clustering.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / "fig_regime_clustering.png", bbox_inches="tight")
    logger.info("Clustering figure saved.")


if __name__ == "__main__":
    main()
