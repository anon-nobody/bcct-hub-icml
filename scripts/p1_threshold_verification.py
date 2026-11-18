"""P1: Verify the prediction 'Δ > 2×R_min → τ < 0.5'.

Collects (Δ, τ, R_min) data points from:
1. Atlas (190 pairwise metrics at native bitrate)
2. Causal verification (10 pairs × 7 compression levels)

Performs logistic regression, ROC analysis, and computes prediction accuracy.

Usage:
    python scripts/p1_threshold_verification.py [--output data/experiments/p1_threshold.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.special import expit  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def load_atlas_points(atlas_path: Path) -> list[dict]:
    """Extract (Δ, τ, R_min) from atlas pairwise data."""
    atlas = json.loads(atlas_path.read_text())
    points = []

    for pair_key, metrics in atlas["pairwise"].items():
        names = pair_key.split(" <-> ")
        # Get individual bitrates
        r_a = atlas["models"][names[0]]["metrics"]["bitrate"]
        r_b = atlas["models"][names[1]]["metrics"]["bitrate"]
        r_min = min(r_a, r_b)

        points.append({
            "pair": pair_key,
            "source": "atlas",
            "delta": metrics["delta"],
            "tau": metrics["tau"],
            "r_a": r_a,
            "r_b": r_b,
            "r_min": r_min,
            "delta_over_r_min": metrics["delta"] / max(r_min, 1e-10),
            "regime": metrics["regime"],
        })

    return points


def load_causal_points(causal_path: Path) -> list[dict]:
    """Extract (Δ, τ, R_min) from causal verification data."""
    data = json.loads(causal_path.read_text())
    points = []

    for pair_result in data["results"]:
        for pt in pair_result["points"]:
            r_min = min(pt["bitrate_a"], pt["bitrate_b"])
            delta = pt["delta"]
            points.append({
                "pair": pair_result["pair"],
                "source": "causal",
                "compression_level": pt["level"],
                "delta": delta,
                "tau": pt["tau"],
                "r_a": pt["bitrate_a"],
                "r_b": pt["bitrate_b"],
                "r_min": r_min,
                "delta_over_r_min": delta / max(r_min, 1e-10),
            })

    return points


def analyze_p1(points: list[dict]) -> dict:
    """Run P1 analysis: logistic regression, ROC, prediction accuracy."""
    deltas = np.array([p["delta"] for p in points])
    taus = np.array([p["tau"] for p in points])
    np.array([p["r_min"] for p in points])
    delta_ratios = np.array([p["delta_over_r_min"] for p in points])

    # Binary labels: τ < 0.5 means transport fails
    y_true = (taus < 0.5).astype(int)

    # P1 prediction: Δ > 2×R_min → τ < 0.5
    # Equivalent to: delta_over_r_min > 2 → y=1
    # But paper says Δ > 3×R_min, let's test both
    results = {}

    for threshold_name, threshold_col, threshold_val in [
        ("delta_gt_2Rmin", delta_ratios, 2.0),
        ("delta_gt_3Rmin", delta_ratios, 3.0),
        ("delta_gt_1", deltas, 1.0),
        ("delta_gt_2", deltas, 2.0),
    ]:
        y_pred = (threshold_col > threshold_val).astype(int)
        n_pos = int(y_pred.sum())
        n_neg = int(len(y_pred) - n_pos)

        if n_pos > 0 and n_neg > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
        else:
            acc = prec = rec = 0.0

        results[threshold_name] = {
            "threshold": threshold_val,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        }

    # Logistic regression: P(τ < 0.5) ~ logistic(Δ)
    if len(np.unique(y_true)) > 1:
        X_delta = deltas.reshape(-1, 1)
        lr_model = LogisticRegression(solver="lbfgs")
        lr_model.fit(X_delta, y_true)
        y_prob = lr_model.predict_proba(X_delta)[:, 1]

        auc = roc_auc_score(y_true, y_prob)
        # Critical Δ where P(fail) = 0.5
        # logistic: P = 1/(1+exp(-(w*x + b))) = 0.5 → w*x + b = 0 → x = -b/w
        w = float(lr_model.coef_[0][0])
        b = float(lr_model.intercept_[0])
        delta_critical = -b / w if abs(w) > 1e-10 else float("inf")

        logistic = {
            "auc_roc": round(float(auc), 4),
            "delta_critical": round(float(delta_critical), 4),
            "coef": round(w, 4),
            "intercept": round(b, 4),
        }

        # Same with Δ/R_min as predictor
        X_ratio = delta_ratios.reshape(-1, 1)
        lr_ratio = LogisticRegression(solver="lbfgs")
        lr_ratio.fit(X_ratio, y_true)
        y_prob_ratio = lr_ratio.predict_proba(X_ratio)[:, 1]
        auc_ratio = roc_auc_score(y_true, y_prob_ratio)
        w_r = float(lr_ratio.coef_[0][0])
        b_r = float(lr_ratio.intercept_[0])
        ratio_critical = -b_r / w_r if abs(w_r) > 1e-10 else float("inf")

        logistic["auc_roc_ratio"] = round(float(auc_ratio), 4)
        logistic["ratio_critical"] = round(float(ratio_critical), 4)
    else:
        logistic = {"error": "All labels identical, cannot fit logistic regression"}

    # Summary statistics
    stats = {
        "n_points": len(points),
        "n_atlas": sum(1 for p in points if p["source"] == "atlas"),
        "n_causal": sum(1 for p in points if p["source"] == "causal"),
        "n_transport_fail": int(y_true.sum()),
        "n_transport_ok": int(len(y_true) - y_true.sum()),
        "delta_range": [round(float(deltas.min()), 4), round(float(deltas.max()), 4)],
        "tau_range": [round(float(taus.min()), 4), round(float(taus.max()), 4)],
        "correlation_delta_tau": round(float(np.corrcoef(deltas, taus)[0, 1]), 4),
    }

    return {
        "thresholds": results,
        "logistic_regression": logistic,
        "statistics": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P1: Δ-τ threshold verification")
    parser.add_argument("--atlas", type=str, default="data/atlas.json")
    parser.add_argument("--causal", type=str, default="data/experiments/causal_bitrate_verification.json")
    parser.add_argument("--output", type=str, default="data/experiments/p1_threshold.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_points = []

    atlas_path = Path(args.atlas)
    if atlas_path.exists():
        atlas_pts = load_atlas_points(atlas_path)
        logger.info("Loaded %d points from atlas", len(atlas_pts))
        all_points.extend(atlas_pts)
    else:
        logger.warning("Atlas not found at %s", atlas_path)

    causal_path = Path(args.causal)
    if causal_path.exists():
        causal_pts = load_causal_points(causal_path)
        logger.info("Loaded %d points from causal verification", len(causal_pts))
        all_points.extend(causal_pts)
    else:
        logger.warning("Causal data not found at %s — run causal_bitrate_verification.py first", causal_path)

    if not all_points:
        logger.error("No data points available")
        return

    analysis = analyze_p1(all_points)

    output = {
        "metadata": {
            "atlas_path": str(atlas_path),
            "causal_path": str(causal_path),
        },
        "points": all_points,
        "analysis": analysis,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Statistics: %s", json.dumps(analysis["statistics"], indent=2))
    logger.info("Thresholds: %s", json.dumps(analysis["thresholds"], indent=2))
    logger.info("Logistic: %s", json.dumps(analysis["logistic_regression"], indent=2))


if __name__ == "__main__":
    main()
