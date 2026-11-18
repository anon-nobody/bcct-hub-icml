"""Held-out predictive validation of BCCT regime/transport prediction.

Split 190 vision pairs into train/test by family-aware grouping.
Fit predictive models on train, evaluate on held-out.
Compare: objective-only, CKA-only, Δ-only, BCCT combined.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ATLAS_PATH = Path("data/atlas.json")
OUTPUT_PATH = Path("data/experiments/holdout_validation.json")


def load_atlas() -> tuple[list[dict], dict]:
    """Load atlas and return (pairs_list, models_dict)."""
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)
    models = atlas["models"]
    pairs_list = []
    for key, v in atlas["pairwise"].items():
        parts = key.split(" <-> ")
        v["model_a"] = parts[0]
        v["model_b"] = parts[1]
        v["family_a"] = models[parts[0]]["config"]["family"]
        v["family_b"] = models[parts[1]]["config"]["family"]
        v["objective_a"] = models[parts[0]]["config"]["objective"]
        v["objective_b"] = models[parts[1]]["config"]["objective"]
        v["same_obj"] = 1 if v["objective_a"] == v["objective_b"] else 0
        v["same_fam"] = 1 if v["family_a"] == v["family_b"] else 0
        v["pair_key"] = key
        pairs_list.append(v)
    return pairs_list, models


def family_aware_split(
    pairs: list[dict], models: dict, test_families: set[str]
) -> tuple[list[dict], list[dict]]:
    """Split pairs: test = any pair involving a test-family model."""
    train, test = [], []
    for p in pairs:
        if p["family_a"] in test_families or p["family_b"] in test_families:
            test.append(p)
        else:
            train.append(p)
    return train, test


def build_features(pairs: list[dict], feature_set: str) -> np.ndarray:
    """Build feature matrix for a given feature set."""
    rows = []
    for p in pairs:
        if feature_set == "objective_only":
            rows.append([p["same_obj"]])
        elif feature_set == "cka_only":
            rows.append([p["s_global"]])
        elif feature_set == "delta_only":
            rows.append([p["delta"]])
        elif feature_set == "s_local_only":
            rows.append([p["s_local"]])
        elif feature_set == "bcct_combined":
            rows.append([p["same_obj"], p["same_fam"], p["delta"], p["s_local"], p["s_global"]])
        elif feature_set == "bcct_no_bitrate":
            rows.append([p["same_obj"], p["same_fam"], p["s_local"], p["s_global"]])
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
    return np.array(rows)


def build_labels(pairs: list[dict], threshold: float = 0.5) -> np.ndarray:
    """Binary label: 1 = Convergent (tau >= threshold), 0 = not."""
    return np.array([1 if p["tau"] >= threshold else 0 for p in pairs])


def evaluate_split(
    train_pairs: list[dict],
    test_pairs: list[dict],
    feature_sets: list[str],
    tau_threshold: float = 0.5,
) -> dict:
    """Evaluate multiple feature sets on a train/test split.

    Uses class_weight='balanced' to handle severe class imbalance (13/190 positive).
    Reports both classification (ROC-AUC, PR-AUC, F1) and regression (Spearman rho
    of predicted vs actual tau) metrics.
    """
    y_train_cls = build_labels(train_pairs, tau_threshold)
    y_test_cls = build_labels(test_pairs, tau_threshold)
    y_train_reg = np.array([p["tau"] for p in train_pairs])
    y_test_reg = np.array([p["tau"] for p in test_pairs])

    results = {}
    for fs in feature_sets:
        X_train = build_features(train_pairs, fs)
        X_test = build_features(test_pairs, fs)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        entry: dict = {
            "n_train": len(y_train_cls),
            "n_test": len(y_test_cls),
            "n_pos_train": int(y_train_cls.sum()),
            "n_pos_test": int(y_test_cls.sum()),
        }

        # --- Classification with balanced class weights ---
        if len(set(y_train_cls)) < 2:
            entry.update({"roc_auc": float("nan"), "pr_auc": float("nan"),
                          "f1": 0.0, "note": "single class in train"})
        else:
            clf = LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            )
            clf.fit(X_train_s, y_train_cls)
            y_prob = clf.predict_proba(X_test_s)[:, 1]
            y_pred = clf.predict(X_test_s)

            if len(set(y_test_cls)) >= 2:
                entry["roc_auc"] = round(float(roc_auc_score(y_test_cls, y_prob)), 3)
                entry["pr_auc"] = round(float(average_precision_score(y_test_cls, y_prob)), 3)
            else:
                entry["roc_auc"] = float("nan")
                entry["pr_auc"] = float("nan")

            entry["f1"] = round(float(f1_score(y_test_cls, y_pred, zero_division=0.0)), 3)
            entry["precision"] = round(float(precision_score(y_test_cls, y_pred, zero_division=0.0)), 3)
            entry["recall"] = round(float(recall_score(y_test_cls, y_pred, zero_division=0.0)), 3)

            if hasattr(clf, "coef_"):
                entry["coef"] = [round(float(c), 4) for c in clf.coef_[0]]

        # --- Regression: predict tau directly, report Spearman ---
        reg = Ridge(alpha=1.0)
        reg.fit(X_train_s, y_train_reg)
        y_pred_reg = reg.predict(X_test_s)
        rho_reg, p_reg = spearmanr(y_test_reg, y_pred_reg)
        entry["reg_spearman"] = round(float(rho_reg), 3)
        entry["reg_spearman_p"] = float(f"{p_reg:.2e}")

        results[fs] = entry

    return results


def main() -> None:
    pairs, models = load_atlas()
    logger.info(f"Loaded {len(pairs)} pairs, {len(models)} models")

    # Get all families
    families = sorted(set(m["config"]["family"] for m in models.values()))
    logger.info(f"Families: {families}")

    feature_sets = [
        "objective_only",
        "cka_only",
        "delta_only",
        "s_local_only",
        "bcct_no_bitrate",
        "bcct_combined",
    ]

    # Strategy 1: Leave-one-family-out cross-validation
    logger.info("\n=== Leave-One-Family-Out Cross-Validation ===")
    lofo_results: dict[str, list[dict]] = {fs: [] for fs in feature_sets}

    for held_out_family in families:
        train, test = family_aware_split(pairs, models, {held_out_family})
        if not test or not train:
            continue

        y_test = build_labels(test)
        if len(set(y_test)) < 1:
            continue

        split_results = evaluate_split(train, test, feature_sets)
        for fs in feature_sets:
            if fs in split_results:
                split_results[fs]["held_out_family"] = held_out_family
                lofo_results[fs].append(split_results[fs])

        logger.info(
            f"  Held out: {held_out_family:12s} | train={len(train):3d}, test={len(test):3d} | "
            f"BCCT ROC={split_results.get('bcct_combined', {}).get('roc_auc', 'N/A')}"
        )

    # Aggregate LOFO results
    logger.info("\n=== Aggregated LOFO Results ===")
    agg_results = {}
    for fs in feature_sets:
        roc_aucs = [r["roc_auc"] for r in lofo_results[fs] if not np.isnan(r.get("roc_auc", float("nan")))]
        pr_aucs = [r["pr_auc"] for r in lofo_results[fs] if not np.isnan(r.get("pr_auc", float("nan")))]
        f1s = [r["f1"] for r in lofo_results[fs]]
        reg_rhos = [r["reg_spearman"] for r in lofo_results[fs] if not np.isnan(r.get("reg_spearman", float("nan")))]
        agg_results[fs] = {
            "mean_roc_auc": round(float(np.mean(roc_aucs)), 3) if roc_aucs else float("nan"),
            "std_roc_auc": round(float(np.std(roc_aucs)), 3) if roc_aucs else float("nan"),
            "mean_pr_auc": round(float(np.mean(pr_aucs)), 3) if pr_aucs else float("nan"),
            "mean_f1": round(float(np.mean(f1s)), 3) if f1s else float("nan"),
            "mean_reg_spearman": round(float(np.mean(reg_rhos)), 3) if reg_rhos else float("nan"),
            "n_folds": len(lofo_results[fs]),
            "n_valid_folds": len(roc_aucs),
            "per_fold": lofo_results[fs],
        }
        logger.info(
            f"  {fs:20s} | ROC={agg_results[fs]['mean_roc_auc']:.3f}±{agg_results[fs]['std_roc_auc']:.3f} | "
            f"PR={agg_results[fs]['mean_pr_auc']:.3f} | F1={agg_results[fs]['mean_f1']:.3f} | "
            f"Reg-ρ={agg_results[fs]['mean_reg_spearman']:.3f}"
        )

    # Strategy 2: 60/40 objective-stratified split (5 random seeds)
    logger.info("\n=== Random Stratified Split (5 seeds) ===")
    from sklearn.model_selection import StratifiedShuffleSplit

    y_all = build_labels(pairs)
    random_results: dict[str, list[dict]] = {fs: [] for fs in feature_sets}

    for seed in range(5):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        idx_all = np.arange(len(pairs))

        # Need at least 2 classes
        if len(set(y_all)) < 2:
            break

        for train_idx, test_idx in sss.split(idx_all, y_all):
            train_p = [pairs[i] for i in train_idx]
            test_p = [pairs[i] for i in test_idx]
            split_results = evaluate_split(train_p, test_p, feature_sets)
            for fs in feature_sets:
                if fs in split_results:
                    split_results[fs]["seed"] = seed
                    random_results[fs].append(split_results[fs])

    logger.info("\n=== Aggregated Random Split Results ===")
    random_agg = {}
    for fs in feature_sets:
        roc_aucs = [r["roc_auc"] for r in random_results[fs] if not np.isnan(r.get("roc_auc", float("nan")))]
        pr_aucs = [r["pr_auc"] for r in random_results[fs] if not np.isnan(r.get("pr_auc", float("nan")))]
        f1s = [r["f1"] for r in random_results[fs]]
        reg_rhos = [r["reg_spearman"] for r in random_results[fs] if not np.isnan(r.get("reg_spearman", float("nan")))]
        random_agg[fs] = {
            "mean_roc_auc": round(float(np.mean(roc_aucs)), 3) if roc_aucs else float("nan"),
            "std_roc_auc": round(float(np.std(roc_aucs)), 3) if roc_aucs else float("nan"),
            "mean_pr_auc": round(float(np.mean(pr_aucs)), 3) if pr_aucs else float("nan"),
            "mean_f1": round(float(np.mean(f1s)), 3) if f1s else float("nan"),
            "mean_reg_spearman": round(float(np.mean(reg_rhos)), 3) if reg_rhos else float("nan"),
            "n_splits": len(random_results[fs]),
        }
        logger.info(
            f"  {fs:20s} | ROC={random_agg[fs]['mean_roc_auc']:.3f}±{random_agg[fs]['std_roc_auc']:.3f} | "
            f"PR={random_agg[fs]['mean_pr_auc']:.3f} | F1={random_agg[fs]['mean_f1']:.3f} | "
            f"Reg-ρ={random_agg[fs]['mean_reg_spearman']:.3f}"
        )

    # Save all results
    output = {
        "description": "Held-out predictive validation of BCCT regime prediction",
        "target": "tau >= 0.5 (Convergent)",
        "lofo_cv": agg_results,
        "random_stratified_cv": random_agg,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
