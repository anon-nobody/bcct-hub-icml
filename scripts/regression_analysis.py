from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

SCIPY_STATS: Any = import_module("scipy.stats")


@dataclass(frozen=True)
class PairRecord:
    tau: float
    bitrate_avg: float
    bitrate_min: float
    bitrate_mismatch: float
    objective_same: int
    objective_pair_type: str
    family_same: int
    dim_a: int
    dim_b: int
    objective_a: str
    objective_b: str


def _normalize_objective(objective: str) -> str:
    value = objective.strip().lower().replace("_", "-")
    if value in {"contrastive", "supervised", "self-supervised"}:
        return value
    if value == "selfsupervised":
        return "self-supervised"
    return value


def _objective_pair_type(obj_a: str, obj_b: str) -> str:
    if obj_a == obj_b and obj_a in {"contrastive", "supervised", "self-supervised"}:
        return obj_a
    return "mixed"


def load_pair_records(atlas_path: Path) -> list[PairRecord]:
    with atlas_path.open("r", encoding="utf-8") as f:
        atlas: dict[str, Any] = json.load(f)

    models = atlas["models"]
    pairwise = atlas["pairwise"]
    records: list[PairRecord] = []

    for pair_name, pair_metrics in pairwise.items():
        if "tau" not in pair_metrics:
            continue
        if " <-> " not in pair_name:
            continue

        model_a, model_b = pair_name.split(" <-> ", maxsplit=1)
        if model_a not in models or model_b not in models:
            continue

        model_a_data = models[model_a]
        model_b_data = models[model_b]

        bitrate_a = float(model_a_data["metrics"]["bitrate"])
        bitrate_b = float(model_b_data["metrics"]["bitrate"])

        objective_a = _normalize_objective(str(model_a_data["config"]["objective"]))
        objective_b = _normalize_objective(str(model_b_data["config"]["objective"]))

        family_a = str(model_a_data["config"]["family"])
        family_b = str(model_b_data["config"]["family"])

        dim_a = int(model_a_data["config"]["dim"])
        dim_b = int(model_b_data["config"]["dim"])

        records.append(
            PairRecord(
                tau=float(pair_metrics["tau"]),
                bitrate_avg=(bitrate_a + bitrate_b) / 2.0,
                bitrate_min=min(bitrate_a, bitrate_b),
                bitrate_mismatch=abs(bitrate_a - bitrate_b),
                objective_same=int(objective_a == objective_b),
                objective_pair_type=_objective_pair_type(objective_a, objective_b),
                family_same=int(family_a == family_b),
                dim_a=dim_a,
                dim_b=dim_b,
                objective_a=objective_a,
                objective_b=objective_b,
            )
        )

    if not records:
        raise ValueError("No pair records found in atlas data.")

    return records


def _build_matrix(records: list[PairRecord], features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([row.tau for row in records], dtype=float)
    cols = [
        np.array([float(getattr(row, feature)) for row in records], dtype=float)
        for feature in features
    ]
    x = np.column_stack(cols) if cols else np.empty((len(records), 0), dtype=float)
    return x, y


def _ols_fit(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    n_obs = len(y)
    x_design = np.column_stack([np.ones(n_obs, dtype=float), x])

    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    y_hat = x_design @ beta
    residuals = y - y_hat

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else 0.0

    n_params = x_design.shape[1]
    dof = n_obs - n_params
    if dof > 0:
        sigma2 = sse / dof
        xtx_inv = np.linalg.pinv(x_design.T @ x_design)
        cov_beta = sigma2 * xtx_inv
        se = np.sqrt(np.diag(cov_beta))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = beta / se
        p_values = 2.0 * (1.0 - SCIPY_STATS.t.cdf(np.abs(t_values), dof))
    else:
        se = np.full_like(beta, np.nan)
        t_values = np.full_like(beta, np.nan)
        p_values = np.full_like(beta, np.nan)

    names = ["intercept", *feature_names]
    coefficients: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        coefficients.append(
            {
                "term": name,
                "coef": float(beta[idx]),
                "std_err": float(se[idx]) if np.isfinite(se[idx]) else None,
                "t": float(t_values[idx]) if np.isfinite(t_values[idx]) else None,
                "p_value": float(p_values[idx]) if np.isfinite(p_values[idx]) else None,
            }
        )

    return {
        "n": n_obs,
        "dof": dof,
        "r2": r2,
        "adj_r2": (1.0 - (1.0 - r2) * (n_obs - 1) / dof) if dof > 0 else None,
        "sse": sse,
        "features": feature_names,
        "coefficients": coefficients,
    }


def _compare_nested_models(reduced: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    reduced_sse = float(cast("float", reduced["sse"]))
    full_sse = float(cast("float", full["sse"]))
    reduced_dof = int(cast("int", reduced["dof"]))
    full_dof = int(cast("int", full["dof"]))

    df_num = reduced_dof - full_dof
    df_den = full_dof
    if df_num <= 0 or df_den <= 0 or full_sse <= 0:
        return {
            "f_stat": None,
            "df_num": df_num,
            "df_den": df_den,
            "p_value": None,
            "delta_r2": float(cast("float", full["r2"])) - float(cast("float", reduced["r2"])),
            "improves": None,
        }

    improvement = max(reduced_sse - full_sse, 0.0)
    f_stat = (improvement / df_num) / (full_sse / df_den)
    p_value = 1.0 - SCIPY_STATS.f.cdf(f_stat, df_num, df_den)
    delta_r2 = float(cast("float", full["r2"])) - float(cast("float", reduced["r2"]))

    return {
        "f_stat": float(f_stat),
        "df_num": df_num,
        "df_den": df_den,
        "p_value": float(p_value),
        "delta_r2": delta_r2,
        "improves": bool(p_value < 0.05 and delta_r2 > 0),
    }


def _residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(target), dtype=float), controls])
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    return target - (x @ beta)


def _partial_corr(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> dict[str, Any]:
    y_res = _residualize(y, controls)
    x_res = _residualize(x, controls)
    corr, p_val = SCIPY_STATS.pearsonr(x_res, y_res)
    return {
        "n": int(len(y)),
        "correlation": float(corr),
        "p_value": float(p_val),
    }


def _one_hot(values: list[str], drop_first: bool = True) -> tuple[np.ndarray, list[str]]:
    categories = sorted(set(values))
    used = categories[1:] if drop_first and len(categories) > 1 else categories

    if not used:
        return np.zeros((len(values), 0), dtype=float), []

    cols: list[np.ndarray] = []
    names: list[str] = []
    for category in used:
        cols.append(np.array([1.0 if value == category else 0.0 for value in values], dtype=float))
        names.append(f"objective_pair_type={category}")
    return np.column_stack(cols), names


def _stratified_same_objective(records: list[PairRecord]) -> dict[str, Any]:
    same = [row for row in records if row.objective_same == 1]
    if len(same) < 3:
        return {
            "n": len(same),
            "note": "Insufficient same-objective pairs for stable analysis.",
        }

    x_all = np.array([row.bitrate_avg for row in same], dtype=float)
    y_all = np.array([row.tau for row in same], dtype=float)
    all_corr, all_p = SCIPY_STATS.pearsonr(x_all, y_all)
    same_fit = _ols_fit(x_all.reshape(-1, 1), y_all, ["bitrate_avg"])

    by_objective: dict[str, Any] = {}
    for objective in ["contrastive", "supervised", "self-supervised"]:
        group = [row for row in same if row.objective_pair_type == objective]
        if len(group) < 3:
            by_objective[objective] = {
                "n": len(group),
                "note": "Insufficient pairs.",
            }
            continue
        x_group = np.array([row.bitrate_avg for row in group], dtype=float)
        y_group = np.array([row.tau for row in group], dtype=float)
        corr, p_val = SCIPY_STATS.pearsonr(x_group, y_group)
        by_objective[objective] = {
            "n": len(group),
            "pearson_r": float(corr),
            "pearson_p_value": float(p_val),
            "regression": _ols_fit(x_group.reshape(-1, 1), y_group, ["bitrate_avg"]),
        }

    return {
        "same_objective_overall": {
            "n": len(same),
            "pearson_r": float(all_corr),
            "pearson_p_value": float(all_p),
            "regression": same_fit,
        },
        "by_objective": by_objective,
    }


def run_analysis(atlas_path: Path) -> dict[str, Any]:
    records = load_pair_records(atlas_path)

    model_specs: list[tuple[str, list[str]]] = [
        ("m1_tau_bitrate_avg", ["bitrate_avg"]),
        ("m2_tau_bitrate_avg_mismatch", ["bitrate_avg", "bitrate_mismatch"]),
        ("m3_tau_bitrate_avg_objective_same", ["bitrate_avg", "objective_same"]),
        (
            "m4_tau_bitrate_and_compatibility",
            ["bitrate_avg", "bitrate_mismatch", "objective_same", "family_same"],
        ),
        ("m5_tau_compatibility_no_bitrate", ["objective_same", "family_same", "dim_a", "dim_b"]),
    ]

    models_out: list[dict[str, Any]] = []
    model_by_name: dict[str, dict[str, Any]] = {}
    for model_name, features in model_specs:
        x, y = _build_matrix(records, features)
        fit = _ols_fit(x, y, features)
        model_entry: dict[str, Any] = {"name": model_name, **fit}
        models_out.append(model_entry)
        model_by_name[model_name] = model_entry

    objective_only_x, objective_only_y = _build_matrix(records, ["objective_same"])
    objective_only_fit = _ols_fit(objective_only_x, objective_only_y, ["objective_same"])

    bitrate_plus_objective = model_by_name["m3_tau_bitrate_avg_objective_same"]
    bitrate_over_objective = _compare_nested_models(objective_only_fit, bitrate_plus_objective)

    y = np.array([row.tau for row in records], dtype=float)
    x_bitrate = np.array([row.bitrate_avg for row in records], dtype=float)
    objective_same_controls = np.array(
        [[float(row.objective_same)] for row in records], dtype=float
    )
    objective_pair_values = [row.objective_pair_type for row in records]
    objective_pair_one_hot, objective_pair_names = _one_hot(objective_pair_values, drop_first=True)

    controls_combo = np.column_stack([objective_same_controls, objective_pair_one_hot])
    partials: dict[str, Any] = {
        "tau_vs_bitrate_control_objective_same": _partial_corr(
            y, x_bitrate, objective_same_controls
        ),
        "tau_vs_bitrate_control_objective_pair_type": _partial_corr(y, x_bitrate, controls_combo),
        "control_terms": ["objective_same", *objective_pair_names],
    }

    stratified = _stratified_same_objective(records)

    best_model = max(models_out, key=lambda entry: float(cast("float", entry["r2"])))
    delta_pct = float(cast("float", bitrate_over_objective["delta_r2"])) * 100.0

    summary = (
        f"Adding bitrate_avg to objective_same increases explained variance in tau by "
        f"{delta_pct:.2f} percentage points (delta R^2), based on observational regression. "
        f"Highest R^2 among specified models is {best_model['name']} at {best_model['r2']:.4f}."
    )

    return {
        "n_pairs": len(records),
        "models": models_out,
        "objective_only_reference": objective_only_fit,
        "bitrate_over_objective": bitrate_over_objective,
        "partial_correlations": partials,
        "stratified": stratified,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regression analysis for transport linearity (tau)."
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path("data/atlas.json"),
        help="Path to atlas JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/regression_analysis.json"),
        help="Path to write analysis JSON.",
    )
    args = parser.parse_args()

    results = run_analysis(args.atlas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_pairs = int(cast("int", results["n_pairs"]))
    print(f"Analyzed {n_pairs} pairs from {args.atlas}.")
    models = cast("list[dict[str, Any]]", results["models"])
    for model in sorted(models, key=lambda m: float(cast("float", m["r2"])), reverse=True):
        print(f"- {model['name']}: R^2={float(cast('float', model['r2'])):.4f}")

    compare = cast("dict[str, Any]", results["bitrate_over_objective"])
    print(
        "Bitrate over objective-only: "
        f"delta R^2={float(cast('float', compare['delta_r2'])):.4f}, "
        f"F-test p={compare['p_value']}"
    )
    print(str(cast("str", results["summary"])))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
