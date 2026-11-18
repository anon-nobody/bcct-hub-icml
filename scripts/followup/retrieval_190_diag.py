"""Diagnose the full-atlas retrieval signal: is S_local -> knn_stitched driven by target quality?

Re-runs retrieval_190.py's exact computation, saves per-pair rows, and reports partial Spearman
correlations controlling for native (target) k-NN accuracy. Writes
data/experiments/retrieval_full_atlas_diag.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.argv = [sys.argv[0]]
ROOT = Path(__file__).resolve().parents[2]


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    return float(np.corrcoef(ex, ey)[0, 1])



if __name__ == "__main__":
    src = (ROOT / "scripts/followup/retrieval_190.py").read_text()
    src = src.replace('"knn_stitched": knn_st,', '"knn_stitched": knn_st, "knn_native": knn_nat, "pair": key,')
    src = src.split('col = lambda')[0]
    ns: dict = {}
    exec(compile(src, "retrieval_190", "exec"), ns)
    rows = ns["rows"]
    col = lambda n: np.array([r[n] for r in rows])  # noqa: E731
    out = {"n_pairs": len(rows), "rows": rows, "stats": {}}
    s = out["stats"]
    for p in ["s_local", "tau", "s_global"]:
        s[f"{p}->knn_stitched"] = round(float(spearmanr(col(p), col("knn_stitched")).correlation), 3)
        s[f"{p}->knn_native"] = round(float(spearmanr(col(p), col("knn_native")).correlation), 3)
        s[f"{p}->knn_retention"] = round(float(spearmanr(col(p), col("knn_retention")).correlation), 3)
        s[f"{p}->knn_stitched | knn_native"] = round(partial_spearman(col(p), col("knn_stitched"), col("knn_native")), 3)
    s["knn_native->knn_stitched"] = round(float(spearmanr(col("knn_native"), col("knn_stitched")).correlation), 3)
    s["retention_quantiles"] = [round(float(q), 3) for q in np.quantile(col("knn_retention"), [0, .1, .5, .9, 1])]
    s["native_quantiles"] = [round(float(q), 3) for q in np.quantile(col("knn_native"), [0, .1, .5, .9, 1])]
    s["stitched_quantiles"] = [round(float(q), 3) for q in np.quantile(col("knn_stitched"), [0, .1, .5, .9, 1])]
    # partial correlations controlling for BOTH encoders' native k-NN accuracy (Appendix B)
    nat = {r["pair"].split(" <-> ")[1]: r["knn_native"] for r in rows}
    src = np.array([nat.get(r["pair"].split(" <-> ")[0], np.nan) for r in rows])
    ok = ~np.isnan(src)
    def pcor2(x, y, zs):
        z = np.column_stack([np.ones(len(x))] + [rankdata(v) for v in zs])
        rx, ry = rankdata(x), rankdata(y)
        ex = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
        ey = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1])
    st, tn = col("knn_stitched"), col("knn_native")
    s["partials"] = {p: {"raw": round(float(spearmanr(col(p), st).correlation), 3),
                         "partial_target": round(partial_spearman(col(p), st, tn), 3),
                         "partial_source_target": round(pcor2(col(p)[ok], st[ok], [src[ok], tn[ok]]), 3)}
                     for p in ["s_local", "tau", "s_global", "delta"]}
    s["partials"]["n_with_source_native"] = int(ok.sum())
    s["partials"]["stitched_vs_min_native"] = round(
        float(spearmanr(np.minimum(src[ok], tn[ok]), st[ok]).correlation), 3)
    with open(ROOT / "data/experiments/retrieval_full_atlas_diag.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(s, indent=1))
