"""Full-atlas (190-pair) retrieval downstream — informational.
Mirrors scripts/retrieval_downstream.py exactly, but over ALL atlas pairs, to
check whether the 20-pair headline (S_local -> knn_stitched, rho=0.76) holds on
the full atlas or was inflated by the curated 20-pair tau-spanning subset.
DOES NOT modify the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
with open(ROOT / "data/atlas.json") as f:
    atlas = json.load(f)
pw = atlas["pairwise"]

lab_tr = torch.load(ROOT / "data/cifar100_train_labels.pt", weights_only=True).to(DEV)
lab_te = torch.load(ROOT / "data/cifar100_test_labels.pt", weights_only=True).to(DEV)

_cache: dict[str, tuple] = {}


def feats(m):
    if m not in _cache:
        tr = (
            torch.load(ROOT / f"data/features_train/{m}.pt", weights_only=True)["features"]
            .float()
            .to(DEV)
        )
        te = (
            torch.load(ROOT / f"data/features/{m}.pt", weights_only=True)["features"]
            .float()
            .to(DEV)
        )
        _cache[m] = (tr, te)
    return _cache[m]


def fit_procrustes(src, tgt):
    sc, tc = src - src.mean(0, keepdim=True), tgt - tgt.mean(0, keepdim=True)
    ds, dt = sc.shape[1], tc.shape[1]
    md = max(ds, dt)
    sp = F.pad(sc, (0, md - ds)) if ds < md else sc
    tp = F.pad(tc, (0, md - dt)) if dt < md else tc
    U, _, Vt = torch.linalg.svd(tp.T @ sp, full_matrices=False)
    return {
        "W": U @ Vt,
        "sm": src.mean(0, keepdim=True),
        "tm": tgt.mean(0, keepdim=True),
        "ds": ds,
        "dt": dt,
        "md": md,
    }


def apply_p(src, st):
    c = src - st["sm"]
    p = F.pad(c, (0, st["md"] - st["ds"])) if st["ds"] < st["md"] else c
    return (p @ st["W"].T)[:, : st["dt"]] + st["tm"]


@torch.no_grad()
def knn(query, qlab, gal, glab, k=10):
    q = F.normalize(query, dim=-1)
    g = F.normalize(gal, dim=-1)
    topk = (q @ g.T).topk(k, dim=1).indices
    preds = torch.mode(glab[topk], dim=1).values
    return float((preds == qlab).float().mean())


@torch.no_grad()
def r10(query, qlab, gal, glab):
    q = F.normalize(query, dim=-1)
    g = F.normalize(gal, dim=-1)
    topk = (q @ g.T).topk(10, dim=1).indices
    return float((glab[topk] == qlab[:, None]).any(dim=1).float().mean())


rows = []
for i, (key, v) in enumerate(pw.items()):
    a, b = key.split(" <-> ")
    fa_tr, fa_te = feats(a)
    fb_tr, fb_te = feats(b)
    n_tr = min(fa_tr.shape[0], fb_tr.shape[0], lab_tr.shape[0])
    n_te = min(fa_te.shape[0], fb_te.shape[0], lab_te.shape[0])
    st = fit_procrustes(fa_tr[:n_tr], fb_tr[:n_tr])
    mapped = apply_p(fa_te[:n_te], st)
    knn_nat = knn(fb_te[:n_te], lab_te[:n_te], fb_tr[:n_tr], lab_tr[:n_tr])
    knn_st = knn(mapped, lab_te[:n_te], fb_tr[:n_tr], lab_tr[:n_tr])
    ret = knn_st / max(knn_nat, 1e-10)
    r10n = r10(fb_te[:n_te], lab_te[:n_te], fb_tr[:n_tr], lab_tr[:n_tr])
    r10s = r10(mapped, lab_te[:n_te], fb_tr[:n_tr], lab_tr[:n_tr])
    rows.append(
        {
            "tau": v["tau"],
            "s_local": v["s_local"],
            "s_global": v["s_global"],
            "delta": v["delta"],
            "knn_stitched": knn_st,
            "knn_retention": ret,
            "ret_r10_retention": r10s / max(r10n, 1e-10),
        }
    )
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/190 done", flush=True)

def col(n):
    return np.array([r[n] for r in rows])
print("\n===== FULL-ATLAS RETRIEVAL (n=190) =====")
print("(compare to 20-pair headline: s_local->knn_stitched rho=0.756)")
res = {}
for p in ["s_local", "s_global", "delta", "tau"]:
    for t in ["knn_stitched", "knn_retention", "ret_r10_retention"]:
        rr = spearmanr(col(p), col(t))
        res[f"{p}->{t}"] = [round(float(rr.correlation), 3), float(f"{rr.pvalue:.2g}")]
        star = " <== headline analog" if (p == "s_local" and t == "knn_stitched") else ""
        print(f"  {p:9s} -> {t:18s} rho={rr.correlation:+.3f} p={rr.pvalue:.2e}{star}")
with open(ROOT / "data/experiments/retrieval_full_atlas.json", "w") as f:
    json.dump(
        {
            "n_pairs": 190,
            "correlations": res,
            "note": "informational full-atlas run; paper reports curated 20-pair subset",
        },
        f,
        indent=2,
    )
print("\nsaved -> data/experiments/retrieval_full_atlas.json")
