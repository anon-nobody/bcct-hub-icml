"""Lineage quantification: is the residual 'family' effect (beta=0.19 after
controlling objective+arch+bitrate) genuinely shared training LINEAGE?

Cleanest isolation: WITHIN the same training objective, do same-lineage pairs
(same recipe/backbone: both DINOv2, both ResNet, ...) transport better than
cross-lineage pairs? If yes, lineage carries signal beyond objective.
Also: (a) nested R^2 (does same_lineage add beyond objective+bitrate),
      (b) tree/cophenetic consistency of the transport-distance matrix.
Atlas-only, no training.
"""

from __future__ import annotations

import json
import statistics as st
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "data/atlas.json") as f:
    d = json.load(f)
models, pw = d["models"], d["pairwise"]
OBJ = {m: models[m]["config"]["objective"] for m in models}
BR = {m: models[m]["metrics"]["bitrate"] for m in models}


# fine LINEAGE label (recipe/backbone), finer than atlas 'family' for supervised bucket
def lineage(m):
    if m.startswith("dinov2"):
        return "dinov2"
    if m.startswith("beit"):
        return "beit"
    if ".mae" in m:
        return "mae"
    if m.startswith("clip"):
        return "clip"
    if m.startswith("resnet"):
        return "resnet"
    if m.startswith("convnext"):
        return "convnext"
    if m.startswith("vit_"):
        return "vit_sup"
    if m.startswith("efficientnet"):
        return "efficientnet"
    if m.startswith("mixer"):
        return "mixer"
    if m.startswith("swin"):
        return "swin"
    return "other"


def tau(a, b):
    return pw.get(f"{a} <-> {b}", pw.get(f"{b} <-> {a}", {})).get("tau")


rows = []
for a, b in combinations(models, 2):
    t = tau(a, b)
    if t is None:
        continue
    rows.append(
        dict(
            a=a,
            b=b,
            tau=t,
            same_obj=int(OBJ[a] == OBJ[b]),
            same_lin=int(lineage(a) == lineage(b)),
            dbr=abs(BR[a] - BR[b]),
        )
    )

print("=== (1) WITHIN same objective: same-lineage vs cross-lineage transport ===")
within_obj = {}
for obj in ["self-supervised", "supervised", "contrastive"]:
    sub = [r for r in rows if r["same_obj"] and OBJ[r["a"]] == obj]
    same = [r["tau"] for r in sub if r["same_lin"]]
    cross = [r["tau"] for r in sub if not r["same_lin"]]
    if same and cross:
        p = mannwhitneyu(same, cross, alternative="greater").pvalue
        within_obj[obj] = {
            "same_lineage_mean_tau": round(st.mean(same), 4),
            "n_same": len(same),
            "cross_lineage_mean_tau": round(st.mean(cross), 4),
            "n_cross": len(cross),
            "mannwhitney_p_greater": float(p),
        }
        print(
            f"  {obj:16s} same-lin mean tau={st.mean(same):.3f} (n={len(same)}) vs cross-lin={st.mean(cross):.3f} (n={len(cross)})  MW p={p:.2e}"
        )
    elif same:
        print(f"  {obj:16s} same-lin only (n={len(same)}), no cross-lineage pairs")
# pooled within-objective
sub = [r for r in rows if r["same_obj"]]
same = [r["tau"] for r in sub if r["same_lin"]]
cross = [r["tau"] for r in sub if not r["same_lin"]]
print(
    f"  POOLED within-objective: same-lin {st.mean(same):.3f} (n={len(same)}) vs cross-lin {st.mean(cross):.3f} (n={len(cross)})  "
    f"MW p={mannwhitneyu(same, cross, alternative='greater').pvalue:.2e}, Cohen-ish d={(st.mean(same) - st.mean(cross)) / st.pstdev(same + cross):.2f}"
)

print("\n=== (2) nested OLS on tau: does same_lineage add beyond objective+bitrate? ===")


def ols(cols):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r["tau"] for r in rows])
    Xa = np.column_stack([np.ones(len(rows)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    yhat = Xa @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return round(r2, 4), dict(zip(cols, beta[1:].round(4), strict=True))


for cols in [["same_obj"], ["same_obj", "dbr"], ["same_obj", "dbr", "same_lin"]]:
    r2, b = ols(cols)
    print(f"  {'+'.join(cols):30s} R2={r2:.3f}  {b}")

print("\n=== (3) within-lineage scale effect (pure recipe, vary size/bitrate) ===")
for lin in ["dinov2", "clip", "resnet", "convnext", "beit"]:
    ms = [m for m in models if lineage(m) == lin]
    if len(ms) < 2:
        continue
    pairs = [
        (abs(BR[a] - BR[b]), tau(a, b)) for a, b in combinations(ms, 2) if tau(a, b) is not None
    ]
    if len(pairs) >= 3:
        rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).correlation
        print(
            f"  {lin:10s} (n_models={len(ms)}, n_pairs={len(pairs)}): corr(|dbitrate|, tau)={rho:+.2f}, mean tau={st.mean(p[1] for p in pairs):.3f}"
        )

print("\n=== (4) does transport structure recover the lineage tree? (cophenetic) ===")
mlist = list(models)
nM = len(mlist)
idx = {m: i for i, m in enumerate(mlist)}
Dm = np.ones((nM, nM))
for a, b in combinations(mlist, 2):
    t = tau(a, b)
    if t is not None:
        Dm[idx[a], idx[b]] = Dm[idx[b], idx[a]] = 1 - t  # transport distance
np.fill_diagonal(Dm, 0.0)
Z = linkage(squareform(Dm, checks=False), method="average")
# metadata tree distance: 0 same lineage, 1 same objective diff lineage, 2 diff objective
Dmeta = np.zeros((nM, nM))
for a, b in combinations(mlist, 2):
    dd = 0 if lineage(a) == lineage(b) else (1 if OBJ[a] == OBJ[b] else 2)
    Dmeta[idx[a], idx[b]] = Dmeta[idx[b], idx[a]] = dd
coph = cophenet(Z, squareform(Dm, checks=False))[0]
r_meta = spearmanr(squareform(Dm, checks=False), squareform(Dmeta, checks=False)).correlation
print(f"  cophenetic corr (transport tree fidelity) = {coph:.3f}")
print(
    f"  corr(transport distance, lineage-tree distance) = {r_meta:.3f}  (higher => transport inherits phylogenetically)"
)

with open(
    Path(__file__).resolve().parents[2] / "data/experiments/lineage_analysis.json", "w"
) as f:
    json.dump(
        {
            "pooled_same_vs_cross_lineage_p": mannwhitneyu(
                same, cross, alternative="greater"
            ).pvalue,
            "coph": float(coph),
            "transport_vs_lineagetree_corr": float(r_meta),
            "within_objective_same_vs_cross_lineage": within_obj,
            "pooled_same_lineage_mean_tau": round(st.mean(same), 4),
            "pooled_cross_lineage_mean_tau": round(st.mean(cross), 4),
        },
        f,
        indent=2,
    )
print("\ndone.")
