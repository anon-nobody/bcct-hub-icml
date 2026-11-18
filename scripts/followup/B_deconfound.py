"""B: Deconfounding 'family' from objective/architecture/bitrate using natural
experiments already present in the atlas (no training).

Key observations from metadata:
  - The atlas 'family' is nearly collinear with training objective, and the
    'supervised' family lumps 6 different architectures into one label.
  - clip/dinov2/beit/mae/vit are ALL ViT-architecture; the real axis of variation
    across most models is the training OBJECTIVE/recipe, not architecture.
We test whether 'same family' still predicts transport once objective, a proper
architecture label, and bitrate mismatch are controlled -- and we mine the two
arch-matched objective-swap contrasts (ViT-B and ViT-L: supervised vs MAE).
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "data/atlas.json") as f:
    d = json.load(f)
models, pw = d["models"], d["pairwise"]

# proper architecture label (collapse all ViT-based encoders to 'vit')
ARCH = {
    "beit_base_patch16_224": "vit",
    "beit_large_patch16_224": "vit",
    "clip_vitb16": "vit",
    "clip_vitb32": "vit",
    "clip_vitl14": "vit",
    "dinov2_vits14": "vit",
    "dinov2_vitb14": "vit",
    "dinov2_vitl14": "vit",
    "dinov2_vitg14": "vit",
    "vit_base_patch16_224": "vit",
    "vit_large_patch16_224": "vit",
    "vit_base_patch16_224.mae": "vit",
    "vit_large_patch16_224.mae": "vit",
    "convnext_base": "convnext",
    "convnext_large": "convnext",
    "efficientnet_b0": "efficientnet",
    "mixer_b16_224": "mlpmixer",
    "resnet50": "resnet",
    "resnet101": "resnet",
    "swin_base_patch4_window7_224": "swin",
}
# fine training recipe (what the atlas 'family' really tracks)
RECIPE = {m: models[m]["config"]["family"] for m in models}  # dinov2/clip/beit/mae/supervised
OBJ = {m: models[m]["config"]["objective"] for m in models}
BR = {m: models[m]["metrics"]["bitrate"] for m in models}
def arch(m):
    return ARCH[m]

rows = []
for key, v in pw.items():
    a, b = key.split(" <-> ")
    rows.append(
        dict(
            a=a,
            b=b,
            tau=v["tau"],
            y=int(v["tau"] >= 0.5),
            same_family=int(RECIPE[a] == RECIPE[b]),  # atlas coarse 'family'
            same_arch=int(arch(a) == arch(b)),  # PROPER architecture
            same_obj=int(OBJ[a] == OBJ[b]),
            dbr=abs(BR[a] - BR[b]),
        )
    )
n = len(rows)

print("=== (0) the confound: is atlas 'family' just 'objective'? ===")
sf = np.array([r["same_family"] for r in rows])
so = np.array([r["same_obj"] for r in rows])
sar = np.array([r["same_arch"] for r in rows])
print(f"  corr(same_family, same_obj) = {spearmanr(sf, so).correlation:.3f}")
print(f"  corr(same_family, same_arch)= {spearmanr(sf, sar).correlation:.3f}")
print(
    f"  same_arch prevalence: {sar.mean():.2f} (most models are ViT-arch -> architecture barely varies)"
)

print("\n=== (1) natural experiment: SAME architecture, objective swap (supervised vs MAE) ===")
mae_swap = {}
ols_fits = {}
lofo = {}
for base, mae in [
    ("vit_base_patch16_224", "vit_base_patch16_224.mae"),
    ("vit_large_patch16_224", "vit_large_patch16_224.mae"),
]:

    def tau_to(m, c):
        k1, k2 = f"{m} <-> {c}", f"{c} <-> {m}"
        return pw.get(k1, pw.get(k2, {})).get("tau")

    anchors = [m for m in models if m not in (base, mae)]
    dt = [(tau_to(base, c), tau_to(mae, c)) for c in anchors]
    dt = [(x, y) for x, y in dt if x is not None and y is not None]
    dsup = st.mean(x for x, y in dt)
    dmae = st.mean(y for x, y in dt)
    mae_swap[base] = {
        "mean_tau_supervised": round(dsup, 4),
        "mean_tau_mae": round(dmae, 4),
        "n_anchors": len(dt),
        "bitrate_supervised": BR[base],
        "bitrate_mae": BR[mae],
    }
    print(
        f"  {base.split('_')[1]}: mean tau-to-anchors  supervised={dsup:.3f}  MAE={dmae:.3f}  "
        f"(Delta={dsup - dmae:+.3f}); bitrate {BR[base]:.2f} vs {BR[mae]:.2f}"
    )
print("  -> same architecture, only objective/bitrate differ; MAE transports far worse.")

print(
    "\n=== (2) does 'same family' survive controlling for objective+arch+bitrate? (LOFO-free OLS on tau) ==="
)


def ols(cols):
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r["tau"] for r in rows])
    Xa = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    yhat = Xa @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(zip(["intercept"] + cols, beta.round(4), strict=True)), round(r2, 4)


for cols in [
    ["same_family"],
    ["same_obj"],
    ["same_arch"],
    ["same_obj", "dbr"],
    ["same_obj", "same_arch", "dbr"],
    ["same_family", "same_obj", "same_arch", "dbr"],
]:
    b, r2 = ols(cols)
    ols_fits["+".join(cols)] = {"r2": float(r2), "coef": {k: float(v) for k, v in b.items()}}
    print(f"  {'+'.join(cols):40s} R2={r2:.3f}  coeffs={b}")

print("\n=== (3) predictor: replace 'family' with objective -- does AUC hold? ===")

fams = sorted(set(RECIPE.values()))


def lofo_auc(cols):
    yt, yp = [], []
    for hf in fams:
        te = [r for r in rows if RECIPE[r["a"]] == hf or RECIPE[r["b"]] == hf]
        tr = [r for r in rows if RECIPE[r["a"]] != hf and RECIPE[r["b"]] != hf]
        if len(te) < 3 or len({r["y"] for r in tr}) < 2:
            continue
        Xtr = np.array([[r[c] for c in cols] for r in tr])
        Xte = np.array([[r[c] for c in cols] for r in te])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            sc.transform(Xtr), [r["y"] for r in tr]
        )
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        yt += [r["y"] for r in te]
        yp += p.tolist()
    return round(roc_auc_score(yt, yp), 3) if len(set(yt)) == 2 else None


for cols in [["same_obj", "dbr"], ["same_arch", "same_obj", "dbr"], ["same_family", "dbr"]]:
    lofo["+".join(cols)] = lofo_auc(cols)
    print(f"  {'+'.join(cols):35s} pooled-OOF AUC={lofo['+'.join(cols)]}")

with open(
    Path(__file__).resolve().parents[2] / "data/experiments/B_deconfound.json", "w"
) as f:
    json.dump(
        {
            "note": "deconfounding on existing atlas; family~objective confound + supervised bucket",
            "same_family_vs_same_obj_corr": round(float(spearmanr(sf, so).correlation), 3),
            "same_family_vs_same_arch_corr": round(float(spearmanr(sf, sar).correlation), 3),
            "supervised_vs_mae_same_architecture": mae_swap,
            "ols_tau": ols_fits,
            "lofo_auc_tau_ge_0p5": lofo,
        },
        f,
        indent=2,
    )
print("\ndone.")
