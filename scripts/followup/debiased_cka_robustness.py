"""Debiased-CKA robustness of Alignment Locality (lambda).

Recomputes S_global with the UNBIASED HSIC estimator (Song et al. 2012;
Nguyen et al. 2021 debiased CKA) on the same 20-model vision atlas features,
then recomputes lambda = S_local / S_global_debiased and checks:
  (1) pipeline fidelity: biased CKA reproduces stored atlas s_global
  (2) rank preservation of S_global (biased vs debiased)
  (3) rank preservation of lambda (original vs debiased)
  (4) regime invariance (classify_regime ignores s_global by construction)
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features"
with open(ROOT / "data/atlas.json") as f:
    ATLAS = json.load(f)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def gram_linear(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T


def center_gram_biased(g: torch.Tensor) -> torch.Tensor:
    # standard HLH centering (Kornblith 2019 biased)
    means = g.mean(0)
    means = means - means.mean() / 2
    g = g - means[:, None] - means[None, :]
    return g


def center_gram_unbiased(g: torch.Tensor) -> torch.Tensor:
    # Song et al. (2012) unbiased HSIC centering (Nguyen et al. 2021 debiased CKA)
    n = g.shape[0]
    g = g.clone()
    g.fill_diagonal_(0)
    means = g.sum(0) / (n - 2)
    means = means - means.sum() / (2 * (n - 1))
    g = g - means[:, None] - means[None, :]
    g.fill_diagonal_(0)
    return g


def cka_from_grams(gx: torch.Tensor, gy: torch.Tensor) -> float:
    hsic = (gx * gy).sum()
    nx = torch.linalg.norm(gx)
    ny = torch.linalg.norm(gy)
    denom = nx * ny
    if denom < 1e-12:
        return 0.0
    return float((hsic / denom).item())


def cka_biased_featform(a: torch.Tensor, b: torch.Tensor) -> float:
    # exact reproduction of src/latent/metrics/alignment.centered_kernel_alignment
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    cross = torch.linalg.norm(a.T @ b, ord="fro").item() ** 2
    na = torch.linalg.norm(a.T @ a, ord="fro").item() ** 2
    nb = torch.linalg.norm(b.T @ b, ord="fro").item() ** 2
    denom = (na * nb) ** 0.5
    return 0.0 if denom < 1e-10 else cross / denom


# load features for the 20 atlas models
models = {}
for name in ATLAS["models"]:
    p = FEAT / f"{name}.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    d = torch.load(p, map_location="cpu", weights_only=True)
    models[name] = d["features"].float()
print(f"loaded {len(models)} models")

rows = []
gram_cache_b: dict[str, torch.Tensor] = {}
gram_cache_u: dict[str, torch.Tensor] = {}


def grams(name: str, n: int):
    key = f"{name}:{n}"
    if key not in gram_cache_b:
        x = models[name][:n].to(DEV)
        g = gram_linear(x)
        gram_cache_b[key] = center_gram_biased(g)
        gram_cache_u[key] = center_gram_unbiased(g)
        del x, g
        torch.cuda.empty_cache() if DEV.startswith("cuda") else None
    return gram_cache_b[key], gram_cache_u[key]


for pair_key, pv in ATLAS["pairwise"].items():
    a, b = pair_key.split(" <-> ")
    n = min(models[a].shape[0], models[b].shape[0])
    gxb, gxu = grams(a, n)
    gyb, gyu = grams(b, n)

    cka_bias_gram = cka_from_grams(gxb, gyb)
    cka_debias = cka_from_grams(gxu, gyu)
    cka_bias_feat = cka_biased_featform(models[a][:n].to(DEV), models[b][:n].to(DEV))

    s_local = pv["s_local"]
    lam_orig = pv["lambda"]
    lam_debias = s_local / cka_debias if cka_debias > 1e-10 else float("inf")

    rows.append(
        {
            "pair": pair_key,
            "s_global_stored": pv["s_global"],
            "cka_bias_feat": cka_bias_feat,
            "cka_bias_gram": cka_bias_gram,
            "cka_debias": cka_debias,
            "s_local": s_local,
            "lambda_orig": lam_orig,
            "lambda_debias": lam_debias,
            "regime": pv["regime"],
        }
    )

# ---- analysis ----
repro_absdiff = [abs(r["cka_bias_feat"] - r["s_global_stored"]) for r in rows]
sg_stored = [r["s_global_stored"] for r in rows]
sg_debias = [r["cka_debias"] for r in rows]
lam_o = [
    r["lambda_orig"] for r in rows if r["lambda_orig"] not in (None,) and r["lambda_orig"] < 1e6
]
# align lambda lists on finite entries
pairs_fin = [
    (r["lambda_orig"], r["lambda_debias"])
    for r in rows
    if r["lambda_orig"] < 1e6 and r["lambda_debias"] < 1e6
]
lo = [p[0] for p in pairs_fin]
ld = [p[1] for p in pairs_fin]

print("\n===== DEBIASED-CKA ROBUSTNESS =====")
print(f"n_pairs = {len(rows)}")
print("\n(1) Pipeline fidelity: biased CKA (feat-form) vs stored s_global")
print(f"    mean|diff| = {st.mean(repro_absdiff):.5f}, max|diff| = {max(repro_absdiff):.5f}")
print(f"    Pearson r  = {pearsonr(sg_stored, [r['cka_bias_feat'] for r in rows])[0]:.5f}")

print("\n(2) S_global rank preservation: stored(biased) vs debiased")
print(f"    Spearman rho = {spearmanr(sg_stored, sg_debias).correlation:.4f}")
print(f"    Pearson  r   = {pearsonr(sg_stored, sg_debias)[0]:.4f}")
print(f"    mean s_global biased  = {st.mean(sg_stored):.4f}")
print(f"    mean s_global debiased= {st.mean(sg_debias):.4f}")

print(f"\n(3) lambda rank preservation: original vs debiased (n_finite={len(lo)})")
print(f"    Spearman rho = {spearmanr(lo, ld).correlation:.4f}")
print(f"    Pearson  r   = {pearsonr(lo, ld)[0]:.4f}")

# (4) regime invariance: classify_regime uses (bitrate, s_local, tau) only.
print("\n(4) Regime invariance: classify_regime does not consume s_global/lambda")
print(f"    -> 0 / {len(rows)} regime labels can change under CKA debiasing (by construction)")

# save
out = ROOT / "data/experiments/debiased_cka_robustness.json"
summary = {
    "description": "Debiased (unbiased-HSIC) CKA robustness of Alignment Locality",
    "estimator": "Song et al. 2012 unbiased HSIC / Nguyen et al. 2021 debiased CKA (linear kernel)",
    "n_pairs": len(rows),
    "pipeline_fidelity_mean_absdiff": round(st.mean(repro_absdiff), 6),
    "pipeline_fidelity_max_absdiff": round(max(repro_absdiff), 6),
    "s_global_spearman_biased_vs_debiased": round(
        float(spearmanr(sg_stored, sg_debias).correlation), 4
    ),
    "s_global_pearson_biased_vs_debiased": round(float(pearsonr(sg_stored, sg_debias)[0]), 4),
    "s_global_mean_biased": round(st.mean(sg_stored), 4),
    "s_global_mean_debiased": round(st.mean(sg_debias), 4),
    "lambda_spearman_orig_vs_debiased": round(float(spearmanr(lo, ld).correlation), 4),
    "lambda_pearson_orig_vs_debiased": round(float(pearsonr(lo, ld)[0]), 4),
    "n_lambda_finite": len(lo),
    "regime_labels_changed": 0,
    "regime_invariance_reason": "classify_regime() branches on (bitrate, s_local, tau) only; s_global/lambda are reported, not decision inputs",
}
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nsaved -> {out}")
print(json.dumps(summary, indent=2))
