"""Regenerate the tau-vs-CKA divergence analysis from data/atlas.json.

data/experiments/tau_vs_cka_divergence.json was produced inline in May 2026 with no
script checked into the repo. This script recomputes the same analysis purely from
the already-committed atlas (no feature files, no re-running of metrics) so the
result is reproducible going forward.

"CKA" in the original file is the atlas's `s_global` field (S_global is computed via
centered_kernel_alignment in latent.metrics.alignment, per CLAUDE.md's module table),
and "tau" is the atlas's `tau` field, which atlas computation already defines as the
bidirectional-conservative min(tau_a_to_b, tau_b_to_a) (see scripts/compute_alignment_atlas.py
compute_pairwise_metrics: `tau_pair = min(transport.a_to_b.tau, transport.b_to_a.tau)`).

Reproduces, from the 190 pairwise entries in data/atlas.json:
  - the "type A" set: pairs with s_global > 0.55 AND tau < 0.2 (high CKA, low tau --
    the divergence the paper highlights), with cross-family / cross-objective counts
  - its aggregate statistics (n, means, ranges)
  - the "type B" set: pairs with s_global < 0.4 AND tau > 0.3 (low CKA, high tau)
  - the "type C" set: the top-5 pairs-of-pairs, among all 190-choose-2 combinations
    with |s_global_i - s_global_j| <= 0.02, ranked by |tau_i - tau_j| descending
  - Spearman(tau, s_global) and 1 - rho^2 ("variance of tau unexplained by CKA"),
    computed over all 190 pairs

Thresholds were reverse-engineered against the original file's numbers, not swept
blindly: the type-A threshold (s_global > 0.55 AND tau < 0.2) is copied verbatim from
the original file's own "full_type_a_aggregate.definition" string. The type-B
threshold (s_global < 0.4 AND tau > 0.3) and the type-C rule (|delta s_global| <=
0.02, top-5 by |delta tau|) were each found by one targeted hypothesis -- informed by
eyeballing the original file's s_global/tau values -- that was checked against all 190
pairs and matched the original entries (membership, values, and order) exactly on the
first attempt; no wider parameter sweep over these two was needed since the match was
already exact. They are hardcoded below since they are the rules being verified, not
free parameters.

The original file's short "type_a_high_cka_low_tau" (8-example) representative list is
NOT reproduced here, but it is *not* unexplained: an actual search (see
scripts/followup -- the sweep is not checked in, just the result) found that 7 of its 8
entries decompose exactly as the union of
  (a) the top-3 type-A pairs by s_global, restricted to the 19 type-A pairs with
      tau < 0.1 (beit_large<->vit_large.mae, dinov2_vitb14<->swin_base,
      beit_base<->vit_large.mae), and
  (b) the four clip_vitl14 pairs that serve as the low-tau anchor in the first four
      entries of the type_c_similar_cka_different_tau list above (vit_large,
      convnext_large, convnext_base, swin_base).
The 8th entry, clip_vitl14 <-> dinov2_vits14, is not produced by (a), by (b), by any
single threshold on (s_global, tau) over the 19-pair tau<0.1 subset (a full grid sweep
over all observed s_global/tau cutoffs found no exact match), nor by top-N rulings by
s_global or by tau within type A or within that subset. Because of this one leftover,
and because (a)+(b) is itself a two-part composite rather than one rule, this file
treats the representative 8-list as a hand/editorial selection and does not try to
emit it.

Input:  data/atlas.json
Output: data/experiments/tau_vs_cka_divergence.json (the May 2026 inline result is kept as
        data/experiments/tau_vs_cka_divergence_may2026.json)

Usage:
    python scripts/followup/cka_tau_divergence.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ATLAS_PATH = ROOT / "data/atlas.json"
OUTPUT_PATH = ROOT / "data/experiments/tau_vs_cka_divergence.json"

TYPE_A_S_GLOBAL_MIN = 0.55
TYPE_A_TAU_MAX = 0.2
TYPE_B_S_GLOBAL_MAX = 0.4
TYPE_B_TAU_MIN = 0.3
TYPE_C_DELTA_S_GLOBAL_MAX = 0.02
TYPE_C_TOP_K = 5


def load_pairs(atlas: dict) -> list[dict]:
    """Flatten atlas["pairwise"] into per-pair rows with model metadata."""
    models = atlas["models"]
    rows = []
    for key, v in atlas["pairwise"].items():
        model_a, model_b = key.split(" <-> ")
        rows.append(
            {
                "pair": key,
                "model_a": model_a,
                "model_b": model_b,
                "s_global": v["s_global"],
                "tau": v["tau"],
                "s_local": v["s_local"],
                "delta": v["delta"],
                "regime": v["regime"],
                "same_family": models[model_a]["config"]["family"]
                == models[model_b]["config"]["family"],
                "same_objective": models[model_a]["config"]["objective"]
                == models[model_b]["config"]["objective"],
            }
        )
    return rows


def type_a_entries(rows: list[dict]) -> list[dict]:
    matches = [r for r in rows if r["s_global"] > TYPE_A_S_GLOBAL_MIN and r["tau"] < TYPE_A_TAU_MAX]
    return sorted(matches, key=lambda r: -r["s_global"])


def type_a_aggregate(entries: list[dict]) -> dict:
    s_global = [e["s_global"] for e in entries]
    tau = [e["tau"] for e in entries]
    s_local = [e["s_local"] for e in entries]
    return {
        "n": len(entries),
        "s_global_mean": float(np.mean(s_global)),
        "s_global_range": [min(s_global), max(s_global)],
        "tau_mean": float(np.mean(tau)),
        "tau_range": [min(tau), max(tau)],
        "s_local_mean": float(np.mean(s_local)),
        "s_local_range": [min(s_local), max(s_local)],
        "definition": (
            f"s_global > {TYPE_A_S_GLOBAL_MIN} AND tau < {TYPE_A_TAU_MAX} "
            "(using bidirectional conservative tau = min(tau_AB, tau_BA))"
        ),
    }


def type_b_entries(rows: list[dict]) -> list[dict]:
    # Preserve atlas.json's natural pairwise iteration order (no re-sorting) --
    # this is what reproduces the original file's entry order exactly.
    return [r for r in rows if r["s_global"] < TYPE_B_S_GLOBAL_MAX and r["tau"] > TYPE_B_TAU_MIN]


def type_c_entries(rows: list[dict]) -> list[dict]:
    """Top-K pairs-of-pairs with near-equal s_global but maximally different tau."""
    candidates = []
    for r1, r2 in itertools.combinations(rows, 2):
        if abs(r1["s_global"] - r2["s_global"]) <= TYPE_C_DELTA_S_GLOBAL_MAX:
            candidates.append((abs(r1["tau"] - r2["tau"]), r1, r2))
    candidates.sort(key=lambda c: -c[0])
    entries = []
    for gap, r1, r2 in candidates[:TYPE_C_TOP_K]:
        entries.append(
            {
                "pair1": {"name": r1["pair"], "cka": r1["s_global"], "tau": r1["tau"]},
                "pair2": {"name": r2["pair"], "cka": r2["s_global"], "tau": r2["tau"]},
                "tau_gap": gap,
            }
        )
    return entries


# The original file's 8-example "type_a_high_cka_low_tau" shortlist, copied verbatim
# for comparison purposes only (this script never overwrites that file). See the
# module docstring: 7 of these 8 are explained by a 2-part composite rule; the 8th is
# not explained by anything found so far.
ORIGINAL_REPRESENTATIVE_8 = {
    "beit_large_patch16_224 <-> vit_large_patch16_224.mae",
    "beit_base_patch16_224 <-> vit_large_patch16_224.mae",
    "clip_vitl14 <-> convnext_base",
    "clip_vitl14 <-> swin_base_patch4_window7_224",
    "clip_vitl14 <-> convnext_large",
    "clip_vitl14 <-> vit_large_patch16_224",
    "dinov2_vitb14 <-> swin_base_patch4_window7_224",
    "clip_vitl14 <-> dinov2_vits14",
}


def representative_type_a_partial_explanation(type_a: list[dict], type_c: list[dict]) -> dict:
    """Check how much of the original 8-pair representative shortlist is explained.

    Composite rule found by inspection (see module docstring): the top-3 type-A pairs
    by s_global, restricted to the 19 type-A pairs with tau < 0.1, union the type-A
    pairs used as the low-tau anchor in the first 4 entries of type_c (the 5th type_c
    entry's type-A member, convnext_base <-> vit_large_patch16_224, is NOT part of the
    original shortlist, so it is deliberately excluded here).
    """
    low_tau_19 = [r for r in type_a if r["tau"] < 0.1]
    top3_by_s_global = {
        r["pair"] for r in sorted(low_tau_19, key=lambda r: -r["s_global"])[:3]
    }
    type_c_low_tau_anchors = {entry["pair1"]["name"] for entry in type_c[:4]}
    composite = top3_by_s_global | type_c_low_tau_anchors

    return {
        "rule": (
            "top-3 type-A pairs by s_global among the 19 with tau<0.1, UNION the "
            "pair1 (low-tau anchor) of the first 4 type_c_similar_cka_different_tau "
            "entries"
        ),
        "composite_pairs": sorted(composite),
        "matches_original_8_list": sorted(composite & ORIGINAL_REPRESENTATIVE_8),
        "original_pairs_not_explained": sorted(ORIGINAL_REPRESENTATIVE_8 - composite),
        "composite_pairs_not_in_original": sorted(composite - ORIGINAL_REPRESENTATIVE_8),
    }


def main() -> None:
    atlas = json.loads(ATLAS_PATH.read_text())
    rows = load_pairs(atlas)

    tau_arr = np.array([r["tau"] for r in rows])
    s_global_arr = np.array([r["s_global"] for r in rows])
    rho, p_value = spearmanr(tau_arr, s_global_arr)

    type_a = type_a_entries(rows)
    type_b = type_b_entries(rows)
    type_c = type_c_entries(rows)
    cross_family = sum(1 for r in type_a if not r["same_family"])
    cross_objective = sum(1 for r in type_a if not r["same_objective"])

    result = {
        "description": (
            "tau vs CKA (s_global) divergence cases -- where they disagree. "
            "Regenerated from data/atlas.json by scripts/followup/cka_tau_divergence.py "
            "(see module docstring for the thresholds and type-C matching rule)."
        ),
        "n_total_pairs": len(rows),
        "type_a_high_cka_low_tau_35": [
            {
                "pair": r["pair"],
                "model_a": r["model_a"],
                "model_b": r["model_b"],
                "s_global": r["s_global"],
                "tau": r["tau"],
                "s_local": r["s_local"],
                "delta": r["delta"],
                "regime": r["regime"],
                "same_family": r["same_family"],
                "same_objective": r["same_objective"],
            }
            for r in type_a
        ],
        "type_a_aggregate": type_a_aggregate(type_a),
        "type_a_cross_family": cross_family,
        "type_a_cross_objective": cross_objective,
        "type_b_low_cka_high_tau": [
            {
                "pair": r["pair"],
                "cka": r["s_global"],
                "tau": r["tau"],
                "regime": r["regime"],
            }
            for r in type_b
        ],
        "type_b_definition": f"s_global < {TYPE_B_S_GLOBAL_MAX} AND tau > {TYPE_B_TAU_MIN}",
        "type_c_similar_cka_different_tau": type_c,
        "type_c_definition": (
            f"top-{TYPE_C_TOP_K} pairs-of-pairs (over all {len(rows)} choose 2 combinations) "
            f"with |s_global_i - s_global_j| <= {TYPE_C_DELTA_S_GLOBAL_MAX}, ranked by "
            "|tau_i - tau_j| descending"
        ),
        "spearman_tau_cka": float(rho),
        "spearman_tau_cka_pvalue": float(p_value),
        "variance_tau_unexplained_by_cka": float(1 - rho**2),
        "representative_type_a_high_cka_low_tau_8": None,
        "representative_type_a_partial_explanation": representative_type_a_partial_explanation(
            type_a, type_c
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"type A: n={len(type_a)}, cross_family={cross_family}, cross_objective={cross_objective}")
    print(f"type B: n={len(type_b)}")
    print(f"type C: n={len(type_c)}")
    print(f"spearman(tau, s_global)={rho:.4f}, 1-rho^2={1 - rho**2:.4f}")


if __name__ == "__main__":
    main()
