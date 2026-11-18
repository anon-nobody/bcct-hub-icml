"""Model-level permutation test for the ImageNet-1k architecture comparison.

Pairs share models, so the pair-level Mann-Whitney test in in1k_arch_test.py is anti-conservative.
Here the statistic mean(tau | same arch) - mean(tau | cross arch) is compared with its distribution
when the architecture labels are permuted across the 12 models (group sizes preserved).
Reads data/experiments/in1k_arch_test.json and writes the result back into its "summary".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[2] / "data/experiments/in1k_arch_test.json"


def stat(rows: list[dict], lab: dict[str, str]) -> float:
    same = [r["tau"] for r in rows if lab[r["a"]] == lab[r["b"]]]
    cross = [r["tau"] for r in rows if lab[r["a"]] != lab[r["b"]]]
    return float(np.mean(same) - np.mean(cross))


def main() -> None:
    with open(P) as f:
        d = json.load(f)
    models = list(d["models"])
    rng = np.random.default_rng(0)
    out = {}
    for tag, rows in [("all", d["pairs"]), ("mlp_fit_ok", [r for r in d["pairs"] if not r["mlp_underfit"]])]:
        obs = stat(rows, d["models"])
        labels = [d["models"][m] for m in models]
        null = [stat(rows, dict(zip(models, rng.permutation(labels), strict=True))) for _ in range(10000)]
        p = (1 + sum(n >= obs for n in null)) / (1 + len(null))
        out[tag] = {"diff_same_minus_cross": round(obs, 4), "perm_p_one_sided": round(p, 4), "n_perm": 10000}
        print(tag, out[tag])
    d["summary"]["model_level_permutation"] = out
    with open(P, "w") as f:
        json.dump(d, f, indent=1)


if __name__ == "__main__":
    main()
