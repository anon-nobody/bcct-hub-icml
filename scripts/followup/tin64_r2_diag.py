"""Add per-direction Procrustes/MLP R^2 to every contrast pair of the Tiny-ImageNet factorial.

tau alone cannot tell "linear fails, nonlinear works" from "nothing transfers"; this records
both R^2 values per direction and summarizes them per contrast.
Reads data/experiments/tin64_factorial_feats/*.pt, updates tin64_factorial_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent.metrics.transport import transport_linearity_score as tls  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
D = ROOT / "data/experiments/tin64_factorial_feats"
OUT = ROOT / "data/experiments/tin64_factorial_results.json"


def load(n: str) -> torch.Tensor:
    return torch.load(D / f"{n}.pt", weights_only=False)["features"].to(DEV)


def main() -> None:
    f = {f"{g}_s{s}": load(f"{g}_s{s}") for g in ("ref", "data", "arch", "obj") for s in range(3)}
    pairs = {"C-init": [(f"ref_s{i}", f"ref_s{j}") for i in range(3) for j in range(i + 1, 3)]}
    for g in ("data", "arch", "obj"):
        pairs[f"C-{g}"] = [(f"ref_s{i}", f"{g}_s{j}") for i in range(3) for j in range(3)]
        pairs[f"{g}-init"] = [(f"{g}_s{i}", f"{g}_s{j}") for i in range(3) for j in range(i + 1, 3)]
    out = {}
    for k, ps in pairs.items():
        rows = []
        for a, b in ps:
            ab, ba = tls(f[a], f[b], mlp_steps=3000), tls(f[b], f[a], mlp_steps=3000)
            rows.append({"a": a, "b": b, "tau": round(min(ab.tau, ba.tau), 4),
                         "lin": [round(ab.perf_linear, 4), round(ba.perf_linear, 4)],
                         "mlp": [round(ab.perf_best, 4), round(ba.perf_best, 4)],
                         "underfit": bool(ab.perf_best < ab.perf_linear or ba.perf_best < ba.perf_linear)})
            print(k, rows[-1], flush=True)
        out[k] = {"pairs": rows,
                  "mean_lin_r2": round(float(np.mean([x for r in rows for x in r["lin"]])), 3),
                  "mean_mlp_r2": round(float(np.mean([x for r in rows for x in r["mlp"]])), 3),
                  "max_mlp_r2": round(float(np.max([x for r in rows for x in r["mlp"]])), 3),
                  "n_underfit": sum(r["underfit"] for r in rows)}
    with open(OUT) as fh:
        res = json.load(fh)
    res["r2_diagnostics"] = out
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != "pairs"} for k, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
