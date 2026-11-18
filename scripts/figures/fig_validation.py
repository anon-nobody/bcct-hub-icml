"""Figure: ground-truth validation of transport linearity (tau) with controlled encoders.

(a) One-factor-apart contrasts (ViT, CIFAR-100, 100 epochs), all with 3 seeds per encoder: tau between a reference encoder and
    an encoder that differs in exactly one training factor. Error bars: 95% CI over seeds where
    available (seed_ci_results.json); single-seed contrasts are marked.
(b) tau vs. fraction of shared training classes: supervised ViT on CIFAR-100 and Tiny-ImageNet,
    supervised and SimCLR ResNet-18 on CIFAR-100 (same windows, seeds, and probe). CIFAR curves: mean over
    seed pairs with min-max range (data_overlap_seeds, overlap_*_mid, overlap_*_zero).

Inputs: data/experiments/{seed_ci_results, seed_ci_arch_results, data_overlap_seeds, overlap_tin_curve,
overlap_sup_r18_mid, overlap_sup_r18_zero, overlap_simclr_v2_mid, overlap_simclr_v2_zero}.json. Output: paper/figures/fig_validation.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "data/experiments"


def load(name: str) -> dict:
    with open(EXP / name) as f:
        return json.load(f)


def main() -> None:
    ci = load("seed_ci_results.json")["contrasts"]
    tin = sorted(load("overlap_tin_curve.json")["curve"])

    arch = load("seed_ci_arch_results.json")["contrasts"]

    def ci_of(d: dict, k: str) -> tuple[float, list[float]]:
        return d[k]["mean"], d[k]["ci95"]

    bars = [  # label, mean, (lo, hi) or None
        ("seed", *ci_of(ci, "C-init (same recipe, diff seed)")),
        ("width\n512→64", *ci_of(arch, "C-w64 (B0 vs w64)")),
        ("width\n512→32", *ci_of(arch, "C-w32 (B0 vs w32)")),
        ("architecture\nViT→ResNet", *ci_of(arch, "C-arch (B0 vs ResNet)")),
        ("data", ci["C-data (data swap)"]["mean"], ci["C-data (data swap)"]["ci95"]),
        ("objective", ci["C-obj (objective swap)"]["mean"], ci["C-obj (objective swap)"]["ci95"]),
    ]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.35, 1]})

    xs = range(len(bars))
    colors = ["#2e8b57", "#8fbc8f", "#8fbc8f", "#b0b0b0", "#c0504d", "#c0504d"]
    for x, (_lab, m, c), col in zip(xs, bars, colors, strict=True):
        err = None if c is None else [[m - c[0]], [c[1] - m]]
        ax1.bar(x, m, color=col, width=0.7, yerr=err, capsize=2, error_kw={"lw": 0.8})
        ax1.text(x, m + 0.02, f"{m:.2f}" + ("" if c else "*"), ha="center", fontsize=7)
    ax1.set_xticks(list(xs), [b[0] for b in bars], fontsize=6.5)
    ax1.set_ylabel(r"$\tau$ to reference encoder")
    ax1.set_ylim(0, 0.72)
    ax1.set_title("(a)", fontsize=8, loc="left")

    def seeded(mid: str, zero: str, ref100: float) -> tuple[list, list, list, list]:
        """Mean and min-max range over seed pairs at 0/26/50/74%; 100% from the reference-seed pair."""
        m, z = load(mid)["windows"], load(zero)
        xs, ys, lo, hi = [100], [ref100], [0.0], [0.0]
        for w in ("74", "50", "26"):
            xs.append(int(w))
            ys.append(m[w]["tau_mean"])
            lo.append(m[w]["tau_mean"] - m[w]["tau_range"][0])
            hi.append(m[w]["tau_range"][1] - m[w]["tau_mean"])
        xs.append(0)
        ys.append(z["tau_0pct_mean"])
        lo.append(z["tau_0pct_mean"] - z["tau_0pct_range"][0])
        hi.append(z["tau_0pct_range"][1] - z["tau_0pct_mean"])
        return xs, ys, lo, hi

    vs = load("data_overlap_seeds.json")["summary"]
    v_x = [100, 74, 50, 26, 0]
    v_y = [vs[str(k)]["tau_mean"] for k in v_x]
    v_lo = [vs[str(k)]["tau_mean"] - vs[str(k)]["tau_range"][0] for k in v_x]
    v_hi = [vs[str(k)]["tau_range"][1] - vs[str(k)]["tau_mean"] for k in v_x]
    r18_ref = [p["tau"] for p in load("overlap_sup_r18_zero.json")["pairs"] if p["overlap"] == 100][0]
    ssl_ref = [p["tau"] for p in load("overlap_simclr_v2_zero.json")["pairs"] if p["overlap"] == 100][0]
    r18 = seeded("overlap_sup_r18_mid.json", "overlap_sup_r18_zero.json", r18_ref)
    ssl = seeded("overlap_simclr_v2_mid.json", "overlap_simclr_v2_zero.json", ssl_ref)
    kw = {"lw": 1.5, "ms": 4, "capsize": 2, "elinewidth": 0.8}
    ax2.errorbar(v_x, v_y, yerr=[v_lo, v_hi], fmt="o-", color="#2e8b57", label="sup. ViT, CIFAR-100", **kw)
    ax2.plot(*zip(*tin, strict=True), "s--", color="#3b6fb0", lw=1.5, ms=4, label="sup. ViT, Tiny-ImageNet (1 seed)")
    ax2.errorbar(r18[0], r18[1], yerr=[r18[2], r18[3]], fmt="^-", color="#c0504d", label="sup. ResNet-18, CIFAR-100", **kw)
    ax2.errorbar(ssl[0], ssl[1], yerr=[ssl[2], ssl[3]], fmt="D-", color="#8064a2", label="SimCLR ResNet-18, CIFAR-100", **kw)
    ax2.set_xlabel("shared training classes (%)")
    ax2.set_ylabel(r"$\tau$ to reference encoder")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(frameon=False, fontsize=6, loc="upper left")
    ax2.set_title("(b)", fontsize=8, loc="left")
    fig.tight_layout()
    out = ROOT / "paper/figures/fig_validation.pdf"
    fig.savefig(out, bbox_inches="tight")
    # ICML accepts Type-1 fonts only; matplotlib embeds Type 3, so convert text to outlines.
    import subprocess

    subprocess.run(["bash", str(ROOT / "scripts/figures/outline_fonts.sh"), str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
