# Supplementary code and data

Code, result files, and atlases for *What Makes Two Encoders Linearly Compatible? Dependence-Aware
Measurement and Controlled Evidence* (anonymous submission). Every number in the paper is read from
a JSON file in `data/` that the listed script writes.

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # metrics library (src/latent) + ruff, pytest
pip install -e ".[full]"         # + torchvision, timm/transformers, matplotlib (experiments, figures)
ruff check src/ tests/ scripts/
pytest                           # 65 tests
```

`requirements-pinned.txt` lists the exact versions used. Scripts locate the repository root from
their own path, so run them from anywhere.

## The six quantities (Section 3)

`src/latent/metrics/`: `bitrate.py` (R, Δ), `alignment.py` (S_local with permutation null, CKA,
λ), `transport.py` (τ: worst-direction ratio of held-out Procrustes R² to MLP R², 80/20 split;
TAI), `regime.py` (regime rule). One-pair report:

```bash
bcct-scorecard --model-features A.pt --ref-features B.pt --format markdown
```

## Where each result comes from

| Paper | Script | Result file (`data/`) |
|---|---|---|
| Vision atlas, 190 pairs | `scripts/extract_features.py`, `scripts/compute_alignment_atlas.py` | `atlas.json` |
| Language atlas, 36 pairs | `scripts/llm_experiment.py` | `llm_atlas.json` |
| Dyadic-robust regression (§4–5) | `scripts/followup/dyadic_regression.py` | `experiments/dyadic_regression.json` |
| Family vs. objective/recipe, LOFO | `scripts/followup/B_deconfound.py`, `scripts/followup/lineage_analysis.py`, `scripts/holdout_predictive_validation.py` | `experiments/B_deconfound.json`, `lineage_analysis.json`, `holdout_validation.json` |
| CKA vs. τ (35 high-CKA, low-τ pairs) | `scripts/followup/cka_tau_divergence.py` | `experiments/tau_vs_cka_divergence.json` |
| Debiased (unbiased-HSIC) CKA | `scripts/followup/debiased_cka_robustness.py` | `experiments/debiased_cka_robustness.json` |
| τ probe sensitivity (ridge, 3×3 MLP grid) | `scripts/followup/probe_sensitivity.py` | `experiments/probe_sensitivity.json` |
| CIFAR-100 one-factor contrasts (Fig. 2a) | `scripts/followup/train_controlled.py`, `run_min_tier_strong.py`, `run_seed_ci.py`, `run_seed_ci_arch.py` | `experiments/seed_ci_results.json`, `seed_ci_arch_results.json`, `min_tier_strong_results.json` |
| Shared-data curves (Fig. 2b) | `run_data_overlap.py`, `run_data_overlap_seeds.py`, `run_data_overlap_tin.py`, `run_overlap_simclr_v2.py` | `experiments/data_overlap_seeds.json`, `overlap_tin_curve.json`, `overlap_{sup_r18,simclr_v2}_{curve,mid,zero}.json`, `overlap_sup_r18_strongaug_zero.json` |
| Tiny-ImageNet 64×64 | `run_tin64_factorial.py`, `tin64_r2_diag.py`, `run_tin64_arch2.py` | `experiments/tin64_factorial_results.json`, `tin64_arch2_results.json` |
| Food-101 128×128 | `run_food128_factorial.py` | `experiments/food128_results.json` |
| Public ImageNet-1k checkpoints | `in1k_arch_test.py`, `in1k_arch_permtest.py`, `in1k_factor_test.py`, `in1k_effective_rank.py` | `experiments/in1k_arch_test.json`, `in1k_factor_test.json`, `in1k_effective_rank.json` |
| Downstream retrieval | `scripts/retrieval_downstream.py`, `scripts/followup/retrieval_190_diag.py` | `experiments/retrieval_full_atlas_diag.json` |
| Robustness (probe, k, sample size, STL-10, estimators) | `scripts/{probe_robustness,knn_sensitivity,sample_size_stability,stl10_robustness,bitrate_estimator_robustness}.py` | matching JSON in `data/` |
| Figure 2 | `scripts/figures/fig_validation.py` | `paper/figures/fig_validation.pdf` |

Scripts without a directory are in `scripts/followup/`. Training scripts need the datasets
(CIFAR-100, Tiny-ImageNet, Food-101 via torchvision) under `data/`; raw images and extracted
feature tensors are not included in this archive because of size. Analysis scripts that only read
`data/atlas.json` or result JSONs run on CPU in minutes.

## Data

For each model pair, `data/atlas.json` lists S_local, S_global, λ, τ (both directions
and their minimum), TAI, Δ, the Procrustes and MLP R² behind τ, and the regime label; per-model
bitrate, effective rank, family, and objective are under `models`.

## License

Apache 2.0 (`LICENSE`) for code, atlases, and result files. Pretrained checkpoints and datasets
keep their own licenses.
