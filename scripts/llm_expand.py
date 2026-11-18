# pyright: reportMissingTypeStubs=none, reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownParameterType=none, reportUnknownArgumentType=none, reportMissingTypeArgument=none, reportArgumentType=none
from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LLM_MODELS: list[dict[str, str | int]] = [
    {
        "name": "pythia-410m",
        "hf_id": "EleutherAI/pythia-410m",
        "family": "pythia",
        "params": "410M",
        "layers": 24,
        "dim": 1024,
    },
    {
        "name": "pythia-1.4b",
        "hf_id": "EleutherAI/pythia-1.4b",
        "family": "pythia",
        "params": "1.4B",
        "layers": 24,
        "dim": 2048,
    },
    {
        "name": "pythia-2.8b",
        "hf_id": "EleutherAI/pythia-2.8b",
        "family": "pythia",
        "params": "2.8B",
        "layers": 32,
        "dim": 2560,
    },
    {
        "name": "opt-1.3b",
        "hf_id": "facebook/opt-1.3b",
        "family": "opt",
        "params": "1.3B",
        "layers": 24,
        "dim": 2048,
    },
    {
        "name": "phi-2",
        "hf_id": "microsoft/phi-2",
        "family": "phi",
        "params": "2.7B",
        "layers": 32,
        "dim": 2560,
    },
    {
        "name": "gpt2-large",
        "hf_id": "openai-community/gpt2-large",
        "family": "gpt2",
        "params": "774M",
        "layers": 36,
        "dim": 1280,
    },
    {
        "name": "pythia-6.9b",
        "hf_id": "EleutherAI/pythia-6.9b",
        "family": "pythia",
        "params": "6.9B",
        "layers": 32,
        "dim": 4096,
    },
    {
        "name": "mistral-7b",
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "family": "mistral",
        "params": "7.2B",
        "layers": 32,
        "dim": 4096,
    },
    {
        "name": "falcon-7b",
        "hf_id": "tiiuae/falcon-7b",
        "family": "falcon",
        "params": "7.2B",
        "layers": 32,
        "dim": 4544,
    },
    {
        "name": "pythia-12b",
        "hf_id": "EleutherAI/pythia-12b",
        "family": "pythia",
        "params": "12B",
        "layers": 36,
        "dim": 5120,
    },
    {
        "name": "llama-3.1-8b",
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "family": "llama",
        "params": "8B",
        "layers": 32,
        "dim": 4096,
    },
]

NEW_MODELS = {"pythia-12b", "llama-3.1-8b"}


def get_layer_indices(n_layers: int, n_points: int = 5) -> list[int]:
    if n_layers <= n_points:
        return list(range(n_layers))
    indices = [int(i * (n_layers - 1) / (n_points - 1)) for i in range(n_points)]
    return sorted(set(indices))


def resolve_llama_hf_id(preferred: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    candidates = [
        preferred,
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B",
    ]
    for model_id in candidates:
        try:
            api.model_info(model_id)
            logger.info("Resolved Llama-3.1-8B model ID: %s", model_id)
            return model_id
        except Exception:
            continue
    logger.warning("Could not validate Llama model ID. Falling back to %s", preferred)
    return preferred


def _model_device_map(device: str) -> str | None:
    if device == "auto":
        return "auto"
    return None


def _batch_device(model: torch.nn.Module, fallback: str) -> torch.device | str:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def extract_llm_features(
    model_spec: dict[str, str | int],
    n_samples: int,
    max_length: int,
    device: str,
    output_dir: Path,
) -> dict[int, Tensor]:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_file = output_dir / f"{model_spec['name']}.pt"
    if output_file.exists():
        logger.info("Loading cached features: %s", output_file)
        data = torch.load(output_file, weights_only=True)
        return data["layer_features"]

    hf_id = str(model_spec["hf_id"])
    if model_spec["name"] == "llama-3.1-8b":
        hf_id = resolve_llama_hf_id(hf_id)
        model_spec["hf_id"] = hf_id

    logger.info("Loading model: %s (%s)", model_spec["name"], hf_id)
    # Phi-2 and Falcon use custom model code not yet merged into transformers.
    _needs_remote_code = model_spec.get("family") in {"phi", "falcon"}
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=_needs_remote_code,  # Required: model uses custom code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map=_model_device_map(device),
        trust_remote_code=_needs_remote_code,  # Required: model uses custom code
        output_hidden_states=True,
    )
    if device != "auto" and _model_device_map(device) is None:
        model = model.to(device)
    model.eval()

    logger.info("Loading WikiText-103 test split...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][: n_samples * 2]

    logger.info("Tokenizing %d texts (max_length=%d)", len(texts), max_length)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    n_layers = int(model_spec["layers"])
    layer_indices = get_layer_indices(n_layers=n_layers, n_points=5)
    logger.info("Extracting layers: %s", layer_indices)

    layer_features: dict[int, list[Tensor]] = {li: [] for li in layer_indices}
    batch_size = 16
    collected = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size), desc=f"Extracting {model_spec['name']}"
        ):
            if collected >= n_samples:
                break
            end = min(start + batch_size, len(texts))

            batch_target = _batch_device(model, device)
            batch_ids = encodings["input_ids"][start:end].to(batch_target)
            batch_mask = encodings["attention_mask"][start:end].to(batch_target)

            try:
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            except Exception as exc:
                logger.warning("Batch %d failed for %s: %s", start, model_spec["name"], exc)
                continue

            hidden_states = outputs.hidden_states
            for li in layer_indices:
                hs = hidden_states[li].float()
                mask = batch_mask.unsqueeze(-1).float()
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                layer_features[li].append(pooled.cpu())

            collected += end - start

    result: dict[int, Tensor] = {}
    for li in layer_indices:
        if layer_features[li]:
            feat = torch.cat(layer_features[li], dim=0)[:n_samples]
            result[li] = feat
            logger.info("  Layer %d -> %s", li, tuple(feat.shape))

    torch.save(
        {
            "layer_features": result,
            "model_spec": model_spec,
            "n_samples": n_samples,
            "max_length": max_length,
            "layer_indices": layer_indices,
        },
        output_file,
    )
    logger.info("Saved features to %s", output_file)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def compute_llm_bcct_metrics(
    features_dict: dict[str, dict[int, Tensor]],
    model_specs: dict[str, dict[str, str | int]],
    device: str,
) -> dict[str, dict]:
    from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
    from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
    from latent.metrics.regime import classify_regime
    from latent.metrics.transport import transport_asymmetry_index

    device_for_metrics = (
        device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    results: dict[str, dict] = {"models": {}, "pairwise": {}}

    for name, layer_feats in features_dict.items():
        spec = model_specs[name]
        rep_layer = get_layer_indices(int(spec["layers"]), 5)[3]
        if rep_layer not in layer_feats:
            rep_layer = max(layer_feats.keys())

        feats = layer_feats[rep_layer].to(device_for_metrics)
        br = representation_bitrate(feats)
        results["models"][name] = {
            "bitrate": round(float(br.bitrate), 3),
            "effective_rank": round(float(br.effective_rank_value), 1),
            "dim": int(feats.shape[1]),
            "n_samples": int(feats.shape[0]),
            "rep_layer": int(rep_layer),
            "family": str(spec["family"]),
            "params": str(spec["params"]),
        }

    model_names = sorted(features_dict.keys())
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if j <= i:
                continue

            spec_a = model_specs[name_a]
            spec_b = model_specs[name_b]
            rep_a = get_layer_indices(int(spec_a["layers"]), 5)[3]
            rep_b = get_layer_indices(int(spec_b["layers"]), 5)[3]
            if rep_a not in features_dict[name_a]:
                rep_a = max(features_dict[name_a].keys())
            if rep_b not in features_dict[name_b]:
                rep_b = max(features_dict[name_b].keys())

            fa = features_dict[name_a][rep_a].to(device_for_metrics)
            fb = features_dict[name_b][rep_b].to(device_for_metrics)
            n = min(fa.shape[0], fb.shape[0])
            fa = fa[:n]
            fb = fb[:n]

            logger.info("Computing pair: %s <-> %s", name_a, name_b)
            s_local, _raw_overlap, _null_mean = null_calibrated_mutual_knn(
                fa, fb, k=10, n_permutations=100, seed=42
            )
            s_global = centered_kernel_alignment(fa, fb)
            delta = bottleneck_mismatch(fa, fb)
            transport = transport_asymmetry_index(fa, fb, mlp_steps=500, seed=42)

            tau = min(transport.a_to_b.tau, transport.b_to_a.tau)
            lambda_val = s_local / s_global if s_global > 1e-10 else float("inf")
            r_a = representation_bitrate(fa).bitrate
            r_b = representation_bitrate(fb).bitrate
            regime = classify_regime(
                bitrate=(r_a + r_b) / 2,
                s_local=s_local,
                tau=tau,
                s_global=s_global,
                lambda_val=lambda_val,
            ).regime.value

            pair_key = f"{name_a} <-> {name_b}"
            results["pairwise"][pair_key] = {
                "s_local": round(float(s_local), 4),
                "s_global": round(float(s_global), 4),
                "lambda": round(float(lambda_val), 4),
                "tau": round(float(tau), 4),
                "tau_a_to_b": round(float(transport.a_to_b.tau), 4),
                "tau_b_to_a": round(float(transport.b_to_a.tau), 4),
                "tai": round(float(transport.tai), 4),
                "delta": round(float(delta), 3),
                "regime": regime,
                "family_a": str(spec_a["family"]),
                "family_b": str(spec_b["family"]),
            }

    return results


def generate_pythia_scaling_plot(results: dict[str, dict], fig_dir: Path) -> Path:
    pythia_models = [
        "pythia-410m",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]
    rows: list[tuple[float, float, str]] = []

    for pair_key, metrics in results["pairwise"].items():
        name_a, name_b = pair_key.split(" <-> ")
        if name_a not in pythia_models or name_b not in pythia_models:
            continue
        bitrate_a = results["models"][name_a]["bitrate"]
        bitrate_b = results["models"][name_b]["bitrate"]
        avg_bitrate = 0.5 * (float(bitrate_a) + float(bitrate_b))
        rows.append((avg_bitrate, float(metrics["tau"]), pair_key))

    if not rows:
        logger.warning("No Pythia scaling rows available; skipping plot.")
        return fig_dir / "fig15_pythia_scaling_tau_vs_bitrate.pdf"

    rows = sorted(rows, key=lambda x: x[0])
    x = np.array([r[0] for r in rows], dtype=np.float32)
    y = np.array([r[1] for r in rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(x, y, c="#1976D2", s=55)
    ax.plot(x, y, color="#1976D2", alpha=0.7)

    for xv, yv, label in rows:
        ax.annotate(
            label.replace(" <-> ", " vs "),
            (xv, yv),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    ax.set_xlabel("Average pair bitrate (bits/dim)")
    ax.set_ylabel("Transport linearity tau")
    ax.set_title("Pythia scaling: tau vs bitrate")
    ax.grid(True, alpha=0.3)

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "fig15_pythia_scaling_tau_vs_bitrate.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pythia scaling plot: %s", out_path)
    return out_path


def _count_new_regimes(pairwise: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair_key, metrics in pairwise.items():
        name_a, name_b = pair_key.split(" <-> ")
        if name_a not in NEW_MODELS and name_b not in NEW_MODELS:
            continue
        regime = str(metrics["regime"])
        counts[regime] = counts.get(regime, 0) + 1
    return counts


def _pythia_scaling_statement(results: dict[str, dict]) -> str:
    pythia_pairs: list[tuple[float, float]] = []
    for pair_key, metrics in results["pairwise"].items():
        name_a, name_b = pair_key.split(" <-> ")
        if not (name_a.startswith("pythia-") and name_b.startswith("pythia-")):
            continue
        r_a = float(results["models"][name_a]["bitrate"])
        r_b = float(results["models"][name_b]["bitrate"])
        pythia_pairs.append(((r_a + r_b) / 2.0, float(metrics["tau"])))

    if len(pythia_pairs) < 2:
        return "insufficient Pythia pairs for scaling trend check"

    arr = np.array(pythia_pairs)
    corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    if np.isnan(corr):
        return "scaling trend undefined (NaN correlation)"
    if corr > 0.2:
        return f"supported (corr(tau, bitrate)={corr:.3f})"
    if corr < -0.2:
        return f"not supported (corr(tau, bitrate)={corr:.3f})"
    return f"weak or mixed evidence (corr(tau, bitrate)={corr:.3f})"


def _llama_integration_statement(results: dict[str, dict]) -> str:
    llama_name = "llama-3.1-8b"
    rows = []
    for pair_key, metrics in results["pairwise"].items():
        name_a, name_b = pair_key.split(" <-> ")
        if name_a != llama_name and name_b != llama_name:
            continue
        rows.append(metrics)

    if not rows:
        return "no Llama pairwise rows available"

    convergent = sum(1 for r in rows if r["regime"] == "Convergent")
    divergent = sum(1 for r in rows if r["regime"] == "Divergent")
    avg_tau = float(np.mean([float(r["tau"]) for r in rows]))
    avg_local = float(np.mean([float(r["s_local"]) for r in rows]))

    if convergent >= divergent:
        return (
            "integrates with existing families "
            f"(Convergent={convergent}, Divergent={divergent}, "
            f"avg_tau={avg_tau:.3f}, avg_s_local={avg_local:.3f})"
        )
    return (
        "shows divergence against existing families "
        f"(Convergent={convergent}, Divergent={divergent}, "
        f"avg_tau={avg_tau:.3f}, avg_s_local={avg_local:.3f})"
    )


def _to_serializable(obj: object) -> object:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand LLM BCCT atlas with new models")
    parser.add_argument("--device", default="cuda:0", help="Device or 'auto'")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--features-dir", default="data/llm_features")
    parser.add_argument("--output", default="data/llm_atlas_expanded.json")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    features_dir = root / args.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {m["name"]: dict(m) for m in LLM_MODELS}
    features_dict: dict[str, dict[int, Tensor]] = {}

    for spec in LLM_MODELS:
        model_name = str(spec["name"])
        if args.skip_extraction:
            pt_file = features_dir / f"{model_name}.pt"
            if not pt_file.exists():
                logger.warning("Missing cached feature file: %s", pt_file)
                continue
            data = torch.load(pt_file, weights_only=True)
            features_dict[model_name] = data["layer_features"]
            continue

        feats = extract_llm_features(
            model_spec=model_specs[model_name],
            n_samples=args.n_samples,
            max_length=args.max_length,
            device=args.device,
            output_dir=features_dir,
        )
        features_dict[model_name] = feats

    if len(features_dict) < 2:
        raise RuntimeError("Need at least 2 models with extracted features.")

    logger.info("Computing BCCT metrics for %d models", len(features_dict))
    results = compute_llm_bcct_metrics(features_dict, model_specs, args.device)

    fig_dir = root / "paper" / "figures"
    scaling_plot = generate_pythia_scaling_plot(results, fig_dir)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enriched = {
        "results": results,
        "metadata": {
            "n_models": len(results["models"]),
            "n_pairs": len(results["pairwise"]),
            "new_models": sorted(NEW_MODELS),
            "scaling_plot": str(scaling_plot),
        },
    }
    serializable_results = json.loads(json.dumps(enriched, default=_to_serializable))
    output_path.write_text(json.dumps(serializable_results, indent=2))
    logger.info("Saved expanded atlas: %s", output_path)

    new_regimes = _count_new_regimes(results["pairwise"])
    pythia_statement = _pythia_scaling_statement(results)
    llama_statement = _llama_integration_statement(results)

    print("\n" + "=" * 72)
    print("EXPANDED LLM ATLAS SUMMARY")
    print("=" * 72)
    print(f"Total models: {len(results['models'])}")
    print(f"Total pairs: {len(results['pairwise'])}")
    print("\nNew-pair regime counts (pairs touching pythia-12b or llama-3.1-8b):")
    for regime in ["Convergent", "Local-Only", "Divergent", "Collapsed"]:
        print(f"  {regime}: {new_regimes.get(regime, 0)}")

    print("\nPythia scaling check (higher bitrate -> higher transport):")
    print(f"  {pythia_statement}")

    print("\nLlama-3.1-8B integration check:")
    print(f"  {llama_statement}")


if __name__ == "__main__":
    main()
