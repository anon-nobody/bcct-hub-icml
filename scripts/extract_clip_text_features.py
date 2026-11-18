"""Extract CLIP text encoder features for cross-modal transport analysis.

Uses CIFAR-100 class templates ("a photo of a {class}") to generate
text embeddings comparable to vision features.

Usage:
    python scripts/extract_clip_text_features.py [--device cuda:0]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

logger = logging.getLogger(__name__)

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="data/features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = args.device

    for clip_name, hf_name in [
        ("clip_vitb16_text", "openai/clip-vit-base-patch16"),
        ("clip_vitl14_text", "openai/clip-vit-large-patch14"),
    ]:
        logger.info("Extracting %s from %s", clip_name, hf_name)

        text_model = CLIPTextModelWithProjection.from_pretrained(hf_name).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(hf_name)

        all_embeddings = []

        text_model.eval()
        with torch.no_grad():
            for cls_name in CIFAR100_CLASSES:
                texts = [t.format(cls_name.replace("_", " ")) for t in TEMPLATES]
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = text_model(**inputs)
                text_embeds = outputs.text_embeds  # (n_templates, dim)
                # L2 normalize then average
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                avg_embed = text_embeds.mean(dim=0)
                avg_embed = avg_embed / avg_embed.norm()
                all_embeddings.append(avg_embed)

        # Stack: (100, dim)
        text_features = torch.stack(all_embeddings).cpu()
        logger.info("Text features shape: %s", text_features.shape)

        # To match vision atlas format (5000 samples), repeat each class 50 times
        # (CIFAR-100 test set has 50 samples per class)
        text_features_expanded = text_features.repeat_interleave(50, dim=0)  # (5000, dim)
        logger.info("Expanded shape: %s", text_features_expanded.shape)

        # Save
        output_path = Path(args.output_dir) / f"{clip_name}.pt"
        torch.save({
            "features": text_features_expanded,
            "model_name": clip_name,
            "model_config": {"type": "text_encoder", "hf_name": hf_name},
        }, output_path)
        logger.info("Saved to %s", output_path)

        del text_model
        torch.cuda.empty_cache()

    # Now compute cross-modal transport
    logger.info("Computing cross-modal transport...")
    from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
    from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
    from latent.metrics.regime import classify_regime
    from latent.metrics.transport import transport_linearity_score

    results = []
    cross_pairs = [
        ("clip_vitb16", "clip_vitb16_text"),
        ("clip_vitl14", "clip_vitl14_text"),
        ("dinov2_vitb14", "clip_vitb16_text"),  # cross-model cross-modal
        ("resnet50", "clip_vitb16_text"),  # supervised vision vs CLIP text
    ]

    for model_a, model_b in cross_pairs:
        logger.info("Cross-modal: %s <-> %s", model_a, model_b)
        try:
            fa = torch.load(Path(args.output_dir) / f"{model_a}.pt", weights_only=True)["features"].to(device)
            fb = torch.load(Path(args.output_dir) / f"{model_b}.pt", weights_only=True)["features"].to(device)
        except FileNotFoundError as e:
            logger.warning("Skip %s: %s", model_a, e)
            continue

        n = min(fa.shape[0], fb.shape[0])
        fa, fb = fa[:n], fb[:n]

        s_local, _, _ = null_calibrated_mutual_knn(fa, fb, k=10, n_permutations=100, seed=42)
        s_global = centered_kernel_alignment(fa, fb)
        r_a = representation_bitrate(fa).bitrate
        r_b = representation_bitrate(fb).bitrate
        delta = bottleneck_mismatch(fa, fb)
        transport = transport_linearity_score(fa, fb, mlp_steps=3000, seed=42)

        regime = classify_regime(bitrate=(r_a + r_b) / 2, s_local=s_local, tau=transport.tau)

        result = {
            "pair": f"{model_a} <-> {model_b}",
            "s_local": round(s_local, 4),
            "s_global": round(s_global, 4),
            "tau": round(transport.tau, 4),
            "delta": round(delta, 3),
            "regime": regime.regime.value,
            "cross_modal": True,
        }
        logger.info("  %s", result)
        results.append(result)

    # Save cross-modal results
    output = {"pairs": results}
    with open("data/experiments/cross_modal_clip.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Cross-modal results saved")


if __name__ == "__main__":
    main()
