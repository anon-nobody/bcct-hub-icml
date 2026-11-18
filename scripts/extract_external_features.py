"""Extract features from external (out-of-atlas) pretrained models on CIFAR-100.

Used for scorecard case study: do BCCT scores predict regime for models
never seen in the 20-model atlas?

Models tested (5): SigLIP, ConvNeXt-V2, EVA02-ViT-B, DeiT-III-B, DINOv2+registers.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_external_features.py \
        --output-dir data/features_external --split test --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoModel, SiglipVisionModel

logger = logging.getLogger(__name__)


# External models: (name, family, objective, loader_kind, source_id, dim)
EXTERNAL_MODELS = [
    (
        "siglip_base_p16_224", "siglip", "contrastive",
        "hf_siglip", "google/siglip-base-patch16-224", 768,
    ),
    (
        "convnextv2_base", "convnextv2", "supervised",
        "timm", "convnextv2_base.fcmae_ft_in22k_in1k", 1024,
    ),
    (
        "eva02_base_p14_224", "eva02", "self-supervised",
        "timm", "eva02_base_patch14_224.mim_in22k", 768,
    ),
    (
        "deit3_base_p16_224", "deit3", "supervised",
        "timm", "deit3_base_patch16_224.fb_in1k", 768,
    ),
    (
        "dinov2_reg_base", "dinov2reg", "self-supervised",
        "hf_automodel", "facebook/dinov2-with-registers-base", 768,
    ),
]


def get_cifar100_loader(split: str, n_samples: int, batch_size: int, image_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    is_train = split == "train"
    dataset = datasets.CIFAR100(
        root="data", train=is_train, download=True, transform=transform
    )
    if n_samples < len(dataset):
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=g)[:n_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


@torch.no_grad()
def extract_timm(model_id: str, loader, device: str) -> torch.Tensor:
    """timm model with num_classes=0 returns pooled features."""
    model = timm.create_model(model_id, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    feats = []
    for images, _ in tqdm(loader, desc=model_id, leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images)
        out = F.normalize(out, dim=-1)
        feats.append(out.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


@torch.no_grad()
def extract_hf_siglip(model_id: str, loader, device: str) -> torch.Tensor:
    """SigLIP vision encoder with projection to image embeddings."""
    model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    feats = []
    for images, _ in tqdm(loader, desc=model_id, leave=False):
        images = images.to(device, non_blocking=True)
        # SigLIP vision: use pooler_output (pooled image representation)
        out = model(pixel_values=images).pooler_output
        out = F.normalize(out, dim=-1)
        feats.append(out.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


@torch.no_grad()
def extract_hf_automodel(model_id: str, loader, device: str) -> torch.Tensor:
    """Generic HF AutoModel: use pooler_output or CLS token from last_hidden_state."""
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    feats = []
    for images, _ in tqdm(loader, desc=model_id, leave=False):
        images = images.to(device, non_blocking=True)
        out = model(pixel_values=images)
        # DINOv2 variants: use CLS token (first position of last_hidden_state)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            feat = out.last_hidden_state[:, 0]
        feat = F.normalize(feat, dim=-1)
        feats.append(feat.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/features_external")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Subset of model names; default: all"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = get_cifar100_loader(
        split=args.split, n_samples=args.n_samples, batch_size=args.batch_size
    )

    specs_to_run = EXTERNAL_MODELS
    if args.models:
        specs_to_run = [s for s in EXTERNAL_MODELS if s[0] in args.models]

    for name, family, objective, loader_kind, source_id, _dim in specs_to_run:
        out_path = output_dir / f"{name}.pt"
        if out_path.exists():
            logger.info("Skipping %s (exists)", name)
            continue

        logger.info("Extracting %s via %s from %s", name, loader_kind, source_id)
        try:
            if loader_kind == "timm":
                features = extract_timm(source_id, loader, args.device)
            elif loader_kind == "hf_siglip":
                features = extract_hf_siglip(source_id, loader, args.device)
            elif loader_kind == "hf_automodel":
                features = extract_hf_automodel(source_id, loader, args.device)
            else:
                raise ValueError(f"Unknown loader: {loader_kind}")

            torch.save(
                {
                    "features": features,
                    "model_name": name,
                    "model_config": {
                        "family": family, "objective": objective,
                        "dim": features.shape[1], "source": source_id,
                    },
                    "split": f"cifar100_{args.split}",
                    "n_samples": features.shape[0],
                },
                out_path,
            )
            logger.info("Saved %s shape=%s", out_path, tuple(features.shape))
        except Exception as e:
            logger.error("Failed %s: %s", name, e)


if __name__ == "__main__":
    main()
