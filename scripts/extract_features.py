"""Extract features from pretrained models for the BCCT alignment atlas.

Usage:
    python scripts/extract_features.py --output-dir data/features --device cuda:0
    python scripts/extract_features.py --models clip dinov2 --n-samples 5000

Produces .pt files in output-dir, one per model, containing:
    {"features": Tensor(N, D), "model_name": str, "model_config": dict}
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry: name → loader function
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Specification for a pretrained model."""

    name: str
    family: str  # clip, dinov2, mae, supervised, generative
    objective: str  # contrastive, self-supervised, supervised, generative
    loader: str  # function name to call
    dim: int  # expected feature dimensionality
    source: str  # timm, transformers, torchvision


# Models we can load without special downloads
MODEL_REGISTRY: list[ModelSpec] = [
    # DINOv2 family (self-supervised)
    ModelSpec("dinov2_vits14", "dinov2", "self-supervised", "load_dinov2", 384, "torch.hub"),
    ModelSpec("dinov2_vitb14", "dinov2", "self-supervised", "load_dinov2", 768, "torch.hub"),
    ModelSpec("dinov2_vitl14", "dinov2", "self-supervised", "load_dinov2", 1024, "torch.hub"),
    ModelSpec("dinov2_vitg14", "dinov2", "self-supervised", "load_dinov2", 1536, "torch.hub"),
    # CLIP family (contrastive)
    ModelSpec("clip_vitb32", "clip", "contrastive", "load_clip", 512, "transformers"),
    ModelSpec("clip_vitb16", "clip", "contrastive", "load_clip", 512, "transformers"),
    ModelSpec("clip_vitl14", "clip", "contrastive", "load_clip", 768, "transformers"),
    # Supervised (timm)
    ModelSpec("resnet50", "supervised", "supervised", "load_timm", 2048, "timm"),
    ModelSpec("resnet101", "supervised", "supervised", "load_timm", 2048, "timm"),
    ModelSpec("vit_base_patch16_224", "supervised", "supervised", "load_timm", 768, "timm"),
    ModelSpec("vit_large_patch16_224", "supervised", "supervised", "load_timm", 1024, "timm"),
    ModelSpec("convnext_base", "supervised", "supervised", "load_timm", 1024, "timm"),
    ModelSpec("convnext_large", "supervised", "supervised", "load_timm", 1536, "timm"),
    ModelSpec("efficientnet_b0", "supervised", "supervised", "load_timm", 1280, "timm"),
    ModelSpec(
        "swin_base_patch4_window7_224", "supervised", "supervised", "load_timm", 1024, "timm"
    ),
    # MAE (self-supervised, timm)
    ModelSpec("vit_base_patch16_224.mae", "mae", "self-supervised", "load_timm", 768, "timm"),
    ModelSpec("vit_large_patch16_224.mae", "mae", "self-supervised", "load_timm", 1024, "timm"),
    # BEiT (self-supervised, timm)
    ModelSpec("beit_base_patch16_224", "beit", "self-supervised", "load_timm", 768, "timm"),
    ModelSpec("beit_large_patch16_224", "beit", "self-supervised", "load_timm", 1024, "timm"),
    # MLP-Mixer
    ModelSpec("mixer_b16_224", "mlpmixer", "supervised", "load_timm", 768, "timm"),
]


def load_dinov2(model_name: str, device: str) -> torch.nn.Module:
    """Load a DINOv2 model from torch hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model = model.to(device).eval()
    return model


def load_clip(model_name: str, device: str) -> torch.nn.Module:
    """Load CLIP visual encoder from transformers."""
    from transformers import CLIPVisionModelWithProjection

    clip_map = {
        "clip_vitb32": "openai/clip-vit-base-patch32",
        "clip_vitb16": "openai/clip-vit-base-patch16",
        "clip_vitl14": "openai/clip-vit-large-patch14",
    }
    hf_name = clip_map[model_name]
    model = CLIPVisionModelWithProjection.from_pretrained(hf_name)
    model = model.to(device).eval()
    return model


def load_timm(model_name: str, device: str) -> torch.nn.Module:
    """Load a model from timm."""
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_features_from_model(
    model: torch.nn.Module,
    spec: ModelSpec,
    dataloader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: str,
    max_samples: int = 5000,
) -> Tensor:
    """Extract features from a model on a dataset.

    Args:
        model: Pretrained model.
        spec: Model specification.
        dataloader: DataLoader for the evaluation dataset.
        device: Device string.
        max_samples: Maximum number of samples to process.

    Returns:
        Feature tensor, shape (N, D).
    """
    features_list: list[Tensor] = []
    n_collected = 0

    for batch_images, _ in tqdm(dataloader, desc=spec.name, leave=False):
        if n_collected >= max_samples:
            break

        batch_images = batch_images.to(device)

        if spec.loader == "load_clip":
            # CLIP: use vision encoder with projection → image_embeds
            out = model(pixel_values=batch_images).image_embeds
        elif spec.loader == "load_dinov2":
            # DINOv2: forward returns CLS token
            out = model(batch_images)
        else:
            # timm models with num_classes=0 return features directly
            out = model(batch_images)

        # Normalize features
        out = F.normalize(out, dim=-1)
        features_list.append(out.cpu())
        n_collected += out.shape[0]

    features = torch.cat(features_list, dim=0)[:max_samples]
    return features


def get_imagenet_val_loader(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    """Create ImageNet validation DataLoader.

    If ImageNet is not available, falls back to CIFAR-100 as a proxy.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        logger.info("Using ImageNet validation set from %s", val_dir)
        dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    else:
        logger.warning("ImageNet not found at %s, falling back to CIFAR-100", data_dir)
        dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from pretrained models")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/imagenet", help="Dataset directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models
    if args.models:
        specs = [s for s in MODEL_REGISTRY if any(m in s.name for m in args.models)]
    else:
        specs = MODEL_REGISTRY

    logger.info("Will extract features from %d models", len(specs))

    # Create dataloader
    dataloader = get_imagenet_val_loader(args.data_dir, batch_size=args.batch_size)

    for spec in specs:
        output_path = output_dir / f"{spec.name}.pt"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", spec.name)
            continue

        logger.info("Loading model: %s (%s)", spec.name, spec.source)
        try:
            if spec.loader == "load_dinov2":
                model = load_dinov2(spec.name, args.device)
            elif spec.loader == "load_clip":
                model = load_clip(spec.name, args.device)
            elif spec.loader == "load_timm":
                model = load_timm(spec.name, args.device)
            else:
                logger.error("Unknown loader: %s", spec.loader)
                continue

            features = extract_features_from_model(
                model, spec, dataloader, args.device, max_samples=args.n_samples
            )

            torch.save(
                {
                    "features": features,
                    "model_name": spec.name,
                    "model_config": {
                        "family": spec.family,
                        "objective": spec.objective,
                        "dim": spec.dim,
                        "source": spec.source,
                    },
                },
                output_path,
            )

            logger.info("Saved %s features: shape=%s", spec.name, features.shape)

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception:
            logger.exception("Failed to extract features from %s", spec.name)
            continue

    logger.info("Done! Features saved to %s", output_dir)


if __name__ == "__main__":
    main()
