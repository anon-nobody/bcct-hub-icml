"""Extract features from CIFAR-100 TRAIN set for probe training.

This is used to fix test-label leakage in downstream validation:
probes are trained on TRAIN features (50K), evaluated on existing TEST features (5K).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_train.py \
        --output-dir data/features_train --device cuda:0 --n-samples 5000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

# Reuse model loaders and registry from existing script
from extract_features import (  # type: ignore[import-not-found]
    MODEL_REGISTRY,
    load_clip,
    load_dinov2,
    load_timm,
)
from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_cifar100_train_loader(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    n_samples: int = 5000,
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    """CIFAR-100 TRAIN loader for probe training (deterministic subset)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform
    )
    if n_samples < len(dataset):
        # Deterministic subset (seed=42) to keep probe training fast & reproducible
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=g)[:n_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def extract(
    model: torch.nn.Module, spec, loader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match extract_features.py::extract_features_from_model exactly so that
    train/test features live in the same space (same projection, same L2 norm)."""
    import torch.nn.functional as F

    feats, labels = [], []
    for images, y in tqdm(loader, desc="extracting", leave=False):
        images = images.to(device, non_blocking=True)

        if spec.loader == "load_clip":
            out = model(pixel_values=images).image_embeds
        elif spec.loader == "load_dinov2":
            out = model(images)
        else:
            out = model(images)

        out = F.normalize(out, dim=-1)
        feats.append(out.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/features_train")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        [s for s in MODEL_REGISTRY if any(m in s.name for m in args.models)]
        if args.models
        else MODEL_REGISTRY
    )
    logger.info("Extracting CIFAR-100 TRAIN features for %d models (n=%d)", len(specs), args.n_samples)

    loader = get_cifar100_train_loader(
        data_dir=args.data_dir, batch_size=args.batch_size, n_samples=args.n_samples
    )

    # Save train labels once (deterministic across models)
    labels_path = Path(args.data_dir) / "cifar100_train_labels.pt"
    if not labels_path.exists():
        all_labels = torch.cat([y for _, y in loader])
        torch.save(all_labels, labels_path)
        logger.info("Saved train labels: %s (n=%d)", labels_path, len(all_labels))

    for spec in specs:
        output_path = output_dir / f"{spec.name}.pt"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", spec.name)
            continue

        logger.info("Loading: %s", spec.name)
        try:
            if spec.loader == "load_dinov2":
                model = load_dinov2(spec.name, args.device)
            elif spec.loader == "load_clip":
                model = load_clip(spec.name, args.device)
            elif spec.loader == "load_timm":
                model = load_timm(spec.name, args.device)
            else:
                continue

            features, _ = extract(model, spec, loader, args.device)
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
                    "split": "cifar100_train",
                    "n_samples": features.shape[0],
                },
                output_path,
            )
            logger.info("Saved %s: shape=%s", output_path, tuple(features.shape))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("Failed %s: %s", spec.name, e)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
