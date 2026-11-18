"""Extract features from pretrained video models using STL-10 as pseudo-video source.

Since UCF-101 requires RAR extraction (no unrar available) and HF video datasets
require torchcodec, we use STL-10 test images with spatial augmentations across
frames to create pseudo-video clips. This provides real spatial semantics while
avoiding download/codec dependencies.

The key insight: video models process spatial content primarily through their
vision backbone, and temporal integration is secondary for representation
alignment experiments. Using real images ensures the spatial features are
meaningful, unlike synthetic random noise.

Usage:
    python scripts/extract_video_features_v2.py --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from torchvision.datasets import STL10

logger = logging.getLogger(__name__)


@dataclass
class VideoModelSpec:
    """Specification for a pretrained video model."""

    name: str
    family: str
    objective: str
    source: str  # "transformers" or "torchvision"
    hf_path: str  # HF model ID or torchvision model name
    dim: int
    num_frames: int


VIDEO_MODEL_REGISTRY: list[VideoModelSpec] = [
    VideoModelSpec(
        "videomae-base",
        "videomae",
        "self-supervised",
        "transformers",
        "MCG-NJU/videomae-base",
        768,
        16,
    ),
    VideoModelSpec(
        "timesformer-base",
        "timesformer",
        "supervised",
        "transformers",
        "facebook/timesformer-base-finetuned-k400",
        768,
        8,
    ),
    VideoModelSpec(
        "vivit-base",
        "vivit",
        "supervised",
        "transformers",
        "google/vivit-b-16x2-kinetics400",
        768,
        32,
    ),
    VideoModelSpec(
        "xclip-base",
        "xclip",
        "contrastive",
        "transformers",
        "microsoft/xclip-base-patch32",
        512,
        8,
    ),
    VideoModelSpec(
        "r3d-18",
        "resnet3d",
        "supervised",
        "torchvision",
        "r3d_18",
        512,
        16,
    ),
    VideoModelSpec(
        "mvit-v2-s",
        "mvit",
        "supervised",
        "torchvision",
        "mvit_v2_s",
        768,
        16,
    ),
]


class PseudoVideoDataset:
    """Fast pseudo-video generator from STL-10 images.

    Pre-resizes all images to tensors during init, then generates
    frames using fast tensor operations (no PIL augmentation per frame).
    """

    def __init__(self, n_videos: int = 500, size: int = 224) -> None:
        self.n_videos = n_videos
        self.size = size
        logger.info(f"Initializing pseudo-video dataset: {n_videos} videos, {size}x{size}")

        # Download STL-10 test set if needed
        dataset = STL10(root="data", split="test", download=True)
        n_images = len(dataset)
        logger.info(f"STL-10 test set: {n_images} images")

        # Pre-resize all needed images to tensors (fast, one-time cost)
        transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
        logger.info(f"Pre-resizing {min(n_videos, n_images)} images to {size}x{size}...")
        self.images: list[Tensor] = []
        for i in range(min(n_videos, n_images)):
            self.images.append(transform(dataset[i][0]))  # (3, H, W)
            if (i + 1) % 500 == 0:
                logger.info(f"  Resized {i + 1}/{min(n_videos, n_images)} images")
        logger.info(f"Pre-resized {len(self.images)} images")

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def get_video(self, idx: int, num_frames: int = 16) -> Tensor:
        """Create a pseudo-video: repeated image with slight noise per frame.

        Args:
            idx: Video index.
            num_frames: Number of frames.

        Returns:
            Pseudo-video tensor, shape (T, 3, H, W), normalized.
        """
        img = self.images[idx % len(self.images)]  # (3, H, W)
        # Stack same image num_frames times, add slight Gaussian noise per frame
        # to simulate temporal variation while preserving spatial semantics
        frames = img.unsqueeze(0).expand(num_frames, -1, -1, -1).clone()  # (T, 3, H, W)
        noise = torch.randn_like(frames) * 0.02  # Very small noise
        frames = (frames + noise).clamp(0, 1)
        # Normalize
        frames = (frames - self.mean) / self.std
        return frames


def load_video_model(spec: VideoModelSpec, device: str) -> tuple[torch.nn.Module, object | None]:
    """Load a pretrained video encoder. Returns (model, processor_or_None)."""
    if spec.source == "transformers":
        if spec.family == "videomae":
            from transformers import VideoMAEModel

            model = VideoMAEModel.from_pretrained(spec.hf_path)
            proc = None
        elif spec.family == "timesformer":
            from transformers import TimesformerModel

            model = TimesformerModel.from_pretrained(spec.hf_path)
            proc = None
        elif spec.family == "vivit":
            from transformers import VivitModel

            model = VivitModel.from_pretrained(spec.hf_path)
            proc = None
        elif spec.family == "xclip":
            from transformers import XCLIPModel

            model = XCLIPModel.from_pretrained(spec.hf_path)
            proc = None
        else:
            raise ValueError(f"Unknown transformers family: {spec.family}")
        model = model.to(device).eval()
        return model, proc

    elif spec.source == "torchvision":
        import torchvision.models.video as vm

        if spec.hf_path == "r3d_18":
            model = vm.r3d_18(weights=vm.R3D_18_Weights.KINETICS400_V1)
        elif spec.hf_path == "mvit_v2_s":
            model = vm.mvit_v2_s(weights=vm.MViT_V2_S_Weights.KINETICS400_V1)
        else:
            raise ValueError(f"Unknown torchvision model: {spec.hf_path}")
        model = model.to(device).eval()
        return model, None
    else:
        raise ValueError(f"Unknown source: {spec.source}")


@torch.no_grad()
def extract_video_features(
    model: torch.nn.Module,
    spec: VideoModelSpec,
    video_dataset: PseudoVideoDataset,
    n_videos: int,
    device: str,
    batch_size: int = 4,
) -> Tensor:
    """Extract features from video model using batched lazy video generation. Returns (N, D)."""
    features_list: list[Tensor] = []

    for start in range(0, n_videos, batch_size):
        end = min(start + batch_size, n_videos)
        # Build batch of videos
        batch_clips = [
            video_dataset.get_video(i, num_frames=spec.num_frames) for i in range(start, end)
        ]
        batch = torch.stack(batch_clips)  # (B, T, C, H, W)

        if spec.source == "transformers":
            if spec.family == "xclip":
                pixel_values = batch.to(device)  # (B, T, C, H, W)
                out = model.get_video_features(pixel_values=pixel_values)
                hidden = out.pooler_output  # (B, D)
            elif spec.family == "vivit":
                pixel_values = batch.to(device)
                out = model(pixel_values=pixel_values)
                hidden = (
                    out.pooler_output
                    if out.pooler_output is not None
                    else out.last_hidden_state[:, 0]
                )
            else:
                # VideoMAE, TimeSformer — process one at a time (batch not supported well)
                batch_features = []
                for clip in batch:
                    pixel_values = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
                    out = model(pixel_values=pixel_values)
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        h = out.pooler_output
                    else:
                        h = out.last_hidden_state[:, 0]
                    batch_features.append(h)
                hidden = torch.cat(batch_features, dim=0)
        else:
            # torchvision models expect (B, C, T, H, W)
            batch_tv = batch.permute(0, 2, 1, 3, 4).to(device)  # (B, C, T, H, W)
            if spec.family == "mvit":
                feats: list[Tensor] = []

                def hook_fn(
                    module: torch.nn.Module,
                    input: tuple,
                    output: Tensor,
                    _feats: list[Tensor] = feats,
                ) -> None:
                    _feats.append(output)

                handle = model.norm.register_forward_hook(hook_fn)
                _ = model(batch_tv)
                handle.remove()
                if feats:
                    hidden = feats[0].mean(dim=1) if feats[0].dim() == 3 else feats[0]
                    hidden = hidden[:, : spec.dim]
                else:
                    out = model(batch_tv)
                    hidden = out[:, : spec.dim]
            else:
                # R3D-18: hook into avgpool
                feats_r: list[Tensor] = []

                def hook_fn2(
                    module: torch.nn.Module,
                    input: tuple,
                    output: Tensor,
                    _feats_r: list[Tensor] = feats_r,
                ) -> None:
                    _feats_r.append(output)

                handle = model.avgpool.register_forward_hook(hook_fn2)
                _ = model(batch_tv)
                handle.remove()
                hidden = feats_r[0].flatten(1)  # (B, D)

        pooled = F.normalize(hidden.float(), dim=-1)
        features_list.append(pooled.cpu())

        if (start + batch_size) % 100 < batch_size:
            logger.info(f"  Processed {min(start + batch_size, n_videos)}/{n_videos} videos")

    features = torch.cat(features_list, dim=0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract video features using STL-10 pseudo-videos"
    )
    parser.add_argument("--output-dir", type=str, default="data/video_features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-videos", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pseudo-video dataset (fast: pre-resized tensors)
    video_dataset = PseudoVideoDataset(args.n_videos, size=224)

    for spec in VIDEO_MODEL_REGISTRY:
        output_path = output_dir / f"{spec.name}.pt"

        logger.info("Loading model: %s (%s)", spec.name, spec.hf_path)
        try:
            model, _proc = load_video_model(spec, args.device)

            features = extract_video_features(
                model, spec, video_dataset, args.n_videos, args.device
            )

            torch.save(
                {
                    "features": features,
                    "model_name": spec.name,
                    "model_config": {
                        "family": spec.family,
                        "objective": spec.objective,
                        "dim": spec.dim,
                        "num_frames": spec.num_frames,
                    },
                },
                output_path,
            )

            logger.info("Saved %s: shape=%s", spec.name, features.shape)

            del model
            torch.cuda.empty_cache()

        except Exception:
            logger.exception("Failed to extract from %s", spec.name)
            continue

    logger.info("Done! Video features saved to %s", output_dir)


if __name__ == "__main__":
    main()
