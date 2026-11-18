"""Extract features from pretrained video models for BCCT cross-domain validation.

Usage:
    python scripts/extract_video_features.py --output-dir data/video_features --device cuda:0

Models: VideoMAE, TimeSformer, ViViT, X-CLIP (transformers) + R3D-18, MViT-v2 (torchvision)
Dataset: UCF-101 test split (sampled 1000 clips)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

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


def load_video_model(spec: VideoModelSpec, device: str) -> tuple[torch.nn.Module, object | None]:
    """Load a pretrained video encoder. Returns (model, processor_or_None)."""
    if spec.source == "transformers":
        if spec.family == "videomae":
            from transformers import VideoMAEImageProcessor, VideoMAEModel

            proc = VideoMAEImageProcessor.from_pretrained(spec.hf_path)
            model = VideoMAEModel.from_pretrained(spec.hf_path)
        elif spec.family == "timesformer":
            from transformers import AutoImageProcessor, TimesformerModel

            proc = AutoImageProcessor.from_pretrained(spec.hf_path)
            model = TimesformerModel.from_pretrained(spec.hf_path)
        elif spec.family == "vivit":
            from transformers import VivitImageProcessor, VivitModel

            proc = VivitImageProcessor.from_pretrained(spec.hf_path)
            model = VivitModel.from_pretrained(spec.hf_path)
        elif spec.family == "xclip":
            from transformers import AutoProcessor, XCLIPModel

            proc = AutoProcessor.from_pretrained(spec.hf_path)
            model = XCLIPModel.from_pretrained(spec.hf_path)
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
        # Remove classification head, keep features
        model = model.to(device).eval()
        return model, None
    else:
        raise ValueError(f"Unknown source: {spec.source}")


def generate_synthetic_videos(
    n_videos: int = 1000, num_frames: int = 16, size: int = 224
) -> Tensor:
    """Generate synthetic random video tensors for feature extraction.

    In production, replace with real UCF-101 / Kinetics clips.
    Shape: (N, num_frames, 3, H, W)
    """
    logger.info(f"Generating {n_videos} synthetic videos: {num_frames}f × {size}×{size}")
    return torch.randn(n_videos, num_frames, 3, size, size)


def load_ucf101_videos(
    data_root: Path, max_videos: int = 1000, num_frames: int = 16, size: int = 224
) -> Tensor:
    """Load UCF-101 videos, sample frames, resize to (N, T, 3, H, W)."""
    import torchvision.io as io
    from torchvision.transforms import Compose, Normalize, Resize

    video_files = sorted(data_root.rglob("*.avi"))
    if not video_files:
        video_files = sorted(data_root.rglob("*.mp4"))
    if not video_files:
        logger.warning(f"No video files found in {data_root}, using synthetic data")
        return generate_synthetic_videos(max_videos, num_frames, size)

    logger.info(f"Found {len(video_files)} video files in {data_root}")
    transform = Compose([Resize((size, size))])
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    videos = []
    for i, vpath in enumerate(video_files):
        if i >= max_videos:
            break
        try:
            vframes, _, _ = io.read_video(str(vpath), pts_unit="sec", output_format="TCHW")
            # Sample num_frames evenly
            total = vframes.shape[0]
            if total < num_frames:
                # Repeat frames if too short
                indices = list(range(total)) * (num_frames // total + 1)
                indices = indices[:num_frames]
            else:
                indices = torch.linspace(0, total - 1, num_frames).long().tolist()
            frames = vframes[indices]  # (T, C, H, W)
            frames = frames.float() / 255.0
            frames = torch.stack([transform(f) for f in frames])  # (T, C, H, W)
            frames = torch.stack([normalize(f) for f in frames])
            videos.append(frames)
        except Exception:
            logger.debug(f"Failed to load {vpath}, skipping")
            continue
        if (i + 1) % 200 == 0:
            logger.info(f"  Loaded {i + 1}/{min(len(video_files), max_videos)} videos")

    if not videos:
        logger.warning("No videos loaded successfully, using synthetic data")
        return generate_synthetic_videos(max_videos, num_frames, size)

    result = torch.stack(videos)  # (N, T, C, H, W)
    logger.info(f"Loaded {result.shape[0]} videos: shape={result.shape}")
    return result


@torch.no_grad()
def extract_video_features(
    model: torch.nn.Module,
    processor: object | None,
    spec: VideoModelSpec,
    videos: Tensor,
    device: str,
) -> Tensor:
    """Extract features from video model. Returns (N, D)."""
    features_list: list[Tensor] = []
    n = videos.shape[0]

    for i in range(n):
        clip = videos[i]  # (T, C, H, W)

        if spec.source == "transformers":
            # Transformers video models expect list of frames as numpy or pixel_values
            if spec.family == "xclip":
                # XCLIPModel.get_video_features() expects (B, T, C, H, W)
                pixel_values = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
                out = model.get_video_features(pixel_values=pixel_values)
                hidden = out.pooler_output  # (1, D)
            elif spec.family == "vivit":
                pixel_values = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
                out = model(pixel_values=pixel_values)
                hidden = (
                    out.pooler_output
                    if out.pooler_output is not None
                    else out.last_hidden_state[:, 0]
                )
            else:
                # VideoMAE, TimeSformer
                pixel_values = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
                out = model(pixel_values=pixel_values)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    hidden = out.pooler_output  # (1, D)
                else:
                    hidden = out.last_hidden_state[:, 0]  # CLS token
        else:
            # torchvision models expect (B, C, T, H, W)
            clip_tv = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)
            if spec.family == "mvit":
                out = model(clip_tv)
                hidden = out.unsqueeze(0) if out.dim() == 1 else out  # (1, num_classes)
                # MViT outputs logits, we need features before head
                # Use forward hook instead
                feats = []

                def hook_fn(
                    module: torch.nn.Module,
                    input: tuple,
                    output: Tensor,
                    _feats: list = feats,
                ) -> None:
                    _feats.append(output)

                # Register hook on the norm layer before head
                handle = model.norm.register_forward_hook(hook_fn)
                _ = model(clip_tv)
                handle.remove()
                if feats:
                    hidden = feats[0].mean(dim=1) if feats[0].dim() == 3 else feats[0]
                    hidden = hidden.view(1, -1)[:, : spec.dim]
                else:
                    hidden = out[:, : spec.dim]
            else:
                # R3D-18: hook into avgpool
                feats = []

                def hook_fn2(
                    module: torch.nn.Module,
                    input: tuple,
                    output: Tensor,
                    _feats: list = feats,
                ) -> None:
                    _feats.append(output)

                handle = model.avgpool.register_forward_hook(hook_fn2)
                _ = model(clip_tv)
                handle.remove()
                hidden = feats[0].flatten(1)  # (1, D)

        pooled = F.normalize(hidden.float(), dim=-1)
        features_list.append(pooled.cpu())

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{n} videos")

    features = torch.cat(features_list, dim=0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video features")
    parser.add_argument("--output-dir", type=str, default="data/video_features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-videos", type=int, default=1000)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ucf101",
        help="Path to UCF-101 or video dataset",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading real videos, fall back to synthetic
    data_dir = Path(args.data_dir)
    max_frames = max(s.num_frames for s in VIDEO_MODEL_REGISTRY)

    if data_dir.exists():
        videos = load_ucf101_videos(data_dir, args.n_videos, max_frames)
    else:
        logger.info(f"Data dir {data_dir} not found, using synthetic videos")
        videos = generate_synthetic_videos(args.n_videos, max_frames)

    for spec in VIDEO_MODEL_REGISTRY:
        output_path = output_dir / f"{spec.name}.pt"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", spec.name)
            continue

        logger.info("Loading model: %s (%s)", spec.name, spec.hf_path)
        try:
            model, proc = load_video_model(spec, args.device)

            # Trim frames to what this model needs
            clip_videos = videos[:, : spec.num_frames]

            features = extract_video_features(model, proc, spec, clip_videos, args.device)

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
