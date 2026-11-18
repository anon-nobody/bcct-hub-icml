"""Extract features from pretrained audio models for BCCT cross-domain validation.

Usage:
    python scripts/extract_audio_features.py --output-dir data/audio_features --device cuda:0

Produces .pt files containing:
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

logger = logging.getLogger(__name__)


@dataclass
class AudioModelSpec:
    """Specification for a pretrained audio model."""

    name: str
    family: str
    objective: str
    hf_path: str
    dim: int
    model_class: str  # Wav2Vec2Model, HubertModel, WhisperModel


AUDIO_MODEL_REGISTRY: list[AudioModelSpec] = [
    AudioModelSpec(
        "wav2vec2-base",
        "wav2vec2",
        "self-supervised",
        "facebook/wav2vec2-base",
        768,
        "Wav2Vec2Model",
    ),
    AudioModelSpec(
        "wav2vec2-large",
        "wav2vec2",
        "self-supervised",
        "facebook/wav2vec2-large",
        1024,
        "Wav2Vec2Model",
    ),
    AudioModelSpec(
        "hubert-base",
        "hubert",
        "self-supervised",
        "facebook/hubert-base-ls960",
        768,
        "HubertModel",
    ),
    AudioModelSpec(
        "hubert-large",
        "hubert",
        "self-supervised",
        "facebook/hubert-large-ll60k",
        1024,
        "HubertModel",
    ),
    AudioModelSpec(
        "whisper-base",
        "whisper",
        "supervised",
        "openai/whisper-base",
        512,
        "WhisperModel",
    ),
    AudioModelSpec(
        "whisper-small",
        "whisper",
        "supervised",
        "openai/whisper-small",
        768,
        "WhisperModel",
    ),
]


def load_audio_model(spec: AudioModelSpec, device: str) -> torch.nn.Module:
    """Load a pretrained audio encoder."""
    from transformers import HubertModel, Wav2Vec2Model, WhisperModel

    model_classes = {
        "Wav2Vec2Model": Wav2Vec2Model,
        "HubertModel": HubertModel,
        "WhisperModel": WhisperModel,
    }
    cls = model_classes[spec.model_class]
    model = cls.from_pretrained(spec.hf_path)
    model = model.to(device).eval()
    return model


def load_librispeech(max_samples: int = 2000) -> list[dict]:
    """Load LibriSpeech test-clean FLAC files using soundfile."""
    import soundfile as sf

    data_root = Path("data/librispeech/LibriSpeech/test-clean")
    if not data_root.exists():
        raise FileNotFoundError(
            f"{data_root} not found. Run: torchaudio LIBRISPEECH download first."
        )

    flac_files = sorted(data_root.rglob("*.flac"))
    logger.info(f"Found {len(flac_files)} FLAC files in {data_root}")

    samples = []
    for i, fpath in enumerate(flac_files):
        if i >= max_samples:
            break
        data, sr = sf.read(fpath, dtype="float32")
        waveform = torch.tensor(data, dtype=torch.float32)
        samples.append({"waveform": waveform, "sr": sr})
        if (i + 1) % 500 == 0:
            logger.info(f"  Loaded {i + 1}/{min(len(flac_files), max_samples)} samples")
    logger.info(f"Loaded {len(samples)} audio samples at {sr}Hz")
    return samples


@torch.no_grad()
def extract_audio_features(
    model: torch.nn.Module,
    spec: AudioModelSpec,
    samples: list[dict],
    device: str,
    max_samples: int = 2000,
) -> Tensor:
    """Extract features from an audio model.

    For wav2vec2/HuBERT: use last_hidden_state, mean-pool over time.
    For whisper: use encoder last_hidden_state, mean-pool over time.
    """
    import torchaudio
    from transformers import WhisperFeatureExtractor

    features_list: list[Tensor] = []

    # Pre-create whisper feature extractor if needed
    whisper_extractor = (
        WhisperFeatureExtractor.from_pretrained(spec.hf_path)
        if spec.model_class == "WhisperModel"
        else None
    )
    for i, sample in enumerate(samples):
        if i >= max_samples:
            break

        waveform = sample["waveform"]
        sr = sample["sr"]

        # Resample to 16kHz if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Truncate/pad to 10 seconds max (160000 samples at 16kHz)
        max_len = 160000
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]

        waveform = waveform.unsqueeze(0).to(device)  # (1, T)

        if spec.model_class == "WhisperModel":
            # Whisper needs mel spectrogram input via transformers processor
            assert whisper_extractor is not None
            inputs = whisper_extractor(
                waveform.squeeze(0).cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device)
            out = model.encoder(input_features)
            hidden = out.last_hidden_state  # (1, T', D)
        else:
            # wav2vec2 / HuBERT: raw waveform input
            out = model(waveform)
            hidden = out.last_hidden_state  # (1, T', D)

        # Mean pool over time dimension
        pooled = hidden.mean(dim=1)  # (1, D)
        pooled = F.normalize(pooled, dim=-1)
        features_list.append(pooled.cpu())

        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i + 1}/{min(len(samples), max_samples)} samples")

    features = torch.cat(features_list, dim=0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--output-dir", type=str, default="data/audio_features")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio data once
    samples = load_librispeech(max_samples=args.n_samples)

    for spec in AUDIO_MODEL_REGISTRY:
        output_path = output_dir / f"{spec.name}.pt"
        if output_path.exists():
            logger.info("Skipping %s (already exists)", spec.name)
            continue

        logger.info("Loading model: %s (%s)", spec.name, spec.hf_path)
        try:
            model = load_audio_model(spec, args.device)

            features = extract_audio_features(
                model, spec, samples, args.device, max_samples=args.n_samples
            )

            torch.save(
                {
                    "features": features,
                    "model_name": spec.name,
                    "model_config": {
                        "family": spec.family,
                        "objective": spec.objective,
                        "dim": spec.dim,
                        "source": "transformers",
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

    logger.info("Done! Audio features saved to %s", output_dir)


if __name__ == "__main__":
    main()
