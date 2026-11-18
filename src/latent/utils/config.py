"""Configuration utilities: dataclass-based configs with YAML loading.

Provides typed configuration for experiments, following the project conventions
in AGENTS.md. Uses dataclasses for in-code config, YAML for experiment files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes:
        encoder: Encoder model name or path (e.g., "dinov2_vitl14").
        latent_channels: Number of latent channels.
        latent_dim: Latent dimensionality (per spatial position).
        noise_schedule: Noise schedule type for diffusion prior.
    """

    encoder: str = "dinov2_vitl14"
    latent_channels: int = 16
    latent_dim: int = 768
    noise_schedule: str = "cosine"


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Attributes:
        batch_size: Per-GPU batch size.
        lr: Peak learning rate.
        epochs: Total training epochs.
        warmup_epochs: Linear warmup duration.
        weight_decay: AdamW weight decay.
        seed: Global random seed.
        gradient_clip: Max gradient norm (0 = no clipping).
    """

    batch_size: int = 256
    lr: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    seed: int = 42
    gradient_clip: float = 1.0


@dataclass
class EvalConfig:
    """Evaluation / BCCT protocol configuration.

    Attributes:
        k: Number of nearest neighbors for mutual k-NN.
        n_permutations: Null calibration permutations.
        reference_encoder: Default reference encoder for Scorecard.
        compression_levels: Default levels for bitrate sweep.
    """

    k: int = 10
    n_permutations: int = 1000
    reference_encoder: str = "openai/clip-vit-large-patch14"
    compression_levels: list[float] = field(
        default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]
    )


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration.

    Attributes:
        project: W&B project name.
        entity: W&B entity (team or user).
        enabled: Whether to log to W&B.
        tags: Run tags.
    """

    project: str = "bcct"
    entity: str = ""
    enabled: bool = True
    tags: list[str] = field(default_factory=list)  # type: ignore[arg-type]


@dataclass
class Config:
    """Top-level experiment configuration.

    Attributes:
        model: Model architecture settings.
        training: Training loop settings.
        evaluation: Evaluation protocol settings.
        wandb: W&B logging settings.
        experiment_name: Human-readable experiment identifier.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Populated Config instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
            wandb=WandbConfig(**raw.get("wandb", {})),
            experiment_name=raw.get("experiment_name", "default"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Output path.
        """
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
