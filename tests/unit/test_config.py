"""Tests for configuration utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from latent.utils.config import Config, ModelConfig, TrainingConfig


class TestConfig:
    """Tests for Config loading and saving."""

    def test_default_config(self) -> None:
        config = Config()
        assert config.model.encoder == "dinov2_vitl14"
        assert config.training.lr == 1e-4
        assert config.evaluation.k == 10

    def test_yaml_roundtrip(self) -> None:
        config = Config(
            experiment_name="test_exp",
            model=ModelConfig(encoder="clip_vitl14", latent_dim=1024),
            training=TrainingConfig(batch_size=128, epochs=50),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(path)
            loaded = Config.from_yaml(path)
            assert loaded.experiment_name == "test_exp"
            assert loaded.model.encoder == "clip_vitl14"
            assert loaded.model.latent_dim == 1024
            assert loaded.training.batch_size == 128

    def test_load_base_yaml(self) -> None:
        base_path = Path(__file__).parent.parent.parent / "configs" / "base.yaml"
        if base_path.exists():
            config = Config.from_yaml(base_path)
            assert config.experiment_name == "bcct_base"
            assert config.evaluation.k == 10

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/config.yaml")

    def test_empty_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            path.write_text("")
            config = Config.from_yaml(path)
            assert config.experiment_name == "default"
