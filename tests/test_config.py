"""Unit tests for configuration loading, validation, and serialization."""

import tempfile
from pathlib import Path

import pytest

from geovision.config import (
    GeoVisionConfig,
    load_config,
)


def test_default_config_instantiation():
    """Verify default config instance is created with expected defaults."""
    cfg = GeoVisionConfig()
    assert cfg.seed == 42
    assert cfg.smoke is False
    assert cfg.detection.model_name == "yolov8n.pt"
    assert cfg.segmentation.num_classes == 5
    assert cfg.change.num_classes == 2
    assert cfg.retrieval.dataset_name == "eurosat"


def test_config_yaml_serialization():
    """Verify config can be saved to YAML and re-loaded identically."""
    cfg = GeoVisionConfig(seed=123, smoke=True)
    cfg.detection.epochs = 10
    cfg.segmentation.learning_rate = 0.0005

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test_config.yaml"
        cfg.save_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = GeoVisionConfig.load_yaml(yaml_path)
        assert loaded.seed == 123
        assert loaded.smoke is True
        assert loaded.detection.epochs == 10
        assert loaded.segmentation.learning_rate == 0.0005


def test_load_config_helper():
    """Verify load_config helper handles CLI smoke overrides."""
    cfg = load_config(smoke=True)
    assert cfg.smoke is True


def test_invalid_config_path():
    """Verify attempting to load a non-existent YAML raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        GeoVisionConfig.load_yaml("non_existent_file_path_12345.yaml")
