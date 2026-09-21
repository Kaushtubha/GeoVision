"""Hierarchical and type-safe configuration system for GeoVision."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from geovision.constants import (
    CONFIGS_DIR,
    DATA_DIR,
    MLRUNS_DIR,
    WEIGHTS_DIR,
)


class BaseConfig(BaseModel):
    """Base configuration model with serialization utilities."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to python dictionary."""
        return self.model_dump()

    def save_yaml(self, path: Path | str) -> None:
        """Save configuration instance to YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)


class DatasetSpec(BaseModel):
    """Dataset configuration specification."""
    name: str
    raw_dir: str = str(DATA_DIR / "raw")
    processed_dir: str = str(DATA_DIR / "processed")
    smoke_samples: int = 20
    batch_size: int = 8
    num_workers: int = 0
    image_size: int = 512
    val_split: float = 0.2
    test_split: float = 0.1
    seed: int = 42


class DetectionConfig(BaseModel):
    """Object detection model and training settings."""
    model_name: str = "yolov8n.pt"
    dataset_name: str = "nwpu_vhr10"
    num_classes: int = 10
    image_size: int = 512
    epochs: int = 50
    smoke_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 0.001
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "cpu"
    save_dir: str = str(WEIGHTS_DIR / "detection")


class SegmentationConfig(BaseModel):
    """Land-cover semantic segmentation settings."""
    architecture: str = "Unet"  # SegFormer or Unet
    encoder_name: str = "mobilenet_v3_small"  # mit_b0 or mobilenet_v3_small
    encoder_weights: str = "imagenet"
    dataset_name: str = "landcover_ai"
    num_classes: int = 5  # Background + 4 classes
    image_size: int = 512
    epochs: int = 30
    smoke_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 0.0003
    loss: str = "dice_ce"
    device: str = "cpu"
    save_dir: str = str(WEIGHTS_DIR / "segmentation")


class ChangeConfig(BaseModel):
    """Bi-temporal change detection settings."""
    architecture: str = "SiameseUnet"
    encoder_name: str = "resnet18"
    encoder_weights: str = "imagenet"
    dataset_name: str = "levir_cd"
    num_classes: int = 2  # No Change / Change
    image_size: int = 256
    epochs: int = 25
    smoke_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 0.0003
    device: str = "cpu"
    save_dir: str = str(WEIGHTS_DIR / "change")


class RetrievalConfig(BaseModel):
    """Satellite scene retrieval & metric learning settings."""
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    dataset_name: str = "eurosat"
    num_classes: int = 10
    embedding_dim: int = 512
    collection_name: str = "geovision_scenes"
    top_k: int = 5
    batch_size: int = 16
    qdrant_path: str = str(DATA_DIR / "qdrant_db")
    device: str = "cpu"


class VLMConfig(BaseModel):
    """Grounded VLM and reasoning assistant settings."""
    vlm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    quantization: str = "4bit"
    max_new_tokens: int = 512
    temperature: float = 0.1
    fallback_to_text_llm: bool = True
    grounding_strictness: str = "strict"  # strict citation required


class TrackingConfig(BaseModel):
    """MLflow experiment tracking settings."""
    enabled: bool = True
    experiment_name: str = "GeoVision"
    tracking_uri: str = f"file:///{MLRUNS_DIR.as_posix()}"
    log_models: bool = True


class GeoVisionConfig(BaseConfig):
    """Master GeoVision configuration class."""
    seed: int = 42
    smoke: bool = False
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    change: ChangeConfig = Field(default_factory=ChangeConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)

    @classmethod
    def load_yaml(cls, path: Path | str) -> "GeoVisionConfig":
        """Load configuration from a YAML file."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Config file not found at: {target}")
        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load_default(cls) -> "GeoVisionConfig":
        """Load default config from configs/default.yaml if exists, else defaults."""
        default_path = CONFIGS_DIR / "default.yaml"
        if default_path.exists():
            return cls.load_yaml(default_path)
        return cls()


def load_config(config_path: str | Path | None = None, smoke: bool = False) -> GeoVisionConfig:
    """Helper function to load configuration with optional CLI overrides."""
    if config_path:
        cfg = GeoVisionConfig.load_yaml(config_path)
    else:
        cfg = GeoVisionConfig.load_default()
    if smoke:
        cfg.smoke = True
    return cfg
