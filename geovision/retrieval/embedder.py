"""Multimodal satellite scene embedding model wrapping OpenCLIP and lightweight vision backbones."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from geovision.constants import EUROSAT_CLASSES
from geovision.logger import get_logger

logger = get_logger("geovision.retrieval.embedder")

try:
    import open_clip
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

# Standard remote sensing prompt templates for ensemble zero-shot classification
DEFAULT_SATELLITE_PROMPTS = [
    "a satellite photograph of a {}.",
    "an aerial remote sensing view of {}.",
    "a top-down satellite image showing {}.",
    "a Sentinel-2 Earth observation scene of {}.",
    "an aerial orthophoto depicting {}.",
]


class LightweightSatelliteEmbedder(nn.Module):
    """Standalone lightweight vision & text projection embedder for fast CPU execution and offline mode."""

    def __init__(self, embedding_dim: int = 512, vocab_size: int = 2000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Image encoder using ResNet18 backbone
        resnet = models.resnet18(weights=None)
        self.visual = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
        )

        # Simple bag-of-words / character n-gram text projection head
        self.text_embed = nn.Embedding(vocab_size, 64)
        self.text_fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.visual(x)
        return F.normalize(feats, p=2, dim=-1)

    def _simple_tokenize(self, text_list: list[str]) -> torch.Tensor:
        """Hash words into token IDs."""
        tokens = []
        for text in text_list:
            words = text.lower().replace("-", " ").replace("_", " ").split()
            hashes = [abs(hash(w)) % 1999 + 1 for w in words] or [1]
            # Pad/truncate to length 16
            hashes = (hashes + [0] * 16)[:16]
            tokens.append(hashes)
        return torch.tensor(tokens, dtype=torch.long)

    def encode_text(self, text_list: list[str], device: torch.device) -> torch.Tensor:
        toks = self._simple_tokenize(text_list).to(device)
        emb = self.text_embed(toks)  # (B, 16, 64)
        mean_emb = emb.mean(dim=1)   # (B, 64)
        feats = self.text_fc(mean_emb)
        return F.normalize(feats, p=2, dim=-1)


class SatelliteSceneEmbedder(nn.Module):
    """Production Multimodal Satellite Scene Embedder with OpenCLIP integration."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str | None = "openai",
        embedding_dim: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.target_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        self.fallback_model = None

        self._initialize_model()
        self.to(self.target_device)

    def _initialize_model(self) -> None:
        """Initialize OpenCLIP model or fallback embedder."""
        if _HAS_OPEN_CLIP:
            try:
                logger.info(f"Loading OpenCLIP model '{self.model_name}' (pretrained='{self.pretrained}')...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                )
                self.clip_model = model
                self.preprocess = preprocess
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                logger.info("OpenCLIP model loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load OpenCLIP model ({e}). Using LightweightSatelliteEmbedder fallback.")

        self.fallback_model = LightweightSatelliteEmbedder(embedding_dim=self.embedding_dim)
        logger.info("Using LightweightSatelliteEmbedder fallback architecture.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Default forward pass encodes images."""
        return self.encode_image(images)

    def encode_image(
        self,
        images: torch.Tensor | np.ndarray | Image.Image | list[Any],
    ) -> torch.Tensor:
        """Extract L2-normalized image embeddings.

        Args:
            images: PyTorch tensor (B, 3, H, W), single PIL/NumPy image, or list of images.

        Returns:
            Normalized tensor of shape (B, embedding_dim).
        """
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]

        if isinstance(images, list):
            tensor_list = []
            for img in images:
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img)
                elif isinstance(img, Image.Image):
                    pil_img = img
                else:
                    raise TypeError(f"Unsupported image type in list: {type(img)}")

                if self.preprocess is not None:
                    tensor_list.append(self.preprocess(pil_img))
                else:
                    arr = np.array(pil_img.convert("RGB").resize((224, 224)))
                    tensor_list.append(torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0)

            img_tensor = torch.stack(tensor_list).to(self.target_device)
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            img_tensor = images.to(self.target_device)
            if img_tensor.shape[2:] != (224, 224):
                img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)
        else:
            raise TypeError(f"Unsupported input type for encode_image: {type(images)}")

        if self.clip_model is not None:
            feats = self.clip_model.encode_image(img_tensor)
            return F.normalize(feats, p=2, dim=-1)
        else:
            return self.fallback_model.encode_image(img_tensor)

    def encode_text(
        self,
        texts: str | list[str],
    ) -> torch.Tensor:
        """Extract L2-normalized text prompt embeddings.

        Args:
            texts: String or list of text strings.

        Returns:
            Normalized tensor of shape (B, embedding_dim).
        """
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = list(texts)

        if self.clip_model is not None and self.tokenizer is not None:
            toks = self.tokenizer(text_list).to(self.target_device)
            feats = self.clip_model.encode_text(toks)
            return F.normalize(feats, p=2, dim=-1)
        else:
            return self.fallback_model.encode_text(text_list, device=self.target_device)

    def build_class_text_embeddings(
        self,
        class_names: list[str] | None = None,
        templates: list[str] | None = None,
    ) -> torch.Tensor:
        """Generate prompt-ensembled normalized text embeddings for all target classes.

        Args:
            class_names: List of class labels (defaults to EUROSAT_CLASSES).
            templates: List of prompt string templates with '{}' placeholder.

        Returns:
            Normalized tensor of shape (num_classes, embedding_dim).
        """
        classes = class_names or EUROSAT_CLASSES
        prompts = templates or DEFAULT_SATELLITE_PROMPTS

        class_embeddings = []
        for c in classes:
            c_text_prompts = [tpl.format(c) for tpl in prompts]
            p_embs = self.encode_text(c_text_prompts)  # (num_templates, D)
            mean_emb = p_embs.mean(dim=0, keepdim=True)
            norm_emb = F.normalize(mean_emb, p=2, dim=-1)
            class_embeddings.append(norm_emb)

        return torch.cat(class_embeddings, dim=0)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save model state dict and metadata."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict(),
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "embedding_dim": self.embedding_dim,
            "metadata": metadata or {},
        }
        torch.save(payload, save_path)
        logger.info(f"Saved SatelliteSceneEmbedder checkpoint to {save_path}")
        return save_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SatelliteSceneEmbedder":
        """Load embedder from saved checkpoint."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        embedder = cls(
            model_name=checkpoint.get("model_name", "ViT-B-32"),
            pretrained=checkpoint.get("pretrained", "openai"),
            embedding_dim=checkpoint.get("embedding_dim", 512),
            device=device,
        )
        embedder.load_state_dict(checkpoint["state_dict"])
        embedder.eval()
        logger.info(f"Loaded SatelliteSceneEmbedder from {load_path}")
        return embedder
