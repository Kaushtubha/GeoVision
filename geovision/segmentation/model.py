"""Segmentation model wrapper supporting SMP architectures and custom backbones."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from geovision.logger import get_logger

logger = get_logger("geovision.segmentation.model")

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LightweightUNet(nn.Module):
    """Standalone native PyTorch lightweight UNet for CPU/smoke execution and zero-dependency fallback."""

    def __init__(self, in_channels: int = 3, num_classes: int = 5, base_filters: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))

        self.up1 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up3 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 2, base_filters)

        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = self.conv_up1(torch.cat([u1, x3], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, x2], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, x1], dim=1))

        return self.outc(u3)


class LandCoverSegmenter(nn.Module):
    """Production Land-Cover Semantic Segmentation model wrapper."""

    def __init__(
        self,
        architecture: str = "Unet",
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 5,
        device: str = "cpu",
    ):
        super().__init__()
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        self.model = self._build_model()
        self.to(self.target_device)

    def _build_model(self) -> nn.Module:
        """Construct the underlying PyTorch / SMP model."""
        if _HAS_SMP:
            try:
                # Attempt to build requested SMP architecture
                arch_cls = getattr(smp, self.architecture, smp.Unet)
                return arch_cls(
                    encoder_name=self.encoder_name,
                    encoder_weights=self.encoder_weights,
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                )
            except Exception as e:
                logger.warning(
                    f"Could not initialize SMP model with weights '{self.encoder_weights}' ({e}). "
                    f"Building without pre-trained weights or falling back to LightweightUNet."
                )
                try:
                    arch_cls = getattr(smp, self.architecture, smp.Unet)
                    return arch_cls(
                        encoder_name=self.encoder_name,
                        encoder_weights=None,
                        in_channels=self.in_channels,
                        classes=self.num_classes,
                    )
                except Exception as ex:
                    logger.warning(f"SMP initialization failed ({ex}), using LightweightUNet fallback.")
                    return LightweightUNet(
                        in_channels=self.in_channels,
                        num_classes=self.num_classes,
                    )

        logger.info("Using standalone LightweightUNet architecture.")
        return LightweightUNet(in_channels=self.in_channels, num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (N, num_classes, H, W)."""
        return self.model(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run evaluation inference returning predicted class masks and softmax probability maps.

        Args:
            x: Input tensor (N, C, H, W) or (C, H, W).

        Returns:
            Tuple of:
            - Predicted class labels (N, H, W)
            - Softmax class probabilities (N, num_classes, H, W)
        """
        self.eval()
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.target_device)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save model checkpoint and metadata dictionary."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict(),
            "architecture": self.architecture,
            "encoder_name": self.encoder_name,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "metadata": metadata or {},
        }
        torch.save(payload, save_path)
        logger.info(f"Saved LandCoverSegmenter checkpoint to {save_path}")
        return save_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "LandCoverSegmenter":
        """Load model from saved checkpoint file."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        segmenter = cls(
            architecture=checkpoint.get("architecture", "Unet"),
            encoder_name=checkpoint.get("encoder_name", "mobilenet_v3_small"),
            encoder_weights=None,
            num_classes=checkpoint.get("num_classes", 5),
            in_channels=checkpoint.get("in_channels", 3),
            device=device,
        )
        segmenter.load_state_dict(checkpoint["state_dict"])
        segmenter.eval()
        logger.info(f"Loaded LandCoverSegmenter from {load_path}")
        return segmenter
