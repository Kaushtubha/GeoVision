"""Siamese Neural Network architectures for bi-temporal remote sensing change detection."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from geovision.logger import get_logger

logger = get_logger("geovision.change.model")


class ConvBlock(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsampling block with skip-connection fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Dual-branch Siamese encoder-decoder network with multi-scale feature difference fusion."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_name: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes

        if encoder_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            # Encoder stages
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2, 64 ch
            self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)           # /4, 64 ch
            self.enc2 = resnet.layer2                                          # /8, 128 ch
            self.enc3 = resnet.layer3                                          # /16, 256 ch
            self.enc4 = resnet.layer4                                          # /32, 512 ch

            # Decoder blocks fusing difference features |f_A - f_B|
            self.dec4 = DecoderBlock(512, 256, 256)
            self.dec3 = DecoderBlock(256, 128, 128)
            self.dec2 = DecoderBlock(128, 64, 64)
            self.dec1 = DecoderBlock(64, 64, 64)
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        else:
            # Standalone lightweight CNN encoder
            self.enc0 = ConvBlock(in_channels, 32)
            self.enc1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
            self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
            self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
            self.enc4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))

            self.dec4 = DecoderBlock(512, 256, 256)
            self.dec3 = DecoderBlock(256, 128, 128)
            self.dec2 = DecoderBlock(128, 64, 64)
            self.dec1 = DecoderBlock(64, 32, 32)
            self.final_up = nn.Identity()
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def _extract_features(self, x: torch.Tensor):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e0, e1, e2, e3, e4

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass taking two temporal co-registered images."""
        h, w = x1.shape[2:]

        # Shared weight Siamese encoder passes
        e0_1, e1_1, e2_1, e3_1, e4_1 = self._extract_features(x1)
        e0_2, e1_2, e2_2, e3_2, e4_2 = self._extract_features(x2)

        # Multi-scale absolute feature difference representations
        d0 = torch.abs(e0_1 - e0_2)
        d1 = torch.abs(e1_1 - e1_2)
        d2 = torch.abs(e2_1 - e2_2)
        d3 = torch.abs(e3_1 - e3_2)
        d4 = torch.abs(e4_1 - e4_2)

        # Decoder fusion
        x = self.dec4(d4, d3)
        x = self.dec3(x, d2)
        x = self.dec2(x, d1)
        x = self.dec1(x, d0)
        x = self.final_up(x)
        logits = self.classifier(x)

        if logits.shape[2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        return logits


class SiameseChangeDetector(nn.Module):
    """Production wrapper for bi-temporal change detection."""

    def __init__(
        self,
        architecture: str = "SiameseUnet",
        encoder_name: str = "resnet18",
        num_classes: int = 2,
        in_channels: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.target_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        self.model = SiameseUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_name=encoder_name,
            pretrained=False,
        )
        self.to(self.target_device)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass returning change logits (N, 2, H, W)."""
        return self.model(x1, x2)

    @torch.no_grad()
    def predict(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference returning binary change mask and change probability map.

        Args:
            x1: First time-step image tensor (N, C, H, W) or (C, H, W).
            x2: Second time-step image tensor (N, C, H, W) or (C, H, W).
            threshold: Probability threshold for positive change.

        Returns:
            Tuple of:
            - Binary change mask (N, H, W) uint8
            - Change probability map (N, H, W) float32
        """
        self.eval()
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)

        x1 = x1.to(self.target_device)
        x2 = x2.to(self.target_device)

        logits = self.forward(x1, x2)
        probs = F.softmax(logits, dim=1)[:, 1]  # Change class probability
        pred_mask = (probs >= threshold).to(torch.uint8)

        return pred_mask, probs

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save model weights and metadata."""
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
        logger.info(f"Saved SiameseChangeDetector checkpoint to {save_path}")
        return save_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SiameseChangeDetector":
        """Load model from checkpoint file."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        detector = cls(
            architecture=checkpoint.get("architecture", "SiameseUnet"),
            encoder_name=checkpoint.get("encoder_name", "resnet18"),
            num_classes=checkpoint.get("num_classes", 2),
            in_channels=checkpoint.get("in_channels", 3),
            device=device,
        )
        detector.load_state_dict(checkpoint["state_dict"])
        detector.eval()
        logger.info(f"Loaded SiameseChangeDetector from {load_path}")
        return detector
