"""Compound loss functions for multi-class semantic segmentation."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Multi-Class Dice Loss."""

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int | None = None,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate soft multi-class Dice loss.

        Args:
            logits: Predicted logits of shape (N, C, H, W).
            targets: Ground-truth class indices of shape (N, H, W).

        Returns:
            Scalar Dice loss tensor.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets to (N, C, H, W)
        target_clamped = torch.clamp(targets, 0, num_classes - 1)
        targets_one_hot = F.one_hot(target_clamped, num_classes=num_classes)
        # Permute from (N, H, W, C) to (N, C, H, W)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Handle ignore_index if provided
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).unsqueeze(1)
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        # Flatten spatial dimensions
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.weight is not None:
            weight = self.weight.to(dice_loss.device)
            return torch.sum(dice_loss * weight) / (torch.sum(weight) + 1e-8)

        return torch.mean(dice_loss)


class FocalLoss(nn.Module):
    """Multi-Class Focal Loss for addressing extreme class imbalance."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


class DiceCELoss(nn.Module):
    """Compound loss combining Cross-Entropy Loss and Soft Dice Loss."""

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = -100,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index if ignore_index >= 0 else None,
            weight=class_weights,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
