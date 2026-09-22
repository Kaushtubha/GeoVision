"""Loss functions for bi-temporal remote sensing change detection."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Soft Dice Loss for binary change detection."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary soft Dice loss.

        Args:
            logits: Output logits of shape (N, 1, H, W) or (N, 2, H, W).
            targets: Ground-truth binary masks of shape (N, H, W) or (N, 1, H, W).
        """
        if logits.shape[1] == 2:
            probs = F.softmax(logits, dim=1)[:, 1:2]
        else:
            probs = torch.sigmoid(logits)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()

        intersection = torch.sum(probs * targets, dim=(2, 3))
        cardinality = torch.sum(probs + targets, dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice_score)


class ChangeLoss(nn.Module):
    """Compound Cross-Entropy and Binary Dice loss for change detection."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = BinaryDiceLoss(smooth=smooth)
        self.pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute compound change detection loss.

        Args:
            logits: Predicted logits of shape (N, 2, H, W) or (N, 1, H, W).
            targets: Ground truth binary masks of shape (N, H, W) with values 0 and 1.
        """
        if logits.shape[1] == 2:
            # 2-class cross-entropy
            ce_loss = F.cross_entropy(logits, targets.long())
        else:
            # Binary cross entropy with logits
            pos_w = self.pos_weight_tensor.to(logits.device) if self.pos_weight_tensor is not None else None
            t_float = targets.unsqueeze(1).float() if targets.dim() == 3 else targets.float()
            ce_loss = F.binary_cross_entropy_with_logits(logits, t_float, pos_weight=pos_w)

        dice_loss = self.dice_loss(logits, targets)
        return self.bce_weight * ce_loss + self.dice_weight * dice_loss
