"""Evaluation routine for semantic land-cover segmentation models."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geovision.constants import LANDCOVER_AI_CLASSES
from geovision.logger import get_logger
from geovision.metrics.segmentation import SegmentationMetricsCalculator

logger = get_logger("geovision.segmentation.evaluate")


def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    criterion: nn.Module | None = None,
    num_classes: int = 5,
    class_names: dict[int, str] | None = None,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Evaluate a segmentation model over a dataset loader.

    Args:
        model: LandCoverSegmenter or PyTorch segmentation module.
        dataloader: PyTorch DataLoader yielding {"image": tensor, "mask": tensor}.
        device: Evaluation device ('cpu' or 'cuda').
        criterion: Optional loss function for loss computation.
        num_classes: Number of segmentation classes.
        class_names: Mapping of class index to human-readable string.
        progress_bar: Whether to display a tqdm progress bar.

    Returns:
        Dictionary containing overall loss, mIoU, per-class metrics, and confusion matrix.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    calc = SegmentationMetricsCalculator(
        num_classes=num_classes,
        class_names=class_names or LANDCOVER_AI_CLASSES,
    )

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating Segmentation", disable=not progress_bar)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(dev)
            masks = batch["mask"].to(dev)

            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, masks)
                total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            calc.update(preds, masks)
            num_batches += 1

    results = calc.compute()
    avg_loss = (total_loss / num_batches) if (criterion is not None and num_batches > 0) else 0.0
    results["loss"] = avg_loss

    logger.info(
        f"Evaluation Results -> Loss: {avg_loss:.4f} | mIoU: {results['mIoU']:.4f} | "
        f"Mean Dice: {results['mean_dice']:.4f} | Overall Accuracy: {results['overall_accuracy']:.4f}"
    )
    for c_name, iou_val in results["per_class_iou"].items():
        logger.info(f"  Class '{c_name}': IoU = {iou_val:.4f} | Dice = {results['per_class_dice'][c_name]:.4f}")

    return results
