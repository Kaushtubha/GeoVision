"""Evaluation pipeline for bi-temporal remote sensing change detection models."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geovision.logger import get_logger
from geovision.metrics.change import ChangeMetricsCalculator

logger = get_logger("geovision.change.evaluate")


def evaluate_change_detection(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    criterion: nn.Module | None = None,
    threshold: float = 0.5,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Evaluate a bi-temporal change detection model on a test/validation dataloader.

    Args:
        model: SiameseChangeDetector or PyTorch Siamese module.
        dataloader: PyTorch DataLoader yielding {"image_a": t1, "image_b": t2, "mask": mask}.
        device: Computation device ('cpu' or 'cuda').
        criterion: Optional loss function for loss computation.
        threshold: Decision threshold for change classification.
        progress_bar: Whether to display a tqdm progress bar.

    Returns:
        Dictionary of change detection metrics (F1, Change-IoU, Precision, Recall, OA).
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model.to(dev)

    calc = ChangeMetricsCalculator(threshold=threshold)
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating Change Detection", disable=not progress_bar)
    with torch.no_grad():
        for batch in pbar:
            img_a = batch["image_a"].to(dev)
            img_b = batch["image_b"].to(dev)
            masks = batch["mask"].to(dev)

            logits = model(img_a, img_b)
            if criterion is not None:
                loss = criterion(logits, masks)
                total_loss += float(loss.item())

            probs = torch.softmax(logits, dim=1)[:, 1]
            calc.update(probs, masks)
            num_batches += 1

    results = calc.compute()
    avg_loss = (total_loss / num_batches) if (criterion is not None and num_batches > 0) else 0.0
    results["loss"] = avg_loss

    logger.info(
        f"Change Eval -> Loss: {avg_loss:.4f} | Change-IoU: {results['change_iou']:.4f} | "
        f"F1-Score: {results['f1_score']:.4f} | Precision: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f} | OA: {results['overall_accuracy']:.4f}"
    )

    return results
