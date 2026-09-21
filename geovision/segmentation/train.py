"""Training pipeline for Land-Cover Semantic Segmentation models."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from geovision.config import SegmentationConfig, load_config
from geovision.constants import LANDCOVER_AI_CLASSES
from geovision.data.datasets import LandCoverAIDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.segmentation.evaluate import evaluate_segmentation
from geovision.segmentation.losses import DiceCELoss
from geovision.segmentation.model import LandCoverSegmenter
from geovision.tracking import ExperimentTracker

logger = get_logger("geovision.segmentation.train")


def train_segmentation(
    config: SegmentationConfig | Path | str | None = None,
    smoke: bool = False,
    experiment_name: str = "LandCover-Segmentation",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train Land-Cover semantic segmentation model with MLflow tracking and checkpointing.

    Args:
        config: SegmentationConfig instance, dict, or path to YAML config.
        smoke: If True, executes a fast 1-epoch CPU smoke-test run.
        experiment_name: Name of the MLflow experiment.
        run_name: Optional label for the tracking run.

    Returns:
        Dictionary containing checkpoint paths, training duration, and best validation metrics.
    """
    if config is None:
        cfg = SegmentationConfig()
    elif isinstance(config, (str, Path)):
        loaded = load_config(config)
        cfg = loaded.segmentation if hasattr(loaded, "segmentation") else SegmentationConfig(**loaded.to_dict())
    elif isinstance(config, SegmentationConfig):
        cfg = config
    else:
        cfg = SegmentationConfig(**config)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg.smoke_epochs if smoke else cfg.epochs
    batch_size = 2 if smoke else cfg.batch_size
    device = "cpu" if smoke else cfg.device
    encoder_weights = None if smoke else cfg.encoder_weights

    logger.info(f"Starting Segmentation Training | Architecture: {cfg.architecture} | Encoder: {cfg.encoder_name} | Epochs: {epochs} | Smoke: {smoke}")

    # Ensure dataset is available
    dataset_path = download_dataset("landcover_ai", smoke=smoke)
    full_dataset = LandCoverAIDataset(root_dir=dataset_path)

    if len(full_dataset) == 0:
        raise ValueError(f"No samples found in LandCover.ai dataset at {dataset_path}")

    # Train / Val split (80 / 20)
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_ds) > batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # Instantiate model
    model = LandCoverSegmenter(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder_name,
        encoder_weights=encoder_weights,
        num_classes=cfg.num_classes,
        device=device,
    )
    dev = model.target_device

    criterion = DiceCELoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    tracker = ExperimentTracker(experiment_name=experiment_name)
    best_miou = -1.0
    best_checkpoint_path = save_dir / "best_segmenter.pth"
    latest_checkpoint_path = save_dir / "latest_segmenter.pth"

    with tracker.start_run(run_name=run_name or f"{cfg.architecture}_{cfg.encoder_name}"):
        tracker.log_params({
            "architecture": cfg.architecture,
            "encoder_name": cfg.encoder_name,
            "encoder_weights": str(encoder_weights),
            "num_classes": cfg.num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": cfg.learning_rate,
            "smoke": smoke,
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
        })

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                images = batch["image"].to(dev)
                masks = batch["mask"].to(dev)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            scheduler.step()
            avg_train_loss = (train_loss / num_batches) if num_batches > 0 else 0.0

            # Evaluate on validation split
            val_results = evaluate_segmentation(
                model=model,
                dataloader=val_loader,
                device=device,
                criterion=criterion,
                num_classes=cfg.num_classes,
                class_names=LANDCOVER_AI_CLASSES,
                progress_bar=False,
            )

            metrics_to_log = {
                "train_loss": avg_train_loss,
                "val_loss": val_results["loss"],
                "val_mIoU": val_results["mIoU"],
                "val_mean_dice": val_results["mean_dice"],
                "val_overall_accuracy": val_results["overall_accuracy"],
            }
            # Log per-class metrics
            for c_name, iou_v in val_results["per_class_iou"].items():
                metrics_to_log[f"val_iou_{c_name}"] = iou_v

            tracker.log_metrics(metrics_to_log, step=epoch)

            # Save latest checkpoint
            model.save(latest_checkpoint_path, metadata={"epoch": epoch, "mIoU": val_results["mIoU"]})

            # Save best checkpoint
            if val_results["mIoU"] > best_miou:
                best_miou = val_results["mIoU"]
                model.save(best_checkpoint_path, metadata={"epoch": epoch, "mIoU": best_miou})
                logger.info(f"New best model saved at epoch {epoch} with mIoU: {best_miou:.4f}")

    logger.info(f"Training completed. Best validation mIoU: {best_miou:.4f}")
    return {
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_mIoU": best_miou,
        "epochs_completed": epochs,
    }
