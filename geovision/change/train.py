"""Training pipeline for bi-temporal remote sensing change detection models."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from geovision.change.evaluate import evaluate_change_detection
from geovision.change.losses import ChangeLoss
from geovision.change.model import SiameseChangeDetector
from geovision.config import ChangeConfig, load_config
from geovision.data.datasets import LEVIRCDDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.tracking import ExperimentTracker

logger = get_logger("geovision.change.train")


def train_change_detection(
    config: ChangeConfig | Path | str | None = None,
    smoke: bool = False,
    experiment_name: str = "BiTemporal-Change-Detection",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train Siamese bi-temporal change detection model with MLflow tracking and checkpointing.

    Args:
        config: ChangeConfig instance, dict, or path to YAML config.
        smoke: If True, executes a fast 1-epoch CPU smoke-test run.
        experiment_name: Name of the MLflow experiment.
        run_name: Optional label for the tracking run.

    Returns:
        Dictionary containing checkpoint paths, training duration, and best validation metrics.
    """
    if config is None:
        cfg = ChangeConfig()
    elif isinstance(config, (str, Path)):
        loaded = load_config(config)
        cfg = loaded.change if hasattr(loaded, "change") else ChangeConfig(**loaded.to_dict())
    elif isinstance(config, ChangeConfig):
        cfg = config
    else:
        cfg = ChangeConfig(**config)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg.smoke_epochs if smoke else cfg.epochs
    batch_size = 2 if smoke else cfg.batch_size
    device = "cpu" if smoke else cfg.device

    logger.info(
        f"Starting Change Detection Training | Architecture: {cfg.architecture} | "
        f"Encoder: {cfg.encoder_name} | Epochs: {epochs} | Smoke: {smoke}"
    )

    # Ensure dataset is available
    dataset_path = download_dataset("levir_cd", smoke=smoke)
    full_dataset = LEVIRCDDataset(root_dir=dataset_path)

    if len(full_dataset) == 0:
        raise ValueError(f"No samples found in LEVIR-CD dataset at {dataset_path}")

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

    # Instantiate Siamese model
    model = SiameseChangeDetector(
        architecture=cfg.architecture,
        encoder_name=cfg.encoder_name,
        num_classes=cfg.num_classes,
        device=device,
    )
    dev = model.target_device

    criterion = ChangeLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    tracker = ExperimentTracker(experiment_name=experiment_name)
    best_iou = -1.0
    best_checkpoint_path = save_dir / "best_change_detector.pth"
    latest_checkpoint_path = save_dir / "latest_change_detector.pth"

    with tracker.start_run(run_name=run_name or f"{cfg.architecture}_{cfg.encoder_name}"):
        tracker.log_params({
            "architecture": cfg.architecture,
            "encoder_name": cfg.encoder_name,
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
                img_a = batch["image_a"].to(dev)
                img_b = batch["image_b"].to(dev)
                masks = batch["mask"].to(dev)

                optimizer.zero_grad()
                logits = model(img_a, img_b)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            scheduler.step()
            avg_train_loss = (train_loss / num_batches) if num_batches > 0 else 0.0

            # Evaluate on validation split
            val_results = evaluate_change_detection(
                model=model,
                dataloader=val_loader,
                device=device,
                criterion=criterion,
                progress_bar=False,
            )

            metrics_to_log = {
                "train_loss": avg_train_loss,
                "val_loss": val_results["loss"],
                "val_change_iou": val_results["change_iou"],
                "val_f1_score": val_results["f1_score"],
                "val_precision": val_results["precision"],
                "val_recall": val_results["recall"],
                "val_oa": val_results["overall_accuracy"],
            }
            tracker.log_metrics(metrics_to_log, step=epoch)

            # Save latest checkpoint
            model.save(latest_checkpoint_path, metadata={"epoch": epoch, "change_iou": val_results["change_iou"]})

            # Save best checkpoint
            if val_results["change_iou"] > best_iou:
                best_iou = val_results["change_iou"]
                model.save(best_checkpoint_path, metadata={"epoch": epoch, "change_iou": best_iou})
                logger.info(f"New best change detector saved at epoch {epoch} with Change-IoU: {best_iou:.4f}")

    logger.info(f"Training completed. Best validation Change-IoU: {best_iou:.4f}")
    return {
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_change_iou": best_iou,
        "epochs_completed": epochs,
    }
