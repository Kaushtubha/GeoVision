"""Contrastive fine-tuning and metric learning pipeline for satellite scene embeddings."""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from geovision.config import RetrievalConfig, load_config
from geovision.constants import EUROSAT_CLASSES, WEIGHTS_DIR
from geovision.data.datasets import EuroSATDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.retrieval.embedder import SatelliteSceneEmbedder
from geovision.retrieval.evaluate import evaluate_retrieval
from geovision.tracking import ExperimentTracker

logger = get_logger("geovision.retrieval.train")


def train_retrieval_embedder(
    config: RetrievalConfig | Path | str | None = None,
    smoke: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    temperature: float = 0.07,
    experiment_name: str = "Satellite-Scene-Retrieval",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train or fine-tune multimodal satellite scene embedder using contrastive loss.

    Args:
        config: RetrievalConfig instance, dict, or path to YAML config.
        smoke: If True, runs a fast 1-epoch CPU smoke verification.
        epochs: Number of training epochs.
        learning_rate: Learning rate for AdamW optimizer.
        temperature: Softmax scaling temperature for contrastive logits.
        experiment_name: MLflow experiment name.
        run_name: Optional run label.

    Returns:
        Dictionary with checkpoint paths and validation metrics.
    """
    if config is None:
        cfg = RetrievalConfig()
    elif isinstance(config, (str, Path)):
        loaded = load_config(config)
        cfg = loaded.retrieval if hasattr(loaded, "retrieval") else RetrievalConfig(**loaded.to_dict())
    elif isinstance(config, RetrievalConfig):
        cfg = config
    else:
        cfg = RetrievalConfig(**config)

    save_dir = Path(WEIGHTS_DIR / "retrieval")
    save_dir.mkdir(parents=True, exist_ok=True)

    actual_epochs = 1 if smoke else epochs
    batch_size = 4 if smoke else cfg.batch_size
    device = "cpu" if smoke else cfg.device

    logger.info(f"Starting Scene Retrieval Training | Model: {cfg.model_name} | Epochs: {actual_epochs} | Smoke: {smoke}")

    dataset_path = download_dataset("eurosat", smoke=smoke)
    train_dataset = EuroSATDataset(root_dir=dataset_path, split="train")
    val_dataset = EuroSATDataset(root_dir=dataset_path, split="val")

    if len(train_dataset) == 0:
        # Fallback to all samples if split has 0
        train_dataset = EuroSATDataset(root_dir=dataset_path, split="all")
        val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    embedder = SatelliteSceneEmbedder(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained if not smoke else None,
        embedding_dim=cfg.embedding_dim,
        device=device,
    )
    dev = embedder.target_device

    # Get class prompt embeddings for supervised contrastive loss
    with torch.no_grad():
        class_text_embeddings = embedder.build_class_text_embeddings(class_names=EUROSAT_CLASSES).to(dev)

    # Optimizer (filtering trainable parameters)
    trainable_params = [p for p in embedder.parameters() if p.requires_grad]
    if not trainable_params:
        # If parameters were frozen, enable requires_grad
        for p in embedder.parameters():
            p.requires_grad = True
        trainable_params = list(embedder.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    tracker = ExperimentTracker(experiment_name=experiment_name)
    best_recall1 = -1.0
    best_checkpoint_path = save_dir / "best_retrieval_embedder.pth"
    latest_checkpoint_path = save_dir / "latest_retrieval_embedder.pth"

    with tracker.start_run(run_name=run_name or f"{cfg.model_name}_EuroSAT"):
        tracker.log_params({
            "model_name": cfg.model_name,
            "embedding_dim": cfg.embedding_dim,
            "epochs": actual_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "smoke": smoke,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        })

        for epoch in range(1, actual_epochs + 1):
            embedder.train()
            train_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{actual_epochs}")
            for batch in pbar:
                images = batch["image"].to(dev)
                labels = batch["label"].to(dev)

                optimizer.zero_grad()

                # Image embeddings: (B, D)
                img_feats = embedder(images)

                # Image-to-Class similarity: (B, num_classes)
                logits = torch.matmul(img_feats, class_text_embeddings.T) / temperature
                loss = F.cross_entropy(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = (train_loss / num_batches) if num_batches > 0 else 0.0

            # Evaluate on validation split
            val_results = evaluate_retrieval(
                embedder=embedder,
                dataloader=val_loader,
                class_names=EUROSAT_CLASSES,
                k_values=(1, 5),
                progress_bar=False,
            )

            metrics_to_log = {
                "train_loss": avg_train_loss,
                "val_recall@1": val_results.get("recall@1", 0.0),
                "val_recall@5": val_results.get("recall@5", 0.0),
                "val_mrr": val_results.get("mrr", 0.0),
                "val_map": val_results.get("map", 0.0),
                "val_zeroshot_top1": val_results.get("zeroshot_top1_accuracy", 0.0),
            }
            tracker.log_metrics(metrics_to_log, step=epoch)

            embedder.save(latest_checkpoint_path, metadata={"epoch": epoch, "recall@1": val_results.get("recall@1", 0.0)})

            rec1 = val_results.get("recall@1", 0.0)
            if rec1 >= best_recall1:
                best_recall1 = rec1
                embedder.save(best_checkpoint_path, metadata={"epoch": epoch, "recall@1": best_recall1})
                logger.info(f"Saved best retrieval embedder at epoch {epoch} with Recall@1: {best_recall1:.4f}")

    logger.info(f"Retrieval training completed. Best Recall@1: {best_recall1:.4f}")
    return {
        "best_checkpoint": str(best_checkpoint_path),
        "latest_checkpoint": str(latest_checkpoint_path),
        "best_recall@1": best_recall1,
        "epochs_completed": actual_epochs,
    }
