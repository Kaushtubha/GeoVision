"""Comprehensive unit tests for Bi-Temporal Change Detection pipeline."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from geovision.change.infer import detect_changes
from geovision.change.losses import BinaryDiceLoss, ChangeLoss
from geovision.change.model import SiameseChangeDetector
from geovision.change.train import train_change_detection
from geovision.metrics.change import (
    ChangeMetricsCalculator,
    compute_change_statistics,
)


def test_change_losses_computation_and_gradients():
    """Test BinaryDiceLoss and compound ChangeLoss computation and backpropagation."""
    batch_size = 2
    h, w = 64, 64

    logits = torch.randn(batch_size, 2, h, w, requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, h, w), dtype=torch.long)

    dice_fn = BinaryDiceLoss()
    loss_dice = dice_fn(logits, targets)
    assert loss_dice.ndim == 0
    assert loss_dice.item() >= 0.0

    compound_fn = ChangeLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=2.0)
    loss_compound = compound_fn(logits, targets)
    assert loss_compound.ndim == 0
    assert loss_compound.item() >= 0.0

    loss_compound.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_change_metrics_calculator_known_cases():
    """Test ChangeMetricsCalculator on controlled ground truth and prediction matrices."""
    calc = ChangeMetricsCalculator(threshold=0.5)

    # Perfect match
    preds = np.array([[0, 1], [0, 1]], dtype=np.float32)
    targets = np.array([[0, 1], [0, 1]], dtype=np.int64)
    calc.update(preds, targets)
    results = calc.compute()

    assert pytest.approx(results["precision"], 0.001) == 1.0
    assert pytest.approx(results["recall"], 0.001) == 1.0
    assert pytest.approx(results["f1_score"], 0.001) == 1.0
    assert pytest.approx(results["change_iou"], 0.001) == 1.0
    assert pytest.approx(results["overall_accuracy"], 0.001) == 1.0

    # Test error cases
    calc.reset()
    # 1 TP, 1 FP, 1 TN, 1 FN
    preds_err = np.array([[1, 1], [0, 0]], dtype=np.float32)
    targets_err = np.array([[1, 0], [0, 1]], dtype=np.int64)
    calc.update(preds_err, targets_err)
    res_err = calc.compute()

    assert pytest.approx(res_err["precision"], 0.01) == 0.5
    assert pytest.approx(res_err["recall"], 0.01) == 0.5
    assert pytest.approx(res_err["f1_score"], 0.01) == 0.5
    assert pytest.approx(res_err["overall_accuracy"], 0.01) == 0.5


def test_change_statistics_computation():
    """Test compute_change_statistics percentage and ground area calculation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:20, :] = 1  # 20% change

    stats = compute_change_statistics(mask, gsd_meters=0.5)

    assert stats["total_pixels"] == 10000
    assert stats["changed_pixels"] == 2000
    assert stats["unchanged_pixels"] == 8000
    assert stats["change_percentage"] == 20.0
    # 2000 * 0.25m2 = 500m2 = 0.05 ha
    assert stats["changed_area_sq_meters"] == 500.0
    assert stats["changed_area_hectares"] == 0.05


def test_siamese_change_detector_forward_predict_save_load():
    """Test Siamese model forward pass, prediction, and checkpoint serialization."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = SiameseChangeDetector(
            architecture="SiameseUnet",
            encoder_name="resnet18",
            num_classes=2,
            device="cpu",
        )
        model.eval()

        t1 = torch.randn(2, 3, 64, 64)
        t2 = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            logits = model(t1, t2)
        assert logits.shape == (2, 2, 64, 64)

        masks, probs = model.predict(t1, t2, threshold=0.5)
        assert masks.shape == (2, 64, 64)
        assert probs.shape == (2, 64, 64)
        assert torch.all((masks == 0) | (masks == 1))

        # Save and load checkpoint
        save_path = temp_dir / "test_change_model.pth"
        model.save(save_path, metadata={"test_metric": 0.85})
        assert save_path.exists()

        loaded_model = SiameseChangeDetector.load(save_path, device="cpu")
        loaded_model.eval()
        with torch.no_grad():
            loaded_logits = loaded_model(t1, t2)
        assert torch.allclose(logits, loaded_logits, atol=1e-5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_change_inference_sliding_window():
    """Test sliding-window bi-temporal inference and multi-panel report generation."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        img_a = np.random.randint(50, 180, size=(350, 350, 3), dtype=np.uint8)
        img_b = img_a.copy()
        img_b[100:200, 100:200] = np.random.randint(200, 255, size=(100, 100, 3), dtype=np.uint8)

        mask_out = temp_dir / "change_mask.png"
        overlay_out = temp_dir / "overlay.png"
        report_out = temp_dir / "tri_panel.png"

        result = detect_changes(
            image_a=img_a,
            image_b=img_b,
            tile_size=128,
            overlap=16,
            device="cpu",
            gsd_meters=0.5,
            output_mask_path=mask_out,
            output_overlay_path=overlay_out,
            output_report_path=report_out,
        )

        assert result["change_mask"].shape == (350, 350)
        assert mask_out.exists()
        assert overlay_out.exists()
        assert report_out.exists()
        assert "statistics" in result
        assert result["statistics"]["total_pixels"] == 350 * 350
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_smoke_train_change_detection():
    """Test end-to-end smoke training and evaluation cycle."""
    results = train_change_detection(
        smoke=True,
        experiment_name="Test-Change-Smoke",
    )

    assert "best_checkpoint" in results
    assert Path(results["best_checkpoint"]).exists()
    assert results["epochs_completed"] == 1
    assert "best_change_iou" in results
