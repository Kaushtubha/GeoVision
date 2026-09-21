"""Comprehensive unit tests for Land-Cover Semantic Segmentation pipeline."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from geovision.constants import LANDCOVER_AI_CLASSES
from geovision.metrics.segmentation import (
    SegmentationMetricsCalculator,
    compute_landcover_distribution,
)
from geovision.segmentation.infer import (
    segment_image,
)
from geovision.segmentation.losses import DiceCELoss, DiceLoss, FocalLoss
from geovision.segmentation.model import LandCoverSegmenter
from geovision.segmentation.train import train_segmentation


def test_dice_and_compound_losses():
    """Test DiceLoss, FocalLoss, and DiceCELoss computation and gradients."""
    batch_size = 2
    num_classes = 5
    h, w = 64, 64

    logits = torch.randn(batch_size, num_classes, h, w, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size, h, w), dtype=torch.long)

    dice_fn = DiceLoss()
    loss_dice = dice_fn(logits, targets)
    assert loss_dice.ndim == 0
    assert loss_dice.item() >= 0.0

    focal_fn = FocalLoss(gamma=2.0)
    loss_focal = focal_fn(logits, targets)
    assert loss_focal.ndim == 0
    assert loss_focal.item() >= 0.0

    compound_fn = DiceCELoss(ce_weight=0.5, dice_weight=0.5)
    loss_compound = compound_fn(logits, targets)
    assert loss_compound.ndim == 0
    assert loss_compound.item() >= 0.0

    # Test backpropagation
    loss_compound.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_segmentation_metrics_calculator_known_case():
    """Test SegmentationMetricsCalculator with controlled ground truth and predictions."""
    calc = SegmentationMetricsCalculator(num_classes=3, class_names=["Bg", "Class1", "Class2"])

    # Perfect match test
    preds = np.array([[0, 1], [2, 1]])
    targets = np.array([[0, 1], [2, 1]])

    calc.update(preds, targets)
    results = calc.compute()

    assert pytest.approx(results["mIoU"], 0.001) == 1.0
    assert pytest.approx(results["mean_dice"], 0.001) == 1.0
    assert pytest.approx(results["overall_accuracy"], 0.001) == 1.0
    assert results["per_class_iou"]["Bg"] == 1.0
    assert results["per_class_iou"]["Class1"] == 1.0

    # Reset and test with known errors
    calc.reset()
    preds_err = np.array([[0, 0], [1, 2]])
    targets_err = np.array([[0, 1], [1, 2]])
    calc.update(preds_err, targets_err)
    res_err = calc.compute()

    assert res_err["overall_accuracy"] == 0.75
    assert res_err["mIoU"] < 1.0


def test_landcover_distribution_calculator():
    """Test compute_landcover_distribution percentage and physical area outputs."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 1  # 50% class 1 (Building)
    mask[50:, :] = 2  # 50% class 2 (Woodland)

    dist = compute_landcover_distribution(
        mask=mask,
        num_classes=5,
        class_names=LANDCOVER_AI_CLASSES,
        gsd_meters=0.5,
    )

    assert dist["total_pixels"] == 10000
    assert dist["percentages"]["Building"] == 50.0
    assert dist["percentages"]["Woodland"] == 50.0
    assert dist["percentages"]["Water"] == 0.0
    # 5000 pixels * 0.25 m2 = 1250 m2 = 0.1250 ha
    assert dist["area_sq_meters"]["Building"] == 1250.0
    assert dist["area_hectares"]["Building"] == 0.1250


def test_landcover_segmenter_model():
    """Test model forward pass, predict method, and save/load serialization."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = LandCoverSegmenter(
            architecture="Unet",
            encoder_name="mobilenet_v3_small",
            encoder_weights=None,
            num_classes=5,
            device="cpu",
        )

        model.eval()
        dummy_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            logits = model(dummy_input)
        assert logits.shape == (2, 5, 128, 128)

        preds, probs = model.predict(dummy_input)
        assert preds.shape == (2, 128, 128)
        assert probs.shape == (2, 5, 128, 128)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

        # Test save and load
        save_path = temp_dir / "test_model.pth"
        model.save(save_path, metadata={"test_key": "test_val"})
        assert save_path.exists()

        loaded_model = LandCoverSegmenter.load(save_path, device="cpu")
        loaded_model.eval()
        with torch.no_grad():
            loaded_logits = loaded_model(dummy_input)
        assert torch.allclose(logits, loaded_logits, atol=1e-5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_segmentation_infer_and_tiling():
    """Test sliding-window segmentation inference on large mock image."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        mock_img = np.random.randint(50, 200, size=(700, 700, 3), dtype=np.uint8)
        mask_path = temp_dir / "pred_mask.png"
        overlay_path = temp_dir / "overlay.png"

        result = segment_image(
            image_input=mock_img,
            tile_size=256,
            overlap=32,
            device="cpu",
            gsd_meters=0.5,
            output_mask_path=mask_path,
            output_overlay_path=overlay_path,
        )

        assert result["mask"].shape == (700, 700)
        assert mask_path.exists()
        assert overlay_path.exists()
        assert "distribution" in result
        assert result["distribution"]["total_pixels"] == 700 * 700
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_smoke_train_segmentation():
    """Test end-to-end smoke training and validation cycle."""
    results = train_segmentation(
        smoke=True,
        experiment_name="Test-Segmentation-Smoke",
    )

    assert "best_checkpoint" in results
    assert Path(results["best_checkpoint"]).exists()
    assert results["epochs_completed"] == 1
    assert "best_mIoU" in results
