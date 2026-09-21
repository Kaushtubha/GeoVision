"""Unit tests for dataset downloaders, smoke datasets, and statistical summaries."""

import tempfile
from pathlib import Path

from geovision.data.downloaders import (
    download_dataset,
    verify_dataset_integrity,
)
from geovision.data.statistics import calculate_dataset_stats


def test_smoke_eurosat_creation():
    """Verify EuroSAT smoke dataset creation and directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = download_dataset("eurosat", target_dir=root, smoke=True)
        report = verify_dataset_integrity("eurosat", path)
        assert report["status"] == "OK"
        assert report["total_files"] == 50  # 10 classes * 5 samples

        stats = calculate_dataset_stats(path, task="classification")
        assert stats["total_samples"] == 50
        assert len(stats["class_counts"]) == 10


def test_smoke_nwpu_vhr10_creation():
    """Verify NWPU VHR-10 smoke dataset creation and YOLO format labels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = download_dataset("nwpu_vhr10", target_dir=root, smoke=True)
        report = verify_dataset_integrity("nwpu_vhr10", path)
        assert report["status"] == "OK"
        assert report["total_files"] == 20

        stats = calculate_dataset_stats(path, task="detection")
        assert stats["total_samples"] == 20
        assert len(stats["class_counts"]) > 0


def test_smoke_landcover_ai_creation():
    """Verify LandCover.ai smoke dataset creation with mask classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = download_dataset("landcover_ai", target_dir=root, smoke=True)
        report = verify_dataset_integrity("landcover_ai", path)
        assert report["status"] == "OK"
        assert report["total_files"] == 40  # 20 images + 20 masks

        stats = calculate_dataset_stats(path, task="segmentation")
        assert stats["total_samples"] == 20
        assert "unique_classes_found" in stats


def test_smoke_levir_cd_creation():
    """Verify LEVIR-CD smoke dataset creation with bi-temporal pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = download_dataset("levir_cd", target_dir=root, smoke=True)
        report = verify_dataset_integrity("levir_cd", path)
        assert report["status"] == "OK"
        assert report["total_files"] == 60  # 20 in A + 20 in B + 20 in label

        stats = calculate_dataset_stats(path, task="change")
        assert stats["total_samples"] == 20
