"""Comprehensive visualization utilities for satellite detection, segmentation, change, and geospatial tiling."""

from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from geovision.constants import (
    LANDCOVER_AI_CLASSES,
    NWPU_VHR10_CLASSES,
)
from geovision.logger import get_logger

logger = get_logger("geovision.visualization")

# Distinct color palette for LandCover.ai (0: bg, 1: building, 2: woodland, 3: water, 4: road)
LANDCOVER_PALETTE = {
    0: (128, 128, 128),  # Background (Gray)
    1: (230, 25, 75),    # Building (Red)
    2: (60, 180, 75),    # Woodland (Green)
    3: (0, 130, 200),    # Water (Blue)
    4: (245, 130, 48),   # Road (Orange)
}


def ensure_numpy_image(img: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Normalize input image into a uint8 RGB numpy array."""
    if isinstance(img, (str, Path)):
        with Image.open(img) as im:
            return np.array(im.convert("RGB"))
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            return img.astype(np.uint8)
        return img
    raise TypeError(f"Unsupported image type: {type(img)}")


def plot_detection_boxes(
    image: np.ndarray | Image.Image | str | Path,
    bboxes: list[list[float]],
    class_labels: list[int],
    scores: list[float] | None = None,
    class_names: list[str] | None = None,
    output_path: str | Path | None = None,
    bbox_format: str = "yolo",  # 'yolo' or 'xyxy'
) -> plt.Figure:
    """Render bounding boxes overlaid on a satellite image with labels and confidence scores."""
    img_arr = ensure_numpy_image(image)
    h, w = img_arr.shape[:2]
    names = class_names or NWPU_VHR10_CLASSES

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_arr)
    ax.set_title(f"GeoVision Object Detection ({len(bboxes)} objects)", fontsize=14, fontweight="bold")
    ax.axis("off")

    for i, bbox in enumerate(bboxes):
        cls_id = class_labels[i]
        label_name = names[cls_id] if 0 <= cls_id < len(names) else f"Class_{cls_id}"
        score_str = f" {scores[i]:.2f}" if scores is not None and i < len(scores) else ""

        if bbox_format == "yolo":
            cx, cy, bw, bh = bbox
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
        else:
            x1, y1, x2, y2 = bbox
            box_w = x2 - x1
            box_h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 4),
            f"{label_name}{score_str}",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="cyan", edgecolor="none", alpha=0.8, pad=1),
        )

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        logger.info(f"Saved detection visualization: {out}")
    return fig


def plot_segmentation_overlay(
    image: np.ndarray | Image.Image | str | Path,
    mask: np.ndarray,
    alpha: float = 0.5,
    class_names: dict[int, str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Render land-cover segmentation color mask overlaid on optical image with legend."""
    img_arr = ensure_numpy_image(image)
    h, w = img_arr.shape[:2]
    c_names = class_names or LANDCOVER_AI_CLASSES

    # Build RGB mask overlay
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_classes = np.unique(mask)

    for cls_idx in unique_classes:
        color = LANDCOVER_PALETTE.get(int(cls_idx), (255, 255, 0))
        color_mask[mask == cls_idx] = color

    # Blend
    blended = (img_arr * (1.0 - alpha) + color_mask * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_arr)
    axes[0].set_title("Input Satellite Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Land-Cover Segmentation Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title(f"Blended Overlay (Alpha={alpha})", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Legend
    legend_elements = [
        patches.Patch(
            facecolor=np.array(LANDCOVER_PALETTE.get(c, (0, 0, 0))) / 255.0,
            label=f"{c_names.get(c, f'Class {c}')} ({np.mean(mask == c)*100:.1f}%)"
        )
        for c in unique_classes
    ]
    axes[2].legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        logger.info(f"Saved segmentation visualization: {out}")
    return fig


def plot_change_detection_pair(
    image_a: np.ndarray | Image.Image | str | Path,
    image_b: np.ndarray | Image.Image | str | Path,
    change_mask: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Render Before (T1), After (T2), Change Mask, and Difference Overlay."""
    img_a = ensure_numpy_image(image_a)
    img_b = ensure_numpy_image(image_b)
    h, w = img_a.shape[:2]

    binary_mask = (change_mask > 0).astype(np.uint8)
    change_pct = np.mean(binary_mask) * 100.0

    overlay = img_b.copy()
    overlay[binary_mask == 1] = [255, 0, 0]  # Red highlight for changed structures
    blended = (img_b * 0.5 + overlay * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_a)
    axes[0].set_title("Time 1 (Before)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_b)
    axes[1].set_title("Time 2 (After)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title(f"Change Mask ({change_pct:.2f}% Changed)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(blended)
    axes[3].set_title("Detected Change Overlay", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        logger.info(f"Saved change detection visualization: {out}")
    return fig


def plot_tiling_overview(
    image: np.ndarray | Image.Image | str | Path,
    windows: list[Any],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Render raster sliding-window grid overlay demonstrating tile coverage."""
    img_arr = ensure_numpy_image(image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_arr)
    ax.set_title(f"Raster Sliding-Window Tiling ({len(windows)} tiles)", fontsize=14, fontweight="bold")
    ax.axis("off")

    for win in windows:
        rect = patches.Rectangle(
            (win.col_off, win.row_off),
            win.width,
            win.height,
            linewidth=1.5,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            win.col_off + 8,
            win.row_off + 20,
            f"T#{win.tile_id}",
            color="black",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="yellow", edgecolor="none", alpha=0.7, pad=1),
        )

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        logger.info(f"Saved tiling overview visualization: {out}")
    return fig
