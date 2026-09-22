"""Inference pipeline for bi-temporal satellite change detection with sliding-window tiling and visual reports."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from geovision.change.model import SiameseChangeDetector
from geovision.geo.raster import GeoRasterReader, write_geotiff
from geovision.geo.tiling import PatchReconstructor, RasterTiler
from geovision.logger import get_logger
from geovision.metrics.change import compute_change_statistics

logger = get_logger("geovision.change.infer")


def create_change_overlay(
    image_b: np.ndarray,
    change_mask: np.ndarray,
    color: tuple[int, int, int] = (255, 30, 30),  # Bright Red
    alpha: float = 0.5,
) -> np.ndarray:
    """Create RGB change overlay highlighting altered regions on the post-event image."""
    h, w = change_mask.shape[:2]
    img = image_b.copy().astype(np.float32)
    mask_bin = (change_mask > 0)

    color_arr = np.zeros((h, w, 3), dtype=np.float32)
    color_arr[mask_bin] = color

    img[mask_bin] = (1.0 - alpha) * img[mask_bin] + alpha * color_arr[mask_bin]
    return np.clip(img, 0, 255).astype(np.uint8)


def create_tri_panel_report(
    image_a: np.ndarray,
    image_b: np.ndarray,
    change_overlay: np.ndarray,
) -> np.ndarray:
    """Concatenate Pre-event, Post-event, and Change Overlay images side-by-side."""
    return np.concatenate([image_a, image_b, change_overlay], axis=1)


def _load_image_array(img_input: str | Path | np.ndarray) -> tuple[np.ndarray, Any | None]:
    """Helper to load image as (H, W, 3) uint8 numpy array with optional GeoTIFF metadata."""
    meta = None
    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        try:
            reader = GeoRasterReader(p)
            meta = reader.metadata
            arr = reader.read()
        except Exception:
            with Image.open(p) as im:
                arr = np.array(im.convert("RGB"))
    else:
        arr = img_input

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[2] > 3:
        arr = arr[:, :, :3]

    return arr, meta


def detect_changes(
    image_a: str | Path | np.ndarray,
    image_b: str | Path | np.ndarray,
    model: SiameseChangeDetector | None = None,
    checkpoint_path: str | Path | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    threshold: float = 0.5,
    device: str = "cpu",
    gsd_meters: float | None = 0.5,
    output_mask_path: str | Path | None = None,
    output_overlay_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Perform bi-temporal change detection with sliding-window tiling and comprehensive output reports.

    Args:
        image_a: Pre-event (T1) image path or NumPy array.
        image_b: Post-event (T2) image path or NumPy array.
        model: Optional pre-loaded SiameseChangeDetector.
        checkpoint_path: Optional path to checkpoint .pth file if model is not provided.
        tile_size: Window dimension for tiled processing.
        overlap: Overlap in pixels between adjacent tiles.
        threshold: Probability threshold for binary change decision.
        device: Computation device ('cpu' or 'cuda').
        gsd_meters: Ground sampling distance in meters per pixel.
        output_mask_path: Optional path to save binary change mask PNG/GeoTIFF.
        output_overlay_path: Optional path to save change overlay on Image B.
        output_report_path: Optional path to save 3-panel comparison image (T1, T2, Change Overlay).

    Returns:
        Dictionary containing binary change mask, change probabilities, statistics, and saved file paths.
    """
    if model is None:
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            model = SiameseChangeDetector.load(checkpoint_path, device=device)
        else:
            logger.info("Initializing baseline SiameseChangeDetector.")
            model = SiameseChangeDetector(device=device)

    model.eval()
    dev = model.target_device

    arr_a, meta_a = _load_image_array(image_a)
    arr_b, meta_b = _load_image_array(image_b)

    if arr_a.shape[:2] != arr_b.shape[:2]:
        raise ValueError(f"Image dimension mismatch: Image A {arr_a.shape[:2]} vs Image B {arr_b.shape[:2]}")

    h, w = arr_a.shape[:2]

    # Run inference: direct or sliding window
    if h <= tile_size and w <= tile_size:
        t_a = torch.from_numpy(arr_a.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t_b = torch.from_numpy(arr_b.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        t_a, t_b = t_a.to(dev), t_b.to(dev)

        with torch.no_grad():
            logits = model(t_a, t_b)
            if logits.shape[2:] != (h, w):
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
            pred_mask = (probs >= threshold).astype(np.uint8)
    else:
        tiler = RasterTiler(tile_size=tile_size, overlap=overlap)
        reconstructor = PatchReconstructor(
            full_height=h,
            full_width=w,
            num_channels=1,
            dtype=np.float32,
        )

        for win, patch_a in tiler.tile_array(arr_a):
            patch_b = arr_b[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
            if patch_b.shape[0] < tile_size or patch_b.shape[1] < tile_size:
                pad_h = tile_size - patch_b.shape[0]
                pad_w = tile_size - patch_b.shape[1]
                patch_b = np.pad(patch_b, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            t_a = torch.from_numpy(patch_a.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            t_b = torch.from_numpy(patch_b.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            t_a, t_b = t_a.to(dev), t_b.to(dev)

            with torch.no_grad():
                logits = model(t_a, t_b)
                if logits.shape[2:] != (tile_size, tile_size):
                    logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                tile_probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()

            reconstructor.add_patch(window=win, patch=tile_probs, blend=True)

        probs = reconstructor.get_reconstruction()
        pred_mask = (probs >= threshold).astype(np.uint8)

    # Statistical shift computation
    stats = compute_change_statistics(
        change_mask=pred_mask,
        gsd_meters=gsd_meters,
    )

    saved_files = {}

    # Save binary mask
    if output_mask_path is not None:
        mask_out = Path(output_mask_path)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        if mask_out.suffix.lower() in (".tif", ".tiff") and (meta_a or meta_b):
            target_meta = meta_b or meta_a
            try:
                write_geotiff(
                    path=mask_out,
                    data=pred_mask * 255,
                    metadata=target_meta,
                )
            except Exception as e:
                logger.warning(f"GeoTIFF write failed ({e}), saving PNG.")
                Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_out)
        else:
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(mask_out)
        saved_files["mask_path"] = str(mask_out)

    overlay = create_change_overlay(arr_b, pred_mask)

    # Save overlay image
    if output_overlay_path is not None:
        over_out = Path(output_overlay_path)
        over_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(over_out)
        saved_files["overlay_path"] = str(over_out)

    # Save 3-panel report image
    if output_report_path is not None:
        rep_out = Path(output_report_path)
        rep_out.parent.mkdir(parents=True, exist_ok=True)
        tri_panel = create_tri_panel_report(arr_a, arr_b, overlay)
        Image.fromarray(tri_panel).save(rep_out)
        saved_files["report_path"] = str(rep_out)

    return {
        "change_mask": pred_mask,
        "change_probs": probs,
        "statistics": stats,
        "saved_files": saved_files,
        "image_shape": (h, w),
    }
