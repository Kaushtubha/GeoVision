"""Inference pipeline for land-cover semantic segmentation with sliding-window tiling and geospatial mask export."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from geovision.constants import LANDCOVER_AI_CLASSES, LANDCOVER_AI_COLORS
from geovision.geo.raster import GeoRasterReader, write_geotiff
from geovision.geo.tiling import PatchReconstructor, RasterTiler
from geovision.logger import get_logger
from geovision.metrics.segmentation import compute_landcover_distribution
from geovision.segmentation.model import LandCoverSegmenter

logger = get_logger("geovision.segmentation.infer")


def colorize_mask(mask: np.ndarray, color_map: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Map integer class indices to 3-channel RGB colorized image.

    Args:
        mask: 2D integer numpy array (H, W).
        color_map: Mapping of class index to RGB tuple.

    Returns:
        RGB numpy array of shape (H, W, 3) with uint8 dtype.
    """
    cmap = color_map or LANDCOVER_AI_COLORS
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in cmap.items():
        match_pixels = (mask == class_id)
        rgb[match_pixels] = color

    return rgb


def overlay_mask_on_image(
    image: np.ndarray,
    mask_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend RGB color mask on top of source aerial/satellite image."""
    img_float = image.astype(np.float32)
    mask_float = mask_rgb.astype(np.float32)
    blended = (1.0 - alpha) * img_float + alpha * mask_float
    return np.clip(blended, 0, 255).astype(np.uint8)


def segment_image(
    image_input: str | Path | np.ndarray,
    model: LandCoverSegmenter | None = None,
    checkpoint_path: str | Path | None = None,
    tile_size: int = 512,
    overlap: int = 64,
    device: str = "cpu",
    gsd_meters: float | None = 0.5,
    output_mask_path: str | Path | None = None,
    output_overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    """Segment a satellite/aerial image with sliding-window tiling and geospatial preservation.

    Args:
        image_input: File path (GeoTIFF/PNG/JPG) or NumPy array (H, W, C).
        model: Optional pre-loaded LandCoverSegmenter instance.
        checkpoint_path: Optional path to checkpoint .pth file if model is not passed.
        tile_size: Window size for sliding-window inference.
        overlap: Overlap in pixels between adjacent sliding windows.
        device: Inference device ('cpu' or 'cuda').
        gsd_meters: Ground sampling distance in meters per pixel.
        output_mask_path: Optional destination to save predicted mask PNG/GeoTIFF.
        output_overlay_path: Optional destination to save visual overlay image.

    Returns:
        Dictionary containing predicted mask, class area distribution, and saved paths.
    """
    if model is None:
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            model = LandCoverSegmenter.load(checkpoint_path, device=device)
        else:
            logger.info("Initializing fresh LandCoverSegmenter for inference.")
            model = LandCoverSegmenter(device=device)

    model.eval()
    dev = model.target_device

    geotiff_meta = None
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")

        # Attempt GeoTIFF reading for geospatial metadata
        try:
            reader = GeoRasterReader(img_path)
            geotiff_meta = reader.metadata
            img_arr = reader.read()
        except Exception:
            with Image.open(img_path) as im:
                img_arr = np.array(im.convert("RGB"))
    else:
        img_arr = image_input

    # Normalize dimensions (H, W, 3)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.ndim == 3 and img_arr.shape[0] in (1, 3, 4) and img_arr.shape[2] not in (1, 3, 4):
        img_arr = np.transpose(img_arr, (1, 2, 0))

    if img_arr.shape[2] > 3:
        img_arr = img_arr[:, :, :3]

    h, w, c = img_arr.shape
    num_classes = model.num_classes

    # Choose direct inference vs sliding-window tiling
    if h <= tile_size and w <= tile_size:
        # Direct padded inference
        tensor_img = torch.from_numpy(img_arr.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        # Resize or pad if needed
        tensor_img = tensor_img.to(dev)
        with torch.no_grad():
            logits = model(tensor_img)
            if logits.shape[2:] != (h, w):
                logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
    else:
        # Sliding window tiling with Hann window blending
        tiler = RasterTiler(tile_size=tile_size, overlap=overlap)
        reconstructor = PatchReconstructor(
            full_height=h,
            full_width=w,
            num_channels=num_classes,
            dtype=np.float32,
        )

        for win, tile_patch in tiler.tile_array(img_arr):
            tensor_tile = torch.from_numpy(tile_patch.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            tensor_tile = tensor_tile.to(dev)

            with torch.no_grad():
                tile_logits = model(tensor_tile)
                if tile_logits.shape[2:] != (tile_size, tile_size):
                    tile_logits = F.interpolate(tile_logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                tile_probs = F.softmax(tile_logits, dim=1)[0].permute(1, 2, 0).cpu().numpy()

            reconstructor.add_patch(window=win, patch=tile_probs, blend=True)

        blended_probs = reconstructor.get_reconstruction()
        pred_mask = np.argmax(blended_probs, axis=-1).astype(np.uint8)

    # Compute land-cover distribution statistics
    distribution = compute_landcover_distribution(
        mask=pred_mask,
        num_classes=num_classes,
        class_names=LANDCOVER_AI_CLASSES,
        gsd_meters=gsd_meters,
    )

    saved_files = {}

    # Save mask if path specified
    if output_mask_path is not None:
        mask_out = Path(output_mask_path)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        # If output requested is .tif and geotiff metadata is available, write GeoTIFF
        if mask_out.suffix.lower() in (".tif", ".tiff") and geotiff_meta is not None:
            try:
                write_geotiff(
                    path=mask_out,
                    data=pred_mask,
                    metadata=geotiff_meta,
                )
            except Exception as e:
                logger.warning(f"Could not write GeoTIFF ({e}), saving PNG mask.")
                Image.fromarray(pred_mask).save(mask_out)
        else:
            Image.fromarray(pred_mask).save(mask_out)
        saved_files["mask_path"] = str(mask_out)

    # Save visual overlay if requested
    if output_overlay_path is not None:
        overlay_out = Path(output_overlay_path)
        overlay_out.parent.mkdir(parents=True, exist_ok=True)
        color_mask = colorize_mask(pred_mask)
        blended = overlay_mask_on_image(img_arr, color_mask)
        Image.fromarray(blended).save(overlay_out)
        saved_files["overlay_path"] = str(overlay_out)

    return {
        "mask": pred_mask,
        "distribution": distribution,
        "saved_files": saved_files,
        "image_shape": (h, w),
    }
