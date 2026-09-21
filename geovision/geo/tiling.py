"""Sliding-window raster tiling and seamless patch reconstruction."""

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
from affine import Affine

from geovision.logger import get_logger

logger = get_logger("geovision.geo.tiling")


@dataclass
class TileWindow:
    """Represents a tiled sub-window with pixel offsets and geospatial sub-affine transform."""
    tile_id: int
    col_off: int
    row_off: int
    width: int
    height: int
    transform: Affine | None = None
    bounds: tuple[float, float, float, float] | None = None


class RasterTiler:
    """Slices large satellite rasters into manageable overlapping patches."""

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        pad_mode: str = "reflect",
    ):
        """Initialize tiler.

        Args:
            tile_size: Square tile dimension (width & height).
            overlap: Overlap in pixels between adjacent tiles.
            pad_mode: Numpy padding mode for edge handling.
        """
        if overlap >= tile_size:
            raise ValueError(f"Overlap ({overlap}) must be strictly less than tile_size ({tile_size})")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.pad_mode = pad_mode

    def generate_windows(
        self,
        image_width: int,
        image_height: int,
        base_transform: Affine | None = None,
    ) -> list[TileWindow]:
        """Compute list of all tile windows covering the full image dimensions."""
        windows: list[TileWindow] = []
        tile_id = 0

        # Calculate stride steps
        y_steps = list(range(0, image_height, self.stride))
        x_steps = list(range(0, image_width, self.stride))

        # Adjust last steps to firmly cover image boundaries without out-of-bounds
        if y_steps and y_steps[-1] + self.tile_size > image_height:
            y_steps[-1] = max(0, image_height - self.tile_size)
        if x_steps and x_steps[-1] + self.tile_size > image_width:
            x_steps[-1] = max(0, image_width - self.tile_size)

        # De-duplicate steps
        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))

        for y in y_steps:
            h = min(self.tile_size, image_height - y)
            for x in x_steps:
                w = min(self.tile_size, image_width - x)
                sub_t = None
                bounds = None
                if base_transform is not None:
                    # Shift affine transform origin to tile's top-left corner
                    sub_t = base_transform @ Affine.translation(x, y)
                    min_x, max_y = sub_t @ (0, 0)
                    max_x, min_y = sub_t @ (w, h)
                    bounds = (min_x, min_y, max_x, max_y)

                windows.append(
                    TileWindow(
                        tile_id=tile_id,
                        col_off=x,
                        row_off=y,
                        width=w,
                        height=h,
                        transform=sub_t,
                        bounds=bounds,
                    )
                )
                tile_id += 1

        return windows

    def tile_array(
        self,
        array: np.ndarray,
        base_transform: Affine | None = None,
    ) -> Generator[tuple[TileWindow, np.ndarray], None, None]:
        """Yield (TileWindow, tile_patch_array) for every window."""
        h, w = array.shape[:2]
        windows = self.generate_windows(w, h, base_transform=base_transform)

        for win in windows:
            patch = array[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
            # Pad patch if it is smaller than tile_size at the right/bottom borders
            if patch.shape[0] < self.tile_size or patch.shape[1] < self.tile_size:
                pad_h = self.tile_size - patch.shape[0]
                pad_w = self.tile_size - patch.shape[1]
                if array.ndim == 3:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode=self.pad_mode)
                else:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode=self.pad_mode)

            yield win, patch


class PatchReconstructor:
    """Seamlessly blends reconstructed overlapping tile predictions into full raster space."""

    def __init__(
        self,
        full_height: int,
        full_width: int,
        num_channels: int = 1,
        dtype: np.dtype = np.float32,
    ):
        self.full_height = full_height
        self.full_width = full_width
        self.num_channels = num_channels
        self.dtype = dtype

        # Allocate canvas & weight accumulation maps
        if num_channels == 1:
            self.canvas = np.zeros((full_height, full_width), dtype=dtype)
            self.weights = np.zeros((full_height, full_width), dtype=np.float32)
        else:
            self.canvas = np.zeros((full_height, full_width, num_channels), dtype=dtype)
            self.weights = np.zeros((full_height, full_width, 1), dtype=np.float32)

        self._hann_cache = {}

    def _get_hann_window(self, height: int, width: int) -> np.ndarray:
        """Create a 2D smooth Hann window for seamless edge blending."""
        key = (height, width)
        if key not in self._hann_cache:
            wy = np.hanning(height + 2)[1:-1]
            wx = np.hanning(width + 2)[1:-1]
            w2d = np.outer(wy, wx).astype(np.float32)
            # Clip minimum weight to prevent zero-division near edges
            w2d = np.maximum(w2d, 1e-4)
            self._hann_cache[key] = w2d
        return self._hann_cache[key]

    def add_patch(self, window: TileWindow, patch: np.ndarray, blend: bool = True) -> None:
        """Add a predicted patch into the canvas.

        Args:
            window: Corresponding TileWindow.
            patch: Predicted patch array of shape (H, W) or (H, W, C).
            blend: Whether to use smooth Hann window blending.
        """
        # Crop patch to actual window dimensions
        p = patch[: window.height, : window.width]

        if blend:
            w_2d = self._get_hann_window(window.height, window.width)
        else:
            w_2d = np.ones((window.height, window.width), dtype=np.float32)

        y1, y2 = window.row_off, window.row_off + window.height
        x1, x2 = window.col_off, window.col_off + window.width

        if self.num_channels == 1:
            p_clean = p if p.ndim == 2 else p.squeeze(-1)
            self.canvas[y1:y2, x1:x2] += p_clean * w_2d
            self.weights[y1:y2, x1:x2] += w_2d
        else:
            p_3d = p if p.ndim == 3 else p[:, :, np.newaxis]
            w_3d = w_2d[:, :, np.newaxis]
            self.canvas[y1:y2, x1:x2] += p_3d * w_3d
            self.weights[y1:y2, x1:x2] += w_3d

    def get_reconstruction(self) -> np.ndarray:
        """Normalize canvas by accumulated weights and return reconstructed array."""
        eps = 1e-6
        safe_weights = np.maximum(self.weights, eps)
        reconstructed = self.canvas / safe_weights
        return reconstructed.astype(self.dtype)
