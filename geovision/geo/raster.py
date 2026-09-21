"""Unified GeoTIFF and satellite raster I/O with dual-engine fallback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from PIL import Image

from geovision.logger import get_logger

logger = get_logger("geovision.geo.raster")

try:
    import rasterio
    from rasterio.windows import Window
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


@dataclass
class GeoRasterMetadata:
    """Metadata schema representing georeferenced satellite rasters."""
    crs: str
    transform: Affine
    bounds: tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    width: int
    height: int
    count: int  # Number of channels/bands
    dtype: str
    gsd_x: float  # Ground Sample Distance (resolution in X)
    gsd_y: float  # Ground Sample Distance (resolution in Y)
    nodata: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "crs": self.crs,
            "transform": [self.transform.a, self.transform.b, self.transform.c,
                          self.transform.d, self.transform.e, self.transform.f],
            "bounds": {
                "min_x": self.bounds[0],
                "min_y": self.bounds[1],
                "max_x": self.bounds[2],
                "max_y": self.bounds[3],
            },
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtype": self.dtype,
            "gsd": (self.gsd_x, self.gsd_y),
            "nodata": self.nodata,
        }


class GeoRasterReader:
    """Unified satellite raster reader supporting both Rasterio and pure-Python Tifffile fallbacks."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Raster file not found: {self.filepath}")

        self.engine = "rasterio" if _HAS_RASTERIO else "tifffile"
        self.meta = self._read_metadata()

    def _read_metadata(self) -> GeoRasterMetadata:
        """Extract spatial reference, dimensions, and affine transform from file."""
        if self.engine == "rasterio":
            try:
                with rasterio.open(self.filepath) as src:
                    crs_str = src.crs.to_string() if src.crs else "EPSG:4326"
                    t = src.transform
                    bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    return GeoRasterMetadata(
                        crs=crs_str,
                        transform=t,
                        bounds=bounds,
                        width=src.width,
                        height=src.height,
                        count=src.count,
                        dtype=str(src.dtypes[0]),
                        gsd_x=abs(t.a),
                        gsd_y=abs(t.e),
                        nodata=src.nodata,
                    )
            except Exception as e:
                logger.warning(f"Rasterio failed on {self.filepath} ({e}). Falling back to tifffile engine.")
                self.engine = "tifffile"

        # Tifffile / PIL pure-Python Fallback
        if _HAS_TIFFFILE and self.filepath.suffix.lower() in [".tif", ".tiff"]:
            with tifffile.TiffFile(self.filepath) as tif:
                page = tif.pages[0]
                shape = page.shape
                # Derive height, width, count
                if len(shape) == 2:
                    h, w, count = shape[0], shape[1], 1
                elif len(shape) == 3:
                    if shape[0] in (1, 3, 4) and shape[0] < shape[1]:
                        count, h, w = shape[0], shape[1], shape[2]
                    else:
                        h, w, count = shape[0], shape[1], shape[2]
                else:
                    h, w, count = shape[-2], shape[-1], 1

                # Extract GeoTIFF ModelPixelScale and ModelTiepoint tags if present
                tags = page.tags
                scale = tags.get(33550)  # ModelPixelScaleTag
                tiepoint = tags.get(33922)  # ModelTiepointTag
                geo_key_dir = tags.get(34735)  # GeoKeyDirectoryTag

                crs_str = "EPSG:4326"
                if geo_key_dir and len(geo_key_dir.value) > 4:
                    # Parse ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
                    keys = geo_key_dir.value
                    for i in range(4, len(keys) - 3, 4):
                        key_id, _, _, val = keys[i:i+4]
                        if key_id in (3072, 2048) and val != 0:
                            crs_str = f"EPSG:{val}"
                            break

                if scale and tiepoint and len(scale.value) >= 2 and len(tiepoint.value) >= 6:
                    sx, sy = scale.value[0], scale.value[1]
                    gx, gy = tiepoint.value[3], tiepoint.value[4]
                    t = Affine(sx, 0.0, gx, 0.0, -sy, gy)
                else:
                    # Default affine (1px = 1 unit)
                    t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(h))

                min_x = t.c
                max_y = t.f
                max_x = min_x + t.a * w
                min_y = max_y + t.e * h

                return GeoRasterMetadata(
                    crs=crs_str,
                    transform=t,
                    bounds=(min_x, min_y, max_x, max_y),
                    width=w,
                    height=h,
                    count=count,
                    dtype=str(page.dtype),
                    gsd_x=abs(t.a),
                    gsd_y=abs(t.e),
                    nodata=None,
                )

        # General image format (PNG/JPG) with default georeferencing
        with Image.open(self.filepath) as im:
            w, h = im.size
            count = len(im.getbands())
            t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(h))
            return GeoRasterMetadata(
                crs="EPSG:4326",
                transform=t,
                bounds=(0.0, 0.0, float(w), float(h)),
                width=w,
                height=h,
                count=count,
                dtype="uint8",
                gsd_x=1.0,
                gsd_y=1.0,
                nodata=None,
            )

    def read_all(self) -> np.ndarray:
        """Read full raster into a numpy array (H, W, C) for standard imagery."""
        if self.engine == "rasterio":
            with rasterio.open(self.filepath) as src:
                arr = src.read()  # (C, H, W)
                if arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
                return arr

        if _HAS_TIFFFILE and self.filepath.suffix.lower() in [".tif", ".tiff"]:
            arr = tifffile.imread(self.filepath)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))
            return arr

        with Image.open(self.filepath) as im:
            return np.array(im)

    def read_window(self, col_off: int, row_off: int, width: int, height: int) -> np.ndarray:
        """Read a sub-window of pixels from the raster."""
        if self.engine == "rasterio":
            with rasterio.open(self.filepath) as src:
                w = Window(col_off, row_off, width, height)
                arr = src.read(window=w)
                if arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                return arr

        # Fallback slicing on full array
        full_arr = self.read_all()
        return full_arr[row_off : row_off + height, col_off : col_off + width]


def write_geotiff(
    output_path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
    nodata: float | None = None,
) -> Path:
    """Write an image array to a valid GeoTIFF with coordinate tags."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        h, w = data.shape
        count = 1
        arr_to_write = data[np.newaxis, ...]
    elif data.ndim == 3:
        h, w, count = data.shape[0], data.shape[1], data.shape[2]
        arr_to_write = np.transpose(data, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported array dimension: {data.ndim}")

    if _HAS_RASTERIO:
        with rasterio.open(
            out,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=count,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr_to_write)
        return out

    if _HAS_TIFFFILE:
        # Save with ModelPixelScale and ModelTiepoint tags for pure Python GeoTIFF compliance
        epsg_code = 4326
        if "epsg:" in crs.lower():
            try:
                epsg_code = int(crs.lower().split("epsg:")[1])
            except ValueError:
                pass

        extratags = [
            (33550, "d", 3, (abs(transform.a), abs(transform.e), 0.0), False),  # ModelPixelScale
            (33922, "d", 6, (0.0, 0.0, 0.0, transform.c, transform.f, 0.0), False),  # ModelTiepoint
            (34735, "H", 8, (1, 1, 0, 1, 3072, 0, 1, epsg_code), False),  # GeoKeyDirectory
        ]
        tifffile.imwrite(out, data, extratags=extratags)
        return out

    # Basic Pillow save if neither is present
    Image.fromarray(data).save(out)
    return out
