"""Geospatial coordinate transformations and CRS reprojection."""

from typing import Any

from affine import Affine
from pyproj import CRS, Transformer

from geovision.logger import get_logger

logger = get_logger("geovision.geo.crs")


class CoordinateTransformer:
    """Handles high-precision bidirectional coordinate mapping between:
    - Pixel space (col, row) -> (x, y) in image
    - Projected world coordinates (Easting, Northing)
    - Geodetic WGS84 coordinates (Longitude, Latitude) [EPSG:4326]
    """

    def __init__(self, src_crs: str | int | CRS = "EPSG:4326"):
        """Initialize transformer with a source CRS.

        Args:
            src_crs: Source Coordinate Reference System (e.g. 'EPSG:4326', 'EPSG:3857', 'EPSG:2180').
        """
        if isinstance(src_crs, int):
            self.src_crs = CRS.from_epsg(src_crs)
        elif isinstance(src_crs, str):
            self.src_crs = CRS.from_user_input(src_crs)
        else:
            self.src_crs = src_crs

        self.wgs84_crs = CRS.from_epsg(4326)

        # Initialize pyproj forward and reverse transformers (always_xy=True ensures (x, y) = (lon, lat))
        if self.src_crs.to_epsg() != 4326:
            self._to_wgs84 = Transformer.from_crs(self.src_crs, self.wgs84_crs, always_xy=True)
            self._from_wgs84 = Transformer.from_crs(self.wgs84_crs, self.src_crs, always_xy=True)
        else:
            self._to_wgs84 = None
            self._from_wgs84 = None

    def pixel_to_projected(
        self,
        pixel_x: float,
        pixel_y: float,
        transform: Affine | tuple[float, ...],
    ) -> tuple[float, float]:
        """Convert pixel coordinate (col, row) to projected world coordinate (X, Y).

        Args:
            pixel_x: Column index (x) in pixels.
            pixel_y: Row index (y) in pixels.
            transform: 6-element affine transform (a, b, c, d, e, f) or Affine instance.

        Returns:
            Tuple of (world_x, world_y) in source CRS units (e.g. meters or degrees).
        """
        if not isinstance(transform, Affine):
            transform = Affine(*transform[:6])
        world_x, world_y = transform @ (pixel_x, pixel_y)
        return float(world_x), float(world_y)

    def projected_to_pixel(
        self,
        world_x: float,
        world_y: float,
        transform: Affine | tuple[float, ...],
    ) -> tuple[float, float]:
        """Convert projected world coordinate (X, Y) to pixel coordinate (col, row).

        Args:
            world_x: World X (easting / longitude).
            world_y: World Y (northing / latitude).
            transform: 6-element affine transform.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        if not isinstance(transform, Affine):
            transform = Affine(*transform[:6])
        inv_transform = ~transform
        pixel_x, pixel_y = inv_transform @ (world_x, world_y)
        return float(pixel_x), float(pixel_y)

    def projected_to_latlon(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Convert projected coordinates (X, Y) to WGS84 (Longitude, Latitude)."""
        if self._to_wgs84 is None:
            return float(world_x), float(world_y)
        lon, lat = self._to_wgs84.transform(world_x, world_y)
        return float(lon), float(lat)

    def latlon_to_projected(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert WGS84 (Longitude, Latitude) to projected coordinates (X, Y)."""
        if self._from_wgs84 is None:
            return float(lon), float(lat)
        world_x, world_y = self._from_wgs84.transform(lon, lat)
        return float(world_x), float(world_y)

    def pixel_to_latlon(
        self,
        pixel_x: float,
        pixel_y: float,
        transform: Affine | tuple[float, ...],
    ) -> tuple[float, float]:
        """Convert pixel (x, y) directly to WGS84 (Longitude, Latitude)."""
        wx, wy = self.pixel_to_projected(pixel_x, pixel_y, transform)
        return self.projected_to_latlon(wx, wy)

    def latlon_to_pixel(
        self,
        lon: float,
        lat: float,
        transform: Affine | tuple[float, ...],
    ) -> tuple[float, float]:
        """Convert WGS84 (Longitude, Latitude) directly to pixel (x, y)."""
        wx, wy = self.latlon_to_projected(lon, lat)
        return self.projected_to_pixel(wx, wy, transform)

    def bbox_pixel_to_geojson(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        transform: Affine | tuple[float, ...],
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert a pixel bounding box [xmin, ymin, xmax, ymax] into a GeoJSON Polygon Feature.

        Args:
            xmin, ymin: Top-left corner in pixel space.
            xmax, ymax: Bottom-right corner in pixel space.
            transform: Affine transform.
            properties: Optional metadata dictionary to attach to GeoJSON feature.

        Returns:
            GeoJSON Feature dictionary.
        """
        # 4 corners in pixel space (counter-clockwise polygon)
        corners_pixel = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),  # Close loop
        ]

        coords_latlon = [
            list(self.pixel_to_latlon(px, py, transform))
            for px, py in corners_pixel
        ]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords_latlon],
            },
            "properties": properties or {},
        }
        return feature


def create_geojson_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a list of GeoJSON features into a standard FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }
