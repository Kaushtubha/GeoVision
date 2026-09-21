"""Demo script verifying GeoTIFF I/O, CRS coordinate reprojections, and sliding-window tiling."""

import sys
from pathlib import Path

import numpy as np
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.constants import EXPERIMENTS_DIR
from geovision.geo.crs import CoordinateTransformer, create_geojson_collection
from geovision.geo.raster import GeoRasterReader, write_geotiff
from geovision.geo.tiling import PatchReconstructor, RasterTiler
from geovision.logger import get_logger
from geovision.visualization import (
    plot_change_detection_pair,
    plot_detection_boxes,
    plot_segmentation_overlay,
    plot_tiling_overview,
)

logger = get_logger("geovision.scripts.demo_geo")


def main():
    vis_dir = EXPERIMENTS_DIR / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 65)
    logger.info("GeoVision: Phase 2 Geospatial & Data Pipeline Demonstration")
    logger.info("=" * 65)

    # 1. Create a Synthetic GeoTIFF centered at Warsaw, Poland (EPSG:2180) for LandCover.ai validation
    # Real coordinate benchmark: Easting=636000, Northing=486000, GSD=0.5m/pixel
    h, w = 1024, 1024
    affine_warsaw = Affine(0.5, 0.0, 636000.0, 0.0, -0.5, 486512.0)
    crs_str = "EPSG:2180"

    # Create synthetic optical satellite raster (RGB)
    rng = np.random.RandomState(42)
    sample_raster = rng.randint(40, 210, size=(h, w, 3), dtype=np.uint8)

    geotiff_path = vis_dir / "sample_geotiff_warsaw.tif"
    write_geotiff(geotiff_path, sample_raster, affine_warsaw, crs=crs_str)
    logger.info(f"1. Created Georeferenced Satellite GeoTIFF: {geotiff_path}")

    # 2. Read back with unified GeoRasterReader
    reader = GeoRasterReader(geotiff_path)
    meta = reader.meta
    logger.info(f"2. Read Metadata (Engine: {reader.engine}):")
    logger.info(f"   • CRS: {meta.crs}")
    logger.info(f"   • Dimensions: {meta.width}x{meta.height} (Channels: {meta.count})")
    logger.info(f"   • Bounds: {meta.bounds}")
    logger.info(f"   • GSD (Resolution): {meta.gsd_x:.2f}m x {meta.gsd_y:.2f}m")

    # 3. Test Bidirectional Coordinate Transforms
    transformer = CoordinateTransformer(meta.crs)
    center_px_x, center_px_y = 512.0, 512.0
    lon, lat = transformer.pixel_to_latlon(center_px_x, center_px_y, meta.transform)
    recon_px_x, recon_px_y = transformer.latlon_to_pixel(lon, lat, meta.transform)

    logger.info("3. Bidirectional Coordinate Transformation Test:")
    logger.info(f"   • Input Pixel: ({center_px_x}, {center_px_y})")
    logger.info(f"   • Reprojected WGS84 Geodetic: (Lon={lon:.6f}°, Lat={lat:.6f}°)")
    logger.info(f"   • Inverted Pixel: ({recon_px_x:.4f}, {recon_px_y:.4f})")
    logger.info(f"   • Roundtrip Error: {abs(center_px_x - recon_px_x):.6e} px")

    # 4. GeoJSON Polygon Generation from Object Detection Bounding Box
    feature = transformer.bbox_pixel_to_geojson(
        xmin=100.0, ymin=100.0, xmax=300.0, ymax=300.0,
        transform=meta.transform,
        properties={"class": "building", "confidence": 0.94}
    )
    geojson = create_geojson_collection([feature])
    logger.info(f"4. Generated GeoJSON Polygon Feature: {geojson['features'][0]['geometry']['type']} with {len(geojson['features'][0]['geometry']['coordinates'][0])} vertices")

    # 5. Sliding-Window Tiling & Seamless Hann-Window Reconstruction
    tiler = RasterTiler(tile_size=512, overlap=64)
    windows = tiler.generate_windows(w, h, base_transform=meta.transform)
    logger.info(f"5. Sliced {w}x{h} Scene into {len(windows)} overlapping patches (Tile=512, Overlap=64px)")

    # Simulate patch predictions and reconstruct
    reconstructor = PatchReconstructor(full_height=h, full_width=w, num_channels=1, dtype=np.float32)
    for win, patch in tiler.tile_array(sample_raster, base_transform=meta.transform):
        # Dummy probability response
        pred_patch = (patch.mean(axis=-1) / 255.0).astype(np.float32)
        reconstructor.add_patch(win, pred_patch, blend=True)

    reconstructed_map = reconstructor.get_reconstruction()
    logger.info(f"   • Reconstructed continuous spatial map: shape={reconstructed_map.shape}, range=[{reconstructed_map.min():.2f}, {reconstructed_map.max():.2f}]")

    # 6. Generate and Save Visualizations
    plot_tiling_overview(sample_raster, windows, output_path=vis_dir / "01_tiling_overview.png")

    # Detection overlay
    boxes = [[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.15, 0.25]]
    labels = [0, 1]  # Airplane, Ship
    scores = [0.95, 0.88]
    plot_detection_boxes(sample_raster[:512, :512], boxes, labels, scores=scores, output_path=vis_dir / "02_detection_overlay.png")

    # Segmentation overlay
    seg_mask = np.zeros((512, 512), dtype=np.uint8)
    seg_mask[50:180, 50:200] = 1  # Building
    seg_mask[220:400, 200:450] = 2  # Woodland
    seg_mask[420:490, 50:480] = 3  # Water
    seg_mask[180:210, :] = 4  # Road
    plot_segmentation_overlay(sample_raster[:512, :512], seg_mask, alpha=0.45, output_path=vis_dir / "03_segmentation_overlay.png")

    # Change Detection overlay
    t1 = sample_raster[:256, :256].copy()
    t2 = t1.copy()
    change_mask = np.zeros((256, 256), dtype=np.uint8)
    t2[50:120, 50:130] = np.random.randint(200, 255, size=(70, 80, 3), dtype=np.uint8)
    change_mask[50:120, 50:130] = 1
    plot_change_detection_pair(t1, t2, change_mask, output_path=vis_dir / "04_change_overlay.png")

    logger.info("All geospatial pipeline demonstrations and visual overlays saved to experiments/visualizations/!")


if __name__ == "__main__":
    main()
