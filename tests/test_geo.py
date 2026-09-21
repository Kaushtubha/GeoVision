"""Unit tests for geospatial transformations, GeoTIFF I/O, tiling, and dataset pipelines."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from geovision.data.datasets import (
    EuroSATDataset,
    LandCoverAIDataset,
    LEVIRCDDataset,
    NWPUVHR10Dataset,
)
from geovision.data.downloaders import download_dataset
from geovision.geo.crs import CoordinateTransformer, create_geojson_collection
from geovision.geo.raster import GeoRasterReader, write_geotiff
from geovision.geo.tiling import PatchReconstructor, RasterTiler


def test_coordinate_transformer_roundtrip_wgs84():
    """Verify bidirectional pixel <-> WGS84 coordinates on EPSG:4326."""
    # San Francisco: Top-Left Lon=-122.5, Lat=37.8, GSD=0.0001 deg/pixel
    transform = Affine(0.0001, 0.0, -122.5, 0.0, -0.0001, 37.8)
    transformer = CoordinateTransformer("EPSG:4326")

    px, py = 250.0, 150.0
    lon, lat = transformer.pixel_to_latlon(px, py, transform)

    # Check exact geodetic coordinates
    assert pytest.approx(lon, 1e-6) == -122.5 + 250.0 * 0.0001
    assert pytest.approx(lat, 1e-6) == 37.8 - 150.0 * 0.0001

    # Roundtrip inverse
    recon_x, recon_y = transformer.latlon_to_pixel(lon, lat, transform)
    assert pytest.approx(recon_x, 1e-5) == px
    assert pytest.approx(recon_y, 1e-5) == py


def test_coordinate_transformer_projected_epsg2180():
    """Verify bidirectional pixel <-> projected EPSG:2180 (Poland / LandCover.ai)."""
    # Warsaw coordinates: Easting=636000, Northing=486000, GSD=0.5m/px
    transform = Affine(0.5, 0.0, 636000.0, 0.0, -0.5, 486000.0)
    transformer = CoordinateTransformer("EPSG:2180")

    px, py = 500.0, 300.0
    lon, lat = transformer.pixel_to_latlon(px, py, transform)

    # Longitude should be ~21.0 deg E, Latitude should be ~52.2 deg N (Warsaw)
    assert 20.0 < lon < 22.0
    assert 51.5 < lat < 53.0

    # Invert back to pixel
    recon_x, recon_y = transformer.latlon_to_pixel(lon, lat, transform)
    assert pytest.approx(recon_x, 1e-4) == px
    assert pytest.approx(recon_y, 1e-4) == py


def test_bbox_to_geojson():
    """Verify pixel bounding box conversion to valid GeoJSON polygon."""
    transform = Affine(0.0001, 0.0, -122.5, 0.0, -0.0001, 37.8)
    transformer = CoordinateTransformer("EPSG:4326")

    feature = transformer.bbox_pixel_to_geojson(
        xmin=50.0, ymin=50.0, xmax=150.0, ymax=150.0,
        transform=transform,
        properties={"label": "airplane", "confidence": 0.92}
    )

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    coords = feature["geometry"]["coordinates"][0]
    assert len(coords) == 5  # Closed loop (5 points)
    assert coords[0] == coords[-1]  # Closed polygon
    assert feature["properties"]["label"] == "airplane"

    collection = create_geojson_collection([feature])
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 1


def test_geotiff_io_roundtrip():
    """Verify GeoTIFF creation and metadata reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        geotiff_file = Path(tmpdir) / "test_satellite.tif"
        data = np.random.randint(10, 240, size=(256, 256, 3), dtype=np.uint8)
        transform = Affine(0.5, 0.0, 500000.0, 0.0, -0.5, 4000000.0)

        write_geotiff(geotiff_file, data, transform, crs="EPSG:3857")
        assert geotiff_file.exists()

        reader = GeoRasterReader(geotiff_file)
        meta = reader.meta

        assert meta.width == 256
        assert meta.height == 256
        assert meta.count == 3
        assert pytest.approx(meta.gsd_x, 1e-4) == 0.5
        assert pytest.approx(meta.gsd_y, 1e-4) == 0.5

        loaded_data = reader.read_all()
        assert loaded_data.shape == (256, 256, 3)
        np.testing.assert_array_equal(data, loaded_data)


def test_raster_tiler_and_reconstruction():
    """Verify sliding-window tiler and seamless reconstruction across overlapping patches."""
    h, w = 1000, 1000
    original_map = np.random.uniform(0.1, 0.9, size=(h, w)).astype(np.float32)

    tiler = RasterTiler(tile_size=512, overlap=64)
    windows = tiler.generate_windows(w, h)
    assert len(windows) >= 4

    reconstructor = PatchReconstructor(full_height=h, full_width=w, num_channels=1)

    for win in windows:
        patch = original_map[win.row_off : win.row_off + win.height, win.col_off : win.col_off + win.width]
        reconstructor.add_patch(win, patch, blend=True)

    reconstructed = reconstructor.get_reconstruction()
    assert reconstructed.shape == (h, w)
    # Reconstructed continuous float map should match original map within floating-point tolerance
    np.testing.assert_allclose(original_map, reconstructed, atol=1e-4)


def test_dataset_loading_pipelines():
    """Verify PyTorch dataset loaders for all 4 satellite tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        # 1. EuroSAT
        eurosat_path = download_dataset("eurosat", target_dir=dest, smoke=True)
        ds_eurosat = EuroSATDataset(eurosat_path)
        assert len(ds_eurosat) > 0
        sample_e = ds_eurosat[0]
        assert "image" in sample_e and "label" in sample_e

        # 2. NWPU VHR-10
        nwpu_path = download_dataset("nwpu_vhr10", target_dir=dest, smoke=True)
        ds_nwpu = NWPUVHR10Dataset(nwpu_path)
        assert len(ds_nwpu) > 0
        sample_n = ds_nwpu[0]
        assert "image" in sample_n and "bboxes" in sample_n

        # 3. LandCover.ai
        lc_path = download_dataset("landcover_ai", target_dir=dest, smoke=True)
        ds_lc = LandCoverAIDataset(lc_path)
        assert len(ds_lc) > 0
        sample_l = ds_lc[0]
        assert "image" in sample_l and "mask" in sample_l

        # 4. LEVIR-CD
        levir_path = download_dataset("levir_cd", target_dir=dest, smoke=True)
        ds_levir = LEVIRCDDataset(levir_path)
        assert len(ds_levir) > 0
        sample_v = ds_levir[0]
        assert "image_a" in sample_v and "image_b" in sample_v and "mask" in sample_v
