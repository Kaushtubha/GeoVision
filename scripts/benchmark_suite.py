"""End-to-End Latency & Accuracy Benchmark Suite for GeoVision.

Benchmarks:
1. Geospatial & CRS Tiling / Reconstruction Throughput
2. Object Detection Inference Latency & Batch Throughput
3. Semantic Land-Cover Segmentation Sliding Window Latency
4. Bi-Temporal Change Detection Sliding Window Latency
5. OpenCLIP Scene Vector Embedding & Qdrant Search Latency
6. Grounded VLM Earth Assistant Citation Verification Throughput

Usage:
    python scripts/benchmark_suite.py --iterations 5 --output benchmark_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from geovision.change import SiameseChangeDetector, detect_changes
from geovision.detection import YOLODetector
from geovision.geo.tiling import PatchReconstructor, RasterTiler
from geovision.logger import get_logger
from geovision.retrieval import SceneVectorIndex
from geovision.segmentation import LandCoverSegmenter, segment_image
from geovision.vlm.assistant import GroundedEarthAssistant
from geovision.vlm.evidence import EvidenceSynthesizer

logger = get_logger("geovision.benchmark")


def benchmark_tiling(iterations: int = 5) -> dict[str, float]:
    """Benchmark Geo raster tiling and reconstruction."""
    large_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    tiler = RasterTiler(tile_size=512, overlap=64)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        reconstructor = PatchReconstructor(full_height=2048, full_width=2048)
        for win, _ in tiler.tile_array(large_img):
            mock_mask = np.ones((512, 512), dtype=np.float32)
            reconstructor.add_patch(win, mock_mask)
        _ = reconstructor.get_reconstruction()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "throughput_megapixels_per_sec": round(
            (2048 * 2048 / 1e6) / (float(np.mean(latencies)) / 1000.0), 2
        ),
    }


def benchmark_detection(iterations: int = 5) -> dict[str, float]:
    """Benchmark YOLO detection latency."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    detector = YOLODetector(model_path="yolov8n.pt", device="cpu")

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = detector.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def benchmark_segmentation(iterations: int = 5) -> dict[str, float]:
    """Benchmark Land-Cover segmentation latency."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    segmenter = LandCoverSegmenter(encoder_name="resnet18", num_classes=5, device="cpu")

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = segment_image(img, model=segmenter, tile_size=512, device="cpu")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def benchmark_change_detection(iterations: int = 5) -> dict[str, float]:
    """Benchmark Bi-temporal Siamese change detector latency."""
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    detector = SiameseChangeDetector(device="cpu")

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = detect_changes(img1, img2, model=detector, tile_size=256, device="cpu")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def benchmark_retrieval(iterations: int = 5) -> dict[str, float]:
    """Benchmark vector database search latency."""
    index = SceneVectorIndex(collection_name="bench_col", vector_size=512, in_memory=True)

    # Populate index with 100 vectors
    vecs = np.random.randn(100, 512).astype(np.float32)
    payloads = [{"label": f"class_{i % 10}"} for i in range(100)]
    index.upsert_scenes(vecs, payloads=payloads)

    query_vec = np.random.randn(512).astype(np.float32)
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = index.search_by_vector(query_vec, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "qps": round(1000.0 / float(np.mean(latencies)), 2),
    }


def benchmark_grounded_vlm(iterations: int = 5) -> dict[str, float]:
    """Benchmark Grounded VLM assistant and citation verification."""
    assistant = GroundedEarthAssistant(strictness="strict")
    ev = EvidenceSynthesizer.synthesize_from_results(
        detection_result={
            "boxes": [[10, 10, 50, 50]],
            "scores": [0.92],
            "class_names": ["airplane"],
        },
        segmentation_result={
            "pixel_counts": {"Building": 400},
            "percentages": {"Building": 40.0},
            "area_hectares": {"Building": 0.04},
            "area_sq_km": {"Building": 0.0004},
        },
    )

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = assistant.ask("How many airplanes were detected?", evidence=ev)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "requests_per_sec": round(1000.0 / float(np.mean(latencies)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoVision System Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="benchmark_report.json", help="Path to save JSON report")
    args = parser.parse_args()

    logger.info(f"Running GeoVision Benchmark Suite ({args.iterations} iterations)...")

    report = {
        "geospatial_tiling": benchmark_tiling(args.iterations),
        "object_detection": benchmark_detection(args.iterations),
        "landcover_segmentation": benchmark_segmentation(args.iterations),
        "change_detection": benchmark_change_detection(args.iterations),
        "vector_retrieval": benchmark_retrieval(args.iterations),
        "grounded_vlm": benchmark_grounded_vlm(args.iterations),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Benchmark completed! Report written to {out_path}")
    print("\n--- GeoVision Performance Benchmark Summary ---")
    for module, metrics in report.items():
        print(f"[{module.upper()}] Avg Latency: {metrics['avg_latency_ms']} ms | Min: {metrics['min_latency_ms']} ms | Max: {metrics['max_latency_ms']} ms")


if __name__ == "__main__":
    main()
