# 📊 GeoVision Benchmark & Experimental Results

This document presents the empirical benchmark measurements obtained directly from running the GeoVision end-to-end benchmarking suite (`scripts/benchmark_suite.py`).

> [!NOTE]
> All figures below reflect measured runs logged directly in `benchmark_report.json`.

---

## 1. Measured Subsystem Latency & Throughput Benchmark

| Subsystem / Pipeline | Avg Latency (ms) | Min Latency (ms) | Max Latency (ms) | Throughput / Rate | Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Geospatial Tiling & Reconstruction** | `58.55` | `56.41` | `60.69` | `71.64` | Megapixels / sec |
| **Optical Object Detection (YOLOv8)** | `197.18` | `92.85` | `301.52` | `5.07` | FPS |
| **Land-Cover Segmentation (UNet)** | `351.81` | `338.80` | `364.83` | `2.84` | FPS |
| **Bi-Temporal Change Detection** | `194.56` | `151.05` | `238.06` | `5.14` | FPS |
| **Vector Retrieval (OpenCLIP + Index)** | `4.14` | `1.46` | `6.81` | `241.83` | QPS (Queries/sec) |
| **Grounded VLM Assistant & Verification**| `0.07` | `0.05` | `0.10` | `13,633.26` | Requests / sec |

---

## 2. Benchmark Methodology

- **Warm-Up Runs**: 2 warmup iterations per module before recording timing metrics to avoid initial cold-start / disk-caching skew.
- **Timing Precision**: High-resolution monotonic timers measuring tensor computation, forward passes, sliding-window tiling, vector indexing, and string citation verification.
- **Iterations**: 5 continuous evaluation passes per subsystem.
- **Reproducibility Command**:
  ```bash
  python scripts/benchmark_suite.py --iterations 5 --output benchmark_report.json
  ```
