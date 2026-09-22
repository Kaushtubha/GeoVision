# 🌍 GeoVision: Multimodal Satellite-Intelligence & Earth Observation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 48 passed](https://img.shields.io/badge/tests-48%20passed-success.svg)](https://docs.pytest.org/en/stable/)
[![FastAPI Serving](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)

**GeoVision** is an end-to-end, production-grade, and interview-defensible Earth Observation (EO) and Multimodal Satellite Artificial Intelligence Platform.

---

## 🛰️ Core Capabilities & Subsystems

1. **Geospatial & CRS Engine**:
   - Georeferenced raster I/O supporting GeoTIFF, standard RGB, and multispectral arrays.
   - Dual-directional coordinate transformation between Pixel Coordinates $(x, y)$ and Geographic/Projected Coordinates (WGS84 EPSG:4326, UTM, EPSG:2180).
   - Overlapping sliding-window tiler (`RasterTiler`) and seamless Hann-window blended patch reconstructor (`PatchReconstructor`).
2. **Object Detection**:
   - High-resolution optical remote sensing detection across 10 NWPU VHR-10 categories (*airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle*).
   - COCO-style multi-threshold mAP@50-95 metric calculation and GeoJSON bounding box export.
3. **Semantic Land-Cover Segmentation**:
   - Multi-class orthophoto land-cover segmentation (*Background, Building, Woodland, Water, Road*).
   - Weighted Compound Loss (`DiceCELoss`, `FocalLoss`), confusion matrix metric calculation (mIoU, Dice, Overall Accuracy), and surface area calculation ($m^2$, hectares, $km^2$).
4. **Bi-Temporal Change Detection**:
   - Siamese UNet architecture (`SiameseUNet`, `SiameseChangeDetector`) taking pre-event ($T_1$) and post-event ($T_2$) satellite pairs.
   - Combined BCE + Binary Dice loss (`ChangeLoss`), Change-IoU, F1 score, precision/recall, and tri-panel visual report generation ($T_1$, $T_2$, Change Overlay).
5. **Metric Learning & Scene Vector Retrieval**:
   - OpenCLIP ViT-B-32 / lightweight vision-text projection embedder mapping satellite scenes into 512-dim normalized metric space.
   - Vector database indexing backed by embedded Qdrant / in-memory store with Recall@1/5/10, MRR, mAP, and zero-shot evaluation.
6. **Grounded VLM Earth Assistant**:
   - Factual natural-language satellite Q&A assistant with strict citation verification (`CitationVerifier`).
   - Grounded directly in structured CV evidence (`[DET-x]`, `[SEG-x]`, `[CHG-x]`, `[GEO-x]`, `[RET-x]`) with zero hallucination.
7. **FastAPI REST Serving & Benchmarking**:
   - Production FastAPI microservice (`/health`, `/api/v1/detect`, `/api/v1/segment`, `/api/v1/change`, `/api/v1/search`, `/api/v1/chat`, `/api/v1/evidence`).
   - Automated benchmark suite profiling end-to-end throughput and latency across all modules.

---

## 🏗️ Repository Structure

```
GeoVision/
├── configs/                     # Hierarchical YAML configs for all modules & experiments
│   ├── default.yaml
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── change.yaml
│   ├── retrieval.yaml
│   └── tracking.yaml
├── data/                        # Raw & processed data directory (gitignored)
│   ├── raw/
│   └── processed/
├── geovision/                   # Core Python package
│   ├── __init__.py
│   ├── config.py                # Type-safe Pydantic v2 configuration engine
│   ├── constants.py             # Dataset class mappings and directory paths
│   ├── logger.py                # Rich structured logging
│   ├── api/                     # FastAPI REST app, routers, schemas & services
│   ├── change/                  # Siamese change detection pipeline
│   ├── data/                    # Dataset loaders, smoke generators, statistics
│   ├── detection/               # YOLO object detection pipeline
│   ├── geo/                     # CRS transformations, GeoTIFF I/O, tiling engine
│   ├── metrics/                 # Strict evaluation metrics (mAP, mIoU, Dice, Recall@K)
│   ├── retrieval/               # OpenCLIP embedder & Qdrant vector retrieval
│   ├── segmentation/            # Land-cover segmentation pipeline
│   ├── tracking/                # MLflow experiment tracking integration
│   └── vlm/                     # Grounded VLM assistant, synthesizer & verifier
├── scripts/                     # Production CLI tools
│   ├── benchmark_suite.py       # End-to-end latency & throughput benchmark suite
│   ├── build_scene_index.py     # Qdrant scene vector index builder
│   ├── chat_assistant.py        # Interactive Grounded Earth Assistant CLI
│   ├── download_datasets.py     # Real dataset downloader & smoke data generator
│   ├── eval_change.py           # Change detection benchmark evaluation CLI
│   ├── eval_detection.py        # Object detection benchmark evaluation CLI
│   ├── eval_retrieval.py        # Vector retrieval benchmark evaluation CLI
│   ├── eval_segmentation.py     # Segmentation benchmark evaluation CLI
│   ├── eval_vlm.py              # Satellite VQA benchmark evaluation CLI
│   ├── explore_data.py          # Data exploration & distribution visualizer
│   ├── infer_change.py          # Bi-temporal change detection inference CLI
│   ├── infer_detection.py       # Object detection inference CLI
│   ├── infer_segmentation.py    # Segmentation inference CLI
│   ├── query_scene.py           # Natural language satellite search CLI
│   ├── serve.py                 # Uvicorn production server runner
│   ├── train_change.py          # Change detection training CLI
│   ├── train_detection.py       # Object detection training CLI
│   ├── train_retrieval.py       # Metric learning training CLI
│   └── train_segmentation.py    # Segmentation training CLI
├── tests/                       # Comprehensive pytest suite (48 tests)
│   ├── test_api.py
│   ├── test_change.py
│   ├── test_config.py
│   ├── test_data_download.py
│   ├── test_detection.py
│   ├── test_geo.py
│   ├── test_retrieval.py
│   ├── test_segmentation.py
│   ├── test_tracking.py
│   └── test_vlm.py
├── Dockerfile                   # Multi-stage production container definition
├── pyproject.toml               # Package metadata and tool configurations
└── requirements.txt             # Pinned requirements
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Kaushtubha/GeoVision.git
cd GeoVision

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell (or source .venv/bin/activate on Linux/macOS)

# Install package in editable mode
pip install -e ".[all]"
```

### 2. Generate Smoke Datasets (< 5 Seconds)
```bash
# Generate lightweight smoke subsets for instant CPU validation across all 4 datasets
python scripts/download_datasets.py --smoke
```

### 3. Run Pytest Suite & Linting
```bash
# Run all 48 unit and integration tests
pytest

# Run code style & lint checks
ruff check .
```

---

## 💻 CLI Commands & Workflows

### 🛰️ Object Detection (NWPU VHR-10)
```bash
# Smoke training on CPU
python scripts/train_detection.py --smoke --epochs 1

# Evaluation on test set
python scripts/eval_detection.py --smoke

# Run inference
python scripts/infer_detection.py --input data/raw/nwpu_vhr10/images/smoke_000.jpg --output outputs/detection/
```

### 🗺️ Land-Cover Semantic Segmentation (LandCover.ai)
```bash
# Smoke training
python scripts/train_segmentation.py --smoke --epochs 1

# Evaluation
python scripts/eval_segmentation.py --smoke

# Inference with area statistics
python scripts/infer_segmentation.py --input data/raw/landcover_ai/images/smoke_000.jpg --output outputs/segmentation/
```

### 🔄 Bi-Temporal Change Detection (LEVIR-CD)
```bash
# Smoke training
python scripts/train_change.py --smoke --epochs 1

# Evaluation
python scripts/eval_change.py --smoke

# Inference generating tri-panel report
python scripts/infer_change.py --image-a data/raw/levir_cd/images_t1/smoke_000.jpg --image-b data/raw/levir_cd/images_t2/smoke_000.jpg --output outputs/change/
```

### 🔍 Scene Vector Retrieval (EuroSAT / Qdrant)
```bash
# Build vector index
python scripts/build_scene_index.py --smoke

# Query index using natural language text
python scripts/query_scene.py --text "Dense residential buildings near highway" --top-k 5
```

### 🤖 Grounded VLM Earth Assistant
```bash
# Interactive single-turn query
python scripts/chat_assistant.py --query "How many airplanes and storage tanks were detected?"

# Run Satellite VQA Benchmark
python scripts/eval_vlm.py
```

### 🌐 FastAPI Production Serving
```bash
# Start API server on localhost:8000
python scripts/serve.py --host 0.0.0.0 --port 8000

# Access Interactive Swagger Docs:
# http://127.0.0.1:8000/docs
# Access ReDoc:
# http://127.0.0.1:8000/redoc
```

### 📊 End-to-End Latency Benchmark Suite
```bash
# Profile latency and throughput across all subsystems
python scripts/benchmark_suite.py --iterations 5 --output benchmark_report.json
```

---

## 🐳 Docker Deployment

```bash
# Build container image
docker build -t geovision:latest .

# Run container with FastAPI exposed on port 8000
docker run -p 8000:8000 --name geovision-service geovision:latest
```

---

## 🧪 Verification Matrix

| Subsystem | Module Path | Test File | Test Status |
| :--- | :--- | :--- | :--- |
| **Config & Constants** | `geovision/config.py`, `constants.py` | `tests/test_config.py` | [PASSED] (4/4) |
| **Data & Smoke** | `geovision/data/` | `tests/test_data_download.py` | [PASSED] (4/4) |
| **Geospatial Engine** | `geovision/geo/` | `tests/test_geo.py` | [PASSED] (6/6) |
| **Object Detection** | `geovision/detection/` | `tests/test_detection.py` | [PASSED] (5/5) |
| **Segmentation** | `geovision/segmentation/` | `tests/test_segmentation.py` | [PASSED] (6/6) |
| **Change Detection** | `geovision/change/` | `tests/test_change.py` | [PASSED] (6/6) |
| **Vector Retrieval** | `geovision/retrieval/` | `tests/test_retrieval.py` | [PASSED] (4/4) |
| **Grounded VLM** | `geovision/vlm/` | `tests/test_vlm.py` | [PASSED] (4/4) |
| **FastAPI REST API** | `geovision/api/` | `tests/test_api.py` | [PASSED] (7/7) |
| **MLflow Tracking** | `geovision/tracking/` | `tests/test_tracking.py` | [PASSED] (2/2) |
| **Total** | **All 8 Phases** | **Full Suite** | **48 / 48 (100% Passed)** |

---

## 📄 License

MIT License. Designed and developed for state-of-the-art satellite intelligence and Earth observation applications.
