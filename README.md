# 🌍 GeoVision: Multimodal Satellite-Intelligence & Earth Observation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tested with: pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://docs.pytest.org/en/stable/)

**GeoVision** is a production-grade, modular, and interview-defensible Earth Observation (EO) and satellite AI intelligence platform.

---

## 🛰️ Core Capabilities

1. **Object Detection**: High-resolution aerial optical remote sensing detection (airplanes, ships, storage tanks, harbors, bridges, vehicles).
2. **Semantic Land-Cover Segmentation**: Orthophoto multi-class land-cover segmentation with class area percentage reporting.
3. **Bi-Temporal Change Detection**: Siamese difference network producing verified binary change masks and statistical shift metrics.
4. **Metric Learning & Scene Retrieval**: Contrastive CLIP-based satellite embeddings indexed in embedded Qdrant vector database (evaluated with Recall@1/5/10).
5. **Grounded VLM Earth Assistant**: Natural-language satellite QA assistant grounded strictly in CV module structured evidence JSON without hallucinated claims.
6. **Geospatial & CRS Engine**: Full GeoTIFF raster parsing, CRS reprojection, affine transform handling, and pixel $\leftrightarrow$ Lat/Lon coordinate conversions.
7. **Serving & Ops**: FastAPI REST endpoints, MLflow experiment tracking, configuration management, and reproducible `--smoke` modes.

---

## 🏗️ Repository Architecture

```
GeoVision/
├── configs/               # Hierarchical YAML configs for all modules & experiments
│   ├── default.yaml
│   ├── detection.yaml
│   ├── segmentation.yaml
│   ├── change.yaml
│   ├── retrieval.yaml
│   └── tracking.yaml
├── data/                  # Raw & processed data directory (gitignored)
│   ├── raw/
│   └── processed/
├── geovision/             # Core Python package
│   ├── __init__.py
│   ├── config.py          # Type-safe Pydantic v2 configuration engine
│   ├── constants.py       # Dataset class mappings and directory paths
│   ├── logger.py          # Rich structured logging
│   ├── data/              # Downloaders, dataset loaders, statistics
│   ├── tracking/          # MLflow experiment tracking integration
│   ├── geo/               # CRS, GeoTIFF, coordinate transformation
│   ├── detection/         # Object detection pipeline
│   ├── segmentation/      # Land-cover segmentation pipeline
│   ├── change/            # Siamese change detection pipeline
│   ├── retrieval/         # Vector search & metric learning
│   ├── vlm/               # Grounded VLM assistant & evidence synthesis
│   ├── metrics/           # Evaluation metrics (mAP, mIoU, Dice, Recall@K)
│   └── api/               # FastAPI routers & schemas
├── notebooks/             # Exploratory notebooks
│   └── 01_data_exploration.ipynb
├── scripts/               # CLI tools for download, exploration, training, eval
│   ├── download_datasets.py
│   └── explore_data.py
├── tests/                 # Comprehensive unit test suite (pytest)
│   ├── test_config.py
│   ├── test_tracking.py
│   └── test_data_download.py
├── pyproject.toml         # Package metadata and tool configurations
└── requirements.txt       # Pinned requirements
```

---

## 🚀 Quickstart & Verification

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download or Generate Smoke Datasets (< 5 Seconds)
```bash
# Generate lightweight smoke-test subsets for fast CPU pipeline verification
python scripts/download_datasets.py --smoke
```

### 3. Explore Dataset Distributions
```bash
python scripts/explore_data.py
```

### 4. Run Pytest Suite & Linting
```bash
# Run unit tests
pytest

# Run linter
ruff check .
```

---

## 📊 Dataset Specifications

| Module | Dataset | Resolution / GSD | Classes | License |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | **NWPU VHR-10** | 0.08–2.0m | 10 classes (airplane, ship, harbor, bridge, etc.) | Academic / Research |
| **Segmentation** | **LandCover.ai** | 0.25–0.5m | 4 classes + background (Building, Woodland, Water, Road) | CC BY-NC-SA 4.0 |
| **Change Detection** | **LEVIR-CD** | 0.5m | Binary building change mask | CC0 / Academic |
| **Retrieval** | **EuroSAT** | 10m (Sentinel-2) | 10 Land Use / Land Cover classes | MIT License |

---

## 📈 Phase Roadmap
- [x] **Phase 0: Plan** — System hardware audit, dataset verification, technical blueprint.
- [x] **Phase 1: Foundation** — Architecture skeleton, Pydantic configs, MLflow tracking, verified downloaders, smoke test pipeline, pytest & ruff CI.
- [x] **Phase 2: Geospatial & Data Pipeline** — GeoTIFF reader, CRS reprojection, affine transforms, windowed raster tiling.
- [x] **Phase 3: Object Detection** — YOLO nano training, `--smoke` mode, mAP@50-95 evaluator, geo-referenced bounding box outputs.
- [x] **Phase 4: Semantic Segmentation** — SegFormer / SMP UNet, mIoU / Dice metrics, class area % reporting.
- [x] **Phase 5: Change Detection** — Siamese difference network, binary change masks & stats.
- [ ] **Phase 6: Scene Retrieval** — Zero-shot vs Fine-tuned CLIP embeddings, embedded Qdrant index, Recall@K benchmark.
- [ ] **Phase 7: Grounded VLM Earth Assistant** — Evidence synthesizer, citation verification, satellite QA eval set.
- [ ] **Phase 8: Serving & Research Benchmarks** — FastAPI REST API, Dockerfile, augmentation ablation study, model cards.
