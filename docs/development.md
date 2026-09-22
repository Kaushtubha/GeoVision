# 💻 GeoVision Developer Guide

This guide covers local environment setup, virtual environment management, dataset generation, automated testing, and development workflows.

---

## 1. Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: v18+ or v20+ (for Frontend dashboard)
- **C/C++ Build Tools**: Required for compiled geospatial/GDAL wheel dependencies (standard Linux `build-essential` or Windows Visual C++ Build Tools).

---

## 2. Python Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/Kaushtubha/GeoVision.git
cd GeoVision

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
```

---

## 3. Synthetic Smoke Dataset Generation

Generate lightweight synthetic smoke subsets across all 4 supported datasets in under 5 seconds:

```bash
python scripts/download_datasets.py --smoke
```

This populates `data/raw/` with standardized miniature datasets for NWPU VHR-10, LandCover.ai, LEVIR-CD, and EuroSAT.

---

## 4. Running Tests & Linters

```bash
# Run all 48 unit and integration tests
pytest tests/ -v

# Run code style & formatting checks
ruff check .
```

---

## 5. Launching Local Services

### Start FastAPI Backend
```bash
python scripts/serve.py --host 127.0.0.1 --port 8000 --reload
```
Swagger UI will be accessible at `http://127.0.0.1:8000/docs`.

### Start React + Vite Frontend
```bash
cd frontend
npm install
npm run dev
```
Interactive UI will be accessible at `http://localhost:3000` (or `http://localhost:5173`).
