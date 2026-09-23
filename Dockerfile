# Multi-stage Dockerfile for GeoVision Serving & Inference
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000

# System dependencies for GDAL/PROJ and OpenCV image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and project files
COPY requirements.txt pyproject.toml README.md ./
COPY geovision/ /app/geovision/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Install CPU-optimized PyTorch first for rapid lightweight deployment, then full requirements
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-deps -e .

# Expose Render standard container port
EXPOSE 10000

# Launch production server with dynamic port support
CMD ["python", "scripts/serve.py"]
