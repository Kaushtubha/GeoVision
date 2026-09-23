# Multi-stage Dockerfile for GeoVision Serving & Inference
FROM python:3.11-slim as base

# Prevent Python from writing .pyc and buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies for GDAL/PROJ and OpenCV image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Copy application source and configs
COPY geovision/ /app/geovision/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Expose FastAPI port
EXPOSE 8000

# Launch production server with dynamic port support
CMD ["python", "scripts/serve.py"]
