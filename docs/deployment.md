# 🐳 GeoVision Deployment & Production Guide

This guide describes containerization, Docker Compose orchestration, and production considerations for deploying the **GeoVision** platform.

---

## 1. Single-Container Docker Deployment

Build and run the standalone FastAPI backend container:

```bash
# Build Docker image
docker build -t geovision:latest .

# Run container exposing port 8000
docker run -p 8000:8000 \
  -e GEOVISION_DEVICE=cpu \
  --name geovision-api \
  geovision:latest
```

The container automatically exposes a healthcheck endpoint at `/health`.

---

## 2. Multi-Service Orchestration with Docker Compose

To run the GeoVision API alongside the Qdrant vector database:

```bash
# Launch API and Qdrant in detached mode
docker compose up -d

# Verify running services
docker compose ps

# Inspect logs
docker compose logs -f api
```

---

## 3. Production Considerations

1. **Hardware Acceleration**:
   To enable NVIDIA GPU acceleration in Docker, ensure NVIDIA Container Toolkit is installed and supply the `--gpus all` flag or `deploy.resources.reservations.devices` block in Docker Compose. Set `GEOVISION_DEVICE=cuda`.

2. **Persistent Storage**:
   Mount host directories for `./data` and `./weights` to retain downloaded raster data and trained model checkpoints across container restarts.

3. **Reverse Proxy & TLS**:
   In production environments, place NGINX, Traefik, or an AWS Application Load Balancer in front of FastAPI and configure TLS certificates.

---

## 4. 💸 Zero-Cost (₹0/Month) Free-Tier Deployment

To host GeoVision publicly with zero infrastructure cost:

### 1. Backend: Hugging Face Spaces (Docker SDK — Free 16GB RAM)
1. Create a new Space on Hugging Face (SDK: **Docker**).
2. Connect your GitHub repository.
3. Hugging Face automatically builds the `Dockerfile` and deploys the FastAPI container with **16 GB RAM** and **2 vCPUs** completely free.
4. Copy the public endpoint URL (e.g., `https://<username>-geovision.hf.space`).

### 2. Frontend: Vercel (Static React SPA — Free CDN)
1. Import your GitHub repository to [Vercel](https://vercel.com).
2. Set Root Directory to `frontend`.
3. Add Environment Variable:
   - `VITE_API_BASE_URL`: `https://<username>-geovision.hf.space`
4. Deploy. Vercel provides a custom global CDN domain with automatic SSL (e.g., `https://geovision.vercel.app`).

