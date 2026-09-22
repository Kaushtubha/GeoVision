# 🏗️ GeoVision Architecture

This document provides an overview of the system architecture, component boundaries, and data flow of the **GeoVision** platform.

---

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    subgraph Client Layer
        UI[React 18 + Vite SPA Dashboard]
        CLI[Python CLI Suite & Scripts]
        EXT[External REST Clients / SDK]
    end

    subgraph API Gateway & Routing
        F莊[FastAPI Application Router]
        F莊 -->|POST /api/v1/detect| SVC_DET[Detection Service]
        F莊 -->|POST /api/v1/segment| SVC_SEG[Segmentation Service]
        F莊 -->|POST /api/v1/change| SVC_CHG[Change Detection Service]
        F莊 -->|POST /api/v1/search| SVC_RET[Vector Search Service]
        F莊 -->|POST /api/v1/chat| SVC_VLM[Grounded VLM Assistant]
        F莊 -->|POST /api/v1/evidence| SVC_EVI[Evidence Synthesizer]
    end

    subgraph Service & ML Core Layer
        SVC_DET --> YOLO[YOLOv8 Optical Remote Sensing Detector]
        SVC_SEG --> UNET[PyTorch Land-Cover UNet / DiceCE]
        SVC_CHG --> SIAMESE[Siamese UNet Bi-Temporal Change Detector]
        SVC_RET --> CLIP[OpenCLIP ViT-B-32 Scene Embedder]
        SVC_VLM --> VERIFY[CitationVerifier & Grounding Engine]
        SVC_EVI --> MULTI[Multimodal CV Evidence Collector]
    end

    subgraph Geospatial Engine
        GEO[CRS Transformer & GeoTIFF I/O]
        TILER[RasterTiler & PatchReconstructor]
        YOLO --> TILER
        UNET --> TILER
        SIAMESE --> TILER
    end

    subgraph Data & Storage Layer
        CLIP --> QDRANT[(Qdrant Vector DB / In-Memory Store)]
        TRACK[MLflow Experiment Tracker]
        FS[(Local / Mounted Geospatial Storage)]
    end

    UI --> F莊
    CLI --> SVC_DET & SVC_SEG & SVC_CHG & SVC_RET & SVC_VLM
    EXT --> F莊
```

---

## 2. Subsystem Descriptions

### 2.1 Geospatial & CRS Engine (`geovision.geo`)
- **`CoordinateTransformer`**: Handles forward and inverse conversions between raster pixel coordinates $(x, y)$ and Geographic/Projected coordinates (EPSG:4326 WGS84, UTM zones, EPSG:2180).
- **`GeoTIFFReader` / `GeoTIFFWriter`**: Reads georeferenced rasters, preserves affine transformation matrices and coordinate reference systems, and exports spatial metadata.
- **`RasterTiler` & `PatchReconstructor`**: Implements overlapping sliding-window inference with 2D Hann window blending to eliminate boundary artifacts across large-scale satellite orthophotos.

### 2.2 Computer Vision & Deep Learning Engines
- **Object Detection (`geovision.detection`)**: High-resolution optical feature detector across 10 remote sensing categories with bounding box extraction and GeoJSON serialization.
- **Land-Cover Segmentation (`geovision.segmentation`)**: Multi-class semantic segmenter predicting 5 classes (*Background, Building, Woodland, Water, Road*) with class distribution and surface area calculations ($m^2$, hectares, $km^2$).
- **Siamese Change Detection (`geovision.change`)**: Dual-encoder Siamese UNet architecture processing pre-event ($T_1$) and post-event ($T_2$) pairs to detect structural land modifications with tri-panel visualization.
- **Metric Learning & Vector Retrieval (`geovision.retrieval`)**: Normalized 512-dim OpenCLIP ViT-B-32 / lightweight vision-text projection embedder indexed in Qdrant for semantic scene discovery.

### 2.3 Grounded VLM Earth Assistant (`geovision.vlm`)
- **`EvidenceSynthesizer`**: Compiles visual discoveries into structured citations: `[DET-x]`, `[SEG-x]`, `[CHG-x]`, `[GEO-x]`, and `[RET-x]`.
- **`CitationVerifier`**: Factual verification layer computing citation precision, recall, and hallucination detection before returning answers.

### 2.4 Production Serving & Experimentation
- **FastAPI Layer (`geovision.api`)**: Async endpoints, Pydantic v2 validation schemas, dependency injection, and health monitoring.
- **MLflow Tracking (`geovision.tracking`)**: Structured metric logging and run lifecycle tracking with offline fallback.
