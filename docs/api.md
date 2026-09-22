# 🌐 GeoVision REST API Reference

The GeoVision backend provides high-performance asynchronous REST endpoints for multimodal Earth observation inference.

Interactive Swagger documentation is available at `/docs`, and ReDoc is available at `/redoc`.

---

## 1. Endpoints Overview

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Subsystem health, device info, and loaded model status |
| `POST` | `/api/v1/detect` | Optical remote sensing object detection (NWPU VHR-10 classes) |
| `POST` | `/api/v1/segment` | Multi-class land-cover segmentation with surface area estimation |
| `POST` | `/api/v1/change` | Bi-temporal change detection on pre/post image pairs |
| `POST` | `/api/v1/search` | Natural language text-to-satellite scene vector search |
| `POST` | `/api/v1/chat` | Grounded Earth AI assistant conversation with citation verification |
| `POST` | `/api/v1/evidence` | Synthesizes structured multimodal CV evidence tokens |

---

## 2. Endpoint Details & Schemas

### `GET /health`
Returns system status and device runtime information.

**Response Schema (`HealthResponse`)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "models_loaded": {
    "detection": true,
    "segmentation": true,
    "change": true,
    "retrieval": true,
    "vlm": true
  }
}
```

---

### `POST /api/v1/detect`
Performs object detection with optional coordinate conversion to GeoJSON.

**Request Schema (`DetectionRequest`)**:
```json
{
  "image_path": "data/raw/nwpu_vhr10/images/smoke_000.jpg",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45,
  "export_geojson": false
}
```

**Response Schema (`DetectionResponse`)**:
```json
{
  "detections": [
    {
      "label": "airplane",
      "class_id": 0,
      "confidence": 0.89,
      "bbox": [120.0, 85.0, 240.0, 195.0],
      "geographic_bbox": null
    }
  ],
  "num_detections": 1,
  "inference_time_ms": 192.5
}
```

---

### `POST /api/v1/segment`
Performs land-cover semantic segmentation and calculates class area percentages.

**Request Schema (`SegmentationRequest`)**:
```json
{
  "image_path": "data/raw/landcover_ai/images/smoke_000.jpg",
  "pixel_resolution_meters": 0.5,
  "patch_size": 256,
  "stride": 128
}
```

**Response Schema (`SegmentationResponse`)**:
```json
{
  "distribution": {
    "Background": 0.35,
    "Building": 0.25,
    "Woodland": 0.15,
    "Water": 0.10,
    "Road": 0.15
  },
  "area_hectares": {
    "Building": 1.63,
    "Woodland": 0.98,
    "Water": 0.65,
    "Road": 0.98
  },
  "inference_time_ms": 340.2
}
```

---

### `POST /api/v1/change`
Detects structural bi-temporal changes between two co-registered satellite images.

**Request Schema (`ChangeRequest`)**:
```json
{
  "image_t1_path": "data/raw/levir_cd/images_t1/smoke_000.jpg",
  "image_t2_path": "data/raw/levir_cd/images_t2/smoke_000.jpg",
  "threshold": 0.5
}
```

**Response Schema (`ChangeResponse`)**:
```json
{
  "change_ratio": 0.125,
  "changed_pixels": 8192,
  "total_pixels": 65536,
  "inference_time_ms": 188.4
}
```

---

### `POST /api/v1/search`
Queries satellite scene index using natural language text embedding.

**Request Schema (`SearchRequest`)**:
```json
{
  "query_text": "Dense residential buildings near highway",
  "top_k": 5
}
```

**Response Schema (`SearchResponse`)**:
```json
{
  "results": [
    {
      "scene_id": "eurosat_patch_042",
      "score": 0.875,
      "metadata": { "label": "Residential", "path": "data/raw/eurosat/smoke_042.jpg" }
    }
  ],
  "total_found": 1,
  "search_time_ms": 3.8
}
```

---

### `POST /api/v1/chat`
Conversational satellite Earth assistant with structured citation verification.

**Request Schema (`ChatRequest`)**:
```json
{
  "query": "How many airplanes and storage tanks were detected?",
  "history": [],
  "evidence": null
}
```

**Response Schema (`ChatResponse`)**:
```json
{
  "response": "Based on satellite imagery, 1 airplane was detected [DET-0].",
  "citations": ["DET-0"],
  "hallucination_detected": false,
  "citation_precision": 1.0
}
```
