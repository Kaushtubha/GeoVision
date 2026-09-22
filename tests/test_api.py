"""Unit and integration tests for GeoVision FastAPI REST API."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geovision.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def make_dummy_image_bytes(width: int = 128, height: int = 128) -> bytes:
    """Helper to generate dummy RGB JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint(client: TestClient):
    """Test GET /health returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "device" in data
    assert "models_status" in data
    assert data["models_status"]["detection"] is True


def test_detect_endpoint(client: TestClient):
    """Test POST /api/v1/detect with dummy image file."""
    img_bytes = make_dummy_image_bytes(256, 256)
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"conf_thresh": "0.3"}

    response = client.post("/api/v1/detect", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "count" in res
    assert "detections" in res
    assert "image_shape" in res
    assert "latency_ms" in res
    assert res["image_shape"] == [256, 256, 3]


def test_segment_endpoint(client: TestClient):
    """Test POST /api/v1/segment with dummy image file."""
    img_bytes = make_dummy_image_bytes(128, 128)
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"resolution_m": "1.5"}

    response = client.post("/api/v1/segment", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "mask_shape" in res
    assert "classes" in res
    assert len(res["classes"]) == 5
    assert res["resolution_m"] == 1.5
    assert "total_area_hectares" in res


def test_change_endpoint(client: TestClient):
    """Test POST /api/v1/change with two dummy images."""
    img1_bytes = make_dummy_image_bytes(128, 128)
    img2_bytes = make_dummy_image_bytes(128, 128)
    files = {
        "file_t1": ("t1.jpg", img1_bytes, "image/jpeg"),
        "file_t2": ("t2.jpg", img2_bytes, "image/jpeg"),
    }
    data = {"resolution_m": "0.5"}

    response = client.post("/api/v1/change", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "changed_pixels" in res
    assert "change_ratio" in res
    assert "changed_area_ha" in res
    assert res["resolution_m"] == 0.5


def test_search_endpoint(client: TestClient):
    """Test POST /api/v1/search with text query."""
    data = {"query_text": "dense urban residential buildings", "top_k": "3"}
    response = client.post("/api/v1/search", data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["query"] == "dense urban residential buildings"
    assert "results" in res
    assert "latency_ms" in res


def test_chat_endpoint(client: TestClient):
    """Test POST /api/v1/chat with structured query."""
    payload = {
        "query": "How many aircraft are on the tarmac?",
        "context_evidence": {
            "detections": [
                {
                    "class_name": "airplane",
                    "confidence": 0.95,
                    "xmin": 10.0,
                    "ymin": 10.0,
                    "xmax": 50.0,
                    "ymax": 50.0,
                },
                {
                    "class_name": "airplane",
                    "confidence": 0.88,
                    "xmin": 60.0,
                    "ymin": 60.0,
                    "xmax": 90.0,
                    "ymax": 90.0,
                },
            ]
        },
        "provider": "deterministic",
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "2" in res["answer"]
    assert any("DET" in c for c in res["citations"])
    assert res["is_grounded"] is True
    assert res["grounding_score"] >= 0.8


def test_evidence_endpoint(client: TestClient):
    """Test POST /api/v1/evidence extraction from raw image."""
    img_bytes = make_dummy_image_bytes(128, 128)
    files = {"file": ("scene.jpg", img_bytes, "image/jpeg")}
    data = {"resolution_m": "1.0"}

    response = client.post("/api/v1/evidence", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert "evidence_tags" in res
    assert "summary_text" in res
    assert "raw_evidence" in res
    assert len(res["evidence_tags"]) > 0
