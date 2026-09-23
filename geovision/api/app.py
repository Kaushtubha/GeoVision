"""FastAPI application for GeoVision Multimodal Earth Observation Serving."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from geovision import __version__
from geovision.api.schemas import (
    ChangeResponse,
    ChatApiRequest,
    ChatApiResponse,
    DetectionResponse,
    EvidenceExtractionResponse,
    HealthResponse,
    SearchResponse,
    SegmentationResponse,
)
from geovision.api.services import GeoVisionServices, decode_image_bytes
from geovision.config import load_config
from geovision.vlm.evidence import EvidenceSynthesizer

# Global services container
services: GeoVisionServices | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model warming & resource allocation."""
    global services
    config = load_config()
    services = GeoVisionServices(config=config)
    yield
    # Teardown / cleanup if needed
    services = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="GeoVision API",
        version=__version__,
        description=(
            "Multimodal Satellite-Intelligence & Earth Observation Platform REST API. "
            "Provides endpoints for object detection, land-cover segmentation, "
            "bi-temporal change detection, scene vector retrieval, and grounded VLM chat."
        ),
        lifespan=lifespan,
    )

    cors_origins_env = os.getenv("CORS_ORIGINS", "*")
    if cors_origins_env.strip() == "*":
        origins = ["*"]
        allow_creds = False
    else:
        origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        allow_creds = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.api_route("/", methods=["GET", "HEAD"], tags=["Root"])
    async def root():
        """Root status probe for cloud platforms and load balancers."""
        return {
            "status": "ok",
            "service": "GeoVision Multimodal Earth Observation API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Check API service health and subsystem status."""
        global services
        if services is None:
            services = GeoVisionServices()

        return HealthResponse(
            status="ok",
            version=__version__,
            device=services.device,
            models_status={
                "detection": True,
                "segmentation": True,
                "change": True,
                "retrieval": True,
                "vlm": True,
            },
        )

    @app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Inference"])
    async def detect_objects(
        file: UploadFile = File(..., description="Satellite image file (PNG, JPEG, GeoTIFF)"),
        conf_thresh: float = Form(0.25, description="Confidence threshold"),
    ):
        """Perform object detection on satellite imagery."""
        global services
        if services is None:
            services = GeoVisionServices()

        try:
            contents = await file.read()
            image_np = decode_image_bytes(contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}",
            )

        res = services.run_detection(image_np, conf_thresh=conf_thresh)
        return DetectionResponse(**res)

    @app.post("/api/v1/segment", response_model=SegmentationResponse, tags=["Inference"])
    async def segment_landcover(
        file: UploadFile = File(..., description="Satellite image file (PNG, JPEG, GeoTIFF)"),
        resolution_m: float = Form(1.0, description="Spatial resolution in meters/pixel"),
    ):
        """Perform 5-class semantic land-cover segmentation and area calculation."""
        global services
        if services is None:
            services = GeoVisionServices()

        try:
            contents = await file.read()
            image_np = decode_image_bytes(contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}",
            )

        res = services.run_segmentation(image_np, resolution_m=resolution_m)
        return SegmentationResponse(**res)

    @app.post("/api/v1/change", response_model=ChangeResponse, tags=["Inference"])
    async def detect_changes(
        file_t1: UploadFile = File(..., description="Time 1 (pre-change) satellite image"),
        file_t2: UploadFile = File(..., description="Time 2 (post-change) satellite image"),
        resolution_m: float = Form(1.0, description="Spatial resolution in meters/pixel"),
    ):
        """Perform bi-temporal change detection between two satellite images."""
        global services
        if services is None:
            services = GeoVisionServices()

        try:
            content1 = await file_t1.read()
            content2 = await file_t2.read()
            img1 = decode_image_bytes(content1)
            img2 = decode_image_bytes(content2)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}",
            )

        res = services.run_change_detection(img1, img2, resolution_m=resolution_m)
        return ChangeResponse(**res)

    @app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
    async def search_scenes(
        query_text: str | None = Form(None, description="Text search query (e.g. 'Dense residential')"),
        file: UploadFile | None = File(None, description="Optional image search query"),
        top_k: int = Form(5, description="Number of results to return"),
    ):
        """Perform multimodal text-to-image or image-to-image vector retrieval."""
        global services
        if services is None:
            services = GeoVisionServices()

        img_np = None
        if file is not None:
            try:
                contents = await file.read()
                if len(contents) > 0:
                    img_np = decode_image_bytes(contents)
            except Exception:
                img_np = None

        if not query_text and img_np is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either 'query_text' or 'file'.",
            )

        res = services.run_search(query_text=query_text, query_image=img_np, top_k=top_k)
        return SearchResponse(**res)

    @app.post("/api/v1/chat", response_model=ChatApiResponse, tags=["VLM Assistant"])
    async def chat_vlm(request: ChatApiRequest):
        """Ask Earth Assistant a question with strict citation verification."""
        global services
        if services is None:
            services = GeoVisionServices()

        res = services.run_chat(
            query=request.query,
            context_evidence=request.context_evidence,
            provider=request.provider,
            api_key=request.api_key,
        )
        return ChatApiResponse(**res)

    @app.post("/api/v1/evidence", response_model=EvidenceExtractionResponse, tags=["VLM Assistant"])
    async def extract_evidence(
        file: UploadFile = File(..., description="Primary satellite image"),
        resolution_m: float = Form(1.0, description="Spatial resolution in meters/pixel"),
    ):
        """Extract unified CV evidence from satellite image for downstream VLM reasoning."""
        global services
        if services is None:
            services = GeoVisionServices()

        t0 = time.perf_counter()
        try:
            contents = await file.read()
            image_np = decode_image_bytes(contents)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {e}",
            )

        det_raw = services.get_detector().predict(image_np)
        seg_res = services.run_segmentation(image_np, resolution_m=resolution_m)

        ev = EvidenceSynthesizer.synthesize_from_results(
            detection_result=det_raw,
            segmentation_result=seg_res,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return EvidenceExtractionResponse(
            timestamp=str(time.time()),
            evidence_tags=[item.tag for item in ev.evidence_items],
            summary_text=ev.to_markdown_context(),
            raw_evidence=ev.model_dump(),
            latency_ms=round(latency_ms, 2),
        )

    return app


app = create_app()
