"""Pydantic schemas for GeoVision FastAPI REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service health status, e.g. 'ok'")
    version: str = Field(..., description="API Version")
    device: str = Field(..., description="Compute device (e.g. cpu, cuda)")
    models_status: dict[str, bool] = Field(
        default_factory=dict, description="Availability status of each subsystem"
    )


class BBoxItem(BaseModel):
    """Single bounding box detection."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    """Object detection inference response."""

    count: int = Field(..., description="Number of detected objects")
    detections: list[BBoxItem] = Field(default_factory=list)
    image_shape: list[int] = Field(..., description="[Height, Width, Channels]")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class ClassBreakdown(BaseModel):
    """Land-cover segmentation breakdown per class."""

    class_id: int
    class_name: str
    pixel_count: int
    percentage: float
    area_hectares: float
    area_sqkm: float


class SegmentationResponse(BaseModel):
    """Semantic land-cover segmentation response."""

    mask_shape: list[int] = Field(..., description="[Height, Width]")
    classes: list[ClassBreakdown] = Field(default_factory=list)
    resolution_m: float = Field(..., description="Ground sample distance in meters")
    total_area_hectares: float
    latency_ms: float


class ChangeResponse(BaseModel):
    """Bi-temporal change detection response."""

    changed_pixels: int
    total_pixels: int
    change_ratio: float
    changed_area_ha: float
    resolution_m: float
    latency_ms: float


class SearchResultItem(BaseModel):
    """Vector search retrieval result item."""

    id: str | int
    score: float
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Scene vector retrieval response."""

    query: str
    count: int
    results: list[SearchResultItem] = Field(default_factory=list)
    latency_ms: float


class ChatApiRequest(BaseModel):
    """VLM chat request schema."""

    query: str = Field(..., description="User question or instruction regarding satellite imagery")
    context_evidence: dict[str, Any] | None = Field(
        default=None, description="Optional raw or structured CV evidence dictionary"
    )
    provider: str = Field(default="deterministic", description="VLM provider: 'deterministic', 'openai', etc.")
    api_key: str | None = Field(default=None, description="Optional LLM API key")


class ChatApiResponse(BaseModel):
    """VLM chat response schema with strict citation verification."""

    query: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    grounding_score: float
    ungrounded_claims: list[str] = Field(default_factory=list)
    is_grounded: bool
    latency_ms: float


class EvidenceExtractionResponse(BaseModel):
    """Structured Earth Observation evidence extraction response."""

    timestamp: str
    evidence_tags: list[str] = Field(default_factory=list)
    summary_text: str
    raw_evidence: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float


class GeoTransformSchema(BaseModel):
    """Georeferencing affine transformation schema."""

    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    bounds: list[float] = Field(..., description="[west, south, east, north] bounding box")
    affine_matrix: list[float] = Field(default_factory=list, description="Affine 6-parameter transformation array")
    resolution_m: float | None = Field(default=None, description="Ground sampling distance in meters")


class GeoTIFFMetadataResponse(BaseModel):
    """Spatial raster metadata response."""

    filename: str
    width: int
    height: int
    bands: int
    crs: str
    transform: GeoTransformSchema


class StructuredEvidenceItem(BaseModel):
    """Granular evidence item linked to a vision subsystem finding."""

    tag: str = Field(..., description="Evidence citation token, e.g. '[DET-0]' or '[SEG-1]'")
    category: str = Field(..., description="Vision modality: 'detection', 'segmentation', 'change', 'retrieval'")
    description: str = Field(..., description="Human-readable factual finding")
    confidence: float = Field(default=1.0, description="Verification confidence score")
    payload: dict[str, Any] = Field(default_factory=dict, description="Raw feature or bbox data")


class CitationVerificationDetails(BaseModel):
    """Detailed verification metrics for grounded VLM responses."""

    total_claims: int = Field(default=0)
    verified_citations: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    hallucination_rate: float = Field(default=0.0)
    precision: float = Field(default=1.0)


