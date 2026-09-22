"""Service layer for GeoVision FastAPI serving."""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from PIL import Image

from geovision.change import SiameseChangeDetector, detect_changes
from geovision.config import GeoVisionConfig, load_config
from geovision.constants import LANDCOVER_CLASSES
from geovision.detection import YOLODetector
from geovision.retrieval import SatelliteSceneEmbedder, SceneVectorIndex
from geovision.segmentation import LandCoverSegmenter, segment_image
from geovision.vlm.assistant import GroundedEarthAssistant
from geovision.vlm.evidence import EarthObservationEvidence, EvidenceSynthesizer


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (PNG, JPEG, TIFF) to RGB numpy array uint8."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


class GeoVisionServices:
    """Singleton/Container holding instantiated subsystems for the API."""

    def __init__(self, config: GeoVisionConfig | None = None) -> None:
        self.config = config or load_config()
        self.device = getattr(self.config.detection, "device", "cpu")

        # Lazy handles
        self._detector: YOLODetector | None = None
        self._segmenter: LandCoverSegmenter | None = None
        self._change_detector: SiameseChangeDetector | None = None
        self._embedder: SatelliteSceneEmbedder | None = None
        self._vector_index: SceneVectorIndex | None = None
        self._assistant: GroundedEarthAssistant | None = None
        self._synthesizer = EvidenceSynthesizer()

    def get_detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(
                model_path=self.config.detection.model_name,
                device=self.device,
            )
        return self._detector

    def get_segmenter(self) -> LandCoverSegmenter:
        if self._segmenter is None:
            self._segmenter = LandCoverSegmenter(
                encoder_name=self.config.segmentation.encoder_name,
                num_classes=self.config.segmentation.num_classes,
                device=self.device,
            )
        return self._segmenter

    def get_change_detector(self) -> SiameseChangeDetector:
        if self._change_detector is None:
            self._change_detector = SiameseChangeDetector(
                device=self.device,
            )
        return self._change_detector

    def get_embedder(self) -> SatelliteSceneEmbedder:
        if self._embedder is None:
            self._embedder = SatelliteSceneEmbedder(
                model_name=self.config.retrieval.model_name,
                pretrained=None,  # Use offline/instant lightweight embedder for API serving
                embedding_dim=self.config.retrieval.embedding_dim,
                device=self.device,
            )
        return self._embedder

    def get_vector_index(self) -> SceneVectorIndex:
        if self._vector_index is None:
            self._vector_index = SceneVectorIndex(
                collection_name="geovision_scenes",
                vector_size=self.config.retrieval.embedding_dim,
                in_memory=True,
            )
        return self._vector_index

    def get_assistant(self) -> GroundedEarthAssistant:
        if self._assistant is None:
            self._assistant = GroundedEarthAssistant(strictness="strict")
        return self._assistant

    def run_detection(self, image_np: np.ndarray, conf_thresh: float = 0.25) -> dict[str, Any]:
        """Run object detection on an RGB image array."""
        t0 = time.perf_counter()
        detector = self.get_detector()
        detections = []
        try:
            preds = detector.predict(image_np, conf_threshold=conf_thresh)
            boxes = preds.get("boxes", [])
            scores = preds.get("scores", [])
            labels = preds.get("labels", [])
            class_names = preds.get("class_names", [])

            for i in range(len(boxes)):
                box = boxes[i]
                detections.append({
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3]),
                    "confidence": float(scores[i]) if i < len(scores) else 1.0,
                    "class_id": int(labels[i]) if i < len(labels) else 0,
                    "class_name": class_names[i] if i < len(class_names) else "object",
                })
        except Exception:
            detections = []

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "count": len(detections),
            "detections": detections,
            "image_shape": list(image_np.shape),
            "latency_ms": round(latency_ms, 2),
        }

    def run_segmentation(self, image_np: np.ndarray, resolution_m: float = 1.0) -> dict[str, Any]:
        """Run land-cover segmentation on an RGB image array."""
        t0 = time.perf_counter()
        segmenter = self.get_segmenter()
        tile_sz = min(self.config.segmentation.image_size, image_np.shape[0], image_np.shape[1])
        if tile_sz < 64:
            tile_sz = min(image_np.shape[0], image_np.shape[1])

        seg_output = segment_image(
            image_input=image_np,
            model=segmenter,
            tile_size=tile_sz,
            gsd_meters=resolution_m,
            device=self.device,
        )
        mask = seg_output["mask"]
        stats = seg_output.get("stats", {})

        pixel_counts = stats.get("pixel_counts", {})
        percentages = stats.get("percentages", {})
        area_ha = stats.get("area_hectares", {})
        area_sqkm = stats.get("area_sq_km", {})

        classes_list = []
        for cid, cname in LANDCOVER_CLASSES.items():
            classes_list.append({
                "class_id": cid,
                "class_name": cname,
                "pixel_count": int(pixel_counts.get(cname, 0)),
                "percentage": float(percentages.get(cname, 0.0)),
                "area_hectares": float(area_ha.get(cname, 0.0)),
                "area_sqkm": float(area_sqkm.get(cname, 0.0)),
            })

        total_area_ha = sum(c["area_hectares"] for c in classes_list)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mask_shape": list(mask.shape),
            "classes": classes_list,
            "resolution_m": resolution_m,
            "total_area_hectares": round(total_area_ha, 4),
            "latency_ms": round(latency_ms, 2),
        }

    def run_change_detection(
        self, img_t1: np.ndarray, img_t2: np.ndarray, resolution_m: float = 1.0
    ) -> dict[str, Any]:
        """Run bi-temporal change detection."""
        t0 = time.perf_counter()
        change_det = self.get_change_detector()
        tile_sz = min(self.config.change.image_size, img_t1.shape[0], img_t1.shape[1])
        if tile_sz < 64:
            tile_sz = min(img_t1.shape[0], img_t1.shape[1])

        chg_output = detect_changes(
            image_a=img_t1,
            image_b=img_t2,
            model=change_det,
            tile_size=tile_sz,
            gsd_meters=resolution_m,
            device=self.device,
        )
        stats = chg_output.get("stats", {})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "changed_pixels": int(stats.get("changed_pixels", 0)),
            "total_pixels": int(stats.get("total_pixels", img_t1.shape[0] * img_t1.shape[1])),
            "change_ratio": round(float(stats.get("change_ratio", 0.0)), 4),
            "changed_area_ha": round(float(stats.get("area_hectares", 0.0)), 4),
            "resolution_m": resolution_m,
            "latency_ms": round(latency_ms, 2),
        }

    def run_search(
        self, query_text: str | None = None, query_image: np.ndarray | None = None, top_k: int = 5
    ) -> dict[str, Any]:
        """Run text or image vector retrieval."""
        t0 = time.perf_counter()
        embedder = self.get_embedder()
        index = self.get_vector_index()

        query_str = query_text or "image query"
        if query_text:
            results = index.search_by_text(query_text, embedder=embedder, top_k=top_k)
        elif query_image is not None:
            results = index.search_by_image(query_image, embedder=embedder, top_k=top_k)
        else:
            results = []

        formatted_results = [
            {
                "id": r["id"],
                "score": round(float(r["score"]), 4),
                "label": r.get("payload", {}).get("label") or r.get("payload", {}).get("class_name"),
                "metadata": r.get("payload", {}),
            }
            for r in results
        ]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": query_str,
            "count": len(formatted_results),
            "results": formatted_results,
            "latency_ms": round(latency_ms, 2),
        }

    def run_chat(
        self,
        query: str,
        context_evidence: dict[str, Any] | None = None,
        provider: str = "deterministic",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Run grounded VLM Earth Assistant inference."""
        t0 = time.perf_counter()
        assistant = GroundedEarthAssistant(strictness="strict")

        # Synthesize evidence if raw context provided
        if context_evidence:
            det_data = context_evidence.get("detection") or context_evidence.get("detections")
            # If detections is list of bboxes
            if isinstance(det_data, list):
                det_dict = {
                    "boxes": [[d.get("xmin", 0), d.get("ymin", 0), d.get("xmax", 0), d.get("ymax", 0)] for d in det_data],
                    "scores": [d.get("confidence", 1.0) for d in det_data],
                    "class_names": [d.get("class_name", "object") for d in det_data],
                }
            else:
                det_dict = det_data

            evidence = EvidenceSynthesizer.synthesize_from_results(
                detection_result=det_dict,
                segmentation_result=context_evidence.get("segmentation"),
                change_result=context_evidence.get("change"),
                retrieval_result=context_evidence.get("retrieval") or context_evidence.get("retrievals"),
                geo_metadata=context_evidence.get("geo_metadata") or context_evidence.get("crs_metadata"),
            )
        else:
            evidence = EarthObservationEvidence()

        answer_text, report = assistant.ask(query=query, evidence=evidence)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query": query,
            "answer": answer_text,
            "citations": report.valid_citations,
            "grounding_score": report.grounding_score,
            "ungrounded_claims": report.invalid_citations,
            "is_grounded": report.is_grounded,
            "latency_ms": round(latency_ms, 2),
        }
