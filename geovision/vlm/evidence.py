"""Structured multimodal evidence synthesis for grounded Earth Observation reasoning."""

from typing import Any

from pydantic import BaseModel, Field

from geovision.logger import get_logger

logger = get_logger("geovision.vlm.evidence")


class EvidenceTagItem(BaseModel):
    """Base schema for an atomic verified evidence item with citation tag."""
    tag: str
    category: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class EarthObservationEvidence(BaseModel):
    """Comprehensive container for structured evidence extracted across computer vision modules."""
    evidence_items: list[EvidenceTagItem] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    def get_tag_map(self) -> dict[str, EvidenceTagItem]:
        """Map tag identifier (e.g. 'DET-1', 'SEG-2') to its evidence object."""
        return {item.tag: item for item in self.evidence_items}

    def to_markdown_context(self) -> str:
        """Render evidence into structured Markdown for VLM prompt injection."""
        lines = [
            "### [VERIFIED SATELLITE COMPUTER VISION EVIDENCE]",
            "Below is the structured, machine-verified evidence for this Earth Observation scene.",
            "You MUST base all quantitative answers and assertions strictly on these evidence tags.\n",
        ]

        if not self.evidence_items:
            lines.append("*No computer vision evidence has been extracted for this scene.*")
            return "\n".join(lines)

        current_category = None
        for item in self.evidence_items:
            if item.category != current_category:
                current_category = item.category
                lines.append(f"\n#### [{current_category.upper()}]")
            lines.append(f"- **[{item.tag}]**: {item.summary}")

        return "\n".join(lines)


class EvidenceSynthesizer:
    """Aggregates raw outputs from Detection, Segmentation, Change, and GeoTIFF into indexed evidence."""

    @staticmethod
    def synthesize_from_results(
        detection_result: dict[str, Any] | None = None,
        segmentation_result: dict[str, Any] | None = None,
        change_result: dict[str, Any] | None = None,
        retrieval_result: list[dict[str, Any]] | None = None,
        geo_metadata: Any | None = None,
        image_paths: list[str] | None = None,
    ) -> EarthObservationEvidence:
        """Synthesize multimodal outputs into verified evidence schema with unique citation tags."""
        items: list[EvidenceTagItem] = []
        raw: dict[str, Any] = {}

        # 1. Geospatial & CRS Metadata
        if geo_metadata is not None:
            raw["geo"] = getattr(geo_metadata, "to_dict", lambda: dict(geo_metadata))()
            crs = getattr(geo_metadata, "crs", "Unknown")
            w = getattr(geo_metadata, "width", None)
            h = getattr(geo_metadata, "height", None)
            gsd_x = getattr(geo_metadata, "gsd_x", None)
            bounds = getattr(geo_metadata, "bounds", None)

            summary = f"Raster dimensions: {w}x{h} px | CRS: {crs}"
            if gsd_x is not None:
                summary += f" | GSD: {gsd_x:.2f} m/px"
            if bounds is not None:
                summary += f" | Bounds: {bounds}"

            items.append(
                EvidenceTagItem(
                    tag="GEO-1",
                    category="Geospatial",
                    summary=summary,
                    data=raw["geo"],
                )
            )

        # 2. Object Detection Evidence
        if detection_result is not None:
            raw["detection"] = detection_result
            boxes = detection_result.get("boxes", [])
            class_names = detection_result.get("class_names", [])

            # Count objects per class
            from collections import Counter
            counts = Counter(class_names)
            total_objs = len(boxes)

            summary = f"Detected {total_objs} total objects across {len(counts)} categories: "
            summary += ", ".join([f"{count} {cls_name}" for cls_name, count in counts.items()]) if counts else "None"

            items.append(
                EvidenceTagItem(
                    tag="DET-1",
                    category="Object Detection",
                    summary=summary,
                    data={
                        "total_objects": total_objs,
                        "class_counts": dict(counts),
                        "boxes_count": len(boxes),
                    },
                )
            )

            # Individual class tag summaries if detected
            det_idx = 2
            for cls_name, count in counts.items():
                items.append(
                    EvidenceTagItem(
                        tag=f"DET-{det_idx}",
                        category="Object Detection",
                        summary=f"{count}x {cls_name} identified with high confidence.",
                        data={"class_name": cls_name, "count": count},
                    )
                )
                det_idx += 1

        # 3. Land-Cover Semantic Segmentation Evidence
        if segmentation_result is not None:
            raw["segmentation"] = segmentation_result
            dist = segmentation_result.get("distribution", {})
            percentages = dist.get("percentages", {})
            hectares = dist.get("area_hectares", {})
            total_area_km2 = dist.get("total_area_sq_km")

            seg_summary = "Land-Cover distribution: "
            seg_summary += ", ".join([f"{cls_name}: {pct}%" for cls_name, pct in percentages.items() if pct > 0])

            items.append(
                EvidenceTagItem(
                    tag="SEG-1",
                    category="Land-Cover Segmentation",
                    summary=seg_summary,
                    data={"percentages": percentages, "total_area_km2": total_area_km2},
                )
            )

            # Detail tags for dominant land-cover classes
            seg_idx = 2
            for cls_name, pct in percentages.items():
                if pct > 0:
                    ha = hectares.get(cls_name, 0.0)
                    items.append(
                        EvidenceTagItem(
                            tag=f"SEG-{seg_idx}",
                            category="Land-Cover Segmentation",
                            summary=f"{cls_name} covers {pct:.2f}% of total scene area ({ha:.2f} hectares).",
                            data={"class_name": cls_name, "percentage": pct, "hectares": ha},
                        )
                    )
                    seg_idx += 1

        # 4. Bi-Temporal Change Detection Evidence
        if change_result is not None:
            raw["change"] = change_result
            stats = change_result.get("statistics", {})
            chg_pct = stats.get("change_percentage", 0.0)
            chg_ha = stats.get("changed_area_hectares", 0.0)
            chg_px = stats.get("changed_pixels", 0)

            items.append(
                EvidenceTagItem(
                    tag="CHG-1",
                    category="Bi-Temporal Change",
                    summary=(
                        f"Detected {chg_pct:.2f}% bi-temporal physical change "
                        f"({chg_ha:.4f} hectares / {chg_px:,} altered pixels between T1 and T2)."
                    ),
                    data=stats,
                )
            )

        # 5. Scene Retrieval / Vector Matches Evidence
        if retrieval_result is not None and len(retrieval_result) > 0:
            raw["retrieval"] = retrieval_result
            top_hit = retrieval_result[0]
            top_class = top_hit.get("payload", {}).get("class_name", "Unknown")
            top_sim = top_hit.get("score", 0.0)

            items.append(
                EvidenceTagItem(
                    tag="RET-1",
                    category="Scene Retrieval",
                    summary=f"Top semantic scene match: '{top_class}' (Cosine Similarity: {top_sim:.4f}).",
                    data={"top_hit": top_hit, "total_hits": len(retrieval_result)},
                )
            )

        return EarthObservationEvidence(
            evidence_items=items,
            image_paths=image_paths or [],
            raw_payload=raw,
        )
