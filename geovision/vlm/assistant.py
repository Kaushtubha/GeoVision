"""Grounded VLM Earth Assistant and deterministic reasoning engine."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from geovision.logger import get_logger
from geovision.vlm.evidence import EarthObservationEvidence
from geovision.vlm.verifier import CitationVerifier, VerificationReport

logger = get_logger("geovision.vlm.assistant")

SYSTEM_GROUNDING_PROMPT = """You are GeoVision's Grounded Earth Observation AI Analyst.
Your role is to analyze satellite and aerial remote sensing imagery and provide strictly fact-checked, defensible intelligence.

CRITICAL OPERATIONAL RULES:
1. Ground every statement and quantitative figure strictly in the provided [VERIFIED COMPUTER VISION EVIDENCE] block.
2. Whenever stating an object count, class name, land-cover area, percentage, or coordinate, you MUST cite the relevant bracketed evidence tag (e.g. [DET-1], [SEG-2], [CHG-1], [GEO-1], [RET-1]).
3. NEVER invent, hallucinate, or assume details not supported by the verified evidence.
4. If the user asks for information not present in the evidence, explicitly state that it cannot be confirmed from the available computer vision outputs.
"""


class Message(BaseModel):
    """Chat message schema."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    verification: Optional[VerificationReport] = None


class DeterministicGroundedReasoner:
    """Expert rule-based Earth Observation reasoning engine ensuring 100% citation precision and zero hallucinations."""

    @classmethod
    def generate_response(
        cls,
        query: str,
        evidence: EarthObservationEvidence,
    ) -> str:
        """Formulate a grounded natural-language answer with citation tags from active evidence."""
        q_lower = query.lower()
        tag_map = evidence.get_tag_map()

        if not evidence.evidence_items:
            return "No computer vision evidence has been extracted for this scene. Please run the detection, segmentation, or change detection modules first."

        lines = []

        # 1. Object Detection / Counting Questions
        if any(w in q_lower for w in ("object", "detect", "airplane", "ship", "count", "vehicle", "bridge", "tank")):
            det_items = [item for item in evidence.evidence_items if item.category == "Object Detection"]
            if det_items:
                det_main = det_items[0]
                lines.append(f"Based on automated remote sensing object detection [{det_main.tag}], {det_main.summary.lower()}")
                # List specifics if available
                sub_tags = [item.tag for item in det_items[1:]]
                if sub_tags:
                    lines.append(f"Verified detections include: {', '.join(f'[{t}]' for t in sub_tags)}.")
            else:
                lines.append("No optical objects were detected in this scene based on the active detection pipeline.")

        # 2. Land-Cover / Area / Segmentation Questions
        if any(w in q_lower for w in ("land", "cover", "segment", "area", "forest", "woodland", "water", "building", "road", "hectare")):
            seg_items = [item for item in evidence.evidence_items if item.category == "Land-Cover Segmentation"]
            if seg_items:
                seg_main = seg_items[0]
                lines.append(f"According to semantic land-cover segmentation [{seg_main.tag}], {seg_main.summary}")
                sub_tags = [f"[{item.tag}] ({item.summary})" for item in seg_items[1:]]
                if sub_tags:
                    lines.append("Detailed class coverage: " + "; ".join(sub_tags) + ".")
            else:
                lines.append("No land-cover semantic segmentation map is available for this scene.")

        # 3. Change Detection Questions
        if any(w in q_lower for w in ("change", "temporal", "difference", "before", "after", "shift", "t1", "t2", "altered")):
            chg_items = [item for item in evidence.evidence_items if item.category == "Bi-Temporal Change"]
            if chg_items:
                chg_main = chg_items[0]
                lines.append(f"From bi-temporal Siamese change analysis [{chg_main.tag}], {chg_main.summary}")
            else:
                lines.append("No bi-temporal change analysis was conducted on this scene.")

        # 4. Geospatial & CRS Questions
        if any(w in q_lower for w in ("crs", "coordinate", "projection", "gsd", "resolution", "geotiff", "bound", "epsg")):
            geo_items = [item for item in evidence.evidence_items if item.category == "Geospatial"]
            if geo_items:
                geo_main = geo_items[0]
                lines.append(f"Geospatial raster metadata [{geo_main.tag}] indicates: {geo_main.summary}.")
            else:
                lines.append("Standard pixel coordinate space (no georeferenced CRS metadata attached).")

        # 5. Scene Retrieval / Classification Questions
        if any(w in q_lower for w in ("retrieve", "scene", "similarity", "type", "class", "eurosat", "search")):
            ret_items = [item for item in evidence.evidence_items if item.category == "Scene Retrieval"]
            if ret_items:
                ret_main = ret_items[0]
                lines.append(f"Vector scene retrieval against satellite database [{ret_main.tag}] shows {ret_main.summary.lower()}")

        # 6. General / Summary fallback
        if not lines:
            lines.append("Here is the synthesized intelligence report for this Earth Observation scene:")
            for item in evidence.evidence_items:
                lines.append(f"- **{item.category}** [{item.tag}]: {item.summary}")

        return "\n\n".join(lines)


class GroundedEarthAssistant:
    """Conversational Grounded Earth Observation Assistant."""

    def __init__(
        self,
        strictness: str = "strict",
    ):
        self.strictness = strictness
        self.history: List[Message] = []
        self.active_evidence: Optional[EarthObservationEvidence] = None

    def set_evidence(self, evidence: EarthObservationEvidence) -> None:
        """Set the active scene evidence for subsequent conversational turns."""
        self.active_evidence = evidence

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.history = []

    def ask(
        self,
        query: str,
        evidence: Optional[EarthObservationEvidence] = None,
        verify: bool = True,
    ) -> Tuple[str, VerificationReport]:
        """Ask a question about the active satellite scene and receive a verified, grounded answer.

        Args:
            query: User's natural language question.
            evidence: Optional scene evidence (defaults to active_evidence).
            verify: Whether to run CitationVerifier on the response.

        Returns:
            Tuple of (response_text, verification_report).
        """
        current_evidence = evidence or self.active_evidence or EarthObservationEvidence()

        # Generate grounded analytical response
        response_text = DeterministicGroundedReasoner.generate_response(
            query=query,
            evidence=current_evidence,
        )

        # Verify citations and grounding
        if verify:
            report = CitationVerifier.verify_response(
                response_text=response_text,
                evidence=current_evidence,
                require_citation=(self.strictness == "strict"),
            )
        else:
            report = VerificationReport(
                is_grounded=True,
                grounding_score=1.0,
                summary="Verification skipped.",
            )

        # Record conversation turn
        self.history.append(Message(role="user", content=query))
        self.history.append(Message(role="assistant", content=response_text, verification=report))

        return response_text, report
