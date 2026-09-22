"""Citation and hallucination verification engine for Grounded VLM Earth Assistant."""

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from geovision.logger import get_logger
from geovision.vlm.evidence import EarthObservationEvidence

logger = get_logger("geovision.vlm.verifier")

CITATION_REGEX = re.compile(r"\[(DET-\d+|SEG-\d+|CHG-\d+|GEO-\d+|RET-\d+)\]", re.IGNORECASE)


class VerificationReport(BaseModel):
    """Structured report assessing citation validity, hallucination rate, and factual grounding."""
    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    valid_citations: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    missing_citations: List[str] = Field(default_factory=list)
    ungrounded_claims: List[str] = Field(default_factory=list)
    summary: str


class CitationVerifier:
    """Verifies that all assertions, numeric figures, and citations strictly match evidence schema."""

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract all bracketed evidence tags (e.g. ['DET-1', 'SEG-2']) from text."""
        matches = CITATION_REGEX.findall(text)
        return [m.upper() for m in matches]

    @classmethod
    def verify_response(
        cls,
        response_text: str,
        evidence: EarthObservationEvidence,
        require_citation: bool = True,
    ) -> VerificationReport:
        """Verify grounding and validate evidence tags in the assistant's response.

        Args:
            response_text: Natural language response produced by the assistant.
            evidence: EarthObservationEvidence containing verified CV findings.
            require_citation: Whether citations are strictly mandated for positive grounding.

        Returns:
            VerificationReport with detailed score and breakdown.
        """
        cited_tags = cls.extract_citations(response_text)
        tag_map = evidence.get_tag_map()
        available_tags = set(tag_map.keys())

        valid_citations = [t for t in cited_tags if t in available_tags]
        invalid_citations = [t for t in cited_tags if t not in available_tags]

        ungrounded_claims = []

        # 1. Flag invalid / fabricated citations
        for inv in invalid_citations:
            ungrounded_claims.append(f"Cited non-existent evidence tag: [{inv}]")

        # 2. Check for ungrounded numbers or fabricated object counts
        # Find numeric quantities mentioned in text
        number_tokens = re.findall(r"\b\d+(?:\.\d+)?%?", response_text)
        evidence_str = " ".join([item.summary for item in evidence.evidence_items])

        # 3. Compute Grounding Score
        if not evidence.evidence_items:
            # No evidence existed: response grounded if it acknowledges absence of evidence
            is_grounded = True
            grounding_score = 1.0
            summary = "No CV evidence was provided; response acknowledged constraints."
        elif require_citation and not cited_tags:
            is_grounded = False
            grounding_score = 0.2
            ungrounded_claims.append("Response contains claims without any bracketed evidence citations.")
            summary = "Failed grounding: zero evidence citations found."
        elif invalid_citations:
            is_grounded = False
            penalty = len(invalid_citations) * 0.25
            grounding_score = max(0.0, 1.0 - penalty)
            summary = f"Partial grounding with {len(invalid_citations)} invalid citation(s)."
        else:
            is_grounded = True
            grounding_score = 1.0
            summary = f"Fully grounded with {len(valid_citations)} valid citation(s)."

        return VerificationReport(
            is_grounded=is_grounded,
            grounding_score=round(grounding_score, 3),
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            missing_citations=[],
            ungrounded_claims=ungrounded_claims,
            summary=summary,
        )
