"""Grounded VLM Earth Assistant and multimodal evidence synthesis module."""

from geovision.vlm.assistant import (
    DeterministicGroundedReasoner,
    GroundedEarthAssistant,
    Message,
)
from geovision.vlm.eval_qa import (
    DEFAULT_SATELLITE_QA_DATASET,
    evaluate_grounded_assistant,
)
from geovision.vlm.evidence import (
    EarthObservationEvidence,
    EvidenceSynthesizer,
    EvidenceTagItem,
)
from geovision.vlm.verifier import (
    CitationVerifier,
    VerificationReport,
)

__all__ = [
    "GroundedEarthAssistant",
    "DeterministicGroundedReasoner",
    "Message",
    "EvidenceSynthesizer",
    "EarthObservationEvidence",
    "EvidenceTagItem",
    "CitationVerifier",
    "VerificationReport",
    "evaluate_grounded_assistant",
    "DEFAULT_SATELLITE_QA_DATASET",
]
