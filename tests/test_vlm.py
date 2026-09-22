"""Comprehensive unit tests for Grounded VLM Earth Assistant and Citation Verifier."""

import pytest

from geovision.vlm.assistant import (
    DeterministicGroundedReasoner,
    GroundedEarthAssistant,
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
from geovision.vlm.verifier import CitationVerifier, VerificationReport


def test_evidence_synthesizer_and_markdown():
    """Test EvidenceSynthesizer schema building and markdown conversion."""
    det_res = {
        "boxes": [[10, 10, 50, 50], [60, 60, 100, 100]],
        "class_names": ["airplane", "airplane"],
    }
    seg_res = {
        "distribution": {
            "percentages": {"Woodland": 70.0, "Water": 30.0},
            "area_hectares": {"Woodland": 14.0, "Water": 6.0},
        }
    }
    geo_meta = {
        "crs": "EPSG:4326",
        "width": 512,
        "height": 512,
        "gsd_x": 0.5,
    }

    evidence = EvidenceSynthesizer.synthesize_from_results(
        detection_result=det_res,
        segmentation_result=seg_res,
        geo_metadata=geo_meta,
    )

    tag_map = evidence.get_tag_map()
    assert "GEO-1" in tag_map
    assert "DET-1" in tag_map
    assert "SEG-1" in tag_map

    md_text = evidence.to_markdown_context()
    assert "[GEO-1]" in md_text
    assert "[DET-1]" in md_text
    assert "[SEG-1]" in md_text
    assert "Woodland" in md_text


def test_citation_verifier_precision_and_hallucination_detection():
    """Test CitationVerifier on grounded vs fabricated responses."""
    evidence = EvidenceSynthesizer.synthesize_from_results(
        detection_result={
            "boxes": [[10, 10, 50, 50]],
            "class_names": ["ship"],
        },
        segmentation_result={
            "distribution": {"percentages": {"Water": 100.0}},
        },
    )

    # 1. Valid Grounded response
    grounded_text = "The scene contains 1 ship [DET-1] in 100.0% water [SEG-1]."
    rep_valid = CitationVerifier.verify_response(grounded_text, evidence)
    assert rep_valid.is_grounded is True
    assert rep_valid.grounding_score == 1.0
    assert "DET-1" in rep_valid.valid_citations
    assert "SEG-1" in rep_valid.valid_citations
    assert len(rep_valid.invalid_citations) == 0

    # 2. Fabricated citation (e.g. DET-99, CHG-5)
    hallucinated_text = "I observed 10 buildings [DET-99] with major urban growth [CHG-5]."
    rep_inv = CitationVerifier.verify_response(hallucinated_text, evidence)
    assert rep_inv.is_grounded is False
    assert "DET-99" in rep_inv.invalid_citations
    assert "CHG-5" in rep_inv.invalid_citations
    assert rep_inv.grounding_score < 1.0

    # 3. Uncited statement
    uncited_text = "There is a ship in the water."
    rep_uncited = CitationVerifier.verify_response(uncited_text, evidence, require_citation=True)
    assert rep_uncited.is_grounded is False
    assert rep_uncited.grounding_score < 0.5


def test_grounded_earth_assistant_multiturn():
    """Test GroundedEarthAssistant reasoning across multiple domains."""
    assistant = GroundedEarthAssistant(strictness="strict")

    evidence = EvidenceSynthesizer.synthesize_from_results(
        detection_result={
            "boxes": [[10, 10, 50, 50], [60, 60, 100, 100]],
            "class_names": ["airplane", "airplane"],
        },
        segmentation_result={
            "distribution": {
                "percentages": {"Woodland": 60.0, "Road": 40.0},
                "area_hectares": {"Woodland": 12.0, "Road": 8.0},
            }
        },
        change_result={
            "statistics": {
                "change_percentage": 5.5,
                "changed_area_hectares": 0.11,
                "changed_pixels": 4400,
            }
        },
        geo_metadata={
            "crs": "EPSG:32633",
            "width": 1024,
            "height": 1024,
        },
    )
    assistant.set_evidence(evidence)

    # Turn 1: Detection query
    resp1, rep1 = assistant.ask("What objects were detected?")
    assert "airplane" in resp1.lower()
    assert "[DET-1]" in resp1
    assert rep1.is_grounded is True

    # Turn 2: Segmentation query
    resp2, rep2 = assistant.ask("How much area does Woodland cover?")
    assert "woodland" in resp2.lower()
    assert "[SEG-1]" in resp2
    assert rep2.is_grounded is True

    # Turn 3: Change query
    resp3, rep3 = assistant.ask("What bi-temporal changes occurred?")
    assert "change" in resp3.lower()
    assert "[CHG-1]" in resp3
    assert rep3.is_grounded is True

    assert len(assistant.history) == 6


def test_vlm_benchmark_evaluation_suite():
    """Test evaluate_grounded_assistant on the full benchmark QA dataset."""
    assistant = GroundedEarthAssistant(strictness="strict")
    results = evaluate_grounded_assistant(assistant=assistant, eval_dataset=DEFAULT_SATELLITE_QA_DATASET)

    assert results["total_questions"] == len(DEFAULT_SATELLITE_QA_DATASET)
    assert results["grounding_accuracy"] == 1.0
    assert results["mean_grounding_score"] == 1.0
    assert results["hallucination_rate"] == 0.0
