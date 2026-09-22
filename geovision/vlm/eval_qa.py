"""Satellite Visual Question Answering (VQA) benchmark evaluation dataset and grounded reasoning evaluator."""

from typing import Any

from geovision.logger import get_logger
from geovision.vlm.assistant import GroundedEarthAssistant
from geovision.vlm.evidence import EvidenceSynthesizer

logger = get_logger("geovision.vlm.eval_qa")

# Curated benchmark QA evaluation set covering all satellite analytics modalities
DEFAULT_SATELLITE_QA_DATASET: list[dict[str, Any]] = [
    {
        "id": "QA-01",
        "category": "Object Detection",
        "question": "How many airplanes and ships are detected in this scene?",
        "mock_results": {
            "detection": {
                "boxes": [[10, 10, 50, 50], [60, 60, 100, 100], [200, 200, 250, 250]],
                "class_names": ["airplane", "airplane", "ship"],
                "scores": [0.92, 0.88, 0.95],
            }
        },
        "expected_tags": ["DET-1"],
    },
    {
        "id": "QA-02",
        "category": "Land-Cover Segmentation",
        "question": "What is the dominant land-cover class and what percentage does Woodland occupy?",
        "mock_results": {
            "segmentation": {
                "distribution": {
                    "percentages": {"Background": 5.0, "Woodland": 65.5, "Water": 20.0, "Building": 9.5},
                    "area_hectares": {"Woodland": 13.1, "Water": 4.0},
                    "total_area_sq_km": 0.20,
                }
            }
        },
        "expected_tags": ["SEG-1"],
    },
    {
        "id": "QA-03",
        "category": "Bi-Temporal Change",
        "question": "How much physical ground area changed between the two time steps?",
        "mock_results": {
            "change": {
                "statistics": {
                    "change_percentage": 12.45,
                    "changed_area_hectares": 0.312,
                    "changed_pixels": 12480,
                }
            }
        },
        "expected_tags": ["CHG-1"],
    },
    {
        "id": "QA-04",
        "category": "Geospatial & CRS",
        "question": "What coordinate reference system (CRS) and spatial resolution is this raster in?",
        "mock_results": {
            "geo_metadata": {
                "crs": "EPSG:32633",
                "width": 1024,
                "height": 1024,
                "gsd_x": 0.5,
                "bounds": (500000.0, 4649760.0, 500512.0, 4650272.0),
            }
        },
        "expected_tags": ["GEO-1"],
    },
    {
        "id": "QA-05",
        "category": "Multimodal Synthesis",
        "question": "Provide a complete comprehensive summary of objects, land-cover, and spatial bounds.",
        "mock_results": {
            "detection": {
                "boxes": [[10, 10, 50, 50]],
                "class_names": ["storage_tank"],
            },
            "segmentation": {
                "distribution": {
                    "percentages": {"Industrial": 80.0, "Road": 20.0},
                }
            },
            "geo_metadata": {
                "crs": "EPSG:4326",
                "width": 512,
                "height": 512,
                "gsd_x": 1.0,
            },
        },
        "expected_tags": ["DET-1", "SEG-1", "GEO-1"],
    },
]


def evaluate_grounded_assistant(
    assistant: GroundedEarthAssistant | None = None,
    eval_dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run benchmark evaluation measuring grounding accuracy and citation adherence across satellite QA dataset.

    Args:
        assistant: GroundedEarthAssistant instance.
        eval_dataset: Optional custom list of QA evaluation items.

    Returns:
        Dictionary of evaluation metrics (grounding_accuracy, mean_score, hallucination_rate).
    """
    qa_set = eval_dataset or DEFAULT_SATELLITE_QA_DATASET
    ast = assistant or GroundedEarthAssistant(strictness="strict")

    total_questions = len(qa_set)
    grounded_count = 0
    scores = []
    category_scores: dict[str, list[float]] = {}

    for item in qa_set:
        cat = item["category"]
        q = item["question"]
        mock = item.get("mock_results", {})

        # Synthesize evidence for this benchmark sample
        evidence = EvidenceSynthesizer.synthesize_from_results(
            detection_result=mock.get("detection"),
            segmentation_result=mock.get("segmentation"),
            change_result=mock.get("change"),
            retrieval_result=mock.get("retrieval"),
            geo_metadata=mock.get("geo_metadata"),
        )

        response, report = ast.ask(q, evidence=evidence, verify=True)

        if report.is_grounded:
            grounded_count += 1

        scores.append(report.grounding_score)
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(report.grounding_score)

    grounding_acc = grounded_count / total_questions if total_questions > 0 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    hallucination_rate = 1.0 - grounding_acc

    results: dict[str, Any] = {
        "total_questions": total_questions,
        "grounding_accuracy": round(grounding_acc, 4),
        "mean_grounding_score": round(mean_score, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "per_category_scores": {
            c: round(sum(s_list) / len(s_list), 4)
            for c, s_list in category_scores.items()
        },
    }

    logger.info(
        f"Grounded VLM Benchmark -> Grounding Accuracy: {results['grounding_accuracy']:.2%} | "
        f"Mean Score: {results['mean_grounding_score']:.4f} | Hallucination Rate: {results['hallucination_rate']:.2%}"
    )

    return results
