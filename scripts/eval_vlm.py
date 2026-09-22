"""CLI script to benchmark Grounded VLM Earth Assistant on Satellite VQA."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.logger import get_logger
from geovision.vlm.assistant import GroundedEarthAssistant
from geovision.vlm.eval_qa import evaluate_grounded_assistant

logger = get_logger("geovision.scripts.eval_vlm")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Grounded VLM Earth Assistant on Satellite QA.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/runs/vlm/eval_results.json",
        help="Path to save VLM benchmark evaluation results JSON.",
    )
    parser.add_argument(
        "--strictness",
        type=str,
        default="strict",
        choices=["strict", "lenient"],
        help="Grounding strictness level.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Grounded VLM Earth Assistant benchmark...")

    assistant = GroundedEarthAssistant(strictness=args.strictness)
    results = evaluate_grounded_assistant(assistant=assistant)

    logger.info("=== Grounded VLM Earth Assistant Benchmark Summary ===")
    logger.info(f"Total Evaluated Questions: {results['total_questions']}")
    logger.info(f"Grounding Accuracy      : {results['grounding_accuracy']:.2%}")
    logger.info(f"Mean Grounding Score    : {results['mean_grounding_score']:.4f}")
    logger.info(f"Hallucination Rate      : {results['hallucination_rate']:.2%}")
    logger.info("Per-Category Grounding Scores:")
    for cat, score in results["per_category_scores"].items():
        logger.info(f"  - {cat:25s}: {score:.4f}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved benchmark results to {out_p}")


if __name__ == "__main__":
    main()
