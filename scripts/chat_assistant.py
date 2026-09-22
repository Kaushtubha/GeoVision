"""Interactive CLI Grounded Earth Observation Assistant."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.logger import get_logger
from geovision.vlm.assistant import GroundedEarthAssistant
from geovision.vlm.evidence import EvidenceSynthesizer

logger = get_logger("geovision.scripts.chat_assistant")


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Grounded Earth Observation Assistant.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to optical satellite image or GeoTIFF.",
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Pre-event image for change detection.",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Post-event image for change detection.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single-turn question to answer. If omitted, starts interactive chat mode.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run assistant with smoke evidence payload.",
    )
    parser.add_argument(
        "--strictness",
        type=str,
        default="strict",
        choices=["strict", "lenient"],
        help="Grounding citation strictness.",
    )
    return parser.parse_args()


def build_smoke_evidence():
    """Construct a rich, multi-modal mock evidence set for quick verification."""
    return EvidenceSynthesizer.synthesize_from_results(
        detection_result={
            "boxes": [[100, 100, 150, 150], [200, 200, 260, 260], [300, 300, 400, 350]],
            "class_names": ["airplane", "airplane", "storage_tank"],
            "scores": [0.94, 0.89, 0.91],
        },
        segmentation_result={
            "distribution": {
                "percentages": {"Background": 10.0, "Woodland": 55.0, "Water": 25.0, "Road": 10.0},
                "area_hectares": {"Woodland": 14.08, "Water": 6.4, "Road": 2.56},
                "total_area_sq_km": 0.256,
            }
        },
        change_result={
            "statistics": {
                "change_percentage": 4.82,
                "changed_area_hectares": 0.1234,
                "changed_pixels": 4936,
            }
        },
        geo_metadata={
            "crs": "EPSG:32633 (UTM Zone 33N)",
            "width": 512,
            "height": 512,
            "gsd_x": 0.5,
            "bounds": (500000.0, 4649744.0, 500256.0, 4650000.0),
        },
        retrieval_result=[
            {"score": 0.895, "payload": {"class_name": "Forest"}}
        ],
    )


def main():
    args = parse_args()
    assistant = GroundedEarthAssistant(strictness=args.strictness)

    # 1. Synthesize Evidence
    if args.smoke or (args.image is None and args.image_a is None):
        logger.info("Initializing Grounded Assistant with multi-modal smoke evidence...")
        evidence = build_smoke_evidence()
    elif args.image:
        logger.info(f"Analyzing satellite scene at: {args.image}")
        # Run detection and segmentation on input image
        from geovision.detection.model import YOLODetector
        from geovision.segmentation.infer import segment_image
        try:
            det_res = YOLODetector(device="cpu").predict(args.image)
        except Exception:
            det_res = None
        try:
            seg_res = segment_image(args.image, device="cpu")
        except Exception:
            seg_res = None

        evidence = EvidenceSynthesizer.synthesize_from_results(
            detection_result=det_res,
            segmentation_result=seg_res,
            image_paths=[args.image],
        )
    elif args.image_a and args.image_b:
        logger.info(f"Analyzing bi-temporal pair: {args.image_a} vs {args.image_b}")
        from geovision.change.infer import detect_changes
        try:
            chg_res = detect_changes(args.image_a, args.image_b, device="cpu")
        except Exception:
            chg_res = None

        evidence = EvidenceSynthesizer.synthesize_from_results(
            change_result=chg_res,
            image_paths=[args.image_a, args.image_b],
        )

    assistant.set_evidence(evidence)

    logger.info("\n" + evidence.to_markdown_context())

    # 2. Single-Turn Query or Interactive Mode
    if args.query:
        logger.info(f"User Query: {args.query}")
        response, report = assistant.ask(args.query, verify=True)
        print("\n--- [Grounded Earth Assistant Response] ---")
        print(response)
        print("\n--- [Citation Verification Report] ---")
        print(f"Grounding Status: {'[PASSED]' if report.is_grounded else '[FAILED]'}")
        print(f"Grounding Score : {report.grounding_score:.2f}")
        print(f"Valid Citations : {report.valid_citations}")
        if report.invalid_citations:
            print(f"Invalid Citations: {report.invalid_citations}")
    else:
        print("\n=== GeoVision Grounded Earth Assistant CLI (Type 'exit' to quit) ===")
        while True:
            try:
                user_q = input("\n[User Question] > ").strip()
                if user_q.lower() in ("exit", "quit", "q"):
                    break
                if not user_q:
                    continue
                response, report = assistant.ask(user_q, verify=True)
                print(f"\n[GeoVision Assistant] >\n{response}")
                print(f"\n[Verification] > Score: {report.grounding_score:.2f} | Citations: {report.valid_citations}")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
