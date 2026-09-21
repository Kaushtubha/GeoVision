"""CLI script to run georeferenced object detection on satellite images / GeoTIFFs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.detection.model import YOLODetector
from geovision.logger import get_logger
from geovision.visualization import plot_detection_boxes

logger = get_logger("geovision.scripts.infer_detection")


def parse_args():
    parser = argparse.ArgumentParser(description="Run georeferenced object detection on satellite imagery.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image or GeoTIFF (.tif, .png, .jpg)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--output-img", type=str, default=None, help="Path to save visualization overlay image")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")

    detector = YOLODetector(model_path=args.weights, device=args.device)

    logger.info("=" * 65)
    logger.info(f"Running GeoVision Detection Inference on: {img_path.name}")
    logger.info("=" * 65)

    is_geotiff = img_path.suffix.lower() in [".tif", ".tiff"]

    if is_geotiff:
        results = detector.predict_georeferenced(
            raster_path=img_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        logger.info(f"Source CRS: {results['crs']}")
        logger.info(f"Total Detections: {results['total_detections']}")
        for d in results["detections"]:
            b = d["pixel_bbox"]
            gb = d["geo_bounds"]
            logger.info(
                f"  • {d['class_name']:<18} (Conf: {d['confidence']:.2f}) | "
                f"Pixel: [{b[0]}, {b[1]}, {b[2]}, {b[3]}] | "
                f"Geo: [Lon: {gb['min_lon']}..{gb['max_lon']}, Lat: {gb['min_lat']}..{gb['max_lat']}]"
            )
    else:
        results = detector.predict(
            image=img_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        logger.info(f"Total Detections: {len(results['boxes'])}")
        for b, s, name in zip(results["boxes"], results["scores"], results["class_names"]):
            logger.info(f"  • {name:<18} (Conf: {s:.2f}) | Pixel BBox: {b}")

    # Save JSON
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved structured detection JSON to: {out_json_path}")

    # Save Visualization Overlay
    if args.output_img:
        boxes = results.get("boxes", [d["pixel_bbox"] for d in results.get("detections", [])])
        labels = results.get("labels", [d["class_id"] for d in results.get("detections", [])])
        scores = results.get("scores", [d["confidence"] for d in results.get("detections", [])])
        plot_detection_boxes(
            image=img_path,
            bboxes=boxes,
            class_labels=labels,
            scores=scores,
            output_path=args.output_img,
            bbox_format="xyxy",
        )
        logger.info(f"Saved detection overlay to: {args.output_img}")


if __name__ == "__main__":
    main()
