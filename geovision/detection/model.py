"""YOLO Object Detector wrapper with georeferenced bounding box inference."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geovision.constants import NWPU_VHR10_CLASSES
from geovision.geo.crs import CoordinateTransformer, create_geojson_collection
from geovision.geo.raster import GeoRasterReader
from geovision.logger import get_logger

logger = get_logger("geovision.detection.model")

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


class YOLODetector:
    """Ultralytics YOLO Aerial Remote Sensing Detector."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or NWPU_VHR10_CLASSES
        self.model_path = str(model_path) if model_path else "yolov8n.pt"

        if _HAS_ULTRALYTICS:
            logger.info(f"Loading YOLO model from '{self.model_path}' on device '{self.device}'...")
            self.model = YOLO(self.model_path)
        else:
            logger.warning("Ultralytics not installed. Operating in stub prediction mode.")
            self.model = None

    def predict(
        self,
        image: np.ndarray | Image.Image | str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        image_size: int = 512,
    ) -> dict[str, Any]:
        """Perform object detection on an image.

        Returns:
            Dictionary containing:
            - 'boxes': list of [x1, y1, x2, y2] in pixel coordinates
            - 'scores': list of confidence scores
            - 'labels': list of integer class IDs
            - 'class_names': list of class name strings
            - 'image_shape': (H, W, C)
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                img_arr = np.array(im.convert("RGB"))
        elif isinstance(image, Image.Image):
            img_arr = np.array(image.convert("RGB"))
        else:
            img_arr = image

        h, w = img_arr.shape[:2]

        if not _HAS_ULTRALYTICS or self.model is None:
            # Deterministic fallback / stub mode for testing without GPU/ultralytics
            return {
                "boxes": [],
                "scores": [],
                "labels": [],
                "class_names": [],
                "image_shape": (h, w, 3),
            }

        results = self.model.predict(
            source=img_arr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=image_size,
            device=self.device,
            verbose=False,
        )

        boxes_list = []
        scores_list = []
        labels_list = []
        class_names_list = []

        if results and len(results) > 0:
            boxes_obj = results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, classes):
                    # Filter within class count
                    cls_id_clamped = int(cls_id) % self.num_classes
                    boxes_list.append([round(float(c), 2) for c in box])
                    scores_list.append(round(float(conf), 4))
                    labels_list.append(cls_id_clamped)
                    name = (
                        self.class_names[cls_id_clamped]
                        if cls_id_clamped < len(self.class_names)
                        else f"class_{cls_id_clamped}"
                    )
                    class_names_list.append(name)

        return {
            "boxes": boxes_list,
            "scores": scores_list,
            "labels": labels_list,
            "class_names": class_names_list,
            "image_shape": (h, w, 3),
            "num_detections": len(boxes_list),
        }

    def predict_georeferenced(
        self,
        raster_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        image_size: int = 512,
    ) -> dict[str, Any]:
        """Perform object detection on a satellite GeoTIFF and reproject detections to geodetic WGS84 coordinates.

        Returns:
            Structured dictionary containing detections with both pixel and lat/lon bounds + GeoJSON FeatureCollection.
        """
        reader = GeoRasterReader(raster_path)
        img_arr = reader.read_all()
        meta = reader.meta
        transformer = CoordinateTransformer(meta.crs)

        raw_preds = self.predict(
            image=img_arr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )

        geo_detections = []
        geojson_features = []

        for box, score, label, class_name in zip(
            raw_preds["boxes"],
            raw_preds["scores"],
            raw_preds["labels"],
            raw_preds["class_names"],
        ):
            x1, y1, x2, y2 = box

            # Reproject corners to WGS84 (Lon, Lat)
            lon_min, lat_max = transformer.pixel_to_latlon(x1, y1, meta.transform)
            lon_max, lat_min = transformer.pixel_to_latlon(x2, y2, meta.transform)

            props = {
                "class_id": label,
                "class_name": class_name,
                "confidence": score,
                "pixel_bbox": [x1, y1, x2, y2],
            }

            feature = transformer.bbox_pixel_to_geojson(
                xmin=x1, ymin=y1, xmax=x2, ymax=y2,
                transform=meta.transform,
                properties=props,
            )
            geojson_features.append(feature)

            geo_detections.append({
                "class_id": label,
                "class_name": class_name,
                "confidence": score,
                "pixel_bbox": [x1, y1, x2, y2],
                "geo_bounds": {
                    "min_lon": round(lon_min, 6),
                    "min_lat": round(lat_min, 6),
                    "max_lon": round(lon_max, 6),
                    "max_lat": round(lat_max, 6),
                },
            })

        return {
            "source_raster": str(raster_path),
            "crs": meta.crs,
            "dimensions": {"width": meta.width, "height": meta.height},
            "gsd_meters": (meta.gsd_x, meta.gsd_y),
            "total_detections": len(geo_detections),
            "detections": geo_detections,
            "geojson": create_geojson_collection(geojson_features),
        }
