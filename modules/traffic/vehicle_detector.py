"""Vehicle Detector using YOLOv8 optimized for CPU inference with Confidence Scoring."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.35,
        target_classes: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """Initialize the YOLO vehicle detector.

        Args:
            model_name: YOLO model file (defaults to lightweight yolov8n.pt).
            confidence: Detection confidence threshold.
            target_classes: List of COCO class IDs (2: car, 3: motorcycle, 5: bus, 7: truck).
            class_names: Mapping of class ID to human-readable label.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.target_classes = target_classes or [2, 3, 5, 7]
        self.class_names = class_names or {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck",
        }

    def detect(
        self,
        frame: np.ndarray,
        road_roi: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[int]], List[str], List[float], List[List[int]], List[str], List[float]]:
        """Run vehicle detection on a frame.

        Args:
            frame: BGR video frame.
            road_roi: Optional polygon (array of (x, y) coordinates) defining road area.

        Returns:
            all_boxes: Vehicle detections from the monitored image region.
            all_labels: Labels for detections in the monitored image region.
            all_confs: Confidences for detections in the monitored image region.
            roi_boxes: Vehicle bounding boxes whose center lies inside road_roi.
            roi_labels: Labels for vehicles inside the road ROI.
            roi_confs: Confidences for vehicles inside the road ROI.
        """
        inference_frame = frame
        offset_x = offset_y = 0
        if road_roi is not None and len(road_roi) >= 3:
            road_roi = np.asarray(road_roi, dtype=np.int32)
            offset_x, offset_y, roi_width, roi_height = cv2.boundingRect(road_roi)
            frame_height, frame_width = frame.shape[:2]
            x2 = min(frame_width, offset_x + roi_width)
            y2 = min(frame_height, offset_y + roi_height)
            inference_frame = frame[offset_y:y2, offset_x:x2]
            if inference_frame.size == 0:
                raise ValueError("Road ROI does not intersect the video frame")

            local_roi = np.asarray(road_roi, dtype=np.int32) - np.array([offset_x, offset_y])
            roi_mask = np.zeros(inference_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [local_roi], 255)
            inference_frame = cv2.bitwise_and(inference_frame, inference_frame, mask=roi_mask)

        results = self.model.predict(
            source=inference_frame,
            conf=self.confidence,
            classes=self.target_classes,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        all_boxes: List[List[int]] = []
        all_labels: List[str] = []
        all_confs: List[float] = []

        roi_boxes: List[List[int]] = []
        roi_labels: List[str] = []
        roi_confs: List[float] = []

        if not results or len(results) == 0:
            return all_boxes, all_labels, all_confs, roi_boxes, roi_labels, roi_confs

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return all_boxes, all_labels, all_confs, roi_boxes, roi_labels, roi_confs

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y

            label = self.class_names.get(cls_id, f"Veh_{cls_id}")
            all_boxes.append([x1, y1, x2, y2])
            all_labels.append(label)
            all_confs.append(conf)

            # Check if vehicle centroid or base is inside the Road ROI
            if road_roi is not None and len(road_roi) >= 3:
                cx = (x1 + x2) // 2
                cy = int(0.8 * y2 + 0.2 * y1)  # Base on road surface
                is_inside = cv2.pointPolygonTest(road_roi, (cx, cy), False) >= 0
                if is_inside:
                    roi_boxes.append([x1, y1, x2, y2])
                    roi_labels.append(label)
                    roi_confs.append(conf)
            else:
                roi_boxes.append([x1, y1, x2, y2])
                roi_labels.append(label)
                roi_confs.append(conf)

        return all_boxes, all_labels, all_confs, roi_boxes, roi_labels, roi_confs

    def get_confidence_stats(self, confidences: List[float], labels: List[str]) -> Dict:
        """Compute statistical breakdown of AI detection confidences."""
        if not confidences:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "by_class": {}}

        by_class: Dict[str, List[float]] = {}
        for c, lbl in zip(confidences, labels):
            by_class.setdefault(lbl, []).append(c)

        class_means = {k: round(float(np.mean(v)) * 100.0, 1) for k, v in by_class.items()}
        return {
            "mean": round(float(np.mean(confidences)) * 100.0, 1),
            "min": round(float(np.min(confidences)) * 100.0, 1),
            "max": round(float(np.max(confidences)) * 100.0, 1),
            "by_class": class_means,
        }
