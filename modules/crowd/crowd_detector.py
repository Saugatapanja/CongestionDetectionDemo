"""Overhead Crowd & Head Detector for Festival Pandals.

Optimized for top-down CCTV/drone perspectives where heads and shoulders
form circular patterns without standard perspective body occlusions.
Uses SCUT-HEAD / CrowdHuman-trained YOLO head detectors for precise head localization.
"""

from typing import List, Optional, Tuple
import os
import cv2
import numpy as np
from ultralytics import YOLO


class OverheadCrowdDetector:
    def __init__(
        self,
        model_name: str = "yolov8n_head.pt",
        confidence: float = 0.25,
        imgsz: int = 640,
    ):
        """Initialize overhead crowd & head detector.

        Args:
            model_name: YOLO detection model. Defaults to 'yolov8n_head.pt' (SCUT-HEAD trained).
            confidence: Detection confidence for heads.
            imgsz: Inference resolution.
        """
        if not os.path.exists(model_name):
            if os.path.exists("yolov8n_head.pt"):
                model_name = "yolov8n_head.pt"
            elif os.path.exists("yolo11n.pt"):
                model_name = "yolo11n.pt"
            else:
                model_name = "yolov8n.pt"

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = max(320, int(imgsz))

        # Check if the model has a dedicated 'head' class (e.g. SCUT-HEAD / CrowdHuman)
        self.is_head_model = any("head" in str(name).lower() for name in self.model.names.values())
        if self.is_head_model:
            self.target_classes = [
                cls_id for cls_id, name in self.model.names.items()
                if "head" in str(name).lower()
            ]
        else:
            self.target_classes = [0]  # COCO class 0 is person

    def detect_heads(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[List[int]], List[float]]:
        """Detect human heads directly from overhead perspective.

        Returns:
            head_centroids: List of (cx, cy) pixel coordinates representing actual head locations.
            boxes: List of [x1, y1, x2, y2] bounding boxes around the heads.
            confidences: List of detection confidence scores.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.target_classes,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False,
        )

        head_centroids: List[Tuple[int, int]] = []
        boxes_out: List[List[int]] = []
        confs_out: List[float] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if self.is_head_model:
                    # Direct head detector: center of the head bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                else:
                    # Fallback for full-body models: upper-body / head position
                    cx = (x1 + x2) // 2
                    cy = int(y1 + 0.25 * (y2 - y1))

                head_centroids.append((cx, cy))
                boxes_out.append([x1, y1, x2, y2])
                confs_out.append(conf)

        return head_centroids, boxes_out, confs_out
