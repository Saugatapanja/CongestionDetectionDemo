"""Overhead Crowd & Head Detector for Festival Pandals.

Optimized for top-down CCTV/drone perspectives where heads and shoulders
form circular patterns without standard perspective body occlusions.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


class OverheadCrowdDetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        imgsz: int = 960,
    ):
        """Initialize overhead crowd detector.

        Args:
            model_name: YOLO detection model.  Use a larger model for wide,
                oblique pandal cameras where devotees occupy few pixels.
            confidence: Detection confidence for persons/heads.
            imgsz: Inference resolution.  A higher value preserves small
                people in distant queues.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = max(320, int(imgsz))
        self.person_class = 0  # COCO class 0 is person

    def detect_heads(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[List[int]], List[float]]:
        """Detect human heads / persons from overhead perspective.

        Returns:
            head_centroids: List of (cx, cy) pixel coordinates representing person/head locations.
            boxes: List of [x1, y1, x2, y2] bounding boxes.
            confidences: List of detection confidence scores.
        """
        # Run YOLO person detector on CPU
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[self.person_class],
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
                # Use upper-body position rather than the box centre.  The
                # supplied pandal camera is oblique, so its box centre falls
                # behind the actual head and can put a devotee in the wrong
                # entry/exit zone.
                cx = (x1 + x2) // 2
                cy = int(y1 + 0.25 * (y2 - y1))

                head_centroids.append((cx, cy))
                boxes_out.append([x1, y1, x2, y2])
                confs_out.append(conf)

        return head_centroids, boxes_out, confs_out
