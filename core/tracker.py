"""Lightweight Euclidean Centroid Tracker with Feature Extraction.

Extracts vehicle and crowd motion features:
- Detection confidence score retention
- Calibrated speed estimation (km/h)
- Trajectory heading vector (degrees & cardinal directions)
- Dwell time & stationary duration (seconds)
- Bounding box dimensional features
"""

from collections import OrderedDict
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class TrackedObject:
    def __init__(
        self,
        object_id: int,
        centroid: Tuple[int, int],
        bbox: List[int],
        class_name: str,
        confidence: float = 0.85,
        fps: float = 25.0,
        meters_per_pixel: float = 0.05,
    ):
        self.object_id = object_id
        self.centroid = centroid
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        self.fps = max(1.0, fps)
        self.meters_per_pixel = meters_per_pixel

        self.history = [centroid]
        self.disappeared = 0
        self.stationary_frames = 0
        self.velocity_px = 0.0  # pixels per frame
        self.speed_kmh = 0.0    # estimated km/h
        self.heading_deg = 0.0  # 0 to 360 degrees
        self.heading_cardinal = "Stationary"
        self.dwell_time_sec = 0.0

        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])
        self.dimensions = {"width": w, "height": h, "area": w * h}

    @property
    def velocity(self) -> float:
        return self.velocity_px

    def update(
        self,
        new_centroid: Tuple[int, int],
        new_bbox: List[int],
        new_confidence: Optional[float] = None,
    ):
        dx = new_centroid[0] - self.centroid[0]
        dy = new_centroid[1] - self.centroid[1]
        dist = math.hypot(dx, dy)

        # Exponential moving average for velocity
        self.velocity_px = 0.7 * self.velocity_px + 0.3 * dist
        # Speed in km/h: (meters / sec) * 3.6
        instant_speed_kmh = (dist * self.meters_per_pixel * self.fps) * 3.6
        self.speed_kmh = round(0.7 * self.speed_kmh + 0.3 * instant_speed_kmh, 1)

        # Heading calculation
        if dist > 2.0:
            deg = math.degrees(math.atan2(dy, dx))
            self.heading_deg = round((deg + 360.0) % 360.0, 1)
            # Cardinal direction (screen coordinates: +x = East, +y = South)
            if 315 <= self.heading_deg or self.heading_deg < 45:
                self.heading_cardinal = "Eastbound"
            elif 45 <= self.heading_deg < 135:
                self.heading_cardinal = "Southbound"
            elif 135 <= self.heading_deg < 225:
                self.heading_cardinal = "Westbound"
            else:
                self.heading_cardinal = "Northbound"
        else:
            self.heading_cardinal = "Stationary"

        # Stationary & dwell tracking
        if dist < 3.0:
            self.stationary_frames += 1
        else:
            self.stationary_frames = max(0, self.stationary_frames - 2)

        self.dwell_time_sec = round(self.stationary_frames / self.fps, 1)

        # Confidence update
        if new_confidence is not None:
            self.confidence = 0.7 * self.confidence + 0.3 * new_confidence

        self.centroid = new_centroid
        self.bbox = new_bbox
        w = max(1, new_bbox[2] - new_bbox[0])
        h = max(1, new_bbox[3] - new_bbox[1])
        self.dimensions = {"width": w, "height": h, "area": w * h}

        self.history.append(new_centroid)
        if len(self.history) > 30:
            self.history.pop(0)
        self.disappeared = 0


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = 25,
        max_distance: float = 60.0,
        fps: float = 25.0,
        meters_per_pixel: float = 0.05,
    ):
        self.next_object_id = 0
        self.objects: Dict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.fps = fps
        self.meters_per_pixel = meters_per_pixel

    def register(
        self,
        centroid: Tuple[int, int],
        bbox: List[int],
        class_name: str,
        confidence: float = 0.85,
    ) -> int:
        self.objects[self.next_object_id] = TrackedObject(
            self.next_object_id,
            centroid,
            bbox,
            class_name,
            confidence=confidence,
            fps=self.fps,
            meters_per_pixel=self.meters_per_pixel,
        )
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]

    def update(
        self,
        rects: List[List[int]],
        class_names: List[str],
        confidences: Optional[List[float]] = None,
    ) -> Dict[int, TrackedObject]:
        """rects: List of [x1, y1, x2, y2] bounding boxes."""
        if len(rects) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        native_centroids = [
            (int(centroid[0]), int(centroid[1])) for centroid in input_centroids
        ]

        if len(self.objects) == 0:
            for i in range(len(rects)):
                conf = confidences[i] if (confidences and i < len(confidences)) else 0.85
                self.register(
                    native_centroids[i],
                    rects[i],
                    class_names[i] if i < len(class_names) else "Unknown",
                    confidence=conf,
                )
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj.centroid for obj in self.objects.values()]

            # Compute pairwise distance matrix
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                conf = confidences[col] if (confidences and col < len(confidences)) else None
                self.objects[obj_id].update(native_centroids[col], rects[col], new_confidence=conf)
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                obj_id = object_ids[row]
                self.objects[obj_id].disappeared += 1
                if self.objects[obj_id].disappeared > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                conf = confidences[col] if (confidences and col < len(confidences)) else 0.85
                self.register(
                    native_centroids[col],
                    rects[col],
                    class_names[col] if col < len(class_names) else "Unknown",
                    confidence=conf,
                )

        return self.objects
