"""Traffic Congestion Meter and Level of Service (LOS) Calculator with Feature Extraction."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from core.tracker import TrackedObject


class CongestionMeter:
    def __init__(
        self,
        road_roi: np.ndarray,
        free_flow_thresh: float = 30.0,
        moderate_thresh: float = 65.0,
        congested_thresh: float = 80.0,
        stationary_sec_thresh: float = 5.0,
        car_threshold: int = 10,
        fps: float = 25.0,
    ):
        """Initialize Congestion Meter.

        Args:
            road_roi: Polygon coordinates of the monitored road segment.
            free_flow_thresh: Occupancy % below which traffic is free-flowing.
            moderate_thresh: Occupancy % threshold for moderate traffic.
            congested_thresh: Occupancy % threshold for severe congestion.
            stationary_sec_thresh: Seconds vehicles stay still before triggering gridlock.
            car_threshold: Number of cars above which police diversion is triggered.
            fps: Video frames per second for converting frames to time.
        """
        self.road_roi = np.array(road_roi, dtype=np.int32)
        self.free_flow_thresh = free_flow_thresh
        self.moderate_thresh = moderate_thresh
        self.congested_thresh = congested_thresh
        self.car_threshold = car_threshold
        self.stationary_frame_thresh = int(stationary_sec_thresh * fps)
        self.fps = max(1.0, fps)

        # Calculate polygon area of road ROI
        self.roi_area = max(1.0, cv2.contourArea(self.road_roi))
        self.smoothed_occupancy = 0.0

    def calculate_occupancy(
        self,
        vehicle_boxes: List[List[int]],
        frame_shape: Tuple[int, int],
    ) -> float:
        """Calculate exact pixel occupancy percentage of vehicles within the road ROI."""
        if not vehicle_boxes or self.roi_area <= 0:
            return 0.0

        h, w = frame_shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [self.road_roi], 255)

        vehicle_mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in vehicle_boxes:
            cv2.rectangle(vehicle_mask, (max(0, x1), max(0, y1)), (min(w, x2), min(h, y2)), 255, -1)

        overlap = cv2.bitwise_and(roi_mask, vehicle_mask)
        overlap_pixels = cv2.countNonZero(overlap)

        occupancy = (overlap_pixels / self.roi_area) * 100.0
        return min(100.0, max(0.0, occupancy))

    def evaluate(
        self,
        vehicle_boxes: List[List[int]],
        vehicle_labels: List[str],
        tracked_objects: Dict[int, TrackedObject],
        frame_shape: Tuple[int, int],
        confidences: Optional[List[float]] = None,
    ) -> Dict:
        """Evaluate traffic state with confidence and kinematic feature extraction.

        Returns comprehensive metric dictionary.
        """
        raw_occupancy = self.calculate_occupancy(vehicle_boxes, frame_shape)
        self.smoothed_occupancy = 0.8 * self.smoothed_occupancy + 0.2 * raw_occupancy

        # Count vehicle types (canonical capitalized)
        counts: Dict[str, int] = {}
        for lbl in vehicle_labels:
            canonical_lbl = lbl.capitalize()
            counts[canonical_lbl] = counts.get(canonical_lbl, 0) + 1

        # Check vehicle count threshold (> 10 cars triggers police diversion)
        num_cars = counts.get("Car", 0)
        car_threshold_exceeded = (num_cars > self.car_threshold)

        # Check stationary vehicles and speed statistics from tracker
        stationary_count = 0
        speeds_kmh: List[float] = []
        headings: List[str] = []
        active_tracked = 0

        for obj in tracked_objects.values():
            cx, cy = obj.centroid
            if cv2.pointPolygonTest(self.road_roi, (cx, cy), False) >= 0:
                active_tracked += 1
                speeds_kmh.append(obj.speed_kmh)
                headings.append(obj.heading_cardinal)
                if obj.stationary_frames >= self.stationary_frame_thresh:
                    stationary_count += 1

        avg_speed = round(float(np.mean(speeds_kmh)), 1) if speeds_kmh else 0.0
        max_speed = round(float(np.max(speeds_kmh)), 1) if speeds_kmh else 0.0

        # Mean detection confidence
        if confidences and len(confidences) > 0:
            avg_conf = round(float(np.mean(confidences)) * 100.0, 1)
        else:
            avg_conf = 88.5  # default baseline

        # Congestion Index (0.0 to 1.0)
        occ = self.smoothed_occupancy
        occ_term = occ / 100.0
        stoppage_term = (stationary_count / max(1, active_tracked))
        velocity_deficit = max(0.0, 1.0 - (avg_speed / 45.0))
        congestion_index = round(0.50 * occ_term + 0.30 * stoppage_term + 0.20 * velocity_deficit, 2)
        congestion_index = min(1.0, max(0.0, congestion_index))

        # Flow Rate estimate (vehicles per minute)
        flow_rate_vpm = round(active_tracked * max(1.0, avg_speed / 15.0), 1)

        # Determine Level of Service (LOS) & Status
        if occ < self.free_flow_thresh and stationary_count == 0 and not car_threshold_exceeded:
            los = "A"
            status = "FREE_FLOW"
            color_bgr = (0, 255, 0)  # Green
            divert = False
        elif occ < self.moderate_thresh and not car_threshold_exceeded:
            los = "C"
            status = "MODERATE"
            color_bgr = (0, 200, 255)  # Amber / Yellow
            divert = False
        elif occ < self.congested_thresh and stationary_count < 3 and not car_threshold_exceeded:
            los = "E"
            status = "CONGESTED"
            color_bgr = (0, 100, 255)  # Orange
            divert = True
        else:
            los = "F" if (occ >= self.congested_thresh or stationary_count >= 3) else "E"
            status = "GRIDLOCK" if los == "F" else "CONGESTED"
            color_bgr = (0, 0, 255) if los == "F" else (0, 100, 255)
            divert = True

        # If car count > 10, enforce police diversion and specialized police directive message
        if car_threshold_exceeded:
            divert = True
            los = "E" if los in ["A", "C"] else los
            status = f"CONGESTED (>{self.car_threshold} CARS)" if status != "GRIDLOCK" else status
            color_bgr = (0, 0, 255) if status == "GRIDLOCK" else (0, 100, 255)
            police_msg = (
                f"KOLKATA POLICE ALERT: Critical vehicle volume detected ({num_cars} cars > {self.car_threshold} threshold). "
                f"Immediate diversion dispatch recommended via alternate corridors to prevent corridor gridlock."
            )
        elif divert:
            police_msg = "KOLKATA POLICE ALERT: Heavy congestion detected. Divert vehicles via alternate secondary corridors."
        else:
            police_msg = "POLICE DIRECTIVE: Traffic flowing within normal capacity. Maintain standard signal timings."

        return {
            "vehicle_count": len(vehicle_boxes),
            "vehicle_breakdown": counts,
            "occupancy_pct": round(occ, 1),
            "level_of_service": los,
            "status": status,
            "status_color": color_bgr,
            "stationary_vehicles": stationary_count,
            "avg_velocity_px": round(avg_speed / 3.6, 1),
            "avg_speed_kmh": avg_speed,
            "max_speed_kmh": max_speed,
            "flow_rate_vpm": flow_rate_vpm,
            "congestion_index": congestion_index,
            "avg_confidence_pct": avg_conf,
            "is_divert_recommended": divert,
            "car_threshold_exceeded": car_threshold_exceeded,
            "car_count": num_cars,
            "police_message": police_msg,
            "recommendation": police_msg,
        }
