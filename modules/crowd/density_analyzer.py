"""Overhead Pandal Crowd Density Analyzer and Gaussian Heatmap Generator."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class PandalDensityAnalyzer:
    def __init__(
        self,
        zones_config: List[Dict],
        normal_thresh: float = 1.5,
        warning_thresh: float = 3.0,
        critical_thresh: float = 4.5,
        blur_radius: int = 35,
        heatmap_alpha: float = 0.45,
    ):
        """Initialize Pandal Density Analyzer.

        Args:
            zones_config: List of zone definitions with id, name, area_sq_meters, polygon.
            normal_thresh: people / sq meter for Green status.
            warning_thresh: people / sq meter for Yellow status.
            critical_thresh: people / sq meter for Red status.
            blur_radius: Gaussian smoothing radius for density heatmap.
            heatmap_alpha: Transparency blending factor for heatmap.
        """
        self.zones = []
        for z in zones_config:
            zone_copy = dict(z)
            zone_copy["polygon_np"] = np.array(z["polygon"], dtype=np.int32)
            self.zones.append(zone_copy)

        self.normal_thresh = normal_thresh
        self.warning_thresh = warning_thresh
        self.critical_thresh = critical_thresh
        self.blur_radius = blur_radius if blur_radius % 2 != 0 else blur_radius + 1
        self.heatmap_alpha = heatmap_alpha

    def analyze_zones(
        self,
        head_centroids: List[Tuple[int, int]],
    ) -> List[Dict]:
        """Classify each detected person into pandal zones and compute density."""
        zone_stats = []

        for zone in self.zones:
            poly = zone["polygon_np"]
            area_m2 = max(1.0, float(zone.get("area_sq_meters", 50.0)))

            # Count heads inside polygon
            count = 0
            for pt in head_centroids:
                if cv2.pointPolygonTest(poly, pt, False) >= 0:
                    count += 1

            density = count / area_m2

            # Status classification
            if density < self.normal_thresh:
                status = "NORMAL"
                color_bgr = (0, 255, 0)      # Green
                level = 1
            elif density < self.warning_thresh:
                status = "WARNING"
                color_bgr = (0, 220, 255)    # Yellow / Amber
                level = 2
            elif density < self.critical_thresh:
                status = "WARNING"
                color_bgr = (0, 220, 255)    # Yellow / Amber
                level = 2
            else:
                status = "CRITICAL_SURGE"
                color_bgr = (0, 0, 255)      # Red
                level = 3

            zone_stats.append({
                "id": zone["id"],
                "name": zone["name"],
                "polygon": zone["polygon"],
                "polygon_np": poly,
                "area_m2": area_m2,
                "head_count": count,
                "density_per_m2": round(density, 2),
                "status": status,
                "status_color": color_bgr,
                "level": level,
            })

        return zone_stats

    def generate_heatmap_overlay(
        self,
        frame: np.ndarray,
        head_centroids: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Create a 2D Gaussian density heatmap overlaid onto the video frame."""
        if len(head_centroids) == 0:
            return frame.copy()

        h, w = frame.shape[:2]
        density_map = np.zeros((h, w), dtype=np.float32)

        # Place unit pulses at each head location
        for cx, cy in head_centroids:
            if 0 <= cx < w and 0 <= cy < h:
                density_map[cy, cx] += 1.0

        # Apply 2D Gaussian filter to simulate spatial density distribution
        ksize = self.blur_radius
        density_smooth = cv2.GaussianBlur(density_map, (ksize, ksize), 0)

        # Normalize to 0-255
        max_val = np.max(density_smooth)
        if max_val > 0:
            density_norm = np.uint8(255 * (density_smooth / max_val))
        else:
            density_norm = np.zeros((h, w), dtype=np.uint8)

        # Apply JET or TURBO colormap
        heatmap_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

        # Mask out areas with negligible density so original video shows clearly
        mask = density_norm > 15
        blended = frame.copy()
        blended[mask] = cv2.addWeighted(
            frame, 1.0 - self.heatmap_alpha, heatmap_color, self.heatmap_alpha, 0
        )[mask]

        return blended
