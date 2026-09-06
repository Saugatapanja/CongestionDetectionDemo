"""Interactive ROI Polygon Calibration Tool.

Run this tool on any video to visually click and define road boundaries or pandal zones.
Saves calibrated polygon vertices directly into config/zones.json.

Usage:
    python tools/roi_drawer.py --video path/to/video.mp4 --mode traffic
    python tools/roi_drawer.py --video path/to/video.mp4 --mode crowd
"""

import argparse
import json
import os
import cv2
import numpy as np


class ROIDrawer:
    def __init__(self, video_path: str, mode: str = "traffic", config_file: str = "config/zones.json"):
        self.video_path = video_path
        self.mode = mode
        self.config_file = config_file
        self.current_points = []
        self.all_polygons = []
        self.window_name = f"ROI Polygon Calibration [{mode.upper()}] - (L-Click: Add point, U: Undo, C: Close, S: Save, Q: Quit)"

        # Load existing config if available
        self.data = {"traffic": {}, "crowd_pandal": {"zones": []}}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    self.data = json.load(f)
            except Exception:
                pass

        # Load first frame
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        if not ret or self.frame is None:
            raise ValueError(f"Could not read frame from: {video_path}")

        # Resize for display
        self.frame = cv2.resize(self.frame, (1024, 576))
        self.clean_frame = self.frame.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self.redraw()

    def redraw(self):
        disp = self.clean_frame.copy()

        # Draw existing saved polygons
        for poly in self.all_polygons:
            pts = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(disp, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(disp, [pts], (0, 255, 0, 40))
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(disp, poly["name"], (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw points of polygon currently being drawn
        if len(self.current_points) > 0:
            for pt in self.current_points:
                cv2.circle(disp, tuple(pt), 4, (0, 0, 255), -1)

            if len(self.current_points) > 1:
                cv2.polylines(disp, [np.array(self.current_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)

        cv2.imshow(self.window_name, disp)

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.redraw()

        print(f"\n--- ROI Calibration for [{self.mode.upper()}] ---")
        print("1. Left Click to drop polygon vertex points on the video.")
        print("2. Press 'c' to finish the current polygon.")
        print("3. Press 'u' to undo the last placed point.")
        print("4. Press 's' to save polygons to config/zones.json.")
        print("5. Press 'q' to exit.\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()
                    self.redraw()
            elif key == ord("c"):
                if len(self.current_points) >= 3:
                    name = input(f"Enter a name for this {self.mode} zone (e.g., 'main_road' or 'sanctum'): ").strip() or f"Zone_{len(self.all_polygons)+1}"
                    area = float(input("Enter estimated real-world area in sq meters (default 40.0): ") or "40.0")
                    self.all_polygons.append({
                        "id": name.lower().replace(" ", "_"),
                        "name": name,
                        "area_sq_meters": area,
                        "points": list(self.current_points),
                    })
                    self.current_points = []
                    self.redraw()
                    print(f"Added zone '{name}'. Press 's' when ready to save.")
            elif key == ord("s"):
                self.save()
                break

        cv2.destroyAllWindows()

    def save(self):
        if self.mode == "traffic":
            if self.all_polygons:
                self.data["traffic"]["road_roi"] = self.all_polygons[0]["points"]
        elif self.mode == "crowd":
            if self.all_polygons:
                self.data["crowd_pandal"]["zones"] = [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "area_sq_meters": p["area_sq_meters"],
                        "polygon": p["points"],
                    }
                    for p in self.all_polygons
                ]

        config_dir = os.path.dirname(self.config_file)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Successfully saved ROI zones to {self.config_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Video ROI Polygons")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--mode", type=str, choices=["traffic", "crowd"], default="traffic")
    parser.add_argument("--config", type=str, default="config/zones.json")
    args = parser.parse_args()

    drawer = ROIDrawer(args.video, args.mode, args.config)
    drawer.run()
