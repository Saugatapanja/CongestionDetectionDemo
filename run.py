"""Central Application Runner: Smart Traffic & Festival Crowd Congestion Management System.

Command-line orchestrator supporting:
  - Traffic congestion detection and dynamic police route diversion
  - Overhead festival pandal crowd density monitoring and surge alerts
  - Interactive ROI calibration
  - Out-of-the-box synthetic demo modes

Usage:
  python run.py --mode demo-traffic
  python run.py --mode demo-crowd
  python run.py --mode traffic --video path/to/traffic.mp4
  python run.py --mode crowd --video path/to/pandal.mp4
  python run.py --mode roi --video path/to/video.mp4 --target traffic
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
import yaml

from core.stream_loader import VideoStreamLoader
from core.tracker import CentroidTracker
from modules.traffic.vehicle_detector import VehicleDetector
from modules.traffic.congestion_meter import CongestionMeter
from modules.traffic.route_recommender import RouteRecommender
from modules.crowd.crowd_detector import OverheadCrowdDetector
from modules.crowd.density_analyzer import PandalDensityAnalyzer
from modules.crowd.alert_system import CrowdAlertSystem
from tools.sample_video_generator import generate_traffic_sample, generate_crowd_sample


def load_config(config_path: str = "config/config.yaml") -> Dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_zones(zones_path: str = "config/zones.json") -> Dict:
    if os.path.exists(zones_path):
        with open(zones_path, "r") as f:
            return json.load(f)
    return {}


def write_session_report(report: Dict, output_path: str = "outputs/session_report.json"):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_traffic_module(
    video_source: str,
    config: Dict,
    zones: Dict,
    save_output: bool = False,
    headless: bool = False,
    max_frames: Optional[int] = None,
):
    print(f"\n[INIT] Starting Traffic Congestion & Route Diversion Engine on: {video_source}")
    t_cfg = config.get("traffic", {})
    r_cfg = config.get("routing", {})

    road_roi = zones.get("traffic", {}).get("road_roi")
    if not road_roi:
        road_roi = [[180, 560], [380, 240], [720, 240], [920, 560]]

    frame_stride = max(1, int(t_cfg.get("frame_stride", 2)))
    loader = VideoStreamLoader(
        source=video_source,
        target_width=config.get("ui", {}).get("display_width", 1024),
        target_height=config.get("ui", {}).get("display_height", 576),
        frame_stride=frame_stride,
        loop=(not headless),
    )

    detector = VehicleDetector(
        model_name=t_cfg.get("model_name", "yolov8n.pt"),
        confidence=t_cfg.get("confidence_threshold", 0.35),
        target_classes=t_cfg.get("target_classes", [2, 3, 5, 7]),
    )

    tracker = CentroidTracker(max_disappeared=20, max_distance=80.0)
    meter = CongestionMeter(
        road_roi=road_roi,
        free_flow_thresh=t_cfg.get("occupancy_thresholds", {}).get("free_flow", 30.0),
        moderate_thresh=t_cfg.get("occupancy_thresholds", {}).get("moderate", 65.0),
        congested_thresh=t_cfg.get("occupancy_thresholds", {}).get("congested", 80.0),
        fps=loader.fps / frame_stride,
    )

    recommender = RouteRecommender(
        city_name=r_cfg.get("default_city", "Kolkata, India"),
        penalty_factor=r_cfg.get("congested_penalty_factor", 12.0),
    )
    route_start = r_cfg.get("default_start_node", "J1_North")
    route_end = r_cfg.get("default_end_node", "J3_South")
    monitored_segment = tuple(r_cfg.get("monitored_segment", ["J1_North", "J2_Central"]))

    writer = None
    if save_output:
        os.makedirs("outputs", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("outputs/traffic_monitored.mp4", fourcc, loader.fps / frame_stride, loader.get_resolution())

    window_name = "Smart Traffic Control - Congestion Detection & Alternate Routing (Press Q to quit, Space to pause)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)

    paused = False
    last_diversion_check = 0.0
    latest_routes: List[Dict] = []
    map_generated = False
    processed_frames = 0
    session_started = time.time()
    peak_occupancy = 0.0

    try:
        for f_idx, frame, fps in loader.frames():
            if max_frames is not None and processed_frames >= max_frames:
                break
            if not paused:
                processed_frames += 1
                # 1. Detection
                all_boxes, all_lbls, all_confs, roi_boxes, roi_lbls, roi_confs = detector.detect(
                    frame, road_roi=meter.road_roi
                )

                # 2. Tracking with kinematic features
                tracked = tracker.update(all_boxes, all_lbls, confidences=all_confs)

                # 3. Congestion evaluation with confidence & flow rate
                metrics = meter.evaluate(roi_boxes, roi_lbls, tracked, frame.shape, confidences=roi_confs)
                peak_occupancy = max(peak_occupancy, metrics["occupancy_pct"])

                # 4. Route Recommender trigger
                now = time.time()
                if metrics["is_divert_recommended"] and (now - last_diversion_check > 3.0):
                    last_diversion_check = now
                    congestion_ratio = min(1.0, metrics["occupancy_pct"] / 100.0)
                    recommender.update_segment_congestion(
                        monitored_segment,
                        True,
                        congestion_ratio=congestion_ratio,
                    )
                    latest_routes = recommender.find_alternate_routes(route_start, route_end, top_k=2)
                    map_file = recommender.generate_folium_map(
                        latest_routes, congested_segment=monitored_segment
                    )
                    if not map_generated:
                        print(f"\n[ALERT] Severe Congestion Detected! Generated Police Alternate Route Map -> {map_file}")
                        map_generated = True
                elif not metrics["is_divert_recommended"] and map_generated:
                    recommender.update_segment_congestion(monitored_segment, False)

                # 5. Render HUD Visuals
                disp = frame.copy()

                # Draw Road ROI
                status_color = metrics["status_color"]
                cv2.polylines(disp, [meter.road_roi], True, status_color, 3)

                # Draw Vehicles with Confidence Scores & Speeds
                for i, (box, lbl) in enumerate(zip(all_boxes, all_lbls)):
                    x1, y1, x2, y2 = box
                    conf = all_confs[i] if i < len(all_confs) else 0.85
                    cv2.rectangle(disp, (x1, y1), (x2, y2), status_color, 2)
                    label_str = f"{lbl} {int(conf * 100)}%"
                    cv2.putText(disp, label_str, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2)

                # Overlay HUD Top Bar
                cv2.rectangle(disp, (0, 0), (1024, 70), (20, 20, 20), -1)
                cv2.line(disp, (0, 70), (1024, 70), status_color, 2)

                # Text: Level of Service badge
                cv2.rectangle(disp, (15, 12), (90, 58), status_color, -1)
                cv2.putText(disp, f"LOS {metrics['level_of_service']}", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

                # Text: Metrics with AI Confidence & Speed
                stat_text = (
                    f"Status: {metrics['status']} | Occupancy: {metrics['occupancy_pct']}% | "
                    f"AI Conf: {metrics['avg_confidence_pct']}% | Speed: {metrics['avg_speed_kmh']} km/h"
                )
                cv2.putText(disp, stat_text, (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

                # Police Instruction
                action_text = f"ACTION: {metrics['recommendation']}"
                action_color = (0, 100, 255) if metrics["is_divert_recommended"] else (180, 255, 180)
                cv2.putText(disp, action_text, (105, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, action_color, 2)

                # If diversion active, render suggested routes banner on bottom
                if metrics["is_divert_recommended"] and latest_routes:
                    best = latest_routes[0]
                    cv2.rectangle(disp, (0, 526), (1024, 576), (10, 35, 10), -1)
                    cv2.line(disp, (0, 526), (1024, 526), (0, 220, 0), 2)
                    div_text = (
                        f"POLICE DIVERSION #{best['rank']}: {best['corridor_name']} "
                        f"({best['confidence_score']}% Conf | Saves {best['time_saved_min']}m) -> {best['vehicle_suitability']}"
                    )
                    cv2.putText(disp, div_text, (15, 558), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)

                if writer:
                    writer.write(disp)

                if not headless:
                    cv2.imshow(window_name, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    finally:
        loader.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        write_session_report({
            "mode": "traffic",
            "video_source": video_source,
            "processed_frames": processed_frames,
            "elapsed_seconds": round(time.time() - session_started, 2),
            "peak_occupancy_pct": peak_occupancy,
            "diversion_map_generated": map_generated,
        })


def run_crowd_module(
    video_source: str,
    config: Dict,
    zones: Dict,
    save_output: bool = False,
    headless: bool = False,
    max_frames: Optional[int] = None,
):
    print(f"\n[INIT] Starting Overhead Pandal Crowd Monitoring Engine on: {video_source}")
    c_cfg = config.get("crowd", {})
    zones_list = zones.get("crowd_pandal", {}).get("zones", [])

    frame_stride = max(1, int(c_cfg.get("frame_stride", 2)))
    loader = VideoStreamLoader(
        source=video_source,
        target_width=config.get("ui", {}).get("display_width", 1024),
        target_height=config.get("ui", {}).get("display_height", 576),
        frame_stride=frame_stride,
        loop=(not headless),
    )

    detector = OverheadCrowdDetector(
        model_name=c_cfg.get("model_name", "yolov8n.pt"),
        confidence=c_cfg.get("confidence_threshold", 0.25),
    )

    analyzer = PandalDensityAnalyzer(
        zones_config=zones_list,
        normal_thresh=c_cfg.get("density_thresholds", {}).get("normal", 1.5),
        warning_thresh=c_cfg.get("density_thresholds", {}).get("warning", 3.0),
        critical_thresh=c_cfg.get("density_thresholds", {}).get("critical", 4.5),
        blur_radius=c_cfg.get("heatmap_blur_radius", 35),
        heatmap_alpha=c_cfg.get("heatmap_alpha", 0.45),
    )

    alert_sys = CrowdAlertSystem(log_file="outputs/crowd_alerts.json")

    writer = None
    if save_output:
        os.makedirs("outputs", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("outputs/pandal_crowd_monitored.mp4", fourcc, loader.fps / frame_stride, loader.get_resolution())

    window_name = "Pandal Crowd Safety Control - Durga Puja (Press Q to quit, Space to pause)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 576)

    paused = False
    processed_frames = 0
    session_started = time.time()
    peak_density = 0.0
    alert_count = 0

    try:
        for f_idx, frame, fps in loader.frames():
            if max_frames is not None and processed_frames >= max_frames:
                break
            if not paused:
                processed_frames += 1
                # 1. Detect overhead people / heads
                head_centroids, boxes, confs = detector.detect_heads(frame)

                # 2. Zonal analysis
                zone_stats = analyzer.analyze_zones(head_centroids)
                peak_density = max(
                    peak_density,
                    max((z["density_per_m2"] for z in zone_stats), default=0.0),
                )

                # 3. Generate Gaussian Heatmap
                heatmap_frame = analyzer.generate_heatmap_overlay(frame, head_centroids)

                # 4. Safety Risk Evaluation & Alerts
                active_alerts = alert_sys.evaluate_safety_risks(zone_stats)
                alert_count += len(active_alerts)

                # 5. Render HUD
                disp = heatmap_frame.copy()

                # Draw Zones
                for z in zone_stats:
                    poly = z["polygon_np"]
                    col = z["status_color"]
                    cv2.polylines(disp, [poly], True, col, 2)
                    cx = int(np.mean(poly[:, 0]))
                    cy = int(np.mean(poly[:, 1]))
                    label = f"{z['name']}: {z['head_count']} ({z['density_per_m2']} p/m2)"
                    cv2.putText(disp, label, (cx - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                # Draw head markers
                for cx, cy in head_centroids:
                    cv2.circle(disp, (cx, cy), 3, (0, 255, 255), -1)

                # Top Alert Banner if any alert is active
                if active_alerts:
                    alert = active_alerts[0]
                    bar_col = (0, 0, 220) if alert["severity"] == "CRITICAL" else (0, 140, 255)
                    cv2.rectangle(disp, (0, 0), (1024, 60), bar_col, -1)
                    cv2.putText(disp, f"[POLICE ACTION] {alert['title']}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(disp, alert["police_action"], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                else:
                    cv2.rectangle(disp, (0, 0), (1024, 40), (25, 25, 25), -1)
                    cv2.putText(disp, f"DURGA PUJA PANDAL SAFETY MONITOR | Total Detected: {len(head_centroids)} | FPS: {fps:.1f}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)

                if writer:
                    writer.write(disp)

                if not headless:
                    cv2.imshow(window_name, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

    finally:
        loader.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        write_session_report({
            "mode": "crowd",
            "video_source": video_source,
            "processed_frames": processed_frames,
            "elapsed_seconds": round(time.time() - session_started, 2),
            "peak_density_per_m2": peak_density,
            "active_alert_observations": alert_count,
        })


def main():
    parser = argparse.ArgumentParser(description="Smart Traffic & Pandal Crowd Congestion System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["traffic", "crowd", "demo-traffic", "demo-crowd", "roi"],
        help="Operating mode: traffic, crowd, demo-traffic, demo-crowd, or roi",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to video file or RTSP stream")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--zones", type=str, default="config/zones.json", help="Path to zones file")
    parser.add_argument("--target", choices=["traffic", "crowd"], default="traffic", help="ROI target type")
    parser.add_argument("--save-output", action="store_true", help="Save annotated output video to outputs/")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    config = load_config(args.config)
    zones = load_zones(args.zones)

    if args.mode == "demo-traffic":
        video_path = "data/sample_videos/traffic_demo.mp4"
        if not os.path.exists(video_path):
            generate_traffic_sample(video_path)
        run_traffic_module(
            video_path,
            config,
            zones,
            save_output=args.save_output,
            headless=args.headless,
            max_frames=args.max_frames,
        )

    elif args.mode == "demo-crowd":
        video_path = "data/sample_videos/crowd_demo.mp4"
        if not os.path.exists(video_path):
            generate_crowd_sample(video_path)
        run_crowd_module(
            video_path,
            config,
            zones,
            save_output=args.save_output,
            headless=args.headless,
            max_frames=args.max_frames,
        )

    elif args.mode == "traffic":
        if not args.video:
            print("[ERROR] Please provide --video <path_to_video>")
            sys.exit(1)
        run_traffic_module(
            args.video,
            config,
            zones,
            save_output=args.save_output,
            headless=args.headless,
            max_frames=args.max_frames,
        )

    elif args.mode == "crowd":
        if not args.video:
            print("[ERROR] Please provide --video <path_to_video>")
            sys.exit(1)
        run_crowd_module(
            args.video,
            config,
            zones,
            save_output=args.save_output,
            headless=args.headless,
            max_frames=args.max_frames,
        )

    elif args.mode == "roi":
        if not args.video:
            print("[ERROR] Please provide --video <path_to_video>")
            sys.exit(1)
        from tools.roi_drawer import ROIDrawer
        drawer = ROIDrawer(args.video, mode=args.target, config_file=args.zones)
        drawer.run()


if __name__ == "__main__":
    main()
