"""FastAPI Web Server for Congestion Detection Command Center.

Provides real-time MJPEG video streaming with AI overlays, live telemetry REST APIs,
Kolkata-specific OpenStreetMap diversion map serving, full-frame video detection,
and upload controls with Source, Destination, and Road Name configuration.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from typing import Dict, Generator, List, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yaml

# Silence benign Windows socket closure exceptions on stream disconnection
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from core.stream_loader import VideoStreamLoader
from core.tracker import CentroidTracker
from modules.traffic.vehicle_detector import VehicleDetector
from modules.traffic.congestion_meter import CongestionMeter
from modules.traffic.route_recommender import RouteRecommender
from modules.crowd.crowd_detector import OverheadCrowdDetector
from modules.crowd.density_analyzer import PandalDensityAnalyzer
from modules.crowd.alert_system import CrowdAlertSystem
from tools.sample_video_generator import generate_traffic_sample, generate_crowd_sample

app = FastAPI(title="Smart Congestion & Crowd Command Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploaded_videos")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AppState:
    def __init__(self):
        self.mode = "traffic"  # "traffic" or "crowd"
        self.stream_version = 0
        self.traffic_video = os.path.join(PROJECT_ROOT, "data", "sample_videos", "traffic_demo.mp4")
        self.crowd_video = os.path.join(PROJECT_ROOT, "data", "sample_videos", "crowd_demo.mp4")
        self.active_road_name = "Central Avenue (CR Avenue)"
        self.source_junction = "Shyambazar_5Point"
        self.dest_junction = "Park_Circus_7Point"
        self.roi_mode = "full"  # "full" for 100% video coverage or "calibrated"
        self.config = self._load_config()
        self.zones = self._load_zones()

        # Pre-initialize AI models once on startup for instantaneous stream start & CPU efficiency
        t_cfg = self.config.get("traffic", {})
        c_cfg = self.config.get("crowd", {})
        self.vehicle_detector = VehicleDetector(
            model_name=t_cfg.get("model_name", "yolov8n.pt"),
            confidence=t_cfg.get("confidence_threshold", 0.35),
            target_classes=t_cfg.get("target_classes", [2, 3, 5, 7]),
        )
        self.crowd_detector = OverheadCrowdDetector(
            model_name=c_cfg.get("model_name", "yolov8n.pt"),
            confidence=c_cfg.get("confidence_threshold", 0.25),
        )
        self.route_recommender = RouteRecommender(city_name="Kolkata, India")

        self.telemetry = {
            "mode": "traffic",
            "fps": 0.0,
            "timestamp": time.time(),
            "traffic": {
                "level_of_service": "A",
                "status": "FREE_FLOW",
                "occupancy_pct": 0.0,
                "vehicle_count": 0,
                "vehicle_breakdown": {},
                "stationary_vehicles": 0,
                "avg_speed_kmh": 0.0,
                "avg_confidence_pct": 92.5,
                "flow_rate_vpm": 0,
                "monitored_road": self.active_road_name,
                "source_junction": self.source_junction,
                "dest_junction": self.dest_junction,
                "roi_mode": self.roi_mode,
                "is_divert_recommended": False,
                "recommendation": "Flow Normal",
                "alternate_routes": [],
            },
            "crowd": {
                "total_heads": 0,
                "avg_confidence_pct": 89.0,
                "zones": [],
                "active_alerts": [],
            },
        }

    def _load_config(self) -> Dict:
        path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_zones(self) -> Dict:
        path = os.path.join(PROJECT_ROOT, "config", "zones.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f) or {}
        return {}


state = AppState()

# Ensure demo videos exist
if not os.path.exists(state.traffic_video):
    generate_traffic_sample(state.traffic_video)
if not os.path.exists(state.crowd_video):
    generate_crowd_sample(state.crowd_video)


def generate_traffic_stream() -> Generator[bytes, None, None]:
    """Processes traffic video stream with YOLO vehicle tracking, full-frame detection,

    and Kolkata route diversion analysis.
    """
    state.config = state._load_config()
    state.zones = state._load_zones()
    
    t_cfg = state.config.get("traffic", {})
    target_w, target_h = 800, 450
    local_version = state.stream_version
    local_video = state.traffic_video

    # Full frame detection vs Calibrated ROI
    if state.roi_mode == "full":
        road_roi = np.array([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]], dtype=np.int32)
    else:
        road_roi_list = state.zones.get("traffic", {}).get("road_roi")
        if not road_roi_list:
            road_roi_list = [[180, 560], [380, 240], [720, 240], [920, 560]]
        road_roi = np.array(road_roi_list, dtype=np.int32)

    frame_stride = t_cfg.get("frame_stride", 3)
    loader = VideoStreamLoader(
        source=local_video,
        target_width=target_w,
        target_height=target_h,
        frame_stride=frame_stride,
        loop=True,
    )

    detector = state.vehicle_detector
    recommender = state.route_recommender
    tracker = CentroidTracker(max_disappeared=20, max_distance=80.0, fps=loader.fps)
    meter = CongestionMeter(
        road_roi=road_roi,
        free_flow_thresh=t_cfg.get("occupancy_thresholds", {}).get("free_flow", 30.0),
        moderate_thresh=t_cfg.get("occupancy_thresholds", {}).get("moderate", 65.0),
        congested_thresh=t_cfg.get("occupancy_thresholds", {}).get("congested", 80.0),
        car_threshold=t_cfg.get("car_count_threshold", 10),
        fps=loader.fps,
    )

    last_diversion_check = 0.0
    latest_routes = []

    try:
        for f_idx, frame, fps in loader.frames():
            if state.mode != "traffic" or state.stream_version != local_version or state.traffic_video != local_video:
                break

            all_boxes, all_lbls, all_confs, roi_boxes, roi_lbls, roi_confs = detector.detect(
                frame, road_roi=meter.road_roi
            )
            tracked = tracker.update(all_boxes, all_lbls, confidences=all_confs)
            metrics = meter.evaluate(roi_boxes, roi_lbls, tracked, frame.shape, confidences=roi_confs)

            now = time.time()
            if metrics["is_divert_recommended"] and (now - last_diversion_check > 3.0):
                last_diversion_check = now
                if metrics.get("car_threshold_exceeded"):
                    # Heavy car congestion (>10 cars): apply strong penalty ratio even if occupancy is moderate
                    congestion_ratio = min(1.0, max(0.65, metrics["occupancy_pct"] / 100.0))
                else:
                    congestion_ratio = min(1.0, metrics["occupancy_pct"] / 100.0)

                recommender.penalize_road_by_name(state.active_road_name, True, congestion_ratio=congestion_ratio)
                latest_routes = recommender.find_alternate_routes(
                    start_node=state.source_junction,
                    target_node=state.dest_junction,
                    top_k=2,
                )

                if metrics.get("car_threshold_exceeded") and latest_routes:
                    car_cnt = metrics.get("car_count", 0)
                    for r in latest_routes:
                        r["police_action"] = (
                            f"KOLKATA POLICE DIRECTIVE (>6 CARS ALERT): Divert traffic onto alternate route "
                            f"to relieve {state.active_road_name} ({car_cnt} cars queued). Saves approx. {r.get('time_saved_min', 0)} mins."
                        )

                map_out = os.path.join(OUTPUTS_DIR, "traffic_diversion_map.html")
                recommender.generate_folium_map(latest_routes, output_path=map_out)
            elif not metrics["is_divert_recommended"]:
                recommender.penalize_road_by_name(state.active_road_name, False)

            # Update shared state telemetry
            metrics["alternate_routes"] = latest_routes
            metrics["monitored_road"] = state.active_road_name
            metrics["source_junction"] = state.source_junction
            metrics["dest_junction"] = state.dest_junction
            metrics["roi_mode"] = state.roi_mode

            state.telemetry["mode"] = "traffic"
            state.telemetry["fps"] = round(fps, 1)
            state.telemetry["timestamp"] = now
            state.telemetry["traffic"] = metrics

            # Render visuals
            disp = frame.copy()
            status_color = metrics["status_color"]

            # If not full frame, draw the ROI polygon border
            if state.roi_mode != "full":
                cv2.polylines(disp, [meter.road_roi], True, status_color, 3)

            for i, (box, lbl) in enumerate(zip(all_boxes, all_lbls)):
                x1, y1, x2, y2 = box
                conf_val = all_confs[i] if i < len(all_confs) else 0.85
                cv2.rectangle(disp, (x1, y1), (x2, y2), status_color, 2)
                label_txt = f"{lbl} {int(conf_val * 100)}%"
                cv2.putText(disp, label_txt, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 2)

            ret, buffer = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        loader.release()


def generate_crowd_stream() -> Generator[bytes, None, None]:
    """Processes pandal crowd video stream with overhead density and heatmap analysis."""
    state.config = state._load_config()
    state.zones = state._load_zones()
    
    c_cfg = state.config.get("crowd", {})
    zones_list = state.zones.get("crowd_pandal", {}).get("zones", [])
    local_version = state.stream_version
    local_video = state.crowd_video

    loader = VideoStreamLoader(
        source=local_video,
        target_width=800,
        target_height=450,
        frame_stride=c_cfg.get("frame_stride", 3),
        loop=True,
    )

    detector = state.crowd_detector
    analyzer = PandalDensityAnalyzer(
        zones_config=zones_list,
        normal_thresh=c_cfg.get("density_thresholds", {}).get("normal", 1.5),
        warning_thresh=c_cfg.get("density_thresholds", {}).get("warning", 3.0),
        critical_thresh=c_cfg.get("density_thresholds", {}).get("critical", 4.5),
        blur_radius=c_cfg.get("heatmap_blur_radius", 35),
        heatmap_alpha=c_cfg.get("heatmap_alpha", 0.45),
    )
    alert_sys = CrowdAlertSystem(log_file=os.path.join(OUTPUTS_DIR, "crowd_alerts.json"))

    try:
        for f_idx, frame, fps in loader.frames():
            if state.mode != "crowd" or state.stream_version != local_version or state.crowd_video != local_video:
                break

            head_centroids, boxes, confs = detector.detect_heads(frame)
            zone_stats = analyzer.analyze_zones(head_centroids)
            heatmap_frame = analyzer.generate_heatmap_overlay(frame, head_centroids)
            active_alerts = alert_sys.evaluate_safety_risks(zone_stats)

            now = time.time()
            clean_zone_stats = [
                {
                    "id": z["id"],
                    "name": z["name"],
                    "head_count": z["head_count"],
                    "density_per_m2": z["density_per_m2"],
                    "status": z["status"],
                    "level": z["level"],
                }
                for z in zone_stats
            ]
            avg_crowd_conf = round(float(np.mean(confs)) * 100.0, 1) if confs else 88.0
            state.telemetry["mode"] = "crowd"
            state.telemetry["fps"] = round(fps, 1)
            state.telemetry["timestamp"] = now
            state.telemetry["crowd"] = {
                "total_heads": len(head_centroids),
                "avg_confidence_pct": avg_crowd_conf,
                "zones": clean_zone_stats,
                "active_alerts": active_alerts,
            }

            disp = heatmap_frame.copy()
            for z in zone_stats:
                poly = z["polygon_np"]
                col = z["status_color"]
                cv2.polylines(disp, [poly], True, col, 2)
                cx = int(np.mean(poly[:, 0]))
                cy = int(np.mean(poly[:, 1]))
                cv2.putText(disp, f"{z['name']}: {z['density_per_m2']} p/m2", (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            for cx, cy in head_centroids:
                cv2.circle(disp, (cx, cy), 3, (0, 255, 255), -1)

            ret, buffer = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    finally:
        loader.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"mode": state.mode})


@app.get("/video_feed")
async def video_feed(request: Request):
    async def stream_wrapper():
        gen = generate_traffic_stream() if state.mode == "traffic" else generate_crowd_stream()
        try:
            while True:
                if await request.is_disconnected():
                    break
                frame_chunk = await asyncio.to_thread(next, gen, None)
                if frame_chunk is None:
                    break
                yield frame_chunk
                await asyncio.sleep(0.001)
        except (asyncio.CancelledError, GeneratorExit, ConnectionResetError, OSError):
            pass
        finally:
            try:
                gen.close()
            except Exception:
                pass

    return StreamingResponse(
        stream_wrapper(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/telemetry")
def get_telemetry():
    return JSONResponse(state.telemetry)


@app.get("/api/kolkata_nodes")
def get_kolkata_nodes():
    """Returns available Kolkata junctions and prominent roads for selection."""
    recommender = RouteRecommender()
    nodes_info = []
    for node_id, data in recommender.graph.nodes(data=True):
        nodes_info.append({
            "id": node_id,
            "label": data.get("label", node_id),
            "zone": data.get("zone", "Kolkata"),
        })
    popular_roads = [
        "Central Avenue (CR Avenue)",
        "Eastern Metropolitan (EM) Bypass",
        "AJC Bose Road / Flyover",
        "Acharya Prafulla Chandra (APC) Road",
        "Park Street / Shakespeare Sarani",
        "Strand Road (Riverfront)",
        "Chowringhee Road (JL Nehru Rd)",
    ]
    return {
        "nodes": nodes_info,
        "popular_roads": popular_roads,
    }


@app.get("/api/map", response_class=HTMLResponse)
def get_map():
    map_path = os.path.join(OUTPUTS_DIR, "traffic_diversion_map.html")
    if not os.path.exists(map_path):
        map_path = os.path.join(OUTPUTS_DIR, "test_diversion_map.html")
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        "<div style='color:white;padding:20px;font-family:sans-serif;'>No diversion map generated yet. Map will appear when severe congestion triggers alternate routing.</div>"
    )


@app.post("/api/set_mode")
async def set_mode(data: Dict):
    new_mode = data.get("mode", "traffic")
    if new_mode in ["traffic", "crowd"]:
        state.stream_version += 1
        state.mode = new_mode
        state.telemetry["mode"] = new_mode
        return {"status": "success", "mode": state.mode}
    raise HTTPException(status_code=400, detail="Invalid mode")


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    mode: str = Form("traffic"),
    road_name: str = Form("Central Avenue (CR Avenue)"),
    source_junction: str = Form("Shyambazar_5Point"),
    dest_junction: str = Form("Park_Circus_7Point"),
    roi_mode: str = Form("full"),
):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files (.mp4, .avi, .mov, .mkv) are supported")

    # Increment stream version immediately to interrupt any running stream & free CPU for upload
    state.stream_version += 1

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    if mode == "traffic":
        state.traffic_video = dest_path
        state.mode = "traffic"
        state.active_road_name = road_name
        state.source_junction = source_junction
        state.dest_junction = dest_junction
        state.roi_mode = roi_mode
        state.telemetry["mode"] = "traffic"
        state.telemetry["traffic"]["monitored_road"] = road_name
        state.telemetry["traffic"]["source_junction"] = source_junction
        state.telemetry["traffic"]["dest_junction"] = dest_junction
        state.telemetry["traffic"]["roi_mode"] = roi_mode
        state.telemetry["traffic"]["vehicle_count"] = 0
        state.telemetry["traffic"]["vehicle_breakdown"] = {}
        state.telemetry["traffic"]["stationary_vehicles"] = 0
        state.telemetry["traffic"]["status"] = "ANALYZING VIDEO"
        state.telemetry["traffic"]["level_of_service"] = "A"
        state.telemetry["traffic"]["occupancy_pct"] = 0.0
        state.telemetry["traffic"]["is_divert_recommended"] = False
        state.telemetry["traffic"]["recommendation"] = f"Analyzing video feed for {road_name}..."
        state.telemetry["traffic"]["alternate_routes"] = []

        # Reset road penalty
        state.route_recommender.penalize_road_by_name(road_name, False)
    elif mode == "crowd":
        state.crowd_video = dest_path
        state.mode = "crowd"
        state.telemetry["mode"] = "crowd"
        state.telemetry["crowd"]["total_heads"] = 0
        state.telemetry["crowd"]["active_alerts"] = []

    return {
        "status": "success",
        "message": f"Successfully uploaded {file.filename}",
        "mode": mode,
        "road_name": road_name,
        "source_junction": source_junction,
        "dest_junction": dest_junction,
        "roi_mode": roi_mode,
        "video_path": dest_path,
        "stream_version": state.stream_version,
    }


@app.post("/api/reset_demo")
async def reset_demo(data: Dict):
    target = data.get("target", "all")
    state.stream_version += 1
    if target in ["traffic", "all"]:
        state.traffic_video = os.path.join(PROJECT_ROOT, "data", "sample_videos", "traffic_demo.mp4")
        state.active_road_name = "Central Avenue (CR Avenue)"
        state.source_junction = "Shyambazar_5Point"
        state.dest_junction = "Park_Circus_7Point"
        state.roi_mode = "full"
        state.route_recommender.penalize_road_by_name("Central Avenue", False)
    if target in ["crowd", "all"]:
        state.crowd_video = os.path.join(PROJECT_ROOT, "data", "sample_videos", "crowd_demo.mp4")
    return {"status": "success", "message": "Reset to default demo simulations"}
