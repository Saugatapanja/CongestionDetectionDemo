# Smart Congestion Detection & Festival Crowd Management System

An AI-powered computer vision and geospatial traffic management platform designed for police operations. It addresses two critical urban enforcement scenarios:

1. **Module A (Traffic Congestion & Alternate Routing):** Identifies vehicle bottlenecks at key road junctions, calculates Level of Service (LOS) and road occupancy %, and dynamically generates optimized alternate diversion routes using OpenStreetMap (OSM) for traffic police dispatch.
2. **Module B (Durga Puja Pandal Crowd Safety):** Overhead CCTV monitoring for festival pandals, computing real-time crowd density ($people/m^2$), 2D Gaussian heatmaps, and automated alerts to prevent sanctum overcrowding and exit corridor stampede hazards.

---

## Key Features

- **CPU-Optimized Inference:** Uses YOLOv8 nano (`yolov8n.pt`) with frame stride processing for smooth 25+ FPS performance on standard laptops.
- **Video-Driven Offline Road Diversion:** Traffic occupancy from the road video updates a BPR-style cost on the monitored graph segment. NetworkX ranks alternate paths and Folium exports the congested link plus recommended diversion routes to `outputs/traffic_diversion_map.html`.
- **Interactive ROI Calibration (`tools/roi_drawer.py`):** Visual GUI allowing officers to click and outline road boundaries or pandal queues directly on video feeds.
- **Overhead Pandal Safety & Heatmaps:** Zonal density breakdown (Barricade Queue, Darshan Sanctum, Exit Corridor), live 2D Gaussian heatmaps, and police advisory triggers.
- **Built-in Synthetic Demos:** Complete out-of-the-box demo simulations (`demo-traffic` and `demo-crowd`) without needing immediate internet video downloads.

---

## Installation

Ensure Python 3.10+ is installed, then run:

```bash
pip install -r requirements.txt
```

---

## Quick Start: Web Command Center (Recommended)

To launch the full interactive web frontend with live video feeds, real-time gauges, and embedded OpenStreetMap routing:

```bash
python start_dashboard.py
```
*This starts the FastAPI server and automatically opens `http://127.0.0.1:8000` in your web browser.*

---

## Command-Line Demos

You can also run standalone OpenCV GUI windows directly:

### 1. Test Traffic Congestion & Route Diversion:
```bash
python run.py --mode demo-traffic
```
*Observe vehicles slowing into a bottleneck, triggering an alert and generating an interactive police diversion map.*

### 2. Test Durga Puja Pandal Crowd Safety:
```bash
python run.py --mode demo-crowd
```
*Observe overhead pandal devotees gathering at the sanctum, generating a live density heatmap and safety alert banner.*

---

## Running With Your Own Downloaded Videos

### Step 1: Calibrate Video Zones (Optional but Recommended)
If your downloaded video has specific road lanes or pandal gates:
```bash
# For traffic road boundaries:
python tools/roi_drawer.py --video path/to/your_traffic_video.mp4 --mode traffic

# For pandal zones (Entry, Sanctum, Exit):
python tools/roi_drawer.py --video path/to/your_pandal_video.mp4 --mode crowd

# Or through the unified runner:
python run.py --mode roi --video path/to/your_pandal_video.mp4 --target crowd
```
- **Left-Click:** Drop polygon vertex points.
- **c:** Complete and name the zone.
- **u:** Undo last point.
- **s:** Save calibrated coordinates to `config/zones.json`.
- **q:** Exit.

### Step 2: Run Traffic Congestion & Alternate Routing
```bash
python run.py --mode traffic --video path/to/traffic_video.mp4 --save-output
```
- When severe congestion (LOS E/F) is detected, the system will compute alternate routes and save an interactive map to:
  `outputs/traffic_diversion_map.html` (open in any web browser).

### Step 3: Run Pandal Crowd Safety Monitor
```bash
python run.py --mode crowd --video path/to/pandal_overhead.mp4 --save-output
```
- Monitors crowd density ($people/m^2$) and logs all critical events to `outputs/crowd_alerts.json`.

---

## Keyboard Controls During Live Video Window

| Key | Action |
|---|---|
| `Space` | Pause / Resume playback |
| `q` | Quit application |

---

## Project Structure

```
CongestionDetection/
├── config/
│   ├── config.yaml               # System thresholds, model settings, and map parameters
│   └── zones.json                # Calibrated polygon coordinates for roads and pandal zones
├── core/
│   ├── stream_loader.py          # Video/RTSP reader with frame stride for CPU speed
│   └── tracker.py                # Centroid tracker measuring velocity and stationary time
├── modules/
│   ├── traffic/
│   │   ├── vehicle_detector.py   # YOLOv8 vehicle detection with ROI masking
│   │   ├── congestion_meter.py   # Occupancy % and Level of Service (LOS A-F) logic
│   │   └── route_recommender.py  # OSMnx / NetworkX dynamic diversion & Folium map
│   └── crowd/
│       ├── crowd_detector.py     # Overhead person/head detector
│       ├── density_analyzer.py   # Zonal people/m² calculation & Gaussian heatmap
│       └── alert_system.py       # Pandal surge and choke-point safety alerts
├── tools/
│   ├── roi_drawer.py             # Visual polygon calibration GUI tool
│   └── sample_video_generator.py # Synthetic traffic & crowd video generator
├── outputs/                      # Generated diversion maps, logs, and monitored videos
├── tests/                        # Automated pytest suite
├── requirements.txt              # Project dependencies
└── run.py                        # Central command-line orchestrator
```

---

## Running Automated Tests

Run the complete test suite anytime:
```bash
python -m pytest -v tests/
```
