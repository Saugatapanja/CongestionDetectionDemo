"""Unit tests for Traffic Congestion Meter, Kinematic Tracking, and Confidence Scoring."""

import numpy as np
import pytest
from core.tracker import CentroidTracker
from modules.traffic.congestion_meter import CongestionMeter


def test_centroid_tracker_features_and_confidence():
    tracker = CentroidTracker(max_disappeared=5, max_distance=50.0, fps=25.0, meters_per_pixel=0.05)
    # Register vehicle with 94% detection confidence
    rects = [[100, 100, 150, 150]]
    objects = tracker.update(rects, ["Car"], confidences=[0.94])
    assert len(objects) == 1
    assert 0 in objects
    obj = objects[0]
    assert obj.class_name == "Car"
    assert round(obj.confidence, 2) == 0.94
    assert obj.dimensions["width"] == 50
    assert obj.dimensions["height"] == 50

    # Move vehicle to the right (+x, simulating eastbound vehicle)
    rects_moved = [[120, 100, 170, 150]]
    objects = tracker.update(rects_moved, ["Car"], confidences=[0.92])
    obj = objects[0]
    assert obj.velocity > 0
    assert obj.speed_kmh > 0
    assert obj.heading_cardinal == "Eastbound"
    assert 0.0 <= obj.heading_deg <= 360.0

    meter = CongestionMeter(road_roi=[[0, 0], [200, 0], [200, 200], [0, 200]], fps=25.0)
    metrics = meter.evaluate(rects_moved, ["Car"], objects, (300, 300, 3), confidences=[0.92])
    assert metrics["vehicle_count"] == 1
    assert metrics["avg_confidence_pct"] == 92.0
    assert metrics["avg_speed_kmh"] > 0
    assert "congestion_index" in metrics


def test_congestion_meter_levels():
    road_roi = [[0, 0], [200, 0], [200, 200], [0, 200]]  # Area = 40,000 px
    meter = CongestionMeter(
        road_roi=road_roi,
        free_flow_thresh=30.0,
        moderate_thresh=65.0,
        congested_thresh=80.0,
    )

    tracked_objects = {}
    frame_shape = (300, 300, 3)

    # 1. Empty road -> Free flow (LOS A)
    res_empty = meter.evaluate([], [], tracked_objects, frame_shape)
    assert res_empty["level_of_service"] == "A"
    assert res_empty["status"] == "FREE_FLOW"
    assert res_empty["is_divert_recommended"] is False

    # 2. Add multiple large vehicles covering > 70% of ROI
    congested_boxes = [[10, 10, 190, 190]]
    res_jam = meter.evaluate(
        congested_boxes, ["Bus"], tracked_objects, frame_shape, confidences=[0.95]
    )
    assert res_jam["occupancy_pct"] > 0
    assert res_jam["avg_confidence_pct"] == 95.0


def test_car_count_threshold_triggers_diversion():
    road_roi = [[0, 0], [800, 0], [800, 450], [0, 450]]  # Full video frame
    meter = CongestionMeter(
        road_roi=road_roi,
        free_flow_thresh=30.0,
        moderate_thresh=65.0,
        congested_thresh=80.0,
        car_threshold=6,
    )
    tracked_objects = {}
    frame_shape = (450, 800, 3)

    # 1. Test with 5 cars (<= 6 cars threshold) and low occupancy
    small_boxes_5 = [[i * 30, 50, i * 30 + 20, 70] for i in range(5)]
    labels_5 = ["Car"] * 5
    res_5 = meter.evaluate(small_boxes_5, labels_5, tracked_objects, frame_shape)
    assert res_5["car_count"] == 5
    assert res_5["car_threshold_exceeded"] is False
    assert res_5["is_divert_recommended"] is False
    assert "FLOW NORMAL" in res_5["police_message"].upper() or "NORMAL" in res_5["police_message"].upper()

    # 2. Test with 8 cars (> 6 cars threshold) -> Must trigger diversion & police alert message
    small_boxes_8 = [[i * 30, 50, i * 30 + 20, 70] for i in range(8)]
    labels_8 = ["Car"] * 8
    res_8 = meter.evaluate(small_boxes_8, labels_8, tracked_objects, frame_shape)
    assert res_8["car_count"] == 8
    assert res_8["car_threshold_exceeded"] is True
    assert res_8["is_divert_recommended"] is True
    assert "CONGESTED" in res_8["status"]
    assert "POLICE ALERT" in res_8["police_message"]
    assert "8 cars > 6" in res_8["police_message"]
    assert res_8["recommendation"] == res_8["police_message"]
