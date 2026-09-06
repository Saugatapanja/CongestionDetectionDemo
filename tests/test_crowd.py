"""Unit tests for Pandal Crowd Density Analyzer and Alert System."""

import os
import numpy as np
from modules.crowd.alert_system import CrowdAlertSystem
from modules.crowd.density_analyzer import PandalDensityAnalyzer


def test_pandal_density_analyzer():
    zones_cfg = [
        {
            "id": "main_sanctum",
            "name": "Main Sanctum",
            "area_sq_meters": 10.0,
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        },
        {
            "id": "exit_corridor",
            "name": "Exit Passage",
            "area_sq_meters": 5.0,
            "polygon": [[110, 0], [200, 0], [200, 100], [110, 100]],
        },
    ]

    analyzer = PandalDensityAnalyzer(
        zones_cfg, normal_thresh=1.0, warning_thresh=2.0, critical_thresh=3.0
    )

    # Place 35 people inside the 10 m^2 sanctum -> 3.5 people/m^2 (CRITICAL)
    heads = [(50, 50) for _ in range(35)]
    stats = analyzer.analyze_zones(heads)

    sanctum_stat = next(s for s in stats if s["id"] == "main_sanctum")
    assert sanctum_stat["head_count"] == 35
    assert sanctum_stat["density_per_m2"] == 3.5
    assert sanctum_stat["status"] == "CRITICAL_SURGE"

    # Heatmap test
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    heat = analyzer.generate_heatmap_overlay(frame, heads)
    assert heat.shape == frame.shape
    assert np.max(heat) > 0


def test_density_threshold_boundaries():
    analyzer = PandalDensityAnalyzer(
        [{
            "id": "zone",
            "name": "Zone",
            "area_sq_meters": 10.0,
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        }],
        normal_thresh=1.5,
        warning_thresh=3.0,
        critical_thresh=4.5,
    )

    warning = analyzer.analyze_zones([(50, 50)] * 35)[0]
    critical = analyzer.analyze_zones([(50, 50)] * 45)[0]
    assert warning["density_per_m2"] == 3.5
    assert warning["status"] == "WARNING"
    assert critical["density_per_m2"] == 4.5
    assert critical["status"] == "CRITICAL_SURGE"


def test_crowd_alert_system(tmp_path):
    alert_sys = CrowdAlertSystem(
        log_file=str(tmp_path / "crowd_alerts.json"), alert_cooldown_sec=60.0
    )
    zone_stats = [
        {
            "id": "main_sanctum",
            "name": "Main Sanctum",
            "density_per_m2": 4.2,
            "head_count": 80,
            "level": 3,
        },
        {
            "id": "exit_corridor",
            "name": "Exit Corridor",
            "density_per_m2": 2.5,
            "head_count": 20,
            "level": 2,
        },
    ]

    alerts = alert_sys.evaluate_safety_risks(zone_stats)
    assert len(alerts) >= 1
    # Check that emergency actions are suggested
    titles = [a["title"] for a in alerts]
    assert any("HOLD ENTRY" in t for t in titles)
    assert any("EXIT CORRIDOR" in t for t in titles)
    history_count = len(alert_sys.alert_history)
    repeated_alerts = alert_sys.evaluate_safety_risks(zone_stats)
    assert repeated_alerts
    assert len(alert_sys.alert_history) == history_count
    assert os.path.exists(alert_sys.log_file)
