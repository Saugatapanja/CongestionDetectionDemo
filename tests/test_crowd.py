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


def test_people_outside_zone_boxes_are_not_counted():
    analyzer = PandalDensityAnalyzer(
        [
            {
                "id": "entry",
                "name": "Entry",
                "area_sq_meters": 10.0,
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            },
            {
                "id": "exit",
                "name": "Exit",
                "area_sq_meters": 10.0,
                "polygon": [[110, 0], [200, 0], [200, 100], [110, 100]],
            },
        ]
    )

    stats = analyzer.analyze_zones([(50, 50), (150, 50), (250, 50), (250, 150)])

    assert [zone["raw_head_count"] for zone in stats] == [1, 1]
    assert sum(zone["raw_head_count"] for zone in stats) == 2


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
    assert any(a["zone_id"] == "exit_corridor" for a in alerts)
    history_count = len(alert_sys.alert_history)
    repeated_alerts = alert_sys.evaluate_safety_risks(zone_stats)
    assert repeated_alerts
    assert len(alert_sys.alert_history) == history_count
    assert os.path.exists(alert_sys.log_file)


def test_exit_corridor_warning_is_not_reported_as_an_emergency(tmp_path):
    alert_sys = CrowdAlertSystem(
        log_file=str(tmp_path / "crowd_alerts.json"), alert_cooldown_sec=0.0
    )
    warning = [{
        "id": "exit_corridor",
        "name": "Exit Corridor",
        "density_per_m2": 0.5,
        "head_count": 10,
        "level": 2,
    }]
    critical = [{
        **warning[0],
        "head_count": 12,
        "level": 3,
    }]
    heavier_warning = [{
        **warning[0],
        "head_count": 11,
    }]

    warning_alert = alert_sys.evaluate_safety_risks(warning)[0]
    heavier_warning_alert = alert_sys.evaluate_safety_risks(heavier_warning)[0]
    emergency_alert = alert_sys.evaluate_safety_risks(critical)[0]

    assert warning_alert["title"] == "EXIT FLOW ADVISORY"
    assert warning_alert["severity"] == "WARNING"
    assert warning_alert["police_action"] != heavier_warning_alert["police_action"]
    assert emergency_alert["title"] == "EMERGENCY: EXIT CORRIDOR BLOCKAGE"
    assert emergency_alert["severity"] == "CRITICAL"


def test_primary_alert_tracks_the_highest_current_risk(tmp_path):
    alert_sys = CrowdAlertSystem(
        log_file=str(tmp_path / "crowd_alerts.json"), alert_cooldown_sec=0.0
    )
    exit_warning = {
        "id": "exit_corridor",
        "name": "Exit Corridor",
        "density_per_m2": 0.5,
        "head_count": 10,
        "level": 2,
    }
    sanctum_normal = {
        "id": "main_sanctum",
        "name": "Main Sanctum",
        "density_per_m2": 0.5,
        "head_count": 5,
        "level": 1,
    }
    sanctum_critical = {
        **sanctum_normal,
        "head_count": 12,
        "level": 3,
    }

    normal_sanctum_alerts = alert_sys.evaluate_safety_risks(
        [sanctum_normal, exit_warning]
    )
    critical_sanctum_alerts = alert_sys.evaluate_safety_risks(
        [sanctum_critical, exit_warning]
    )

    assert normal_sanctum_alerts[0]["zone_id"] == "exit_corridor"
    assert critical_sanctum_alerts[0]["zone_id"] == "main_sanctum"
    assert critical_sanctum_alerts[0]["title"] == "HOLD ENTRY QUEUE BARRICADE"


def test_crowd_alert_on_sanctum_level_increase(tmp_path):
    alert_sys = CrowdAlertSystem(
        log_file=str(tmp_path / "crowd_alerts.json"), alert_cooldown_sec=0.0
    )

    def sanctum_stats(head_count, level):
        return [{
            "id": "main_sanctum",
            "name": "Main Sanctum",
            "density_per_m2": head_count / 10,
            "head_count": head_count,
            "level": level,
        }]

    assert not any(
        "HOLD ENTRY" in alert["title"]
        for alert in alert_sys.evaluate_safety_risks(sanctum_stats(5, 1))
    )
    alerts = alert_sys.evaluate_safety_risks(sanctum_stats(12, 3))

    assert any("HOLD ENTRY" in alert["title"] for alert in alerts)
