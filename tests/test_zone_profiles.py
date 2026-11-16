import json
from pathlib import Path

from core.zone_profiles import crowd_zones, profile_for_video, traffic_roi


def test_road_traffic_2_uses_left_roadway_roi_and_other_samples_use_full_frame():
    zones = {
        "traffic": {
            "road_roi": [[0, 0], [1024, 0], [1024, 576], [0, 576]],
        },
        "video_profiles": {
            "traffic": [
                {
                    "filename": "road_traffic_2.mp4",
                    "reference_size": [478, 850],
                    "road_roi": [[0, 150], [409, 150], [355, 849], [0, 849]],
                    "overrides": {},
                }
            ]
        },
    }

    roi, overrides = traffic_roi(zones, "road_traffic_2.mp4", (478, 850))
    assert profile_for_video(zones, "traffic", "road_traffic_2.mp4") is not None
    assert roi == [[0, 150], [409, 150], [355, 849], [0, 849]]
    assert overrides == {}

    for filename in ("road_traffic_1.mp4", "road_traffic_3.mp4"):
        assert profile_for_video(zones, "traffic", filename) is None
        roi, overrides = traffic_roi(zones, filename, (478, 850))
        assert roi == [[0, 0], [478, 0], [478, 850], [0, 850]]
        assert overrides == {}


def test_video_profile_filename_matching_is_case_insensitive():
    zones = {
        "video_profiles": {
            "traffic": [
                {
                    "filename": "road_traffic_2.mp4",
                }
            ]
        }
    }

    profile = profile_for_video(zones, "traffic", r"data\sample_videos\ROAD_TRAFFIC_2.MP4")
    assert profile is zones["video_profiles"]["traffic"][0]


def test_pandal_zone_boxes_have_equal_heights():
    zones_path = Path(__file__).resolve().parents[1] / "config" / "zones.json"
    zones = json.loads(zones_path.read_text(encoding="utf-8"))

    default_zones = zones["crowd_pandal"]["zones"]
    profile_zones, _ = crowd_zones(
        zones,
        "Durga Puja Pandal Overcrowded_Gemini generated.mp4",
        (1024, 576),
    )

    for configured_zones in (default_zones, profile_zones):
        heights = {
            zone["id"]: max(point[1] for point in zone["polygon"])
            - min(point[1] for point in zone["polygon"])
            for zone in configured_zones
        }
        assert len(set(heights.values())) == 1
