import json
from pathlib import Path

from core.zone_profiles import crowd_zones, profile_for_video, traffic_roi


def test_traffic_video_profiles_for_sample_videos():
    zones_path = Path(__file__).resolve().parents[1] / "config" / "zones.json"
    zones = json.loads(zones_path.read_text(encoding="utf-8"))

    # 1. road_traffic_1.mp4: full roadway ROI capturing all vehicles
    profile1 = profile_for_video(zones, "traffic", "road_traffic_1.mp4")
    assert profile1 is not None
    roi1, overrides1 = traffic_roi(zones, "road_traffic_1.mp4", (1080, 1920))
    assert roi1 == [[0, 800], [1080, 800], [1080, 1920], [0, 1920]]
    assert overrides1.get("roi_thickness") == 6
    assert overrides1.get("box_thickness") == 4

    # 2. road_traffic_2.mp4: left outgoing lane ROI
    profile2 = profile_for_video(zones, "traffic", "road_traffic_2.mp4")
    assert profile2 is not None
    roi2, overrides2 = traffic_roi(zones, "road_traffic_2.mp4", (1920, 1080))
    assert roi2 == [[650, 150], [880, 150], [880, 1080], [650, 1080]]
    assert overrides2 == {}

    # 3. Unprofiled video falls back to default traffic ROI scaled to frame size
    unprofiled = "road_traffic_3.mp4"
    assert profile_for_video(zones, "traffic", unprofiled) is None
    roi_unprofiled, _ = traffic_roi(zones, unprofiled, (1024, 576))
    assert roi_unprofiled == [[0, 0], [1024, 0], [1024, 576], [0, 576]]


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
