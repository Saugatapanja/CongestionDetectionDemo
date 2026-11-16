"""Video-specific, resolution-aware monitoring zone selection.

The project ships with demonstration zones as well as calibrated zones for the
two supplied Kolkata CCTV clips.  Keeping the calibration separate from the
detectors means a feed can be resized without silently moving its ROI.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _scale_points(points: Iterable[Sequence[float]], reference_size: Sequence[float], frame_size: Tuple[int, int]) -> List[List[int]]:
    """Scale polygon points from their calibration resolution to a frame."""
    ref_w, ref_h = reference_size
    width, height = frame_size
    sx, sy = width / float(ref_w), height / float(ref_h)
    return [[round(x * sx), round(y * sy)] for x, y in points]


def profile_for_video(zones: Dict, mode: str, video_source: str) -> Optional[Dict]:
    """Return the matching profile, falling back to the ordinary mode zones."""
    name = Path(str(video_source)).name.lower()
    for profile in zones.get("video_profiles", {}).get(mode, []):
        filenames = profile.get("filenames", [profile.get("filename", "")])
        if any(filename.lower() == name for filename in filenames):
            return profile
    return None


def traffic_roi(zones: Dict, video_source: str, frame_size: Tuple[int, int]) -> Tuple[List[List[int]], Dict]:
    """Get a traffic ROI and its profile-specific runtime overrides."""
    profile = profile_for_video(zones, "traffic", video_source)
    if profile:
        reference_size = profile["reference_size"]
        points = profile.get("road_roi") or zones.get("traffic", {}).get("road_roi")
        if not points:
            ref_w, ref_h = reference_size
            points = [[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]
        return (
            _scale_points(points, reference_size, frame_size),
            profile.get("overrides", {}),
        )
    points = zones.get("traffic", {}).get("road_roi") or [[0, 0], [1024, 0], [1024, 576], [0, 576]]
    return _scale_points(points, (1024, 576), frame_size), {}


def crowd_zones(zones: Dict, video_source: str, frame_size: Tuple[int, int]) -> Tuple[List[Dict], Dict]:
    """Get crowd zones, scaled from the calibration resolution when available."""
    profile = profile_for_video(zones, "crowd", video_source)
    if not profile:
        return zones.get("crowd_pandal", {}).get("zones", []), {}

    scaled = []
    for zone in profile.get("zones", []):
        item = dict(zone)
        item["polygon"] = _scale_points(zone["polygon"], profile["reference_size"], frame_size)
        scaled.append(item)
    return scaled, profile.get("overrides", {})
