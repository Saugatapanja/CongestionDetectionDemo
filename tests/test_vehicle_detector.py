import numpy as np

from modules.traffic.vehicle_detector import VehicleDetector


class FakeValue:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def tolist(self):
        return self.coordinates


class FakeBox:
    cls = [FakeValue(2)]
    conf = [FakeValue(0.9)]

    def __init__(self, coordinates):
        self.xyxy = [FakeCoordinates(coordinates)]


class FakeResult:
    def __init__(self):
        self.boxes = [
            FakeBox([10, 70, 20, 90]),
            FakeBox([65, 70, 75, 90]),
        ]


class FakeModel:
    def __init__(self):
        self.source = None

    def predict(self, **kwargs):
        self.source = kwargs["source"]
        return [FakeResult()]


def test_vehicle_detector_limits_inference_to_masked_road_roi():
    detector = VehicleDetector.__new__(VehicleDetector)
    detector.model = FakeModel()
    detector.confidence = 0.35
    detector.target_classes = [2, 3, 5, 7]
    detector.class_names = {2: "Car"}
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    road_roi = np.array([[0, 0], [80, 0], [50, 99], [0, 99]], dtype=np.int32)

    _, _, _, roi_boxes, _, _ = detector.detect(frame, road_roi=road_roi)

    assert detector.model.source.shape == (100, 81, 3)
    assert np.any(detector.model.source[80, 30])
    assert not np.any(detector.model.source[80, 70])
    assert roi_boxes == [[10, 70, 20, 90]]
