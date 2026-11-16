import cv2
import numpy as np

from core.stream_loader import VideoStreamLoader


class FakeCapture:
    def __init__(self, _source):
        self.frame = np.zeros((90, 160, 3), dtype=np.uint8)
        self.read_count = 0

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 160,
            cv2.CAP_PROP_FRAME_HEIGHT: 90,
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 1,
        }.get(prop, 0)

    def read(self):
        self.read_count += 1
        if self.read_count == 1:
            return True, self.frame.copy()
        return False, None

    def release(self):
        pass

    def isOpened(self):
        return True


def test_loader_preserves_source_resolution_when_resize_is_disabled(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    loader = VideoStreamLoader(
        "source.mp4",
        target_width=None,
        target_height=None,
        loop=False,
    )

    try:
        assert loader.get_resolution() == (160, 90)
        _, frame, _ = next(loader.frames())
        assert frame.shape == (90, 160, 3)
    finally:
        loader.release()


def test_loader_keeps_existing_resize_behavior(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    loader = VideoStreamLoader("source.mp4", target_width=80, target_height=45, loop=False)

    try:
        assert loader.get_resolution() == (80, 45)
        _, frame, _ = next(loader.frames())
        assert frame.shape == (45, 80, 3)
    finally:
        loader.release()
