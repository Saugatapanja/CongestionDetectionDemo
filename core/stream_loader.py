"""Video Stream Loader with CPU performance optimizations.

Supports local video files, RTSP IP camera streams, and webcams.
Provides frame skipping, automatic looping, and real-time FPS measurement.
"""

import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


class VideoStreamLoader:
    def __init__(
        self,
        source: str,
        target_width: Optional[int] = 1024,
        target_height: Optional[int] = 576,
        frame_stride: int = 1,
        loop: bool = True,
    ):
        """Initialize the video stream loader.

        Args:
            source: Path to video file or RTSP URL or integer for webcam.
            target_width: Rescaled width, or None with target_height to preserve source size.
            target_height: Rescaled height, or None with target_width to preserve source size.
            frame_stride: Process every Nth frame (e.g., 2 skips alternate frames).
            loop: Whether to loop the video upon reaching the end.
        """
        self.source = source
        if (target_width is None) != (target_height is None):
            raise ValueError("target_width and target_height must both be set or both be None")
        self.target_width = target_width
        self.target_height = target_height
        self.frame_stride = max(1, frame_stride)
        self.loop = loop

        # Webcam check
        if str(source).isdigit():
            self.source = int(source)

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video stream at: {source}")

        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if target_width is None and (self.source_width <= 0 or self.source_height <= 0):
            self.cap.release()
            raise ValueError(f"Unable to determine source resolution for: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # Performance monitoring
        self._last_time = time.time()
        self._measured_fps = self.fps

    def get_resolution(self) -> Tuple[int, int]:
        if self.target_width is None:
            return self.source_width, self.source_height
        assert self.target_height is not None
        return self.target_width, self.target_height

    def frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Yield (source frame index, frame, measured processing FPS)."""
        while True:
            # Fast-skip intermediate frames with grab() to save CPU decoding
            for _ in range(self.frame_stride - 1):
                if not self.cap.grab():
                    if self.loop and self.total_frames > 0:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_idx = 0
                        break
                    else:
                        return
                self.current_frame_idx += 1

            ret, frame = self.cap.read()
            if not ret:
                if self.loop and self.total_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    continue
                else:
                    break

            self.current_frame_idx += 1

            # Resize frame for uniform display and CPU efficiency
            if self.target_width is not None and self.target_height is not None:
                frame = cv2.resize(frame, (self.target_width, self.target_height))

            # FPS calculation
            now = time.time()
            elapsed = now - self._last_time
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                self._measured_fps = 0.9 * self._measured_fps + 0.1 * instant_fps
            self._last_time = now

            yield self.current_frame_idx, frame, self._measured_fps

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
