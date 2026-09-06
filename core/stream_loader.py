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
        target_width: int = 1024,
        target_height: int = 576,
        frame_stride: int = 1,
        loop: bool = True,
    ):
        """Initialize the video stream loader.

        Args:
            source: Path to video file or RTSP URL or integer for webcam.
            target_width: Rescaled width for CPU performance and display.
            target_height: Rescaled height.
            frame_stride: Process every Nth frame (e.g., 2 skips alternate frames).
            loop: Whether to loop the video upon reaching the end.
        """
        self.source = source
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

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0

        # Performance monitoring
        self._last_time = time.time()
        self._measured_fps = self.fps

    def get_resolution(self) -> Tuple[int, int]:
        return (self.target_width, self.target_height)

    def frames(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Yield (source frame index, resized frame, measured processing FPS)."""
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
            if self.target_width and self.target_height:
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
