"""A drop-in ``cv2.VideoCapture``-shaped object backed by a streamrelay
``FrameReader``. Pass it to any code that expects a ``VideoCapture``.

This is the integration trick used by the Rope/Rope-Live recipe in the
README — most face-swap tools accept anything with .read()/.get()/.isOpened().
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from streamrelay import FrameReader, protocol


class StreamRelayCapture:
    """Quack-like-cv2.VideoCapture wrapper around a ``FrameReader``."""

    def __init__(self, shm_name: str = protocol.DEFAULT_SHM_NAME,
                 attach_timeout: float = 10.0):
        self._reader = FrameReader(shm_name=shm_name,
                                   attach_timeout=attach_timeout)
        self._last_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── cv2.VideoCapture-shaped API ──────────────────────────────────────────
    def read(self) -> Tuple[bool, np.ndarray]:
        frame = self._reader.read_latest()
        if frame is not None:
            self._last_frame = frame
        return True, self._last_frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._last_frame.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._last_frame.shape[0])
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        # Stream cannot be seeked.
        return False

    def isOpened(self) -> bool:  # noqa: N802 (matching cv2 API)
        return self._reader.attached

    def release(self) -> None:
        self._reader.close()


if __name__ == "__main__":
    cap = StreamRelayCapture()
    while cap.isOpened():
        ok, frame = cap.read()
        cv2.imshow("via cap", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
