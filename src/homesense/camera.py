"""Camera capture and a single-latest-frame handoff."""

from dataclasses import dataclass
import logging
from homesense import diagnostic_logging as diagnostic
import threading
import time
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Frame:
    frame_id: int
    captured_at: float
    image: NDArray[np.uint8]


class LatestFrameBuffer:
    """One producer, any number of consumers; close wakes all waiters.

    Publishing and reading each copy the image. The producer must not mutate its
    input while publish is copying it. A closed buffer cannot be reused.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self._last_id = 0

    def publish(self, frame: Frame) -> bool:
        image = frame.image.copy()
        with self._condition:
            if self._closed:
                return False
            if frame.frame_id <= self._last_id:
                raise ValueError("frame IDs must increase")
            self._last_id = frame.frame_id
            self._frame = Frame(frame.frame_id, frame.captured_at, image)
            self._condition.notify_all()
            return True

    def wait_for_newer(self, after_id: int = 0, timeout: float | None = None) -> Frame | None:
        """Return a private copy of the latest newer frame, or None on timeout/close."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or (self._frame is not None and self._frame.frame_id > after_id),
                timeout=timeout,
            )
            if self._closed or self._frame is None or self._frame.frame_id <= after_id:
                return None
            frame = self._frame
        return Frame(frame.frame_id, frame.captured_at, frame.image.copy())

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._frame = None
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


class FrameSource(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...
    def release(self) -> None: ...


def _open_camera(camera_id: int) -> FrameSource:
    import cv2
    return cv2.VideoCapture(camera_id)


class CameraWorker:
    """Single-use worker. Construct a fresh worker and buffer for each search."""

    def __init__(
        self,
        buffer: LatestFrameBuffer,
        camera_id: int = 0,
        *,
        source_factory: Callable[[int], FrameSource] = _open_camera,
        clock: Callable[[], float] = time.monotonic,
        diagnostic_session_id=None,
    ) -> None:
        self._buffer = buffer
        self._camera_id = camera_id
        self._source_factory = source_factory
        self._clock = clock
        self._diagnostic_session_id = diagnostic_session_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._release_error: Exception | None = None

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    @property
    def release_error(self) -> Exception | None:
        """Expose failed release separately without replacing the primary error."""
        with self._lock:
            return self._release_error

    def start(self) -> None:
        """Start capture asynchronously; check error when capture ends."""
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("camera worker is single-use")
            self._thread = threading.Thread(target=self._run, name="camera-worker")
            try:
                self._thread.start()
            except Exception as exc:
                # Failed startup is terminal, but there is no started thread to join.
                self._thread = None
                self._stop.set()
                self._error = exc
                raise

    def stop(self, timeout: float = 2.0) -> bool:
        """Request stop and wait up to timeout; return whether the worker ended."""
        self._stop.set()
        self._buffer.close()
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        source: FrameSource | None = None
        try:
            if self._stop.is_set():
                return
            source = self._source_factory(self._camera_id)
            if not source.isOpened():
                raise RuntimeError(f"could not open camera {self._camera_id}")
            diagnostic.event('camera', 'camera_ready', self._diagnostic_session_id, camera_id=self._camera_id)
            frame_id = 0
            while not self._stop.is_set():
                ok, image = source.read()
                captured_at = self._clock()
                if self._stop.is_set():
                    break
                if not ok or image is None:
                    raise RuntimeError("camera capture failed")
                frame_id += 1
                if not self._buffer.publish(Frame(frame_id, captured_at, image)):
                    break
        except Exception as exc:
            diagnostic.event('camera', 'camera_error', self._diagnostic_session_id, severity='ERROR', error=str(exc))
            with self._lock:
                self._error = exc
        finally:
            if source is not None:
                try:
                    source.release()
                    diagnostic.event('camera', 'camera_released', self._diagnostic_session_id)
                except Exception as exc:
                    with self._lock:
                        self._release_error = exc
                        if self._error is None:
                            self._error = exc
                    # Preserve the primary error while still reporting release failures.
                    logging.getLogger(__name__).exception("Camera release failed")
            self._buffer.close()
