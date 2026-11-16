"""Latest-frame inference with session and monotonic freshness gates."""

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Callable, Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from homesense.camera import Frame, LatestFrameBuffer
from homesense.detector import Detection
from homesense.session import SearchSession, SessionManager


class Detector(Protocol):
    def predict(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]: ...


@dataclass(frozen=True)
class FreshnessSettings:
    # Development test limits, not measured responsiveness guarantees.
    max_frame_age: float = 0.25
    max_result_age: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.max_frame_age, self.max_result_age):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("age limits must be positive finite seconds")


@dataclass(frozen=True)
class DetectionBatch:
    session_id: UUID
    frame: Frame
    completed_at: float
    inference_seconds: float
    detections: tuple[Detection, ...]

    @property
    def capture_to_result_seconds(self) -> float:
        return self.completed_at - self.frame.captured_at


@dataclass(frozen=True)
class WorkerStats:
    processed: int = 0
    skipped: int = 0
    stale_frames: int = 0
    rejected_results: int = 0
    published: int = 0
    processed_fps: float = 0.0


class DetectionWorker:
    """One worker per session/buffer; only this worker calls its detector.

    A latest-result slot replaces previous results. Consumers receive private
    images. Session lock precedes worker lock; neither is held during inference.
    """

    def __init__(
        self, buffer: LatestFrameBuffer, detector: Detector,
        sessions: SessionManager, session: SearchSession, *,
        settings: FreshnessSettings = FreshnessSettings(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._buffer = buffer
        self._detector = detector
        self._sessions = sessions
        self._session = session
        self._settings = settings
        self._clock = clock
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._latest: DetectionBatch | None = None
        self._stats = WorkerStats()
        self._started_at: float | None = None
        self._ended_at: float | None = None

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    @property
    def stats(self) -> WorkerStats:
        with self._lock:
            end = self._ended_at if self._ended_at is not None else self._clock()
            elapsed = end - self._started_at if self._started_at is not None else 0
            return replace(self._stats, processed_fps=self._stats.processed / elapsed if elapsed > 0 else 0.0)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("detection worker is single-use")
            self._started_at = self._clock()
            self._thread = threading.Thread(target=self._run, name="detection-worker")
            try:
                self._thread.start()
            except Exception as exc:
                # Failed startup is terminal, but there is no started thread to join.
                self._thread = None
                self._stop.set()
                self._error = exc
                self._ended_at = self._clock()
                self._finished.set()
                raise

    def stop(self, timeout: float = 2.0) -> bool:
        with self._lock:
            self._stop.set()
            self._latest = None
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def latest_result(self, after_id: int = 0) -> DetectionBatch | None:
        """Return a fresh eligible snapshot, never a permanent validity promise."""
        with self._lock:
            batch = self._latest
        if batch is None or batch.frame.frame_id <= after_id:
            return None
        # Copy outside locks; recheck after copying in case state/age changed.
        frame = replace(batch.frame, image=batch.frame.image.copy())
        with self._sessions.guard_active(self._session.session_id) as session:
            with self._lock:
                if session is None or self._stop.is_set() or self._error is not None or self._buffer.closed:
                    return None
                if not self._fresh(frame, self._settings.max_result_age):
                    return None
                return replace(batch, frame=frame)

    def _fresh(self, frame: Frame, limit: float) -> bool:
        now = self._clock()
        age = now - frame.captured_at
        return (
            now < self._session.deadline
            and frame.captured_at >= self._session.started_at
            and 0 <= age < limit
        )

    def _count(self, **increments: int) -> None:
        with self._lock:
            self._stats = replace(self._stats, **{
                key: getattr(self._stats, key) + value for key, value in increments.items()
            })

    def _run(self) -> None:
        last_id = 0
        try:
            while not self._stop.is_set() and self._sessions.is_eligible(self._session.session_id):
                frame = self._buffer.wait_for_newer(last_id, timeout=0.05)
                if self._buffer.closed:
                    break
                if frame is None:
                    continue
                self._count(skipped=max(0, frame.frame_id - last_id - 1))
                last_id = frame.frame_id
                if self._stop.is_set() or not self._sessions.is_eligible(self._session.session_id):
                    break
                if not self._fresh(frame, self._settings.max_frame_age):
                    self._count(stale_frames=1)
                    continue
                started = self._clock()
                detections = self._detector.predict(frame.image)
                completed = self._clock()
                self._count(processed=1)
                batch = DetectionBatch(
                    self._session.session_id, frame, completed, completed - started,
                    tuple(d for d in detections if d.class_name == self._session.target.value),
                )
                with self._sessions.guard_active(self._session.session_id) as session:
                    with self._lock:
                        if session is None or self._stop.is_set() or self._buffer.closed or not self._fresh(frame, self._settings.max_result_age):
                            self._stats = replace(self._stats, rejected_results=self._stats.rejected_results + 1)
                        else:
                            self._latest = batch
                            self._stats = replace(self._stats, published=self._stats.published + 1)
        except Exception as exc:
            with self._lock:
                self._error = exc
                self._latest = None
        finally:
            with self._lock:
                self._ended_at = self._clock()
            self._finished.set()
