"""Image-relative target direction; consumes perception, never runs inference."""

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from typing import Callable
from uuid import UUID

from homesense.perception import DetectionBatch
from homesense.session import SessionManager, Target
from homesense.stability import Visibility, VisibilitySnapshot


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class GuidanceStatus(str, Enum):
    DIRECTION = "direction"
    UNCONFIRMED = "unconfirmed"
    UNCERTAIN = "uncertain"
    NO_DETECTION = "no_detection"
    AMBIGUOUS = "ambiguous"
    INVALID_INPUT = "invalid_input"
    WEAK_DETECTION = "weak_detection"
    STALE = "stale"
    INACTIVE = "inactive"
    MISMATCH = "mismatched_observation"
    OUTDATED = "outdated_observation"


@dataclass(frozen=True)
class GuidanceSettings:
    left_boundary: float = 1 / 3
    right_boundary: float = 2 / 3
    hysteresis: float = 0.02
    max_position_age: float = 0.5
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        for name in ("left_boundary", "right_boundary", "hysteresis", "max_position_age", "min_confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.left_boundary < self.right_boundary < 1:
            raise ValueError("require 0 < left_boundary < right_boundary < 1")
        if not 0 <= self.hysteresis < min(self.left_boundary, 1 - self.right_boundary,
                                        (self.right_boundary - self.left_boundary) / 2):
            raise ValueError("hysteresis must be nonnegative and smaller than the zone margins")
        if self.max_position_age <= 0 or not 0 <= self.min_confidence <= 1:
            raise ValueError("require positive position age and confidence between 0 and 1")


@dataclass(frozen=True)
class GuidanceResult:
    session_id: UUID | None
    target: Target | None
    status: GuidanceStatus
    direction: Direction | None
    normalized_x: float | None
    frame_id: int | None
    captured_at: float | None
    published_at: float
    valid_until: float | None

    @property
    def actionable(self) -> bool:
        """Snapshot status only; use latest() to recheck at consumption time."""
        return self.status is GuidanceStatus.DIRECTION and self.direction is not None


class GuidanceProcessor:
    """Publish to one latest-output slot, coordinated with session changes.

    Call update from one owner (the preview UI). latest() may be called from
    other threads. No callbacks, I/O, or rendering run while locks are held.
    Lock order is session then output. Returned snapshots are valid at handoff,
    not a promise that later physical presentation will still be eligible.
    """

    def __init__(self, sessions: SessionManager, *, settings: GuidanceSettings = GuidanceSettings(),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._sessions = sessions
        self._settings = settings
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: GuidanceResult | None = None
        self._bound_id: UUID | None = None
        self._last_id = 0
        self._last_capture: float | None = None
        self._previous_direction: Direction | None = None

    def _position(self, batch: DetectionBatch | None, target: Target | None) -> tuple[GuidanceStatus, float | None]:
        if batch is None or target is None:
            return GuidanceStatus.NO_DETECTION, None
        try:
            shape = batch.frame.image.shape
            if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
                return GuidanceStatus.INVALID_INPUT, None
            height, width = shape[:2]
            detections = [d for d in batch.detections if d.class_name == target.value]
            if not detections:
                return GuidanceStatus.NO_DETECTION, None
            if len(detections) != 1:
                return GuidanceStatus.AMBIGUOUS, None
            detection = detections[0]
            x1, y1, x2, y2 = (float(value) for value in detection.bbox)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, detection.confidence)) or not (
                0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 0 <= detection.confidence <= 1
            ):
                return GuidanceStatus.INVALID_INPUT, None
            if detection.confidence < self._settings.min_confidence:
                return GuidanceStatus.WEAK_DETECTION, None
            return GuidanceStatus.DIRECTION, ((x1 + x2) / 2) / width
        except (AttributeError, TypeError, ValueError, OverflowError):
            return GuidanceStatus.INVALID_INPUT, None

    def _direction(self, x: float) -> Direction:
        left, right, margin = self._settings.left_boundary, self._settings.right_boundary, self._settings.hysteresis
        previous = self._previous_direction
        if previous is Direction.LEFT and x < left + margin:
            return Direction.LEFT
        if previous is Direction.RIGHT and x >= right - margin:
            return Direction.RIGHT
        if previous is Direction.CENTER and left - margin <= x < right + margin:
            return Direction.CENTER
        if x < left:
            return Direction.LEFT
        if x < right:
            return Direction.CENTER
        return Direction.RIGHT

    def update(self, batch: DetectionBatch | None, visibility: VisibilitySnapshot) -> None:
        """Publish only from matching fresh evidence; duplicate frames do not republish."""
        status, x = self._position(batch, visibility.target)
        session_id = visibility.session_id if visibility.session_id is not None else (batch.session_id if batch else None)
        with self._sessions.guard_active(session_id) as session:
            with self._lock:
                now = self._clock()
                if self._bound_id != session_id:
                    self._bound_id = session_id
                    self._last_id = 0
                    self._last_capture = None
                    self._previous_direction = None
                if self._latest is not None and self._latest.valid_until is not None and now >= self._latest.valid_until:
                    self._previous_direction = None
                valid_until = None
                accepted = False
                if session is None or now >= session.deadline:
                    status = GuidanceStatus.INACTIVE
                elif visibility.session_id != session.session_id or visibility.target != session.target:
                    status = GuidanceStatus.MISMATCH
                elif batch is not None:
                    captured = batch.frame.captured_at
                    if (batch.session_id != session.session_id or visibility.source_frame_id != batch.frame.frame_id
                            or visibility.last_observation_at != captured):
                        status = GuidanceStatus.MISMATCH
                    elif (not all(math.isfinite(v) for v in (captured, batch.completed_at, batch.inference_seconds))
                          or not session.started_at <= captured <= batch.completed_at <= now
                          or batch.inference_seconds < 0
                          or now - captured >= self._settings.max_position_age):
                        status = GuidanceStatus.STALE
                    elif (type(batch.frame.frame_id) is not int or batch.frame.frame_id < self._last_id or batch.frame.frame_id <= 0
                          or (self._last_capture is not None and captured < self._last_capture)):
                        status = GuidanceStatus.OUTDATED
                    else:
                        accepted = True
                        if visibility.state is Visibility.UNCONFIRMED:
                            status = GuidanceStatus.UNCONFIRMED
                        elif visibility.state is not Visibility.VISIBLE:
                            status = GuidanceStatus.UNCERTAIN
                        elif (visibility.last_positive_at != captured or visibility.valid_until is None
                              or not math.isfinite(visibility.valid_until) or now >= visibility.valid_until):
                            status = GuidanceStatus.STALE
                        elif status is GuidanceStatus.DIRECTION:
                            if batch.frame.frame_id == self._last_id:
                                return
                            if self._last_capture is not None and captured == self._last_capture:
                                status = GuidanceStatus.OUTDATED
                            else:
                                valid_until = min(session.deadline, captured + self._settings.max_position_age, visibility.valid_until)
                elif visibility.state is Visibility.UNCONFIRMED:
                    status = GuidanceStatus.UNCONFIRMED
                elif visibility.state is not Visibility.VISIBLE:
                    status = GuidanceStatus.UNCERTAIN
                direction = self._direction(x) if status is GuidanceStatus.DIRECTION and x is not None and valid_until is not None else None
                if direction is None:
                    x = None
                    self._previous_direction = None
                else:
                    self._previous_direction = direction
                if accepted and batch is not None and batch.frame.frame_id > self._last_id:
                    self._last_id = batch.frame.frame_id
                    self._last_capture = batch.frame.captured_at
                self._latest = GuidanceResult(
                    session_id, session.target if session else None, status, direction, x,
                    batch.frame.frame_id if batch else None, batch.frame.captured_at if batch else None,
                    now, valid_until,
                )

    def latest(self) -> GuidanceResult | None:
        """Recheck eligibility/age at retrieval; an old copy must not be replayed."""
        with self._lock:
            result = self._latest
        if result is None:
            return None
        with self._sessions.guard_active(result.session_id) as session:
            with self._lock:
                # A concurrent update may have replaced the slot since the first read.
                if self._latest is not result:
                    return None
                now = self._clock()
                status = result.status
                if session is None or now >= session.deadline:
                    status = GuidanceStatus.INACTIVE
                elif result.actionable and (result.valid_until is None or now >= result.valid_until):
                    status = GuidanceStatus.STALE
                if status is not result.status:
                    self._previous_direction = None
                    self._latest = replace(result, status=status, direction=None, normalized_x=None, valid_until=None)
                return self._latest
