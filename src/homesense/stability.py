"""Time-based target presence, without instance association or position memory."""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable
from uuid import UUID

from homesense.perception import DetectionBatch
from homesense.session import SearchSession, SessionManager, SessionStatus, Target


class Visibility(str, Enum):
    UNCONFIRMED = "unconfirmed"
    VISIBLE = "visible"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class StabilitySettings:
    confirmation_span: float = 0.4
    min_positive_observations: int = 3
    max_positive_gap: float = 0.6
    uncertain_after: float = 0.4
    forget_after: float = 0.8
    max_observation_age: float = 0.5
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        for name in ("confirmation_span", "max_positive_gap", "uncertain_after",
                     "forget_after", "max_observation_age"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite seconds")
        if type(self.min_positive_observations) is not int or self.min_positive_observations < 2:
            raise ValueError("min_positive_observations must be at least 2")
        if not self.uncertain_after < self.forget_after or self.max_positive_gap > self.forget_after:
            raise ValueError("require uncertain_after < forget_after and max_positive_gap <= forget_after")
        if (isinstance(self.min_confidence, bool) or not isinstance(self.min_confidence, (int, float))
                or not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1):
            raise ValueError("min_confidence must be between 0 and 1")


@dataclass(frozen=True)
class VisibilitySnapshot:
    session_id: UUID | None
    target: Target | None
    state: Visibility
    last_positive_at: float | None
    last_observation_at: float | None
    source_frame_id: int | None
    # A saved VISIBLE snapshot must not authorize output indefinitely.
    valid_until: float | None = None


class StabilityProcessor:
    """Single-owner processor. Call snapshot regularly to age silent periods.

    Only accepted fresh inference batches are observations. Nothing here marks a
    search found, retains a box, or promises that visibility implies localization.
    All clocks and input timestamps must share a monotonic time base.
    """

    def __init__(
        self, sessions: SessionManager, *, settings: StabilitySettings = StabilitySettings(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sessions = sessions
        self._settings = settings
        self._clock = clock
        self._reset(None)

    def _reset(self, session: SearchSession | None) -> None:
        self._session = session
        self._state = Visibility.UNCONFIRMED
        self._confirmed = False
        self._candidate_start: float | None = None
        self._positive_count = 0
        self._last_positive: float | None = None
        self._last_capture: float | None = None
        self._last_completed: float | None = None
        self._last_id: int | None = None

    def _sync_session(self) -> SearchSession | None:
        session = self._sessions.current
        if session is not None and session.status is not SessionStatus.ACTIVE:
            session = None
        if (session.session_id if session else None) != (self._session.session_id if self._session else None):
            self._reset(session)
        return session

    def _age(self, now: float) -> None:
        if self._last_positive is None:
            return
        age = now - self._last_positive
        if age >= self._settings.forget_after:
            self._confirmed = False
            self._candidate_start = None
            self._positive_count = 0
            self._last_positive = None
            self._state = Visibility.UNCONFIRMED
        elif self._confirmed and age >= self._settings.uncertain_after:
            self._state = Visibility.UNCERTAIN

    def observe(self, batch: DetectionBatch) -> bool:
        """Accept a fresh, ordered batch once; False means no observation added."""
        session = self._sync_session()
        if session is None:
            return False
        with self._sessions.guard_active(session.session_id) as active:
            if active is None:
                self._reset(None)
                return False
            now = self._clock()
            self._age(now)
            captured = batch.frame.captured_at
            completed = batch.completed_at
            if (
                batch.session_id != active.session_id
                or now >= active.deadline
                or not all(math.isfinite(v) for v in (captured, completed, batch.inference_seconds))
                or not active.started_at <= captured <= completed <= now
                or batch.inference_seconds < 0
                or now - captured >= self._settings.max_observation_age
                or batch.frame.frame_id <= 0
                or (self._last_id is not None and batch.frame.frame_id <= self._last_id)
                or (self._last_capture is not None and captured <= self._last_capture)
                or (self._last_completed is not None and completed < self._last_completed)
            ):
                return False
            self._last_id = batch.frame.frame_id
            self._last_capture = captured
            self._last_completed = completed
            positive = any(
                detection.class_name == active.target.value
                and math.isfinite(detection.confidence)
                and self._settings.min_confidence <= detection.confidence <= 1
                for detection in batch.detections
            )
            if positive:
                if not self._confirmed:
                    if (self._last_positive is None or self._candidate_start is None
                            or captured - self._last_positive >= self._settings.max_positive_gap):
                        self._candidate_start = captured
                        self._positive_count = 0
                    self._positive_count += 1
                    if (self._positive_count >= self._settings.min_positive_observations
                            and captured - self._candidate_start >= self._settings.confirmation_span):
                        self._confirmed = True
                self._last_positive = captured
                self._state = Visibility.VISIBLE if self._confirmed else Visibility.UNCONFIRMED
            elif self._confirmed:
                self._state = Visibility.UNCERTAIN
            else:
                # A real negative breaks an unconfirmed run; silence does not.
                self._candidate_start = None
                self._positive_count = 0
            self._age(now)
            return True

    def snapshot(self) -> VisibilitySnapshot:
        """Return current presence only; cancellation/expiry resets on access."""
        session = self._sync_session()
        if session is not None:
            with self._sessions.guard_active(session.session_id) as active:
                if active is None or self._clock() >= active.deadline:
                    self._reset(None)
                else:
                    self._age(self._clock())
        return VisibilitySnapshot(
            self._session.session_id if self._session else None,
            self._session.target if self._session else None, self._state,
            self._last_positive, self._last_capture, self._last_id,
            min(self._session.deadline, self._last_positive + self._settings.uncertain_after)
            if self._session is not None and self._state is Visibility.VISIBLE and self._last_positive is not None
            else None,
        )
