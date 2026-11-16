"""Search session lifecycle, independent of camera and detection code."""

from dataclasses import dataclass, replace
from contextlib import contextmanager
from enum import Enum
import math
import threading
import time
from typing import Callable, Iterator
from uuid import UUID, uuid4


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class Target(str, Enum):
    PHONE = "phone"
    KEYS = "keys"
    WALLET = "wallet"


@dataclass(frozen=True)
class SearchSession:
    session_id: UUID
    target: Target
    started_at: float
    deadline: float
    status: SessionStatus


class SessionManager:
    """Own one search at a time, with synchronized individual operations.

    Returned sessions are immutable snapshots; query the manager for current state.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._current: SearchSession | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> SearchSession | None:
        """Return the current session, updating its timeout first."""
        with self._lock:
            self._expire_if_needed()
            return self._current

    def start(self, target: str, timeout_seconds: float) -> SearchSession:
        """Start a search, replacing any active search after validation."""
        if not isinstance(target, str):
            raise ValueError("target must be phone, keys, or wallet")
        try:
            normalized_target = Target(target.strip().lower())
        except ValueError as exc:
            raise ValueError("target must be phone, keys, or wallet") from exc
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout must be a positive finite number of seconds")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")

        with self._lock:
            now = self._clock()
            deadline = now + timeout_seconds
            if not math.isfinite(deadline) or deadline <= now:
                raise ValueError("timeout cannot be represented as a future deadline")
            self._expire_if_needed(now)
            self._current = SearchSession(uuid4(), normalized_target, now, deadline, SessionStatus.ACTIVE)
            return self._current

    def cancel(self, session_id: UUID) -> bool:
        """Cancel the current active session if its identity matches."""
        with self._lock:
            self._expire_if_needed()
            if self._current is None or self._current.session_id != session_id:
                return False
            if self._current.status is not SessionStatus.ACTIVE:
                return False
            self._current = replace(self._current, status=SessionStatus.CANCELLED)
            return True

    def is_eligible(self, session_id: UUID) -> bool:
        """Check eligibility at result delivery time, including the deadline."""
        with self._lock:
            self._expire_if_needed()
            return (
                self._current is not None
                and self._current.session_id == session_id
                and self._current.status is SessionStatus.ACTIVE
            )

    def _expire_if_needed(self, now: float | None = None) -> None:
        if self._current is None or self._current.status is not SessionStatus.ACTIVE:
            return
        if (self._clock() if now is None else now) >= self._current.deadline:
            self._current = replace(self._current, status=SessionStatus.TIMED_OUT)

    @contextmanager
    def guard_active(self, session_id: UUID | None) -> Iterator[SearchSession | None]:
        """Coordinate a brief result handoff with cancellation/replacement.

        Never perform inference, I/O, or rendering inside this guard. Eligibility
        is checked on entry; a returned result must be rechecked at later use.
        """
        with self._lock:
            self._expire_if_needed()
            session = self._current
            active = (
                session is not None
                and session.session_id == session_id
                and session.status is SessionStatus.ACTIVE
            )
            yield session if active else None
