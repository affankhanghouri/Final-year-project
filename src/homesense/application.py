"""Single-control-thread ownership of search sessions and perception resources."""

from dataclasses import dataclass
from enum import Enum
import logging
import math
import threading
import time
from typing import Callable
from uuid import UUID

from homesense.camera import CameraWorker, Frame, FrameSource, LatestFrameBuffer
from homesense.guidance import GuidanceProcessor, GuidanceResult, GuidanceSettings, GuidanceStatus
from homesense.perception import DetectionBatch, DetectionWorker, Detector, FreshnessSettings, WorkerStats
from homesense.session import SearchSession, SessionManager, SessionStatus
from homesense.stability import StabilityProcessor, StabilitySettings, VisibilitySnapshot, Visibility
from homesense.motion import MotionEvidence, MotionSettings, CameraMotionWorker
from homesense.search_strategy import SearchStrategyController, ScanSettings, StrategySnapshot, SearchPhase, ScanStatus
from homesense import diagnostic_logging as diagnostic


class ApplicationState(str, Enum):
    IDLE = "idle"
    SEARCHING = "searching"
    STOPPING = "stopping"
    FAILED = "failed"
    CLOSED = "closed"


class SearchOutcome(str, Enum):
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    REPLACED = "replaced"
    FAILED = "failed"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class ApplicationError:
    stage: str
    exception: Exception


@dataclass(frozen=True)
class ApplicationSnapshot:
    state: ApplicationState
    session: SearchSession | None
    outcome: SearchOutcome | None
    outcome_session_id: UUID | None
    errors: tuple[ApplicationError, ...]
    cleanup_complete: bool
    batch: DetectionBatch | None = None
    visibility: VisibilitySnapshot | None = None
    guidance: GuidanceResult | None = None
    stats: WorkerStats = WorkerStats()
    motion: MotionEvidence | None = None
    scan: StrategySnapshot | None = None
    search_reason: str | None = None
    # Historical evidence only: never a substitute for current `guidance`.
    last_target: GuidanceResult | None = None


class ApplicationCoordinator:
    """Own one camera and detector across successive searches.

    Construct and call all control methods on one application thread. The detector
    is supplied already loaded and must not be used elsewhere while owned here.
    Call poll regularly for timeout cleanup and evidence ageing; always close in
    finally. Returned results are expiring snapshots, not replayable commands.
    """

    def __init__(
        self, detector: Detector, *, camera_id: int = 0,
        source_factory: Callable[[int], FrameSource] | None = None,
        freshness: FreshnessSettings = FreshnessSettings(),
        stability: StabilitySettings = StabilitySettings(),
        guidance: GuidanceSettings = GuidanceSettings(),
        motion: MotionSettings | None = None,
        scan: ScanSettings = ScanSettings(),
        scan_speech_busy: Callable[[], bool] | None = None,
        shutdown_timeout: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (isinstance(shutdown_timeout, bool) or not isinstance(shutdown_timeout, (int, float))
                or not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0):
            raise ValueError("shutdown_timeout must be positive finite seconds")
        self._owner = threading.current_thread()
        self._detector = detector
        self._camera_id = camera_id
        self._source_factory = source_factory
        self._clock = clock
        self._freshness = freshness
        self._stability_settings = stability
        self._guidance_settings = guidance
        self._motion_settings, self._scan_settings = motion, scan
        self._scan_speech_busy = scan_speech_busy
        self._motion: CameraMotionWorker | None = None
        self._scan: SearchStrategyController | None = None
        self._shutdown_timeout = shutdown_timeout
        self._sessions = SessionManager(clock=clock)
        self._camera: CameraWorker | None = None
        self._inference: DetectionWorker | None = None
        self._buffer: LatestFrameBuffer | None = None
        self._stability: StabilityProcessor | None = None
        self._guidance: GuidanceProcessor | None = None
        self._state = ApplicationState.IDLE
        self._outcome: SearchOutcome | None = None
        self._outcome_id: UUID | None = None
        self._search_reason: str | None = None
        self._last_target: GuidanceResult | None = None
        self._last_scan_state = None
        self._errors: dict[str, ApplicationError] = {}
        self._closed = False
        self._stop_deadline: float | None = None
        self._stopped = {"camera": True, "inference": True, "motion": True}
        self._last_id = 0
        self._diagnostic_state = None
        self._diagnostic_batch = None
        self._diagnostic_terminal = None

    def _check_owner(self) -> None:
        if threading.current_thread() is not self._owner:
            raise RuntimeError("coordinator control methods belong to their creating thread")

    def _error(self, stage: str, error: Exception) -> None:
        # Fixed stage names bound retained diagnostics; keep the first cause per stage.
        new_error = stage not in self._errors
        if new_error:
            self._errors[stage] = ApplicationError(stage, error)
            logging.getLogger(__name__).error("%s: %s", stage, error)
        self._state = ApplicationState.FAILED
        self._outcome = SearchOutcome.FAILED
        current = self._sessions.current
        self._outcome_id = current.session_id if current else None
        if new_error:
            diagnostic.event('coordinator', 'resource_error', self._outcome_id,
                             severity='ERROR', stage=stage, error=str(error))

    def _inspect_errors(self) -> None:
        for label, worker in self._workers():
            if worker is not None and worker.error is not None:
                self._error(label + ".worker", worker.error)
        if self._camera is not None and self._camera.release_error is not None:
            self._error("camera.release", self._camera.release_error)

    def _cleanup_complete(self) -> bool:
        return (all(self._stopped.values())
                and (self._camera is None or self._camera.release_error is None))

    def _snapshot(self, batch: DetectionBatch | None = None,
                  visibility: VisibilitySnapshot | None = None,
                  guidance: GuidanceResult | None = None,
                  motion: MotionEvidence | None = None, scan: StrategySnapshot | None = None) -> ApplicationSnapshot:
        session = self._sessions.current
        # Final eligibility/age check after all result processing. No slow work here.
        with self._sessions.guard_active(session.session_id if session else None) as active:
            now = self._clock()
            if active is None or now >= active.deadline or self._state is not ApplicationState.SEARCHING:
                batch = visibility = guidance = None
                motion = scan = None
            else:
                if batch is not None and now - batch.frame.captured_at >= self._freshness.max_result_age:
                    batch = None
                if guidance is not None and guidance.actionable and (
                        guidance.valid_until is None or now >= guidance.valid_until):
                    guidance = None
                if motion is not None and (self._motion_settings is None or
                        not 0 <= now-motion.captured_at < self._motion_settings.max_age):
                    motion = None
                if scan is not None and now >= scan.valid_until:
                    scan = None
                if (batch is not None and visibility is not None and guidance is not None
                        and batch.session_id == guidance.session_id == visibility.session_id == active.session_id
                        and guidance.target == visibility.target == active.target
                        and visibility.state is Visibility.VISIBLE
                        and guidance.frame_id == batch.frame.frame_id
                        and guidance.captured_at == batch.frame.captured_at
                        and visibility.valid_until is not None and now < visibility.valid_until
                        and (guidance.actionable and guidance.valid_until is not None and now < guidance.valid_until
                             or guidance.status is GuidanceStatus.AMBIGUOUS)):
                    self._last_target = guidance
        result = ApplicationSnapshot(
            self._state, session, self._outcome, self._outcome_id,
            tuple(self._errors.values()), self._cleanup_complete(), batch, visibility,
            guidance, self._inference.stats if self._inference else WorkerStats(),
            motion, scan,
            self._search_reason,
            self._last_target,
        )
        if diagnostic.enabled():
            self._record_diagnostics(result)
        return result

    def _record_diagnostics(self, snapshot):
        """Observe existing decisions only; never feed diagnostics into lifecycle."""
        session = snapshot.session
        sid = session.session_id if session else None
        state = (sid, snapshot.state, snapshot.outcome)
        if state != self._diagnostic_state:
            local_outcome = snapshot.outcome if snapshot.outcome_session_id == sid else None
            reason = snapshot.search_reason or (local_outcome.value if local_outcome else {
                ApplicationState.SEARCHING: 'search_started', ApplicationState.STOPPING: 'resources_stopping',
                ApplicationState.IDLE: 'resources_released', ApplicationState.FAILED: 'resource_failure',
                ApplicationState.CLOSED: 'application_closed'}[snapshot.state])
            diagnostic.event('coordinator', 'application_transition', sid,
                previous_state=self._diagnostic_state[1].value if self._diagnostic_state else None,
                state=snapshot.state.value, outcome=local_outcome, reason=reason,
                cleanup_complete=snapshot.cleanup_complete)
            self._diagnostic_state = state
        batch = snapshot.batch
        if batch is not None and (sid, batch.frame.frame_id) != self._diagnostic_batch:
            diagnostic.event('perception', 'detection_sample', sid, important=False,
                target=session.target.value, frame_id=batch.frame.frame_id,
                captured_at=batch.frame.captured_at, completed_at=batch.completed_at,
                result_age=self._clock()-batch.frame.captured_at,
                inference_seconds=batch.inference_seconds,
                detections=[dict(class_name=d.class_name, confidence=d.confidence, box=d.bbox)
                            for d in batch.detections[:8]],
                visibility=snapshot.visibility.state.value if snapshot.visibility else None,
                guidance_status=snapshot.guidance.status.value if snapshot.guidance else None,
                actionable=bool(snapshot.guidance and snapshot.guidance.actionable))
            self._diagnostic_batch = (sid, batch.frame.frame_id)
        if session and snapshot.outcome is not None and snapshot.outcome_session_id == sid:
            key = (sid, snapshot.outcome, snapshot.cleanup_complete, tuple(e.stage for e in snapshot.errors))
            if key != self._diagnostic_terminal:
                diagnostic.event('coordinator', 'session_finished', sid,
                    target=session.target.value, outcome=snapshot.outcome.value,
                    reason=snapshot.search_reason or snapshot.outcome.value,
                    duration_seconds=max(0., self._clock()-session.started_at),
                    cleanup_complete=snapshot.cleanup_complete,
                    ever_acquired=snapshot.last_target is not None,
                    last_target_at=snapshot.last_target.captured_at if snapshot.last_target else None,
                    last_target_direction=(snapshot.last_target.direction.value
                        if snapshot.last_target and snapshot.last_target.direction else None),
                    errors=[dict(stage=e.stage, error=str(e.exception)) for e in snapshot.errors],
                    final_inference_stats=dict(processed=snapshot.stats.processed, published=snapshot.stats.published,
                        stale_frames=snapshot.stats.stale_frames, rejected_results=snapshot.stats.rejected_results,
                        inference_fps=snapshot.stats.processed_fps,
                        last_inference_seconds=snapshot.stats.last_inference_seconds,
                        last_capture_to_result_seconds=snapshot.stats.last_result_age))
                self._diagnostic_terminal = key

    def start_search(self, target: str, timeout_seconds: float) -> ApplicationSnapshot:
        """Start or replace a search only after confirmed cleanup; invalid input raises.

        Failed cleanup latches failure and refuses reuse for this coordinator's
        lifetime. No replacement request is queued. May wait up to shutdown_timeout.
        """
        self._check_owner()
        self._sessions.validate_request(target, timeout_seconds)
        diagnostic.event('coordinator', 'search_requested', target=target, timeout_seconds=timeout_seconds)
        if self._closed:
            raise RuntimeError("coordinator is closed")
        if self._errors:
            return self._snapshot()
        if self._state in (ApplicationState.SEARCHING, ApplicationState.STOPPING):
            current = self._sessions.current
            outcome = (SearchOutcome.TIMED_OUT if current and current.status is SessionStatus.TIMED_OUT
                       else SearchOutcome.REPLACED)
            self._end(outcome, wait=True)
        if self._errors or not self._cleanup_complete():
            return self._snapshot()
        self._stop_deadline = None
        self._search_reason = None
        self._last_scan_state = None
        self._last_id = 0
        self._camera = self._inference = None
        self._motion = self._scan = None
        self._buffer = None
        try:
            session = self._sessions.start(target, timeout_seconds)
            self._last_target = None
            diagnostic.event('coordinator', 'session_started', session.session_id,
                target=session.target.value, started_at=session.started_at, deadline=session.deadline,
                guided_scan=self._motion_settings is not None)
            self._state = ApplicationState.SEARCHING
            self._buffer = LatestFrameBuffer()
            # Fresh processors prevent any evidence or hysteresis crossing sessions.
            self._stability = StabilityProcessor(self._sessions, settings=self._stability_settings, clock=self._clock)
            self._guidance = GuidanceProcessor(self._sessions, settings=self._guidance_settings, clock=self._clock)
            camera_options = {} if self._source_factory is None else {"source_factory": self._source_factory}
            camera_options['diagnostic_session_id'] = session.session_id
            self._camera = CameraWorker(self._buffer, self._camera_id, clock=self._clock, **camera_options)
            self._stopped["camera"] = False
            self._inference = DetectionWorker(self._buffer, self._detector, self._sessions,
                                              session, settings=self._freshness, clock=self._clock)
            self._stopped["inference"] = False
            if self._motion_settings is not None:
                self._motion = CameraMotionWorker(self._buffer, self._sessions, session, self._motion_settings, clock=self._clock)
                self._scan = SearchStrategyController(session, self._scan_settings, min_confidence=self._stability_settings.min_confidence)
                self._stopped['motion'] = False
            self._camera.start()
            diagnostic.event('camera', 'camera_worker_started', session.session_id, camera_id=self._camera_id)
            self._inference.start()
            diagnostic.event('perception', 'inference_worker_started', session.session_id)
            if self._motion is not None:
                self._motion.start()
        except Exception as error:
            self._error("startup", error)
            self._end(SearchOutcome.FAILED, wait=True)
        return self._snapshot()

    def _end(self, outcome: SearchOutcome, *, wait: bool) -> None:
        current = self._sessions.current
        if current is not None:
            if outcome in (SearchOutcome.COMPLETED, SearchOutcome.EXHAUSTED):
                operation = self._sessions.complete if outcome is SearchOutcome.COMPLETED else self._sessions.exhaust
                completed = operation(current.session_id)
                if not completed and self._sessions.current.status is SessionStatus.TIMED_OUT:
                    outcome = SearchOutcome.TIMED_OUT
            else:
                self._sessions.cancel(current.session_id)
            if not self._errors:
                self._outcome, self._outcome_id = outcome, current.session_id
            logging.getLogger(__name__).info('Search ended: target=%s outcome=%s elapsed=%.3fs reason=%s',
                current.target.value, self._outcome.value if self._outcome else outcome.value,
                max(0.0, self._clock()-current.started_at), self._search_reason or '')
            diagnostic.event('coordinator', 'search_end_requested', current.session_id,
                outcome=self._outcome or outcome, reason=self._search_reason or outcome.value,
                elapsed_seconds=max(0., self._clock()-current.started_at))
        # Invalidate session first. No saved downstream output is returned during stop.
        if self._stop_deadline is None:
            self._stop_deadline = time.monotonic() + self._shutdown_timeout
        self._state = ApplicationState.STOPPING
        self._cleanup(wait=wait)
        if diagnostic.enabled():
            self._snapshot()  # Preserve the old session's end before replacement.

    def _cleanup(self, *, wait: bool) -> None:
        before_stopped = self._stopped.copy()
        # Signal every worker before any bounded join; retain objects for retries.
        for label, worker in self._workers():
            if worker is None:
                continue
            try:
                done = worker.stop(0)
                self._stopped[label] = done
            except Exception as error:
                self._error(label + ".stop", error)
        if wait:
            for label, worker in self._workers():
                if worker is None or self._stopped[label]:
                    continue
                try:
                    remaining = max(0.0, self._stop_deadline - time.monotonic())
                    self._stopped[label] = worker.stop(remaining)
                except Exception as error:
                    self._error(label + ".stop", error)
        self._inspect_errors()
        if not all(self._stopped.values()):
            if wait or time.monotonic() >= self._stop_deadline:
                for label in self._stopped:
                    if not self._stopped[label]:
                        self._error(label + ".blocked", RuntimeError(label + " worker has not stopped; reuse refused"))
            self._state = ApplicationState.FAILED if self._errors else ApplicationState.STOPPING
        elif self._closed and self._cleanup_complete():
            self._state = ApplicationState.CLOSED
        else:
            self._state = ApplicationState.FAILED if self._errors else ApplicationState.IDLE
        if before_stopped != self._stopped or wait:
            current = self._sessions.current
            diagnostic.event('coordinator', 'resource_cleanup', current.session_id if current else None,
                stopped=self._stopped, complete=self._cleanup_complete(), waiting=wait,
                errors=[e.stage for e in self._errors.values()])

    def cancel_search(self, session_id: UUID) -> ApplicationSnapshot:
        """Invalidate the matching search and attempt bounded cleanup; wrong IDs do nothing."""
        self._check_owner()
        current = self._sessions.current
        if current is not None and current.session_id == session_id and self._state is ApplicationState.SEARCHING:
            outcome = (SearchOutcome.TIMED_OUT if current.status is SessionStatus.TIMED_OUT
                       else SearchOutcome.CANCELLED)
            self._end(outcome, wait=True)
        return self._snapshot()

    def complete_search(self, session_id: UUID) -> ApplicationSnapshot:
        """User-confirmed completion, distinct from cancellation; never auto-found."""
        self._check_owner()
        current = self._sessions.current
        if current is not None and current.session_id == session_id and self._state is ApplicationState.SEARCHING:
            outcome = (SearchOutcome.TIMED_OUT if current.status is SessionStatus.TIMED_OUT
                       else SearchOutcome.COMPLETED)
            self._end(outcome, wait=True)
        return self._snapshot()

    def poll(self) -> ApplicationSnapshot:
        """Retrieve available evidence and service lifecycle; never wait for prediction.

        Timeout cleanup uses zero-wait stops. Keep polling to observe termination;
        a missed shutdown budget becomes explicit failure on a subsequent poll.
        """
        self._check_owner()
        if self._state in (ApplicationState.STOPPING, ApplicationState.FAILED) or self._closed:
            if self._stop_deadline is not None:
                self._cleanup(wait=False)
            return self._snapshot()
        if self._state is not ApplicationState.SEARCHING:
            return self._snapshot()
        try:
            self._inspect_errors()
            if self._errors:
                self._end(SearchOutcome.FAILED, wait=False)
                return self._snapshot()
            current = self._sessions.current
            if current.status is not SessionStatus.ACTIVE:
                self._end(SearchOutcome.TIMED_OUT, wait=False)
                return self._snapshot()
            if self._inference.finished or self._buffer.closed:
                self._error("termination", RuntimeError("perception ended unexpectedly"))
                self._end(SearchOutcome.FAILED, wait=False)
                return self._snapshot()
            if self._motion is not None and self._motion.finished:
                self._error('motion.termination', RuntimeError('motion worker ended unexpectedly'))
                self._end(SearchOutcome.FAILED, wait=False)
                return self._snapshot()
            batch = self._inference.latest_result()
            if batch is not None and batch.frame.frame_id > self._last_id:
                self._last_id = batch.frame.frame_id
                self._stability.observe(batch)
            visibility = self._stability.snapshot()
            self._guidance.update(batch, visibility)
            guidance = self._guidance.latest()
            motion = self._motion.latest() if self._motion is not None else None
            scan = self._scan.update(current, batch, visibility, guidance, motion, self._clock(),
                instruction_busy=bool(self._scan_speech_busy and self._scan_speech_busy())) if self._scan else None
            if scan is not None:
                state = (scan.phase, scan.status)
                if state != self._last_scan_state:
                    log = (logging.getLogger(__name__).info if self._last_scan_state is None or
                           state[0] != self._last_scan_state[0] else logging.getLogger(__name__).debug)
                    if scan.status is ScanStatus.RECOVERING:
                        log = logging.getLogger(__name__).info
                    log('Guided search: target=%s phase=%s status=%s progress=%.2f idle=%.1fs effort=%.1fs retry=%d reason=%s',
                        current.target.value, scan.phase.value, scan.status.value, scan.scan.progress,
                        scan.scan.inactive_seconds, scan.scan.leg_seconds, scan.scan.retry, scan.reason)
                    self._last_scan_state = state
            if scan is not None and scan.phase is SearchPhase.SEARCH_EXHAUSTED:
                self._search_reason = scan.reason
                self._end(SearchOutcome.EXHAUSTED, wait=False)
                return self._snapshot()
            return self._snapshot(batch, visibility, guidance, motion, scan)
        except Exception as error:
            self._error("processing", error)
            self._end(SearchOutcome.FAILED, wait=False)
            return self._snapshot()

    def close(self) -> ApplicationSnapshot:
        """Permanently refuse new searches and retry bounded cleanup on every call.

        Live workers remain owned. CLOSED means cleanup is confirmed; errors are
        retained even after eventual closure. A release error keeps state FAILED.
        """
        self._check_owner()
        self._closed = True
        if self._stop_deadline is None:
            self._end(SearchOutcome.CANCELLED, wait=True)
        else:
            self._cleanup(wait=True)
        return self._snapshot()

    def latest_frame(self, after_id: int = 0) -> Frame | None:
        """Read a private preview copy without waiting or consuming inference input.

        Diagnostic access only, on the control thread. Never opens hardware or
        changes session/worker state. No images survive cancellation/replacement.
        """
        self._check_owner()
        session = self._sessions.current
        if (self._state is not ApplicationState.SEARCHING or self._buffer is None
                or session is None or session.status is not SessionStatus.ACTIVE):
            return None
        return self._buffer.wait_for_newer(after_id, timeout=0)

    def _workers(self):
        return (('camera', self._camera), ('inference', self._inference), ('motion', self._motion))
