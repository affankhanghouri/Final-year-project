"""Session-scoped local search decisions; no camera, detector or speech I/O."""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from homesense.config import positive
from homesense.guidance import GuidanceResult, GuidanceSettings, GuidanceStatus
from homesense.motion import MotionEvidence, MotionKind
from homesense.perception import DetectionBatch
from homesense.session import SearchSession, SessionStatus
from homesense.stability import Visibility, VisibilitySnapshot
from homesense import diagnostic_logging as diagnostic


class SearchPhase(str, Enum):
    OBSERVE_CURRENT_VIEW = 'observe_current_view'
    SCAN_LEFT = 'scan_left'
    SCAN_RIGHT = 'scan_right'
    TARGET_ACQUIRED = 'target_acquired'
    TARGET_LOST = 'target_lost'
    SEARCH_EXHAUSTED = 'search_exhausted'
    INACTIVE = 'inactive'


class ScanStatus(str, Enum):
    OBSERVING = 'observing'
    SCANNING = 'scanning'
    PAUSED = 'paused'
    WRONG_WAY = 'wrong_way'
    UNKNOWN = 'unknown'
    TOO_FAST = 'too_fast'
    RECOVERING = 'recovering'
    HOLD = 'hold'
    TARGET_VISIBLE = 'target_visible'
    EXHAUSTED = 'exhausted'
    INACTIVE = 'inactive'


@dataclass(frozen=True)
class ScanSettings:
    initial_observe_seconds: float = 5.0
    min_negative_frames: int = 3
    min_observation_span: float = .4
    left_extent: float = .35
    right_extent: float = .7
    max_leg_seconds: float = 30.0
    inactivity_seconds: float = 8.0
    paused_after: float = 3.0
    wrong_way_seconds: float = .45
    wrong_way_samples: int = 3
    candidate_hold: float = .8
    max_age: float = .5
    response_grace: float = 2.0
    max_retries: int = 2
    continuity_grace: float = .5
    uncertain_after: float = 1.0
    target_loss_grace: float = 1.5
    target_recovery_seconds: float = 6.0
    target_recovery_wall_seconds: float = 12.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == 'max_retries':
                if type(self.max_retries) is not int or not 0 <= self.max_retries <= 3:
                    raise ValueError('max_retries must be an integer between 0 and 3')
            elif name in ('min_negative_frames', 'wrong_way_samples'):
                if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                    raise ValueError(f'{name} must be an integer >= 2')
            else:
                positive(name, getattr(self, name), zero=name in ('initial_observe_seconds', 'response_grace', 'uncertain_after'))
        if self.paused_after >= self.inactivity_seconds:
            raise ValueError('pause prompt must precede inactivity limit')


@dataclass(frozen=True)
class ScanProgress:
    direction: str | None = None
    leg_id: int = 0
    progress: float = 0.0
    attempted_left: bool = False
    attempted_right: bool = False
    completed_left: bool = False
    completed_right: bool = False
    wrong_way: bool = False
    inactive_seconds: float = 0.0
    leg_seconds: float = 0.0
    retry: int = 0
    response_seconds: float = 0.0
    uncertain_seconds: float = 0.0


@dataclass(frozen=True)
class StrategySnapshot:
    session_id: UUID
    phase: SearchPhase
    status: ScanStatus
    scan: ScanProgress
    valid_until: float
    reason: str = ''
    last_target: GuidanceResult | None = None  # History only; never current guidance.
    recovery_id: int = 0
    recovery_seconds: float = 0.0

    @property
    def condition(self):
        # A single candidate pauses progress without cutting an instruction into
        # fragments. Confirmed acquisition changes phase and invalidates the leg;
        # stale target-position speech still has its separate freshness condition.
        if self.phase is SearchPhase.TARGET_LOST:
            return f'target_lost:{self.recovery_id}'
        return f'scan:{self.scan.leg_id}:{self.phase.value}'


class ScanProgressTracker:
    """Net measured image-width displacement, with explicit continuity breaks.

    No elapsed-time completion. Reverse motion removes progress. Each motion
    frame is consumed once. Brief uncertainty freezes progress; only recovery
    against the same estimator reference may continue it. Epoch changes or long
    gaps reset credit. Already completed legs are local effort, not coverage.
    """

    def __init__(self, settings=ScanSettings()):
        self.settings = settings
        self.direction = None
        self.leg_id = 0
        self.attempted = set()
        self.completed = set()
        self._baseline = self._epoch = None
        self._last_id = 0
        self._progress = 0.0
        self._last_update = None
        self._leg_seconds = self._idle_seconds = 0.0
        self._wrong_since = None
        self._wrong_count = 0
        self._wrong_way = False
        self._gap_since = None
        self._ready_at = 0.0
        self._response_seconds = 0.0
        self._retry = 0
        self._uncertain_seconds = 0.0

    def begin(self, direction: str, now: float) -> None:
        if direction not in ('left', 'right'):
            raise ValueError('scan direction must be left or right')
        self.direction = direction
        self.leg_id += 1
        self.attempted.add(direction)
        self._baseline = self._epoch = None
        self._last_id = 0
        self._progress = self._leg_seconds = self._idle_seconds = 0.0
        self._last_update = now
        self._wrong_since = None
        self._wrong_count = 0
        self._wrong_way = False
        self._gap_since = None
        self._ready_at = now+self.settings.response_grace
        self._response_seconds = self.settings.response_grace
        self._retry = 0
        self._uncertain_seconds = 0.0

    def retry(self, now: float) -> None:
        """One bounded recovery opportunity; never manufacture movement credit."""
        self._retry += 1
        self._idle_seconds = 0.0
        self._ready_at = now+self.settings.response_grace
        self._gap_since = None
        self._uncertain_seconds = 0.0

    def _break_continuity(self):
        self._baseline = None
        self._progress = 0.0

    def pause(self, now: float) -> None:
        # Acquisition/candidate time must not consume the scan's movement budget.
        self._last_update = now
        self._break_continuity()
        self._ready_at = now+self.settings.response_grace
        self._gap_since = None
        self._uncertain_seconds = 0.0
        self._wrong_since = None
        self._wrong_count = 0
        self._wrong_way = False

    def observe(self, motion: MotionEvidence | None, now: float, *, instruction_busy=False) -> ScanProgress:
        if instruction_busy:
            self._ready_at = now+self.settings.response_grace
        delta = max(0.0, now-max(self._last_update, self._ready_at)) if self._last_update is not None else 0.0
        self._last_update = now
        self._response_seconds = max(0.0, self._ready_at-now)
        self._leg_seconds += delta
        self._idle_seconds += delta
        if motion is None or not motion.reliable:
            if self._gap_since is None:
                self._gap_since = now
            self._uncertain_seconds = max(0.0, now-self._gap_since)
            if (now-self._gap_since >= self.settings.continuity_grace or
                    motion is not None and (motion.epoch != self._epoch or motion.kind is MotionKind.TOO_FAST)):
                self._break_continuity()
            self._wrong_since = None
            self._wrong_count = 0
            self._wrong_way = False
            return self.snapshot()
        if motion.frame_id <= self._last_id:
            return self.snapshot()
        self._last_id = motion.frame_id
        if self._gap_since is not None and now-self._gap_since >= self.settings.continuity_grace:
            self._break_continuity()
        self._gap_since = None
        self._uncertain_seconds = 0.0
        if self._baseline is None or self._epoch != motion.epoch:
            self._baseline, self._epoch = motion.offset, motion.epoch
            self._progress = 0.0
        extent = self.settings.left_extent if self.direction == 'left' else self.settings.right_extent
        progress = (motion.offset-self._baseline)*(-1 if self.direction == 'left' else 1)/extent
        if progress > self._progress + 1e-6:
            self._idle_seconds = 0.0
        self._progress = max(0.0, min(1.0, progress))
        if motion.kind in (MotionKind.LEFT, MotionKind.RIGHT) and motion.kind.value != self.direction:
            if self._wrong_since is None:
                self._wrong_since = motion.captured_at
            self._wrong_count += 1
            self._wrong_way = (self._wrong_count >= self.settings.wrong_way_samples
                               and motion.captured_at-self._wrong_since >= self.settings.wrong_way_seconds)
        else:
            self._wrong_since = None
            self._wrong_count = 0
            self._wrong_way = False
        if self._progress >= 1:
            self.completed.add(self.direction)
        return self.snapshot()

    def snapshot(self) -> ScanProgress:
        return ScanProgress(self.direction, self.leg_id, self._progress,
                            'left' in self.attempted, 'right' in self.attempted,
                            'left' in self.completed, 'right' in self.completed,
                            self._wrong_way, self._idle_seconds, self._leg_seconds,
                            self._retry, self._response_seconds, self._uncertain_seconds)


class SearchStrategyController:
    """Decide one bounded left/right attempt inside the existing search session.

    The coordinator still owns session status and resource cleanup. TARGET_ACQUIRED
    means temporal presence with eligible guidance, never user-confirmed success.
    Optical flow extents are approximate image widths, not yaw or room coverage.
    """

    def __init__(self, session: SearchSession, settings=ScanSettings(), *, min_confidence=GuidanceSettings().min_confidence):
        self.session, self.settings = session, settings
        self.min_confidence = min_confidence
        self.progress = ScanProgressTracker(settings)
        self.phase = SearchPhase.OBSERVE_CURRENT_VIEW
        self._last_batch_id = 0
        self._negative_times = []
        self._hold_until = float('-inf')
        self._acquired = False
        self._reason = ''
        self.last_target: GuidanceResult | None = None
        self._loss_started = None
        self._recovery_started = None
        self._recovery_effort = 0.0
        self._recovery_last_update = None
        self._recovery_ready_at = 0.0
        self._recovery_id = 0
        self._observation_started = session.started_at
        self._diagnostic_last = None
        self._diagnostic_at = float('-inf')
        self._diagnostic_input = {}

    def _snapshot(self, status, now, reason=''):
        result = StrategySnapshot(self.session.session_id, self.phase, status, self.progress.snapshot(),
                                min(self.session.deadline, now+self.settings.max_age), reason,
                                self.last_target, self._recovery_id, self._recovery_effort)
        if diagnostic.enabled():
            scan = result.scan
            previous = self._diagnostic_last
            state = (self.phase.value, status.value, reason, scan.retry)
            identity = (state, self._diagnostic_input.get('motion_frame_id'), self._diagnostic_input.get('batch_frame_id'))
            if previous != identity or now-self._diagnostic_at >= 1:
                old_state = previous[0] if previous else None
                old_progress = self._diagnostic_progress if previous else 0.
                action = ('reset' if old_progress > 0 and scan.progress == 0 else
                          'increased' if scan.progress > old_progress else
                          'decreased' if scan.progress < old_progress else 'froze')
                timer_pause = ('speech_pending_or_playing' if self._diagnostic_input.get('instruction_busy') else
                               'target_hold' if status in (ScanStatus.HOLD, ScanStatus.TARGET_VISIBLE) else
                               'reaction_grace' if scan.response_seconds > 0 else None)
                diagnostic.event('strategy', 'search_decision', self.session.session_id,
                    important=old_state != state, previous_phase=old_state[0] if old_state else None,
                    previous_status=old_state[1] if old_state else None,
                    phase=self.phase.value, status=status.value, reason=reason or status.value,
                    decision_at=now, direction=scan.direction, leg_id=scan.leg_id,
                    progress=scan.progress, previous_progress=old_progress, progress_action=action,
                    baseline_offset=self.progress._baseline, baseline_epoch=self.progress._epoch,
                    progress_rejection=('no fresh perception' if not self._diagnostic_input.get('batch_fresh') else
                        (self._diagnostic_input.get('motion_reason') or 'no eligible fresh motion')
                        if not self._diagnostic_input.get('motion_eligible') else
                        'opposite_requested_direction' if self._diagnostic_input.get('motion_direction') in ('left', 'right')
                            and self._diagnostic_input.get('motion_direction') != scan.direction else None),
                    leg_completed=scan.completed_left if scan.direction == 'left' else scan.completed_right,
                    completed_left=scan.completed_left, completed_right=scan.completed_right,
                    wrong_way_correction=status is ScanStatus.WRONG_WAY,
                    observation_seconds=max(0., now-self.session.started_at),
                    view_observation_seconds=max(0., now-self._observation_started),
                    target_loss_seconds=max(0., now-self._loss_started) if self._loss_started is not None else 0.,
                    target_recovery_effort=self._recovery_effort,
                    target_recovery_wall=max(0., now-self._recovery_started) if self._recovery_started is not None else 0.,
                    target_recovery_effort_limit=self.settings.target_recovery_seconds,
                    target_recovery_wall_limit=self.settings.target_recovery_wall_seconds,
                    recovery_timer_running=(self.phase is SearchPhase.TARGET_LOST
                        and not self._diagnostic_input.get('instruction_busy') and now >= self._recovery_ready_at),
                    last_target_at=self.last_target.captured_at if self.last_target else None,
                    last_target_direction=self.last_target.direction.value if self.last_target and self.last_target.direction else None,
                    observation_required=self.settings.initial_observe_seconds,
                    negative_frames=len(self._negative_times),
                    negative_span=self._negative_times[-1]-self._negative_times[0] if self._negative_times else 0.,
                    required_negative_frames=self.settings.min_negative_frames,
                    required_negative_span=self.settings.min_observation_span,
                    target_hold_remaining=max(0., self._hold_until-now),
                    min_target_confidence=self.min_confidence, freshness_limit=self.settings.max_age,
                    response_seconds=scan.response_seconds, inactivity_seconds=scan.inactive_seconds,
                    inactivity_limit=self.settings.inactivity_seconds, effort_seconds=scan.leg_seconds,
                    effort_limit=self.settings.max_leg_seconds, retry=scan.retry, retry_limit=self.settings.max_retries,
                    remaining_session_seconds=max(0., self.session.deadline-now),
                    timers_running=timer_pause is None, timer_pause_reason=timer_pause,
                    effort_timer_running=self.phase in (SearchPhase.SCAN_LEFT, SearchPhase.SCAN_RIGHT) and timer_pause is None,
                    **self._diagnostic_input)
                self._diagnostic_last, self._diagnostic_at = identity, now
                self._diagnostic_progress = scan.progress
        return result

    def update(self, session: SearchSession, batch: DetectionBatch | None, visibility: VisibilitySnapshot | None,
               guidance: GuidanceResult | None, motion: MotionEvidence | None, now: float,
               *, instruction_busy=False) -> StrategySnapshot:
        sid = self.session.session_id
        if diagnostic.enabled():
            self._diagnostic_input = dict(instruction_busy=bool(instruction_busy),
                batch_frame_id=batch.frame.frame_id if batch else None,
                batch_session_id=batch.session_id if batch else None,
                batch_age=now-batch.frame.captured_at if batch else None,
                motion_frame_id=motion.frame_id if motion else None,
                motion_age=now-motion.captured_at if motion else None,
                motion_epoch=motion.epoch if motion else None,
                motion_direction=motion.kind.value if motion else None,
                motion_offset=motion.offset if motion else None,
                motion_reason=motion.reason if motion else 'no current motion',
                motion_eligible=bool(motion and motion.session_id == sid and motion.reliable
                                     and 0 <= now-motion.captured_at < self.settings.max_age))
        if session.session_id != sid or session.status is not SessionStatus.ACTIVE or now >= session.deadline:
            self.phase = SearchPhase.INACTIVE
            return self._snapshot(ScanStatus.INACTIVE, now)
        if self.phase is SearchPhase.INACTIVE:
            return self._snapshot(ScanStatus.INACTIVE, now)
        if self.phase is SearchPhase.SEARCH_EXHAUSTED:
            return self._snapshot(ScanStatus.EXHAUSTED, now, self._reason)
        fresh = (batch is not None and batch.session_id == sid and
                 session.started_at <= batch.frame.captured_at <= now and now-batch.frame.captured_at < self.settings.max_age)
        positive = fresh and any(d.class_name == session.target.value and d.confidence >= self.min_confidence for d in batch.detections)
        if diagnostic.enabled():
            self._diagnostic_input.update(batch_fresh=bool(fresh), target_candidate=bool(positive),
                visibility=visibility.state.value if visibility else None,
                guidance_status=guidance.status.value if guidance else None)
        if fresh and batch.frame.frame_id > self._last_batch_id:
            self._last_batch_id = batch.frame.frame_id
            if positive:
                self._hold_until = now+self.settings.candidate_hold
                self._negative_times.clear()
            else:
                self._negative_times.append(batch.frame.captured_at)
                if len(self._negative_times) > self.settings.min_negative_frames:
                    self._negative_times = self._negative_times[:self.settings.min_negative_frames-1] + self._negative_times[-1:]
        acquired = (fresh and positive and visibility is not None and visibility.session_id == sid and visibility.target == session.target
                    and visibility.state is Visibility.VISIBLE and visibility.valid_until is not None and now < visibility.valid_until
                    and guidance is not None and guidance.session_id == sid and guidance.target == session.target
                    and guidance.frame_id == batch.frame.frame_id and guidance.captured_at == batch.frame.captured_at
                    and (guidance.actionable and guidance.valid_until is not None and now < guidance.valid_until
                         or guidance.status is GuidanceStatus.AMBIGUOUS))
        if acquired:
            self.phase = SearchPhase.TARGET_ACQUIRED
            self._acquired = True
            self.last_target = guidance
            self._loss_started = self._recovery_started = self._recovery_last_update = None
            self._recovery_effort = 0.0
            self.progress.pause(now)
            return self._snapshot(ScanStatus.TARGET_VISIBLE, now)
        if self._acquired:
            self.progress.pause(now)
            if self._loss_started is None:
                self._loss_started = now
            if now-self._loss_started < self.settings.target_loss_grace:
                return self._snapshot(ScanStatus.HOLD, now, 'brief_target_loss')
            if self._recovery_started is None:
                self._recovery_id += 1
                self._recovery_started = self._recovery_last_update = now
                self._recovery_ready_at = now+self.settings.response_grace
            self.phase = SearchPhase.TARGET_LOST
            if instruction_busy:
                self._recovery_ready_at = now+self.settings.response_grace
            self._recovery_effort += max(0., now-max(self._recovery_last_update, self._recovery_ready_at))
            self._recovery_last_update = now
            if (self._recovery_effort < self.settings.target_recovery_seconds and
                    now-self._recovery_started < self.settings.target_recovery_wall_seconds):
                return self._snapshot(ScanStatus.RECOVERING, now, 'target_lost_reacquiring')
            # A bounded reacquisition opportunity precedes a new local scan.
            # Candidate flicker never restarts this recovery budget.
            self._acquired = False
            previous_leg_id = self.progress.leg_id
            self.progress = ScanProgressTracker(self.settings)
            self.progress.leg_id = previous_leg_id
            self.phase = SearchPhase.OBSERVE_CURRENT_VIEW
            self._observation_started = now
            self._negative_times.clear()
            self._hold_until = float('-inf')
            return self._snapshot(ScanStatus.OBSERVING, now, 'target_recovery_failed')
        if now < self._hold_until:
            self.progress.pause(now)
            return self._snapshot(ScanStatus.HOLD, now)
        if self.phase is SearchPhase.OBSERVE_CURRENT_VIEW:
            enough = (len(self._negative_times) >= self.settings.min_negative_frames
                      and self._negative_times[-1]-self._negative_times[0] >= self.settings.min_observation_span)
            if not fresh or not enough or now-self._observation_started < self.settings.initial_observe_seconds:
                return self._snapshot(ScanStatus.OBSERVING, now)
            self.phase = SearchPhase.SCAN_LEFT
            self.progress.begin('left', now)
        if motion is not None and (motion.session_id != sid or not 0 <= now-motion.captured_at < self.settings.max_age):
            motion = None
        progress = self.progress.observe(motion if fresh else None, now, instruction_busy=instruction_busy)
        if diagnostic.enabled():
            self._diagnostic_input.update(input_leg_direction=progress.direction, input_leg_progress=progress.progress)
        if progress.progress >= 1:
            if self.phase is SearchPhase.SCAN_LEFT:
                self.phase = SearchPhase.SCAN_RIGHT
                self.progress.begin('right', now)
                self.progress.observe(motion, now)  # Establish the turn-around baseline, credit nothing yet.
                return self._snapshot(ScanStatus.SCANNING, now)
            self.phase, self._reason = SearchPhase.SEARCH_EXHAUSTED, 'local_limit'
            return self._snapshot(ScanStatus.EXHAUSTED, now, self._reason)
        inactivity = progress.inactive_seconds >= self.settings.inactivity_seconds
        if inactivity and progress.leg_seconds < self.settings.max_leg_seconds and progress.retry < self.settings.max_retries:
            self.progress.retry(now)
            return self._snapshot(ScanStatus.RECOVERING, now, 'retry_after_inactivity')
        if progress.leg_seconds >= self.settings.max_leg_seconds or inactivity:
            self.phase = SearchPhase.SEARCH_EXHAUSTED
            self._reason = ('no_fresh_results' if not fresh else
                            'motion_unconfirmed' if motion is None or not motion.reliable else 'no_progress')
            return self._snapshot(ScanStatus.EXHAUSTED, now, self._reason)
        if progress.response_seconds > 0:
            return self._snapshot(ScanStatus.SCANNING, now, 'speech_or_response_grace')
        if not fresh or motion is None or motion.kind is MotionKind.UNKNOWN:
            if progress.uncertain_seconds < self.settings.uncertain_after:
                return self._snapshot(ScanStatus.SCANNING, now, 'brief_tracking_loss')
            return self._snapshot(ScanStatus.UNKNOWN, now, 'no reliable current motion/perception')
        if motion.kind is MotionKind.TOO_FAST:
            return self._snapshot(ScanStatus.TOO_FAST, now)
        if progress.wrong_way:
            return self._snapshot(ScanStatus.WRONG_WAY, now)
        if progress.inactive_seconds >= self.settings.paused_after:
            return self._snapshot(ScanStatus.PAUSED, now)
        return self._snapshot(ScanStatus.SCANNING, now)
