"""The sole owner of spoken wording, priorities, repeat rules, and cooldowns."""

import time
from homesense.application import ApplicationSnapshot, ApplicationState, SearchOutcome
from homesense.audio.speech_output import SpeechMessage, SpeechValidity, SpeechDelivery, DeliveryStatus
from homesense.commands.models import CommandProblem
from homesense.config import FeedbackSettings
from homesense.guidance import GuidanceSettings, GuidanceStatus
from homesense.interaction.state import InteractionEvent, InteractionEventKind
from homesense.session import SessionStatus
from homesense.stability import Visibility
from homesense.search_strategy import SearchPhase, ScanStatus
from homesense import diagnostic_logging as diagnostic


class FeedbackManager:
    def __init__(self, output, validity: SpeechValidity, *, settings=FeedbackSettings(), clock=time.monotonic):
        self.output, self.validity, self.settings, self.clock = output, validity, settings, clock
        self.generation = 0
        self.listening = False
        self._context = (None, False)
        self._snapshot: ApplicationSnapshot | None = None
        self._condition = None
        self._last_direction = None
        self._last_direction_at = float("-inf")
        self._last_missing_at = float("-inf")
        self._last_loss_at = float("-inf")
        self._last_loss_state = None
        self._loss_since = None
        self._ever_visible = False
        self._terminal = None
        self._error_announced = False
        self._scan_leg = None
        self._scan_alerts = set()
        self._last_scan_alert_at = float('-inf')
        self._strategy_phase = None
        self._target_loss_notice = None
        self._diagnostic_feedback = None
        self._pending_direction = None
        self._direction_request_at = float('-inf')
        self._direction_retry_at = float('-inf')

    def _say(self, text, *, condition=None, priority=50, fresh_guidance=False):
        sid = self._snapshot.session.session_id if self._snapshot and self._snapshot.session else None
        message = SpeechMessage(text, self.generation, condition, priority, session_id=sid,
                                fresh_guidance=fresh_guidance)
        diagnostic.event('feedback', 'speech_requested', sid, important=False, message_id=message.message_id,
                         text=text, condition=condition, priority=priority)
        accepted = self.output.submit(message)
        if accepted and fresh_guidance:
            self._pending_direction = message
            self._direction_request_at = self.clock()
        return accepted

    def _refresh_direction_delivery(self):
        """Only completed output counts as announced; workers never call policy."""
        pending = self._pending_direction
        if pending is None:
            return
        lookup = getattr(self.output, 'delivery', None)
        receipt = lookup(pending.message_id) if lookup else None
        if receipt is None and lookup is not None:
            # Receipt eviction is uncertainty, never proof that speech succeeded.
            self._pending_direction = None
            self._direction_request_at = float('-inf')
            return
        if not isinstance(receipt, SpeechDelivery):
            return  # Lightweight output adapters may not expose delivery receipts.
        if receipt.status is DeliveryStatus.COMPLETED:
            self._last_direction = pending.condition
            self._last_direction_at = receipt.started_at if receipt.started_at is not None else receipt.changed_at
            self._pending_direction = None
        elif receipt.status in (DeliveryStatus.DISCARDED, DeliveryStatus.REJECTED,
                                DeliveryStatus.INTERRUPTED, DeliveryStatus.FAILED):
            self._pending_direction = None
            self._direction_request_at = float('-inf')
            self._direction_retry_at = (receipt.started_at+self.settings.direction_cooldown
                if receipt.status is DeliveryStatus.INTERRUPTED and receipt.started_at is not None else float('-inf'))

    def set_listening(self, listening: bool):
        """Defer unsolicited speech while the user completes a command."""
        self.listening = listening
        if listening:
            self.validity.update(self.generation, None, None)

    def on_event(self, event: InteractionEvent):
        kind = event.kind
        if kind is InteractionEventKind.READY:
            self._say("HomeSense is ready.")
        elif kind is InteractionEventKind.ACKNOWLEDGE:
            self._say("I'm listening. What would you like me to find?")
        elif kind is InteractionEventKind.COMMAND_TIMEOUT:
            self._say("I didn't catch a command. Please try again.")
        elif kind is InteractionEventKind.HELP:
            self._say("I can search for your phone, keys, or wallet. Say find my phone. You can also cancel search, repeat the direction, or say I found it.")
        elif kind is InteractionEventKind.NO_SEARCH:
            self._say("There is no active search. Ask me to find your phone, keys, or wallet.")
        elif kind is InteractionEventKind.TARGET_MISMATCH:
            self._say("That is a different object from your current search. Please clarify what you found.")
        elif kind is InteractionEventKind.REPEAT:
            self._repeat()
        elif kind is InteractionEventKind.ERROR:
            # Audio failure cannot leave queued directional instructions eligible.
            self.generation += 1
            self.validity.update(self.generation, None, None)
            self._say("I'm having trouble with the audio or search system. Please check the device connections.", priority=100)
        elif kind is InteractionEventKind.UNKNOWN:
            problem = event.command.problem if event.command else None
            if problem is CommandProblem.AMBIGUOUS:
                self._say("Please choose one object: phone, keys, or wallet.")
            elif problem is CommandProblem.UNSUPPORTED:
                self._say("I can only find phones, keys, and wallets for now.")
            else:
                self._say("I didn't understand. Ask me to find your phone, keys, or wallet, or ask for help.")

    def _direction_text(self, target, direction):
        if direction == "center":
            return f"{target.capitalize()} in the center."
        return f"{target.capitalize()} on your {direction}."

    def _repeat(self):
        snapshot = self._snapshot
        guidance = snapshot.guidance if snapshot else None
        if (snapshot and snapshot.state is ApplicationState.SEARCHING and snapshot.session
                and snapshot.session.status is SessionStatus.ACTIVE and self.clock() < snapshot.session.deadline
                and guidance and guidance.actionable and guidance.valid_until is not None
                and self.clock() < guidance.valid_until and self._condition == guidance.direction.value):
            self._say(self._direction_text(snapshot.session.target.value, guidance.direction.value),
                      condition=self._condition, priority=50, fresh_guidance=True)
        else:
            scan = snapshot.scan if snapshot else None
            if (snapshot and snapshot.state is ApplicationState.SEARCHING and snapshot.session
                    and scan and scan.session_id == snapshot.session.session_id
                    and self.clock() < min(snapshot.session.deadline, scan.valid_until)
                    and self._condition == scan.condition and scan.scan.direction
                    and scan.phase in (SearchPhase.SCAN_LEFT, SearchPhase.SCAN_RIGHT)
                    and scan.status is not ScanStatus.HOLD):
                self._say(self._scan_text(scan), condition=scan.condition, priority=50)
            elif (snapshot and snapshot.session and scan and scan.phase is SearchPhase.TARGET_LOST
                  and snapshot.state is ApplicationState.SEARCHING
                  and self.clock() < min(snapshot.session.deadline, scan.valid_until)
                  and self._condition == scan.condition):
                self._say(self._target_loss_text(snapshot.session.target.value), condition=scan.condition, priority=50)
            else:
                self._say("I don't have a current confirmed direction to repeat.")

    def _scan_text(self, scan):
        direction = scan.scan.direction
        if scan.status is ScanStatus.UNKNOWN:
            return f"Pause briefly, then turn the camera {direction} slowly and steadily."
        if scan.status is ScanStatus.TOO_FAST:
            return f"Move the camera more slowly to your {direction}."
        if scan.status is ScanStatus.PAUSED:
            return f"Continue turning the camera slowly {direction} when ready."
        return f"Please turn the camera {direction} slowly."

    def _target_loss_text(self, target):
        # A remembered image position cannot direct recovery after camera motion.
        return f"I've lost sight of your {target}. Hold steady briefly."

    def _scan_feedback(self, scan, target, now):
        if (scan.phase not in (SearchPhase.SCAN_LEFT, SearchPhase.SCAN_RIGHT)
                or not scan.scan.direction or scan.status is ScanStatus.HOLD):
            return
        key = (scan.phase, scan.scan.leg_id, scan.scan.retry)
        if key != self._scan_leg:
            text = ('Slowly turn the camera left.'
                    if scan.scan.direction == 'left' else 'Now turn the camera right.')
            if scan.scan.retry:
                text = f"Let's try again. Keep the camera level and slowly turn it {scan.scan.direction}."
            if self._say(text, condition=scan.condition, priority=20):
                self._scan_leg = key
                self._scan_alerts.clear()
                self._last_scan_alert_at = now
                self._last_direction = None
                self._last_direction_at = float('-inf')
            return
        if scan.status in (ScanStatus.WRONG_WAY, ScanStatus.UNKNOWN, ScanStatus.TOO_FAST, ScanStatus.PAUSED):
            if scan.status not in self._scan_alerts and now-self._last_scan_alert_at >= self.settings.scan_alert_gap:
                if self._say(self._scan_text(scan), condition=scan.condition, priority=20):
                    self._scan_alerts.add(scan.status)
                    self._last_scan_alert_at = now

    def on_snapshot(self, snapshot: ApplicationSnapshot):
        now = self.clock()
        session = snapshot.session
        active = (snapshot.state is ApplicationState.SEARCHING and session is not None
                  and session.status is SessionStatus.ACTIVE and now < session.deadline and not snapshot.errors)
        context = (session.session_id if session else None, active)
        changed = context != self._context
        new_session = context[0] != self._context[0]
        self._snapshot = snapshot
        if changed:
            self._context = context
            self.generation += 1
            self._last_direction = None
            self._last_direction_at = self._last_missing_at = self._last_loss_at = float("-inf")
            self._last_loss_state = None
            self._loss_since = None
            if new_session:
                self._ever_visible = False
            self._error_announced = False
            self._scan_leg = None
            self._scan_alerts.clear()
            self._last_scan_alert_at = float('-inf')
            self._strategy_phase = None
            self._target_loss_notice = None
            self._pending_direction = None
            self._direction_request_at = self._direction_retry_at = float('-inf')
        history = snapshot.last_target
        if (history is not None and session is not None and history.session_id == session.session_id
                and history.target == session.target):
            self._ever_visible = True
        condition, deadline = None, None
        guidance, visibility = snapshot.guidance, snapshot.visibility
        scan = snapshot.scan
        scan_valid = (active and scan is not None and scan.session_id == session.session_id and now < scan.valid_until)
        newly_acquired = scan_valid and scan.status is ScanStatus.TARGET_VISIBLE and self._strategy_phase is not SearchPhase.TARGET_ACQUIRED
        if scan_valid:
            self._strategy_phase = scan.phase
        if active:
            deadline = session.deadline
            if guidance and guidance.actionable and guidance.session_id == session.session_id and guidance.valid_until is not None and now < guidance.valid_until:
                condition, deadline = guidance.direction.value, min(deadline, guidance.valid_until)
            elif (guidance and guidance.status is GuidanceStatus.AMBIGUOUS and snapshot.batch is not None
                  and snapshot.batch.session_id == session.session_id and now - snapshot.batch.frame.captured_at < GuidanceSettings().max_position_age):
                condition, deadline = "ambiguous", min(deadline, snapshot.batch.frame.captured_at + GuidanceSettings().max_position_age)
            elif visibility and visibility.state is Visibility.UNCERTAIN:
                condition = "uncertain"
            else:
                condition = "missing"
            if scan_valid and scan.status is not ScanStatus.TARGET_VISIBLE:
                condition, deadline = scan.condition, min(deadline, scan.valid_until)
        if self.listening:
            condition, deadline = None, None
        self._condition = condition
        self.validity.update(self.generation, condition, deadline)
        self._refresh_direction_delivery()
        if diagnostic.enabled():
            key = (self._context, condition, self.listening,
                   guidance.status if guidance else None, scan.status if scan else None)
            if key != self._diagnostic_feedback:
                diagnostic.event('feedback', 'feedback_state', session.session_id if session else None,
                    important=False, condition=condition, valid_until=deadline, listening=self.listening,
                    suppression_reason='voice_listening' if self.listening else
                        'unchanged_direction' if guidance and guidance.actionable and
                        self._last_direction == guidance.direction else None,
                    presentation_state=('idle' if not active else 'guiding' if condition in {'left', 'center', 'right'}
                        else 'target_lost' if scan_valid and scan.phase is SearchPhase.TARGET_LOST
                        else 'target_acquired' if scan_valid and scan.phase is SearchPhase.TARGET_ACQUIRED else 'searching'),
                    guidance_status=guidance.status.value if guidance else None)
                self._diagnostic_feedback = key
        if snapshot.errors:
            if not self._error_announced:
                camera = any(e.stage.startswith("camera") for e in snapshot.errors)
                text = ("I'm having trouble accessing or releasing the camera. Please check the camera connection."
                        if camera else "The search system encountered a problem. Please restart HomeSense after checking the devices.")
                self._say(text, priority=100)
                self._error_announced = True
            return
        if not active:
            terminal = (snapshot.outcome_session_id, snapshot.outcome)
            if (snapshot.cleanup_complete and snapshot.outcome is not None and terminal != self._terminal
                    and session is not None and snapshot.outcome_session_id == session.session_id):
                self._terminal = terminal
                target = session.target.value if session else "object"
                exhausted = {
                    'motion_unconfirmed': "I couldn't confirm camera movement. This search attempt has ended.",
                    'no_fresh_results': "Search results were too old to continue. This search attempt has ended.",
                    'no_progress': "Not enough scan progress. This search attempt has ended.",
                }.get(snapshot.search_reason, f"Sorry, I couldn't locate your {target} in the views we checked.")
                if self._ever_visible:
                    exhausted = f"This search attempt has ended. I located your {target} earlier, but lost sight of it."
                timeout = (f"Search time is up. I located your {target} during this search."
                           if self._ever_visible else
                           f"Search time is up. I couldn't locate your {target} in the views checked.")
                texts = {SearchOutcome.CANCELLED: "Search cancelled.",
                         SearchOutcome.COMPLETED: "Great. Search completed.",
                         SearchOutcome.EXHAUSTED: exhausted,
                         SearchOutcome.TIMED_OUT: timeout}
                if snapshot.outcome in texts:
                    self._say(texts[snapshot.outcome], priority=90)
            return
        if self.listening:
            return
        if changed:
            self._say(f"Okay, I'm looking for your {session.target.value}.")
        if scan_valid and scan.status is not ScanStatus.TARGET_VISIBLE:
            if scan.phase is SearchPhase.TARGET_LOST and self._target_loss_notice != scan.condition:
                if self._say(self._target_loss_text(session.target.value), condition=scan.condition, priority=40):
                    self._target_loss_notice = scan.condition
                    self._last_direction = None
                    self._pending_direction = None
                    self._direction_request_at = float('-inf')
            self._scan_feedback(scan, session.target.value, now)
            return
        if visibility and visibility.state is Visibility.VISIBLE:
            self._ever_visible = True
            self._last_loss_state = None
        target = session.target.value
        if condition in {"left", "center", "right", "ambiguous"}:
            self._loss_since = None
            if newly_acquired:
                self._last_direction_at = float('-inf')
            pending_same = self._pending_direction is not None and self._pending_direction.condition == condition
            last_attempt = max(self._last_direction_at,
                               self._direction_request_at if self._pending_direction else float('-inf'))
            if (condition != self._last_direction and not pending_same and now >= self._direction_retry_at
                    and (newly_acquired or now-last_attempt >= self.settings.direction_cooldown)):
                text = ("I can see multiple possible objects. I can't tell which one you're looking for."
                        if condition == "ambiguous" else self._direction_text(target, condition))
                if condition == 'ambiguous':
                    # Ambiguity has no direction; retain its existing paced policy.
                    if self._say(text, condition=condition, priority=20):
                        self._last_direction, self._last_direction_at = condition, now
                else:
                    self._say(text, condition=condition, priority=60 if newly_acquired else 10, fresh_guidance=True)
        else:
            # Brief stale/weak intervals invalidate playback, not the last
            # announcement. Recovery in the same direction is not a new message.
            if self._ever_visible:
                # A position-age gap alone is not confirmed object loss. Require
                # persistent non-visible evidence before a recovery can reannounce.
                if visibility and visibility.state is Visibility.VISIBLE:
                    self._loss_since = None
                    return
                if self._loss_since is None:
                    self._loss_since = now
                if now - self._loss_since < self.settings.loss_delay:
                    return
                if condition != self._last_loss_state and now - self._last_loss_at >= self.settings.loss_cooldown:
                    text = (f"I can't confirm your {target} anymore. Slowly adjust your camera view."
                            if condition == "uncertain" else f"I've lost sight of your {target}. Slowly scan the area again.")
                    if self._say(text, condition=condition, priority=20):
                        self._last_loss_at, self._last_loss_state = now, condition
                        self._last_direction = None  # Recovery after announced loss is meaningful.
                        self._pending_direction = None
                        self._direction_request_at = float('-inf')
            elif now - session.started_at >= self.settings.no_detection_delay and self._last_missing_at == float('-inf'):
                if self._say(f"I can't see your {target} yet. Slowly scan your camera view from left to right.", condition=condition, priority=20):
                    self._last_missing_at = now
