"""Bounded recovery and truthful feedback; historical positions never authorize speech."""
from dataclasses import replace
import unittest
from unittest.mock import Mock
from uuid import uuid4

from homesense.application import ApplicationState, SearchOutcome, ApplicationError
from homesense.audio.speech_output import SpeechValidity
from homesense.detector import Detection
from homesense.guidance import GuidanceStatus
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.state import InteractionEvent, InteractionEventKind
from homesense.motion import MotionKind
from homesense.search_strategy import SearchPhase, ScanStatus
from homesense.session import SessionStatus
from homesense.stability import Visibility
from test_search_strategy import StrategyFixture


class TargetRecoveryTests(StrategyFixture):
    def setUp(self):
        super().setUp()
        self.messages = []
        self.output = Mock()
        self.output.submit.side_effect = lambda message: self.messages.append(message) or True
        self.validity = SpeechValidity(clock=self.clock)
        self.feedback = FeedbackManager(self.output, self.validity, clock=self.clock)

    def acquired(self):
        result, snap = self.update(detections=(Detection(0, 'phone', .9, (1, 1, 5, 5)),), visible=True)
        self.feedback.on_snapshot(snap)
        self.assertIs(result.phase, SearchPhase.TARGET_ACQUIRED)
        return snap

    def lost(self):
        for _ in range(8):
            self.clock.now += .25
            result, snap = self.update(kind=MotionKind.UNKNOWN)
            self.feedback.on_snapshot(snap)
        self.assertIs(result.phase, SearchPhase.TARGET_LOST)
        return snap

    def test_brief_fluctuation_retains_position_history_without_repeating_direction(self):
        original = self.acquired()
        self.clock.now += .2
        result, snap = self.update(kind=MotionKind.RIGHT, offset=.4)
        self.feedback.on_snapshot(snap)
        self.assertEqual(result.last_target, original.guidance)
        self.assertIs(result.status, ScanStatus.HOLD)
        self.assertEqual(result.scan.progress, 0)
        self.assertFalse(self.validity.allows(self.messages[-1]))
        self.acquired()
        self.assertEqual(sum(m.text == 'Phone on your left.' for m in self.messages), 1)
        self.assertFalse(any('lost sight' in m.text for m in self.messages))

    def test_loss_instruction_once_reacquisition_preempts_it_and_announces_fresh_position(self):
        self.begin()
        self.acquired()
        lost = self.lost()
        loss_message = self.messages[-1]
        self.assertIn('Hold steady', loss_message.text)
        self.assertNotIn('left', loss_message.text)
        for _ in range(5):
            self.feedback.on_snapshot(lost)
        self.assertEqual(sum('lost sight' in m.text for m in self.messages), 1)
        self.acquired()
        self.assertFalse(self.validity.allows(loss_message))
        self.assertEqual(sum(m.text == 'Phone on your left.' for m in self.messages), 2)
        self.assertEqual(self.messages[-1].priority, 60)
        self.assertIs(self.session.status, SessionStatus.ACTIVE)

    def test_repeat_while_lost_never_repeats_old_scan_direction(self):
        self.begin()
        self.acquired()
        self.lost()
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertIn('Hold steady', self.messages[-1].text)
        self.clock.now += 1
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertIn("don't have a current confirmed direction", self.messages[-1].text)

    def test_recovery_budget_survives_unconfirmed_candidate_flicker_then_observes_new_view(self):
        self.acquired()
        self.lost()
        for _ in range(24):
            self.clock.now += .25
            result, _ = self.update(detections=(Detection(0, 'phone', .9, (1, 1, 5, 5)),))
            if result.phase is SearchPhase.OBSERVE_CURRENT_VIEW:
                break
        self.assertIs(result.phase, SearchPhase.OBSERVE_CURRENT_VIEW)
        self.assertEqual(result.reason, 'target_recovery_failed')
        # Original session clock cannot skip the new observation period.
        self.clock.now += .01
        result, _ = self.update()
        self.assertIs(result.phase, SearchPhase.OBSERVE_CURRENT_VIEW)
        for _ in range(5):
            self.clock.now += .15
            result, _ = self.update()
        self.assertIs(result.phase, SearchPhase.SCAN_LEFT)
        self.assertEqual(result.scan.progress, 0)
        self.assertIsNotNone(result.last_target)

    def test_speech_pauses_recovery_effort_but_wall_bound_and_session_deadline_win(self):
        self.acquired()
        snap = self.lost()
        for _ in range(11):
            self.clock.now += 1
            result = self.strategy.update(self.session, None, None, None, None, self.clock(), instruction_busy=True)
            self.assertIs(result.phase, SearchPhase.TARGET_LOST)
        self.assertLess(result.recovery_seconds, 1)
        self.clock.now += 1
        result = self.strategy.update(self.session, None, None, None, None, self.clock(), instruction_busy=True)
        self.assertIs(result.phase, SearchPhase.OBSERVE_CURRENT_VIEW)
        self.clock.now = self.session.deadline
        result = self.strategy.update(self.session, snap.batch, snap.visibility, snap.guidance, snap.motion, self.clock())
        self.assertIs(result.phase, SearchPhase.INACTIVE)

    def test_mismatched_frame_cannot_acquire_or_restore_historical_direction(self):
        snap = self.acquired()
        self.lost()
        self.clock.now += .1
        fresh = self.factory.make()
        result = self.strategy.update(self.session, replace(fresh.batch, detections=snap.batch.detections),
            fresh.visibility, replace(fresh.guidance, frame_id=999), None, self.clock())
        self.assertIs(result.phase, SearchPhase.TARGET_LOST)
        self.assertEqual(result.last_target, snap.guidance)

    def test_deadline_during_recovery_cannot_reacquire_from_late_positive_evidence(self):
        acquired = self.acquired()
        self.lost()
        self.clock.now = self.session.deadline
        result, _ = self.update(detections=acquired.batch.detections, visible=True)
        self.assertIs(result.phase, SearchPhase.INACTIVE)
        self.assertEqual(result.last_target, acquired.guidance)
        self.clock.now += .1
        result, _ = self.update(detections=acquired.batch.detections, visible=True)
        self.assertIs(result.phase, SearchPhase.INACTIVE)

    def test_timeout_with_acquisition_history_is_truthful_even_without_feedback_history(self):
        snap = self.acquired()
        self.feedback = FeedbackManager(self.output, self.validity, clock=self.clock)
        terminal = replace(snap, state=ApplicationState.IDLE, session=replace(self.session, status=SessionStatus.TIMED_OUT),
            outcome=SearchOutcome.TIMED_OUT, outcome_session_id=self.session.session_id, cleanup_complete=True,
            batch=None, visibility=None, guidance=None, scan=None, last_target=snap.guidance)
        self.feedback.on_snapshot(terminal)
        self.assertIn('located your phone during this search', self.messages[-1].text)
        self.assertNotIn("couldn't", self.messages[-1].text)
        self.assertNotIn('left', self.messages[-1].text)
        self.assertEqual(self.messages[-1].priority, 90)

    def test_timeout_without_acquisition_and_wrong_session_history_cannot_claim_found(self):
        snap = self.factory.make(status=GuidanceStatus.UNCONFIRMED)
        terminal = replace(snap, state=ApplicationState.IDLE, outcome=SearchOutcome.TIMED_OUT,
            outcome_session_id=self.session.session_id, cleanup_complete=True,
            last_target=replace(snap.guidance, session_id=uuid4()))
        self.feedback.on_snapshot(terminal)
        self.assertIn("couldn't locate", self.messages[-1].text)

    def test_cancel_complete_error_outrank_recovery_and_new_session_forgets_history(self):
        for outcome in (SearchOutcome.CANCELLED, SearchOutcome.COMPLETED):
            with self.subTest(outcome=outcome):
                self.setUp()
                acquired = self.acquired()
                lost = self.lost()
                old = self.messages[-1]
                stopped = replace(lost, state=ApplicationState.IDLE, outcome=outcome,
                    outcome_session_id=self.session.session_id, cleanup_complete=True, last_target=acquired.guidance)
                self.feedback.on_snapshot(stopped)
                self.assertFalse(self.validity.allows(old))
                self.assertEqual(self.messages[-1].priority, 90)
                self.feedback.on_snapshot(replace(stopped, state=ApplicationState.FAILED,
                    errors=(ApplicationError('camera.release', RuntimeError('release failed')),)))
                self.assertEqual(self.messages[-1].priority, 100)
                self.factory.session = self.factory.sessions.start('wallet', 60)
                new = self.factory.make(status=GuidanceStatus.UNCONFIRMED, visibility=Visibility.UNCONFIRMED)
                self.feedback.on_snapshot(new)
                self.feedback.on_snapshot(replace(new, state=ApplicationState.IDLE, outcome=SearchOutcome.TIMED_OUT,
                    outcome_session_id=new.session.session_id, cleanup_complete=True))
                self.assertIn("couldn't locate your wallet", self.messages[-1].text)
