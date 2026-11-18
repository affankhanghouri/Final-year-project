"""Deterministic progress, strategy and feedback policies; no hardware or weights."""

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from homesense.application import ApplicationState, SearchOutcome
from homesense.audio.speech_output import SpeechValidity
from homesense.detector import Detection
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.state import InteractionEvent, InteractionEventKind
from homesense.motion import MotionEvidence, MotionKind
from homesense.search_strategy import ScanSettings, ScanProgressTracker, ScanStatus, SearchPhase, SearchStrategyController
from homesense.session import SessionStatus
from homesense.guidance import GuidanceStatus
from homesense.stability import Visibility
from test_voice_interaction import SnapshotFactory, Clock


class StrategyFixture(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = SnapshotFactory(self.clock)
        self.session = self.factory.session
        # Fast, no-retry profile isolates the original progression contracts.
        # Separate recovery tests below exercise the real runtime defaults.
        self.settings = ScanSettings(initial_observe_seconds=.4, response_grace=0,
                                     max_retries=0, max_leg_seconds=15, uncertain_after=0)
        self.strategy = SearchStrategyController(self.session, self.settings)
        self.frame_id = 0

    def motion(self, offset=0.0, kind=MotionKind.STATIONARY, epoch=1, sid=None):
        return MotionEvidence(sid or self.session.session_id, self.frame_id, self.clock(), kind, epoch, offset)

    def update(self, *, offset=0, kind=MotionKind.STATIONARY, epoch=1, detections=(), visible=False,
               motion=True, age=0, batch_sid=None):
        self.frame_id += 1
        snap = self.factory.make(visibility=Visibility.VISIBLE if visible else Visibility.UNCONFIRMED,
                                 status=GuidanceStatus.DIRECTION if visible else GuidanceStatus.UNCONFIRMED)
        frame = replace(snap.batch.frame, frame_id=self.frame_id, captured_at=self.clock()-age)
        batch = replace(snap.batch, frame=frame, detections=detections, session_id=batch_sid or self.session.session_id)
        view = snap.visibility
        guide = replace(snap.guidance, frame_id=frame.frame_id, captured_at=frame.captured_at)
        m = self.motion(offset, kind, epoch) if motion else None
        result = self.strategy.update(self.session, batch, view, guide, m, self.clock())
        return result, replace(snap, batch=batch, guidance=guide, scan=result, motion=m)

    def begin(self):
        for _ in range(4):
            self.update()
            self.clock.now += .15
        result, snap = self.update()
        self.assertIs(result.phase, SearchPhase.SCAN_LEFT)
        return result, snap


class ProgressTests(StrategyFixture):
    def test_net_progress_reverse_motion_and_no_duplicate_credit(self):
        tracker = ScanProgressTracker(self.settings)
        tracker.begin('left', 100)
        self.frame_id = 1
        tracker.observe(self.motion(), 100)
        self.frame_id = 2
        value = tracker.observe(self.motion(-.2, MotionKind.LEFT), 100.2)
        self.assertGreater(value.progress, .5)
        duplicate = tracker.observe(self.motion(-.2, MotionKind.LEFT), 100.3)
        self.assertEqual(value.progress, duplicate.progress)
        self.frame_id = 3
        reverse = tracker.observe(self.motion(-.1, MotionKind.RIGHT), 100.4)
        self.assertLess(reverse.progress, value.progress)
        self.assertFalse(reverse.completed_left)

    def test_unknown_and_epoch_change_do_not_bridge_unseen_motion(self):
        self.begin()
        self.clock.now += .15
        value, _ = self.update(offset=-.2, kind=MotionKind.LEFT)
        self.assertGreater(value.scan.progress, 0)
        self.clock.now += .15
        value, _ = self.update(kind=MotionKind.UNKNOWN, epoch=2)
        self.assertEqual(value.scan.progress, 0)
        self.clock.now += .15
        value, _ = self.update(offset=-1, kind=MotionKind.LEFT, epoch=2)
        self.assertEqual(value.scan.progress, 0)
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)

    def test_wrong_way_requires_sustained_distinct_samples(self):
        self.begin()
        self.clock.now += .2
        first, _ = self.update(offset=.1, kind=MotionKind.RIGHT)
        self.assertIsNot(first.status, ScanStatus.WRONG_WAY)
        self.clock.now += .25
        self.update(offset=.15, kind=MotionKind.RIGHT)
        self.clock.now += .25
        third, _ = self.update(offset=.2, kind=MotionKind.RIGHT)
        self.assertIs(third.status, ScanStatus.WRONG_WAY)
        self.assertEqual(third.scan.progress, 0)

    def test_stationary_and_unreliable_motion_never_complete_a_leg(self):
        for kind in (MotionKind.STATIONARY, MotionKind.UNKNOWN, MotionKind.TOO_FAST):
            self.setUp()
            self.begin()
            for _ in range(35):
                self.clock.now += .25
                value, _ = self.update(kind=kind)
            self.assertFalse(value.scan.completed_left)
            self.assertFalse(value.scan.attempted_right)
            self.assertIs(value.phase, SearchPhase.SEARCH_EXHAUSTED)

    def test_repeated_partial_reversals_cannot_keep_searching_beyond_leg_budget(self):
        self.begin()
        for index in range(34):
            self.clock.now += .5
            offset, kind = (-.2, MotionKind.LEFT) if index % 2 == 0 else (-.1, MotionKind.RIGHT)
            value, _ = self.update(offset=offset, kind=kind)
        self.assertIs(value.phase, SearchPhase.SEARCH_EXHAUSTED)
        self.assertFalse(value.scan.completed_left)
        self.assertFalse(value.scan.attempted_right)
        self.assertEqual(value.reason, 'no_progress')


class StrategyTests(StrategyFixture):
    def test_real_optical_flow_sequence_drives_both_legs_without_timer_completion(self):
        import cv2
        import numpy as np
        from homesense.camera import Frame
        from homesense.motion import CameraMotionEstimator
        from homesense.perception import DetectionBatch
        from homesense.stability import VisibilitySnapshot
        estimator = CameraMotionEstimator()
        gray = np.random.default_rng(91).integers(0, 256, (240, 1600), dtype=np.uint8)
        panorama = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        position = 600
        phases, wrong_seen = set(), False
        for index in range(110):
            if index > 4:
                # First deliberately turn the wrong way, then correct left,
                # finally scan right past the original view.
                if index < 10:
                    position += 6
                else:
                    position += 6 if self.strategy.phase is SearchPhase.SCAN_RIGHT else -6
            self.clock.now += .15
            frame = Frame(index+1, self.clock(), panorama[:, position:position+320].copy())
            motion = estimator.observe(frame, self.session.session_id)
            batch = DetectionBatch(self.session.session_id, frame, self.clock(), 0, ())
            view = VisibilitySnapshot(self.session.session_id, self.session.target, Visibility.UNCONFIRMED,
                                      None, frame.captured_at, frame.frame_id)
            value = self.strategy.update(self.session, batch, view, None, motion, self.clock())
            phases.add(value.phase)
            wrong_seen |= value.status is ScanStatus.WRONG_WAY
            if value.phase is SearchPhase.SEARCH_EXHAUSTED:
                break
        self.assertTrue(wrong_seen)
        self.assertIn(SearchPhase.SCAN_LEFT, phases)
        self.assertIn(SearchPhase.SCAN_RIGHT, phases)
        self.assertEqual(value.reason, 'local_limit')
        self.assertTrue(value.scan.completed_left)
        self.assertTrue(value.scan.completed_right)

    def test_initial_observation_requires_real_frames_and_time(self):
        self.clock.now += 6
        result = self.strategy.update(self.session, None, None, None, None, self.clock())
        self.assertIs(result.phase, SearchPhase.OBSERVE_CURRENT_VIEW)
        value, _ = self.update()
        self.assertIs(value.phase, SearchPhase.OBSERVE_CURRENT_VIEW)
        self.clock.now += .1
        self.update()
        self.clock.now += .4
        value, _ = self.update()
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)

    def test_high_fps_observation_eventually_accumulates_a_time_span(self):
        self.clock.now += 5
        for _ in range(60):
            value, _ = self.update()
            self.clock.now += .01
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)

    def test_left_then_right_exhaustion_requires_measured_progress(self):
        self.begin()
        self.clock.now += .15
        value, _ = self.update(offset=-.36, kind=MotionKind.LEFT)
        self.assertIs(value.phase, SearchPhase.SCAN_RIGHT)
        self.assertTrue(value.scan.completed_left)
        self.assertEqual(value.scan.progress, 0)
        self.clock.now += .15
        value, _ = self.update(offset=.36, kind=MotionKind.RIGHT)
        self.assertIs(value.phase, SearchPhase.SEARCH_EXHAUSTED)
        self.assertTrue(value.scan.completed_right)
        self.assertEqual(value.reason, 'local_limit')
        self.clock.now += .1
        value, _ = self.update(detections=(Detection(0, 'phone', .9, (1, 1, 5, 5)),), visible=True)
        self.assertIs(value.phase, SearchPhase.SEARCH_EXHAUSTED)

    def test_stale_inference_does_not_credit_movement_or_claim_target_absence(self):
        self.begin()
        self.clock.now += self.settings.inactivity_seconds+.1
        value, _ = self.update(offset=-1, kind=MotionKind.LEFT, age=1)
        self.assertEqual(value.reason, 'no_fresh_results')
        self.assertFalse(value.scan.completed_left)

    def test_only_requested_target_can_pause_or_acquire(self):
        wrong = (Detection(1, 'keys', .99, (1, 1, 5, 5)), Detection(2, 'wallet', .99, (1, 1, 5, 5)))
        for _ in range(5):
            value, _ = self.update(detections=wrong)
            self.clock.now += .15
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)
        self.assertIsNot(value.status, ScanStatus.HOLD)
        positive = (Detection(0, 'phone', .9, (1, 1, 5, 5)),)
        value, _ = self.update(detections=positive, visible=True)
        self.assertIs(value.phase, SearchPhase.TARGET_ACQUIRED)
        self.assertIs(self.session.status, SessionStatus.ACTIVE)

    def test_inconsistent_visibility_cannot_promote_wrong_class_or_weak_detection(self):
        for detections in ((Detection(1, 'keys', .99, (1, 1, 5, 5)),),
                           (Detection(0, 'phone', .2, (1, 1, 5, 5)),)):
            result, _ = self.update(detections=detections, visible=True)
            self.assertIsNot(result.phase, SearchPhase.TARGET_ACQUIRED)
            self.assertIsNot(result.status, ScanStatus.HOLD)

    def test_brief_target_loss_holds_then_recovers_without_scan_oscillation(self):
        self.begin()
        positive = (Detection(0, 'phone', .9, (1, 1, 5, 5)),)
        self.update(detections=positive, visible=True)
        self.clock.now += .2
        held, _ = self.update()
        self.assertIs(held.status, ScanStatus.HOLD)
        self.assertIs(held.phase, SearchPhase.TARGET_ACQUIRED)
        self.clock.now += .2
        recovered, _ = self.update(detections=positive, visible=True)
        self.assertIs(recovered.phase, SearchPhase.TARGET_ACQUIRED)

    def test_acquisition_interrupts_each_stage_for_each_canonical_target(self):
        for target in ('phone', 'keys', 'wallet'):
            for phase in (SearchPhase.OBSERVE_CURRENT_VIEW, SearchPhase.SCAN_LEFT, SearchPhase.SCAN_RIGHT):
                with self.subTest(target=target, phase=phase):
                    self.setUp()
                    self.factory.session = self.factory.sessions.start(target, 60)
                    self.session = self.factory.session
                    self.strategy = SearchStrategyController(self.session, self.settings)
                    if phase is not SearchPhase.OBSERVE_CURRENT_VIEW:
                        self.begin()
                    if phase is SearchPhase.SCAN_RIGHT:
                        self.clock.now += .15
                        value, _ = self.update(offset=-.36, kind=MotionKind.LEFT)
                        self.assertIs(value.phase, phase)
                    value, _ = self.update(detections=(Detection(0, target, .9, (1, 1, 5, 5)),), visible=True)
                    self.assertIs(value.phase, SearchPhase.TARGET_ACQUIRED)
                    self.assertIs(self.session.status, SessionStatus.ACTIVE)

    def test_stale_and_wrong_session_motion_cannot_finish_scan(self):
        self.begin()
        snap = self.factory.make()
        for motion in (replace(self.motion(-1, MotionKind.LEFT), captured_at=99), self.motion(-1, MotionKind.LEFT, sid=uuid4())):
            value = self.strategy.update(self.session, snap.batch, snap.visibility, snap.guidance, motion, self.clock())
            self.assertEqual(value.scan.progress, 0)
            self.assertIs(value.status, ScanStatus.UNKNOWN)

    def test_acquisition_does_not_consume_scan_budget_and_real_loss_resets_progress(self):
        self.begin()
        positive = (Detection(0, 'phone', .9, (1, 1, 5, 5)),)
        self.update(detections=positive, visible=True)
        for _ in range(10):
            self.clock.now += 1
            self.update(detections=positive, visible=True)
        for _ in range(5):
            self.clock.now += .25
            value, _ = self.update()
        self.assertIs(value.phase, SearchPhase.TARGET_ACQUIRED)
        self.assertIs(value.status, ScanStatus.HOLD)
        for _ in range(40):
            self.clock.now += .25
            value, _ = self.update()
            if value.phase is SearchPhase.SCAN_LEFT:
                break
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)
        self.assertLess(value.scan.leg_seconds, 1)

    def test_cancel_timeout_and_new_session_are_inactive(self):
        self.begin()
        snap = self.factory.make()
        for session in (replace(self.session, status=SessionStatus.CANCELLED), replace(self.session, session_id=uuid4()),
                        replace(self.session, deadline=self.clock())):
            value = self.strategy.update(session, snap.batch, snap.visibility, snap.guidance, self.motion(), self.clock())
            self.assertIs(value.phase, SearchPhase.INACTIVE)


class GuidedFeedbackTests(StrategyFixture):
    def setUp(self):
        super().setUp()
        self.messages = []
        output = Mock()
        output.submit.side_effect = lambda message: self.messages.append(message) or True
        self.validity = SpeechValidity(clock=self.clock)
        self.feedback = FeedbackManager(output, self.validity, clock=self.clock)

    def test_scan_instruction_once_then_repeat_and_sustained_correction_once(self):
        _, snap = self.begin()
        for _ in range(20):
            self.feedback.on_snapshot(snap)
        self.assertEqual(sum('turn the camera left' in m.text for m in self.messages), 1)
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertIn('turn the camera left', self.messages[-1].text)
        self.clock.now += 4
        snap = replace(snap, scan=replace(snap.scan, status=ScanStatus.WRONG_WAY, valid_until=self.clock()+.5))
        for _ in range(10):
            self.feedback.on_snapshot(snap)
        self.assertEqual(sum(m.text == 'Please turn the camera left slowly.' for m in self.messages), 2)

    def test_target_acquisition_invalidates_scan_and_outranks_start_ack(self):
        _, snap = self.begin()
        self.feedback.on_snapshot(snap)
        old = self.messages[-1]
        self.clock.now += .15
        _, target = self.update(detections=(Detection(0, 'phone', .9, (1, 1, 5, 5)),), visible=True)
        self.feedback.on_snapshot(target)
        self.assertFalse(self.validity.allows(old))
        self.assertEqual(self.messages[-1].text, 'Phone on your left.')
        self.assertEqual(self.messages[-1].priority, 60)

    def test_single_candidate_does_not_chop_scan_speech_or_announce_target(self):
        _, snap = self.begin()
        self.feedback.on_snapshot(snap)
        old = self.messages[-1]
        count = len(self.messages)
        self.clock.now += .15
        value, held = self.update(detections=(Detection(0, 'phone', .9, (1, 1, 5, 5)),))
        self.assertIs(value.status, ScanStatus.HOLD)
        self.feedback.on_snapshot(held)
        self.assertTrue(self.validity.allows(old))
        self.assertEqual(len(self.messages), count)
        self.assertFalse(any(message.text == 'Phone on your left.' for message in self.messages))

    def test_exhaustion_only_announced_after_successful_cleanup(self):
        _, snap = self.begin()
        self.feedback.on_snapshot(snap)
        count = len(self.messages)
        stopped = replace(snap, state=ApplicationState.STOPPING, outcome=SearchOutcome.EXHAUSTED,
                          outcome_session_id=self.session.session_id, cleanup_complete=False)
        self.feedback.on_snapshot(stopped)
        self.assertEqual(len(self.messages), count)
        self.feedback.on_snapshot(replace(stopped, state=ApplicationState.IDLE, cleanup_complete=True))
        self.assertIn('views we checked', self.messages[-1].text)
        self.assertEqual(self.messages[-1].priority, 90)

    def test_unfinished_attempt_is_not_announced_as_a_completed_scan(self):
        _, snap = self.begin()
        self.feedback.on_snapshot(snap)
        stopped = replace(snap, state=ApplicationState.IDLE, outcome=SearchOutcome.EXHAUSTED,
                          outcome_session_id=self.session.session_id, cleanup_complete=True, search_reason='no_progress')
        self.feedback.on_snapshot(stopped)
        self.assertIn('Not enough scan progress', self.messages[-1].text)
        self.assertNotIn('views we checked', self.messages[-1].text)


class CompositionTests(unittest.TestCase):
    def test_new_flag_preserves_existing_positional_runtime_settings(self):
        from pathlib import Path
        from homesense.config import RuntimeSettings
        settings = RuntimeSettings(Path('models/best.pt'), 1, 12.0)
        self.assertEqual(settings.search_timeout, 12.0)
        self.assertFalse(settings.guided_scan)

    def test_runtime_factory_enables_motion_only_for_explicit_flag(self):
        from homesense.config import RuntimeSettings
        from homesense.runtime import perception_factory
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch('homesense.runtime.ApplicationCoordinator') as factory, \
                 patch('homesense.detector.YoloDetector') as detector:
                perception_factory(RuntimeSettings(guided_scan=enabled))()
                detector.assert_called_once()
                self.assertEqual('motion' in factory.call_args.kwargs, enabled)

    def test_cli_accepts_guided_scan_without_opening_any_hardware_in_test(self):
        from scripts import run_homesense
        runtime = Mock(failed=False)
        runtime.step.side_effect = KeyboardInterrupt
        with patch('sys.argv', ['run_homesense.py', '--guided-scan', '--text', '--console-feedback']), \
             patch.object(run_homesense, 'compose', return_value=runtime) as compose:
            self.assertEqual(run_homesense.main(), 0)
            self.assertTrue(compose.call_args.args[0].guided_scan)

    def test_exhaustion_status_is_distinct_and_deadline_wins(self):
        clock = Clock()
        factory = SnapshotFactory(clock)
        sessions, session = factory.sessions, factory.session
        self.assertFalse(sessions.exhaust(uuid4()))
        self.assertTrue(sessions.exhaust(session.session_id))
        self.assertIs(sessions.current.status, SessionStatus.EXHAUSTED)
        self.assertFalse(sessions.complete(session.session_id))
        new = sessions.start('phone', 1)
        clock.now += 1
        self.assertFalse(sessions.exhaust(new.session_id))
        self.assertIs(sessions.current.status, SessionStatus.TIMED_OUT)
