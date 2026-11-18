"""Voice-to-real-session/worker/feedback journeys with deterministic simulated I/O."""

from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from homesense.application import ApplicationCoordinator, ApplicationState, SearchOutcome
from homesense.audio.events import VoiceEvent, VoiceEventKind, VoiceEvents
from homesense.audio.input import PlaybackGate, VoiceWorker
from homesense.audio.speech_output import SpeechValidity, SpeechWorker
from homesense.config import AudioSettings
from homesense.detector import Detection
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.manager import InteractionManager
from homesense.motion import CameraMotionWorker, MotionEvidence, MotionKind, MotionSettings
from homesense.preview import LaptopPreview
from homesense.runtime import HomeSenseRuntime
from homesense.search_strategy import ScanSettings, ScanStatus, SearchPhase
from homesense.session import SessionStatus
from test_laptop_runtime import fake_cv
from .test_voice_runtime import Clock, AudioSource, Recognizer, RecordingSpeech


class Scene:
    def __init__(self):
        self.lock = threading.Lock()
        self.offset = 0.0
        self.kind = MotionKind.STATIONARY
        self.target = None
        self.active = 0
        self.maximum = 0
        self.opens = 0
        self.detector_calls = 0
        self.releases = 0
        self.epoch = 1

    def source(self, camera_id):
        with self.lock:
            self.active += 1
            self.opens += 1
            self.maximum = max(self.maximum, self.active)
        scene = self
        class Source:
            def isOpened(self):
                return True

            def read(self):
                time.sleep(.002)
                image = np.zeros((60, 120, 3), dtype=np.uint8)
                with scene.lock:
                    image[0, 0] = [round(128+scene.offset*100), list(MotionKind).index(scene.kind), scene.epoch]
                return True, image

            def release(self):
                with scene.lock:
                    scene.active -= 1
                    scene.releases += 1
        return Source()

    def predict(self, image):
        with self.lock:
            self.detector_calls += 1
            target = self.target
        wrong = Detection(1, 'keys', .99, (60, 5, 70, 30))
        return (wrong,) if target is None else (wrong, Detection(0, target, .9, (5, 5, 15, 30)))

    def motion(self, frame, sid):
        offset, kind, epoch = frame.image[0, 0]
        return MotionEvidence(sid, frame.frame_id, frame.captured_at, list(MotionKind)[int(kind)],
                              int(epoch), (int(offset)-128)/100)


class GuidedJourneyTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.scene = Clock(), Scene()
        self.settings = ScanSettings(initial_observe_seconds=.3, min_observation_span=.1,
            left_extent=.1, right_extent=.2, max_leg_seconds=6, inactivity_seconds=4,
            paused_after=.5, wrong_way_seconds=.2, wrong_way_samples=2, response_grace=0, max_retries=0, uncertain_after=0)
        self.patcher = patch('homesense.motion.CameraMotionEstimator')
        estimator = self.patcher.start().return_value
        estimator.observe.side_effect = self.scene.motion
        self.addCleanup(self.patcher.stop)
        self.events = VoiceEvents()
        validity = SpeechValidity(clock=self.clock)
        playback = PlaybackGate(tail=0, clock=self.clock)
        self.backend = RecordingSpeech()
        self.speech = SpeechWorker(self.backend, validity, playback, listening_gap=0, clock=self.clock)
        feedback = FeedbackManager(self.speech, validity, clock=self.clock)
        self.app = None
        def create():
            self.app = ApplicationCoordinator(self.scene, source_factory=self.scene.source,
                motion=MotionSettings(sample_period=.005), scan=self.settings,
                scan_speech_busy=lambda: self.speech.busy,
                shutdown_timeout=.1, clock=self.clock)
            return self.app
        self.factory = Mock(side_effect=create)
        self.manager = InteractionManager(self.factory, feedback)
        self.source = AudioSource()
        self.voice = VoiceWorker(self.source, Recognizer(), self.events, playback,
                                 AudioSettings(echo_tail=0), clock=self.clock)
        self.runtime = HomeSenseRuntime(self.manager, feedback, self.speech, self.events,
            voice=self.voice, clock=self.clock, speech_shutdown_grace=.1)
        self.preview = None
        self.addCleanup(self.cleanup)
        self.runtime.start()
        self.until(lambda: not self.speech.busy)

    def cleanup(self):
        if self.preview:
            self.preview.close()
        self.assertTrue(self.runtime.close())
        self.assertEqual(self.scene.active, 0)
        self.assertTrue(self.source.closed.is_set())
        self.assertFalse(any(t.name == 'camera-motion' and t.is_alive() for t in threading.enumerate()))

    def until(self, predicate):
        deadline = time.monotonic()+1.5
        while True:
            self.runtime.step()
            if self.preview:
                self.preview.update(self.manager.snapshot, self.manager.coordinator)
            if predicate():
                return
            if time.monotonic() > deadline:
                self.fail(f'expected journey state not reached: {self.manager.snapshot}')
            time.sleep(.002)

    def say(self, text):
        self.until(lambda: not self.speech.busy)
        time.sleep(.12)  # Existing playback epoch drains before genuinely new speech.
        self.source.queue.put(text.encode())

    def advance(self, offset=None, kind=None, amount=.15):
        with self.scene.lock:
            if offset is not None:
                self.scene.offset = offset
            if kind is not None:
                self.scene.kind = kind
            expected_offset, expected_kind = self.scene.offset, self.scene.kind
        self.clock.set(self.clock()+amount)
        minimum = self.clock()
        self.until(lambda: self.manager.snapshot.batch is not None and self.manager.snapshot.motion is not None
                   and self.manager.snapshot.batch.frame.captured_at >= minimum
                   and self.manager.snapshot.motion.captured_at >= minimum
                   and self.manager.snapshot.motion.kind is expected_kind
                   and abs(self.manager.snapshot.motion.offset-expected_offset) < .011)

    def begin(self):
        self.say('find my phone')
        self.until(lambda: self.app is not None and self.manager.snapshot.batch is not None)
        for _ in range(4):
            self.advance()
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.SCAN_LEFT)

    def test_voice_left_right_exhaustion_spoken_result_cleanup_and_new_search(self):
        self.preview = LaptopPreview(cv=fake_cv(), clock=self.clock)
        self.begin()
        self.until(lambda: self.backend.count('turn the camera left') == 1)
        self.assertFalse(self.manager.snapshot.batch.detections)  # Wrong-class YOLO output filtered.
        self.advance(-.06, MotionKind.LEFT)
        self.advance(-.12, MotionKind.LEFT)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.SCAN_RIGHT)
        self.advance(0, MotionKind.RIGHT)
        # Final step can finish resources before advance's active-evidence predicate.
        with self.scene.lock:
            self.scene.offset, self.scene.kind = .12, MotionKind.RIGHT
        self.clock.set(self.clock()+.15)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.EXHAUSTED and self.manager.snapshot.cleanup_complete)
        self.until(lambda: self.backend.count('views we checked') == 1)
        self.assertIs(self.manager.snapshot.session.status, SessionStatus.EXHAUSTED)
        self.assertEqual(self.scene.active, 0)
        self.assertEqual(self.scene.maximum, 1)
        old_session = self.manager.snapshot.session.session_id
        self.scene.offset, self.scene.kind = 0, MotionKind.STATIONARY
        self.say('find my wallet')
        self.until(lambda: self.manager.snapshot.session.session_id != old_session and self.manager.snapshot.scan is not None)
        self.assertFalse(self.manager.snapshot.scan.scan.attempted_right)
        self.assertEqual(self.manager.snapshot.scan.scan.progress, 0)
        self.assertEqual(self.scene.maximum, 1)
        self.assertEqual(self.factory.call_count, 1)

    def test_acquisition_during_scan_is_immediate_and_brief_loss_does_not_restart(self):
        self.begin()
        with self.scene.lock:
            self.scene.target = 'phone'
        for _ in range(4):
            self.advance(-.03, MotionKind.LEFT)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.TARGET_ACQUIRED)
        self.until(lambda: self.backend.count('Phone on your left') == 1)
        self.assertIs(self.manager.snapshot.session.status, SessionStatus.ACTIVE)
        self.scene.target = None
        self.advance(amount=.2)
        self.assertIs(self.manager.snapshot.scan.phase, SearchPhase.TARGET_ACQUIRED)
        self.scene.target = 'phone'
        for _ in range(4):
            self.advance()
        self.assertEqual(self.backend.count('turn the camera left'), 1)
        self.say('I found it')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.COMPLETED)
        self.assertEqual(self.scene.active, 0)

    def test_sustained_wrong_way_then_correct_motion_repeat_and_cancel(self):
        self.begin()
        for offset in (.03, .06, .09):
            self.advance(offset, MotionKind.RIGHT)
        self.until(lambda: self.manager.snapshot.scan.status is ScanStatus.WRONG_WAY)
        self.assertEqual(self.manager.snapshot.scan.scan.progress, 0)
        self.say('repeat the direction')
        self.until(lambda: self.backend.count('Please turn the camera left slowly') >= 1)
        self.advance(.03, MotionKind.LEFT)
        self.assertIsNot(self.manager.snapshot.scan.status, ScanStatus.WRONG_WAY)
        self.say('cancel')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.CANCELLED)
        self.assertEqual(self.scene.active, 0)

    def test_automatic_wrong_way_correction_is_sustained_and_not_repeated(self):
        self.settings = replace(self.settings, inactivity_seconds=5.5)
        self.begin()
        for index in range(1, 24):
            self.advance(index*.02, MotionKind.RIGHT)
        self.until(lambda: self.backend.count('Please turn the camera left slowly.') == 1)
        for index in range(24, 27):
            self.advance(index*.02, MotionKind.RIGHT)
        self.assertEqual(self.backend.count('Please turn the camera left slowly.'), 1)
        self.assertEqual(self.manager.snapshot.scan.scan.progress, 0)

    def test_unsupported_request_and_idle_do_not_load_resources(self):
        self.say('find my clothes')
        self.until(lambda: self.backend.count('only find phones') == 1)
        self.assertEqual(self.scene.opens, 0)
        self.assertEqual(self.scene.detector_calls, 0)
        self.factory.assert_not_called()

    def test_search_timeout_remains_distinct_from_exhaustion(self):
        self.begin()
        self.clock.set(self.manager.snapshot.session.deadline)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.TIMED_OUT and self.manager.snapshot.cleanup_complete)
        self.assertIsNone(self.manager.snapshot.scan)
        self.assertEqual(self.scene.active, 0)

    def test_unknown_motion_ends_bounded_attempt_without_claiming_scan_completion(self):
        self.begin()
        self.advance(kind=MotionKind.UNKNOWN)
        self.assertIs(self.manager.snapshot.scan.status, ScanStatus.UNKNOWN)
        self.assertFalse(self.manager.snapshot.scan.scan.completed_left)
        # Keep inference fresh while unknown motion persists, rather than making
        # both streams stale with one large artificial clock jump.
        for _ in range(40):
            if self.manager.snapshot.outcome is SearchOutcome.EXHAUSTED:
                break
            self.clock.set(self.clock()+.15)
            minimum = self.clock()
            self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.EXHAUSTED or
                       self.manager.snapshot.batch is not None and self.manager.snapshot.batch.frame.captured_at >= minimum)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.EXHAUSTED and self.manager.snapshot.cleanup_complete)
        self.assertEqual(self.manager.snapshot.search_reason, 'motion_unconfirmed')
        self.until(lambda: self.backend.count("couldn't confirm camera movement") == 1)
        self.assertEqual(self.scene.active, 0)

    def test_live_failure_recovers_after_delayed_user_movement_then_finishes_both_legs(self):
        self.settings = replace(self.settings, max_leg_seconds=30, inactivity_seconds=8,
                                response_grace=2, max_retries=2, uncertain_after=1)
        self.begin()
        for _ in range(14):
            self.advance(kind=MotionKind.UNKNOWN, amount=.9)
            self.assertIs(self.manager.snapshot.state, ApplicationState.SEARCHING)
            self.assertFalse(self.manager.snapshot.scan.scan.completed_left)
        self.until(lambda: self.backend.count('slowly and steadily') + self.backend.count("Let's try again") >= 1)
        self.advance(-.02, MotionKind.LEFT)
        self.advance(-.08, MotionKind.LEFT)
        before_blur = self.manager.snapshot.scan.scan.progress
        self.advance(kind=MotionKind.UNKNOWN)
        self.assertEqual(self.manager.snapshot.scan.scan.progress, before_blur)
        self.advance(-.16, MotionKind.LEFT)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.SCAN_RIGHT)
        self.advance(-.05, MotionKind.RIGHT)
        with self.scene.lock:
            self.scene.offset, self.scene.kind = .10, MotionKind.RIGHT
        self.clock.set(self.clock()+.15)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.EXHAUSTED and self.manager.snapshot.cleanup_complete)
        self.assertEqual(self.manager.snapshot.search_reason, 'local_limit')
        self.assertEqual(self.scene.maximum, 1)
        self.assertEqual(self.scene.active, 0)

    def test_actual_pending_and_playing_speech_pause_effort_but_not_inference_or_acquisition(self):
        self.settings = replace(self.settings, max_leg_seconds=30, inactivity_seconds=8,
                                response_grace=2, max_retries=2, uncertain_after=1)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = self.backend.speak
        def slow_instruction(text, should_stop):
            original(text, should_stop)
            if 'turn the camera left' in text:
                entered.set()
                while not release.wait(.002):
                    if should_stop():
                        return
        self.backend.speak = slow_instruction
        self.begin()
        self.until(entered.is_set)
        calls = self.scene.detector_calls
        # Refresh speech permits between small clock steps: a one-second jump
        # would intentionally invalidate the 0.5 s freshness permit mid-speech.
        for _ in range(95):
            self.advance(kind=MotionKind.UNKNOWN, amount=.15)
            self.assertIs(self.manager.snapshot.state, ApplicationState.SEARCHING)
            self.assertEqual(self.manager.snapshot.scan.scan.leg_seconds, 0)
            self.assertTrue(self.speech.busy)
        self.assertGreater(self.scene.detector_calls, calls)
        self.scene.target = 'phone'
        for _ in range(4):
            self.advance(-.02, MotionKind.LEFT)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.TARGET_ACQUIRED)
        self.until(lambda: self.backend.count('Phone on your left') == 1)
        self.assertGreaterEqual(self.speech.stats.cancelled, 1)
        self.assertIsNone(self.speech.error)

    def test_overall_timeout_still_wins_while_a_scan_instruction_is_playing(self):
        self.settings = replace(self.settings, response_grace=2, max_retries=2)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original = self.backend.speak
        def blocked_instruction(text, should_stop):
            original(text, should_stop)
            if 'turn the camera left' in text:
                entered.set()
                while not release.wait(.002):
                    if should_stop():
                        return
        self.backend.speak = blocked_instruction
        self.begin()
        self.until(entered.is_set)
        self.clock.set(self.manager.snapshot.session.deadline)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.TIMED_OUT and self.manager.snapshot.cleanup_complete)
        self.assertEqual(self.scene.active, 0)
        self.assertIsNone(self.speech.error)


    def acquire_phone(self):
        self.scene.target = 'phone'
        for _ in range(4):
            self.advance(amount=.15)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.TARGET_ACQUIRED)
        self.until(lambda: self.backend.count('Phone on your left') >= 1)

    def test_visible_at_deadline_ends_truthfully_and_releases_camera(self):
        self.begin()
        self.acquire_phone()
        self.clock.set(self.manager.snapshot.session.deadline-1.1)
        self.acquire_phone()
        snapshot = self.manager.snapshot
        self.assertIsNotNone(snapshot.last_target)
        self.assertEqual(snapshot.last_target.session_id, snapshot.session.session_id)
        self.assertTrue(snapshot.guidance.actionable)
        self.clock.set(snapshot.session.deadline)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.TIMED_OUT and self.manager.snapshot.cleanup_complete)
        self.until(lambda: self.backend.count('located your phone during this search') == 1)
        self.assertFalse(any("couldn't locate" in text for text in self.backend.messages))
        self.assertIsNone(self.manager.snapshot.guidance)
        self.assertEqual(self.scene.active, 0)
        self.assertIs(self.manager.snapshot.session.status, SessionStatus.TIMED_OUT)

    def test_loss_recovery_interrupts_real_speech_then_reacquires_without_rescanning(self):
        self.backend.block_left = True
        self.begin()
        self.acquire_phone()
        self.until(self.backend.entered.is_set)
        before = self.scene.detector_calls
        self.scene.target = None
        for _ in range(11):
            self.advance(kind=MotionKind.UNKNOWN, amount=.2)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.TARGET_LOST)
        self.until(lambda: self.backend.count('Hold steady') == 1)
        self.assertGreater(self.speech.stats.cancelled, 0)
        self.assertGreater(self.scene.detector_calls, before)
        self.assertEqual(self.backend.count('turn the camera left'), 1)
        self.assertEqual(self.manager.snapshot.scan.scan.progress, 0)
        self.backend.block_left = False
        self.backend.release.set()
        self.acquire_phone()
        self.until(lambda: self.backend.count('Phone on your left') == 2)
        # Two guidance attempts protect a full listening window after the echo
        # tail. Keep evidence fresh while allowing that new policy to elapse.
        for _ in range(4):
            self.advance(amount=.8)
        self.say('repeat the direction')
        self.until(lambda: self.backend.count('Phone on your left') == 3)
        self.say('I found it')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.COMPLETED)
        self.until(lambda: self.backend.count('Search completed') == 1)
        self.assertEqual(self.scene.active, 0)
        self.assertIsNone(self.speech.error)

    def test_cancel_during_loss_then_new_search_has_no_target_history_or_old_speech(self):
        self.begin()
        self.acquire_phone()
        old = self.manager.snapshot.session.session_id
        self.scene.target = None
        for _ in range(11):
            self.advance(amount=.2)
        self.until(lambda: self.manager.snapshot.scan.phase is SearchPhase.TARGET_LOST)
        self.say('cancel')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.CANCELLED)
        self.until(lambda: self.backend.count('Search cancelled') == 1)
        self.assertEqual(self.scene.active, 0)
        self.say('find my wallet')
        self.until(lambda: self.manager.snapshot.session.session_id != old and self.manager.snapshot.batch is not None)
        self.assertIsNone(self.manager.snapshot.last_target)
        self.assertIsNone(self.manager.snapshot.scan.last_target)
        self.clock.set(self.manager.snapshot.session.deadline)
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.TIMED_OUT and self.manager.snapshot.cleanup_complete)
        self.until(lambda: self.backend.count("couldn't locate your wallet") == 1)
        self.assertEqual(self.scene.maximum, 1)


class GuidedWorkerCleanupTests(unittest.TestCase):
    def test_failed_motion_start_cleans_existing_workers_and_never_joins_unstarted(self):
        scene = Scene()
        app = ApplicationCoordinator(scene, source_factory=scene.source, motion=MotionSettings(), shutdown_timeout=.2)
        original = threading.Thread.start
        def start(thread):
            if thread.name == 'camera-motion':
                raise RuntimeError('motion start failed')
            return original(thread)
        try:
            with patch.object(threading.Thread, 'start', start), self.assertLogs('homesense.application', level='ERROR'):
                result = app.start_search('phone', 10)
            self.assertTrue(result.cleanup_complete)
            self.assertTrue(result.errors)
            self.assertEqual(scene.active, 0)
            self.assertTrue(app.close().cleanup_complete)
        finally:
            app.close()

    def test_blocked_motion_refuses_replacement_and_preserves_cleanup_ownership(self):
        entered, release = threading.Event(), threading.Event()
        scene = Scene()
        def blocked(frame, sid):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test did not release motion')
            return scene.motion(frame, sid)
        app = ApplicationCoordinator(scene, source_factory=scene.source, motion=MotionSettings(), shutdown_timeout=.03)
        with patch('homesense.motion.CameraMotionEstimator') as estimator:
            estimator.return_value.observe.side_effect = blocked
            try:
                app.start_search('phone', 10)
                self.assertTrue(entered.wait(1))
                with self.assertLogs('homesense.application', level='ERROR'):
                    result = app.cancel_search(app.poll().session.session_id)
                self.assertFalse(result.cleanup_complete)
                self.assertIs(result.outcome, SearchOutcome.FAILED)
                count = scene.opens
                app.start_search('keys', 10)
                self.assertEqual(scene.opens, count)
                self.assertEqual(scene.active, 0)
                release.set()
                deadline = time.monotonic()+1
                while not app._motion.finished and time.monotonic() < deadline:
                    time.sleep(.002)
                self.assertTrue(app.close().cleanup_complete)
            finally:
                release.set()
                app.close()
