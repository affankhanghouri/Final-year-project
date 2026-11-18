"""Perspective camera surrogates and human-paced recovery; no physical devices."""

from dataclasses import replace
import math
import unittest
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from homesense.camera import Frame
from homesense.motion import CameraMotionEstimator, MotionKind, MotionSettings
from homesense.search_strategy import ScanSettings, SearchPhase, SearchStrategyController, ScanStatus
from homesense.audio.speech_output import SpeechStats, SpeechValidity
from homesense.diagnostics import RuntimeMetrics
from homesense.interaction.feedback_manager import FeedbackManager
from test_search_strategy import StrategyFixture


def furnished_view(seed=771):
    """Uneven, sparse scene edges and plain regions, not uniform random texture."""
    rng = np.random.default_rng(seed)
    image = np.full((480, 640, 3), 125, np.uint8)
    for _ in range(150):
        x, y = map(int, rng.integers([0, 0], [620, 460]))
        color = tuple(map(int, rng.integers(25, 230, 3)))
        cv2.rectangle(image, (x, y), (x+15, y+18), color, -1)
    return image


def pan_view(image, yaw, *, roll=0, exposure=1.0, noise=0):
    """Known pinhole perspective rotation, with deterministic exposure/sensor noise.

    Degrees are test construction inputs, never estimator outputs or calibration.
    Positive scene yaw corresponds to apparent LEFT camera motion.
    """
    a, b = math.radians(yaw), math.radians(roll)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
    k = np.array([[520., 0, 320], [0, 520., 240], [0, 0, 1]])
    view = cv2.warpPerspective(image, k @ rz @ ry @ np.linalg.inv(k), (640, 480),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(125,)*3)
    variation = np.random.default_rng(90).normal(0, noise, view.shape)
    return np.clip(view.astype(float)*exposure+variation, 0, 255).astype(np.uint8)


class PerspectiveMotionTests(unittest.TestCase):
    def test_realistic_perspective_pan_keeps_consensus_without_lowering_threshold(self):
        image = furnished_view()
        for yaw in (-3, -2, -1, 1, 2, 3):
            with self.subTest(yaw=yaw):
                estimator, sid = CameraMotionEstimator(), uuid4()
                estimator.observe(Frame(1, 100, image), sid)
                value = estimator.observe(Frame(2, 100.15, pan_view(image, yaw)), sid)
                self.assertIs(value.kind, MotionKind.LEFT if yaw > 0 else MotionKind.RIGHT)
                self.assertGreaterEqual(value.support_fraction, MotionSettings().min_inlier_fraction)
                self.assertAlmostEqual(value.step, -520*math.tan(math.radians(yaw))/640, delta=.002)

    def test_pan_with_small_roll_exposure_change_noise_and_foreground(self):
        image = furnished_view()
        for yaw in (-2, 2):
            estimator, sid = CameraMotionEstimator(), uuid4()
            estimator.observe(Frame(1, 100, image), sid)
            view = pan_view(image, yaw, roll=.6, exposure=.88, noise=1)
            view[170:310, 230:410] = 30  # Local foreground is not the motion reference.
            value = estimator.observe(Frame(2, 100.15, view), sid)
            self.assertIs(value.kind, MotionKind.LEFT if yaw > 0 else MotionKind.RIGHT)

    def test_brief_blur_recovers_only_by_tracking_against_last_good_image(self):
        image, sid = furnished_view(), uuid4()
        estimator = CameraMotionEstimator()
        estimator.observe(Frame(1, 100, image), sid)
        first = estimator.observe(Frame(2, 100.15, pan_view(image, 1)), sid)
        blur = estimator.observe(Frame(3, 100.30, np.full_like(image, 125)), sid)
        recovered = estimator.observe(Frame(4, 100.45, pan_view(image, 3)), sid)
        self.assertIs(blur.kind, MotionKind.UNKNOWN)
        self.assertEqual(blur.offset, first.offset)
        self.assertEqual(recovered.epoch, first.epoch)
        self.assertIs(recovered.kind, MotionKind.LEFT)
        self.assertAlmostEqual(recovered.offset, -520*math.tan(math.radians(3))/640, delta=.003)
        later = estimator.observe(Frame(5, 101.1, pan_view(image, -3)), sid)
        self.assertIs(later.kind, MotionKind.UNKNOWN)
        self.assertGreater(later.epoch, recovered.epoch)
        self.assertEqual(later.offset, 0)

    def test_opposite_rotation_during_blur_is_measured_as_reverse_not_extra_credit(self):
        image, sid = furnished_view(), uuid4()
        estimator = CameraMotionEstimator()
        estimator.observe(Frame(1, 100, image), sid)
        first = estimator.observe(Frame(2, 100.15, pan_view(image, 1)), sid)
        estimator.observe(Frame(3, 100.3, np.full_like(image, 125)), sid)
        reverse = estimator.observe(Frame(4, 100.45, image), sid)
        self.assertIs(reverse.kind, MotionKind.RIGHT)
        self.assertLess(abs(reverse.offset), abs(first.offset)/5)

    def test_sensor_noise_does_not_create_movement_in_stationary_view(self):
        image, sid = furnished_view(), uuid4()
        estimator = CameraMotionEstimator()
        estimator.observe(Frame(1, 100, image), sid)
        value = estimator.observe(Frame(2, 100.15, pan_view(image, 0, noise=2)), sid)
        self.assertIs(value.kind, MotionKind.STATIONARY)
        self.assertEqual(value.offset, 0)

    def test_sustained_slow_pan_exceeds_same_displacement_floor_without_losing_every_step(self):
        image = furnished_view()
        for sign in (-1, 1):
            estimator, sid = CameraMotionEstimator(), uuid4()
            estimator.observe(Frame(1, 100, image), sid)
            kinds = []
            for index in range(1, 26):
                # Half a degree/second: below the old instantaneous speed gate.
                yaw = sign*index*.075
                value = estimator.observe(Frame(index+1, 100+index*.15, pan_view(image, yaw, noise=.5)), sid)
                kinds.append(value.kind)
            self.assertIn(MotionKind.LEFT if sign > 0 else MotionKind.RIGHT, kinds)
            self.assertNotIn(MotionKind.RIGHT if sign > 0 else MotionKind.LEFT, kinds)
            self.assertAlmostEqual(value.offset, -520*math.tan(math.radians(yaw))/640, delta=.004)

    def test_small_sensor_jitter_and_noise_do_not_accumulate_false_slow_progress(self):
        image, sid = furnished_view(), uuid4()
        estimator = CameraMotionEstimator()
        estimator.observe(Frame(1, 100, image), sid)
        for index in range(1, 21):
            value = estimator.observe(Frame(index+1, 100+index*.15,
                pan_view(image, .015*math.sin(index), noise=2)), sid)
            self.assertIs(value.kind, MotionKind.STATIONARY)
            self.assertEqual(value.offset, 0)


class RecoveryStrategyTests(StrategyFixture):
    def setUp(self):
        super().setUp()
        self.settings = ScanSettings(initial_observe_seconds=.4)
        self.strategy = SearchStrategyController(self.session, self.settings)

    def busy_update(self, busy, *, kind=MotionKind.UNKNOWN):
        self.frame_id += 1
        snap = self.factory.make()
        frame = replace(snap.batch.frame, frame_id=self.frame_id, captured_at=self.clock())
        batch = replace(snap.batch, frame=frame, detections=())
        return self.strategy.update(self.session, batch, None, None,
            self.motion(kind=kind), self.clock(), instruction_busy=busy)

    def test_reported_thirteen_second_end_is_replaced_by_recovery_opportunity(self):
        self.begin()
        for _ in range(28):
            self.clock.now += .15
            value = self.busy_update(True)  # About four seconds of queued/playing TTS.
        self.assertEqual(value.scan.leg_seconds, 0)
        for _ in range(7):
            self.clock.now += 1
            value = self.busy_update(False)
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)
        self.assertEqual(value.scan.progress, 0)
        # Continue uncertainty until a bounded retry is offered, not exhaustion.
        for _ in range(5):
            self.clock.now += 1
            value = self.busy_update(False)
            if value.status is ScanStatus.RECOVERING:
                break
        self.assertIs(value.status, ScanStatus.RECOVERING)
        self.assertEqual(value.scan.retry, 1)
        self.assertFalse(value.scan.completed_left)
        self.clock.now += .15
        self.update(kind=MotionKind.LEFT, offset=-.1)
        self.clock.now += .15
        value, _ = self.update(kind=MotionKind.LEFT, offset=-.5)
        self.assertIs(value.phase, SearchPhase.SCAN_RIGHT)

    def test_brief_same_epoch_unknown_freezes_progress_and_recovery_uses_net_offset(self):
        self.begin()
        self.clock.now += .15
        value, _ = self.update(offset=-.15, kind=MotionKind.LEFT)
        credit = value.scan.progress
        self.clock.now += .15
        unknown, _ = self.update(offset=-.15, kind=MotionKind.UNKNOWN)
        self.assertEqual(unknown.scan.progress, credit)
        self.assertFalse(unknown.scan.completed_left)
        self.clock.now += .15
        recovered, _ = self.update(offset=-.2, kind=MotionKind.LEFT)
        self.assertAlmostEqual(recovered.scan.progress, .2/self.settings.left_extent)
        self.assertIs(recovered.phase, SearchPhase.SCAN_LEFT)

    def test_long_unknown_or_new_epoch_discards_unverifiable_credit(self):
        for epoch in (1, 2):
            with self.subTest(epoch=epoch):
                self.setUp()
                self.begin()
                self.clock.now += .15
                self.update(offset=-.2, kind=MotionKind.LEFT)
                for _ in range(5):
                    self.clock.now += .15
                    value, _ = self.update(kind=MotionKind.UNKNOWN, epoch=epoch)
                self.assertEqual(value.scan.progress, 0)
                self.clock.now += .15
                recovered, _ = self.update(offset=-.8, kind=MotionKind.LEFT, epoch=epoch)
                self.assertEqual(recovered.scan.progress, 0)
                self.assertFalse(recovered.scan.completed_left)

    def test_response_grace_and_speech_never_credit_unknown_or_block_session_deadline(self):
        self.begin()
        for _ in range(10):
            self.clock.now += 1
            value = self.busy_update(True)
        self.assertEqual(value.scan.leg_seconds, 0)
        self.assertEqual(value.scan.progress, 0)
        self.clock.now = self.session.deadline
        value = self.busy_update(True)
        self.assertIs(value.phase, SearchPhase.INACTIVE)

    def test_uncertainty_has_exactly_two_retries_then_bounded_unsuccessful_end(self):
        self.begin()
        retries = set()
        for _ in range(150):
            self.clock.now += .25
            value = self.busy_update(False)
            if value.status is ScanStatus.RECOVERING:
                retries.add(value.scan.retry)
            if value.phase is SearchPhase.SEARCH_EXHAUSTED:
                break
        self.assertEqual(retries, {1, 2})
        self.assertIs(value.phase, SearchPhase.SEARCH_EXHAUSTED)
        self.assertEqual(value.reason, 'motion_unconfirmed')
        self.assertFalse(value.scan.completed_left)
        self.assertFalse(value.scan.attempted_right)
        self.assertLess(value.scan.leg_seconds, 31)

    def test_wrong_direction_after_recovery_never_completes_left(self):
        self.begin()
        for _ in range(12):
            self.clock.now += .25
            self.update(kind=MotionKind.UNKNOWN)
        for index in range(1, 9):
            self.clock.now += .15
            value, _ = self.update(kind=MotionKind.RIGHT, offset=index*.05)
        self.assertIs(value.status, ScanStatus.WRONG_WAY)
        self.assertEqual(value.scan.progress, 0)
        self.assertFalse(value.scan.completed_left)
        self.assertFalse(value.scan.attempted_right)

    def test_perspective_pan_journey_with_wrong_way_blur_and_real_left_right_progress(self):
        from homesense.perception import DetectionBatch
        panorama = np.hstack([furnished_view(seed) for seed in (71, 771, 19)])
        k = np.array([[520., 0, 960], [0, 520., 240], [0, 0, 1]])
        crop = np.array([[1., 0, -640], [0, 1., 0], [0, 0, 1.]])
        estimator = CameraMotionEstimator()
        yaw, wrong_seen, blurred = 0., False, False
        phases = set()
        for index in range(220):
            if self.strategy.phase is SearchPhase.SCAN_LEFT:
                yaw += .6 if wrong_seen else -.6
            elif self.strategy.phase is SearchPhase.SCAN_RIGHT:
                yaw -= .6
            a = math.radians(yaw)
            rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            image = cv2.warpPerspective(panorama, crop @ k @ rotation @ np.linalg.inv(k), (640, 480))
            image = np.clip(image.astype(float)*(1+.02*math.sin(index))+
                np.random.default_rng(index).normal(0, 1, image.shape), 0, 255).astype(np.uint8)
            if wrong_seen and not blurred and yaw > 5:
                image.fill(125)
                blurred = True
            self.clock.now += .15
            frame = Frame(index+1, self.clock(), image)
            motion = estimator.observe(frame, self.session.session_id)
            batch = DetectionBatch(self.session.session_id, frame, self.clock(), 0, ())
            value = self.strategy.update(self.session, batch, None, None, motion, self.clock())
            phases.add(value.phase)
            wrong_seen |= value.status is ScanStatus.WRONG_WAY
            if value.phase is SearchPhase.SEARCH_EXHAUSTED:
                break
        self.assertTrue(wrong_seen)
        self.assertTrue(blurred)
        self.assertIn(SearchPhase.SCAN_RIGHT, phases)
        self.assertEqual(value.reason, 'local_limit')
        self.assertTrue(value.scan.completed_left)
        self.assertTrue(value.scan.completed_right)

    def test_settings_bound_retries_and_validate_response_and_continuity(self):
        for kwargs in ({'max_retries': -1}, {'max_retries': 4}, {'max_retries': True},
                       {'response_grace': -1}, {'continuity_grace': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ScanSettings(**kwargs)

    def test_short_unknown_does_not_request_a_correction_but_sustained_loss_does(self):
        self.begin()
        self.clock.now += 2.1
        self.update()
        self.clock.now += .15
        value, _ = self.update(kind=MotionKind.UNKNOWN)
        self.assertIs(value.status, ScanStatus.SCANNING)
        self.assertEqual(value.reason, 'brief_tracking_loss')
        for _ in range(8):
            self.clock.now += .15
            value, _ = self.update(kind=MotionKind.UNKNOWN)
        self.assertIs(value.status, ScanStatus.UNKNOWN)
        self.assertIs(value.phase, SearchPhase.SCAN_LEFT)


class RecoveryPresentationTests(StrategyFixture):
    def test_recovery_instruction_is_once_per_retry_and_old_session_is_invalidated(self):
        messages = []
        output = Mock()
        output.submit.side_effect = lambda message: messages.append(message) or True
        validity = SpeechValidity(clock=self.clock)
        feedback = FeedbackManager(output, validity, clock=self.clock)
        _, snapshot = self.begin()
        feedback.on_snapshot(snapshot)
        old = messages[-1]
        initial_count = len(messages)
        retry = replace(snapshot.scan.scan, retry=1)
        snapshot = replace(snapshot, scan=replace(snapshot.scan, status=ScanStatus.RECOVERING, scan=retry))
        for _ in range(20):
            feedback.on_snapshot(snapshot)
        self.assertEqual(sum("Let's try again" in message.text for message in messages), 1)
        self.assertEqual(len(messages), initial_count+1)
        recovery = messages[-1]
        self.factory.session = self.factory.sessions.start('wallet', 60)
        feedback.on_snapshot(self.factory.make())
        self.assertFalse(validity.allows(old))
        self.assertFalse(validity.allows(recovery))

    def test_periodic_diagnostics_expose_rejection_reason_direction_and_budgets(self):
        _, snapshot = self.begin()
        motion = replace(self.motion(kind=MotionKind.UNKNOWN), reason='conflicting scene motion',
                         features=100, inliers=56, pair_seconds=.15)
        snapshot = replace(snapshot, motion=motion)
        app = Mock()
        app.latest_frame.return_value = None
        runtime = SimpleNamespace(interaction=SimpleNamespace(snapshot=snapshot, coordinator=app),
                                  speech=SimpleNamespace(stats=SpeechStats()))
        metrics = RuntimeMetrics(5, clock=self.clock, cpu_clock=lambda: 0)
        self.assertIsNone(metrics.sample(runtime))
        self.clock.now += 5
        with self.assertLogs('homesense.diagnostics', level='INFO') as logs:
            values = metrics.sample(runtime)
            self.assertIsNone(metrics.sample(runtime))
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(values['motion_direction'], 'unknown')
        self.assertEqual(values['motion_reason'], 'conflicting scene motion')
        self.assertEqual(values['motion_support'], .56)
        self.assertEqual(values['motion_pair_ms'], 150)
        for field in ('scan_idle_s', 'scan_effort_s', 'scan_response_s', 'scan_retry', 'scan_decision'):
            self.assertIn(field, values)
