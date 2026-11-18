"""Real optical-flow algorithms on deterministic synthetic images; no devices."""

from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import cv2
import numpy as np

from homesense.camera import Frame, LatestFrameBuffer
from homesense.motion import CameraMotionEstimator, CameraMotionWorker, MotionKind, MotionSettings
from homesense.session import SessionManager


def texture(seed=172):
    gray = np.random.default_rng(seed).integers(0, 256, (240, 320), dtype=np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def translate(image, dx, dy=0):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (image.shape[1], image.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


class MotionEstimatorTests(unittest.TestCase):
    def pair(self, old, new, dt=.15, settings=MotionSettings()):
        estimator, sid = CameraMotionEstimator(settings), uuid4()
        initial = estimator.observe(Frame(1, 100, old), sid)
        result = estimator.observe(Frame(2, 100+dt, new), sid)
        return initial, result

    def test_scene_motion_has_opposite_camera_direction(self):
        image = texture()
        for dx, direction in ((6, MotionKind.LEFT), (-6, MotionKind.RIGHT)):
            with self.subTest(dx=dx):
                first, result = self.pair(image, translate(image, dx))
                self.assertIs(first.kind, MotionKind.UNKNOWN)
                self.assertIs(result.kind, direction)
                self.assertAlmostEqual(result.step, -dx/320, delta=.001)
                self.assertGreater(result.support_fraction, .9)

    def test_stationary_is_not_progress(self):
        _, result = self.pair(texture(), texture())
        self.assertIs(result.kind, MotionKind.STATIONARY)
        self.assertEqual(result.offset, 0)

    def test_blank_and_blurred_views_are_unknown(self):
        images = [np.zeros((240, 320, 3), dtype=np.uint8), cv2.GaussianBlur(texture(), (51, 51), 15)]
        for image in images:
            _, result = self.pair(image, translate(image, 6))
            self.assertIs(result.kind, MotionKind.UNKNOWN)
            self.assertEqual(result.offset, 0)

    def test_moving_central_object_is_not_camera_rotation(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[60:180, 80:240] = texture()[60:180, 80:240]
        _, result = self.pair(image, translate(image, 6))
        self.assertIs(result.kind, MotionKind.UNKNOWN)

    def test_static_background_with_moving_foreground_stays_stationary_or_unknown(self):
        image = texture()
        moving = image.copy()
        moving[60:180, 80:240] = translate(image, 12)[60:180, 80:240]
        _, result = self.pair(image, moving)
        self.assertIn(result.kind, (MotionKind.STATIONARY, MotionKind.UNKNOWN))

    def test_vertical_roll_zoom_and_conflicting_scene_are_not_horizontal_claims(self):
        image = texture()
        cases = [translate(image, 0, 15), texture(993),
                 cv2.warpAffine(image, cv2.getRotationMatrix2D((160, 120), 8, 1), (320, 240)),
                 cv2.warpAffine(image, cv2.getRotationMatrix2D((160, 120), 0, 1.12), (320, 240))]
        for changed in cases:
            _, result = self.pair(image, changed)
            self.assertIs(result.kind, MotionKind.UNKNOWN)

    def test_fast_reliable_translation_is_rejected_without_credit(self):
        first, result = self.pair(texture(), translate(texture(), 10), dt=.02)
        self.assertIs(result.kind, MotionKind.TOO_FAST)
        self.assertGreater(result.epoch, first.epoch)
        self.assertEqual(result.offset, 0)

    def test_gap_duplicate_and_new_session_break_continuity(self):
        estimator, sid = CameraMotionEstimator(), uuid4()
        image = texture()
        estimator.observe(Frame(1, 100, image), sid)
        good = estimator.observe(Frame(2, 100.15, translate(image, 6)), sid)
        bad = estimator.observe(Frame(2, 101, translate(image, 12)), sid)
        self.assertIs(bad.kind, MotionKind.UNKNOWN)
        self.assertGreater(bad.epoch, good.epoch)
        reset = estimator.observe(Frame(1, 102, image), uuid4())
        self.assertIs(reset.kind, MotionKind.UNKNOWN)
        self.assertEqual(reset.offset, 0)

    def test_settings_reject_invalid_budgets(self):
        for kwargs in ({'sample_period': 0}, {'min_features': 200}, {'min_grid_cells': 13}, {'max_age': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MotionSettings(**kwargs)


class MotionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.buffer = LatestFrameBuffer()
        self.sessions = SessionManager()
        self.session = self.sessions.start('phone', 10)
        self.worker = CameraMotionWorker(self.buffer, self.sessions, self.session,
                                         MotionSettings(sample_period=.01))
        self.addCleanup(self.stop)

    def stop(self):
        self.buffer.close()
        self.assertTrue(self.worker.stop(1))

    def until(self, predicate):
        deadline = time.monotonic()+1
        while not predicate():
            if time.monotonic() > deadline:
                self.fail('motion worker did not reach expected state')
            time.sleep(.002)

    def test_worker_samples_same_slot_and_cancellation_hides_late_results(self):
        self.worker.start()
        self.buffer.publish(Frame(1, time.monotonic(), texture()))
        self.until(lambda: self.worker.latest() is not None)
        first = self.worker.latest()
        copy = self.buffer.wait_for_newer(0, 0)
        self.assertEqual(copy.frame_id, first.frame_id)
        self.sessions.cancel(self.session.session_id)
        self.assertIsNone(self.worker.latest())

    def test_failed_start_never_joins_unstarted_thread(self):
        with patch.object(threading.Thread, 'start', side_effect=RuntimeError('cannot start')):
            with self.assertRaises(RuntimeError):
                self.worker.start()
        self.assertTrue(self.worker.stop())

    def test_stale_frame_not_processed_and_no_busy_spin(self):
        with patch('homesense.motion.CameraMotionEstimator') as constructor:
            self.worker.start()
            self.buffer.publish(Frame(1, time.monotonic()-2, texture()))
            time.sleep(.05)
            constructor.return_value.observe.assert_not_called()
            self.assertIsNone(self.worker.latest())

    def test_native_failure_is_exposed_and_thread_finishes(self):
        with patch('homesense.motion.CameraMotionEstimator') as constructor:
            constructor.return_value.observe.side_effect = RuntimeError('flow failure')
            self.worker.start()
            self.buffer.publish(Frame(1, time.monotonic(), texture()))
            self.until(lambda: self.worker.finished)
            self.assertIsInstance(self.worker.error, RuntimeError)
            self.assertIsNone(self.worker.latest())
