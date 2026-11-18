import threading
import time
import unittest

import numpy as np

from homesense.camera import Frame, LatestFrameBuffer
from homesense.detector import Detection
from homesense.perception import DetectionWorker, FreshnessSettings
from homesense.session import SessionManager


class Clock:
    def __init__(self):
        self.now = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def set(self, now):
        with self.lock:
            self.now = now


class FakeDetector:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail = False

    def predict(self, image):
        self.calls.append(int(image[0, 0, 0]))
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError("test detector was not released")
        if self.fail:
            raise RuntimeError("fake inference failure")
        return (Detection(0, "phone", 0.9, (0, 0, 1, 1)),
                Detection(1, "keys", 0.8, (0, 0, 1, 1)))


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sessions = SessionManager(clock=self.clock)
        self.session = self.sessions.start("phone", 10)
        self.buffer = LatestFrameBuffer()
        self.detector = FakeDetector()
        self.worker = DetectionWorker(self.buffer, self.detector, self.sessions, self.session, clock=self.clock)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.detector.release.set()
        self.buffer.close()
        self.assertTrue(self.worker.stop(1))

    def publish(self, frame_id, captured_at=100.0):
        self.buffer.publish(Frame(frame_id, captured_at, np.full((2, 2, 3), frame_id, dtype=np.uint8)))

    def until(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("worker did not reach expected state")
            time.sleep(0.001)

    def begin_inference(self):
        self.publish(1)
        self.worker.start()
        self.assertTrue(self.detector.entered.wait(1))

    def test_newest_frame_after_slow_inference_no_duplicate_and_target_filter(self):
        self.begin_inference()
        self.publish(2)
        self.publish(3)
        self.detector.release.set()
        self.until(lambda: self.worker.stats.published == 2)
        batch = self.worker.latest_result()
        self.assertEqual(self.detector.calls, [1, 3])
        self.assertEqual(batch.frame.frame_id, 3)
        self.assertEqual(batch.session_id, self.session.session_id)
        self.assertEqual([d.class_name for d in batch.detections], ["phone"])
        self.assertEqual(self.worker.stats.skipped, 1)
        self.assertIsNone(self.worker.latest_result(after_id=3))
        batch.frame.image[:] = 9
        self.assertTrue(np.all(self.worker.latest_result().frame.image == 3))
        self.buffer.close()
        self.until(lambda: self.worker.finished)
        self.assertEqual(self.detector.calls, [1, 3])

    def test_stale_frame_boundary_is_rejected_before_inference(self):
        self.clock.set(100.25)
        self.publish(1)
        self.worker.start()
        self.until(lambda: self.worker.stats.stale_frames == 1)
        self.assertEqual(self.detector.calls, [])
        self.assertIsNone(self.worker.latest_result())

    def test_pre_session_and_future_timestamps_are_rejected(self):
        self.publish(1, 99.99)
        self.worker.start()
        self.until(lambda: self.worker.stats.stale_frames == 1)
        self.publish(2, 101)
        self.until(lambda: self.worker.stats.stale_frames == 2)
        self.assertEqual(self.detector.calls, [])

    def test_result_at_age_limit_is_rejected_after_inference(self):
        self.begin_inference()
        self.clock.set(101)
        self.detector.release.set()
        self.until(lambda: self.worker.stats.rejected_results == 1)
        self.assertIsNone(self.worker.latest_result())
        self.assertAlmostEqual(self.worker.stats.last_inference_seconds, 1.0)
        self.assertAlmostEqual(self.worker.stats.last_result_age, 1.0)

    def test_cancel_replace_and_timeout_suppress_inflight_results(self):
        for action in ("cancel", "replace", "timeout"):
            with self.subTest(action=action):
                buffer = LatestFrameBuffer()
                detector = FakeDetector()
                session = self.sessions.start("phone", 10)
                worker = DetectionWorker(buffer, detector, self.sessions, session, clock=self.clock)
                buffer.publish(Frame(1, self.clock(), np.zeros((2, 2, 3), dtype=np.uint8)))
                worker.start()
                try:
                    self.assertTrue(detector.entered.wait(1))
                    if action == "cancel":
                        self.sessions.cancel(session.session_id)
                    elif action == "replace":
                        self.sessions.start("keys", 10)
                    else:
                        self.clock.set(session.deadline)
                    detector.release.set()
                    self.until(lambda: worker.finished)
                    self.assertIsNone(worker.latest_result())
                    self.assertEqual(worker.stats.rejected_results, 1)
                finally:
                    detector.release.set()
                    buffer.close()
                    self.assertTrue(worker.stop(1))

    def test_published_result_expires_or_is_cancelled_before_consumption(self):
        self.begin_inference()
        self.clock.set(100.1)
        self.detector.release.set()
        self.until(lambda: self.worker.stats.published == 1)
        batch = self.worker.latest_result()
        self.assertAlmostEqual(batch.inference_seconds, 0.1)
        self.assertAlmostEqual(batch.capture_to_result_seconds, 0.1)
        self.clock.set(101)
        self.assertIsNone(self.worker.latest_result())
        self.sessions.cancel(self.session.session_id)
        self.assertIsNone(self.worker.latest_result())

    def test_cancellation_rejects_a_still_fresh_published_result(self):
        self.begin_inference()
        self.clock.set(100.1)
        self.detector.release.set()
        self.until(lambda: self.worker.stats.published == 1)
        self.assertIsNotNone(self.worker.latest_result())
        self.assertTrue(self.sessions.cancel(self.session.session_id))
        # Clock stays unchanged: cancellation alone must invalidate this result.
        self.assertIsNone(self.worker.latest_result())

    def test_stop_suppresses_inflight_result_and_reports_bounded_join(self):
        self.begin_inference()
        self.assertFalse(self.worker.stop(0.01))
        self.detector.release.set()
        self.assertTrue(self.worker.stop(1))
        self.assertIsNone(self.worker.latest_result())
        self.assertIsNone(self.worker.error)
        with self.assertRaises(RuntimeError):
            self.worker.start()

    def test_closed_buffer_exits_and_prediction_errors_are_reported(self):
        self.buffer.close()
        self.worker.start()
        self.until(lambda: self.worker.finished)
        self.assertEqual(self.detector.calls, [])

    def test_prediction_error_clears_results_and_finishes(self):
        self.detector.fail = True
        self.detector.release.set()
        self.begin_inference()
        self.until(lambda: self.worker.finished)
        self.assertIsInstance(self.worker.error, RuntimeError)
        self.assertIsNone(self.worker.latest_result())

    def test_invalid_age_limits(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FreshnessSettings(max_frame_age=value)

    def test_already_replaced_session_never_starts_inference(self):
        self.publish(1)
        self.sessions.start("keys", 10)
        self.worker.start()
        self.until(lambda: self.worker.finished)
        self.assertEqual(self.detector.calls, [])

    def test_empty_detection_batch_does_not_complete_search(self):
        self.detector.predict = lambda image: ()
        self.publish(1)
        self.worker.start()
        self.until(lambda: self.worker.stats.published == 1)
        self.assertEqual(self.worker.latest_result().detections, ())
        self.assertTrue(self.sessions.is_eligible(self.session.session_id))


if __name__ == "__main__":
    unittest.main()
