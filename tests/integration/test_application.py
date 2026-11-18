"""Real capture/inference threads with controlled sources and detector outputs."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from homesense import application
from homesense.application import ApplicationCoordinator, ApplicationState, SearchOutcome
from homesense.detector import Detection
from homesense.guidance import Direction
from homesense.session import SessionStatus
from homesense.stability import Visibility


class Clock:
    def __init__(self):
        self.value = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = value


class Cameras:
    def __init__(self):
        self.sources = []
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()
        self.block = False
        self.release_failure = None
        self.read_failure = None
        self.open_failure = None

    def __call__(self, camera_id):
        source = Source(self)
        with self.lock:
            self.sources.append(source)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        return source


class Source:
    def __init__(self, cameras):
        self.cameras = cameras
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.release_called = threading.Event()
        if not cameras.block:
            self.unblock.set()

    def isOpened(self):
        if self.cameras.open_failure:
            raise self.cameras.open_failure
        return True

    def read(self):
        self.entered.set()
        if not self.unblock.wait(2):
            raise RuntimeError("test source was not unblocked")
        # Prevent an unrealistic fake camera from spinning at full CPU.
        threading.Event().wait(0.001)
        if self.cameras.read_failure:
            raise self.cameras.read_failure
        return True, np.zeros((60, 120, 3), dtype=np.uint8)

    def release(self):
        self.release_called.set()
        if self.cameras.release_failure:
            raise self.cameras.release_failure
        with self.cameras.lock:
            self.cameras.active -= 1


class Detector:
    def __init__(self):
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.unblock.set()
        self.failure = None
        self.active = 0
        self.maximum = 0
        self.lock = threading.Lock()

    def predict(self, image):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        self.entered.set()
        try:
            if not self.unblock.wait(2):
                raise RuntimeError("test inference was not unblocked")
            if self.failure:
                raise self.failure
            return (Detection(0, "phone", 0.9, (5, 5, 15, 30)),
                    Detection(1, "keys", 0.9, (90, 5, 110, 30)))
        finally:
            with self.lock:
                self.active -= 1


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cameras = Cameras()
        self.detector = Detector()
        self.app = ApplicationCoordinator(self.detector, source_factory=self.cameras,
                                          clock=self.clock, shutdown_timeout=0.1)
        self.addCleanup(self.cleanup)
        self.logs = patch.object(application.logging.getLogger(application.__name__), "error")
        self.logs.start()
        self.addCleanup(self.logs.stop)

    def cleanup(self):
        self.detector.unblock.set()
        for source in self.cameras.sources:
            source.unblock.set()
        # Closing retries retained workers; give native thread scheduling a bound.
        deadline = time.monotonic() + 1
        while True:
            result = self.app.close()
            if self.detector.active == 0 and all(s.release_called.is_set() for s in self.cameras.sources):
                break
            if time.monotonic() >= deadline:
                self.fail("test workers did not finish cleanup")
            time.sleep(0.001)
        self.assertEqual(self.detector.active, 0)
        self.assertTrue(all(self.app._stopped.values()))

    def until(self, predicate):
        deadline = time.monotonic() + 1
        while True:
            snapshot = self.app.poll()
            if predicate(snapshot):
                return snapshot
            if time.monotonic() >= deadline:
                self.fail(f"expected state not reached: {snapshot}")
            time.sleep(0.001)

    def start(self, target="phone", timeout=10):
        result = self.app.start_search(target, timeout)
        self.assertEqual(result.state, ApplicationState.SEARCHING)
        return result.session

    def test_preview_reads_owned_latest_frames_while_inference_is_blocked(self):
        self.detector.unblock.clear()
        self.start()
        self.assertTrue(self.detector.entered.wait(1))
        first = self.app.latest_frame()
        self.assertIsNotNone(first)
        first.image[:] = 255
        self.assertFalse(np.any(self.app.latest_frame().image))
        self.until(lambda snapshot: self.app.latest_frame().frame_id > first.frame_id)
        self.assertEqual(self.cameras.maximum, 1)
        self.assertEqual(len(self.cameras.sources), 1)
        self.assertEqual(self.detector.maximum, 1)
        self.assertIsNone(self.app.poll().batch)
        self.detector.unblock.set()
        self.until(lambda snapshot: snapshot.batch is not None)
        session = self.app.poll().session
        self.app.cancel_search(session.session_id)
        self.assertIsNone(self.app.latest_frame())

    def positive(self, captured):
        self.clock.set(captured)
        return self.until(lambda s: s.batch is not None and s.batch.frame.captured_at == captured)

    def test_fresh_detection_visibility_and_guidance_without_auto_completion(self):
        session = self.start()
        self.positive(100)
        self.positive(100.25)
        result = self.positive(100.5)
        self.assertEqual(result.visibility.state, Visibility.VISIBLE)
        self.assertEqual(result.guidance.direction, Direction.LEFT)
        self.assertTrue(result.guidance.actionable)
        self.assertEqual(result.guidance.session_id, session.session_id)
        self.assertEqual([d.class_name for d in result.batch.detections], ["phone"])
        self.assertEqual(result.state, ApplicationState.SEARCHING)
        self.assertEqual(result.session.status, SessionStatus.ACTIVE)

    def test_acquisition_history_survives_headless_timeout_but_never_authorizes_guidance(self):
        session = self.start(timeout=1)
        for at in (100, 100.25, 100.5):
            snapshot = self.positive(at)
        self.assertIsNotNone(snapshot.last_target)
        self.assertEqual(snapshot.last_target.frame_id, snapshot.guidance.frame_id)
        self.clock.set(session.deadline)
        ended = self.until(lambda s: s.cleanup_complete)
        self.assertIs(ended.outcome, SearchOutcome.TIMED_OUT)
        self.assertIsNone(ended.guidance)
        self.assertIsNone(ended.batch)
        self.assertEqual(ended.last_target, snapshot.last_target)
        new = self.app.start_search('wallet', 1)
        self.assertIsNone(new.last_target)
        self.assertEqual(self.cameras.maximum, 1)

    def test_cancel_during_inference_suppresses_late_result(self):
        self.detector.unblock.clear()
        session = self.start()
        self.assertTrue(self.detector.entered.wait(1))
        worker = self.app._inference
        result = self.app.cancel_search(session.session_id)
        self.assertEqual(result.state, ApplicationState.FAILED)
        self.assertEqual(result.session.status, SessionStatus.CANCELLED)
        self.assertIsNone(result.guidance)
        self.detector.unblock.set()
        self.until(lambda s: s.cleanup_complete)
        self.assertIsNone(worker.latest_result())
        self.assertEqual(worker.stats.published, 0)

    def test_normal_cancellation_and_repeat_close(self):
        session = self.start()
        self.positive(100)
        result = self.app.cancel_search(session.session_id)
        self.assertEqual(result.state, ApplicationState.IDLE)
        self.assertEqual(result.outcome, SearchOutcome.CANCELLED)
        self.assertTrue(result.cleanup_complete)
        self.assertFalse(result.errors)
        self.assertEqual(self.app.close().state, ApplicationState.CLOSED)
        self.assertEqual(self.app.close().state, ApplicationState.CLOSED)
        with self.assertRaises(RuntimeError):
            self.app.start_search("keys", 10)

    def test_timeout_cleanup_is_nonblocking_and_expired_results_rejected(self):
        self.detector.unblock.clear()
        session = self.start(timeout=1)
        self.assertTrue(self.detector.entered.wait(1))
        worker = self.app._inference
        self.clock.set(session.deadline)
        started = time.monotonic()
        result = self.app.poll()
        self.assertLess(time.monotonic() - started, 0.05)
        self.assertEqual(result.state, ApplicationState.STOPPING)
        self.assertEqual(result.outcome, SearchOutcome.TIMED_OUT)
        self.assertEqual(result.session.status, SessionStatus.TIMED_OUT)
        self.assertIsNone(result.batch)
        self.detector.unblock.set()
        result = self.until(lambda s: s.cleanup_complete)
        self.assertEqual(result.state, ApplicationState.IDLE)
        self.assertIsNone(worker.latest_result())

    def test_replacement_releases_resources_and_resets_evidence(self):
        old = self.start()
        for value in (100, 100.25, 100.5):
            result = self.positive(value)
        self.assertTrue(result.guidance.actionable)
        old_worker = self.app._inference
        self.clock.set(100.625)
        new = self.start("keys")
        self.assertNotEqual(old.session_id, new.session_id)
        result = self.positive(100.625)
        self.assertEqual(result.visibility.state, Visibility.UNCONFIRMED)
        self.assertFalse(result.guidance.actionable)
        self.assertEqual(result.outcome, SearchOutcome.REPLACED)
        self.assertEqual(result.outcome_session_id, old.session_id)
        self.assertTrue(self.cameras.sources[0].release_called.is_set())
        self.assertIsNone(old_worker.latest_result())
        self.positive(100.875)
        result = self.positive(101.125)
        self.assertEqual(result.guidance.direction, Direction.RIGHT)
        self.assertEqual(self.cameras.maximum, 1)
        self.assertEqual(self.detector.maximum, 1)

    def test_blocked_camera_refuses_replacement_and_retains_ownership(self):
        self.cameras.block = True
        old = self.start()
        deadline = time.monotonic() + 1
        while not self.cameras.sources:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        source = self.cameras.sources[0]
        self.assertTrue(source.entered.wait(1))
        result = self.app.start_search("keys", 10)
        self.assertEqual(result.state, ApplicationState.FAILED)
        self.assertEqual(result.session.session_id, old.session_id)
        self.assertFalse(result.cleanup_complete)
        self.assertEqual(len(self.cameras.sources), 1)
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.app.start_search("wallet", 10)
        self.assertEqual(len(self.cameras.sources), 1)
        source.unblock.set()
        recovered = self.until(lambda s: s.cleanup_complete)
        self.assertEqual(recovered.state, ApplicationState.FAILED)
        self.assertTrue(recovered.errors)
        self.app.start_search("keys", 10)
        self.assertEqual(len(self.cameras.sources), 1)
        self.assertEqual(self.app.close().errors, recovered.errors)

    def test_blocked_model_refuses_replacement_even_after_eventual_cleanup(self):
        self.detector.unblock.clear()
        old = self.start()
        self.assertTrue(self.detector.entered.wait(1))
        result = self.app.start_search("keys", 10)
        self.assertEqual(result.state, ApplicationState.FAILED)
        self.assertEqual(result.session.session_id, old.session_id)
        self.assertEqual(len(self.cameras.sources), 1)
        self.detector.unblock.set()
        self.until(lambda s: s.cleanup_complete)
        self.assertEqual(self.app.start_search("keys", 10).state, ApplicationState.FAILED)
        self.assertEqual(self.detector.maximum, 1)
        self.assertEqual(len(self.cameras.sources), 1)

    def test_poll_returns_while_model_is_blocked(self):
        self.detector.unblock.clear()
        self.start()
        self.assertTrue(self.detector.entered.wait(1))
        started = time.monotonic()
        result = self.app.poll()
        self.assertLess(time.monotonic() - started, 0.05)
        self.assertEqual(result.state, ApplicationState.SEARCHING)
        self.assertIsNone(result.batch)
        self.assertEqual(self.detector.active, 1)

    def test_late_inference_exception_survives_close_retries(self):
        self.detector.unblock.clear()
        self.start()
        self.assertTrue(self.detector.entered.wait(1))
        result = self.app.close()
        self.assertFalse(result.cleanup_complete)
        failure = RuntimeError("late prediction failed")
        self.detector.failure = failure
        self.detector.unblock.set()
        result = self.until(lambda s: s.cleanup_complete)
        self.assertEqual(result.state, ApplicationState.CLOSED)
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertIn(failure, [e.exception for e in result.errors])
        self.assertEqual(self.app.close().errors, result.errors)

    def test_release_failure_remains_failed_and_refuses_reuse(self):
        self.cameras.release_failure = RuntimeError("release failed")
        session = self.start()
        self.positive(100)
        with self.assertLogs("homesense.camera", level="ERROR"):
            result = self.app.cancel_search(session.session_id)
        self.assertEqual(result.state, ApplicationState.FAILED)
        self.assertFalse(result.cleanup_complete)
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertEqual(self.app.start_search("keys", 10).state, ApplicationState.FAILED)
        self.assertEqual(self.app.close().state, ApplicationState.FAILED)
        self.assertEqual(len(self.cameras.sources), 1)

    def test_initialization_and_release_errors_are_both_reported(self):
        primary = RuntimeError("initialization failed")
        secondary = RuntimeError("release failed")
        self.cameras.open_failure = primary
        self.cameras.release_failure = secondary
        with self.assertLogs("homesense.camera", level="ERROR"):
            self.app.start_search("phone", 10)
            result = self.until(lambda s: s.errors)
        self.assertIn(primary, [e.exception for e in result.errors])
        self.assertIn(secondary, [e.exception for e in result.errors])
        self.assertFalse(result.cleanup_complete)

    def test_partial_thread_start_failure_releases_started_camera(self):
        original_start = threading.Thread.start
        failure = RuntimeError("cannot start inference thread")

        def start(thread):
            if thread.name == "detection-worker":
                raise failure
            return original_start(thread)

        with patch.object(threading.Thread, "start", start):
            result = self.app.start_search("phone", 10)
        self.assertEqual(result.state, ApplicationState.FAILED)
        self.assertTrue(result.cleanup_complete)
        self.assertIn(failure, [e.exception for e in result.errors])
        self.assertEqual(self.cameras.active, 0)
        self.assertEqual(self.detector.active, 0)

    def test_invalid_request_and_wrong_cancel_leave_active_search_unchanged(self):
        session = self.start()
        for target, timeout in (("glasses", 10), ("keys", 0), ("keys", float("nan")), ("keys", 1e-100)):
            with self.assertRaises(ValueError):
                self.app.start_search(target, timeout)
            self.assertEqual(self.app.poll().session.session_id, session.session_id)
        from uuid import uuid4
        self.assertEqual(self.app.cancel_search(uuid4()).state, ApplicationState.SEARCHING)

    def test_wrong_control_thread_is_rejected(self):
        errors = []

        def call():
            try:
                self.app.poll()
            except RuntimeError as error:
                errors.append(error)

        thread = threading.Thread(target=call)
        thread.start()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)

    def test_camera_read_failure_is_application_failure_not_cancellation(self):
        self.cameras.read_failure = RuntimeError("camera disconnected")
        self.start()
        result = self.until(lambda s: bool(s.errors))
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertEqual(result.session.status, SessionStatus.CANCELLED)
        self.until(lambda s: s.cleanup_complete)
        self.assertEqual(self.cameras.active, 0)

    def test_inference_failure_stops_camera_and_preserves_original_cause(self):
        failure = RuntimeError("inference failed")
        self.detector.failure = failure
        self.start()
        result = self.until(lambda s: bool(s.errors))
        self.assertIn(failure, [e.exception for e in result.errors])
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertIsNone(result.guidance)
        self.until(lambda s: s.cleanup_complete)

    def test_stop_exception_does_not_skip_other_worker(self):
        session = self.start()
        self.positive(100)
        camera = self.app._camera
        original_stop = camera.stop
        calls = []

        def stop(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise RuntimeError("stop failed once")
            return original_stop(timeout)

        with patch.object(camera, "stop", side_effect=stop), patch.object(
                self.app._inference, "stop", wraps=self.app._inference.stop) as inference_stop:
            result = self.app.cancel_search(session.session_id)
        self.assertTrue(inference_stop.called)
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertTrue(result.cleanup_complete)
        self.assertIn("camera.stop", [e.stage for e in result.errors])
        self.assertEqual(self.app.start_search("keys", 10).state, ApplicationState.FAILED)

    def test_unexpected_worker_completion_is_failure(self):
        self.start()
        self.positive(100)
        self.app._buffer.close()
        result = self.app.poll()
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertIn("termination", [e.stage for e in result.errors])

    def test_silence_ages_guidance_without_new_detection(self):
        self.start()
        for value in (100, 100.25, 100.5):
            result = self.positive(value)
        self.assertTrue(result.guidance.actionable)
        self.detector.unblock.clear()
        self.detector.entered.clear()
        self.assertTrue(self.detector.entered.wait(1))
        self.clock.set(101)
        result = self.app.poll()
        self.assertEqual(result.visibility.state, Visibility.UNCERTAIN)
        self.assertFalse(result.guidance.actionable)
        self.clock.set(101.5)
        result = self.app.poll()
        self.assertEqual(result.visibility.state, Visibility.UNCONFIRMED)
        self.assertFalse(result.guidance.actionable)

    def test_explicit_restart_after_confirmed_failure_cleanup(self):
        self.detector.failure = RuntimeError("prediction failed")
        self.start()
        failed = self.until(lambda s: bool(s.errors))
        self.until(lambda s: s.cleanup_complete)
        closed = self.app.close()
        self.assertEqual(closed.state, ApplicationState.CLOSED)
        self.assertTrue(closed.cleanup_complete)
        self.assertTrue(closed.errors)
        self.assertEqual(self.cameras.active, 0)
        self.assertEqual(self.detector.active, 0)
        self.detector.failure = None
        # Explicit new application, only after the old owner confirms release.
        self.app = ApplicationCoordinator(self.detector, source_factory=self.cameras,
                                          clock=self.clock, shutdown_timeout=0.1)
        session = self.start("keys")
        result = self.positive(100)
        self.assertNotEqual(session.session_id, failed.session.session_id)
        self.assertFalse(result.errors)
        self.assertEqual(self.cameras.maximum, 1)
        self.assertEqual(self.detector.maximum, 1)

    def test_empty_coordinator_close_is_idempotent(self):
        result = self.app.close()
        self.assertEqual(result.state, ApplicationState.CLOSED)
        self.assertTrue(result.cleanup_complete)
        self.assertIsNone(result.session)
        self.assertFalse(result.errors)
        self.assertEqual(self.app.close(), result)
        self.assertEqual(self.cameras.sources, [])

    def test_user_confirmed_completion_is_distinct_and_stops_resources(self):
        session = self.start()
        self.positive(100)
        result = self.app.complete_search(session.session_id)
        self.assertEqual(result.state, ApplicationState.IDLE)
        self.assertEqual(result.outcome, SearchOutcome.COMPLETED)
        self.assertEqual(result.session.status, SessionStatus.COMPLETED)
        self.assertTrue(result.cleanup_complete)
        self.assertEqual(self.cameras.active, 0)
        self.assertIsNone(self.app.poll().guidance)

    def test_completion_at_deadline_remains_timeout(self):
        session = self.start(timeout=1)
        self.positive(100)
        self.clock.set(session.deadline)
        result = self.app.complete_search(session.session_id)
        self.assertEqual(result.outcome, SearchOutcome.TIMED_OUT)
        self.assertEqual(result.session.status, SessionStatus.TIMED_OUT)

    def test_completion_blocked_cleanup_is_failure_not_success(self):
        self.detector.unblock.clear()
        session = self.start()
        self.assertTrue(self.detector.entered.wait(1))
        result = self.app.complete_search(session.session_id)
        self.assertEqual(result.outcome, SearchOutcome.FAILED)
        self.assertFalse(result.cleanup_complete)
        self.assertEqual(result.session.status, SessionStatus.COMPLETED)
        self.detector.unblock.set()

    def test_deadline_crossing_during_completion_does_not_report_success(self):
        session = self.start(timeout=1)
        operation = self.app._sessions.complete

        def complete_at_deadline(session_id):
            self.clock.set(session.deadline)
            return operation(session_id)

        with patch.object(self.app._sessions, "complete", side_effect=complete_at_deadline):
            result = self.app.complete_search(session.session_id)
        self.assertEqual(result.outcome, SearchOutcome.TIMED_OUT)
        self.assertEqual(result.session.status, SessionStatus.TIMED_OUT)
        self.assertTrue(result.cleanup_complete)
