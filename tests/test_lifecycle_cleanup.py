"""Regression tests for partial startup and shutdown failures; no hardware needed."""

from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

from homesense.camera import CameraWorker, LatestFrameBuffer
from homesense.perception import DetectionWorker, WorkerStats
from homesense.session import SessionManager
from scripts import live_detection_preview as preview


class StartupFailureTests(unittest.TestCase):
    def test_failed_thread_start_is_terminal_and_safe_to_stop(self):
        sessions = SessionManager()
        session = sessions.start("phone", 60)
        for kind in ("camera", "inference"):
            with self.subTest(worker=kind):
                buffer = LatestFrameBuffer()
                dependency = Mock()
                worker = (CameraWorker(buffer, source_factory=dependency) if kind == "camera"
                          else DetectionWorker(buffer, dependency, sessions, session))
                failure = RuntimeError("cannot start thread")
                with patch("threading.Thread.start", side_effect=failure), patch("threading.Thread.join") as join:
                    with self.assertRaises(RuntimeError) as caught:
                        worker.start()
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(worker.stop())
                    self.assertTrue(worker.stop())
                    join.assert_not_called()
                self.assertIs(worker.error, failure)
                with self.assertRaises(RuntimeError):
                    worker.start()
                dependency.assert_not_called()
                dependency.predict.assert_not_called()
                if kind == "camera":
                    self.assertTrue(buffer.closed)
                else:
                    self.assertTrue(worker.finished)
                    self.assertEqual(worker.stats.processed, 0)
                buffer.close()

    def test_camera_release_failure_is_retained_after_join(self):
        buffer = LatestFrameBuffer()
        source = Mock()
        source.isOpened.return_value = True
        failure = RuntimeError("camera release failed")
        source.release.side_effect = failure

        def read():
            import numpy as np
            buffer.close()
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        source.read.side_effect = read
        worker = CameraWorker(buffer, source_factory=lambda _: source)
        worker.start()
        self.assertIsNone(buffer.wait_for_newer(timeout=1))
        self.assertTrue(worker.stop(1))
        self.assertIs(worker.error, failure)
        source.release.assert_called_once_with()


class PreviewCleanupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("sys.argv", ["live_detection_preview.py"]))
        self.stack.enter_context(patch.object(preview, "YoloDetector"))
        for method in ("namedWindow", "destroyAllWindows"):
            setattr(self, method, self.stack.enter_context(patch.object(preview.cv2, method)))
        self.camera = Mock(error=None)
        self.camera.stop.return_value = True
        self.inference = Mock(error=None, finished=True, stats=WorkerStats())
        self.inference.stop.return_value = True
        self.stack.enter_context(patch.object(preview, "CameraWorker", return_value=self.camera))
        self.stack.enter_context(patch.object(preview, "DetectionWorker", return_value=self.inference))
        self.stack.enter_context(patch.object(preview.logging, "exception"))
        self.stack.enter_context(patch.object(preview.logging, "error"))
        self.stack.enter_context(patch.object(preview.logging, "info"))

    def assert_all_cleanup_attempted(self):
        self.camera.stop.assert_called_once_with()
        self.inference.stop.assert_called_once_with()
        self.destroyAllWindows.assert_called_once_with()

    def test_normal_shutdown_succeeds(self):
        self.assertEqual(preview.main(), 0)
        self.assert_all_cleanup_attempted()

    def test_errors_first_reported_during_stop_fail_the_preview(self):
        for worker in (self.camera, self.inference):
            with self.subTest(worker=worker):
                self.camera.error = self.inference.error = None
                self.camera.stop.reset_mock(side_effect=True)
                self.inference.stop.reset_mock(side_effect=True)
                self.destroyAllWindows.reset_mock()

                def stop():
                    worker.error = RuntimeError("release or late prediction failed")
                    return True

                worker.stop.side_effect = stop
                self.assertEqual(preview.main(), 1)
                self.assert_all_cleanup_attempted()

    def test_stop_exception_does_not_skip_other_resources(self):
        self.camera.stop.side_effect = RuntimeError("camera cleanup failed")
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()

    def test_inference_stop_exception_still_cleans_windows(self):
        self.inference.stop.side_effect = RuntimeError("inference cleanup failed")
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()

    def test_invalidation_exception_does_not_skip_resources(self):
        self.stack.enter_context(patch.object(preview.StabilityProcessor, "snapshot",
                                            side_effect=RuntimeError("invalidation failed")))
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()

    def test_partial_startup_failure_still_cleans_both_workers(self):
        self.inference.start.side_effect = RuntimeError("cannot start thread")
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()

    def test_bounded_stop_failure_is_explicit(self):
        self.camera.stop.return_value = False
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()

    def test_window_cleanup_failure_is_explicit(self):
        self.destroyAllWindows.side_effect = RuntimeError("window cleanup failed")
        self.assertEqual(preview.main(), 1)
        self.assert_all_cleanup_attempted()


if __name__ == "__main__":
    unittest.main()
