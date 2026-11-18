"""Check manual-tool shutdown outcomes without touching hardware or weights."""

from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from homesense.application import ApplicationError, ApplicationSnapshot, ApplicationState, SearchOutcome
from homesense.session import SessionManager
from scripts import coordinator_preview as preview


class CoordinatorPreviewTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("sys.argv", ["coordinator_preview.py"]))
        self.detector = self.stack.enter_context(patch.object(preview, "YoloDetector"))
        self.app = Mock()
        self.constructor = self.stack.enter_context(patch.object(preview, "ApplicationCoordinator", return_value=self.app))
        self.window_cleanup = self.stack.enter_context(patch.object(preview.cv2, "destroyAllWindows"))
        for method in ("namedWindow", "imshow", "putText"):
            self.stack.enter_context(patch.object(preview.cv2, method))
        self.stack.enter_context(patch.object(preview.cv2, "waitKey", return_value=ord("q")))
        self.snapshot = ApplicationSnapshot(ApplicationState.SEARCHING, SessionManager().start("phone", 60),
                                             None, None, (), False)
        self.app.poll.return_value = self.snapshot
        self.final = replace(self.snapshot, state=ApplicationState.CLOSED, outcome=SearchOutcome.CANCELLED,
                             cleanup_complete=True)
        self.app.close.return_value = self.final
        for method in ("info", "error", "exception"):
            self.stack.enter_context(patch.object(preview.logging, method))

    def test_quit_closes_coordinator_and_windows(self):
        self.assertEqual(preview.main(), 0)
        self.app.start_search.assert_called_once_with("phone", 60.0)
        self.app.close.assert_called_once_with()
        self.window_cleanup.assert_called_once_with()

    def test_historical_shutdown_error_still_returns_failure_when_closed(self):
        self.app.close.return_value = replace(self.final, errors=(ApplicationError("inference.worker", RuntimeError("late failure")),))
        self.assertEqual(preview.main(), 1)
        self.window_cleanup.assert_called_once_with()

    def test_coordinator_cleanup_exception_still_cleans_windows(self):
        self.app.close.side_effect = RuntimeError("cleanup failed")
        self.assertEqual(preview.main(), 1)
        self.window_cleanup.assert_called_once_with()

    def test_start_exception_still_closes_coordinator(self):
        self.app.start_search.side_effect = RuntimeError("startup failed")
        self.assertEqual(preview.main(), 1)
        self.app.close.assert_called_once_with()
        self.window_cleanup.assert_called_once_with()
