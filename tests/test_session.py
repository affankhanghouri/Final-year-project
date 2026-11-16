import unittest
from uuid import uuid4

from homesense.session import SessionManager, SessionStatus, Target


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.manager = SessionManager(clock=self.clock)

    def test_creation_normalizes_target_and_sets_deadline(self) -> None:
        session = self.manager.start("  KeYs ", 5)
        self.assertEqual(session.target, Target.KEYS)
        self.assertEqual((session.started_at, session.deadline), (100, 105))
        self.assertEqual(session.status, SessionStatus.ACTIVE)
        self.assertTrue(self.manager.is_eligible(session.session_id))

    def test_invalid_requests_do_not_replace_active_session(self) -> None:
        session = self.manager.start("phone", 5)
        for target, timeout in [("glasses", 5), (None, 5), ("keys", 0),
                                ("keys", -1), ("keys", float("nan")),
                                ("keys", float("inf")), ("keys", True)]:
            with self.subTest(target=target, timeout=timeout):
                with self.assertRaises(ValueError):
                    self.manager.start(target, timeout)
                self.assertTrue(self.manager.is_eligible(session.session_id))

    def test_timeout_boundary_and_terminal_status(self) -> None:
        session = self.manager.start("wallet", 5)
        self.clock.now = 104.999
        self.assertTrue(self.manager.is_eligible(session.session_id))
        self.clock.now = 105
        self.assertFalse(self.manager.is_eligible(session.session_id))
        self.assertEqual(self.manager.current.status, SessionStatus.TIMED_OUT)
        self.assertFalse(self.manager.cancel(session.session_id))

    def test_cancel_and_wrong_identity(self) -> None:
        session = self.manager.start("phone", 5)
        self.assertFalse(self.manager.cancel(uuid4()))
        self.assertTrue(self.manager.cancel(session.session_id))
        self.assertFalse(self.manager.cancel(session.session_id))
        self.assertFalse(self.manager.is_eligible(session.session_id))
        self.assertEqual(self.manager.current.status, SessionStatus.CANCELLED)

    def test_replacement_rejects_old_results_and_cancel(self) -> None:
        old = self.manager.start("phone", 5)
        new = self.manager.start("keys", 5)
        self.assertNotEqual(old.session_id, new.session_id)
        self.assertFalse(self.manager.is_eligible(old.session_id))
        self.assertFalse(self.manager.cancel(old.session_id))
        self.assertTrue(self.manager.is_eligible(new.session_id))

    def test_no_current_session(self) -> None:
        self.assertIsNone(self.manager.current)
        self.assertFalse(self.manager.is_eligible(uuid4()))
        self.assertFalse(self.manager.cancel(uuid4()))


class ConcurrentSessionTests(unittest.TestCase):
    def test_guard_coordinates_result_handoff_with_cancellation(self) -> None:
        import threading
        manager = SessionManager()
        session = manager.start("phone", 60)
        entered = threading.Event()
        cancelled = threading.Event()

        def cancel() -> None:
            entered.set()
            manager.cancel(session.session_id)
            cancelled.set()

        with manager.guard_active(session.session_id) as active:
            self.assertEqual(active, session)
            thread = threading.Thread(target=cancel)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(cancelled.wait(0.01))
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(cancelled.is_set())
        with manager.guard_active(session.session_id) as active:
            self.assertIsNone(active)

    def test_only_one_concurrent_cancel_succeeds(self) -> None:
        import threading
        manager = SessionManager()
        session = manager.start("keys", 60)
        barrier = threading.Barrier(9)
        results = []

        def cancel() -> None:
            barrier.wait()
            results.append(manager.cancel(session.session_id))

        threads = [threading.Thread(target=cancel) for _ in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(1)
            self.assertFalse(thread.is_alive())
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 7)
        self.assertFalse(manager.is_eligible(session.session_id))


if __name__ == "__main__":
    unittest.main()
