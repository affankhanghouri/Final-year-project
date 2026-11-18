"""Scheduling and delivery accounting with real output threads and fake playback."""
import threading
import time
import unittest
from unittest.mock import Mock

from homesense.audio.input import PlaybackGate
from homesense.audio.speech_output import SpeechWorker, SpeechValidity, SpeechMessage, DeliveryStatus
from homesense.guidance import GuidanceStatus
from homesense.stability import Visibility
from homesense.interaction.feedback_manager import FeedbackManager
from test_voice_interaction import Clock, SnapshotFactory


def until(predicate):
    deadline = time.monotonic()+1
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Expected output state was not reached')
        time.sleep(.002)


class SchedulingTests(unittest.TestCase):
    def make(self):
        self.clock = Clock()
        self.validity = SpeechValidity(clock=self.clock)
        self.validity.update(0, 'left', 200)
        self.backend = Mock()
        self.playback = PlaybackGate(tail=.35, clock=self.clock)
        self.worker = SpeechWorker(self.backend, self.validity, self.playback, clock=self.clock)
        self.addCleanup(lambda: self.assertTrue(self.worker.stop(1)))
        self.worker.start()

    def send(self, text, *, fresh=True, priority=10):
        message = SpeechMessage(text, 0, 'left', priority, fresh_guidance=fresh)
        self.assertTrue(self.worker.submit(message))
        return message

    def test_fresh_guidance_bypasses_gap_then_full_unmuted_window_is_protected(self):
        self.make()
        self.send('ack', fresh=False)
        until(lambda: not self.worker.busy)
        self.send('first fresh')
        until(lambda: not self.worker.busy)
        self.send('second fresh')
        until(lambda: not self.worker.busy)
        third = self.send('third fresh')
        self.clock.now = 103.34  # Echo tail .35 + three full seconds for listening.
        time.sleep(.06)
        self.assertEqual(self.worker.delivery(third.message_id).status, DeliveryStatus.ACCEPTED)
        self.assertFalse(self.playback.snapshot()[0])
        self.clock.now = 103.36
        until(lambda: not self.worker.busy)
        self.assertEqual(self.worker.delivery(third.message_id).status, DeliveryStatus.COMPLETED)

    def test_terminal_priority_bypasses_window_and_replaced_direction_is_discarded(self):
        self.make()
        for text in ('left one', 'left two'):
            self.send(text)
            until(lambda: not self.worker.busy)
        pending = self.send('old left')
        terminal = SpeechMessage('Search cancelled.', 0, priority=90)
        self.assertTrue(self.worker.submit(terminal))
        until(lambda: not self.worker.busy)
        self.assertEqual(self.worker.delivery(pending.message_id).status, DeliveryStatus.DISCARDED)
        self.assertEqual(self.worker.delivery(terminal.message_id).status, DeliveryStatus.COMPLETED)
        self.assertNotIn('old left', [c.args[0] for c in self.backend.speak.call_args_list])

    def test_stale_pending_is_discarded_during_window_without_waiting_for_expiry(self):
        self.make()
        for text in ('one', 'two'):
            self.send(text)
            until(lambda: not self.worker.busy)
        message = self.send('obsolete')
        self.validity.update(1, 'right', 200)
        until(lambda: not self.worker.busy)
        self.assertEqual(self.worker.delivery(message.message_id).status, DeliveryStatus.DISCARDED)
        self.assertNotIn('obsolete', [c.args[0] for c in self.backend.speak.call_args_list])

    def test_receipt_ledger_is_bounded_and_shutdown_discards_pending(self):
        self.make()
        self.worker.stop()
        ids = []
        for index in range(100):
            message = SpeechMessage(str(index), 0)
            ids.append(message.message_id)
            self.assertFalse(self.worker.submit(message))
        self.assertIsNone(self.worker.delivery(ids[0]))
        self.assertEqual(self.worker.delivery(ids[-1]).status, DeliveryStatus.REJECTED)
        self.assertLessEqual(len(self.worker._deliveries), 64)

    def test_pending_shutdown_has_no_started_receipt(self):
        validity = SpeechValidity()
        worker = SpeechWorker(Mock(), validity, PlaybackGate(0))
        message = SpeechMessage('pending', 0)
        self.assertTrue(worker.submit(message))
        self.assertTrue(worker.stop())
        receipt = worker.delivery(message.message_id)
        self.assertEqual(receipt.status, DeliveryStatus.DISCARDED)
        self.assertIsNone(receipt.started_at)

    def test_genuine_backend_failure_is_failed_not_completed(self):
        validity = SpeechValidity()
        backend = Mock()
        backend.speak.side_effect = RuntimeError('device failed')
        worker = SpeechWorker(backend, validity, PlaybackGate(0))
        self.addCleanup(worker.stop)
        message = SpeechMessage('test', 0)
        with self.assertLogs('homesense.audio.speech_output', level='ERROR'):
            worker.start()
            worker.submit(message)
            until(lambda: worker.error is not None)
        self.assertEqual(worker.delivery(message.message_id).status, DeliveryStatus.FAILED)
        self.assertEqual(worker.stats.completed, 0)


class FeedbackDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = SnapshotFactory(self.clock)
        self.validity = SpeechValidity(clock=self.clock)
        self.backend = Mock()
        self.entered, self.release = threading.Event(), threading.Event()

        def speak(text, stop):
            if 'looking for' in text:
                self.entered.set()
                while not self.release.wait(.002) and not stop():
                    pass

        self.backend.speak.side_effect = speak
        self.worker = SpeechWorker(self.backend, self.validity, PlaybackGate(0, clock=self.clock), clock=self.clock)
        self.feedback = FeedbackManager(self.worker, self.validity, clock=self.clock)
        self.addCleanup(lambda: self.assertTrue(self.worker.stop(1)))
        self.addCleanup(self.release.set)
        self.worker.start()
        self.feedback.on_snapshot(self.factory.make(status=GuidanceStatus.UNCONFIRMED, visibility=Visibility.UNCONFIRMED))
        self.assertTrue(self.entered.wait(1))

    def test_queued_discarded_direction_does_not_suppress_future_fresh_direction(self):
        self.feedback.on_snapshot(self.factory.make())
        pending = self.feedback._pending_direction
        self.assertIsNotNone(pending)
        self.assertIsNone(self.feedback._last_direction)
        self.validity.update(self.feedback.generation, 'right', 200)
        self.release.set()
        until(lambda: not self.worker.busy)
        self.assertEqual(self.worker.delivery(pending.message_id).status, DeliveryStatus.DISCARDED)
        self.feedback.on_snapshot(self.factory.make())
        until(lambda: not self.worker.busy)
        self.feedback.on_snapshot(self.factory.make())
        self.assertEqual(self.feedback._last_direction, 'left')
        self.assertEqual(sum(c.args[0] == 'Phone on your left.' for c in self.backend.speak.call_args_list), 1)

    def test_started_is_not_completed_and_interruption_can_retry(self):
        self.release.set()
        until(lambda: not self.worker.busy)
        entered = threading.Event()

        def interrupted(text, stop):
            entered.set()
            while not stop():
                time.sleep(.002)

        self.backend.speak.side_effect = interrupted
        self.feedback.on_snapshot(self.factory.make())
        self.assertTrue(entered.wait(1))
        pending = self.feedback._pending_direction
        self.assertEqual(self.worker.delivery(pending.message_id).status, DeliveryStatus.STARTED)
        self.assertIsNone(self.feedback._last_direction)
        self.feedback.set_listening(True)
        until(lambda: not self.worker.busy)
        self.assertEqual(self.worker.delivery(pending.message_id).status, DeliveryStatus.INTERRUPTED)
        self.feedback.set_listening(False)
        self.clock.now += 3.01
        self.backend.speak.side_effect = None
        self.feedback.on_snapshot(self.factory.make())
        until(lambda: not self.worker.busy)
        self.feedback.on_snapshot(self.factory.make())
        self.assertEqual(self.feedback._last_direction, 'left')
