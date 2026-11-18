"""Full voice-event/perception/feedback flow, real workers, synthetic I/O only."""

from queue import Empty, Queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

from homesense.application import ApplicationCoordinator, ApplicationState, SearchOutcome
from homesense.audio.events import VoiceEvent, VoiceEventKind, VoiceEvents
from homesense.audio.input import PlaybackGate, VoiceWorker
from homesense.audio.speech_recognition import Recognition
from homesense.audio.speech_output import SpeechValidity, SpeechWorker
from homesense.config import AudioSettings
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.manager import InteractionManager
from homesense.runtime import HomeSenseRuntime
from homesense.session import SessionStatus
from .test_application import Cameras, Detector


class Clock:
    """Advance between real camera frames, with explicit jumps for confirmation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.set(100.0)

    def __call__(self):
        with self.lock:
            return self.value + time.monotonic() - self.origin

    def set(self, value):
        with self.lock:
            self.value, self.origin = value, time.monotonic()


class AudioSource:
    def __init__(self):
        self.queue = Queue()
        self.closed = threading.Event()

    def open(self):
        pass

    def read(self, timeout):
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def drain(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return

    def close(self):
        self.closed.set()


class Recognizer:
    def feed(self, chunk):
        return Recognition(chunk.decode(), True)

    def reset(self):
        pass


class RecordingSpeech:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()
        self.block_left = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def speak(self, text, should_stop):
        if should_stop():
            return
        with self.lock:
            self.messages.append(text)
        if self.block_left and "on your left" in text:
            self.entered.set()
            deadline = time.monotonic() + 2
            while not should_stop() and not self.release.wait(0.002):
                if time.monotonic() > deadline:
                    raise RuntimeError("test speech not released")

    def count(self, fragment):
        with self.lock:
            return sum(fragment in text for text in self.messages)


class VoiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.cameras, self.detector = Clock(), Cameras(), Detector()
        self.validity = SpeechValidity(clock=self.clock)
        self.playback = PlaybackGate(tail=0, clock=self.clock)
        self.backend = RecordingSpeech()
        self.speech = SpeechWorker(self.backend, self.validity, self.playback, listening_gap=0, clock=self.clock)
        self.feedback = FeedbackManager(self.speech, self.validity, clock=self.clock)
        self.app = None

        def factory():
            self.app = ApplicationCoordinator(self.detector, source_factory=self.cameras,
                                              shutdown_timeout=0.1, clock=self.clock)
            return self.app

        self.factory = Mock(side_effect=factory)
        self.manager = InteractionManager(self.factory, self.feedback)
        self.source, self.events = AudioSource(), VoiceEvents()
        self.voice = VoiceWorker(self.source, Recognizer(), self.events, self.playback,
                                 AudioSettings(echo_tail=0), clock=self.clock)
        self.runtime = HomeSenseRuntime(self.manager, self.feedback, self.speech, self.events,
                                        voice=self.voice, clock=self.clock, speech_shutdown_grace=0.1)
        self.addCleanup(self.cleanup)
        self.runtime.start()
        self.until(lambda: not self.speech.busy)

    def cleanup(self):
        self.backend.release.set()
        self.detector.unblock.set()
        for source in self.cameras.sources:
            source.unblock.set()
        self.assertEqual(self.runtime.close(), not getattr(self, "expected_cleanup_failure", False))
        self.assertTrue(self.source.closed.is_set())
        self.assertEqual(self.cameras.active, 0)
        self.assertEqual(self.detector.active, 0)

    def until(self, predicate):
        deadline = time.monotonic() + 1.5
        while True:
            self.runtime.step()
            if predicate():
                return
            if time.monotonic() > deadline:
                self.fail("runtime did not reach expected state")
            time.sleep(0.002)

    def say(self, text):
        self.until(lambda: not self.speech.busy)
        time.sleep(0.12)  # Recognition observes the completed playback epoch and drains buffered echo.
        self.source.queue.put(text.encode())

    def confirm_visibility(self):
        for value in (100, 100.25, 100.5):
            self.clock.set(value)
            self.until(lambda: self.manager.snapshot.batch is not None and
                       self.manager.snapshot.batch.frame.captured_at >= value)

    def test_single_utterance_to_guidance_repeat_and_user_completion(self):
        self.assertIsNone(self.app)
        self.assertEqual(self.cameras.sources, [])
        self.say("hey agent find my phone")
        self.until(lambda: self.app is not None)
        self.confirm_visibility()
        self.until(lambda: self.backend.count("on your left") == 1)
        self.assertEqual(self.manager.snapshot.session.status, SessionStatus.ACTIVE)
        self.say("hey agent repeat direction")
        self.until(lambda: self.backend.count("on your left") == 2)
        self.say("hey agent i found it")
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.COMPLETED)
        self.until(lambda: self.backend.count("Search completed") == 1)
        self.assertEqual(self.manager.snapshot.session.status, SessionStatus.COMPLETED)
        self.assertTrue(self.manager.snapshot.cleanup_complete)
        self.assertEqual(self.cameras.active, 0)
        self.factory.assert_called_once_with()

    def test_launch_listens_without_wake_and_keeps_idle_camera_off(self):
        for _ in range(20):
            self.runtime.step()
        self.factory.assert_not_called()
        self.assertEqual(self.cameras.sources, [])
        self.say('Can you find my phone?')
        self.until(lambda: self.app is not None)
        self.assertEqual(self.manager.snapshot.session.target.value, 'phone')
        self.confirm_visibility()
        self.until(lambda: self.backend.count('on your left') == 1)
        self.say('Repeat the direction.')
        self.until(lambda: self.backend.count('on your left') == 2)
        self.say('I found it.')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.COMPLETED)
        self.assertEqual(self.cameras.active, 0)

    def test_unsupported_unrelated_negated_and_partial_requests_never_open_camera(self):
        for phrase in ('Find my clothes.', 'Do not find my phone.', 'please close the window',
                       'find my phone or keys', 'find my', '[unk] find my phone'):
            self.say(phrase)
            self.until(lambda: self.source.queue.empty())
            time.sleep(.02)
            self.runtime.step()
        self.factory.assert_not_called()
        self.assertEqual(self.cameras.sources, [])
        self.assertGreaterEqual(self.backend.count('I can only find phones, keys, and wallets'), 1)

    def test_plain_cancel_complete_and_multiple_searches_reuse_real_coordinator(self):
        self.say('find my phone')
        self.until(lambda: self.app is not None)
        self.say('cancel search')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.CANCELLED)
        self.assertEqual(self.cameras.active, 0)
        self.say('hey agent please find my keys')
        self.until(lambda: self.manager.snapshot.session.target.value == 'keys')
        self.say('i found it')
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.COMPLETED)
        self.assertEqual(self.cameras.active, 0)
        self.say('locate my wallet')
        self.until(lambda: self.manager.snapshot.session.target.value == 'wallet')
        self.assertEqual(self.cameras.maximum, 1)
        self.assertEqual(self.detector.maximum, 1)
        self.factory.assert_called_once_with()

    def test_help_does_not_load_detector_and_idle_repeat_does_not_invent_direction(self):
        self.say('Help me.')
        self.until(lambda: self.backend.count('I can search') == 1)
        self.say('Repeat the direction.')
        self.until(lambda: self.backend.count("don't have a current confirmed direction") == 1)
        self.factory.assert_not_called()

    def test_microphone_error_during_search_closes_real_workers(self):
        self.say('find my phone')
        self.until(lambda: self.app is not None)
        self.events.put(VoiceEvent(VoiceEventKind.ERROR, 'microphone disconnected', self.clock()))
        with self.assertLogs('homesense.interaction.manager', level='ERROR'):
            self.runtime.step()
        self.assertTrue(self.runtime.failed)
        self.assertEqual(self.cameras.active, 0)
        self.assertEqual(self.detector.active, 0)

    def test_duplicate_result_id_does_not_replace_search_but_new_utterance_can(self):
        recognizer = Mock()
        recognizer.feed.return_value = Recognition('find my phone', True, 77)
        self.voice.recognizer = recognizer
        self.say('first audio')
        self.until(lambda: self.app is not None)
        session_id = self.manager.snapshot.session.session_id
        self.say('duplicate audio callback')
        self.until(lambda: recognizer.feed.call_count >= 2)
        self.runtime.step()
        self.assertEqual(self.manager.snapshot.session.session_id, session_id)
        recognizer.feed.return_value = Recognition('find my phone', True, 78)
        self.say('new utterance, same command')
        self.until(lambda: self.manager.snapshot.session.session_id != session_id)

    def test_two_turn_command_then_wake_cancellation(self):
        self.say("hey agent")
        self.until(lambda: self.backend.count("I'm listening") == 1)
        self.assertIsNone(self.app)
        self.say("find my keys")
        self.until(lambda: self.app is not None)
        self.say("hey agent cancel search")
        self.until(lambda: self.manager.snapshot.outcome is SearchOutcome.CANCELLED)
        self.until(lambda: self.backend.count("Search cancelled") == 1)
        self.assertEqual(self.manager.snapshot.state, ApplicationState.IDLE)
        self.assertEqual(self.cameras.active, 0)

    def test_voice_replacement_has_no_overlap_or_old_session_speech(self):
        self.say("hey agent find my phone")
        self.until(lambda: self.app is not None)
        self.confirm_visibility()
        self.until(lambda: self.backend.count("on your left") == 1)
        old = self.manager.snapshot.session.session_id
        self.say("hey agent find my keys")
        self.until(lambda: self.manager.snapshot.session.session_id != old)
        self.assertEqual(self.cameras.maximum, 1)
        self.assertEqual(self.detector.maximum, 1)
        self.assertTrue(self.manager.snapshot.guidance is None or not self.manager.snapshot.guidance.actionable)
        self.factory.assert_called_once_with()

    def test_speech_does_not_block_coordinator_polling_or_inference(self):
        self.backend.block_left = True
        self.say("hey agent find my phone")
        self.until(lambda: self.app is not None)
        self.confirm_visibility()
        self.until(lambda: self.backend.entered.is_set())
        before = self.app._inference.stats.processed
        with patch.object(self.app, "poll", wraps=self.app.poll) as poll:
            for _ in range(30):
                self.runtime.step()
                time.sleep(0.002)
            self.assertEqual(poll.call_count, 30)
        self.assertGreater(self.app._inference.stats.processed, before)
        self.backend.release.set()

    def test_expired_voice_commands_are_not_executed(self):
        self.events.put(VoiceEvent(VoiceEventKind.TRANSCRIPT, "find my phone", 90))
        self.runtime.step()
        self.assertIsNone(self.app)
        self.factory.assert_not_called()

    def test_audio_failure_stops_runtime_without_starting_camera(self):
        self.events.put(VoiceEvent(VoiceEventKind.ERROR, "device disconnected", self.clock()))
        with self.assertLogs("homesense.interaction.manager", level="ERROR"):
            self.runtime.step()
        self.assertTrue(self.runtime.failed)
        self.factory.assert_not_called()

    def test_close_attempts_audio_cleanup_if_coordinator_close_raises(self):
        self.say("hey agent find my phone")
        self.until(lambda: self.app is not None)
        with patch.object(self.app, "close", side_effect=RuntimeError("close failed")), \
             self.assertLogs(level="ERROR"):
            self.assertFalse(self.runtime.close())
        self.assertTrue(self.source.closed.is_set())
        self.assertTrue(self.runtime.failed)
        # The real coordinator reference remains available for the cleanup retry.

    def test_search_timeout_stops_camera_and_announces_timeout(self):
        self.say("hey agent find my phone")
        self.until(lambda: self.app is not None)
        self.clock.set(self.manager.snapshot.session.deadline)
        self.until(lambda: self.manager.snapshot.cleanup_complete)
        self.until(lambda: self.backend.count("couldn't locate") == 1)
        self.assertEqual(self.manager.snapshot.outcome, SearchOutcome.TIMED_OUT)
        self.assertEqual(self.cameras.active, 0)

    def test_expired_command_clears_listening_without_executing(self):
        self.events.put(VoiceEvent(VoiceEventKind.LISTENING, occurred_at=self.clock()))
        self.runtime.step()
        self.assertTrue(self.manager.listening)
        self.events.put(VoiceEvent(VoiceEventKind.TRANSCRIPT, "find my phone", 90))
        self.runtime.step()
        self.assertFalse(self.manager.listening)
        self.factory.assert_not_called()

    def test_microphone_release_failure_remains_explicit_on_repeated_close(self):
        with patch.object(self.source, "close", side_effect=RuntimeError("mic release failed")):
            self.assertFalse(self.runtime.close())
        self.assertIsNotNone(self.voice.cleanup_error)
        self.assertTrue(self.runtime.failed)
        self.assertFalse(self.runtime.close())
        # Manually release the fake for test hygiene; production diagnostics stay latched.
        self.source.close()
        self.expected_cleanup_failure = True
