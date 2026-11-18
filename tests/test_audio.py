from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from homesense.audio.events import VoiceEvent, VoiceEventKind, VoiceEvents
from homesense.audio.input import PlaybackGate, SoundDeviceInput, VoiceWorker
from homesense.audio.speech_recognition import Recognition, VoskRecognizer
from homesense.audio.speech_output import SpeechWorker, SpeechValidity, SpeechMessage, EspeakSpeech
from homesense.audio.wake_word import WakePhraseGate
from homesense.config import AudioSettings


def until(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("worker did not reach expected state")
        time.sleep(0.002)


class WakeTests(unittest.TestCase):
    def test_wrong_phrase_never_activates(self):
        gate = WakePhraseGate()
        for text in ["hey jarvis find my phone", "agent find my phone", "hey agents", "find my keys"]:
            self.assertEqual(gate.accept(text, True, 100), [])

    def test_one_utterance_keeps_command_after_wake(self):
        gate = WakePhraseGate()
        first = gate.accept("hey agent find", False, 100)
        self.assertEqual([e.kind for e in first], [VoiceEventKind.LISTENING])
        final = gate.accept("hey agent find my phone", True, 100.5)
        self.assertEqual([e.text for e in final], ["find my phone"])
        self.assertIsNone(gate.deadline)

    def test_two_turn_fallback_acknowledges_only_at_final(self):
        gate = WakePhraseGate()
        self.assertEqual(gate.accept("hey agent", False, 100)[0].kind, VoiceEventKind.LISTENING)
        self.assertEqual(gate.accept("hey agent", True, 100.5)[0].kind, VoiceEventKind.ACKNOWLEDGE)
        result = gate.accept("locate my keys", True, 101)
        self.assertEqual(result[0].text, "locate my keys")

    def test_repeated_activation_and_timeout(self):
        gate = WakePhraseGate(timeout=2)
        for now in [100, 101]:
            result = gate.accept("hey agent cancel search", True, now)
            self.assertEqual(result[-1].text, "cancel search")
        gate.accept("hey agent", True, 102)
        self.assertEqual(gate.tick(104)[0].kind, VoiceEventKind.TIMEOUT)
        self.assertEqual(gate.accept("find my phone", True, 104), [])

    def test_revised_or_expired_partial_does_not_execute_command(self):
        gate = WakePhraseGate(timeout=1)
        gate.accept("hey agent find", False, 100)
        revised = gate.accept("find my keys", True, 100.5)
        self.assertEqual([e.kind for e in revised], [VoiceEventKind.TIMEOUT])
        gate.accept("hey agent find", False, 101)
        self.assertEqual([e.kind for e in gate.accept("hey agent find my phone", True, 102)], [VoiceEventKind.TIMEOUT])

    def test_event_queue_overflow_is_explicit(self):
        events = VoiceEvents(capacity=1)
        events.put(VoiceEvent(VoiceEventKind.TRANSCRIPT, "cancel"))
        events.put(VoiceEvent(VoiceEventKind.TRANSCRIPT, "find my keys"))
        result = events.drain()
        self.assertEqual([e.kind for e in result], [VoiceEventKind.ERROR])


class Source:
    def __init__(self):
        self.chunks = Queue()
        self.closed = threading.Event()
        self.opened = threading.Event()
        self.failure = None

    def open(self):
        self.opened.set()
        if self.failure:
            raise self.failure

    def read(self, timeout):
        try:
            return self.chunks.get(timeout=timeout)
        except Empty:
            return None

    def drain(self):
        while True:
            try:
                self.chunks.get_nowait()
            except Empty:
                return

    def close(self):
        self.closed.set()


class AudioWorkerTests(unittest.TestCase):
    def make(self, recognizer=None):
        self.source, self.events = Source(), VoiceEvents()
        self.playback = PlaybackGate(tail=0)
        self.recognizer = recognizer or Mock()
        self.recognizer.feed.return_value = Recognition("hey agent find my phone", True)
        self.worker = VoiceWorker(self.source, self.recognizer, self.events, self.playback, AudioSettings())
        self.addCleanup(lambda: self.assertTrue(self.worker.stop(1)))
        return self.worker

    def test_worker_routes_commands_without_perception_calls(self):
        self.make().start()
        self.source.chunks.put(b"pcm")
        until(lambda: not self.events.queue.empty())
        result = self.events.drain()
        until(lambda: self.recognizer.feed.called)
        from homesense.commands.parser import parse_command
        from homesense.session import Target
        self.assertTrue(any(e.kind is VoiceEventKind.TRANSCRIPT and
                            parse_command(e.text).target is Target.PHONE for e in result))

    def test_microphone_failure_and_close_are_reported(self):
        worker = self.make()
        failure = RuntimeError("microphone missing")
        self.source.failure = failure
        worker.start()
        until(lambda: self.source.closed.is_set())
        self.assertIs(worker.error, failure)
        self.assertEqual(self.events.drain()[0].kind, VoiceEventKind.ERROR)

    def test_playback_suppresses_recognition_and_resumes(self):
        worker = self.make()
        self.playback.set_playing(True)
        worker.start()
        self.assertTrue(self.source.opened.wait(1))
        self.source.chunks.put(b"own speech")
        time.sleep(0.03)
        self.recognizer.feed.assert_not_called()
        self.playback.set_playing(False)
        time.sleep(0.12)  # Observe resume/drain before supplying genuinely new audio.
        self.source.chunks.put(b"user speech")
        until(lambda: self.recognizer.feed.called)
        self.assertTrue(any(e.kind is VoiceEventKind.TRANSCRIPT for e in self.events.drain()))

    def test_playback_start_during_recognition_discards_that_result(self):
        worker = self.make()
        entered, release = threading.Event(), threading.Event()

        def feed(chunk):
            entered.set()
            if not release.wait(1):
                raise RuntimeError("not released")
            return Recognition("hey agent cancel search", True)

        self.recognizer.feed.side_effect = feed
        worker.start()
        self.source.chunks.put(b"pcm")
        try:
            self.assertTrue(entered.wait(1))
            self.playback.set_playing(True)
        finally:
            release.set()
        time.sleep(0.03)
        self.assertEqual(self.events.drain(), [])

    def test_recognition_failure_closes_microphone_and_reports_error(self):
        worker = self.make()
        error = RuntimeError("recognition failed")
        self.recognizer.feed.side_effect = error
        worker.start()
        self.source.chunks.put(b"pcm")
        until(lambda: self.source.closed.is_set())
        self.assertIs(worker.error, error)
        self.assertEqual(self.events.drain()[0].kind, VoiceEventKind.ERROR)

    def test_decoder_cleanup_failure_still_releases_microphone_and_preserves_errors(self):
        worker = self.make()
        failure = RuntimeError("recognition failed")
        cleanup_failure = RuntimeError("decoder reset failed")
        self.recognizer.feed.side_effect = failure
        self.recognizer.reset.side_effect = cleanup_failure
        worker.start()
        self.source.chunks.put(b"pcm")
        self.assertTrue(self.source.closed.wait(1))
        self.assertTrue(worker.stop(1))
        self.assertIs(worker.error, failure)
        self.assertIs(worker.cleanup_error, cleanup_failure)
        errors = self.events.drain()
        self.assertEqual([e.kind for e in errors], [VoiceEventKind.ERROR, VoiceEventKind.ERROR])
        self.assertEqual([e.text for e in errors], [str(failure), str(cleanup_failure)])

    def test_decode_crossing_deadline_discards_unfinished_utterance(self):
        worker = self.make()
        worker.clock = Mock(return_value=100.0)

        def feed(chunk):
            if chunk == b"late":
                worker.clock.return_value = 111.0
            return Recognition("find my", False)

        self.recognizer.feed.side_effect = feed
        worker.start()
        self.source.chunks.put(b"start")
        until(lambda: worker.gate.deadline is not None)
        self.source.chunks.put(b"late")
        until(lambda: self.recognizer.discard.called)
        events = self.events.drain()
        self.assertEqual([e.kind for e in events], [VoiceEventKind.LISTENING, VoiceEventKind.TIMEOUT])
        self.assertFalse(any(e.kind is VoiceEventKind.TRANSCRIPT for e in events))

    def test_failed_voice_thread_start_safe_to_stop(self):
        worker = self.make()
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start")):
            with self.assertRaises(RuntimeError):
                worker.start()
        self.assertTrue(worker.stop())
        self.assertFalse(self.source.opened.is_set())

    def test_source_callback_overflow_is_reported(self):
        source = SoundDeviceInput(AudioSettings())
        for _ in range(source._chunks.maxsize + 1):
            source._callback(b"pcm", 1, None, None)
        with self.assertRaises(RuntimeError):
            source.read(0)


class RecognizerTests(unittest.TestCase):
    def test_missing_model_has_no_automatic_download(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            VoskRecognizer(Path(directory))

    def test_installed_api_adapter_and_software_silence_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "am").mkdir()
            (path / "am/final.mdl").touch()
            native = Mock()
            native.AcceptWaveform.return_value = False
            native.PartialResult.return_value = '{"partial":"hey agent find my phone"}'
            native.FinalResult.return_value = '{"text":"hey agent find my phone"}'
            module = SimpleNamespace(Model=Mock(), KaldiRecognizer=Mock(return_value=native))
            with patch.dict("sys.modules", {"vosk": module}):
                recognizer = VoskRecognizer(path, sample_rate=16000, silence_seconds=0.1)
                result = recognizer.feed(b"\0" * 3200)
                self.assertTrue(result.final)
                self.assertEqual(result.text, "hey agent find my phone")
                native.Reset.assert_called_once_with()
                module.Model.assert_called_once_with(str(path))


class SpeechWorkerTests(unittest.TestCase):
    def make(self):
        self.validity = SpeechValidity()
        self.validity.update(1, "left", time.monotonic() + 5)
        self.playback = PlaybackGate(tail=0)
        self.backend = Mock()
        self.worker = SpeechWorker(self.backend, self.validity, self.playback, listening_gap=0)
        self.addCleanup(lambda: self.assertTrue(self.worker.stop(1)))

    def test_stale_pending_message_never_plays(self):
        self.make()
        self.assertTrue(self.worker.submit(SpeechMessage("left", 1, "left")))
        self.validity.update(2, "right", time.monotonic() + 5)
        self.worker.start()
        until(lambda: not self.worker.busy)
        self.backend.speak.assert_not_called()

    def test_new_session_interrupts_old_playback(self):
        self.make()
        entered, cancelled = threading.Event(), threading.Event()

        def speak(text, should_stop):
            if text == "old left":
                entered.set()
                deadline = time.monotonic() + 1
                while not should_stop():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("not interrupted")
                    time.sleep(0.002)
                cancelled.set()

        self.backend.speak.side_effect = speak
        self.worker.start()
        self.worker.submit(SpeechMessage("old left", 1, "left"))
        self.assertTrue(entered.wait(1))
        self.assertTrue(self.playback.snapshot()[0])
        self.validity.update(2, "right", time.monotonic() + 5)
        self.worker.submit(SpeechMessage("new right", 2, "right"))
        self.assertTrue(cancelled.wait(1))
        until(lambda: self.backend.speak.call_count == 2)

    def test_priority_and_single_pending_slot(self):
        self.make()
        self.worker.submit(SpeechMessage("left", 1, "left"))
        self.worker.submit(SpeechMessage("cancelled", 1, priority=90))
        self.assertFalse(self.worker.submit(SpeechMessage("obsolete left", 1, "left")))
        self.worker.start()
        until(lambda: self.backend.speak.called)
        self.assertEqual(self.backend.speak.call_args.args[0], "cancelled")

    def test_backend_error_is_explicit_and_gate_resumes(self):
        self.make()
        self.backend.speak.side_effect = RuntimeError("speaker failed")
        with self.assertLogs("homesense.audio.speech_output", level="ERROR"):
            self.worker.start()
            self.worker.submit(SpeechMessage("hello", 1))
            until(lambda: self.worker.error is not None)
        self.assertFalse(self.playback.snapshot()[0])

    def test_missing_tts_binary_is_explicit(self):
        with patch("shutil.which", return_value=None), self.assertRaises(RuntimeError):
            EspeakSpeech(AudioSettings())

    def test_failed_speech_thread_start_is_safe_to_stop(self):
        self.make()
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("cannot start")):
            with self.assertRaises(RuntimeError):
                self.worker.start()
        self.assertTrue(self.worker.stop())


class SpeechCleanupTests(unittest.TestCase):
    def test_blocked_backend_is_retained_and_shutdown_can_be_retried(self):
        entered, release = threading.Event(), threading.Event()
        backend = Mock()
        backend.speak.side_effect = lambda *args: (entered.set(), release.wait(2))
        validity = SpeechValidity()
        worker = SpeechWorker(backend, validity, PlaybackGate(tail=0), listening_gap=0)
        worker.start()
        try:
            worker.submit(SpeechMessage("test", 0))
            self.assertTrue(entered.wait(1))
            self.assertFalse(worker.stop(0.01))
            self.assertTrue(worker._thread.is_alive())
        finally:
            release.set()
            self.assertTrue(worker.stop(1))

    def test_each_external_process_cleanup_attempted_after_one_raises(self):
        synth, player = Mock(), Mock()
        synth.poll.return_value = player.poll.return_value = None
        with patch("shutil.which", return_value="/fake/espeak-ng"):
            backend = EspeakSpeech(AudioSettings(output_device="default"))
        failure = RuntimeError("player termination failed")
        with patch("subprocess.Popen", side_effect=[synth, player]), \
             patch.object(backend, "_terminate", side_effect=[failure, None]) as cleanup, \
             self.assertLogs("homesense.audio.speech_output", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "player termination"):
                backend.speak("test", Mock(side_effect=[False, True]))
        self.assertEqual(cleanup.call_count, 2)
        self.assertIs(cleanup.call_args_list[1].args[0], synth)
