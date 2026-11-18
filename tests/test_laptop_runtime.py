"""Laptop rendering, telemetry and subprocess regressions, without live devices."""

from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from homesense.application import ApplicationError, ApplicationState
from homesense.audio.input import PlaybackGate
from homesense.audio.speech_output import EspeakSpeech, SpeechMessage, SpeechStats, SpeechValidity, SpeechWorker
from homesense.camera import Frame
from homesense.config import AudioSettings
from homesense.detector import Detection
from homesense.diagnostics import RuntimeMetrics
from homesense.guidance import Direction, GuidanceStatus
from homesense.interaction.state import InteractionEvent, InteractionEventKind
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.preview import LaptopPreview
from homesense.stability import Visibility
from test_voice_interaction import SnapshotFactory, Clock


def fake_cv():
    return SimpleNamespace(WINDOW_NORMAL=0, FONT_HERSHEY_SIMPLEX=0, WND_PROP_VISIBLE=0,
        namedWindow=Mock(), destroyWindow=Mock(), imshow=Mock(), putText=Mock(), rectangle=Mock(),
        resize=lambda image, shape: np.zeros((shape[1], shape[0], 3), dtype=np.uint8),
        waitKey=Mock(return_value=-1), getWindowProperty=Mock(return_value=1))


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.cv = Clock(), fake_cv()
        self.snapshot = SnapshotFactory(self.clock).make()
        self.snapshot = replace(self.snapshot, batch=replace(self.snapshot.batch,
            detections=(Detection(1, 'phone', .91, (1, 1, 8, 8)),)))
        self.app = Mock()
        self.app.latest_frame.return_value = Frame(42, self.clock(), np.zeros((10, 10, 3), dtype=np.uint8))
        self.preview = LaptopPreview(cv=self.cv, clock=self.clock)
        self.addCleanup(self.preview.close)

    def test_idle_never_requests_camera_and_close_is_repeatable(self):
        idle = replace(self.snapshot, state=ApplicationState.IDLE, session=None)
        self.assertTrue(self.preview.update(idle, self.app))
        self.app.latest_frame.assert_not_called()
        self.cv.waitKey.assert_called_once_with(1)
        self.preview.close()
        self.preview.close()
        self.cv.destroyWindow.assert_called_once()

    def test_idle_explains_camera_off_and_displays_command_instruction(self):
        self.preview.update(replace(self.snapshot, state=ApplicationState.IDLE, session=None), self.app)
        text = ' '.join(call.args[1] for call in self.cv.putText.call_args_list)
        self.assertIn('Camera OFF', text)
        self.assertIn('Find my phone', text)
        self.assertIn('No active inference', text)
        self.app.latest_frame.assert_not_called()

    def test_failed_and_stopping_states_are_not_reported_as_idle(self):
        failure = ApplicationError('camera.worker', RuntimeError('could not open camera 0'))
        self.preview.update(replace(self.snapshot, state=ApplicationState.FAILED, errors=(failure,)), self.app)
        text = ' '.join(call.args[1] for call in self.cv.putText.call_args_list)
        self.assertIn('camera.worker: could not open camera 0', text)
        self.assertNotIn('Camera OFF. After feedback', text)
        self.assertIn('Target: failed', text)
        self.clock.now += 1
        self.cv.putText.reset_mock()
        self.preview.update(replace(self.snapshot, state=ApplicationState.STOPPING), self.app)
        text = ' '.join(call.args[1] for call in self.cv.putText.call_args_list)
        self.assertIn('Target: stopping', text)
        self.assertIn('waiting for resource cleanup', text)
        self.app.latest_frame.assert_not_called()

    def test_waiting_for_capture_is_distinguished_from_idle_and_stale_capture(self):
        self.app.latest_frame.return_value = None
        self.preview.update(self.snapshot, self.app)
        text = ' '.join(call.args[1] for call in self.cv.putText.call_args_list)
        self.assertIn('Search active; waiting for the first camera frame', text)
        self.clock.now += 1
        self.cv.putText.reset_mock()
        self.app.latest_frame.return_value = self.snapshot.batch.frame
        self.preview.update(self.snapshot, self.app)
        text = ' '.join(call.args[1] for call in self.cv.putText.call_args_list)
        self.assertIn('Camera frame age: 1.00s (STALE)', text)

    def test_exact_detection_image_is_copied_and_not_drawn_on_live_frame(self):
        batch = replace(self.snapshot.batch, detections=(Detection(1, 'phone', .91, (1, 1, 8, 8)),))
        self.preview.update(replace(self.snapshot, batch=batch), self.app)
        self.app.latest_frame.assert_called_once_with()
        drawn_image = self.cv.rectangle.call_args.args[0]
        self.assertFalse(np.shares_memory(drawn_image, batch.frame.image))
        self.assertFalse(np.shares_memory(drawn_image, self.app.latest_frame.return_value.image))
        self.assertTrue(any('phone 0.91' in call.args[1] for call in self.cv.putText.call_args_list))
        self.assertTrue(any('Target: phone' in call.args[1] for call in self.cv.putText.call_args_list))

    def test_stale_cancelled_and_wrong_session_boxes_are_not_rendered(self):
        other = replace(self.snapshot.batch, session_id=SnapshotFactory(self.clock).session.session_id)
        cases = [replace(self.snapshot, batch=other),
                 replace(self.snapshot, state=ApplicationState.IDLE), self.snapshot]
        for case in cases:
            self.preview.update(case, self.app)
            self.clock.now += 1
        self.cv.rectangle.assert_not_called()

    def test_render_is_throttled_and_window_close_requests_exit(self):
        self.preview.update(self.snapshot, self.app)
        self.preview.update(self.snapshot, self.app)
        self.app.latest_frame.assert_called_once()
        self.clock.now += .1
        self.cv.waitKey.return_value = ord('q')
        self.assertFalse(self.preview.update(self.snapshot, self.app))
        self.clock.now += .1
        self.cv.waitKey.return_value = -1
        self.cv.getWindowProperty.return_value = 0
        self.assertFalse(self.preview.update(self.snapshot, self.app))

    def test_gui_failure_still_allows_owned_window_cleanup(self):
        self.cv.imshow.side_effect = RuntimeError('display failed')
        with self.assertRaisesRegex(RuntimeError, 'display failed'):
            self.preview.update(self.snapshot, self.app)
        self.preview.close()
        self.cv.destroyWindow.assert_called_once()

    def test_render_cannot_be_moved_to_camera_or_audio_thread(self):
        errors = []
        def render():
            try:
                self.preview.update(self.snapshot, self.app)
            except RuntimeError as error:
                errors.append(error)
        thread = threading.Thread(target=render)
        thread.start()
        thread.join(1)
        self.assertEqual(len(errors), 1)
        self.cv.namedWindow.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'Linux desktop environment check')
    def test_missing_desktop_fails_explicitly_without_initializing_gui(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'graphical desktop'):
                LaptopPreview()


class GuidanceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = SnapshotFactory(self.clock)
        self.messages = []
        self.output = Mock()
        self.output.submit.side_effect = lambda message: self.messages.append(message) or True
        self.validity = SpeechValidity(clock=self.clock)
        self.feedback = FeedbackManager(self.output, self.validity, clock=self.clock)

    def test_brief_loss_does_not_repeat_same_direction_after_cooldown(self):
        self.feedback.on_snapshot(self.factory.make())
        self.clock.now += 4
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCERTAIN, status=GuidanceStatus.UNCERTAIN))
        self.assertFalse(self.validity.allows(self.messages[-1]))
        self.clock.now += .2
        self.feedback.on_snapshot(self.factory.make())
        self.assertEqual(len(self.messages), 2)
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertEqual(self.messages[-1].text, 'Phone on your left.')
        self.assertEqual(len(self.messages), 3)

    def test_position_gap_with_visible_object_is_not_announced_as_loss(self):
        self.feedback.on_snapshot(self.factory.make())
        for _ in range(5):
            self.clock.now += 1
            self.feedback.on_snapshot(self.factory.make(status=GuidanceStatus.STALE))
        self.assertEqual(len(self.messages), 2)

    def test_initial_absence_prompt_is_not_a_periodic_announcement(self):
        for advance in (5, 20, 20):
            self.clock.now += advance
            self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCONFIRMED, status=GuidanceStatus.UNCONFIRMED))
        self.assertEqual(sum("can't see" in message.text for message in self.messages), 1)

    def test_recovery_after_persistent_announced_loss_is_meaningful(self):
        self.feedback.on_snapshot(self.factory.make())
        self.clock.now += 4
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCERTAIN, status=GuidanceStatus.UNCERTAIN))
        self.clock.now += 1
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCERTAIN, status=GuidanceStatus.UNCERTAIN))
        self.assertIn("can't confirm", self.messages[-1].text)
        self.clock.now += 1
        self.feedback.on_snapshot(self.factory.make(direction=Direction.RIGHT))
        self.assertEqual(self.messages[-1].text, 'Phone on your right.')


class MetricsTests(unittest.TestCase):
    def test_metrics_are_interval_based_and_do_not_construct_resources(self):
        clock = Clock()
        snapshot = SnapshotFactory(clock).make()
        snapshot = replace(snapshot, stats=replace(snapshot.stats, processed_fps=10,
                           last_inference_seconds=.15, last_result_age=.2, rejected_results=2))
        app = Mock()
        app.latest_frame.return_value = Frame(50, 105, np.zeros((1, 1, 3), dtype=np.uint8))
        runtime = SimpleNamespace(interaction=SimpleNamespace(snapshot=snapshot, coordinator=app),
                                  speech=SimpleNamespace(stats=SpeechStats(queue_delay_seconds=.2, playback_seconds=1)))
        cpu = Mock(return_value=10)
        metrics = RuntimeMetrics(5, clock=clock, cpu_clock=cpu)
        self.assertIsNone(metrics.sample(runtime))
        app.latest_frame.assert_not_called()
        clock.now += 5
        cpu.return_value += 7.5
        values = metrics.sample(runtime)
        self.assertEqual(values['cpu_percent'], 150)
        self.assertEqual(values['camera_fps'], 10)
        self.assertEqual(values['inference_ms'], 150)
        self.assertEqual(values['tts_queue_ms'], 200)
        self.assertEqual(values['rejected_results'], 2)


class PlaybackRegressionTests(unittest.TestCase):
    def backend(self, **kwargs):
        with patch('shutil.which', return_value='/fake/espeak-ng'):
            return EspeakSpeech(AudioSettings(**kwargs))

    def test_genuine_playback_error_reports_device_exit_code_and_stderr(self):
        process = Mock(returncode=3)
        process.poll.return_value = 3
        def launch(*args, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            kwargs['stderr'].write(b'invalid voice or audio device')
            return process
        with patch('subprocess.Popen', side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, 'exited 3: invalid voice or audio device'):
                self.backend().speak('test', lambda: False)

    def test_output_routing_keeps_alsa_device_and_closes_pipe(self):
        synth, player = Mock(returncode=0), Mock(returncode=0)
        synth.poll.return_value = player.poll.return_value = 0
        with patch('subprocess.Popen', side_effect=[synth, player]) as launch:
            self.backend(output_device='hw:1,0').speak('test', lambda: False)
        self.assertEqual(launch.call_args_list[1].args[0], ['aplay', '-q', '-D', 'hw:1,0'])
        synth.stdout.close.assert_called_once()
        self.assertTrue(all(call.kwargs['start_new_session'] for call in launch.call_args_list))

    def test_cancel_after_last_poll_does_not_turn_into_playback_failure(self):
        process = Mock(returncode=-15)
        process.poll.return_value = -15
        with patch('subprocess.Popen', return_value=process):
            self.backend().speak('test', Mock(side_effect=[False, True]))

    def test_low_priority_duplicates_do_not_overlap_and_metrics_are_bounded(self):
        entered, release = threading.Event(), threading.Event()
        backend = Mock()
        def speak(text, stop):
            entered.set()
            while not release.wait(.002) and not stop():
                pass
        backend.speak.side_effect = speak
        validity = SpeechValidity()
        worker = SpeechWorker(backend, validity, PlaybackGate(tail=0), listening_gap=0)
        worker.start()
        try:
            message = SpeechMessage('phone left', 0, priority=10)
            worker.submit(message)
            self.assertTrue(entered.wait(1))
            self.assertFalse(worker.submit(message))
            for i in range(20):
                worker.submit(SpeechMessage(str(i), 0, priority=10))
            worker.submit(SpeechMessage('Search cancelled.', 0, priority=90))
            self.assertFalse(worker.submit(message))
            self.assertLessEqual(worker.stats.pending, 1)
            release.set()
            deadline = time.monotonic() + 1
            while worker.busy and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertEqual(backend.speak.call_args.args[0], 'Search cancelled.')
            self.assertGreaterEqual(worker.stats.cancelled, 1)
            self.assertGreaterEqual(worker.stats.replaced, 1)
            self.assertGreaterEqual(worker.stats.queue_delay_seconds, 0)
        finally:
            release.set()
            self.assertTrue(worker.stop(1))

    def test_failure_drops_unplayable_pending_work_and_releases_microphone_gate(self):
        entered, release = threading.Event(), threading.Event()
        backend = Mock()
        def speak(text, stop):
            entered.set()
            if not release.wait(1):
                raise AssertionError('test did not release backend')
            raise RuntimeError('speaker disconnected')
        backend.speak.side_effect = speak
        playback = PlaybackGate(tail=0)
        worker = SpeechWorker(backend, SpeechValidity(), playback, listening_gap=0)
        worker.start()
        try:
            with self.assertLogs('homesense.audio.speech_output', level='ERROR'):
                worker.submit(SpeechMessage('first', 0))
                self.assertTrue(entered.wait(1))
                self.assertTrue(playback.snapshot()[0])
                worker.submit(SpeechMessage('pending', 0))
                release.set()
                self.assertTrue(worker.stop(1))
            self.assertIsInstance(worker.error, RuntimeError)
            self.assertFalse(worker.busy)
            self.assertFalse(playback.snapshot()[0])
            self.assertEqual(worker.stats.failed, 1)
        finally:
            release.set()
            worker.stop(1)


class EntryPointTests(unittest.TestCase):
    def test_preview_quit_and_runtime_close_error_both_release_window(self):
        from scripts import run_homesense
        for failure in (None, RuntimeError('cleanup failed')):
            with self.subTest(failure=failure):
                runtime = Mock(failed=False)
                runtime.close.side_effect = failure
                with patch('sys.argv', ['run_homesense.py', '--preview', '--text', '--console-feedback']), \
                     patch.object(run_homesense, 'compose', return_value=runtime), \
                     patch('homesense.preview.LaptopPreview') as constructor, \
                     patch.object(run_homesense.logging, 'exception'):
                    constructor.return_value.update.return_value = False
                    self.assertEqual(run_homesense.main(), int(failure is not None))
                    runtime.close.assert_called_once()
                    constructor.return_value.close.assert_called_once()

    def test_headless_default_does_not_create_preview(self):
        from scripts import run_homesense
        runtime = Mock(failed=False)
        runtime.step.side_effect = KeyboardInterrupt
        with patch('sys.argv', ['run_homesense.py', '--text', '--console-feedback']), \
             patch.object(run_homesense, 'compose', return_value=runtime), \
             patch('homesense.preview.LaptopPreview') as constructor:
            self.assertEqual(run_homesense.main(), 0)
            constructor.assert_not_called()
            runtime.close.assert_called_once()

    @unittest.skipUnless(os.name == 'posix', 'terminal process groups require POSIX')
    def test_terminal_sigint_does_not_kill_speech_child_or_create_false_error(self):
        with tempfile.TemporaryDirectory() as folder:
            ready, fake = Path(folder)/'ready', Path(folder)/'fake-espeak'
            fake.write_text(f'#!{sys.executable}\nimport time\nfrom pathlib import Path\nPath({str(ready)!r}).touch()\ntime.sleep(10)\n')
            fake.chmod(0o700)
            driver = '''import json,sys,time
from pathlib import Path
from homesense.audio.speech_output import EspeakSpeech,SpeechWorker,SpeechValidity,SpeechMessage
from homesense.audio.input import PlaybackGate
from homesense.config import AudioSettings
backend=EspeakSpeech(AudioSettings()); backend.binary=sys.argv[1]
worker=SpeechWorker(backend,SpeechValidity(),PlaybackGate(tail=0),listening_gap=0)
worker.start(); worker.submit(SpeechMessage("test",0))
deadline=time.monotonic()+3
while not Path(sys.argv[2]).exists():
    if time.monotonic()>deadline: raise RuntimeError("child did not start")
    time.sleep(.01)
try:
    print("READY",flush=True)
    time.sleep(10)
except KeyboardInterrupt:
    time.sleep(.15)
finally:
    worker.stop(1)
print(json.dumps({"error":str(worker.error) if worker.error else None}))
'''
            child = subprocess.Popen([sys.executable, '-c', driver, str(fake), str(ready)],
                start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'READY')
                os.killpg(child.pid, signal.SIGINT)
                out, err = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, err)
                self.assertIsNone(json.loads(out)['error'])
                self.assertNotIn('Speech output failed', err)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
