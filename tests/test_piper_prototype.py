"""Fake neural synthesis and real disposable player processes; no audio hardware."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.piper.prototype import PiperSpeech
from experiments.piper import listen
from homesense.audio.input import PlaybackGate
from homesense.audio.speech_output import SpeechMessage, SpeechValidity, SpeechWorker, DeliveryStatus
from homesense.config import AudioSettings


def backend():
    speech = PiperSpeech.__new__(PiperSpeech)
    speech.voice = Mock()
    speech.voice.synthesize.return_value = iter((SimpleNamespace(
        sample_rate=22050, sample_width=2, sample_channels=1, audio_int16_bytes=b'\x01\x00'*2205),))
    speech.synthesis_config = object()
    speech.sample_rate = 22050
    speech.settings = AudioSettings(output_device='null')
    speech.cache, speech.preparation = {}, []
    speech.reset_observations()
    return speech


class PiperPrototypeTests(unittest.TestCase):
    def executable(self, body):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'fake-player'
        path.write_text(f'#!{sys.executable}\n'+body)
        path.chmod(0o700)
        return str(path)

    def test_fixed_phrase_cache_reuses_pcm_and_has_explicit_capacity(self):
        speech = backend()
        speech.prepare(('Turn left.', 'Turn left.'))
        first = speech.render('Turn left.')
        self.assertEqual(len(first), 4410)
        self.assertEqual(speech.render('Turn left.'), first)
        speech.voice.synthesize.assert_called_once()
        speech.cache.update({str(i): b'x' for i in range(15)})
        with self.assertRaisesRegex(ValueError, 'full'):
            speech.prepare(('new phrase',))

    def test_format_mismatch_cannot_reach_player(self):
        speech = backend()
        speech.voice.synthesize.return_value = iter((SimpleNamespace(
            sample_rate=16000, sample_width=2, sample_channels=1, audio_int16_bytes=b'00'),))
        with patch('subprocess.Popen') as popen, self.assertRaisesRegex(RuntimeError, 'format changed'):
            speech.speak('test', lambda: False)
        popen.assert_not_called()

    def test_invalidation_during_native_synthesis_prevents_any_playback(self):
        speech = backend()
        invalid = threading.Event()
        def chunks(*args, **kwargs):
            invalid.set()  # Model finishes after cancellation/replacement.
            yield SimpleNamespace(sample_rate=22050, sample_width=2, sample_channels=1, audio_int16_bytes=b'00')
        speech.voice.synthesize.side_effect = chunks
        with patch('subprocess.Popen') as popen:
            speech.speak('test', invalid.is_set)
        popen.assert_not_called()
        self.assertIsNotNone(speech.calls[0]['first_stop_observed_at'])

    def test_player_receives_correct_rate_pcm_and_preserves_success_warnings(self):
        speech = backend()
        speech.player = self.executable("import sys\nsys.stdin.buffer.read()\nsys.stderr.write('underrun warning')\n")
        speech.speak('test', lambda: False)
        call = speech.calls[0]
        self.assertEqual(call['returncodes_after_cleanup'], [0])
        self.assertIn('underrun warning', call['stderr'][0]['text'])
        command = call['launches'][0]['command']
        self.assertEqual(command[command.index('-r')+1], '22050')
        self.assertEqual(command[command.index('-f')+1], 'S16_LE')
        self.assertIsNone(call['acoustic_start_seconds'])

    def test_real_player_failure_is_explicit_and_child_is_reaped(self):
        speech = backend()
        speech.player = self.executable("import sys\nsys.stdin.buffer.read()\nsys.stderr.write('device unavailable')\nsys.exit(3)\n")
        with self.assertRaisesRegex(RuntimeError, 'device unavailable'):
            speech.speak('test', lambda: False)
        self.assertEqual(speech.calls[0]['returncodes_after_cleanup'], [3])

    def test_real_blocked_player_is_cancelled_by_existing_worker_and_cleaned(self):
        speech = backend()
        speech.cache['test'] = b'00'*30000
        speech.player = self.executable('import time\ntime.sleep(10)\n')
        validity = SpeechValidity()
        validity.update(0, 'left', time.monotonic()+20)
        worker = SpeechWorker(speech, validity, PlaybackGate(0), listening_gap=0)
        speech.worker = worker
        self.addCleanup(worker.stop)
        worker.start()
        message = SpeechMessage('test', 0, 'left', fresh_guidance=True)
        worker.submit(message)
        self.assertTrue(speech.spawned.wait(1))
        validity.update(1, None, None)
        self.assertTrue(worker.stop(1))
        self.assertIsNone(worker.error)
        self.assertEqual(worker.delivery(message.message_id).status, DeliveryStatus.INTERRUPTED)
        self.assertTrue(all(c is not None for c in speech.calls[0]['returncodes_after_cleanup']))

    def test_loader_uses_one_cpu_session_and_disables_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'voice.onnx'
            model.touch()
            Path(str(model)+'.json').write_text('{}')
            ort, piper = Mock(), Mock()
            piper.PiperConfig.from_dict.return_value.sample_rate = 22050
            piper.PiperVoice.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
            with patch.dict('sys.modules', {'onnxruntime': ort, 'piper': piper}), \
                    patch('shutil.which', return_value='/fake/aplay'):
                speech = PiperSpeech(model, AudioSettings(), threads=1)
            ort.disable_telemetry_events.assert_called_once()
            ort.InferenceSession.assert_called_once()
            options = ort.SessionOptions.return_value
            self.assertEqual(options.intra_op_num_threads, 1)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(speech.sample_rate, 22050)


class ListeningCliTests(unittest.TestCase):
    def test_explicit_playback_or_silent_choice_required_before_loading(self):
        with patch.object(listen, 'PiperSpeech') as model, patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                listen.main([])
            self.assertEqual(error.exception.code, 2)
            model.assert_not_called()

    def test_existing_report_is_preserved_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(listen, 'PiperSpeech') as model:
            path = Path(folder)/'old.json'
            path.write_text('preserved')
            with self.assertRaises(FileExistsError):
                listen.main(['--silent', '--output', str(path)])
            model.assert_not_called()
            self.assertEqual(path.read_text(), 'preserved')
