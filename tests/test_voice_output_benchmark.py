"""Isolated diagnostic contracts; no physical audio, camera, model or network."""

import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import wave

from homesense.config import AudioSettings
from scripts import voice_output_benchmark as bench


def wav_bytes():
    memory = io.BytesIO()
    with wave.open(memory, 'wb') as output:
        output.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\x01\x00'*441)
    return memory.getvalue()


class SynthesisDiagnosticTests(unittest.TestCase):
    def executable(self, body):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name)/'synthetic-executable'
        path.write_text(f'#!{sys.executable}\n'+body)
        path.chmod(0o700)
        return str(path)

    def test_actual_pcm_length_with_streaming_placeholder_header(self):
        data = bytearray(wav_bytes())
        data[40:44] = (0x7fffffff).to_bytes(4, 'little')
        result = bench.wav_info(bytes(data))
        self.assertEqual(result['sample_rate'], 22050)
        self.assertAlmostEqual(result['audio_seconds'], .02)
        self.assertEqual(result['pcm_bytes'], 882)
        self.assertEqual(result['peak_amplitude'], 1)

    def test_pcm_offset_handles_extra_wav_chunks_and_partial_output(self):
        data = wav_bytes()
        augmented = data[:36]+b'JUNK\x02\x00\x00\x00xx'+data[36:]
        self.assertIsNone(bench.pcm_offset(b'RIFF'))
        self.assertEqual(bench.pcm_offset(augmented), 54)
        self.assertEqual(bench.pcm_offset(data), 44)

    def test_real_pipe_synthesis_reports_format_without_a_sound_device(self):
        executable = self.executable(f'import sys\nsys.stdout.buffer.write({wav_bytes()!r})\n')
        result = bench.synthesize(executable, AudioSettings(), 'test')
        self.assertEqual(result['channels'], 1)
        self.assertAlmostEqual(result['audio_seconds'], .02)
        self.assertIsNotNone(result['first_pcm_pipe_seconds'])
        self.assertLessEqual(result['first_pcm_pipe_seconds'], result['synthesis_seconds'])

    def test_native_failure_is_not_reported_as_success(self):
        executable = self.executable("import sys\nsys.stderr.write('diagnostic failure')\nsys.exit(4)\n")
        with self.assertRaisesRegex(RuntimeError, 'exit 4: diagnostic failure'):
            bench.synthesize(executable, AudioSettings(), 'test')

    def test_native_timeout_is_bounded_and_child_is_reaped(self):
        executable = self.executable('import time\ntime.sleep(10)\n')
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            bench.synthesize(executable, AudioSettings(), 'test', timeout=.05)
        self.assertLess(time.monotonic()-started, 2)


class TimedBackend:
    """A paced fake backend under the actual production SpeechWorker."""
    def __init__(self, settings):
        self.calls = []
        self.entered, self.spawned = threading.Event(), threading.Event()

    def speak(self, text, should_stop):
        call = dict(text=text, entered_at=time.monotonic(), first_stop_observed_at=None,
                    error=None, launches=[], stderr=[])
        self.calls.append(call)
        self.entered.set()
        self.spawned.set()
        deadline = time.monotonic()+.03
        while time.monotonic() < deadline:
            if should_stop():
                call['first_stop_observed_at'] = time.monotonic()
                break
            time.sleep(.001)
        call['returned_at'] = time.monotonic()
        call['backend_seconds'] = call['returned_at']-call['entered_at']


class WorkerDiagnosticTests(unittest.TestCase):
    def trial(self, name, gap=0):
        with patch.object(bench, 'ObservedEspeak', TimedBackend):
            result = bench.worker_trial(name, AudioSettings(), gap=gap)
        self.assertIsNone(result['error'])
        self.assertTrue(result['cleanup_complete'])
        self.assertEqual(result['stats']['pending'], 0)
        self.assertIsNone(result['acoustic_start_seconds'])
        return result

    def test_rapid_instructions_replace_pending_and_never_overlap(self):
        result = self.trial('rapid_latest')
        self.assertEqual([c['text'] for c in result['calls']], [bench.PHRASES[1], bench.PHRASES[4]])
        self.assertEqual(result['stats']['replaced'], 2)
        self.assertGreaterEqual(result['calls'][1]['entered_at'], result['calls'][0]['returned_at'])

    def test_urgent_and_stale_interrupts_are_api_timings_only(self):
        for scenario in ('urgent_interrupt', 'stale_interrupt'):
            with self.subTest(scenario=scenario):
                result = self.trial(scenario)
                self.assertTrue(result['cancellation_applicable'])
                self.assertGreaterEqual(result['stats']['cancelled'], 1)
                self.assertLess(result['cancellation_to_backend_return_seconds'], .2)
                self.assertIsNone(result['acoustic_cancellation_seconds'])

    def test_fresh_direction_can_expire_while_worker_observes_listening_gap(self):
        result = self.trial('expires_in_gap', gap=.2)
        self.assertEqual(result['stats']['submitted'], 2)
        self.assertEqual(result['stats']['started'], 1)
        self.assertEqual(result['accepted_without_backend_call'], [bench.PHRASES[5]])

    def test_stderr_warning_on_success_is_observed_without_changing_backend(self):
        with patch('homesense.audio.speech_output.shutil.which', return_value='/fake/espeak'):
            backend = bench.ObservedEspeak(AudioSettings())
        backend.calls.append(dict(launches=[], stderr=[]))
        with tempfile.TemporaryFile() as errors:
            errors.write(b'underrun!!!')
            errors.flush()
            from unittest.mock import Mock
            process = Mock()
            process.poll.return_value = 0
            backend._speak(['/fake/espeak'], 'test', lambda: False,
                lambda *a, **k: process, [], [('fake-player', errors)])
            self.assertEqual(backend.calls[0]['stderr'][0]['text'], 'underrun!!!')


class BenchmarkCliTests(unittest.TestCase):
    def test_help_and_missing_consent_never_start_backend(self):
        for args, code in ((['--help'], 0), (['--mode', 'audible'], 2),
                           (['--output-device', 'default'], 2), (['--listening-gap', 'nan'], 2)):
            with self.subTest(args=args), patch.object(bench, 'EspeakSpeech') as backend, \
                    patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main(args)
                self.assertEqual(error.exception.code, code)
                backend.assert_not_called()

    def test_report_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bench, 'EspeakSpeech') as backend:
            path = Path(folder)/'report.json'
            path.write_text('existing')
            with self.assertRaises(FileExistsError):
                bench.main(['--output', str(path)])
            self.assertEqual(path.read_text(), 'existing')
            backend.assert_not_called()

    def test_synthesis_only_reports_failure_and_never_runs_worker_trial(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(bench, 'EspeakSpeech') as backend, \
                patch.object(bench, 'worker_trial') as trial, patch('sys.stdout', io.StringIO()):
            backend.side_effect = RuntimeError('no synthesizer')
            path = Path(folder)/'failed.json'
            self.assertEqual(bench.main(['--mode', 'synthesis', '--output', str(path)]), 1)
            self.assertIn('no synthesizer', json.loads(path.read_text())['error'])
            trial.assert_not_called()
