"""Speech diagnostics use synthetic PCM/fake engines, never a microphone."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import wave

from homesense.audio.comparison import (Clip, PROMPTS, VoskReplay, WhisperCppReplay,
    grammar_phrases, load_manifest, pcm_stats, score, wake_replay, wav_bytes, word_errors)
from homesense.audio.speech_recognition import VoskRecognizer
from homesense.config import AudioSettings
from scripts import speech_diagnostic as cli


class AudioContractTests(unittest.TestCase):
    def test_pcm16_roundtrip_and_statistics(self):
        pcm = b'\xff\x7f\x00\x80' * 800
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.wav'
            path.write_bytes(wav_bytes(pcm))
            clip = Clip.load(path, 'hello')
            self.assertEqual(clip.pcm, pcm)
        stats = pcm_stats(pcm)
        self.assertEqual(stats['peak_pcm'], 32768)
        self.assertEqual(stats['duration_seconds'], 0.1)
        self.assertEqual(stats['clipped_fraction'], 1)

    def test_stereo_and_wrong_rate_rejected_instead_of_mislabelled(self):
        for channels, rate in [(2, 16000), (1, 44100)]:
            with self.subTest(channels=channels, rate=rate), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'wrong.wav'
                with wave.open(str(path), 'wb') as file:
                    file.setparams((channels, 2, rate, 0, 'NONE', ''))
                    file.writeframes(b'\0' * 3200)
                with self.assertRaisesRegex(ValueError, '16000 Hz'):
                    Clip.load(path, 'hello')

    def test_modified_audio_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.wav'
            path.write_bytes(wav_bytes(b'\0' * 3200))
            with self.assertRaisesRegex(ValueError, 'changed'):
                Clip.load(path, '', 'incorrect hash')

    def test_manifest_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'clips':[{'file':'../private.wav','reference':''}]}))
            with self.assertRaisesRegex(ValueError, 'inside'):
                load_manifest(path)

    def test_empty_and_odd_pcm_rejected(self):
        for pcm in (b'', b'x'):
            with self.assertRaises(ValueError):
                pcm_stats(pcm)

    def test_truncated_wav_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truncated.wav'
            path.write_bytes(wav_bytes(b'\0' * 3200)[:-2])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                Clip.load(path, '')


class ConsentTests(unittest.TestCase):
    def test_no_consent_never_queries_or_opens_microphone(self):
        args = cli.parser().parse_args(['record','--output-dir','recordings/test'])
        with patch.object(cli, 'inspect_device') as inspect, patch.object(cli, 'SoundDeviceInput') as source:
            with self.assertRaisesRegex(ValueError, 'consent'):
                cli.record(args)
        inspect.assert_not_called()
        source.assert_not_called()

    def test_inspection_only_checks_format_never_opens_stream(self):
        sd = Mock()
        sd.query_devices.return_value = {'default_samplerate':44100}
        with patch.dict('sys.modules', {'sounddevice':sd}):
            result = cli.inspect_device(12)
        self.assertEqual(result['requested']['sample_rate'],16000)
        sd.RawInputStream.assert_not_called()
        sd.check_input_settings.assert_called_once_with(device=12,samplerate=16000,channels=1,dtype='int16')

    def test_recording_requires_enter_for_each_clip_and_closes_source(self):
        source = Mock()
        source._stream = SimpleNamespace(samplerate=16000,channels=1,dtype='int16',blocksize=1600,latency=0.1)
        source.read.return_value = b'\0' * 3200
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'trial'
            args = cli.parser().parse_args(['record','--consent-recording','--output-dir',str(output),'--repetitions','1','--seconds','0.1'])
            with patch.object(cli,'inspect_device',return_value={}), patch.object(cli,'SoundDeviceInput',return_value=source), patch('builtins.input',return_value='') as consent, patch('builtins.print'):
                cli.record(args)
            self.assertEqual(consent.call_count,len(PROMPTS))
            self.assertEqual(source.close.call_count,len(PROMPTS))
            manifest, clips = load_manifest(output / 'manifest.json')
            self.assertEqual([c.reference for c in clips],list(PROMPTS))
            self.assertEqual((output.stat().st_mode & 0o777),0o700)
            self.assertEqual((clips[0].path.stat().st_mode & 0o777),0o600)

    def test_capture_failure_still_closes_stream(self):
        source = Mock()
        source._stream = SimpleNamespace(samplerate=16000,channels=1,dtype='int16',blocksize=1600,latency=0.1)
        source.read.side_effect = RuntimeError('device disconnected')
        with tempfile.TemporaryDirectory() as directory:
            args = cli.parser().parse_args(['record','--consent-recording','--output-dir',str(Path(directory)/'trial'),'--repetitions','1'])
            with patch.object(cli,'inspect_device',return_value={}), patch.object(cli,'SoundDeviceInput',return_value=source), patch('builtins.input',return_value=''), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'disconnected'):
                    cli.record(args)
            source.close.assert_called_once()
            self.assertEqual(list((Path(directory)/'trial').glob('*.wav')),[])

    def test_repeated_trial_path_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            args = cli.parser().parse_args(['record','--consent-recording','--output-dir',directory])
            with patch.object(cli,'inspect_device',return_value={}), self.assertRaises(FileExistsError):
                cli.record(args)


class VoskDiagnosticTests(unittest.TestCase):
    def fake(self, path):
        (path/'am').mkdir()
        (path/'am/final.mdl').touch()
        native = Mock()
        native.AcceptWaveform.return_value = False
        native.PartialResult.return_value = '{"partial":"hey agent"}'
        native.FinalResult.return_value = '{"text":"hey agent"}'
        module = SimpleNamespace(Model=Mock(),KaldiRecognizer=Mock(return_value=native))
        return module,native

    def test_quiet_speech_can_be_finalized_by_current_rms_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            module,native=self.fake(path)
            with patch.dict('sys.modules',{'vosk':module}):
                recognizer=VoskRecognizer(path)
                # Nonzero quiet PCM is below the 0.015 threshold, not actual silence.
                results=[recognizer.feed(b'\x64\x00'*1600) for _ in range(6)]
            self.assertTrue(results[-1].final)
            native.Reset.assert_called_once()

    def test_native_mode_does_not_force_quiet_speech_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            module,native=self.fake(path)
            with patch.dict('sys.modules',{'vosk':module}):
                recognizer=VoskRecognizer(path,software_endpoint=False)
                for _ in range(10):
                    self.assertFalse(recognizer.feed(b'\x64\x00'*1600).final)
                self.assertEqual(recognizer.finish().text,'hey agent')
            native.FinalResult.assert_called_once()

    def test_grammar_is_explicit_and_preserved_after_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            module,native=self.fake(path)
            with patch.dict('sys.modules',{'vosk':module}):
                recognizer=VoskRecognizer(path,grammar=('hey agent','[unk]'))
                recognizer.reset()
            self.assertEqual(json.loads(module.KaldiRecognizer.call_args.args[2]),['hey agent','[unk]'])
        self.assertIn('[unk]',grammar_phrases())

    def test_replay_keeps_all_endpoints_and_flushes_tail(self):
        recognizer=Mock()
        recognizer.feed.side_effect=[SimpleNamespace(text='hey agent',final=True),SimpleNamespace(text='find my phone',final=True)]
        recognizer.finish.return_value=SimpleNamespace(text='please')
        with patch('homesense.audio.comparison.VoskRecognizer',return_value=recognizer):
            backend=VoskReplay(Path('unused'),'vosk-current',AudioSettings())
            result=backend.transcribe(b'\0'*6400)
        self.assertEqual(result['transcript'],'hey agent find my phone please')
        self.assertEqual(wake_replay(result['segments']),['find my phone'])


class WhisperDiagnosticTests(unittest.TestCase):
    def test_whisper_receives_same_pcm_via_memory_only_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            model,binary=Path(directory)/'model',Path(directory)/'whisper-cli'
            model.touch(); binary.touch()
            backend=WhisperCppReplay(binary,model)
            with patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=b'Find my keys.\n',stderr=b'')) as run:
                result=backend.transcribe(b'\0'*3200)
            args=run.call_args.args[0]
            self.assertEqual(args[args.index('-f')+1],'-')
            self.assertIn('-ng',args)
            self.assertIn('-otxt',args)
            self.assertEqual(args[args.index('-of')+1],'-')
            with wave.open(io.BytesIO(run.call_args.kwargs['input']),'rb') as wav:
                self.assertEqual(wav.readframes(1600),b'\0'*3200)
            self.assertEqual(result['transcript'],'Find my keys.')
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),['model','whisper-cli'])

    def test_missing_model_fails_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'binary'; binary.touch()
            with self.assertRaises(FileNotFoundError):
                WhisperCppReplay(binary,Path(directory)/'missing')

    def test_native_failure_is_not_treated_as_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'placeholder'; path.touch()
            backend=WhisperCppReplay(path,path)
            with patch('subprocess.run',return_value=SimpleNamespace(returncode=1,stderr=b'bad model')), self.assertRaisesRegex(RuntimeError,'bad model'):
                backend.transcribe(b'\0'*3200)

    def test_timeout_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'placeholder'; path.touch()
            backend=WhisperCppReplay(path,path)
            with patch('subprocess.run',side_effect=subprocess.TimeoutExpired('whisper',1)), self.assertRaises(subprocess.TimeoutExpired):
                backend.transcribe(b'\0'*3200)


class MetricsTests(unittest.TestCase):
    def test_wer_is_normalized_and_can_exceed_one(self):
        self.assertEqual(word_errors('Find my keys.','find MY keys'),0)
        self.assertEqual(word_errors('keys','phone wallet keys'),2)
        self.assertEqual(score('keys','phone wallet keys')['wer'],2)
        self.assertIsNone(score('','hallucination')['wer'])

    def test_metrics_reflect_newly_supported_repeat_wording(self):
        result=score('Repeat the direction.','Repeat the direction.')
        self.assertTrue(result['exact_match'])
        self.assertTrue(result['parser_accepts_reference'])
        self.assertTrue(result['intent_target_match'])
        self.assertEqual(result['expected_intent'],'repeat_guidance')

    def test_failures_not_hidden_in_summary(self):
        rows=[dict(backend='whisper',error='timeout')]
        result=cli.summary(rows)['whisper']
        self.assertEqual(result['failures'],1)
        self.assertIsNone(result['corpus_wer'])
        self.assertIsNone(result['median_cold_seconds'])
        self.assertEqual(result['silence_samples_scored'],0)
        self.assertIsNone(result['non_speech_insertions'])

    def test_timeout_kills_entire_worker_process_group(self):
        process=Mock(pid=123)
        process.communicate.side_effect=[subprocess.TimeoutExpired('worker',1),(b'',b'')]
        with patch('subprocess.Popen',return_value=process),patch('os.killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                cli.run_isolated(['worker'],1)
        kill.assert_called_once_with(123,signal_number())


def signal_number():
    import signal
    return signal.SIGKILL
