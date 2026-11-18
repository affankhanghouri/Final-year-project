"""Actual isolated CLI process, with a fake native executable and synthetic WAV."""

import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from homesense.audio.comparison import wav_bytes


class SpeechDiagnosticIntegrationTests(unittest.TestCase):
    def test_comparison_replays_verified_pcm_and_reports_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            binary=root/'fake whisper'
            binary.write_text('''#!/usr/bin/env python3
import io,sys,wave
with wave.open(io.BytesIO(sys.stdin.buffer.read()), 'rb') as audio:
    assert (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())==(16000,1,2)
    assert audio.readframes(audio.getnframes())==b'\\x64\\x00'*1600
print('Find my keys.')
''')
            binary.chmod(0o700)
            model=root/'fake model.bin'; model.touch()
            (root/'clip.wav').write_bytes(wav_bytes(b'\x64\x00'*1600))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'clips':[{'file':'clip.wav','reference':'Find my keys.'}]}))
            report=root/'report.json'
            result=subprocess.run([sys.executable,'-B','scripts/speech_diagnostic.py','compare',str(manifest),
                '--backends','whisper','--whisper-cli',str(binary),'--whisper-model',str(model),
                '--output',str(report)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(report.read_text())
            row=data['rows'][0]
            self.assertTrue(row['exact_match'])
            self.assertEqual(row['wer'],0)
            self.assertGreater(row['cold_wall_seconds'],0)
            self.assertGreater(row['peak_process_rss_kib'],0)
            self.assertIsNone(row['wake_gate_commands'])
            self.assertEqual(row['audio_sha256'],hashlib.sha256((root/'clip.wav').read_bytes()).hexdigest())
            self.assertEqual(sorted(p.name for p in root.iterdir()),['clip.wav','fake model.bin','fake whisper','manifest.json','report.json'])

    def test_record_cli_without_consent_fails_before_device_access(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'never-created'
            result=subprocess.run([sys.executable,'-B','scripts/speech_diagnostic.py','record','--output-dir',str(target)],
                                   capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--consent-recording',result.stderr)
            self.assertFalse(target.exists())
