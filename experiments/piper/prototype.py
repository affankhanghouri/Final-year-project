"""Isolated resident neural voice; only the existing SpeechWorker calls speak."""

import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import threading
import time

from homesense.audio.speech_output import EspeakSpeech


class PiperSpeech:
    """One resident CPU session, bounded fixed-phrase PCM cache, native ALSA output.

    Entire short phrases are synthesized before playback to avoid inference
    starvation between words. Cancellation is checked before/after ONNX, before
    opening output, while feeding the pipe, and while waiting for the player.
    A single native ONNX call is not preemptible; fixed phrases are pre-generated.
    """

    def __init__(self, model, settings, *, threads=1):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from piper import PiperVoice, PiperConfig, SynthesisConfig

        model = Path(model)
        if not model.is_file() or not Path(str(model)+'.json').is_file():
            raise FileNotFoundError(f'Missing local Piper model/config: {model}')
        self.player = shutil.which('aplay')
        if not self.player:
            raise RuntimeError('The trial needs the existing aplay executable')
        self.settings = settings
        started = time.monotonic()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.voice = PiperVoice(session=ort.InferenceSession(str(model), sess_options=options,
            providers=['CPUExecutionProvider']),
            config=PiperConfig.from_dict(json.loads(Path(str(model)+'.json').read_text())))
        self.synthesis_config = SynthesisConfig(volume=settings.volume)
        self.load_seconds = time.monotonic()-started
        self.sample_rate = self.voice.config.sample_rate
        self.cache = {}
        self.preparation = []
        self.reset_observations()

    def reset_observations(self):
        self.calls = []
        self.entered, self.spawned = threading.Event(), threading.Event()
        self.worker = None

    def render(self, text, should_stop=lambda: False):
        """Return RAM-only PCM; never download resources or save speech."""
        if len(text) > 160:
            raise ValueError('Trial phrases must be at most 160 characters')
        if should_stop():
            return b''
        if text in self.cache:
            return self.cache[text]
        pcm = bytearray()
        for chunk in self.voice.synthesize(text, syn_config=self.synthesis_config):
            if should_stop():
                return b''
            if (chunk.sample_rate, chunk.sample_width, chunk.sample_channels) != (self.sample_rate, 2, 1):
                raise RuntimeError('Piper output format changed during synthesis')
            pcm.extend(chunk.audio_int16_bytes)
            if len(pcm) > 4*1024*1024:
                raise RuntimeError('Synthesized phrase exceeded trial memory budget')
        return bytes(pcm)

    def prepare(self, phrases):
        """Explicit fixed vocabulary only; at most 16 cached phrases / 4 MiB total."""
        for text in dict.fromkeys(phrases):
            if text in self.cache:
                continue
            if len(self.cache) >= 16:
                raise ValueError('Fixed phrase cache is full')
            started, cpu = time.monotonic(), time.process_time()
            pcm = self.render(text)
            if sum(map(len, self.cache.values()))+len(pcm) > 4*1024*1024:
                raise ValueError('Fixed phrase cache exceeds 4 MiB')
            self.cache[text] = pcm
            self.preparation.append(dict(text=text, synthesis_seconds=time.monotonic()-started,
                cpu_seconds=time.process_time()-cpu, audio_seconds=len(pcm)/(2*self.sample_rate), pcm_bytes=len(pcm)))

    def speak(self, text, should_stop):
        call = dict(text=text, entered_at=time.monotonic(), launches=[], stderr=[], error=None,
                    first_stop_observed_at=None, worker_queue_seconds=self.worker.stats.queue_delay_seconds
                    if self.worker else None, cache_hit=text in self.cache, acoustic_start_seconds=None)
        self.calls.append(call)
        self.entered.set()
        process = None

        def stop():
            result = should_stop()
            if result and call['first_stop_observed_at'] is None:
                call['first_stop_observed_at'] = time.monotonic()
            return result

        try:
            synth = time.monotonic()
            pcm = self.render(text, stop)
            call.update(synthesis_seconds=time.monotonic()-synth, sample_rate=self.sample_rate,
                        channels=1, sample_width=2, audio_seconds=len(pcm)/(2*self.sample_rate))
            if stop() or not pcm:
                return
            with tempfile.TemporaryFile() as errors:
                command = [self.player, '-q', '-D', self.settings.output_device or 'default',
                    '-t', 'raw', '-f', 'S16_LE', '-r', str(self.sample_rate), '-c', '1',
                    '--buffer-time=200000', '--period-time=50000']
                at = time.monotonic()
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=errors, start_new_session=True)
                call['launches'].append(dict(command=command, pid=process.pid,
                    requested_at=at, spawned_at=time.monotonic()))
                self.spawned.set()
                try:
                    os.set_blocking(process.stdin.fileno(), False)
                    offset = 0
                    deadline = time.monotonic()+call['audio_seconds']+5
                    with selectors.DefaultSelector() as selector:
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                        while offset < len(pcm) and process.poll() is None:
                            if stop():
                                return
                            if time.monotonic() >= deadline:
                                raise TimeoutError('Playback pipe exceeded diagnostic budget')
                            if selector.select(.01):
                                try:
                                    count = os.write(process.stdin.fileno(), pcm[offset:offset+4096])
                                except BlockingIOError:
                                    continue
                                except BrokenPipeError:
                                    break
                                if offset == 0:
                                    call['first_pipe_write_at'] = time.monotonic()
                                offset += count
                    process.stdin.close()
                    while process.poll() is None:
                        if stop():
                            return
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Player exceeded diagnostic budget')
                        time.sleep(.01)
                    if not stop() and (process.returncode or offset != len(pcm)):
                        errors.seek(0)
                        raise RuntimeError(f'Piper playback failed, exit {process.returncode}: '
                                           +errors.read().decode(errors='replace')[-8192:])
                finally:
                    try:
                        EspeakSpeech._terminate(process)
                    finally:
                        process.stdin.close()
                        errors.seek(0)
                        call['stderr'].append(dict(executable=self.player,
                            text=errors.read().decode(errors='replace')[-8192:]))
                        call['returncodes_after_cleanup'] = [process.poll()]
        except Exception as error:
            call['error'] = str(error)
            raise
        finally:
            call['returned_at'] = time.monotonic()
            call['backend_seconds'] = call['returned_at']-call['entered_at']
