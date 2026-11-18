"""Finite, isolated TTS diagnostics; silent by default, never opens a microphone.

Silent worker trials use the real EspeakSpeech -> aplay ALSA `null` path. That
sink drains immediately: its wall times are not acoustic playback durations.
Audible mode explicitly exercises the selected production playback path.
"""

import argparse
from array import array
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import selectors
import subprocess
import sys
import threading
import time
import wave

from homesense.audio.input import PlaybackGate
from homesense.audio.speech_output import EspeakSpeech, SpeechMessage, SpeechValidity, SpeechWorker
from homesense.config import AudioSettings


PHRASES = (
    'HomeSense is ready.', 'Searching for your phone.', 'Turn left.', 'Turn right.',
    'Phone detected.', 'Phone on your left.', "I've lost sight of your phone.",
    'Search cancelled.',
)


def wav_info(data):
    """Use actual PCM length: eSpeak stdout WAV headers have placeholder sizes."""
    with wave.open(io.BytesIO(data), 'rb') as source:
        if source.getcomptype() != 'NONE':
            raise ValueError('Expected uncompressed PCM')
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        pcm = source.readframes(source.getnframes())
    if width != 2 or len(pcm) % (width*channels):
        raise ValueError('Expected complete signed PCM16 frames')
    samples = array('h', pcm)
    if sys.byteorder != 'little':
        samples.byteswap()
    return dict(sample_rate=rate, sample_width=width, channels=channels,
                pcm_format='signed PCM16 little endian', pcm_bytes=len(pcm),
                audio_seconds=len(pcm)/(rate*width*channels),
                peak_amplitude=max(map(abs, samples), default=0),
                clipped_samples=sum(abs(v) >= 32767 for v in samples),
                sha256=hashlib.sha256(data).hexdigest())


def pcm_offset(data):
    """Locate a WAV data chunk without assuming a 44-byte header."""
    if len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        return None
    offset = 12
    while offset+8 <= len(data):
        size = int.from_bytes(data[offset+4:offset+8], 'little')
        if data[offset:offset+4] == b'data':
            return offset+8
        offset += 8+size+(size % 2)
    return None


def synthesize(binary, settings, text, *, timeout=8):
    """Capture synthesized WAV only in RAM; timestamps refer to a pipe, not sound."""
    started = time.monotonic()
    args = [binary, '--stdout', '-v', settings.voice, '-s', str(settings.speaking_rate),
            '-a', str(round(settings.volume*100)), text]
    # stdout is drained during synthesis; stderr remains small for fixed phrases.
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    first_pcm = None
    data = bytearray()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic()-started >= timeout:
                    raise TimeoutError('Synthesis exceeded diagnostic budget')
                for key, _ in selector.select(.05):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    data.extend(block)
                    if len(data) > 4*1024*1024:
                        raise ValueError('Unexpectedly large synthesized output')
                    offset = pcm_offset(data)
                    if first_pcm is None and offset is not None and len(data) > offset:
                        first_pcm = time.monotonic()-started
        process.wait(timeout=max(.01, timeout-(time.monotonic()-started)))
        stderr = process.stderr.read().decode(errors='replace')[-8192:]
        if process.returncode:
            raise RuntimeError(f'Synthesis exit {process.returncode}: {stderr}')
        return dict(text=text, synthesis_seconds=time.monotonic()-started,
                    first_pcm_pipe_seconds=first_pcm, stderr=stderr, **wav_info(bytes(data)))
    finally:
        try:
            EspeakSpeech._terminate(process)
        finally:
            process.stdout.close()
            process.stderr.close()


class ObservedEspeak(EspeakSpeech):
    """Observe existing backend calls/launches without changing playback behavior."""

    def __init__(self, settings):
        super().__init__(settings)
        self.calls = []
        self.entered, self.spawned = threading.Event(), threading.Event()
        self.worker = None

    def speak(self, text, should_stop):
        call = dict(text=text, entered_at=time.monotonic(), launches=[], stderr=[],
                    first_stop_observed_at=None, error=None,
                    worker_queue_seconds=self.worker.stats.queue_delay_seconds if self.worker else None)
        self.calls.append(call)
        self.entered.set()

        def stop():
            result = should_stop()
            if result and call['first_stop_observed_at'] is None:
                call['first_stop_observed_at'] = time.monotonic()
            return result

        try:
            super().speak(text, stop)
        except Exception as error:
            call['error'] = str(error)
            raise
        finally:
            call['returned_at'] = time.monotonic()
            call['backend_seconds'] = call['returned_at']-call['entered_at']
            call['returncodes_after_cleanup'] = [p.poll() for p in self._processes]

    def _speak(self, args, text, should_stop, launch, processes, diagnostics):
        call = self.calls[-1]
        self._processes = processes

        def observed_launch(command, **kwargs):
            at = time.monotonic()
            process = launch(command, **kwargs)
            call['launches'].append(dict(command=command, pid=process.pid,
                requested_at=at, spawned_at=time.monotonic()))
            self.spawned.set()
            return process

        try:
            return super()._speak(args, text, should_stop, observed_launch, processes, diagnostics)
        finally:
            # Production retains stderr on failure only. Also inspect successful
            # calls here: ALSA can warn about underruns without a nonzero exit.
            for name, stream in diagnostics:
                # A cancelled child may still be writing: pread does not move
                # the shared open-file offset used by its stderr descriptor.
                size = os.fstat(stream.fileno()).st_size
                detail = os.pread(stream.fileno(), 8192, max(0, size-8192))
                call['stderr'].append(dict(executable=name, text=detail.decode(errors='replace')))
            call['returncodes_before_cleanup'] = [p.poll() for p in processes]


def wait_for(predicate, worker, timeout=12):
    deadline = time.monotonic()+timeout
    while not predicate():
        if worker.error:
            raise RuntimeError(str(worker.error))
        if time.monotonic() >= deadline:
            raise TimeoutError('Speech worker did not reach expected diagnostic state')
        time.sleep(.001)


def resource_snapshot():
    parent = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(at=time.monotonic(), parent_cpu=parent.ru_utime+parent.ru_stime,
                child_cpu=children.ru_utime+children.ru_stime,
                parent_peak_rss_mib=parent.ru_maxrss/1024,
                largest_child_peak_rss_mib=children.ru_maxrss/1024)


def resource_delta(before):
    after = resource_snapshot()
    wall = after['at']-before['at']
    return dict(wall_seconds=wall, parent_cpu_seconds=after['parent_cpu']-before['parent_cpu'],
                child_cpu_seconds=after['child_cpu']-before['child_cpu'],
                parent_cpu_percent=100*(after['parent_cpu']-before['parent_cpu'])/wall,
                child_cpu_percent=100*(after['child_cpu']-before['child_cpu'])/wall,
                parent_peak_rss_mib=after['parent_peak_rss_mib'],
                largest_child_peak_rss_mib=after['largest_child_peak_rss_mib'])


def worker_trial(name, settings, *, gap, audible=False, backend_factory=None):
    """Exercise production ownership, replacement, urgency and freshness rules."""
    before = resource_snapshot()
    backend = backend_factory(settings) if backend_factory else ObservedEspeak(settings)
    validity, playback = SpeechValidity(), PlaybackGate(settings.echo_tail)
    worker = SpeechWorker(backend, validity, playback, listening_gap=gap)
    backend.worker = worker
    requests = []
    cancellation_at = None
    trial_error = None

    def submit(text, priority=10, condition=None):
        message = SpeechMessage(text, 0, condition, priority)
        row = dict(text=text, priority=priority, condition=condition, requested_at=time.monotonic())
        row['accepted'] = worker.submit(message)
        row['submit_seconds'] = time.monotonic()-row['requested_at']
        requests.append(row)

    worker.start()
    try:
        if name == 'idle_sequence':
            for text in PHRASES:
                submit(text, 90 if text == 'Search cancelled.' else 10)
                wait_for(lambda: not worker.busy, worker)
        elif name == 'rapid_latest':
            submit(PHRASES[1])
            wait_for(backend.entered.is_set, worker)
            for text in PHRASES[2:5]:
                submit(text)
            wait_for(lambda: not worker.busy, worker)
        elif name in ('urgent_interrupt', 'stale_interrupt'):
            validity.update(0, 'left', time.monotonic()+20)
            submit(PHRASES[5], condition='left')
            wait_for(backend.spawned.is_set, worker)
            if audible:
                time.sleep(.25)  # An explicit audible partial-phrase interruption.
            cancellation_at = time.monotonic()
            validity.update(0, 'right', cancellation_at+20)
            if name == 'urgent_interrupt':
                submit('Search cancelled.', 90)
            wait_for(lambda: not worker.busy, worker)
        elif name == 'expires_in_gap':
            submit(PHRASES[0])
            wait_for(lambda: not worker.busy, worker)
            validity.update(0, 'left', time.monotonic()+.15)
            submit(PHRASES[5], 60, 'left')
            wait_for(lambda: not worker.busy, worker)
        else:
            raise ValueError(f'Unknown trial: {name}')
    except Exception as error:
        trial_error = str(error)
    finally:
        stopped = worker.stop(2)
    calls = backend.calls
    previous_return = None
    for call in calls:
        # Unique text within each scenario; superseded requests never get calls.
        request = next(r for r in reversed(requests) if r['text'] == call['text'] and r['accepted'])
        call['request_to_backend_seconds'] = call['entered_at']-request['requested_at']
        call['gap_after_previous_backend_seconds'] = (call['entered_at']-previous_return
                                                     if previous_return is not None else None)
        previous_return = call['returned_at']
    applicable = bool(cancellation_at is not None and calls and
                      calls[0]['returned_at'] >= cancellation_at and calls[0]['first_stop_observed_at'] is not None)
    return dict(scenario=name, requests=requests, calls=calls, stats=asdict(worker.stats),
        accepted_without_backend_call=[r['text'] for r in requests if r['accepted']
            and not any(c['text'] == r['text'] for c in calls)],
        cleanup_complete=stopped, error=trial_error or (str(worker.error) if worker.error else None),
        cancellation_requested_at=cancellation_at, cancellation_applicable=applicable,
        cancellation_to_backend_return_seconds=calls[0]['returned_at']-cancellation_at if applicable else None,
        acoustic_cancellation_seconds=None, acoustic_start_seconds=None,
        resources=resource_delta(before))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mode', choices=('silent', 'synthesis', 'audible'), default='silent')
    result.add_argument('--allow-playback', action='store_true', help='Explicitly permit speaker playback in audible mode')
    result.add_argument('--output-device', help='ALSA name for audible production path; omit for eSpeak system default')
    result.add_argument('--output', type=Path, help='New JSON metadata report; never overwritten')
    result.add_argument('--voice', default='en-us')
    result.add_argument('--rate', type=int, default=155)
    result.add_argument('--volume', type=float, default=.8)
    result.add_argument('--listening-gap', type=float, default=1., help='Experiment only; production default is 1 second')
    result.add_argument('--synthesis-repetitions', type=int, default=3)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.mode == 'audible' and not args.allow_playback:
        cli.error('Audible mode requires --allow-playback; no playback has started')
    if args.mode != 'audible' and (args.allow_playback or args.output_device):
        cli.error('Playback/device flags are only valid in audible mode')
    if not math.isfinite(args.listening_gap) or not 0 <= args.listening_gap <= 3:
        cli.error('Use a listening gap between 0 and 3 seconds')
    if not 1 <= args.synthesis_repetitions <= 10:
        cli.error('Use 1 to 10 synthesis repetitions')
    settings = AudioSettings(voice=args.voice, speaking_rate=args.rate, volume=args.volume,
        output_device=args.output_device if args.mode == 'audible' else 'null' if args.mode == 'silent' else None)
    # Reserve a new report before starting any subprocess or audible output.
    report_file = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report_file = args.output.open('x', encoding='utf-8')
    before = resource_snapshot()
    report = dict(schema=1, mode=args.mode, environment=dict(python=platform.python_version(),
        platform=platform.platform(), cpu_count=os.cpu_count()), settings=asdict(settings),
        listening_gap=args.listening_gap, microphone_opened=False, camera_opened=False,
        yolo_or_asr_loaded=False, audio_saved=False, synthesis=[], worker_trials=[], error=None,
        timing_scope='Software boundaries only. Speaker start/underruns/conversions may be unobservable.',
        playback_sink='ALSA null, unpaced' if args.mode == 'silent' else
                      'none' if args.mode == 'synthesis' else args.output_device or 'eSpeak system default')
    try:
        backend = EspeakSpeech(settings)
        report['synthesizer'] = backend.binary
        for repetition in range(args.synthesis_repetitions):
            for text in PHRASES:
                result = synthesize(backend.binary, settings, text)
                report['synthesis'].append(dict(repetition=repetition+1, **result))
        if args.mode != 'synthesis':
            for name in ('idle_sequence', 'rapid_latest', 'urgent_interrupt', 'stale_interrupt', 'expires_in_gap'):
                if args.mode == 'audible':
                    print(f'AUDIBLE trial: {name}. Interruption trials deliberately cut speech short.',
                          file=sys.stderr, flush=True)
                result = worker_trial(name, settings, gap=args.listening_gap, audible=args.mode == 'audible')
                report['worker_trials'].append(result)
                if result['error'] or not result['cleanup_complete']:
                    raise RuntimeError(result['error'] or 'Speech worker remains live')
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['resources'] = resource_delta(before)
        if report_file:
            json.dump(report, report_file, indent=2, default=str)
            report_file.write('\n')
            report_file.close()
        print(json.dumps(report, indent=2, default=str))
    return 1 if report['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
