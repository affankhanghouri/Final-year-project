"""Listen to resident Piper or eSpeak, without application, camera or microphone."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from homesense.config import AudioSettings
from scripts.voice_output_benchmark import (
    PHRASES, ObservedEspeak, resource_snapshot, resource_delta, synthesize, worker_trial,
)
from .prototype import PiperSpeech


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'models/speech/piper/en_US-ljspeech-medium/en_US-ljspeech-medium.onnx'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('piper', 'espeak', 'compare'), default='piper')
    parser.add_argument('--scenario', choices=('normal', 'rapid', 'interrupt', 'all'), default='normal')
    consent = parser.add_mutually_exclusive_group(required=True)
    consent.add_argument('--allow-playback', action='store_true', help='Explicitly play speech on your selected output')
    consent.add_argument('--silent', action='store_true', help='Use ALSA null; never opens a physical audio device')
    parser.add_argument('--output-device', default='default', help='ALSA output name, not microphone index')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--no-cache', action='store_true', help='Measure uncached resident synthesis')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4:
        parser.error('Use 1 to 4 CPU threads')
    if args.silent and args.output_device != 'default':
        parser.error('--silent always uses null; omit --output-device')
    output = args.output or ROOT/'outputs/piper-trial'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites before loading any model or opening output.
    with output.open('x', encoding='utf-8') as file:
        settings = AudioSettings(output_device='null' if args.silent else args.output_device)
        result = dict(schema=1, audible=args.allow_playback, camera_opened=False, microphone_opened=False,
            audio_saved=False, acoustic_latency_seconds=None, output_device=settings.output_device,
            backends=[], error=None, mode='output-only; no listening gap because no microphone is running')
        try:
            for name in ('espeak', 'piper') if args.backend == 'compare' else (args.backend,):
                before = resource_snapshot()
                print(f'Preparing {name}; no sound yet.', flush=True)
                if name == 'piper':
                    started = time.monotonic()
                    backend = PiperSpeech(args.model, settings, threads=args.threads)
                    load = time.monotonic()-started
                    if not args.no_cache:
                        backend.prepare(PHRASES)
                    model_info = json.loads(args.model.with_name('provenance.json').read_text())
                    entry = dict(backend=name, load_seconds=load, session_load_seconds=backend.load_seconds,
                        piper_version=importlib.metadata.version('piper-tts'),
                        onnxruntime_version=importlib.metadata.version('onnxruntime'),
                        model=model_info, threads=args.threads, preparation=backend.preparation,
                        cache_bytes=sum(map(len, backend.cache.values())), sample_rate=backend.sample_rate,
                        trials=[])
                else:
                    backend = ObservedEspeak(settings)
                    entry = dict(backend=name, preparation=[synthesize(backend.binary, settings, t) for t in PHRASES],
                                 trials=[])
                result['backends'].append(entry)
                scenarios = ('normal', 'rapid', 'interrupt') if args.scenario == 'all' else (args.scenario,)
                for scenario in scenarios:
                    if args.allow_playback:
                        input(f'[{name.upper()} / {scenario}] Press Enter to listen. '
                              'Interruption trials deliberately cut speech short: ')
                    names = {'normal': ('idle_sequence',), 'rapid': ('rapid_latest',),
                             'interrupt': ('urgent_interrupt', 'stale_interrupt')}[scenario]
                    for trial in names:
                        print(f'{name.upper()}: {trial}', flush=True)
                        # Models/cache stay resident; only finite trace metadata resets.
                        if name == 'piper':
                            backend.reset_observations()
                        else:
                            backend.calls = []
                            backend.entered.clear()
                            backend.spawned.clear()
                        measured = worker_trial(trial, settings, gap=0, audible=args.allow_playback,
                                                backend_factory=lambda _: backend)
                        entry['trials'].append(measured)
                        if measured['error'] or not measured['cleanup_complete']:
                            raise RuntimeError(measured['error'] or 'Output worker did not stop')
                entry['resources_including_preparation'] = resource_delta(before)
        except (Exception, KeyboardInterrupt) as error:
            result['error'] = f'{type(error).__name__}: {error}'
            print(result['error'], file=sys.stderr)
        finally:
            json.dump(result, file, indent=2, default=str)
            file.write('\n')
            print(f'Measurements: {output}')
    return 1 if result['error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
