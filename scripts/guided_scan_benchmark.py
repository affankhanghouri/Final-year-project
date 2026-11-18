"""Finite hardware-free load probe: synthetic camera, local models, no playback.

Compare the same resident YOLO with/without guided scan. Optional existing PCM
replay and eSpeak stdout synthesis run identically in both modes. No camera,
microphone, GUI, speaker or network is opened. Not a live acceptance benchmark.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import threading
import time

import cv2
import numpy as np

from homesense.application import ApplicationCoordinator
from homesense.detector import InferenceSettings, YoloDetector
from homesense.motion import MotionSettings
from homesense.search_strategy import ScanSettings


class SyntheticCamera:
    def __init__(self):
        gray = np.random.default_rng(172).integers(0, 256, (480, 640), dtype=np.uint8)
        self.image = cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        self.started = time.monotonic()
        self.released = False
        self._next = self.started

    def isOpened(self):
        return True

    def read(self):
        self._next += 1/30
        time.sleep(max(0, self._next-time.monotonic()))
        dx = 30*np.sin((time.monotonic()-self.started)*.8)
        image = cv2.warpAffine(self.image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480), borderMode=cv2.BORDER_REFLECT)
        return True, image

    def release(self):
        self.released = True


def trial(detector, enabled, seconds, recognizer=None, pcm=None, synthesize=False):
    sources = []
    def source(camera_id):
        camera = SyntheticCamera()
        sources.append(camera)
        return camera
    app = ApplicationCoordinator(detector, source_factory=source,
        motion=MotionSettings() if enabled else None,
        scan=ScanSettings(initial_observe_seconds=seconds+5))
    stop = threading.Event()
    started, cpu = time.monotonic(), time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    motion_times, inference_times, states = deque(maxlen=512), deque(maxlen=512), []
    last_motion = last_processed = 0
    def speech():
        if recognizer is None:
            return {'calls': 0, 'max_feed_seconds': 0}
        recognizer.reset()
        offset, calls, maximum, next_at = 0, 0, 0.0, time.monotonic()
        while not stop.is_set():
            if offset >= len(pcm):
                offset = 0
                recognizer.reset()
            before = time.monotonic()
            recognizer.feed(pcm[offset:offset+3200])
            maximum = max(maximum, time.monotonic()-before)
            offset += 3200
            calls += 1
            next_at += .1
            stop.wait(max(0, next_at-time.monotonic()))
        return {'calls': calls, 'max_feed_seconds': maximum}
    def synthesis():
        count = 0
        binary = shutil.which('espeak-ng') or shutil.which('espeak')
        if synthesize and binary is None:
            raise RuntimeError('eSpeak is required only for --synthesize-speech')
        while synthesize and not stop.is_set():
            subprocess.run([binary, '-v', 'en-us', '-s', '155', '--stdout', 'Phone on your left.'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, start_new_session=True, timeout=5)
            count += 1
            stop.wait(1.5)
        return count
    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(speech), pool.submit(synthesis)]
        try:
            app.start_search('phone', seconds+10)
            while time.monotonic()-started < seconds:
                snapshot = app.poll()
                if snapshot.errors:
                    raise RuntimeError(str(snapshot.errors))
                if snapshot.motion is not None and snapshot.motion.frame_id != last_motion:
                    last_motion = snapshot.motion.frame_id
                    motion_times.append(snapshot.motion.processing_seconds)
                if snapshot.stats.processed != last_processed and snapshot.stats.last_inference_seconds > 0:
                    last_processed = snapshot.stats.processed
                    inference_times.append(snapshot.stats.last_inference_seconds)
                if snapshot.scan is not None:
                    state = snapshot.scan.phase.value
                    if not states or state != states[-1]:
                        states.append(state)
                stop.wait(.02)
            snapshot = app.poll()
            frame = app.latest_frame()
            elapsed = time.monotonic()-started
            process_cpu = time.process_time()-cpu
            frame_count = frame.frame_id if frame else 0
        finally:
            stop.set()
            cleanup_start = time.monotonic()
            final = app.close()
            cleanup_seconds = time.monotonic()-cleanup_start
        audio_result, synth_count = [worker.result() for worker in workers]
    if final.errors or not final.cleanup_complete or not all(s.released for s in sources):
        raise RuntimeError(f'Probe cleanup failed: {final.errors}')
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(guided_scan=enabled, elapsed_seconds=elapsed, process_cpu_percent=100*process_cpu/elapsed,
        synthetic_camera_fps=frame_count/elapsed, inference_fps=snapshot.stats.processed/elapsed,
        published_result_fps=snapshot.stats.published/elapsed,
        inference_median_ms=1000*statistics.median(inference_times) if inference_times else 0,
        stats=asdict(snapshot.stats), motion_samples=len(motion_times),
        motion_median_ms=1000*statistics.median(motion_times) if motion_times else 0,
        motion_max_ms=1000*max(motion_times, default=0), cleanup_seconds=cleanup_seconds,
        camera_instances=len(sources), cleanup_complete=final.cleanup_complete,
        tts_synthesis_calls=synth_count,
        tts_child_cpu_seconds=children.ru_utime+children.ru_stime-children_before.ru_utime-children_before.ru_stime,
        asr=audio_result, states=states)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/best.pt'))
    parser.add_argument('--seconds', type=float, default=6)
    parser.add_argument('--speech-manifest', type=Path, help='optional existing consented WAV manifest; never captures audio')
    parser.add_argument('--synthesize-speech', action='store_true', help='eSpeak stdout is discarded; no speaker playback')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30 or args.output.exists():
        parser.error('seconds must be 1..30; choose a new output path')
    detector = YoloDetector(args.checkpoint, settings=InferenceSettings(device='cpu'))
    detector.predict(SyntheticCamera().image)  # Resident warm-up, no device.
    recognizer = pcm = None
    if args.speech_manifest:
        import vosk
        from homesense.audio.comparison import load_manifest
        from homesense.audio.speech_recognition import VoskCommandRecognizer
        vosk.SetLogLevel(-1)
        _, clips = load_manifest(args.speech_manifest)
        pcm = clips[0].pcm
        recognizer = VoskCommandRecognizer(Path('models/speech/vosk-model-small-en-us-0.15'))
        # Warm both resident decoder paths before comparing modes. Otherwise the
        # first baseline alone includes cold native decoder initialization.
        for offset in range(0, len(pcm), 3200):
            recognizer.feed(pcm[offset:offset+3200])
        recognizer.reset()
    # Alternating order reduces first/second-mode bias; still a short pilot.
    rows = []
    for enabled in (False, True, True, False):
        row = trial(detector, enabled, args.seconds, recognizer, pcm, args.synthesize_speech)
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(scope='Synthetic capture and optional PCM/synthesis load; no physical hardware or GUI; no TTS/mic suppression simulation',
                  python=platform.python_version(), machine=platform.machine(), rows=rows,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
