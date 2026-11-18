"""Local HomeSense voice application and explicit hardware diagnostic modes."""

import argparse
import logging
import math
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

from homesense.audio.events import VoiceEvent, VoiceEventKind
from homesense.audio.input import SoundDeviceInput
from homesense.config import AudioSettings, FeedbackSettings, RuntimeSettings
from homesense.runtime import compose
from homesense.diagnostics import RuntimeMetrics
from homesense.diagnostic_logging import DiagnosticRun, event


def device(value):
    try:
        return int(value)
    except ValueError:
        return value


def diagnostic(args, audio):
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.list_voices:
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if binary is None:
            raise RuntimeError("Install espeak-ng; see docs/audio-setup.md")
        subprocess.run([binary, "--voices"], check=True)
        return
    if args.tts_test:
        runtime = compose(RuntimeSettings(audio=audio), text_mode=True, console_output=args.console_feedback)
        try:
            runtime.start()  # READY wording is owned by FeedbackManager.
            deadline = time.monotonic() + args.seconds
            while runtime.speech.busy and time.monotonic() < deadline:
                if runtime.speech.error:
                    raise runtime.speech.error
                time.sleep(0.02)
        finally:
            runtime.close()
        if runtime.failed:
            raise RuntimeError("Speech test failed")
        return
    recognizer = None
    if args.stt_test:
        from homesense.audio.speech_recognition import VoskCommandRecognizer
        recognizer = VoskCommandRecognizer(audio.model_path, audio.sample_rate, max_seconds=audio.command_timeout)
    source = SoundDeviceInput(audio)
    chunks, peak = 0, 0
    try:
        source.open()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            chunk = source.read(0.1)
            if chunk is None:
                continue
            chunks += 1
            if recognizer is not None:
                result = recognizer.feed(chunk)
                if result.final:
                    print(result.text or "[unintelligible/empty]", flush=True)
            else:
                from array import array
                samples = array("h", chunk)
                peak = max(peak, max((abs(v) for v in samples), default=0))
    finally:
        source.close()
    print(f"Read {chunks} audio chunks; peak PCM amplitude: {peak if recognizer is None else 'not measured'}. No recording saved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--speech-model", type=Path, default=AudioSettings().model_path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--input-device", type=device)
    parser.add_argument("--output-device", help="ALSA output name for aplay; not a PortAudio device index")
    parser.add_argument("--voice", default="en-us")
    parser.add_argument("--rate", type=int, default=155)
    parser.add_argument("--volume", type=float, default=0.8)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--command-timeout", type=float, default=10.0)
    parser.add_argument("--silence-seconds", type=float, default=0.6, help="legacy diagnostic setting; production uses native endpoints")
    parser.add_argument("--silence-rms", type=float, default=0.015, help="legacy diagnostic setting; production uses no RMS cutoff")
    parser.add_argument("--search-timeout", type=float, default=60.0)
    parser.add_argument("--shutdown-timeout", type=float, default=2.0)
    parser.add_argument("--no-detection-delay", type=float, default=5.0)
    parser.add_argument("--feedback-cooldown", type=float, default=3.0)
    parser.add_argument("--text", action="store_true", help="typed commands; no microphone/ASR")
    parser.add_argument("--console-feedback", action="store_true", help="print feedback instead of audio (explicit debug mode)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="save bounded metadata-only JSONL logs under outputs/logs")
    parser.add_argument("--preview", action="store_true", help="optional live/annotated laptop window; q exits")
    parser.add_argument("--guided-scan", action="store_true", help="opt-in bounded motion-aware left/right object search")
    parser.add_argument("--metrics-interval", type=float, default=5.0, help="seconds between process/worker timing reports")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-audio-devices", action="store_true")
    modes.add_argument("--list-voices", action="store_true")
    modes.add_argument("--mic-test", action="store_true")
    modes.add_argument("--stt-test", action="store_true")
    modes.add_argument("--tts-test", action="store_true")
    parser.add_argument("--seconds", type=float, default=15.0, help="hardware diagnostic duration")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    runtime = None
    preview = None
    exit_code = 0
    run_log = None
    try:
        if args.diagnostics:
            run_log = DiagnosticRun().start(preview=args.preview, guided_scan=args.guided_scan,
                camera=args.camera, input_device=args.input_device, output_device=args.output_device,
                text_mode=args.text, console_feedback=args.console_feedback,
                checkpoint=str(args.checkpoint), speech_model=str(args.speech_model),
                search_timeout=args.search_timeout, metrics_interval=args.metrics_interval)
            print(f"Diagnostic log: {run_log.path}\nSession summaries: {run_log.summary_path}", flush=True)
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            raise ValueError("seconds must be positive finite")
        if not math.isfinite(args.metrics_interval) or args.metrics_interval <= 0:
            raise ValueError("metrics-interval must be positive finite")
        audio = AudioSettings(model_path=args.speech_model, input_device=args.input_device,
                              output_device=args.output_device, voice=args.voice, speaking_rate=args.rate,
                              volume=args.volume, sample_rate=args.sample_rate, command_timeout=args.command_timeout,
                              silence_seconds=args.silence_seconds, silence_rms=args.silence_rms)
        if any((args.list_audio_devices, args.list_voices, args.mic_test, args.stt_test, args.tts_test)):
            diagnostic(args, audio)
            return 0
        settings = RuntimeSettings(checkpoint=args.checkpoint, camera_id=args.camera, audio=audio,
                                   guided_scan=args.guided_scan,
                                   search_timeout=args.search_timeout, shutdown_timeout=args.shutdown_timeout,
                                   feedback=FeedbackSettings(no_detection_delay=args.no_detection_delay,
                                                             direction_cooldown=args.feedback_cooldown))
        runtime = compose(settings, text_mode=args.text, console_output=args.console_feedback)
        event('application', 'components_initialized', preview=args.preview, guided_scan=args.guided_scan)
        if args.preview:
            from homesense.preview import LaptopPreview
            preview = LaptopPreview()
        metrics = RuntimeMetrics(args.metrics_interval)
        runtime.start()
        if args.text:
            print("Type commands without a wake phrase. Type quit or exit to close.", flush=True)
        while not runtime.failed:
            runtime.step()
            if preview is not None and not preview.update(runtime.interaction.snapshot, runtime.interaction.coordinator):
                break
            metrics.sample(runtime, preview)
            if args.text and select.select([sys.stdin], [], [], settings.poll_interval)[0]:
                line = sys.stdin.readline()
                if not line or line.strip().lower() in {"quit", "exit"}:
                    break
                runtime.events.put(VoiceEvent(VoiceEventKind.TRANSCRIPT, line.strip(), time.monotonic()))
            elif not args.text:
                time.sleep(settings.poll_interval)
        exit_code = int(runtime.failed)
    except KeyboardInterrupt:
        event('application', 'shutdown_requested', reason='keyboard_interrupt')
    except Exception as error:
        logging.error("HomeSense could not continue: %s. See docs/audio-setup.md for setup and device checks.", error)
        if args.debug:
            logging.exception("Diagnostic details")
        exit_code = 1
    finally:
        event('application', 'shutdown_started', exit_code=exit_code)
        if runtime is not None:
            try:
                runtime.close()
                exit_code = max(exit_code, int(runtime.failed))
            except Exception:
                logging.exception("Runtime cleanup failed")
                exit_code = 1
        if preview is not None:
            try:
                preview.close()
            except Exception:
                logging.exception("Preview cleanup failed")
                exit_code = 1
        if run_log is not None:
            if not run_log.close(exit_code=exit_code):
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
