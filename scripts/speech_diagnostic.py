"""Consent-gated recording and identical-audio offline speech comparisons."""

import argparse
import hashlib
import json
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

from homesense.audio.comparison import (Clip, CONTROLS, MAX_SECONDS, PROMPTS, SAMPLE_RATE,
    VoskReplay, WhisperCppReplay, load_manifest, pcm_stats, score, wake_replay, wav_bytes)
from homesense.audio.input import SoundDeviceInput
from homesense.config import AudioSettings

BACKENDS = ("vosk-current", "vosk-native", "vosk-grammar", "whisper")


def device(value):
    try:
        return int(value)
    except ValueError:
        return value


def inspect_device(index):
    """Query only; this command never opens a recording stream."""
    import sounddevice as sd
    info = dict(sd.query_devices(index, "input"))
    sd.check_input_settings(device=index, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    return {"device": info, "portaudio": sd.get_portaudio_version(),
            "requested": {"sample_rate": SAMPLE_RATE, "channels": 1, "dtype": "int16", "chunk_ms": 100},
            "format_supported": True, "note": "No stream opened; device default rate is not actual stream or ADC rate"}


def private_write(path, data):
    """Never overwrite a previous trial; recordings/reports are private by default."""
    with path.open("xb") as file:
        os.chmod(path, 0o600)
        file.write(data)


def capture(source, seconds):
    target = int(seconds * SAMPLE_RATE) * 2
    pcm = bytearray()
    deadline = time.monotonic() + seconds + 2
    while len(pcm) < target:
        if time.monotonic() >= deadline:
            raise TimeoutError("Audio input did not deliver the requested recording duration")
        chunk = source.read(0.1)
        if chunk:
            pcm.extend(chunk)
    return bytes(pcm[:target])


def record(args):
    if not args.consent_recording:
        raise ValueError("Recording requires --consent-recording and per-utterance Enter confirmation")
    if not 0 < args.seconds <= MAX_SECONDS or not math.isfinite(args.seconds) or args.repetitions < 1:
        raise ValueError("Use 0 < seconds <= 30 and positive repetitions")
    # Obtain metadata first; do not leave a misleading empty trial on device failure.
    metadata = inspect_device(args.input_device)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    manifest = {"schema": 1, "purpose": "Explicit-consent local ASR comparison", "capture": metadata,
                "seconds_per_clip": args.seconds, "clips": []}
    print("Local WAV files will be retained in", args.output_dir)
    print("No upload. Stop with Ctrl+C; completed clips remain for local review/deletion.")
    prompts = PROMPTS + (CONTROLS if args.include_controls else ())
    for repetition in range(args.repetitions):
        for index, prompt in enumerate(prompts):
            input(f"Take {repetition + 1}, prompt {index + 1}: {prompt or '[remain silent]'}\nPress Enter to record {args.seconds:g} seconds: ")
            source = SoundDeviceInput(AudioSettings(input_device=args.input_device))
            try:
                source.open()
                # Stream properties audit the delivery contract, not native ADC hardware.
                stream = source._stream
                actual = {"sample_rate": stream.samplerate, "channels": stream.channels,
                          "dtype": stream.dtype, "blocksize": stream.blocksize, "latency": stream.latency}
                if (stream.samplerate, stream.channels, stream.dtype) != (SAMPLE_RATE, 1, "int16"):
                    raise RuntimeError(f"Unexpected microphone stream format: {actual}")
                print("RECORDING — speak now", flush=True)
                pcm = capture(source, args.seconds)
            finally:
                source.close()
            name = f"take-{repetition + 1:02d}-prompt-{index + 1:02d}.wav"
            data = wav_bytes(pcm)
            private_write(args.output_dir / name, data)
            manifest["clips"].append({"file": name, "reference": prompt,
                "sha256": hashlib.sha256(data).hexdigest(), "actual_stream": actual,
                "audio": pcm_stats(pcm)})
            # Atomic checkpoint preserves completed clips even if the next capture fails.
            temporary = args.output_dir / ".manifest.tmp"
            private_write(temporary, json.dumps(manifest, indent=2).encode())
            temporary.replace(args.output_dir / "manifest.json")
            print("Saved", name, "— check playback locally before scoring.")
    print("Manifest:", args.output_dir / "manifest.json")


def worker(args):
    clip = Clip.load(args.wav, "", args.sha256)
    settings = AudioSettings(chunk_ms=args.chunk_ms, silence_seconds=args.silence_seconds, silence_rms=args.silence_rms)
    backend = (WhisperCppReplay(args.whisper_cli, args.whisper_model, threads=args.threads, timeout=args.timeout)
               if args.backend == "whisper" else VoskReplay(args.vosk_model, args.backend, settings))
    result = backend.transcribe(clip.pcm)
    result["audio_sha256"] = clip.sha256
    result["wake_gate_commands"] = wake_replay(result["segments"])
    print(json.dumps(result))


def run_isolated(command, timeout):
    """A separate process per trial makes cold latency and Linux RSS comparable.

    Kill the process group on timeout/interrupt, including the native CLI child.
    """
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    elapsed = time.monotonic() - started
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace")[-3000:])
    result = json.loads(stdout)
    for line in stderr.decode(errors="replace").splitlines():
        if line.startswith("HOMESENSE_RESOURCES "):
            _, rss, user, system = line.split()
            result.update(peak_process_rss_kib=int(rss), cpu_seconds=float(user) + float(system))
    result["cold_wall_seconds"] = elapsed
    return result


def summary(rows):
    result = {}
    for name in dict.fromkeys(r["backend"] for r in rows):
        group = [r for r in rows if r["backend"] == name]
        valid = [r for r in group if "error" not in r]
        speech = [r for r in valid if r["reference_words"]]
        words = sum(r["reference_words"] for r in speech)
        result[name] = {"attempted": len(group), "successful_runs": len(valid), "failures": len(group) - len(valid),
                       "speech_samples_scored": len(speech),
                       "exact_match_rate": sum(r["exact_match"] for r in speech) / len(speech) if speech else None,
                       "corpus_wer": sum(r["word_errors"] for r in speech) / words if words else None,
                       "median_cold_seconds": statistics.median(r["cold_wall_seconds"] for r in valid) if valid else None,
                       "max_peak_process_rss_kib": max((r.get("peak_process_rss_kib", 0) for r in valid), default=None),
                       "silence_samples_scored": sum(not r["reference_words"] for r in valid),
                       "non_speech_insertions": (sum(bool(r["transcript"].strip()) for r in valid if not r["reference_words"])
                                                 if any(not r["reference_words"] for r in valid) else None)}
    return result


def compare(args):
    manifest, clips = load_manifest(args.manifest)
    if args.runs < 1 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Positive runs, threads and finite timeout required")
    selected = args.backends or list(BACKENDS)
    if "whisper" in selected:
        WhisperCppReplay(args.whisper_cli, args.whisper_model)  # Fail before an expensive incomplete run.
    if not Path("/usr/bin/time").is_file():
        raise RuntimeError("Install GNU time to measure CPU and peak process RSS")
    report = {"schema": 1, "platform": platform.platform(), "python": sys.version,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "capture": manifest.get("capture"), "configuration": {
                  "backends": selected, "threads": args.threads, "chunk_ms": args.chunk_ms,
                  "silence_seconds": args.silence_seconds, "silence_rms": args.silence_rms,
                  "vosk_model": str(args.vosk_model), "whisper_model": str(args.whisper_model),
                  "whisper_cli": str(args.whisper_cli)},
              "timing_scope": "Cold process per recording; model load included. No capture/real-time pacing or concurrent YOLO. RSS is peak process, not combined simultaneous memory.",
              "rows": []}
    if args.output.exists():
        raise FileExistsError("Choose a new report path; previous reports are never overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["package_versions"] = {}
    for name in ("vosk", "sounddevice"):
        try:
            report["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][name] = None
    report["artifacts"] = {}
    artifacts = []
    if any(backend.startswith("vosk-") for backend in selected):
        artifacts.append(("vosk_acoustic", args.vosk_model / "am/final.mdl"))
    if "whisper" in selected:
        artifacts.extend([("whisper_model", args.whisper_model), ("whisper_cli", args.whisper_cli)])
    for name, path in artifacts:
        with path.open("rb") as artifact:
            report["artifacts"][name] = {"sha256": hashlib.file_digest(artifact, "sha256").hexdigest(), "bytes": path.stat().st_size}
    # Rotate order to reduce a fixed order's cache/thermal bias; still not a randomized study.
    for run in range(args.runs):
        for index, clip in enumerate(clips):
            order = selected[(run + index) % len(selected):] + selected[:(run + index) % len(selected)]
            for backend in order:
                row = {"backend": backend, "run": run + 1, "file": clip.path.name, "reference": clip.reference,
                       "audio_sha256": clip.sha256, "audio": pcm_stats(clip.pcm)}
                command = ["/usr/bin/time", "-f", "HOMESENSE_RESOURCES %M %U %S", sys.executable,
                           str(Path(__file__).resolve()), "worker", "--backend", backend, "--wav", str(clip.path),
                           "--sha256", clip.sha256, "--vosk-model", str(args.vosk_model),
                           "--whisper-cli", str(args.whisper_cli), "--whisper-model", str(args.whisper_model),
                           "--threads", str(args.threads), "--timeout", str(args.timeout),
                           "--chunk-ms", str(args.chunk_ms), "--silence-seconds", str(args.silence_seconds),
                           "--silence-rms", str(args.silence_rms)]
                try:
                    row.update(run_isolated(command, args.timeout + 5))
                    if row["audio_sha256"] != clip.sha256:
                        raise ValueError("Replay audio hash mismatch")
                    row.update(score(clip.reference, row["transcript"]))
                    row["cold_real_time_factor"] = row["cold_wall_seconds"] / row["audio"]["duration_seconds"]
                except Exception as error:
                    row["error"] = str(error)
                report["rows"].append(row)
                print(json.dumps({k: row.get(k) for k in ("backend", "file", "transcript", "wer", "cold_wall_seconds", "error")}), flush=True)
    report["summary"] = summary(report["rows"])
    private_write(args.output, json.dumps(report, indent=2).encode())
    print(json.dumps(report["summary"], indent=2))
    print("Report:", args.output)
    return int(any("error" in row for row in report["rows"]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    inspect = sub.add_parser("inspect", help="query device metadata without recording")
    inspect.add_argument("--input-device", type=device, default=12)
    recorder = sub.add_parser("record", help="explicit-consent, prompted local recordings")
    recorder.add_argument("--consent-recording", action="store_true")
    recorder.add_argument("--input-device", type=device, default=12)
    recorder.add_argument("--output-dir", type=Path, required=True)
    recorder.add_argument("--repetitions", type=int, default=3)
    recorder.add_argument("--seconds", type=float, default=5)
    recorder.add_argument("--include-controls", action="store_true")
    comparison = sub.add_parser("compare", help="replay identical WAV files to selected local backends")
    comparison.add_argument("manifest", type=Path)
    comparison.add_argument("--backends", nargs="+", choices=BACKENDS)
    comparison.add_argument("--runs", type=int, default=1)
    comparison.add_argument("--output", type=Path, default=Path("outputs/speech-comparison.json"))
    internal = sub.add_parser("worker", help="internal isolated replay worker")
    internal.add_argument("--backend", choices=BACKENDS, required=True)
    internal.add_argument("--wav", type=Path, required=True)
    internal.add_argument("--sha256", required=True)
    for p in (comparison, internal):
        p.add_argument("--vosk-model", type=Path, default=AudioSettings().model_path)
        p.add_argument("--whisper-cli", type=Path, default=Path("tools/whisper.cpp/build/bin/whisper-cli"))
        p.add_argument("--whisper-model", type=Path, default=Path("models/speech/ggml-small.en-q5_1.bin"))
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--timeout", type=float, default=120)
        p.add_argument("--chunk-ms", type=int, default=100)
        p.add_argument("--silence-seconds", type=float, default=0.6)
        p.add_argument("--silence-rms", type=float, default=0.015)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.mode == "inspect":
            print(json.dumps(inspect_device(args.input_device), indent=2))
        elif args.mode == "record":
            record(args)
        elif args.mode == "worker":
            worker(args)
        else:
            return compare(args)
    except KeyboardInterrupt:
        print("Stopped; no audio uploaded.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Speech diagnostic failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
