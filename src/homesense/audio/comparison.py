"""Offline, opt-in ASR experiments. Never imported by the application runtime."""

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Protocol
import wave

from homesense.audio.speech_recognition import VoskRecognizer
from homesense.audio.wake_word import WakePhraseGate
from homesense.audio.events import VoiceEventKind
from homesense.commands.models import Intent
from homesense.commands.parser import ALIASES, parse_command
from homesense.config import AudioSettings

PROMPTS = (
    "Hey Agent, find my phone.", "Find my keys.", "Locate my wallet.",
    "Cancel search.", "Repeat the direction.", "I found it.", "Hey Agent, help me.",
)
CONTROLS = ("Please close the window.", "")
SAMPLE_RATE = 16000
MAX_SECONDS = 30


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def word_errors(reference: str, actual: str) -> int:
    """Levenshtein distance on normalized words; insertions may make WER > 1."""
    reference, actual = normalize(reference).split(), normalize(actual).split()
    previous = list(range(len(actual) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def wav_bytes(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)
    return output.getvalue()


def pcm_stats(pcm: bytes, *, chunk_ms: int = 100, silence_rms: float = 0.015) -> dict:
    if not pcm or len(pcm) % 2:
        raise ValueError("Expected nonempty, complete signed-16-bit PCM samples")
    samples = array("h", pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    def rms(values):
        return math.sqrt(sum(v * v for v in values) / len(values)) / 32768
    block = SAMPLE_RATE * chunk_ms // 1000
    levels = [rms(samples[i:i + block]) for i in range(0, len(samples), block)]
    return {"duration_seconds": len(samples) / SAMPLE_RATE,
            "peak_pcm": max(abs(v) for v in samples), "rms_normalized": rms(samples),
            "dc_offset_normalized": sum(samples) / len(samples) / 32768,
            "clipped_fraction": sum(v in {-32768, 32767} for v in samples) / len(samples),
            "blocks_below_runtime_silence_threshold": sum(v < silence_rms for v in levels),
            "block_rms": levels, "chunk_ms": chunk_ms, "silence_rms": silence_rms}


@dataclass(frozen=True)
class Clip:
    path: Path
    reference: str
    sha256: str
    pcm: bytes

    @classmethod
    def load(cls, path: Path, reference: str, expected_hash: str | None = None):
        # Bound memory before reading untrusted/imported recordings.
        if path.stat().st_size > SAMPLE_RATE * 2 * MAX_SECONDS + 65536:
            raise ValueError("Recording exceeds the 30-second diagnostic limit")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"Recording changed: {path.name}")
        with wave.open(io.BytesIO(data), "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (SAMPLE_RATE, 1, 2, "NONE"):
                raise ValueError("Comparison requires 16000 Hz, mono, uncompressed PCM16 WAV; no implicit resampling")
            count = stream.getnframes()
            if not 0 < count <= SAMPLE_RATE * MAX_SECONDS:
                raise ValueError("Recording must contain between 0 and 30 seconds of audio")
            pcm = stream.readframes(count)
            if len(pcm) != count * 2:
                raise ValueError("Truncated WAV data")
        return cls(path.resolve(), reference, digest, pcm)


def load_manifest(path: Path) -> tuple[dict, list[Clip]]:
    manifest = json.loads(path.read_text())
    clips = []
    root = path.parent.resolve()
    for item in manifest["clips"]:
        audio = (root / item["file"]).resolve()
        if not audio.is_relative_to(root):
            raise ValueError("Manifest audio paths must remain inside the trial directory")
        clips.append(Clip.load(audio, item["reference"], item.get("sha256")))
    if not clips:
        raise ValueError("Manifest has no recordings")
    return manifest, clips


def grammar_phrases() -> tuple[str, ...]:
    """Fixed task vocabulary, never learned from the held-out reference transcript.

    This diagnostic can force false positives. Include out-of-vocabulary controls.
    """
    commands = {"cancel", "cancel search", "stop searching", "repeat", "repeat direction",
                "repeat the direction", "say that again", "i found it", "i have found it",
                "help", "help me", "what can you do", "what objects can you find"}
    commands.update(f"{verb} {alias}" for verb in ("find my", "locate my", "where is my", "look for my") for alias in ALIASES)
    return tuple(sorted(commands | {"hey agent"} | {f"hey agent {c}" for c in commands})) + ("[unk]",)


class ReplayBackend(Protocol):
    """One finite utterance; deliberately distinct from continuous wake detection."""
    def transcribe(self, pcm: bytes) -> dict: ...


class VoskReplay:
    def __init__(self, model: Path, mode: str, settings: AudioSettings):
        if mode not in {"vosk-current", "vosk-native", "vosk-grammar"}:
            raise ValueError("Unknown Vosk comparison mode")
        self.settings = settings
        started = time.monotonic()
        self.recognizer = VoskRecognizer(model, SAMPLE_RATE, settings.silence_seconds, settings.silence_rms,
                                         software_endpoint=mode == "vosk-current",
                                         grammar=grammar_phrases() if mode == "vosk-grammar" else None)
        self.load_seconds = time.monotonic() - started

    def transcribe(self, pcm: bytes) -> dict:
        size = SAMPLE_RATE * self.settings.chunk_ms // 1000 * 2
        started = time.monotonic()
        segments = []
        empty_finals = 0
        # Capture all final segments plus the EOF tail, rather than silently
        # scoring only the last fragment of an utterance split by endpointing.
        for offset in range(0, len(pcm), size):
            result = self.recognizer.feed(pcm[offset:offset + size])
            if result.final:
                if result.text:
                    segments.append({"text": result.text, "audio_offset_seconds": min(offset + size, len(pcm)) / (SAMPLE_RATE * 2)})
                else:
                    empty_finals += 1
        tail = self.recognizer.finish()
        if tail.text:
            segments.append({"text": tail.text, "audio_offset_seconds": len(pcm) / (SAMPLE_RATE * 2), "eof_flush": True})
        return {"transcript": " ".join(s["text"] for s in segments), "segments": segments,
                "empty_finals": empty_finals, "model_load_seconds": self.load_seconds,
                "decode_seconds": time.monotonic() - started}


class WhisperCppReplay:
    """Finite-utterance CPU CLI adapter. WAV is piped in memory, never written."""
    def __init__(self, binary: Path, model: Path, *, threads: int = 2, timeout: float = 120):
        if not binary.is_file():
            raise FileNotFoundError(f"whisper-cli missing: {binary}")
        if not model.is_file():
            raise FileNotFoundError(f"Local Whisper model missing: {model}")
        if threads < 1 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive threads and finite timeout required")
        self.binary, self.model, self.threads, self.timeout = binary.resolve(), model.resolve(), threads, timeout

    def transcribe(self, pcm: bytes) -> dict:
        started = time.monotonic()
        args = [str(self.binary), "-m", str(self.model), "-f", "-", "-l", "en", "-t", str(self.threads),
                "-ng", "-nt", "-np", "-nf", "-tp", "0", "-otxt", "-of", "-"]
        result = subprocess.run(args, input=wav_bytes(pcm), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=self.timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"whisper-cli failed ({result.returncode}): {result.stderr.decode(errors='replace')[-2000:]}")
        return {"transcript": " ".join(result.stdout.decode(errors="strict").split()),
                "segments": None, "model_load_seconds": None,
                "decode_seconds": time.monotonic() - started,
                "timing_scope": "cold CLI invocation including model load; not resident-model inference"}


def score(reference: str, transcript: str) -> dict:
    expected_command = re.sub(r"^hey agent\s*", "", normalize(reference))
    actual_command = re.sub(r"^hey agent\s*", "", normalize(transcript))
    expected, actual = parse_command(expected_command), parse_command(actual_command)
    words = len(normalize(reference).split())
    # Replay through the actual wake gate as well: concatenated WER alone can
    # conceal a command lost when endpointing split its wake into separate turns.
    return {"exact_match": normalize(reference) == normalize(transcript),
            "word_errors": word_errors(reference, transcript), "reference_words": words,
            "wer": word_errors(reference, transcript) / words if words else None,
            "expected_intent": expected.intent.value, "recognized_intent": actual.intent.value,
            "expected_parser_problem": expected.problem.value if expected.problem else None,
            "parser_accepts_reference": expected.intent is not Intent.UNKNOWN,
            "intent_target_match": expected.intent is not Intent.UNKNOWN and (expected.intent, expected.target) == (actual.intent, actual.target),
            "wake_phrase_present": bool(re.search(r"\bhey agent\b", normalize(transcript)))}


def wake_replay(segments: list[dict] | None) -> list[str] | None:
    if segments is None:
        return None  # Whisper is intentionally not used as a wake engine.
    gate = WakePhraseGate()
    return [event.text for segment in segments
            for event in gate.accept(segment["text"], True, segment["audio_offset_seconds"])
            if event.kind is VoiceEventKind.TRANSCRIPT]
