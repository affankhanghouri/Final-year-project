"""Typed settings for voice/runtime behavior; perception defaults remain unchanged."""

from dataclasses import dataclass, field
import math
from pathlib import Path


def positive(name: str, value: float, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} finite")


@dataclass(frozen=True)
class AudioSettings:
    model_path: Path = Path("models/speech/vosk-model-small-en-us-0.15")
    input_device: int | str | None = None
    output_device: str | None = None  # ALSA name for aplay, independent of PortAudio input indices.
    sample_rate: int = 16000
    chunk_ms: int = 100
    capture_buffer_seconds: float = 3.0
    silence_seconds: float = 0.6
    silence_rms: float = 0.015
    command_timeout: float = 10.0
    wake_phrase: str = "hey agent"
    voice: str = "en-us"
    speaking_rate: int = 155
    volume: float = 0.8
    echo_tail: float = 0.35
    listening_gap: float = 1.0
    guidance_burst_limit: int = 2
    listening_window: float = 3.0

    def __post_init__(self):
        for name in ("sample_rate", "chunk_ms", "speaking_rate"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_rate != 16000:
            raise ValueError('The audio pipeline requires 16000 Hz mono PCM16')
        for name in ("silence_seconds", "command_timeout", "capture_buffer_seconds"):
            positive(name, getattr(self, name))
        for name in ("echo_tail", "listening_gap"):
            positive(name, getattr(self, name), zero=True)
        positive('listening_window', self.listening_window)
        if type(self.guidance_burst_limit) is not int or self.guidance_burst_limit < 1:
            raise ValueError('guidance_burst_limit must be a positive integer')
        if isinstance(self.volume, bool) or not isinstance(self.volume, (int, float)) or not 0 <= self.volume <= 1:
            raise ValueError("volume must be between 0 and 1")
        if not 0 <= self.silence_rms <= 1:
            raise ValueError("silence_rms must be between 0 and 1")
        if not self.voice.strip() or self.wake_phrase.strip().lower() != "hey agent":
            raise ValueError("voice is required; the supported wake phrase is Hey Agent")


@dataclass(frozen=True)
class FeedbackSettings:
    no_detection_delay: float = 5.0
    no_detection_repeat: float = 15.0  # Compatibility only; unchanged absence is no longer repeated.
    direction_cooldown: float = 3.0
    loss_cooldown: float = 6.0
    loss_delay: float = 0.8
    scan_alert_gap: float = 3.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            positive(name, getattr(self, name), zero=True)


@dataclass(frozen=True)
class RuntimeSettings:
    checkpoint: Path = Path("models/best.pt")
    camera_id: int = 0
    search_timeout: float = 60.0
    shutdown_timeout: float = 2.0
    poll_interval: float = 0.02
    event_max_age: float = 5.0
    speech_shutdown_grace: float = 8.0
    audio: AudioSettings = field(default_factory=AudioSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    guided_scan: bool = False

    def __post_init__(self):
        if type(self.guided_scan) is not bool:
            raise ValueError('guided_scan must be boolean')
        for name in ("search_timeout", "shutdown_timeout", "poll_interval", "event_max_age"):
            positive(name, getattr(self, name))
        positive("speech_shutdown_grace", self.speech_shutdown_grace, zero=True)
