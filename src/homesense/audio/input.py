"""One microphone stream, bounded PCM handoff, and a single recognition worker."""

from queue import Empty, Full, Queue
import threading
import time
import math
from typing import Protocol

from homesense.audio.events import VoiceEvent, VoiceEventKind, VoiceEvents
from homesense.audio.speech_recognition import Recognizer
from homesense.audio.command_gate import CommandGate
from homesense.config import AudioSettings
from homesense import diagnostic_logging as diagnostic


class AudioSource(Protocol):
    def open(self) -> None: ...
    def read(self, timeout: float) -> bytes | None: ...
    def drain(self) -> None: ...
    def close(self) -> None: ...


class PlaybackGate:
    """Speech worker publishes mute epochs; recognizer discards straddling audio."""

    def __init__(self, tail: float = 0.35, *, clock=time.monotonic):
        self._lock = threading.Lock()
        self._playing = False
        self._until = 0.0
        self._epoch = 0
        self._tail = tail
        self._clock = clock

    def set_playing(self, playing: bool):
        with self._lock:
            self._playing = playing
            self._epoch += 1
            if not playing:
                self._until = self._clock() + self._tail

    def snapshot(self) -> tuple[bool, int]:
        with self._lock:
            return self._playing or self._clock() < self._until, self._epoch

    @property
    def listening_ready_at(self) -> float:
        """Echo-tail boundary after output returns; never unmute active playback."""
        with self._lock:
            return float('inf') if self._playing else self._until


class SoundDeviceInput:
    def __init__(self, settings: AudioSettings):
        self.settings = settings
        # Short native decoding spikes must not overflow an eight-block (~0.8 s)
        # queue. A combined CPU probe exceeded two seconds; retain a bounded
        # three seconds / 96 KB of PCM by default, never disk.
        self._chunks = Queue(maxsize=max(1, math.ceil(settings.capture_buffer_seconds * 1000 / settings.chunk_ms)))
        self._failure = threading.Event()
        self._stream = None

    def _callback(self, data, frames, timing, status):
        if status:
            self._failure.set()
            return
        try:
            self._chunks.put_nowait(bytes(data))
        except Full:
            self._failure.set()  # Do not concatenate discontinuous command audio silently.

    def open(self):
        import sounddevice as sd
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.settings.sample_rate, channels=1, dtype="int16",
                blocksize=self.settings.sample_rate * self.settings.chunk_ms // 1000,
                device=self.settings.input_device, callback=self._callback)
            self._stream.start()
            if diagnostic.enabled():
                diagnostic.event('audio', 'microphone_stream_ready', input_device=self.settings.input_device,
                    configured_rate=self.settings.sample_rate, actual_rate=float(self._stream.samplerate),
                    channels=1, pcm_format='signed_int16', chunk_ms=self.settings.chunk_ms)
        except Exception:
            try:
                self.close()
            except Exception as cleanup_error:
                import logging
                logging.getLogger(__name__).error("Microphone cleanup after startup failed: %s", cleanup_error)
            raise

    def read(self, timeout=0.1):
        if self._failure.is_set():
            raise RuntimeError("Microphone stream overflow or device error")
        if self._stream is not None and not self._stream.active:
            raise RuntimeError("Microphone stream ended unexpectedly")
        try:
            return self._chunks.get(timeout=timeout)
        except Empty:
            return None

    def drain(self):
        while True:
            try:
                self._chunks.get_nowait()
            except Empty:
                return

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        self.drain()


class VoiceWorker:
    """Own source and recognizer; send events only, never call perception controls."""

    def __init__(self, source: AudioSource, recognizer: Recognizer, events: VoiceEvents,
                 playback: PlaybackGate, settings: AudioSettings, *, clock=time.monotonic):
        self.source, self.recognizer, self.events = source, recognizer, events
        self.playback, self.settings, self.clock = playback, settings, clock
        self.gate = CommandGate(settings.command_timeout)
        self._stop = threading.Event()
        self._thread = None
        self.error: Exception | None = None
        self.cleanup_error: Exception | None = None

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("voice worker is single-use")
        thread = threading.Thread(target=self._run, name="voice-input")
        try:
            thread.start()
        except Exception as error:
            self.error = error
            self._stop.set()
            raise
        self._thread = thread

    def stop(self, timeout=2.0) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _run(self):
        muted_before = False
        epoch_before = self.playback.snapshot()[1]
        try:
            self.source.open()
            diagnostic.event('audio', 'microphone_ready', sample_rate=self.settings.sample_rate,
                             chunk_ms=self.settings.chunk_ms)
            while not self._stop.is_set():
                muted, epoch = self.playback.snapshot()
                if muted:
                    self.source.drain()
                if muted != muted_before or epoch != epoch_before:
                    diagnostic.event('audio', 'recognition_suppression', important=False,
                                     suppressed=muted, playback_epoch=epoch)
                    self.source.drain()
                    self.recognizer.reset()
                    # Preserve the deliberate two-turn command window across its acknowledgement.
                    self.gate.reset(keep_listening=True)
                muted_before, epoch_before = muted, epoch
                chunk = self.source.read(0.1)
                expired = self.gate.tick(self.clock())
                for event in expired:
                    self.events.put(event)
                    discard = getattr(self.recognizer, 'discard', self.recognizer.reset)
                    discard()
                if expired or muted or chunk is None:
                    continue
                update = self.recognizer.feed(chunk)
                muted_after, epoch_after = self.playback.snapshot()
                if muted_after or epoch_after != epoch:
                    if update.final:
                        diagnostic.event('audio', 'stt_discarded', text=update.text,
                            reason='playback_epoch_changed_during_decode', utterance_id=update.utterance_id)
                    muted_before = True
                    continue
                if update.final:
                    diagnostic.event('audio', 'stt_final', text=update.text, utterance_id=update.utterance_id)
                    import logging
                    logging.getLogger(__name__).info('Final transcript: %r', update.text)
                for event in self.gate.accept(update.text, update.final, self.clock(), update.utterance_id):
                    self.events.put(event)
                    if event.kind is VoiceEventKind.TIMEOUT and not update.final:
                        # A native decode can cross the deadline after tick().
                        # Quarantine that unfinished utterance, including its tail.
                        discard = getattr(self.recognizer, 'discard', self.recognizer.reset)
                        discard()
        except Exception as error:
            diagnostic.event('audio', 'audio_error', severity='ERROR', error=str(error))
            self.error = error
            self.events.put(VoiceEvent(VoiceEventKind.ERROR, str(error), self.clock()))
        finally:
            errors = []
            # Clear transient utterance PCM, then release the microphone even if
            # decoder cleanup fails. Retain every cleanup failure for close().
            for cleanup in (self.recognizer.reset, self.source.close):
                try:
                    cleanup()
                except Exception as error:
                    errors.append(error)
                    self.events.put(VoiceEvent(VoiceEventKind.ERROR, str(error), self.clock()))
            if errors:
                self.cleanup_error = errors[0] if len(errors) == 1 else ExceptionGroup('Voice cleanup failed', errors)
            diagnostic.event('audio', 'microphone_closed', errors=[str(e) for e in errors])
