"""Cancelable offline speech and a single bounded latest-message output worker."""

from dataclasses import dataclass, replace, field
from collections import OrderedDict
from enum import Enum
from contextlib import ExitStack
import logging
import math
import shutil
import subprocess
import sys
import threading
import time
import tempfile
from typing import Callable, Protocol
from uuid import UUID, uuid4

from homesense.audio.input import PlaybackGate
from homesense.config import AudioSettings
from homesense import diagnostic_logging as diagnostic


@dataclass(frozen=True)
class SpeechMessage:
    text: str
    generation: int
    condition: str | None = None
    priority: int = 10
    session_id: UUID | None = field(default=None, compare=False)
    message_id: UUID = field(default_factory=uuid4, compare=False)
    fresh_guidance: bool = False


class DeliveryStatus(str, Enum):
    ACCEPTED = 'accepted'
    STARTED = 'started'
    COMPLETED = 'completed'
    DISCARDED = 'discarded'
    INTERRUPTED = 'interrupted'
    REJECTED = 'rejected'
    FAILED = 'failed'


@dataclass(frozen=True)
class SpeechDelivery:
    message_id: UUID
    status: DeliveryStatus
    changed_at: float
    started_at: float | None = None
    reason: str = ''


def _speech_event(kind, message, *, important=True, **data):
    diagnostic.event('speech', kind, message.session_id, important=important,
        message_id=message.message_id, text=message.text, generation=message.generation,
        condition=message.condition, priority=message.priority, fresh_guidance=message.fresh_guidance, **data)


class SpeechValidity:
    """Control-thread published permission; output thread never queries a coordinator."""

    def __init__(self, *, clock=time.monotonic):
        self._lock = threading.Lock()
        self.clock = clock
        self.generation = 0
        self.condition = None
        self.deadline = None

    def update(self, generation: int, condition: str | None, deadline: float | None):
        with self._lock:
            self.generation, self.condition, self.deadline = generation, condition, deadline

    def allows(self, message: SpeechMessage) -> bool:
        with self._lock:
            return (message.generation == self.generation and
                    (message.condition is None or (message.condition == self.condition and
                     self.deadline is not None and self.clock() < self.deadline)))


class SpeechBackend(Protocol):
    def speak(self, text: str, should_stop: Callable[[], bool]) -> None: ...


class ConsoleSpeech:
    """Explicit debug output; never presented as functioning audio playback."""

    def speak(self, text, should_stop):
        if not should_stop():
            print(f"HomeSense: {text}", flush=True)


class EspeakSpeech:
    def __init__(self, settings: AudioSettings):
        self.settings = settings
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.binary is None:
            raise RuntimeError("eSpeak NG is missing; install espeak-ng (see docs/audio-setup.md)")
        if settings.output_device and shutil.which("aplay") is None:
            raise RuntimeError("ALSA output selection requires aplay (alsa-utils)")

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)

    def speak(self, text, should_stop):
        args = [self.binary, "-v", self.settings.voice, "-s", str(self.settings.speaking_rate),
                "-a", str(round(self.settings.volume * 100))]
        processes = []
        diagnostics = []
        with ExitStack() as stack:
            def launch(command, **kwargs):
                error_output = stack.enter_context(tempfile.TemporaryFile())
                # The control thread handles terminal SIGINT and cancels through
                # should_stop. Children must not die independently from Ctrl+C.
                process = subprocess.Popen(command, stderr=error_output,
                                           start_new_session=True, **kwargs)
                processes.append(process)
                diagnostics.append((command[0], error_output))
                return process
            try:
                self._speak(args, text, should_stop, launch, processes, diagnostics)
            finally:
                primary_error = sys.exception()
                cleanup_errors = []
                for process in reversed(processes):
                    try:
                        self._terminate(process)
                    except Exception as error:
                        cleanup_errors.append(error)
                        logging.getLogger(__name__).error("Speech process cleanup failed: %s", error)
                if cleanup_errors and primary_error is None:
                    raise cleanup_errors[0]

    def _speak(self, args, text, should_stop, launch, processes, diagnostics):
        if should_stop():
            return
        if self.settings.output_device:
            synth = launch(args + ["--stdout", text], stdout=subprocess.PIPE)
            try:
                launch(["aplay", "-q", "-D", self.settings.output_device],
                       stdin=synth.stdout, stdout=subprocess.DEVNULL)
            finally:
                synth.stdout.close()
        else:
            launch(args + [text], stdout=subprocess.DEVNULL)
        while any(p.poll() is None for p in processes):
            if should_stop():
                return
            time.sleep(0.02)
        # Cancellation can race the last poll. It is not a device failure.
        if should_stop():
            return
        failures = []
        for process, (name, errors) in zip(processes, diagnostics):
            if process.returncode != 0:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 2048))
                detail = errors.read().decode(errors='replace').strip()
                failures.append(f"{name} exited {process.returncode}: {detail}")
        if failures:
            raise RuntimeError(f"Speech playback failed (voice={self.settings.voice}, "
                               f"output={self.settings.output_device or 'system default'}): " + '; '.join(failures))


@dataclass(frozen=True)
class SpeechStats:
    submitted: int = 0
    started: int = 0
    completed: int = 0
    cancelled: int = 0
    replaced: int = 0
    rejected: int = 0
    failed: int = 0
    queue_delay_seconds: float = 0.0
    playback_seconds: float = 0.0
    pending: int = 0
    discarded: int = 0


class SpeechWorker:
    """At most one pending message; high priority may interrupt current playback.

    Only mechanics live here. FeedbackManager owns all wording and policy.
    Backends must honor should_stop; bounded stop reports a misbehaving backend.
    """

    def __init__(self, backend: SpeechBackend, validity: SpeechValidity, playback: PlaybackGate,
                 *, listening_gap=1.0, guidance_burst_limit=2, listening_window=3.0, clock=time.monotonic):
        if type(guidance_burst_limit) is not int or guidance_burst_limit < 1:
            raise ValueError('guidance_burst_limit must be positive')
        if not math.isfinite(listening_window) or listening_window < 0:
            raise ValueError('listening_window must be finite and nonnegative')
        self.backend, self.validity, self.playback = backend, validity, playback
        self.listening_gap, self.clock = listening_gap, clock
        self._condition = threading.Condition()
        self._pending: SpeechMessage | None = None
        self._pending_at = 0.0
        self._current: SpeechMessage | None = None
        self._interrupt = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._next_allowed = 0.0
        self.error: Exception | None = None
        self._stats = SpeechStats()
        self._deliveries: OrderedDict[UUID, SpeechDelivery] = OrderedDict()
        self._guidance_streak = 0
        self._listen_until = 0.0
        self._burst_limit, self._listening_window = guidance_burst_limit, listening_window

    def delivery(self, message_id: UUID) -> SpeechDelivery | None:
        """Control-thread receipt lookup; a bounded ledger, not a speech queue."""
        with self._condition:
            return self._deliveries.get(message_id)

    def _delivered(self, message, status, reason=''):
        with self._condition:
            old = self._deliveries.get(message.message_id)
            if (status is DeliveryStatus.REJECTED and old is not None
                    and old.status in (DeliveryStatus.ACCEPTED, DeliveryStatus.STARTED)):
                return  # Re-submitting the same ID cannot undo its first acceptance.
            now = self.clock()
            self._deliveries[message.message_id] = SpeechDelivery(message.message_id, status, now,
                now if status is DeliveryStatus.STARTED else old.started_at if old else None, reason)
            self._deliveries.move_to_end(message.message_id)
            while len(self._deliveries) > 64:
                self._deliveries.popitem(last=False)

    @property
    def stats(self) -> SpeechStats:
        with self._condition:
            return replace(self._stats, pending=int(self._pending is not None))

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("speech worker is single-use")
        thread = threading.Thread(target=self._run, name="speech-output")
        try:
            thread.start()
        except Exception as error:
            self.error = error
            self._stop.set()
            raise
        self._thread = thread

    def submit(self, message: SpeechMessage) -> bool:
        if self._stop.is_set() or not self.validity.allows(message):
            self._delivered(message, DeliveryStatus.REJECTED, 'stopped_or_obsolete')
            _speech_event('speech_rejected', message, important=False, reason='stopped_or_obsolete')
            return False
        with self._condition:
            if self._pending is not None and self.validity.allows(self._pending) and self._pending.priority > message.priority:
                self._delivered(message, DeliveryStatus.REJECTED, 'higher_priority_pending')
                _speech_event('speech_rejected', message, important=False, reason='higher_priority_pending')
                self._stats = replace(self._stats, rejected=self._stats.rejected + 1)
                return False
            if message.priority <= 20 and message in (self._pending, self._current):
                self._delivered(message, DeliveryStatus.REJECTED, 'duplicate_current_or_pending')
                _speech_event('speech_rejected', message, important=False, reason='duplicate_current_or_pending')
                self._stats = replace(self._stats, rejected=self._stats.rejected + 1)
                return False
            self._stats = replace(self._stats, submitted=self._stats.submitted + 1,
                                  replaced=self._stats.replaced + int(self._pending is not None))
            if self._pending is not None:
                self._delivered(self._pending, DeliveryStatus.DISCARDED, 'replaced')
                self._stats = replace(self._stats, discarded=self._stats.discarded+1)
                _speech_event('speech_replaced', self._pending, replacement_id=message.message_id)
            self._pending = message
            self._pending_at = self.clock()
            self._delivered(message, DeliveryStatus.ACCEPTED)
            if self._current is not None and (message.priority > self._current.priority or
                                              not self.validity.allows(self._current)):
                self._interrupt.set()
            self._condition.notify_all()
            _speech_event('speech_queued', message)
            return True

    @property
    def busy(self) -> bool:
        with self._condition:
            return self._current is not None or self._pending is not None

    def stop(self, timeout=2.0) -> bool:
        self._stop.set()
        self._interrupt.set()
        with self._condition:
            if self._pending is not None:
                self._delivered(self._pending, DeliveryStatus.DISCARDED, 'shutdown_before_playback')
                self._stats = replace(self._stats, discarded=self._stats.discarded+1)
                _speech_event('speech_cancelled', self._pending, reason='shutdown_before_playback', playback_seconds=0.)
            self._pending = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                with self._condition:
                    self._condition.wait_for(lambda: self._stop.is_set() or self._pending is not None)
                    if self._stop.is_set():
                        return
                    message = self._pending
                    # Discard invalid work promptly, including during a protected
                    # listening window. New fresh evidence can be requested later.
                    if not self.validity.allows(message):
                        self._pending = None
                        self._delivered(message, DeliveryStatus.DISCARDED, 'obsolete_before_playback')
                        self._stats = replace(self._stats, discarded=self._stats.discarded+1)
                        _speech_event('speech_discarded', message, reason='obsolete_before_playback')
                        continue
                    ready_at = max(self._listen_until,
                                   0.0 if message.fresh_guidance else self._next_allowed)
                    if message.priority < 90 and self.clock() < ready_at:
                        self._condition.wait(timeout=min(0.05, ready_at - self.clock()))
                        continue
                    self._pending = None
                    self._current = message
                    queued_at = self._pending_at
                    self._interrupt.clear()
                should_stop = lambda: (self._stop.is_set() or self._interrupt.is_set() or
                                       not self.validity.allows(message))
                if should_stop():
                    self._delivered(message, DeliveryStatus.DISCARDED, 'obsolete_or_stopped_before_playback')
                    _speech_event('speech_discarded', message, reason='obsolete_or_stopped_before_playback')
                    with self._condition:
                        self._current = None
                        self._stats = replace(self._stats, discarded=self._stats.discarded+1)
                    continue
                self.playback.set_playing(True)
                started = self.clock()
                self._delivered(message, DeliveryStatus.STARTED)
                _speech_event('speech_started', message, queue_seconds=max(0., started-queued_at), microphone_suppressed=True)
                with self._condition:
                    self._stats = replace(self._stats, started=self._stats.started + 1,
                                          queue_delay_seconds=max(0.0, started - queued_at))
                failed = False
                try:
                    self.backend.speak(message.text, should_stop)
                except Exception:
                    failed = True
                    raise
                finally:
                    self.playback.set_playing(False)
                    self._next_allowed = self.clock() + self.listening_gap
                    with self._condition:
                        cancelled = should_stop()
                        self._delivered(message, DeliveryStatus.FAILED if failed else
                            DeliveryStatus.INTERRUPTED if cancelled else DeliveryStatus.COMPLETED)
                        if message.fresh_guidance:
                            self._guidance_streak += 1
                            if self._guidance_streak >= self._burst_limit:
                                self._listen_until = self.playback.listening_ready_at+self._listening_window
                                self._guidance_streak = 0
                                _speech_event('speech_listening_window', message, important=False,
                                    listening_ready_at=self.playback.listening_ready_at,
                                    listening_until=self._listen_until, seconds=self._listening_window)
                        elif message.priority >= 90:
                            self._guidance_streak = 0
                        self._stats = replace(self._stats,
                            completed=self._stats.completed + int(not failed and not cancelled),
                            cancelled=self._stats.cancelled + int(not failed and cancelled),
                            failed=self._stats.failed + int(failed),
                            playback_seconds=max(0.0, self.clock() - started))
                        self._current = None
                    _speech_event('speech_failed' if failed else 'speech_cancelled' if cancelled else 'speech_finished',
                        message, playback_seconds=max(0., self.clock()-started),
                        reason='backend_error' if failed else 'stop_interrupt_or_obsolete' if cancelled else 'backend_returned')
        except Exception as error:
            self.error = error
            diagnostic.event('speech', 'speech_error', message.session_id if 'message' in locals() else None,
                             severity='ERROR', error=str(error))
            self._stop.set()
            with self._condition:
                if self._pending is not None:
                    self._delivered(self._pending, DeliveryStatus.DISCARDED, 'output_failed')
                    self._stats = replace(self._stats, discarded=self._stats.discarded+1)
                self._pending = None
                self._condition.notify_all()
            logging.getLogger(__name__).error("Speech output failed: %s", error)
