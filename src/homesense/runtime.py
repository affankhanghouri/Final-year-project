"""Composition and control loop; audio workers communicate only through events."""

import logging
import time

from homesense.application import ApplicationCoordinator
from homesense.audio.events import VoiceEvent, VoiceEventKind, VoiceEvents
from homesense.audio.input import PlaybackGate, SoundDeviceInput, VoiceWorker
from homesense.audio.speech_output import ConsoleSpeech, EspeakSpeech, SpeechValidity, SpeechWorker
from homesense.config import RuntimeSettings
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.manager import InteractionManager
from homesense.interaction.state import InteractionEvent, InteractionEventKind
from homesense import diagnostic_logging as diagnostic


def perception_factory(settings: RuntimeSettings, *, speech_busy=None):
    def create():
        from homesense.detector import InferenceSettings, YoloDetector
        detector = YoloDetector(settings.checkpoint, settings=InferenceSettings(device="cpu"))
        diagnostic.event('detector', 'detector_ready', checkpoint=str(settings.checkpoint),
                         class_names=detector.class_names)
        options = {}
        if settings.guided_scan:
            from homesense.motion import MotionSettings
            from homesense.search_strategy import ScanSettings
            options = dict(motion=MotionSettings(), scan=ScanSettings(initial_observe_seconds=settings.feedback.no_detection_delay))
            if speech_busy is not None:
                options['scan_speech_busy'] = speech_busy
        return ApplicationCoordinator(detector, camera_id=settings.camera_id, shutdown_timeout=settings.shutdown_timeout, **options)
    return create


class HomeSenseRuntime:
    """Construct/use on the control thread. start/step/close permit deterministic tests."""

    def __init__(self, interaction, feedback, speech, events, *, voice=None, poll_interval=0.02, event_max_age=5.0, speech_shutdown_grace=8.0, clock=time.monotonic):
        self.interaction, self.feedback, self.speech = interaction, feedback, speech
        self.events, self.voice, self.poll_interval = events, voice, poll_interval
        self.clock, self.event_max_age = clock, event_max_age
        self.speech_shutdown_grace = speech_shutdown_grace
        self.failed = False
        self.started = False

    def start(self):
        self.speech.start()
        self.started = True
        try:
            if self.voice is not None:
                self.voice.start()
            self.feedback.on_event(InteractionEvent(InteractionEventKind.READY))
        except Exception:
            self.close()
            raise

    def step(self):
        """Poll on every tick; bounded events cannot starve perception indefinitely."""
        try:
            self.interaction.poll()
        except Exception as error:
            self.failed = True
            self.interaction.fail(error)
            return
        if self.speech.error is not None:
            self.failed = True
            self.interaction.fail(self.speech.error)
            return
        for event in self.events.drain():
            if event.kind is not VoiceEventKind.ERROR and self.clock() - event.occurred_at >= self.event_max_age:
                diagnostic.event('runtime', 'voice_event_discarded', reason='event_too_old',
                    voice_event_kind=event.kind.value, age_seconds=self.clock()-event.occurred_at, max_age=self.event_max_age)
                self.interaction.on_voice_event(VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=self.clock()))
                continue
            self.interaction.on_voice_event(event)
            if self.interaction.error is not None:
                self.failed = True
                break
        if self.interaction.snapshot.errors:
            self.failed = True

    def close(self):
        """Attempt every owned cleanup independently; safe to retry retained workers."""
        complete = True
        try:
            snapshot = self.interaction.close()
            complete = snapshot.cleanup_complete
            self.failed |= bool(snapshot.errors) or not complete
        except Exception:
            logging.exception("Perception shutdown failed")
            self.failed, complete = True, False
        if self.voice is not None:
            try:
                if not self.voice.stop():
                    self.failed, complete = True, False
                self.failed |= self.voice.error is not None or self.voice.cleanup_error is not None
                if self.voice.cleanup_error is not None:
                    complete = False
            except Exception:
                logging.exception("Audio input shutdown failed")
                self.failed, complete = True, False
        # Perception is invalidated before waiting briefly for terminal feedback.
        deadline = time.monotonic() + self.speech_shutdown_grace
        while self.started and self.speech.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        try:
            if not self.speech.stop():
                self.failed, complete = True, False
            self.failed |= self.speech.error is not None
        except Exception:
            logging.exception("Speech output shutdown failed")
            self.failed, complete = True, False
        return complete


def compose(settings: RuntimeSettings, *, text_mode=False, console_output=False) -> HomeSenseRuntime:
    validity = SpeechValidity()
    playback = PlaybackGate(settings.audio.echo_tail)
    backend = ConsoleSpeech() if console_output else EspeakSpeech(settings.audio)
    speech = SpeechWorker(backend, validity, playback, listening_gap=settings.audio.listening_gap,
        guidance_burst_limit=settings.audio.guidance_burst_limit, listening_window=settings.audio.listening_window)
    feedback = FeedbackManager(speech, validity, settings=settings.feedback)
    # Read the existing bounded output's pending/playback state on the control
    # thread. The strategy pauses effort timers, not camera/motion/inference.
    interaction = InteractionManager(perception_factory(settings, speech_busy=lambda: speech.busy), feedback, search_timeout=settings.search_timeout)
    events = VoiceEvents()
    voice = None
    if not text_mode:
        from homesense.audio.speech_recognition import VoskCommandRecognizer
        recognizer = VoskCommandRecognizer(settings.audio.model_path, settings.audio.sample_rate,
                                          max_seconds=settings.audio.command_timeout)
        voice = VoiceWorker(SoundDeviceInput(settings.audio), recognizer, events, playback, settings.audio)
    return HomeSenseRuntime(interaction, feedback, speech, events, voice=voice, poll_interval=settings.poll_interval, event_max_age=settings.event_max_age, speech_shutdown_grace=settings.speech_shutdown_grace)
