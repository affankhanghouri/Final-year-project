"""ASR-transcript phrase gate, NOT a specialized acoustic keyword model."""

import re
from homesense.audio.events import VoiceEvent, VoiceEventKind


class WakePhraseGate:
    def __init__(self, timeout: float = 10.0, phrase: str = "hey agent"):
        self.phrase = phrase.lower()
        self.timeout = timeout
        self.deadline: float | None = None
        self.utterance_has_wake = False

    def reset(self, *, keep_listening: bool = False):
        self.utterance_has_wake = False
        if not keep_listening:
            self.deadline = None

    def tick(self, now: float) -> list[VoiceEvent]:
        if self.deadline is not None and now >= self.deadline:
            self.reset()
            return [VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=now)]
        return []

    def accept(self, text: str, final: bool, now: float) -> list[VoiceEvent]:
        events = self.tick(now)
        if events:
            return events  # Do not re-arm from the same utterance that just expired.
        text = " ".join(re.findall(r"[a-z]+", text.lower()))
        match = re.search(r"\b" + re.escape(self.phrase) + r"\b", text)
        if match:
            if self.deadline is None:
                self.deadline = now + self.timeout
                events.append(VoiceEvent(VoiceEventKind.LISTENING, occurred_at=now))
            self.utterance_has_wake = True
            command = text[match.end():].strip()
        else:
            command = text
        if not final or self.deadline is None:
            return events
        if self.utterance_has_wake and not match:
            # A revised final transcript lost its partial wake: require a new wake.
            self.reset()
            events.append(VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=now))
            return events
        if command:
            events.append(VoiceEvent(VoiceEventKind.TRANSCRIPT, command, now))
            self.reset()
        elif match:
            # Only acknowledge after endpointing, preserving command speech in one utterance.
            events.append(VoiceEvent(VoiceEventKind.ACKNOWLEDGE, occurred_at=now))
            self.utterance_has_wake = False
        return events
