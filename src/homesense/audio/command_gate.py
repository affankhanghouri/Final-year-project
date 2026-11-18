"""Final-only command delivery while launch-activated listening is enabled."""

from collections import deque
from homesense.audio.events import VoiceEvent, VoiceEventKind
from homesense.commands.parser import words


class CommandGate:
    """No activation phrase required; one delivery per explicit utterance ID.

    Partial text only defers unsolicited TTS and starts a bounded command timer.
    Identical commands from different utterances remain valid (not a cooldown).
    The unused wake implementation stays independently replaceable for later work.
    """

    def __init__(self, timeout=10.0):
        self.timeout = timeout
        self.deadline = None
        self._delivered = deque(maxlen=64)

    def reset(self, *, keep_listening=False):
        if not keep_listening:
            self.deadline = None

    def tick(self, now):
        if self.deadline is not None and now >= self.deadline:
            self.reset()
            return [VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=now)]
        return []

    def accept(self, text, final, now, utterance_id=None):
        expired = self.tick(now)
        if expired:
            if utterance_id is not None:
                self._delivered.append(utterance_id)
            return expired
        if utterance_id is not None and utterance_id in self._delivered:
            return []
        events = []
        if text and self.deadline is None:
            self.deadline = now + self.timeout
            events.append(VoiceEvent(VoiceEventKind.LISTENING, occurred_at=now))
        if not final:
            return events
        if utterance_id is not None:
            self._delivered.append(utterance_id)
        if words(text) in {"hey agent", "heyagent"}:
            events.append(VoiceEvent(VoiceEventKind.ACKNOWLEDGE, occurred_at=now))
            return events
        had_command = self.deadline is not None
        self.reset()
        if text:
            events.append(VoiceEvent(VoiceEventKind.TRANSCRIPT, text, now))
        elif had_command:
            events.append(VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=now))
        return events
