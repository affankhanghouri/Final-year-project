from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full, Queue
import threading


class VoiceEventKind(str, Enum):
    LISTENING = "listening"
    ACKNOWLEDGE = "acknowledge"
    TRANSCRIPT = "transcript"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceEvent:
    kind: VoiceEventKind
    text: str = ""
    occurred_at: float = 0.0


class VoiceEvents:
    """Bounded worker-to-control handoff. Overflow is explicit, never silent action loss."""

    def __init__(self, capacity: int = 32):
        self.queue = Queue(maxsize=capacity)
        self.overflow = threading.Event()

    def put(self, event: VoiceEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except Full:
            self.overflow.set()

    def drain(self) -> list[VoiceEvent]:
        if self.overflow.is_set():
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            self.overflow.clear()
            return [VoiceEvent(VoiceEventKind.ERROR, "Voice event buffer overflow; command discarded.")]
        result = []
        while True:
            try:
                result.append(self.queue.get_nowait())
            except Empty:
                return result
