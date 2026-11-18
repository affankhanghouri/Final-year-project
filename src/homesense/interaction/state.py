from dataclasses import dataclass
from enum import Enum
from homesense.commands.models import Command


class InteractionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SEARCHING = "searching"
    STOPPING = "stopping"
    FAILED = "failed"
    CLOSED = "closed"


class InteractionEventKind(str, Enum):
    READY = "ready"
    ACKNOWLEDGE = "acknowledge"
    COMMAND_TIMEOUT = "command_timeout"
    UNKNOWN = "unknown"
    NO_SEARCH = "no_search"
    TARGET_MISMATCH = "target_mismatch"
    HELP = "help"
    REPEAT = "repeat"
    ERROR = "error"


@dataclass(frozen=True)
class InteractionEvent:
    kind: InteractionEventKind
    command: Command | None = None
    detail: str = ""
