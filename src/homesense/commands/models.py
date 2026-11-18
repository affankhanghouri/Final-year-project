from dataclasses import dataclass
from enum import Enum
from homesense.session import Target


class Intent(str, Enum):
    FIND_OBJECT = "find_object"
    CANCEL_SEARCH = "cancel_search"
    REPEAT_GUIDANCE = "repeat_guidance"
    CONFIRM_FOUND = "confirm_found"
    HELP = "help"
    UNKNOWN = "unknown"


class CommandProblem(str, Enum):
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True)
class Command:
    intent: Intent
    target: Target | None = None
    problem: CommandProblem | None = None
    candidates: tuple[Target, ...] = ()
