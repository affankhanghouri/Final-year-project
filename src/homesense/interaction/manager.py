"""Interpret structured commands on the application control thread."""

import logging
from homesense.application import ApplicationSnapshot, ApplicationState
from homesense.audio.events import VoiceEvent, VoiceEventKind
from homesense.commands.models import Command, Intent, CommandProblem
from homesense.commands.parser import parse_command
from homesense.interaction.state import InteractionEvent, InteractionEventKind, InteractionState
from homesense.session import SessionStatus
from homesense import diagnostic_logging as diagnostic


class InteractionManager:
    def __init__(self, coordinator_factory, feedback, *, search_timeout=60.0):
        self.factory, self.feedback, self.search_timeout = coordinator_factory, feedback, search_timeout
        self.coordinator = None
        self.snapshot = ApplicationSnapshot(ApplicationState.IDLE, None, None, None, (), True)
        self.state = InteractionState.IDLE
        self.listening = False
        self.error: Exception | None = None

    def _accept(self, snapshot):
        if snapshot.session and (self.snapshot.session is None or
                                 self.snapshot.session.session_id != snapshot.session.session_id or
                                 self.snapshot.state != snapshot.state):
            logging.getLogger(__name__).info('Search target=%s state=%s', snapshot.session.target.value, snapshot.state.value)
        self.snapshot = snapshot
        self.feedback.on_snapshot(snapshot)
        if snapshot.errors or self.error:
            self.state = InteractionState.FAILED
        elif snapshot.state is ApplicationState.CLOSED:
            self.state = InteractionState.CLOSED
        elif snapshot.state is ApplicationState.STOPPING:
            self.state = InteractionState.STOPPING
        elif self.listening:
            self.state = InteractionState.LISTENING
        elif snapshot.state is ApplicationState.SEARCHING:
            self.state = InteractionState.SEARCHING
        else:
            self.state = InteractionState.IDLE
        return snapshot

    def poll(self):
        """Service perception regularly even while voice command capture is active."""
        return self._accept(self.coordinator.poll() if self.coordinator else self.snapshot)

    def on_voice_event(self, event: VoiceEvent):
        if event.kind is VoiceEventKind.LISTENING:
            self.listening = True
            self.feedback.set_listening(True)
            self.state = InteractionState.LISTENING
        elif event.kind is VoiceEventKind.ACKNOWLEDGE:
            self.listening = True
            self.feedback.set_listening(True)
            self.feedback.on_event(InteractionEvent(InteractionEventKind.ACKNOWLEDGE))
        elif event.kind is VoiceEventKind.TIMEOUT:
            self.listening = False
            self.feedback.set_listening(False)
            self.feedback.on_event(InteractionEvent(InteractionEventKind.COMMAND_TIMEOUT))
        elif event.kind is VoiceEventKind.TRANSCRIPT:
            diagnostic.event('interaction', 'stt_transcript',
                self.snapshot.session.session_id if self.snapshot.session else None,
                text=event.text, occurred_at=event.occurred_at)
            self.handle_command(parse_command(event.text))
        elif event.kind is VoiceEventKind.ERROR:
            self.fail(RuntimeError(event.text))

    def fail(self, error: Exception):
        self.error = error
        logging.getLogger(__name__).error("Interaction failed: %s", error)
        try:
            if self.coordinator is not None:
                self._accept(self.coordinator.close())
        except Exception as cleanup_error:
            logging.getLogger(__name__).error("Cleanup after interaction failure: %s", cleanup_error)
        finally:
            self.state = InteractionState.FAILED
            self.feedback.on_event(InteractionEvent(InteractionEventKind.ERROR))

    def handle_command(self, command: Command):
        """No capture, inference, parsing, or speech synthesis occurs in this method."""
        self.listening = False
        self.feedback.set_listening(False)
        self.poll()  # Repeat/confirmation must use current, eligible perception state.
        diagnostic.event('interaction', 'parser_decision',
            self.snapshot.session.session_id if self.snapshot.session else None,
            intent=command.intent.value, target=command.target.value if command.target else None,
            accepted=command.intent is not Intent.UNKNOWN,
            problem=command.problem.value if command.problem else None)
        self.state = InteractionState.PROCESSING
        logging.getLogger(__name__).info('Parsed command: intent=%s target=%s %s%s', command.intent.value,
                command.target.value if command.target else None,
                'rejected' if command.intent is Intent.UNKNOWN else 'accepted',
                f' ({command.problem.value})' if command.problem else '')
        try:
            if command.intent is Intent.FIND_OBJECT:
                if command.target is None:
                    raise ValueError("find command requires a canonical target")
                if self.coordinator is None:
                    self.coordinator = self.factory()  # Load YOLO once, only on first search.
                self._accept(self.coordinator.start_search(command.target.value, self.search_timeout))
            elif command.intent in (Intent.CANCEL_SEARCH, Intent.CONFIRM_FOUND):
                session = self.snapshot.session
                if self.snapshot.state is not ApplicationState.SEARCHING or session is None or session.status is not SessionStatus.ACTIVE:
                    self.feedback.on_event(InteractionEvent(InteractionEventKind.NO_SEARCH))
                elif command.target is not None and command.target != session.target:
                    self.feedback.on_event(InteractionEvent(InteractionEventKind.TARGET_MISMATCH))
                else:
                    operation = (self.coordinator.complete_search if command.intent is Intent.CONFIRM_FOUND
                                 else self.coordinator.cancel_search)
                    self._accept(operation(session.session_id))
            else:
                event = {Intent.HELP: InteractionEventKind.HELP,
                         Intent.REPEAT_GUIDANCE: InteractionEventKind.REPEAT}.get(command.intent, InteractionEventKind.UNKNOWN)
                if command.intent is not Intent.UNKNOWN or command.problem in (CommandProblem.UNSUPPORTED, CommandProblem.AMBIGUOUS):
                    self.feedback.on_event(InteractionEvent(event, command))
        except Exception as error:
            self.fail(error)
        self._accept(self.snapshot)

    def close(self):
        """Retain the coordinator for repeated cleanup if a native worker is blocked."""
        self.listening = False
        if self.coordinator is not None:
            return self._accept(self.coordinator.close())
        self.state = InteractionState.CLOSED
        return self.snapshot
