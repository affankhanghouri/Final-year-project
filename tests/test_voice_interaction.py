from dataclasses import replace
from unittest.mock import Mock
import unittest

from homesense.application import ApplicationSnapshot, ApplicationState, ApplicationError, SearchOutcome
from homesense.audio.events import VoiceEvent, VoiceEventKind
from homesense.audio.speech_output import SpeechValidity
from homesense.camera import Frame
from homesense.commands.parser import parse_command
from homesense.config import FeedbackSettings
from homesense.guidance import Direction, GuidanceResult, GuidanceStatus
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.interaction.manager import InteractionManager
from homesense.interaction.state import InteractionState, InteractionEvent, InteractionEventKind
from homesense.session import SessionManager, SessionStatus
from homesense.stability import Visibility, VisibilitySnapshot
from homesense.perception import DetectionBatch
import numpy as np


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SnapshotFactory:
    def __init__(self, clock):
        self.clock = clock
        self.sessions = SessionManager(clock=clock)
        self.session = self.sessions.start("phone", 60)

    def make(self, *, direction=Direction.LEFT, visibility=Visibility.VISIBLE, status=GuidanceStatus.DIRECTION):
        now = self.clock()
        frame = Frame(1, now, np.zeros((10, 10, 3), dtype=np.uint8))
        batch = DetectionBatch(self.session.session_id, frame, now, 0, ())
        view = VisibilitySnapshot(self.session.session_id, self.session.target, visibility, now, now, 1, now+0.4)
        guide = GuidanceResult(self.session.session_id, self.session.target, status,
                               direction if status is GuidanceStatus.DIRECTION else None,
                               .25, 1, now, now, now+0.4)
        return ApplicationSnapshot(ApplicationState.SEARCHING, self.session, None, None, (), False, batch, view, guide)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = SnapshotFactory(self.clock)
        self.messages = []
        self.output = Mock()
        self.output.submit.side_effect = lambda message: self.messages.append(message) or True
        self.validity = SpeechValidity(clock=self.clock)
        self.feedback = FeedbackManager(self.output, self.validity, clock=self.clock)

    def texts(self):
        return [m.text for m in self.messages]

    def test_no_identical_direction_at_frame_rate(self):
        for _ in range(100):
            self.feedback.on_snapshot(self.factory.make())
        self.assertEqual(len(self.messages), 2)  # start + one direction
        self.assertEqual("Phone on your left.", self.messages[-1].text)

    def test_direction_change_cooldown_and_latest_direction(self):
        self.feedback.on_snapshot(self.factory.make())
        self.clock.now += 1
        self.feedback.on_snapshot(self.factory.make(direction=Direction.RIGHT))
        self.assertEqual(len(self.messages), 2)
        self.clock.now += 3
        self.feedback.on_snapshot(self.factory.make(direction=Direction.CENTER))
        self.assertIn("center", self.messages[-1].text)
        self.assertFalse(any("on your right" in t for t in self.texts()))

    def test_unconfirmed_delay_and_repeat_cooldown(self):
        snap = self.factory.make(visibility=Visibility.UNCONFIRMED, status=GuidanceStatus.UNCONFIRMED)
        self.feedback.on_snapshot(snap)
        self.assertEqual(len(self.messages), 1)
        self.clock.now += 5
        self.feedback.on_snapshot(snap)
        self.assertIn("can't see", self.messages[-1].text)
        for _ in range(20):
            self.feedback.on_snapshot(snap)
        self.assertEqual(len(self.messages), 2)
        self.assertFalse(any("lost sight" in text for text in self.texts()))

    def test_lost_and_uncertain_are_distinct_from_initial_absence(self):
        self.feedback.on_snapshot(self.factory.make())
        self.clock.now += 1
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCERTAIN, status=GuidanceStatus.UNCERTAIN))
        self.assertEqual(len(self.messages), 2)
        self.clock.now += 1
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCERTAIN, status=GuidanceStatus.UNCERTAIN))
        self.assertIn("can't confirm", self.messages[-1].text)
        self.clock.now += 7
        self.feedback.on_snapshot(self.factory.make(visibility=Visibility.UNCONFIRMED, status=GuidanceStatus.UNCONFIRMED))
        self.assertIn("lost sight", self.messages[-1].text)

    def test_ambiguity_is_non_directional_and_rate_limited(self):
        snap = self.factory.make(status=GuidanceStatus.AMBIGUOUS)
        self.feedback.on_snapshot(snap)
        self.assertIn("multiple possible", self.messages[-1].text)
        for _ in range(10):
            self.feedback.on_snapshot(snap)
        self.assertEqual(len(self.messages), 2)

    def test_repeat_rechecks_current_freshness(self):
        self.feedback.on_snapshot(self.factory.make())
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertIn("on your left", self.messages[-1].text)
        self.clock.now += 1
        self.feedback.on_event(InteractionEvent(InteractionEventKind.REPEAT))
        self.assertIn("don't have a current confirmed direction", self.messages[-1].text)

    def test_old_session_speech_invalidated_even_if_its_deadline_is_future(self):
        self.feedback.on_snapshot(self.factory.make())
        old = self.messages[-1]
        self.factory.session = self.factory.sessions.start("keys", 60)
        self.feedback.on_snapshot(self.factory.make(direction=Direction.RIGHT))
        self.assertFalse(self.validity.allows(old))
        self.assertTrue(self.validity.allows(self.messages[-1]))

    def test_expired_guidance_is_not_spoken(self):
        snap = self.factory.make()
        self.clock.now += 1
        self.feedback.on_snapshot(snap)
        self.assertFalse(any("on your left" in text for text in self.texts()))

    def test_terminal_feedback_waits_for_confirmed_cleanup(self):
        snap = self.factory.make()
        self.feedback.on_snapshot(snap)
        for outcome, text in [(SearchOutcome.COMPLETED, "Search completed"),
                              (SearchOutcome.CANCELLED, "Search cancelled"),
                              (SearchOutcome.TIMED_OUT, "located your phone during this search")]:
            with self.subTest(outcome=outcome):
                before = len(self.messages)
                stopped = replace(snap, state=ApplicationState.STOPPING, outcome=outcome,
                                  outcome_session_id=snap.session.session_id, cleanup_complete=False)
                self.feedback.on_snapshot(stopped)
                self.assertEqual(len(self.messages), before)
                self.feedback.on_snapshot(replace(stopped, state=ApplicationState.IDLE, cleanup_complete=True))
                self.assertIn(text, self.messages[-1].text)

    def test_cleanup_failure_does_not_announce_success(self):
        snap = self.factory.make()
        failure = ApplicationError("camera.blocked", RuntimeError("blocked"))
        self.feedback.on_snapshot(replace(snap, state=ApplicationState.FAILED, outcome=SearchOutcome.FAILED, errors=(failure,)))
        self.assertIn("camera", self.messages[-1].text)
        self.assertFalse(any("completed" in text for text in self.texts()))
        self.assertNotIn("RuntimeError", self.messages[-1].text)

    def test_command_capture_suppresses_unsolicited_speech(self):
        self.feedback.set_listening(True)
        self.feedback.on_snapshot(self.factory.make())
        self.assertEqual(self.messages, [])
        self.feedback.on_event(InteractionEvent(InteractionEventKind.ACKNOWLEDGE))
        self.assertEqual(len(self.messages), 1)
        self.assertIn("listening", self.messages[-1].text)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = SnapshotFactory(self.clock)
        self.snapshot = self.factory.make(visibility=Visibility.UNCONFIRMED, status=GuidanceStatus.UNCONFIRMED)
        self.feedback = Mock()
        self.coordinator = Mock()
        self.coordinator.start_search.return_value = self.snapshot
        self.coordinator.poll.return_value = self.snapshot
        self.coordinator.close.return_value = replace(self.snapshot, state=ApplicationState.CLOSED, cleanup_complete=True)
        self.loader = Mock(return_value=self.coordinator)
        self.manager = InteractionManager(self.loader, self.feedback)

    def command(self, text):
        self.manager.handle_command(parse_command(text))

    def test_help_unknown_and_idle_do_not_load_model(self):
        self.manager.poll()
        for text in ["help", "hello", "cancel", "repeat"]:
            self.command(text)
        self.loader.assert_not_called()

    def test_find_replacement_and_control_thread_routing(self):
        self.command("find my phone")
        self.command("find my keys")
        self.loader.assert_called_once_with()
        self.assertEqual(self.coordinator.start_search.call_args.args, ("keys", 60.0))
        self.assertEqual(self.manager.state, InteractionState.SEARCHING)

    def test_user_completion_calls_complete_not_cancel(self):
        self.command("find my phone")
        completed = replace(self.snapshot, state=ApplicationState.IDLE, cleanup_complete=True,
                            outcome=SearchOutcome.COMPLETED)
        self.coordinator.complete_search.return_value = completed
        self.command("I found it")
        self.coordinator.complete_search.assert_called_once_with(self.snapshot.session.session_id)
        self.coordinator.cancel_search.assert_not_called()

    def test_mismatching_confirmation_target_is_clarified(self):
        self.command("find my phone")
        self.command("I have found my wallet")
        self.coordinator.complete_search.assert_not_called()
        self.assertEqual(self.feedback.on_event.call_args.args[0].kind, InteractionEventKind.TARGET_MISMATCH)

    def test_cancel_and_followup_wake_while_searching(self):
        self.command("find my phone")
        self.manager.on_voice_event(VoiceEvent(VoiceEventKind.LISTENING))
        self.manager.poll()
        self.assertEqual(self.manager.state, InteractionState.LISTENING)
        self.coordinator.cancel_search.return_value = replace(self.snapshot, state=ApplicationState.IDLE, cleanup_complete=True)
        self.manager.on_voice_event(VoiceEvent(VoiceEventKind.TRANSCRIPT, "cancel search"))
        self.coordinator.cancel_search.assert_called_once_with(self.snapshot.session.session_id)

    def test_empty_and_timed_out_speech_do_not_start_search(self):
        self.manager.on_voice_event(VoiceEvent(VoiceEventKind.TRANSCRIPT, ""))
        self.manager.on_voice_event(VoiceEvent(VoiceEventKind.TIMEOUT))
        self.loader.assert_not_called()

    def test_factory_failure_has_no_successful_search_announcement(self):
        self.loader.side_effect = RuntimeError("checkpoint unavailable")
        with self.assertLogs("homesense.interaction.manager", level="ERROR"):
            self.command("find my phone")
        self.assertEqual(self.manager.state, InteractionState.FAILED)
        self.assertIsNotNone(self.manager.error)
        self.assertEqual(self.feedback.on_event.call_args.args[0].kind, InteractionEventKind.ERROR)
