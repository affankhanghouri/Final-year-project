"""Experimental wake sequence -> real parser/InteractionManager, fake resources."""

import unittest
from unittest.mock import Mock

from homesense.application import ApplicationSnapshot, ApplicationState
from homesense.interaction.manager import InteractionManager
from homesense.interaction.state import InteractionEventKind
from homesense.session import SessionManager
from scripts.voice_readiness_validation import SequenceProbe


class VoiceReadinessIntegrationTests(unittest.TestCase):
    def setUp(self):
        session = SessionManager(clock=lambda: 0).start('phone', 60)
        snapshot = ApplicationSnapshot(ApplicationState.SEARCHING, session, None, None, (), False)
        self.coordinator = Mock()
        self.coordinator.start_search.return_value = snapshot
        self.coordinator.poll.return_value = snapshot
        self.coordinator.cancel_search.return_value = snapshot
        self.coordinator.complete_search.return_value = snapshot
        self.factory = Mock(return_value=self.coordinator)
        self.feedback = Mock()
        self.manager = InteractionManager(self.factory, self.feedback)
        self.probe = SequenceProbe()
        self.session = session

    def feed(self, text, *, final=True, now=0):
        for event in self.probe.accept(text, final, now):
            self.manager.on_voice_event(event)

    def test_two_turn_find_then_woken_cancel_dispatches_to_fake_coordinator(self):
        self.feed('heyagent')
        self.factory.assert_not_called()
        self.feed('find my phone', now=1)
        self.coordinator.start_search.assert_called_once_with('phone', 60.0)
        self.feed('cancel search', now=2)
        self.coordinator.cancel_search.assert_not_called()
        self.feed('hey agent cancel search', now=3)
        self.coordinator.cancel_search.assert_called_once_with(self.session.session_id)
        self.coordinator.complete_search.assert_not_called()

    def test_confirmation_is_completion_and_repeat_is_feedback_only(self):
        self.feed('hey agent find my phone')
        self.feed('hey agent repeat direction', now=1)
        self.assertTrue(any(call.args[0].kind is InteractionEventKind.REPEAT
                            for call in self.feedback.on_event.call_args_list))
        self.feed('hey agent i found it', now=2)
        self.coordinator.complete_search.assert_called_once_with(self.session.session_id)
        self.coordinator.cancel_search.assert_not_called()

    def test_final_unknown_releases_listening_without_loading_resources(self):
        self.feed('hey agent', final=False)
        self.assertTrue(self.manager.listening)
        self.feed('hey agent [unk] find my phone', now=1)
        self.assertFalse(self.manager.listening)
        self.factory.assert_not_called()

    def test_new_help_wording_is_dispatched_as_help_without_camera(self):
        self.feed('hey agent help me')
        self.factory.assert_not_called()
        self.assertEqual(self.feedback.on_event.call_args.args[0].kind, InteractionEventKind.HELP)


if __name__ == '__main__':
    unittest.main()
