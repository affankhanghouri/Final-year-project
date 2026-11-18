import unittest
from homesense.commands.models import Intent, CommandProblem
from homesense.commands.parser import parse_command
from homesense.session import Target, SessionManager, SessionStatus
from homesense.config import AudioSettings, FeedbackSettings, RuntimeSettings


class CommandTests(unittest.TestCase):
    def test_find_aliases_and_natural_questions(self):
        for text, target in [("Find my phone.", Target.PHONE), ("Find my mobile", Target.PHONE),
                             ("Where is my phone?", Target.PHONE), ("locate my keys", Target.KEYS),
                             ("please find the key", Target.KEYS), ("find my purse", Target.WALLET),
                             ("locate my wallet please", Target.WALLET)]:
            with self.subTest(text=text):
                result = parse_command(text)
                self.assertEqual(result.intent, Intent.FIND_OBJECT)
                self.assertEqual(result.target, target)

    def test_control_intents(self):
        for phrases, intent in [(["cancel", "cancel search", "stop searching"], Intent.CANCEL_SEARCH),
                                (["repeat", "repeat direction", "say that again"], Intent.REPEAT_GUIDANCE),
                                (["I found it", "I've got it", "I have found my phone"], Intent.CONFIRM_FOUND),
                                (["help", "what can you do", "what objects can you find"], Intent.HELP)]:
            for phrase in phrases:
                with self.subTest(phrase=phrase):
                    self.assertEqual(parse_command(phrase).intent, intent)
        self.assertEqual(parse_command("I have found my phone").target, Target.PHONE)

    def test_ambiguous_unsupported_empty_and_negations(self):
        for text, problem in [("find my phone and keys", CommandProblem.AMBIGUOUS),
                              ("find phone or glasses", CommandProblem.AMBIGUOUS),
                              ("find my glasses", CommandProblem.UNSUPPORTED),
                              ("", CommandProblem.EMPTY), ("don't cancel", CommandProblem.UNRECOGNIZED),
                              ("cancel and find my keys", CommandProblem.UNRECOGNIZED),
                              ("I haven't found it", CommandProblem.UNRECOGNIZED)]:
            with self.subTest(text=text):
                result = parse_command(text)
                self.assertEqual(result.intent, Intent.UNKNOWN)
                self.assertEqual(result.problem, problem)

    def test_completion_is_not_cancellation_and_expires_old_results(self):
        manager = SessionManager(clock=lambda: 100)
        session = manager.start("phone", 10)
        from uuid import uuid4
        self.assertFalse(manager.complete(uuid4()))
        self.assertTrue(manager.complete(session.session_id))
        self.assertEqual(manager.current.status, SessionStatus.COMPLETED)
        self.assertFalse(manager.is_eligible(session.session_id))
        self.assertFalse(manager.cancel(session.session_id))
        self.assertFalse(manager.complete(session.session_id))

    def test_timeout_cannot_be_completed(self):
        now = [100]
        manager = SessionManager(clock=lambda: now[0])
        session = manager.start("phone", 1)
        now[0] = 101
        self.assertFalse(manager.complete(session.session_id))
        self.assertEqual(manager.current.status, SessionStatus.TIMED_OUT)

    def test_invalid_configuration(self):
        for cls, kwargs in [(AudioSettings, {"sample_rate": 0}), (AudioSettings, {"volume": float("nan")}),
                            (AudioSettings, {"command_timeout": 0}), (AudioSettings, {"wake_phrase": "hey jarvis"}),
                            (FeedbackSettings, {"direction_cooldown": -1}), (RuntimeSettings, {"search_timeout": True})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cls(**kwargs)
