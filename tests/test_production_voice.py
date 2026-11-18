"""Production command understanding, native endpoints, bounded fallback and IDs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from homesense.audio.command_gate import CommandGate
from homesense.audio.events import VoiceEventKind
from homesense.audio.speech_recognition import VoskCommandRecognizer, resolve_command
from homesense.audio.input import SoundDeviceInput
from homesense.commands.models import Intent, CommandProblem
from homesense.commands.parser import command_grammar, parse_command
from homesense.config import RuntimeSettings, AudioSettings
from homesense.runtime import compose


class CommandUnderstandingTests(unittest.TestCase):
    def test_required_phrases_with_optional_prefix_and_politeness(self):
        cases = [('find my phone', 'find_object', 'phone'), ('find my mobile', 'find_object', 'phone'),
                 ('find my keys', 'find_object', 'keys'), ('find my key', 'find_object', 'keys'),
                 ('find my wallet', 'find_object', 'wallet'), ('locate my wallet', 'find_object', 'wallet'),
                 ('can you find my phone', 'find_object', 'phone'), ('please find my keys', 'find_object', 'keys'),
                 ('could you please find the wallet', 'find_object', 'wallet'),
                 ('cancel search', 'cancel_search', None), ('stop searching', 'cancel_search', None),
                 ('repeat the direction', 'repeat_guidance', None), ('repeat direction', 'repeat_guidance', None),
                 ('say that again', 'repeat_guidance', None), ('help me', 'help', None),
                 ('what can you find', 'help', None), ('i found it', 'confirm_found', None)]
        for prefix in ('', 'Hey Agent, ', 'HEYAGENT! '):
            for text, intent, target in cases:
                with self.subTest(text=prefix + text):
                    result = parse_command(prefix + text + '.')
                    self.assertEqual(result.intent.value, intent)
                    self.assertEqual(result.target.value if result.target else None, target)

    def test_negations_unknown_fragments_and_embedded_commands_are_rejected(self):
        phrases = ('do not find my phone', 'hey agent do not find my phone', "don't cancel",
                   'hey agent i have not found it', 'never find my keys', 'hey agent [unk] find my phone',
                   'find my', 'find', 'mobile', 'please close the window', 'cancel search cancel search',
                   'tell me to find my phone', 'say hey agent find my phone', 'find my phone maybe',
                   'hey agent hey agent find my phone', 'help me [unk]', 'can you not find my phone')
        for text in phrases:
            with self.subTest(text=text):
                self.assertIs(parse_command(text).intent, Intent.UNKNOWN)

    def test_unsupported_and_ambiguous_are_distinguishable_from_unrelated(self):
        for phrase in ('find my clothes', 'hey agent please find my glasses', 'can you find the shoes'):
            self.assertIs(parse_command(phrase).problem, CommandProblem.UNSUPPORTED)
        for phrase in ('find my phone and wallet', 'can you find my keys or clothes'):
            self.assertIs(parse_command(phrase).problem, CommandProblem.AMBIGUOUS)
        self.assertIs(parse_command('please close the window').problem, CommandProblem.UNRECOGNIZED)

    def test_grammar_has_unknown_natural_phrases_and_rejection_contexts(self):
        grammar = command_grammar()
        for phrase in ('[unk]', 'find my keys', 'hey agent please find my wallet', 'can you find my phone',
                       'repeat the direction', 'help me', 'what can you find', 'find my [unk]',
                       'find my clothes', 'do not find my phone'):
            self.assertIn(phrase, grammar)
        self.assertEqual(len(grammar), len(set(grammar)))

    def test_fallback_cannot_create_a_supported_command(self):
        for primary in ('[unk]', 'find my [unk]', 'mobile', 'please close the window'):
            self.assertEqual(resolve_command(primary, 'find my phone'), primary)

    def test_fallback_identifies_unsupported_request_without_substituting_target(self):
        for primary in ('find my [unk]', 'find my phone'):
            self.assertIs(parse_command(resolve_command(primary, 'find my clothes')).problem, CommandProblem.UNSUPPORTED)

    def test_fallback_vetoes_negation_or_a_conflicting_supported_intent(self):
        for primary, fallback in (('cancel search', 'do not cancel'), ('find my phone', 'find my keys'),
                                  ('i found it', 'i have not found it'), ('find my phone', 'cancel search')):
            self.assertIs(parse_command(resolve_command(primary, fallback)).intent, Intent.UNKNOWN)


class CommandDeliveryTests(unittest.TestCase):
    def test_plain_final_is_delivered_without_wake(self):
        gate = CommandGate()
        events = gate.accept('find my phone', True, 0, 1)
        self.assertEqual([e.text for e in events if e.kind is VoiceEventKind.TRANSCRIPT], ['find my phone'])

    def test_partials_do_not_execute_and_one_id_cannot_deliver_twice(self):
        gate = CommandGate()
        self.assertFalse(any(e.kind is VoiceEventKind.TRANSCRIPT for e in gate.accept('find my phone', False, 0)))
        self.assertTrue(any(e.kind is VoiceEventKind.TRANSCRIPT for e in gate.accept('find my phone', True, 1, 7)))
        self.assertEqual(gate.accept('find my phone', True, 2, 7), [])
        self.assertTrue(any(e.kind is VoiceEventKind.TRANSCRIPT for e in gate.accept('find my phone', True, 3, 8)))

    def test_late_final_and_duplicate_of_late_final_are_discarded(self):
        gate = CommandGate(timeout=1)
        gate.accept('find', False, 0)
        self.assertEqual(gate.accept('find my phone', True, 1, 5)[0].kind, VoiceEventKind.TIMEOUT)
        self.assertEqual(gate.accept('find my phone', True, 2, 5), [])

    def test_standalone_prefix_acknowledges_but_does_not_activate_resources(self):
        events = CommandGate().accept('hey agent', True, 0)
        self.assertTrue(any(e.kind is VoiceEventKind.ACKNOWLEDGE for e in events))
        self.assertFalse(any(e.kind is VoiceEventKind.TRANSCRIPT for e in events))

    def test_composition_uses_production_recognizer_once_and_never_loads_idle_perception(self):
        factory = Mock()
        with patch('homesense.runtime.perception_factory', return_value=factory), \
             patch('homesense.audio.speech_recognition.VoskCommandRecognizer') as recognizer, \
             patch('homesense.runtime.SoundDeviceInput') as source:
            runtime = compose(RuntimeSettings(), console_output=True)
            recognizer.assert_called_once()
            self.assertIs(runtime.voice.recognizer, recognizer.return_value)
            source.return_value.open.assert_not_called()
            runtime.interaction.poll()
            factory.assert_not_called()


class NativeCommandRecognizerTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        (self.path / 'am').mkdir()
        (self.path / 'am/final.mdl').touch()
        self.primary, self.fallback = Mock(), Mock()
        self.primary.AcceptWaveform.return_value = False
        self.primary.PartialResult.return_value = json.dumps({'partial': 'find my phone'})
        self.primary.Result.return_value = json.dumps({'alternatives': [{'text': 'find my phone'}]})
        self.primary.FinalResult.return_value = self.primary.Result.return_value
        self.fallback.AcceptWaveform.return_value = False
        self.fallback.FinalResult.return_value = json.dumps({'text': ''})
        self.module = SimpleNamespace(Model=Mock(), KaldiRecognizer=Mock(side_effect=[self.primary, self.fallback]))
        self.patcher = patch.dict('sys.modules', {'vosk': self.module})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def make(self, **kwargs):
        return VoskCommandRecognizer(self.path, **kwargs)

    def test_two_decoders_share_one_model_and_reset_without_reloading(self):
        recognizer = self.make()
        recognizer.reset()
        recognizer.reset()
        self.module.Model.assert_called_once_with(str(self.path))
        self.assertEqual(self.module.KaldiRecognizer.call_count, 2)
        calls = self.module.KaldiRecognizer.call_args_list
        self.assertIs(calls[0].args[0], calls[1].args[0])
        self.assertIn('[unk]', json.loads(calls[0].args[2]))
        self.primary.SetMaxAlternatives.assert_called_once_with(3)

    def test_quiet_pcm_never_uses_rms_to_force_a_final(self):
        recognizer = self.make()
        for _ in range(12):
            self.assertFalse(recognizer.feed(b'\x64\x00' * 1600).final)
        self.primary.FinalResult.assert_not_called()
        self.fallback.AcceptWaveform.assert_not_called()

    def test_idle_pcm_uses_bounded_preroll_without_timeout_commands(self):
        self.primary.PartialResult.return_value = '{"partial":""}'
        recognizer = self.make()
        for _ in range(150):
            self.assertFalse(recognizer.feed(b'\0' * 3200).final)
        self.assertLessEqual(len(recognizer._audio), 32000)

    def test_native_final_fallback_can_report_unsupported_target(self):
        recognizer = self.make()
        self.primary.AcceptWaveform.return_value = True
        self.fallback.FinalResult.return_value = '{"text":"find my clothes"}'
        update = recognizer.feed(b'\0' * 3200)
        self.assertTrue(update.final)
        self.assertEqual(update.utterance_id, 1)
        self.assertIs(parse_command(update.text).problem, CommandProblem.UNSUPPORTED)
        self.assertEqual(recognizer._audio, bytearray())

    def test_finite_file_flush_reads_top_alternative_and_has_unique_ids(self):
        recognizer = self.make()
        recognizer.feed(b'\0' * 3200)
        first = recognizer.finish()
        self.assertEqual(first.text, 'find my phone')
        second = recognizer.finish()
        self.assertNotEqual(first.utterance_id, second.utterance_id)

    def test_timeout_discards_suffix_until_native_endpoint(self):
        recognizer = self.make(max_seconds=.2)
        recognizer.feed(b'\0' * 3200)
        recognizer.feed(b'\0' * 3200)
        self.assertEqual(recognizer.feed(b'\0' * 3200).text, '')
        self.primary.AcceptWaveform.return_value = True
        self.assertFalse(recognizer.feed(b'\0' * 3200).final)
        update = recognizer.feed(b'\0' * 3200)
        self.assertEqual(update.text, 'find my phone')

    def test_invalid_pcm_and_native_errors_are_explicit(self):
        recognizer = self.make()
        with self.assertRaises(ValueError):
            recognizer.feed(b'\0')
        self.primary.AcceptWaveform.side_effect = RuntimeError('decoder failed')
        with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
            recognizer.feed(b'\0' * 3200)

    def test_fallback_failure_does_not_silently_execute_primary(self):
        recognizer = self.make()
        self.primary.AcceptWaveform.return_value = True
        self.fallback.AcceptWaveform.side_effect = RuntimeError('fallback failed')
        with self.assertRaisesRegex(RuntimeError, 'fallback failed'):
            recognizer.feed(b'\0' * 3200)


class MicrophoneFailureTests(unittest.TestCase):
    def test_failed_stream_start_releases_the_stream(self):
        stream = Mock()
        stream.start.side_effect = RuntimeError('device unavailable')
        module = SimpleNamespace(RawInputStream=Mock(return_value=stream))
        with patch.dict('sys.modules', {'sounddevice': module}):
            source = SoundDeviceInput(AudioSettings())
            with self.assertRaisesRegex(RuntimeError, 'device unavailable'):
                source.open()
        stream.close.assert_called_once()

    def test_disconnected_stream_fails_and_close_attempts_release_after_stop_failure(self):
        source = SoundDeviceInput(AudioSettings())
        stream = Mock(active=False)
        source._stream = stream
        with self.assertRaisesRegex(RuntimeError, 'ended unexpectedly'):
            source.read(0)
        stream.stop.side_effect = RuntimeError('stop failed')
        with self.assertRaisesRegex(RuntimeError, 'stop failed'):
            source.close()
        stream.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
