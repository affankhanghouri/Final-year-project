"""Isolated recognition readiness contracts; no audio device/model is needed."""

import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

from homesense.audio.events import VoiceEventKind
from scripts.voice_readiness_validation import (
    SequenceProbe, command_result, score_commands, score_wake, sequence_rows,
    starts_wake, text_cases, wake_normalize, energy_sweep, confidence_sweep,
)


def row(reference, transcript, *, partial=None, file='clip.wav', sha='verified'):
    segments = []
    if partial is not None:
        segments.append({'text': partial, 'final': False})
    segments.append({'text': transcript, 'final': True})
    return {'file': file, 'sha256': sha, 'reference': reference,
            'transcript': transcript, 'segments': segments}


class ReadinessTests(unittest.TestCase):
    def test_only_complete_heyagent_token_is_aliased(self):
        self.assertEqual(wake_normalize('HEYAGENT, find my phone'), 'hey agent find my phone')
        for text in ('heyagentphone', 'theyagent', 'hey agents', 'hey Adrian', 'hey urgent'):
            self.assertFalse(starts_wake(text))

    def test_all_text_contract_probes(self):
        for case in text_cases():
            with self.subTest(text=case['text']):
                # Historical probe labels retain the old report's parser gaps;
                # production now explicitly supports these two full phrases.
                added = {'hey agent help me': 'help',
                         'hey agent repeat the direction': 'repeat_guidance'}
                if case['text'] in added:
                    self.assertEqual(case['actions'], [(added[case['text']], None)])
                else:
                    self.assertTrue(case['passed'])

    def test_two_turn_wake_then_command_uses_real_parser(self):
        probe = SequenceProbe()
        events = probe.accept('heyagent', True, 0)
        self.assertTrue(any(e.kind is VoiceEventKind.ACKNOWLEDGE for e in events))
        probe.accept('find my keys', True, 1)
        self.assertEqual([(a['intent'], a['target']) for a in probe.actions], [('find_object', 'keys')])
        probe.accept('cancel search', True, 2)
        self.assertEqual(len(probe.actions), 1, 'Command window must close after one command')

    def test_partial_wake_and_partial_command_never_dispatch(self):
        probe = SequenceProbe()
        probe.accept('hey agent', False, 0)
        probe.accept('hey agent find my phone', False, 1)
        self.assertEqual(probe.actions, [])
        probe.accept('agent find my phone', True, 2)
        self.assertEqual(probe.actions, [], 'Final wake revision must revoke partial activation')

    def test_timeout_discards_command_at_exact_deadline(self):
        probe = SequenceProbe(timeout=2)
        probe.accept('hey agent', True, 0)
        probe.accept('find my keys', True, 2)
        self.assertEqual(probe.actions, [])

    def test_unknown_final_revokes_open_window(self):
        probe = SequenceProbe()
        probe.accept('hey agent', True, 0)
        probe.accept('find my keys [unk]', True, 1)
        probe.accept('find my keys', True, 2)
        self.assertEqual(probe.actions, [])

    def test_embedded_wake_rejected_even_in_open_command_window(self):
        probe = SequenceProbe()
        probe.accept('hey agent', True, 0)
        probe.accept('do not heyagent find my phone', True, 1)
        self.assertEqual(probe.actions, [])

    def test_dry_run_actions_require_new_wake_and_are_ordered(self):
        probe = SequenceProbe()
        for offset, text in enumerate(('hey agent find my phone', 'heyagent cancel search',
                                       'hey agent repeat direction', 'hey agent i found it', 'hey agent help')):
            probe.accept(text, True, offset)
        self.assertEqual([a['intent'] for a in probe.actions],
                         ['find_object', 'cancel_search', 'repeat_guidance', 'confirm_found', 'help'])

    def test_correct_asr_now_has_supported_repeat_and_help_intents(self):
        scores = score_commands([row('Repeat the direction.', 'repeat the direction')])
        self.assertEqual(scores['exact_transcripts'], 1)
        self.assertEqual(scores['correct_intents'], 1)
        self.assertEqual(scores['rejected_positive_clips'], 0)
        self.assertTrue(command_result('help me')['accepted'])

    def test_asr_error_can_still_be_a_correct_intent(self):
        scores = score_commands([row('Repeat the direction.', 'repeat direction')])
        self.assertEqual(scores['exact_transcripts'], 0)
        self.assertEqual(scores['correct_intents'], 1)

    def test_negative_and_positive_denominators_are_distinct(self):
        scores = score_commands([row('Find my keys.', 'find my keys'), row('', 'mobile'),
                                 row('Please close the window.', '[unk]')], bare_only=True)
        self.assertEqual(scores['positive_clips'], 1)
        self.assertEqual(scores['negative_clips'], 2)
        self.assertEqual(scores['false_command_accepts'], 0)
        self.assertEqual(scores['silence_nonempty'], 1)

    def test_partial_final_and_embedded_wakes_report_separately(self):
        scores = score_wake([row('Hey Agent, find my phone.', 'agent', partial='hey agent'),
                             row('Cancel search.', '[unk] hey agent')])
        self.assertEqual(scores['partial_or_final_prefix_true'], 1)
        self.assertEqual(scores['final_prefix_true'], 0)
        self.assertEqual(scores['final_anywhere_false'], 1)
        self.assertEqual(scores['final_prefix_false'], 0)

    def test_sequence_requires_wake_and_preserves_extra_unknowns(self):
        wake = row('Hey Agent, find my phone.', 'hey agent [unk]')
        command = row('Hey Agent, find my phone.', '[unk] find my phone')
        result = sequence_rows([wake], [command])[0]
        self.assertTrue(result['activated'])
        self.assertFalse(result['correct'])
        self.assertEqual(result['actions'], [])

    def test_sequence_refuses_mismatched_audio(self):
        with self.assertRaises(ValueError):
            sequence_rows([row('', '', sha='one')], [row('', '', sha='other')])

    def test_duplicate_or_partial_commands_never_silently_become_valid(self):
        for text in ('cancel search cancel search', 'find', 'find my', 'mobile', '[unk] help'):
            self.assertFalse(command_result(text)['accepted'])

    def test_energy_sweep_measures_tradeoff_without_changing_parser(self):
        clips = [SimpleNamespace(path=Path('quiet.wav'), pcm=array('h', [100] * 1600).tobytes()),
                 SimpleNamespace(path=Path('noisy.wav'), pcm=array('h', [1500] * 1600).tobytes())]
        rows = [row('Find my keys.', 'find my keys', file='quiet.wav'),
                row('', 'mobile', file='noisy.wav')]
        result = energy_sweep(rows, clips)
        at_runtime_energy = next(r for r in result if r['exploratory_rms_minimum'] == .015)
        self.assertEqual(at_runtime_energy['correct_bare_commands_retained'], 0)
        self.assertEqual(at_runtime_energy['nonempty_silence_retained'], 1)
        self.assertEqual(at_runtime_energy['false_commands_retained'], 0)

    def test_confidence_sweep_does_not_treat_alternative_scores_as_probabilities(self):
        command = row('Find my keys.', 'find my keys')
        command['evaluation'] = {'minimum_word_confidence': .6}
        alternative = row('', 'cancel search')
        alternative['evaluation'] = {'minimum_word_confidence': None}
        result = confidence_sweep([command, alternative])
        self.assertEqual(result[1]['correct_bare_commands_retained'], 1)
        self.assertEqual(result[2]['correct_bare_commands_retained'], 0)
        self.assertTrue(all(r['false_commands_retained'] == 0 for r in result))


if __name__ == '__main__':
    unittest.main()
