"""Phase 2 analysis contracts; no microphones, recordings or model loading."""

import unittest

from scripts.speech_phase2_investigation import (
    contains_wake, evaluate, grammar_for, native_result, parsed, rejection_sweep, role, summarize,
)


def segment(text, confidence=1.0, *, final=True, offset=2):
    return {'text': text, 'final': final, 'offset_seconds': offset,
            'result': {'text': text, 'result': [{'word': word, 'conf': confidence} for word in text.split()]}}


def row(reference, transcript, confidence=1.0):
    segments = [segment(transcript, confidence)] if transcript else []
    return {'reference': reference, 'transcript': transcript, 'segments': segments,
            'decode_wall_seconds': .2, 'evaluation': evaluate(reference, transcript, segments)}


class Phase2Tests(unittest.TestCase):
    def test_profiles_are_explicit_and_separate_wake_from_command_words(self):
        self.assertEqual(len(grammar_for('mixed')), 100)
        self.assertEqual(grammar_for('wake'), ('hey agent', '[unk]'))
        self.assertEqual(len(grammar_for('commands')), 50)
        self.assertTrue(all(not phrase.startswith('hey agent') for phrase in grammar_for('commands')))
        self.assertNotIn('[unk]', grammar_for('mixed-no-unk'))
        self.assertEqual(grammar_for('mixed-nbest'), grammar_for('mixed'))

    def test_silence_word_is_not_falsely_reported_as_an_executed_command(self):
        result = row('', 'mobile', .33572)['evaluation']
        self.assertFalse(result['parsed']['accepted'])
        self.assertFalse(result['final_wake'])
        self.assertEqual(result['delivered_commands'], [])
        self.assertAlmostEqual(result['minimum_word_confidence'], .33572)

    def test_new_parser_supports_previously_missing_help_wording(self):
        result = row('Hey Agent, help me.', 'hey agent help me')['evaluation']
        self.assertTrue(result['exact'])
        self.assertTrue(result['final_wake'])
        self.assertTrue(result['intended_command_correct'])
        self.assertTrue(result['delivered_intended_command'])
        self.assertEqual(result['delivered_commands'], ['help me'])

    def test_partial_wake_that_disappears_is_counted_separately(self):
        segments = [segment('hey agent', final=False, offset=1), segment('agent find my phone')]
        result = evaluate('Hey Agent, find my phone.', 'agent find my phone', segments)
        self.assertTrue(result['partial_or_final_wake'])
        self.assertFalse(result['final_wake'])
        self.assertFalse(result['delivered_commands'])

    def test_exact_wake_requires_whole_words(self):
        for value in ('hey agents', 'hey Adrian', 'hey urgent', 'hi agent'):
            self.assertFalse(contains_wake(value))
        self.assertTrue(contains_wake('Hey Agent, help.'))

    def test_accepted_command_without_wake_is_not_dispatched(self):
        result = row('Cancel search.', 'cancel search')['evaluation']
        self.assertTrue(result['intended_command_correct'])
        self.assertFalse(result['delivered_commands'])

    def test_unknown_token_and_duplicate_full_transcript_remain_unmodified(self):
        self.assertFalse(parsed('locate my wallet [unk]')['accepted'])
        self.assertFalse(parsed('cancel search cancel search')['accepted'])

    def test_summary_uses_command_only_denominator_and_distinct_negative_groups(self):
        rows = [row('Find my keys.', 'find my keys'), row('Please close the window.', '[unk]'), row('', 'mobile', .3)]
        result = summarize(rows)
        self.assertEqual(result['commands'], 1)
        self.assertEqual(result['command_exact'], 1)
        self.assertEqual(result['unrelated_clips'], 1)
        self.assertEqual(result['silence_clips'], 1)
        self.assertEqual(result['silence_nonempty'], 1)
        self.assertEqual(result['negative_delivered_commands'], 0)

    def test_sweep_reports_in_sample_tradeoffs_without_accepting_unknowns(self):
        rows = [row('', 'mobile', .33572), row('Find my keys.', 'find my keys', .6),
                row('Please close the window.', '[unk]', 1)]
        result = rejection_sweep(rows)
        at_half = next(r for r in result if r['threshold'] == .5)
        at_nine = next(r for r in result if r['threshold'] == .9)
        self.assertEqual(at_half['confidence_only_nonempty_silence'], 0)
        self.assertEqual(at_half['strict_correct_commands'], 1)
        self.assertEqual(at_nine['strict_correct_commands'], 0)
        self.assertEqual(at_half['strict_negative_commands'], 0)

    def test_high_confidence_does_not_make_wrong_transcript_correct(self):
        result = row('Cancel search.', 'cancel search cancel search', 1)['evaluation']
        self.assertEqual(result['minimum_word_confidence'], 1)
        self.assertFalse(result['exact'])
        self.assertFalse(result['intended_command_correct'])

    def test_nbest_preserves_non_probability_score_and_has_no_fake_word_confidence(self):
        result = {'alternatives': [{'text': 'hey agent', 'confidence': 125.5, 'result': [{'word': 'hey'}]},
                                   {'text': 'agent', 'confidence': 123.4}]}
        record = native_result(result, 2, True)
        self.assertEqual(record['text'], 'hey agent')
        self.assertEqual(record['result']['alternatives'][0]['confidence'], 125.5)
        evaluation = evaluate('Hey Agent, help me.', 'hey agent', [record])
        self.assertIsNone(evaluation['minimum_word_confidence'])

    def test_new_unlabelled_references_do_not_silently_become_negative_controls(self):
        with self.assertRaises(ValueError):
            role('A new command that needs annotation')
