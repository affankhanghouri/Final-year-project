"""Offline proof-of-concept metrics and PCM contracts, no optional dependencies."""

from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from scripts.openwakeword_poc import activation_summary, chunks, load_pcm, replay, summarize


class WakePocTests(unittest.TestCase):
    def test_eof_tail_is_preserved_and_only_last_block_is_padded(self):
        pcm = np.arange(1600, dtype=np.int16)
        blocks = list(chunks(pcm))
        self.assertEqual(len(blocks), 2)
        np.testing.assert_array_equal(blocks[0][0], pcm[:1280])
        np.testing.assert_array_equal(blocks[1][0][:320], pcm[1280:])
        self.assertTrue(np.all(blocks[1][0][320:] == 0))
        self.assertEqual(blocks[1][1], .1)
        self.assertEqual(pcm.shape, (1600,))

    def test_long_score_plateau_is_one_episode_and_second_rise_is_distinct(self):
        result = activation_summary([(0.08, .1), (.16, .5), (.24, .9), (.32, .2), (.4, .8)])
        self.assertTrue(result['activated'])
        self.assertEqual(result['activation_offsets'], [.16, .4])
        self.assertEqual(result['activation_episodes'], 2)

    def test_subthreshold_and_empty_clips_do_not_activate(self):
        self.assertFalse(activation_summary([])['activated'])
        self.assertFalse(activation_summary([(.08, .4999)])['activated'])

    def test_invalid_model_scores_fail_explicitly(self):
        for value in (float('nan'), float('inf'), -1, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                activation_summary([(.08, value)])

    def test_misses_false_positive_clips_and_episodes_have_distinct_denominators(self):
        def row(expected, values):
            return dict(activation_summary(list(enumerate(values))), expected_wake=expected, audio_seconds=5)
        summary = summarize([row(True, [.7]), row(True, [.1]), row(False, [.6, .1, .8]), row(False, [.2])])
        self.assertEqual(summary['positive_clips'], 2)
        self.assertEqual(summary['detections'], 1)
        self.assertEqual(summary['misses'], 1)
        self.assertEqual(summary['negative_clips'], 2)
        self.assertEqual(summary['false_positive_clips'], 1)
        self.assertEqual(summary['false_activation_episodes'], 2)
        self.assertEqual(summary['negative_audio_seconds'], 10)

    def test_model_is_reset_between_recordings(self):
        class FakeModel:
            resets = 0
            def reset(self):
                self.resets += 1
            def predict(self, block):
                return {'hey_agent': .1}
        model = FakeModel()
        for _ in range(2):
            result = replay(model, np.zeros(1600, dtype=np.int16))
            self.assertFalse(result['activated'])
            self.assertEqual(len(result['scores']), 2)
        self.assertEqual(model.resets, 2)

    def test_wrong_sample_rate_is_rejected_without_implicit_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wrong.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, 44100, 0, 'NONE', 'uncompressed'))
                wav.writeframes(b'\x00\x00' * 100)
            with self.assertRaises(ValueError):
                load_pcm(path)


if __name__ == '__main__':
    unittest.main()
