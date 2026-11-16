from dataclasses import replace
import unittest

import numpy as np

from homesense.camera import Frame
from homesense.detector import Detection
from homesense.perception import DetectionBatch
from homesense.session import SessionManager
from homesense.stability import StabilityProcessor, StabilitySettings, Visibility


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sessions = SessionManager(clock=self.clock)
        self.session = self.sessions.start("phone", 20)
        self.processor = StabilityProcessor(self.sessions, clock=self.clock)
        self.frame_id = 0

    def batch(self, captured, *, confidence=0.9, target="phone", completed=None):
        self.frame_id += 1
        detections = () if confidence is None else (Detection(0, target, confidence, (0, 0, 1, 1)),)
        return DetectionBatch(
            self.session.session_id,
            Frame(self.frame_id, captured, np.zeros((2, 2, 3), dtype=np.uint8)),
            captured if completed is None else completed, 0, detections,
        )

    def observe(self, captured, **kwargs):
        self.clock.now = captured
        self.assertTrue(self.processor.observe(self.batch(captured, **kwargs)))
        return self.processor.snapshot().state

    def confirm(self):
        self.observe(0)
        self.observe(0.25)
        self.assertEqual(self.observe(0.5), Visibility.VISIBLE)

    def test_initial_confirmation_requires_count_and_elapsed_time(self):
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0.05), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0.1), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0.5), Visibility.VISIBLE)
        self.assertTrue(self.sessions.is_eligible(self.session.session_id))

    def test_isolated_false_positive_never_confirms(self):
        self.observe(0)
        self.clock.now = 0.8
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)
        self.assertIsNone(self.processor.snapshot().last_positive_at)
        self.assertEqual(self.observe(1), Visibility.UNCONFIRMED)

    def test_elapsed_time_alone_does_not_replace_positive_count(self):
        self.observe(0)
        self.assertEqual(self.observe(0.5), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0.625), Visibility.VISIBLE)

    def test_exact_silence_boundaries_and_positive_gap(self):
        self.processor = StabilityProcessor(self.sessions, clock=self.clock, settings=StabilitySettings(
            confirmation_span=0.5, max_positive_gap=0.5,
            uncertain_after=0.25, forget_after=0.75,
        ))
        self.confirm()
        self.clock.now = 0.75
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCERTAIN)
        self.clock.now = 1.25
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)
        self.observe(1.5)
        self.observe(1.75)
        # Exactly max_positive_gap restarts evidence instead of confirming it.
        self.assertEqual(self.observe(2.25), Visibility.UNCONFIRMED)

    def test_two_widely_spaced_positives_do_not_confirm(self):
        self.observe(0)
        self.assertEqual(self.observe(0.7), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(0.9), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(1.125), Visibility.VISIBLE)

    def test_brief_miss_is_uncertain_and_support_restores_visibility(self):
        self.confirm()
        self.assertEqual(self.observe(0.625, confidence=None), Visibility.UNCERTAIN)
        self.assertEqual(self.observe(0.75), Visibility.VISIBLE)

    def test_sustained_disappearance_forgets_and_requires_reconfirmation(self):
        self.confirm()
        self.observe(0.625, confidence=None)
        self.observe(1, confidence=None)
        self.assertEqual(self.observe(1.5, confidence=None), Visibility.UNCONFIRMED)
        self.assertEqual(self.observe(1.625), Visibility.UNCONFIRMED)

    def test_silence_ages_visibility_without_an_explicit_negative(self):
        self.confirm()
        original = self.processor.snapshot()
        self.clock.now = 0.875
        self.assertEqual(self.processor.snapshot().state, Visibility.VISIBLE)
        self.clock.now = 1
        uncertain = self.processor.snapshot()
        self.assertEqual(uncertain.state, Visibility.UNCERTAIN)
        self.assertEqual(uncertain.last_observation_at, original.last_observation_at)
        self.clock.now = 5
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)

    def test_visible_snapshot_carries_support_expiry_for_downstream_consumers(self):
        self.confirm()
        saved = self.processor.snapshot()
        self.assertAlmostEqual(saved.valid_until, 0.9)
        self.clock.now = 1
        current = self.processor.snapshot()
        self.assertEqual(current.state, Visibility.UNCERTAIN)
        self.assertIsNone(current.valid_until)
        self.assertEqual(saved.state, Visibility.VISIBLE)  # Old copies need their deadline checked.

    def test_variable_intervals_and_dropped_frame_ids_can_confirm(self):
        self.observe(0)
        self.frame_id = 25
        self.observe(0.125)
        self.frame_id = 80
        self.assertEqual(self.observe(0.5), Visibility.VISIBLE)

    def test_duplicate_and_out_of_order_results_do_not_add_evidence(self):
        first = self.batch(0)
        self.assertTrue(self.processor.observe(first))
        for _ in range(3):
            self.assertFalse(self.processor.observe(first))
        self.clock.now = 0.25
        second = self.batch(0.25)
        self.assertTrue(self.processor.observe(second))
        self.assertFalse(self.processor.observe(first))
        same_time = self.batch(0.25)
        self.assertFalse(self.processor.observe(same_time))
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)

    def test_regressing_completion_time_rejected(self):
        self.clock.now = 0.25
        self.assertTrue(self.processor.observe(self.batch(0, completed=0.25)))
        self.clock.now = 0.375
        self.assertFalse(self.processor.observe(self.batch(0.125, completed=0.2)))

    def test_stale_positive_and_negative_rejected_before_evidence_update(self):
        self.confirm()
        self.clock.now = 0.75
        for confidence in (None, 0.99):
            stale = self.batch(0.25, confidence=confidence)
            self.assertFalse(self.processor.observe(stale))
        snapshot = self.processor.snapshot()
        self.assertEqual(snapshot.state, Visibility.VISIBLE)
        self.assertEqual(snapshot.source_frame_id, 3)
        self.assertEqual(snapshot.last_positive_at, 0.5)

    def test_new_but_stale_result_cannot_confirm(self):
        self.observe(0)
        self.observe(0.25)
        self.clock.now = 1
        self.assertFalse(self.processor.observe(self.batch(0.5)))
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)

    def test_wrong_target_and_low_confidence_are_not_support(self):
        self.observe(0)
        self.observe(0.25, target="keys")
        self.assertEqual(self.observe(0.5), Visibility.UNCONFIRMED)
        self.observe(0.625, confidence=0.49)
        self.assertEqual(self.observe(0.75), Visibility.UNCONFIRMED)

    def test_multiple_boxes_in_one_frame_are_only_one_observation(self):
        batch = self.batch(0)
        batch = replace(batch, detections=batch.detections * 10)
        self.assertTrue(self.processor.observe(batch))
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)
        # Visibility snapshots expose no remembered localization.
        self.assertFalse(hasattr(self.processor.snapshot(), "bbox"))
        self.assertFalse(hasattr(self.processor.snapshot(), "detections"))

    def test_replacement_resets_and_rejects_old_session(self):
        self.confirm()
        old = self.batch(0.625)
        self.clock.now = 0.625
        self.session = self.sessions.start("phone", 20)
        self.assertFalse(self.processor.observe(old))
        self.assertEqual(self.processor.snapshot().state, Visibility.UNCONFIRMED)
        self.assertEqual(self.processor.snapshot().session_id, self.session.session_id)
        self.frame_id = 0
        self.assertEqual(self.observe(0.75), Visibility.UNCONFIRMED)

    def test_cancel_and_timeout_reset_on_access(self):
        self.confirm()
        self.sessions.cancel(self.session.session_id)
        snapshot = self.processor.snapshot()
        self.assertIsNone(snapshot.session_id)
        self.assertEqual(snapshot.state, Visibility.UNCONFIRMED)
        self.assertFalse(self.processor.observe(self.batch(0.5)))
        self.session = self.sessions.start("keys", 0.25)
        self.assertEqual(self.processor.snapshot().target.value, "keys")
        self.clock.now = self.session.deadline
        self.assertIsNone(self.processor.snapshot().session_id)

    def test_invalid_timestamp_metadata_rejected(self):
        self.clock.now = 0.25
        batch = self.batch(0.125)
        for bad in (replace(batch, completed_at=0.5),
                    replace(batch, completed_at=0),
                    replace(batch, completed_at=float("nan")),
                    replace(batch, frame=replace(batch.frame, captured_at=-0.1))):
            self.assertFalse(self.processor.observe(bad))
        self.assertIsNone(self.processor.snapshot().source_frame_id)

    def test_invalid_settings(self):
        for kwargs in ({"confirmation_span": 0}, {"max_positive_gap": float("nan")},
                       {"min_positive_observations": 1}, {"min_positive_observations": True},
                       {"uncertain_after": 1}, {"max_positive_gap": 2},
                       {"min_confidence": -1}, {"min_confidence": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    StabilitySettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
