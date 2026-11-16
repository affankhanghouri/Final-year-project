from dataclasses import replace
import threading
import unittest

import numpy as np

from homesense.camera import Frame
from homesense.detector import Detection
from homesense.guidance import Direction, GuidanceProcessor, GuidanceSettings, GuidanceStatus
from homesense.perception import DetectionBatch
from homesense.session import SessionManager
from homesense.stability import StabilityProcessor, Visibility


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sessions = SessionManager(clock=self.clock)
        self.session = self.sessions.start("phone", 10)
        self.stability = StabilityProcessor(self.sessions, clock=self.clock)
        self.guidance = GuidanceProcessor(self.sessions, clock=self.clock)
        self.frame_id = 0
        for captured in (0, 0.25, 0.5):
            self.sample(0.5, captured)

    def sample(self, normalized_x, captured, *, width=300, height=100):
        self.clock.now = captured
        self.frame_id += 1
        center = normalized_x * width
        self.batch = DetectionBatch(
            self.session.session_id,
            Frame(self.frame_id, captured, np.zeros((height, width, 3), dtype=np.uint8)),
            captured, 0,
            (Detection(0, "phone", 0.9, (center - 3, 10, center + 3, 30)),),
        )
        self.assertTrue(self.stability.observe(self.batch))
        self.visibility = self.stability.snapshot()

    def publish(self, batch=None, visibility=None):
        self.guidance.update(self.batch if batch is None else batch,
                             self.visibility if visibility is None else visibility)
        return self.guidance.latest()

    def test_left_center_right_and_exact_initial_boundaries(self):
        for x, expected in ((0.1, Direction.LEFT), (1 / 3, Direction.CENTER),
                            (0.5, Direction.CENTER), (2 / 3, Direction.RIGHT), (0.9, Direction.RIGHT)):
            with self.subTest(x=x):
                self.guidance = GuidanceProcessor(self.sessions, clock=self.clock)
                center = x * 300
                detection = replace(self.batch.detections[0], bbox=(center - 3, 10, center + 3, 30))
                result = self.publish(replace(self.batch, detections=(detection,)))
                self.assertTrue(result.actionable)
                self.assertEqual(result.direction, expected)
                self.assertAlmostEqual(result.normalized_x, x)

    def test_boxes_at_image_edges_and_full_image(self):
        for box, expected in (((0, 0, 10, 100), Direction.LEFT),
                              ((290, 0, 300, 100), Direction.RIGHT),
                              ((0, 0, 300, 100), Direction.CENTER)):
            with self.subTest(box=box):
                self.guidance = GuidanceProcessor(self.sessions, clock=self.clock)
                detection = replace(self.batch.detections[0], bbox=box)
                self.assertEqual(self.publish(replace(self.batch, detections=(detection,))).direction, expected)

    def test_original_image_width_and_no_mirroring(self):
        for width in (300, 1200):
            with self.subTest(width=width):
                self.guidance = GuidanceProcessor(self.sessions, clock=self.clock)
                frame = replace(self.batch.frame, image=np.zeros((200, width, 3), dtype=np.uint8))
                detection = replace(self.batch.detections[0], bbox=(10, 10, 30, 50))
                result = self.publish(replace(self.batch, frame=frame, detections=(detection,)))
                self.assertEqual(result.direction, Direction.LEFT)
                self.assertAlmostEqual(result.normalized_x, 20 / width)

    def test_unconfirmed_and_uncertain_are_explicit_nonactionable_results(self):
        for state, expected in ((Visibility.UNCONFIRMED, GuidanceStatus.UNCONFIRMED),
                                (Visibility.UNCERTAIN, GuidanceStatus.UNCERTAIN)):
            with self.subTest(state=state):
                result = self.publish(visibility=replace(self.visibility, state=state, valid_until=None))
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.direction)

    def test_stale_position_boundary_and_future_timestamp(self):
        self.guidance = GuidanceProcessor(self.sessions, clock=self.clock,
                                         settings=GuidanceSettings(max_position_age=0.125))
        self.clock.now = 0.625
        self.assertEqual(self.publish().status, GuidanceStatus.STALE)
        self.clock.now = 0.25
        self.assertEqual(self.publish().status, GuidanceStatus.STALE)

    def test_saved_visible_snapshot_cannot_outlive_support(self):
        self.assertAlmostEqual(self.visibility.valid_until, 0.9)
        self.clock.now = 1
        self.assertEqual(self.publish().status, GuidanceStatus.STALE)

    def test_retrieval_rechecks_age_and_session_without_another_update(self):
        self.assertTrue(self.publish().actionable)
        self.clock.now = 1
        self.assertEqual(self.guidance.latest().status, GuidanceStatus.STALE)
        self.sessions.cancel(self.session.session_id)
        self.assertEqual(self.guidance.latest().status, GuidanceStatus.INACTIVE)

    def test_cancel_replacement_and_timeout_before_publication(self):
        for action in ("cancel", "replace", "timeout"):
            with self.subTest(action=action):
                self.setUp()
                if action == "cancel":
                    self.sessions.cancel(self.session.session_id)
                elif action == "replace":
                    self.sessions.start("keys", 10)
                else:
                    self.clock.now = self.session.deadline
                result = self.publish()
                self.assertEqual(result.status, GuidanceStatus.INACTIVE)
                self.assertFalse(result.actionable)

    def test_missing_wrong_class_and_weak_detection(self):
        for detections, expected in (
            ((), GuidanceStatus.NO_DETECTION),
            ((replace(self.batch.detections[0], class_name="keys"),), GuidanceStatus.NO_DETECTION),
            ((replace(self.batch.detections[0], confidence=0.4),), GuidanceStatus.WEAK_DETECTION),
        ):
            with self.subTest(expected=expected):
                result = self.publish(replace(self.batch, detections=detections))
                self.assertEqual(result.status, expected)
                self.assertFalse(result.actionable)
        self.guidance.update(None, self.visibility)
        self.assertEqual(self.guidance.latest().status, GuidanceStatus.NO_DETECTION)

    def test_multiple_target_instances_are_ambiguous_even_if_one_is_weak(self):
        other = replace(self.batch.detections[0], confidence=0.3, bbox=(200, 0, 230, 50))
        result = self.publish(replace(self.batch, detections=self.batch.detections + (other,)))
        self.assertEqual(result.status, GuidanceStatus.AMBIGUOUS)
        self.assertIsNone(result.direction)
        self.assertIsNone(result.normalized_x)

    def test_invalid_bounding_boxes_and_dimensions(self):
        for box in ((1, 2, 3), (-1, 0, 10, 30), (10, 10, 10, 30), (20, 10, 10, 30),
                    (0, 0, 301, 50), (0, 0, 30, 101), (float("nan"), 0, 30, 50),
                    (0, 0, float("inf"), 50), ("bad", 0, 30, 50)):
            with self.subTest(box=box):
                detection = replace(self.batch.detections[0], bbox=box)
                result = self.publish(replace(self.batch, detections=(detection,)))
                self.assertEqual(result.status, GuidanceStatus.INVALID_INPUT)
                self.assertFalse(result.actionable)
        for shape in ((0, 300, 3), (100, 0, 3), (10,)):
            with self.subTest(shape=shape):
                frame = replace(self.batch.frame, image=np.zeros(shape, dtype=np.uint8))
                self.assertEqual(self.publish(replace(self.batch, frame=frame)).status, GuidanceStatus.INVALID_INPUT)

    def test_visibility_must_describe_the_same_positive_observation(self):
        for snapshot in (replace(self.visibility, source_frame_id=1),
                         replace(self.visibility, last_observation_at=0.25),
                         replace(self.visibility, last_positive_at=0.25),
                         replace(self.visibility, valid_until=None)):
            self.assertFalse(self.publish(visibility=snapshot).actionable)

    def test_duplicate_does_not_republish_and_outdated_observation_is_rejected(self):
        first = self.publish()
        self.clock.now = 0.625
        self.assertIs(self.publish(), first)
        old = replace(self.batch, frame=replace(self.batch.frame, frame_id=1))
        visibility = replace(self.visibility, source_frame_id=1)
        self.assertEqual(self.publish(old, visibility).status, GuidanceStatus.OUTDATED)

    def test_hysteresis_uses_only_fresh_single_object_observations(self):
        self.sample(0.3, 0.625)
        self.assertEqual(self.publish().direction, Direction.LEFT)
        self.sample(0.34, 0.75)
        self.assertEqual(self.publish().direction, Direction.LEFT)
        self.sample(0.37, 0.875)
        self.assertEqual(self.publish().direction, Direction.CENTER)
        # Ambiguity clears the previous direction; no remembered box is used.
        self.publish(replace(self.batch, detections=self.batch.detections * 2))
        self.sample(0.32, 1)
        self.assertEqual(self.publish().direction, Direction.LEFT)

    def test_configurable_boundaries_without_hysteresis(self):
        self.guidance = GuidanceProcessor(self.sessions, clock=self.clock,
                                         settings=GuidanceSettings(left_boundary=0.2, right_boundary=0.8, hysteresis=0))
        self.sample(0.25, 0.625)
        self.assertEqual(self.publish().direction, Direction.CENTER)

    def test_new_session_cannot_inherit_direction_hysteresis_or_frame_ids(self):
        self.sample(0.3, 0.625)
        self.assertEqual(self.publish().direction, Direction.LEFT)
        self.session = self.sessions.start("phone", 10)
        self.frame_id = 0
        for captured in (0.75, 1, 1.25):
            self.sample(0.34, captured)
        # A previous LEFT would retain 0.34, but this is a new session.
        self.assertEqual(self.publish().direction, Direction.CENTER)

    def test_invalid_settings(self):
        for kwargs in ({"left_boundary": 0}, {"right_boundary": 1}, {"hysteresis": -1},
                       {"hysteresis": 0.5}, {"max_position_age": 0}, {"min_confidence": 2},
                       {"left_boundary": float("nan")}, {"min_confidence": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    GuidanceSettings(**kwargs)

    def test_cancellation_or_replacement_during_geometry_preparation_wins(self):
        for action in ("cancel", "replace"):
            with self.subTest(action=action):
                self.setUp()
                entered, release = threading.Event(), threading.Event()
                original = self.guidance._position

                def slow_position(batch, target):
                    result = original(batch, target)
                    entered.set()
                    if not release.wait(1):
                        raise RuntimeError("test handoff was not released")
                    return result

                self.guidance._position = slow_position
                thread = threading.Thread(target=self.guidance.update, args=(self.batch, self.visibility))
                thread.start()
                try:
                    self.assertTrue(entered.wait(1))
                    if action == "cancel":
                        self.sessions.cancel(self.session.session_id)
                    else:
                        self.sessions.start("keys", 10)
                finally:
                    release.set()
                    thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(self.guidance.latest().status, GuidanceStatus.INACTIVE)

    def test_cancellation_waits_for_guarded_publication_then_invalidates_output(self):
        entered, release, cancel_started, cancelled = (threading.Event() for _ in range(4))

        def publication_clock():
            entered.set()
            if not release.wait(1):
                raise RuntimeError("test publication was not released")
            return self.clock()

        self.guidance = GuidanceProcessor(self.sessions, clock=publication_clock)
        publisher = threading.Thread(target=self.guidance.update, args=(self.batch, self.visibility))
        publisher.start()

        def cancel():
            cancel_started.set()
            self.sessions.cancel(self.session.session_id)
            cancelled.set()

        canceller = threading.Thread(target=cancel)
        try:
            self.assertTrue(entered.wait(1))
            canceller.start()
            self.assertTrue(cancel_started.wait(1))
            self.assertFalse(cancelled.wait(0.01))
        finally:
            release.set()
            publisher.join(1)
            if canceller.ident is not None:
                canceller.join(1)
        self.assertFalse(publisher.is_alive())
        self.assertFalse(canceller.is_alive())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.guidance.latest().status, GuidanceStatus.INACTIVE)


if __name__ == "__main__":
    unittest.main()
