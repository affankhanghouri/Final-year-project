"""Measurement semantics and isolated diagnostics, without weights or real cameras."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from homesense.application import ApplicationSnapshot, ApplicationState
from homesense.camera import Frame
from homesense.detector import Detection
from homesense.guidance import GuidanceResult, GuidanceStatus, Direction
from homesense.perception import DetectionBatch
from homesense.session import SessionManager
from homesense.stability import Visibility, VisibilitySnapshot
from scripts import perception_diagnostic as diagnostic


PHONE = Detection(0, "phone", 0.9, (0, 0, 5, 5))
KEY = Detection(1, "keys", 0.06, (1, 1, 4, 4))
WALLET = Detection(2, "wallet", 0.6, (2, 2, 6, 6))


class DiagnosticTests(unittest.TestCase):
    def test_box_counts_frame_counts_confidence_bands_and_no_ground_truth_claim(self):
        counts = diagnostic.Predictions("keys")
        counts.add((KEY, replace(KEY, confidence=0.53), PHONE))
        counts.add(())
        counts.add((PHONE,))
        result = counts.summary()
        self.assertEqual(result["completed_frames"], 3)
        self.assertEqual(result["classes"]["keys"]["boxes"], 2)
        self.assertEqual(result["classes"]["keys"]["frames"], 1)
        self.assertEqual(result["frames_with_no_boxes_any_class"], 1)
        self.assertEqual(result["frames_without_target"], 2)
        self.assertEqual(result["frames_with_other_classes"], 2)
        self.assertEqual(result["frames_with_target_at_stability_confidence"], 1)
        self.assertAlmostEqual(result["target_hit_percent"], 100 / 3)
        self.assertEqual(result["classes"]["keys"]["bands"]["below_0.25"], 1)
        self.assertEqual(result["classes"]["keys"]["bands"]["0.50_to_1"], 1)
        self.assertNotIn("recall", result)

    def test_tap_returns_unchanged_output_and_counts_attempts_including_errors(self):
        records = (PHONE, WALLET)
        backend = Mock()
        backend.predict.side_effect = [records, RuntimeError("failed")]
        tap = diagnostic.DetectorTap(backend, "phone")
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertIs(tap.predict(image), records)
        self.assertIs(backend.predict.call_args.args[0], image)
        with self.assertRaises(RuntimeError):
            tap.predict(image)
        result = tap.summary()
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["failed_attempts"], 1)
        self.assertEqual(result["all_class_post_nms_before_worker_target_filter"]["completed_frames"], 1)

    def test_counted_source_preserves_pixels_and_release(self):
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        source = Mock()
        source.read.side_effect = [(True, image), (False, None)]
        reads = diagnostic.CameraReads(lambda _: source)
        counted = reads.open(0)
        self.assertIs(counted.read()[1], image)
        self.assertEqual(counted.read(), (False, None))
        counted.release()
        source.release.assert_called_once_with()
        self.assertEqual(reads.summary()["successful_camera_reads"], 1)
        self.assertEqual(reads.summary()["source_shapes_hwc"], {"(10, 12, 3)": 1})

    def test_pair_uses_identical_unannotated_pixels_and_counts_separately(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        first, second = Mock(), Mock()

        def normal_predict(pixels):
            np.testing.assert_array_equal(pixels, image)
            pixels[:] = 255  # Even a mutating backend must not contaminate the other input.
            return ()

        first.predict.side_effect = normal_predict
        second.predict.return_value = (KEY,)
        normal, low = diagnostic.Predictions("keys"), diagnostic.Predictions("keys")
        result = diagnostic.compare_frame(image, first, second, normal, low)
        np.testing.assert_array_equal(image, np.zeros_like(image))
        np.testing.assert_array_equal(second.predict.call_args.args[0], image)
        self.assertEqual(result, ((), (KEY,)))
        self.assertEqual(normal.summary()["target_hit_percent"], 0)
        self.assertEqual(low.summary()["target_hit_percent"], 100)

    def test_observations_do_not_double_count_and_detect_nonaccepted_batch(self):
        session = SessionManager(clock=lambda: 100).start("phone", 20)
        frame = Frame(1, 100, np.zeros((10, 10, 3), dtype=np.uint8))
        batch = DetectionBatch(session.session_id, frame, 100.1, 0.1, (PHONE,))
        visibility = VisibilitySnapshot(session.session_id, session.target, Visibility.VISIBLE,
                                        100, 100, 1, 100.4)
        guidance = GuidanceResult(session.session_id, session.target, GuidanceStatus.DIRECTION,
                                  Direction.LEFT, 0.25, 1, 100, 100.1, 100.4)
        snapshot = ApplicationSnapshot(ApplicationState.SEARCHING, session, None, None, (), False,
                                       batch, visibility, guidance)
        observations = diagnostic.PipelineObservations()
        observations.add(snapshot, 100.2)
        observations.add(snapshot, 100.25)
        later_batch = replace(batch, frame=replace(frame, frame_id=2, captured_at=100.3))
        observations.add(replace(snapshot, batch=later_batch), 100.4)
        observations.add(replace(snapshot, batch=None,
                                 visibility=replace(visibility, state=Visibility.UNCERTAIN)), 100.5)
        result = observations.summary()
        self.assertEqual(result["distinct_retrieved_batches"], 2)
        self.assertEqual(result["stability_observations_accepted"], 1)
        self.assertEqual(result["stability_observations_not_accepted"], 1)
        self.assertEqual(result["visibility_state_entries"], {"visible": 1, "uncertain": 1})

    def test_offline_compare_never_opens_camera_or_writes_images(self):
        args = SimpleNamespace(target="keys", image=Path("existing.jpg"), seconds=20,
                               camera=0, show=False, low_confidence=0.05)
        first, second = Mock(), Mock()
        first.predict.return_value = ()
        second.predict.return_value = (KEY,)
        with patch.object(diagnostic.cv2, "imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch.object(diagnostic.cv2, "VideoCapture") as camera, \
             patch.object(diagnostic.cv2, "imwrite") as write:
            report, failed = diagnostic.run_compare(first, second, args)
        self.assertFalse(failed)
        camera.assert_not_called()
        write.assert_not_called()
        self.assertEqual(report["production_post_nms"]["frames_with_target"], 0)
        self.assertEqual(report["lower_confidence_post_nms"]["frames_with_target"], 1)

    def test_pipeline_counts_all_classes_before_real_worker_filters(self):
        class Source:
            def isOpened(self):
                return True

            def read(self):
                threading.Event().wait(0.002)
                return True, np.zeros((10, 10, 3), dtype=np.uint8)

            def release(self):
                pass

        detector = Mock()
        detector.predict.return_value = (PHONE, WALLET)
        reads = diagnostic.CameraReads(lambda _: Source())
        args = SimpleNamespace(target="phone", camera=0, seconds=0.15, show=False)
        report, failed = diagnostic.run_pipeline(detector, args, camera_reads=reads)
        self.assertFalse(failed, report)
        self.assertGreater(report["successful_camera_reads"], 0)
        all_classes = report["all_class_post_nms_before_worker_target_filter"]
        self.assertGreater(all_classes["classes"]["wallet"]["boxes"], 0)
        self.assertEqual(report["attempts"], report["worker"]["processed"])
        self.assertGreater(report["observations"]["stability_observations_accepted"], 0)
        self.assertTrue(report["cleanup_complete"])

    def test_invalid_cli_never_loads_weights(self):
        for args in (["--target", "keys", "--seconds", "nan"],
                     ["--target", "wallet", "--image", "private.jpg"],
                     ["--target", "keys", "--low-confidence", "0.3"]):
            with self.subTest(args=args), patch("sys.argv", ["diagnostic"] + args), \
                 patch.object(diagnostic, "YoloDetector") as loader, patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    diagnostic.main()
                loader.assert_not_called()

    def test_compare_prediction_failure_and_stop_failure_are_both_reported(self):
        import homesense.camera as camera_module

        args = SimpleNamespace(target="keys", image=None, seconds=20, camera=0,
                               show=False, low_confidence=0.05)
        camera = Mock(error=None)
        camera.stop.side_effect = RuntimeError("stop failed")
        buffer = Mock(closed=False)
        buffer.wait_for_newer.return_value = Frame(1, 100, np.zeros((10, 10, 3), dtype=np.uint8))
        normal = Mock()
        normal.predict.side_effect = RuntimeError("prediction failed")
        with patch.object(camera_module, "CameraWorker", return_value=camera), \
             patch.object(camera_module, "LatestFrameBuffer", return_value=buffer):
            report, failed = diagnostic.run_compare(normal, Mock(), args)
        self.assertTrue(failed)
        self.assertTrue(any("prediction failed" in message for message in report["errors"]))
        self.assertTrue(any("stop failed" in message for message in report["errors"]))
        camera.stop.assert_called_once_with()
