from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from homesense.detector import (
    Detection, InferenceSettings, ModelLoadError, PredictionError, YoloDetector,
)


class FakeBoxes:
    def __init__(self, rows=()) -> None:
        data = np.asarray(rows, dtype=float).reshape(-1, 6)
        self.xyxy = data[:, :4]
        self.conf = data[:, 4]
        self.cls = data[:, 5]

    def cpu(self):
        return self

    def numpy(self):
        return self


class DetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "best.pt"
        self.path.touch()
        self.image = np.zeros((40, 80, 3), dtype=np.uint8)
        # Deliberately different from any assumed canonical ordering.
        self.backend = SimpleNamespace(
            names={0: "wallet", 1: "phone", 2: "keys"}, task="detect",
            predict=Mock(return_value=[SimpleNamespace(boxes=FakeBoxes([
                [1, 2, 30, 35, 0.9, 2], [0, 0, 80, 40, 0.8, 0],
            ]))]),
        )
        self.factory = Mock(return_value=self.backend)

    def detector(self, **kwargs):
        return YoloDetector(self.path, backend_factory=self.factory, **kwargs)

    def test_load_once_map_actual_ids_and_return_plain_records(self) -> None:
        detector = self.detector()
        result = detector.predict(self.image)
        detector.predict(self.image)
        self.factory.assert_called_once_with(str(self.path.resolve()))
        self.assertEqual(self.backend.predict.call_count, 2)
        self.assertEqual(result, (
            Detection(2, "keys", 0.9, (1.0, 2.0, 30.0, 35.0)),
            Detection(0, "wallet", 0.8, (0.0, 0.0, 80.0, 40.0)),
        ))
        with self.assertRaises(FrozenInstanceError):
            result[0].confidence = 0
        mapping = detector.class_names
        mapping[2] = "wrong"
        self.assertEqual(detector.class_names[2], "keys")

    def test_settings_forwarded_and_persistence_disabled(self) -> None:
        detector = self.detector(settings=InferenceSettings(confidence=0.85, iou=0.5, device="cpu"))
        result = detector.predict(self.image)
        self.assertEqual(len(result), 1)
        kwargs = self.backend.predict.call_args.kwargs
        self.assertIs(kwargs["source"], self.image)
        self.assertEqual((kwargs["conf"], kwargs["iou"], kwargs["imgsz"], kwargs["device"]),
                         (0.85, 0.5, 640, "cpu"))
        for key in ("stream", "save", "save_txt", "save_crop", "show", "verbose"):
            self.assertFalse(kwargs[key])

    def test_empty_detections(self) -> None:
        self.backend.predict.return_value = [SimpleNamespace(boxes=FakeBoxes())]
        self.assertEqual(self.detector().predict(self.image), ())

    def test_missing_checkpoint_never_invokes_loader(self) -> None:
        with self.assertRaises(FileNotFoundError):
            YoloDetector(self.path.parent / "missing.pt", backend_factory=self.factory)
        self.factory.assert_not_called()

    def test_loading_failure_preserves_cause(self) -> None:
        self.factory.side_effect = OSError("bad weights")
        with self.assertRaises(ModelLoadError) as caught:
            self.detector()
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_incompatible_class_mapping_or_task(self) -> None:
        for mapping in ({0: "phone", 1: "keys"},
                        {0: "phone", 1: "keys", 2: "person"},
                        {0: "phone", 1: "keys", 3: "wallet"},
                        {"0": "phone", "1": "keys", "2": "wallet"},
                        {0: "phone", 1: "phone", 2: "keys"}):
            with self.subTest(mapping=mapping):
                self.backend.names = mapping
                with self.assertRaises(ModelLoadError):
                    self.detector()
        self.backend.names = {0: "wallet", 1: "phone", 2: "keys"}
        self.backend.task = "segment"
        with self.assertRaises(ModelLoadError):
            self.detector()

    def test_invalid_input_never_runs_prediction(self) -> None:
        detector = self.detector()
        for image in (None, [], np.zeros((4, 4), dtype=np.uint8),
                      np.zeros((4, 4, 4), dtype=np.uint8),
                      np.zeros((0, 4, 3), dtype=np.uint8),
                      np.zeros((4, 4, 3), dtype=float)):
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(ValueError):
                    detector.predict(image)
        self.backend.predict.assert_not_called()

    def test_prediction_failure_preserves_cause(self) -> None:
        self.backend.predict.side_effect = RuntimeError("inference failed")
        with self.assertRaises(PredictionError) as caught:
            self.detector().predict(self.image)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_malformed_output_is_not_silently_accepted(self) -> None:
        detector = self.detector()
        for row in ([1, 2, 3, 4, 0.9, 9], [1, 2, 3, 4, 0.9, 1.5],
                    [1, 2, 3, 4, float("nan"), 0], [1, 2, 3, 4, 1.2, 0],
                    [4, 2, 3, 4, 0.9, 0], [1, 2, 90, 4, 0.9, 0],
                    [float("nan"), 2, 3, 4, 0.9, 0]):
            with self.subTest(row=row):
                self.backend.predict.return_value = [SimpleNamespace(boxes=FakeBoxes([row]))]
                with self.assertRaises(PredictionError):
                    detector.predict(self.image)
        for results in ([], [SimpleNamespace(boxes=None)]):
            self.backend.predict.return_value = results
            with self.assertRaises(PredictionError):
                detector.predict(self.image)

    def test_invalid_settings(self) -> None:
        for kwargs in ({"confidence": -1}, {"confidence": True}, {"iou": float("nan")},
                       {"iou": 2}, {"image_size": 0}, {"image_size": True}, {"device": ""}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    InferenceSettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
