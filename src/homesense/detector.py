"""Synchronous YOLO adapter; no camera, session, or guidance ownership."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from homesense.session import Target


class ModelLoadError(RuntimeError):
    """The configured checkpoint could not be loaded or is incompatible."""


class PredictionError(RuntimeError):
    """Inference failed or returned invalid detection data."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceSettings:
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    device: str | None = None

    def __post_init__(self) -> None:
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number between 0 and 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if type(self.image_size) is not int or self.image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("device must be a nonempty string or None")


class _PredictionBackend(Protocol):
    names: Mapping[int, str]
    task: str

    def predict(self, **kwargs: Any) -> Any: ...


def _load_yolo(path: str) -> _PredictionBackend:
    from ultralytics import YOLO
    return YOLO(path)


class YoloDetector:
    """Load once, predict repeatedly. Use from one inference owner at a time.

    The caller owns the BGR image and must not mutate it during predict().
    Frame/session metadata and freshness checks belong to the future worker.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        settings: InferenceSettings = InferenceSettings(),
        backend_factory: Callable[[str], _PredictionBackend] = _load_yolo,
    ) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        # Check before invoking Ultralytics, which can download missing weights.
        if not path.is_file():
            raise FileNotFoundError(f"Local checkpoint not found: {path}")
        if path.suffix.lower() != ".pt":
            raise ValueError("checkpoint must be a local .pt file")
        try:
            backend = backend_factory(str(path))
            names = dict(backend.names)
            if backend.task != "detect":
                raise ValueError("checkpoint must be an object detection model")
            if (
                any(type(key) is not int for key in names)
                or set(names) != {0, 1, 2}
                or set(names.values()) != {target.value for target in Target}
            ):
                raise ValueError(f"expected exactly phone, keys, wallet; got {names!r}")
        except Exception as exc:
            raise ModelLoadError(f"Could not load compatible checkpoint {path}: {exc}") from exc
        self._backend = backend
        self._names = names
        self._settings = settings

    @property
    def class_names(self) -> dict[int, str]:
        """Return a copy of the verified checkpoint mapping."""
        return self._names.copy()

    def predict(self, image: NDArray[np.uint8]) -> tuple[Detection, ...]:
        """Return original-image pixel boxes; an empty tuple means no detections."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("image must be a nonempty H x W x 3 BGR uint8 NumPy array")
        try:
            results = self._backend.predict(
                source=image, stream=False,
                conf=self._settings.confidence, iou=self._settings.iou,
                imgsz=self._settings.image_size, device=self._settings.device,
                save=False, save_txt=False, save_crop=False, show=False, verbose=False,
            )
            if len(results) != 1 or results[0].boxes is None:
                raise ValueError("expected one detection result with boxes")
            boxes = results[0].boxes.cpu().numpy()
            coordinates = np.asarray(boxes.xyxy)
            confidences = np.asarray(boxes.conf)
            classes = np.asarray(boxes.cls)
            count = len(classes)
            if coordinates.shape != (count, 4) or confidences.shape != (count,) or classes.shape != (count,):
                raise ValueError("inconsistent detection array shapes")
            detections = []
            height, width = image.shape[:2]
            for xyxy, confidence, class_value in zip(coordinates, confidences, classes):
                class_number = float(class_value)
                confidence = float(confidence)
                x1, y1, x2, y2 = (float(value) for value in xyxy)
                if not math.isfinite(class_number) or not class_number.is_integer() or int(class_number) not in self._names:
                    raise ValueError(f"unknown class ID: {class_value}")
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("invalid detection confidence")
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (
                    0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                ):
                    raise ValueError("invalid original-image bounding box")
                if confidence < self._settings.confidence:
                    continue
                class_id = int(class_number)
                detections.append(Detection(class_id, self._names[class_id], confidence, (x1, y1, x2, y2)))
            return tuple(detections)
        except Exception as exc:
            raise PredictionError(f"YOLO prediction failed: {exc}") from exc
