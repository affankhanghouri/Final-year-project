"""Isolated perception evidence collection; never saves camera images or changes production settings."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from homesense.application import ApplicationCoordinator, ApplicationState
from homesense.detector import InferenceSettings, YoloDetector
from homesense.guidance import GuidanceSettings
from homesense.perception import FreshnessSettings
from homesense.session import Target
from homesense.stability import StabilitySettings


class Distribution:
    """Bounded-memory numeric summary (no stored observations or images)."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def summary(self):
        return {"count": self.count, "min": self.minimum,
                "mean": self.total / self.count if self.count else None, "max": self.maximum}


class Predictions:
    """Counts post-NMS prediction boxes separately from frames containing them."""

    def __init__(self, target):
        self.target = target
        self.frames = 0
        self.empty_frames = 0
        self.target_frames = 0
        self.support_frames = 0
        self.other_class_frames = 0
        self.classes = {t.value: {"boxes": 0, "frames": 0, "confidence": Distribution(),
                                  "bands": Counter(), "examples": {}} for t in Target}

    def add(self, detections):
        self.frames += 1
        self.empty_frames += not detections
        names = {d.class_name for d in detections}
        self.target_frames += self.target in names
        self.other_class_frames += bool(names - {self.target})
        self.support_frames += any(d.class_name == self.target and
                                   d.confidence >= StabilitySettings().min_confidence for d in detections)
        for name in names:
            self.classes[name]["frames"] += 1
        for detection in detections:
            values = self.classes[detection.class_name]
            values["boxes"] += 1
            values["confidence"].add(detection.confidence)
            band = ("below_0.25" if detection.confidence < 0.25 else
                    "0.25_to_below_0.50" if detection.confidence < 0.5 else "0.50_to_1")
            values["bands"][band] += 1
            # At most one example per confidence band and class, never an image.
            values["examples"].setdefault(band, asdict(detection))

    def summary(self):
        return {
            "completed_frames": self.frames, "frames_with_no_boxes_any_class": self.empty_frames,
            "frames_with_target": self.target_frames, "frames_without_target": self.frames - self.target_frames,
            "frames_with_target_at_stability_confidence": self.support_frames,
            "frames_with_other_classes": self.other_class_frames,
            "target_hit_percent": 100 * self.target_frames / self.frames if self.frames else None,
            "classes": {name: {**{k: v for k, v in values.items() if k not in ("confidence", "examples")},
                               "confidence": values["confidence"].summary(),
                               "example_boxes_by_band": values["examples"]}
                        for name, values in self.classes.items()},
        }


class DetectorTap:
    """Observe all classes before DetectionWorker filters; return output unchanged."""

    def __init__(self, detector, target):
        self.detector = detector
        self.predictions = Predictions(target)
        self.attempts = 0
        self.failures = 0
        self.durations = Distribution()
        self.first_call_seconds = None
        self.lock = threading.Lock()

    def predict(self, image):
        started = time.monotonic()
        with self.lock:
            self.attempts += 1
        try:
            result = self.detector.predict(image)
        except Exception:
            with self.lock:
                self.failures += 1
            raise
        elapsed = time.monotonic() - started
        with self.lock:
            if self.first_call_seconds is None:
                self.first_call_seconds = elapsed
            self.durations.add(elapsed)
            self.predictions.add(result)
        return result

    def summary(self):
        with self.lock:
            return {"attempts": self.attempts, "failed_attempts": self.failures,
                    "prediction_call_seconds": self.durations.summary(),
                    "first_prediction_call_seconds": self.first_call_seconds,
                    "all_class_post_nms_before_worker_target_filter": self.predictions.summary()}


class CameraReads:
    """Count successful source reads, including a possible final unpublished read."""

    def __init__(self, source_factory=None):
        self.source_factory = source_factory or cv2.VideoCapture
        self.successful_reads = 0
        self.shapes = Counter()
        self.lock = threading.Lock()

    def open(self, camera_id):
        return CountedSource(self.source_factory(camera_id), self)

    def summary(self):
        with self.lock:
            return {"successful_camera_reads": self.successful_reads,
                    "source_shapes_hwc": dict(self.shapes)}


class CountedSource:
    def __init__(self, source, counts):
        self.source, self.counts = source, counts

    def isOpened(self):
        return self.source.isOpened()

    def read(self):
        ok, image = self.source.read()
        if ok and image is not None:
            with self.counts.lock:
                self.counts.successful_reads += 1
                self.counts.shapes[str(tuple(image.shape))] += 1
        return ok, image

    def release(self):
        return self.source.release()


class PipelineObservations:
    """Count unique retrieved batches; infer observe acceptance from its source metadata."""

    def __init__(self):
        self.last_id = 0
        self.retrieved = 0
        self.accepted = 0
        self.rejected = 0
        self.support = 0
        self.actionable_observations = 0
        self.age_at_retrieval = Distribution()
        self.capture_to_result = Distribution()
        self.inference = Distribution()
        self.visibility_entries = Counter()
        self.guidance_statuses = Counter()
        self.previous_visibility = None

    def add(self, snapshot, now):
        visibility = snapshot.visibility
        if visibility is not None:
            if visibility.state != self.previous_visibility:
                self.visibility_entries[visibility.state.value] += 1
            self.previous_visibility = visibility.state
        batch = snapshot.batch
        if batch is None or batch.frame.frame_id <= self.last_id:
            return
        self.last_id = batch.frame.frame_id
        self.retrieved += 1
        self.age_at_retrieval.add(now - batch.frame.captured_at)
        self.capture_to_result.add(batch.capture_to_result_seconds)
        self.inference.add(batch.inference_seconds)
        accepted = (visibility is not None and visibility.source_frame_id == batch.frame.frame_id
                    and visibility.last_observation_at == batch.frame.captured_at)
        self.accepted += accepted
        self.rejected += not accepted
        self.support += accepted and any(d.confidence >= StabilitySettings().min_confidence for d in batch.detections)
        guidance = snapshot.guidance
        if guidance is not None:
            self.guidance_statuses[guidance.status.value] += 1
            self.actionable_observations += guidance.actionable

    def summary(self):
        return {"distinct_retrieved_batches": self.retrieved,
                "stability_observations_accepted": self.accepted,
                "stability_observations_not_accepted": self.rejected,
                "accepted_observations_with_confident_target": self.support,
                "visibility_state_entries": dict(self.visibility_entries),
                "guidance_statuses_on_distinct_batches": dict(self.guidance_statuses),
                "actionable_distinct_batches": self.actionable_observations,
                "retrieval_age_seconds": self.age_at_retrieval.summary(),
                "read_return_to_inference_complete_seconds": self.capture_to_result.summary(),
                "worker_inference_seconds": self.inference.summary()}


def draw(image, detections, label):
    """Render on a display copy only; returned pixels are never passed to inference."""
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(v) for v in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"{detection.class_name} {detection.confidence:.3f}",
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(output, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return output


def show(image):
    cv2.imshow("HomeSense diagnostic - q: end trial", image)
    return (cv2.waitKey(10) & 0xFF == ord("q")
            or cv2.getWindowProperty("HomeSense diagnostic - q: end trial", cv2.WND_PROP_VISIBLE) < 1)


def run_pipeline(detector, args, *, camera_reads=None):
    reads = camera_reads or CameraReads()
    tap = DetectorTap(detector, args.target)
    app = ApplicationCoordinator(tap, camera_id=args.camera, source_factory=reads.open)
    observations = PipelineObservations()
    final = None
    started = time.monotonic()
    try:
        app.start_search(args.target, args.seconds)
        while True:
            snapshot = app.poll()
            observations.add(snapshot, time.monotonic())
            if snapshot.errors or snapshot.state in (ApplicationState.IDLE, ApplicationState.CLOSED):
                break
            if args.show:
                visibility = snapshot.visibility.state.value if snapshot.visibility else "unconfirmed"
                image = (draw(snapshot.batch.frame.image, snapshot.batch.detections,
                              f"production target only; visibility: {visibility}")
                         if snapshot.batch is not None else np.zeros((480, 640, 3), dtype=np.uint8))
                if show(image):
                    break
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        final = app.close()
    report = {**reads.summary(), **tap.summary(), "worker": asdict(final.stats),
              "session_wall_seconds_including_cleanup": time.monotonic() - started,
              "observations": observations.summary(), "outcome": final.outcome.value if final.outcome else None,
              "cleanup_complete": final.cleanup_complete,
              "errors": [{"stage": e.stage, "message": str(e.exception)} for e in final.errors]}
    return report, bool(final.errors) or not final.cleanup_complete


def compare_frame(image, normal, low, normal_counts, low_counts):
    # Identical unannotated pixels, settings differ only in the diagnostic confidence.
    ordinary = normal.predict(image.copy())
    lowered = low.predict(image.copy())
    normal_counts.add(ordinary)
    low_counts.add(lowered)
    return ordinary, lowered


def run_compare(normal, low, args, *, camera_reads=None):
    from homesense.camera import CameraWorker, LatestFrameBuffer

    normal_counts, low_counts = Predictions(args.target), Predictions(args.target)
    examples = []
    reads = camera_reads or CameraReads()
    camera = None
    errors = []
    try:
        if args.image:
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode local image: {args.image}")
            frames = [(1, image)]
        else:
            buffer = LatestFrameBuffer()
            camera = CameraWorker(buffer, args.camera, source_factory=reads.open)
            camera.start()
            frames = None
        deadline = time.monotonic() + args.seconds
        frame_id = 0
        while True:
            if frames is not None:
                if not frames:
                    break
                frame_id, image = frames.pop()
            else:
                if camera.error is not None:
                    raise RuntimeError("Camera failed") from camera.error
                if time.monotonic() >= deadline:
                    break
                frame = buffer.wait_for_newer(frame_id, timeout=0.05)
                if frame is None:
                    if buffer.closed:
                        break
                    continue
                frame_id, image = frame.frame_id, frame.image
            ordinary, lowered = compare_frame(image, normal, low, normal_counts, low_counts)
            # Bounded numerical examples; all frames still contribute to counters.
            if len(examples) < 5:
                examples.append({"frame_id": frame_id, "source_shape_hwc": list(image.shape),
                                 "production": [asdict(d) for d in ordinary],
                                 "lower_confidence": [asdict(d) for d in lowered]})
            if args.show:
                display = np.concatenate([draw(image, ordinary, "production 0.25 post-NMS"),
                                          draw(image, lowered, f"diagnostic {args.low_confidence} post-NMS")], axis=1)
                if frames is not None:
                    while time.monotonic() < deadline:
                        if show(display):
                            break
                elif show(display):
                    break
    except KeyboardInterrupt:
        pass
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        if camera is not None:
            try:
                if not camera.stop():
                    errors.append("camera worker still running")
            except Exception as error:
                errors.append(f"camera stop failed: {error}")
            if camera.error is not None:
                errors.append(str(camera.error))
    return {**reads.summary(), "production_post_nms": normal_counts.summary(),
            "lower_confidence_post_nms": low_counts.summary(), "paired_examples": examples,
            "errors": errors,
            "note": "Paired mode bypasses session/freshness/stability gates. Two sequential calls per frame; FPS/age are not production measurements."}, bool(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[t.value for t in Target], required=True)
    parser.add_argument("--mode", choices=("pipeline", "compare"), default="pipeline")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument("--image", type=Path, help="existing local image; compare mode only")
    parser.add_argument("--show", action="store_true", help="display only; never records images")
    parser.add_argument("--trial", default="unnamed", help="short condition label for the summary")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be positive and finite")
    if not math.isfinite(args.low_confidence) or not 0 < args.low_confidence < InferenceSettings().confidence:
        parser.error("low-confidence must be between zero and production confidence (0.25)")
    if args.image is not None and args.mode != "compare":
        parser.error("--image requires --mode compare")
    report = {"trial": args.trial, "mode": args.mode, "target": args.target,
              "requested_seconds": args.seconds, "prediction_level": "post-NMS and confidence filtering; not raw model tensors",
              "production_inference": asdict(InferenceSettings(device="cpu")),
              "freshness": asdict(FreshnessSettings()), "stability": asdict(StabilitySettings()),
              "guidance": asdict(GuidanceSettings())}
    failed = True
    started = time.monotonic()
    try:
        normal = YoloDetector(args.checkpoint, settings=InferenceSettings(device="cpu"))
        report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.expanduser().read_bytes()).hexdigest()
        report["class_mapping"] = normal.class_names
        if args.mode == "pipeline":
            report["measurements"], failed = run_pipeline(normal, args)
        else:
            low = YoloDetector(args.checkpoint, settings=InferenceSettings(confidence=args.low_confidence, device="cpu"))
            report["diagnostic_confidence"] = args.low_confidence
            report["measurements"], failed = run_compare(normal, low, args)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception as error:
                report["window_cleanup_error"] = str(error)
                failed = True
    report["wall_seconds_including_model_loading"] = time.monotonic() - started
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
