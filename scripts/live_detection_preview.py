"""Manual camera/detector integration test; press q to cancel."""

import argparse
from dataclasses import asdict
import logging
import time

import cv2
import numpy as np

from homesense.camera import CameraWorker, LatestFrameBuffer
from homesense.detector import InferenceSettings, YoloDetector
from homesense.guidance import GuidanceProcessor, GuidanceSettings
from homesense.perception import DetectionWorker, FreshnessSettings
from homesense.session import SessionManager, Target
from homesense.stability import StabilityProcessor, StabilitySettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[target.value for target in Target], default="phone")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-frame-age", type=float, default=0.25, help="seconds; test parameter")
    parser.add_argument("--max-result-age", type=float, default=1.0, help="capture-to-now seconds; test parameter")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    defaults = StabilitySettings()
    parser.add_argument("--confirmation-span", type=float, default=defaults.confirmation_span)
    parser.add_argument("--min-positive-observations", type=int, default=defaults.min_positive_observations)
    parser.add_argument("--max-positive-gap", type=float, default=defaults.max_positive_gap)
    parser.add_argument("--uncertain-after", type=float, default=defaults.uncertain_after)
    parser.add_argument("--forget-after", type=float, default=defaults.forget_after)
    parser.add_argument("--stability-max-age", type=float, default=defaults.max_observation_age)
    parser.add_argument("--stability-confidence", type=float, default=defaults.min_confidence)
    direction_defaults = GuidanceSettings()
    parser.add_argument("--left-boundary", type=float, default=direction_defaults.left_boundary)
    parser.add_argument("--right-boundary", type=float, default=direction_defaults.right_boundary)
    parser.add_argument("--direction-hysteresis", type=float, default=direction_defaults.hysteresis)
    parser.add_argument("--max-position-age", type=float, default=direction_defaults.max_position_age)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sessions = SessionManager()
    camera = None
    inference = None
    session = None
    stability = None
    guidance = None
    window = "HomeSense live detection - q to cancel"
    exit_code = 0
    try:
        settings = FreshnessSettings(args.max_frame_age, args.max_result_age)
        stability = StabilityProcessor(sessions, settings=StabilitySettings(
            confirmation_span=args.confirmation_span,
            min_positive_observations=args.min_positive_observations,
            max_positive_gap=args.max_positive_gap,
            uncertain_after=args.uncertain_after,
            forget_after=args.forget_after,
            max_observation_age=args.stability_max_age,
            min_confidence=args.stability_confidence,
        ))
        guidance = GuidanceProcessor(sessions, settings=GuidanceSettings(
            left_boundary=args.left_boundary, right_boundary=args.right_boundary,
            hysteresis=args.direction_hysteresis, max_position_age=args.max_position_age,
            min_confidence=args.stability_confidence,
        ))
        detector = YoloDetector(args.checkpoint, settings=InferenceSettings(confidence=args.confidence, device=args.device))
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        session = sessions.start(args.target, args.timeout)
        buffer = LatestFrameBuffer()
        camera = CameraWorker(buffer, args.camera)
        inference = DetectionWorker(buffer, detector, sessions, session, settings=settings)
        camera.start()
        inference.start()
        while sessions.is_eligible(session.session_id):
            if camera.error is not None:
                raise RuntimeError("Camera failed") from camera.error
            if inference.error is not None:
                raise RuntimeError("Inference failed") from inference.error
            if inference.finished:
                break
            batch = inference.latest_result()
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            lines = [f"Searching for {args.target}: waiting for a fresh result"]
            if batch is not None:
                # Repeated display reads are harmless: the processor rejects duplicates.
                stability.observe(batch)
                image = batch.frame.image
                for detection in batch.detections:
                    x1, y1, x2, y2 = (round(v) for v in detection.bbox)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"raw {detection.class_name} {detection.confidence:.1%}",
                                (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                lines = [
                    f"Frame {batch.frame.frame_id} | target: {args.target} | q: cancel",
                    f"Capture-to-result {batch.capture_to_result_seconds * 1000:.0f} ms | inference {batch.inference_seconds * 1000:.0f} ms",
                    f"Processed FPS {inference.stats.processed_fps:.1f} | current age {(time.monotonic() - batch.frame.captured_at) * 1000:.0f} ms",
                ]
                # Rendering can take time: suppress a result that expired meanwhile.
                if time.monotonic() - batch.frame.captured_at >= settings.max_result_age:
                    image = np.zeros_like(image)
                    lines = ["Result expired; waiting for a fresh result"]
            visibility = stability.snapshot()
            guidance.update(batch, visibility)
            lines.append(f"Visibility: {visibility.state.value} (presence only; support >= {args.stability_confidence:.0%})")
            for index, line in enumerate(lines):
                cv2.putText(image, line, (10, 25 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            direction = guidance.latest()
            label = direction.direction.value.upper() if direction is not None and direction.actionable else (
                f"none ({direction.status.value})" if direction is not None else "none"
            )
            cv2.putText(image, f"Image-relative direction: {label}", (10, 25 + 25 * len(lines)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if not sessions.is_eligible(session.session_id):
                break
            cv2.imshow(window, image)
            if cv2.waitKey(10) & 0xFF == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Live detection preview failed")
        exit_code = 1
    finally:
        # Each cleanup operation must run even if a previous operation fails.
        cleanup = []
        if session is not None:
            cleanup.append(("Session cancellation", lambda: sessions.cancel(session.session_id)))
        if stability is not None:
            cleanup.append(("Visibility invalidation", stability.snapshot))
        if guidance is not None:
            cleanup.append(("Guidance invalidation", guidance.latest))
        for label, operation in cleanup:
            try:
                operation()
            except Exception:
                logging.exception("%s failed during shutdown", label)
                exit_code = 1
        for label, worker in (("Camera", camera), ("Inference", inference)):
            if worker is None:
                continue
            try:
                if not worker.stop():
                    logging.error("%s has not stopped: backend operation may be blocked", label)
                    exit_code = 1
            except Exception:
                logging.exception("%s stop failed", label)
                exit_code = 1
            try:
                if worker.error is not None:
                    logging.error("%s worker error: %s", label, worker.error)
                    exit_code = 1
            except Exception:
                logging.exception("%s error inspection failed", label)
                exit_code = 1
        if inference is not None:
            try:
                logging.info("Inference statistics: %s", asdict(inference.stats))
            except Exception:
                logging.exception("Inference statistics unavailable")
                exit_code = 1
        try:
            cv2.destroyAllWindows()
        except Exception:
            logging.exception("Window cleanup failed")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
