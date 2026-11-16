"""Verify a local checkpoint and optionally infer on one local image."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

from homesense.detector import InferenceSettings, ModelLoadError, PredictionError, YoloDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="local test image path")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        detector = YoloDetector(
            args.checkpoint,
            settings=InferenceSettings(confidence=args.confidence, device=args.device),
        )
        print(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
                          "class_mapping": detector.class_names}, indent=2), flush=True)
        if args.image is None:
            print("No test image supplied; model loaded, image inference NOT performed.", file=sys.stderr)
            return 1
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Local test image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode test image: {image_path}")
        detections = detector.predict(image)
        height, width = image.shape[:2]
        # The adapter rejects boxes outside these original-image pixel bounds.
        print(json.dumps({
            "image": str(image_path),
            "original_width": width,
            "original_height": height,
            "bbox_format": "xyxy, original-image pixels",
            "detections": [asdict(detection) for detection in detections],
        }, indent=2))
        if not detections:
            print("Inference completed with no detections; no boxes available to inspect.", file=sys.stderr)
        return 0
    except (OSError, ValueError, ModelLoadError, PredictionError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
