"""Manual coordinator smoke test; q exits, 1/2/3 replace with phone/keys/wallet."""

import argparse
from dataclasses import asdict
import logging
import time

import cv2
import numpy as np

from homesense.application import ApplicationCoordinator, ApplicationState
from homesense.detector import InferenceSettings, YoloDetector
from homesense.session import Target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[target.value for target in Target], default="phone")
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--shutdown-timeout", type=float, default=2.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = None
    exit_code = 0
    window = "HomeSense coordinator - q: exit, 1: phone, 2: keys, 3: wallet"
    try:
        detector = YoloDetector(args.checkpoint, settings=InferenceSettings(device="cpu"))
        app = ApplicationCoordinator(detector, camera_id=args.camera, shutdown_timeout=args.shutdown_timeout)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        app.start_search(args.target, args.timeout)
        previous = None
        while True:
            snapshot = app.poll()
            summary = (snapshot.state, snapshot.outcome, snapshot.outcome_session_id)
            if summary != previous:
                logging.info("Application: %s; outcome: %s; outcome session: %s", *summary)
                previous = summary
            if snapshot.errors or snapshot.state in (ApplicationState.IDLE, ApplicationState.CLOSED):
                break
            image = (snapshot.batch.frame.image if snapshot.batch is not None
                     else np.zeros((480, 640, 3), dtype=np.uint8))
            direction = snapshot.guidance
            now = time.monotonic()
            if snapshot.session is not None and now >= snapshot.session.deadline:
                continue  # Let poll handle timeout; do not present an expired session.
            label = (direction.direction.value.upper() if direction is not None and direction.actionable
                     and direction.valid_until is not None and now < direction.valid_until else "none")
            visibility = snapshot.visibility.state.value if snapshot.visibility else "unconfirmed"
            target = snapshot.session.target.value if snapshot.session else args.target
            cv2.putText(image, f"{target}: {visibility}; image-relative direction: {label}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(window, image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            target = {ord("1"): "phone", ord("2"): "keys", ord("3"): "wallet"}.get(key)
            if target is not None:
                app.start_search(target, args.timeout)
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Coordinator preview failed")
        exit_code = 1
    finally:
        try:
            if app is not None:
                final = app.close()
                logging.info("Final state: %s; cleanup complete: %s; statistics: %s",
                             final.state.value, final.cleanup_complete, asdict(final.stats))
                for error in final.errors:
                    logging.error("%s: %s", error.stage, error.exception)
                if final.errors or not final.cleanup_complete:
                    exit_code = 1
        except Exception:
            logging.exception("Coordinator cleanup failed")
            exit_code = 1
        try:
            cv2.destroyAllWindows()
        except Exception:
            logging.exception("Window cleanup failed")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
