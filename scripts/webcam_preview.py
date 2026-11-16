"""Manual webcam smoke test. Run from the project root; press q to exit."""

import argparse
import logging
import time

import cv2

from homesense.camera import CameraWorker, LatestFrameBuffer


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    buffer = LatestFrameBuffer()
    worker = CameraWorker(buffer)
    window = "HomeSense webcam preview - q to exit"
    last_id = 0
    displayed = 0
    fps = 0.0
    interval_start = time.monotonic()
    exit_code = 0
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        worker.start()
        while True:
            frame = buffer.wait_for_newer(last_id, timeout=0.05)
            if worker.error is not None:
                raise RuntimeError("Camera worker failed") from worker.error
            if frame is not None:
                last_id = frame.frame_id
                displayed += 1
                now = time.monotonic()
                elapsed = now - interval_start
                if elapsed >= 1.0:
                    fps = displayed / elapsed
                    displayed = 0
                    interval_start = now
                # The buffer gives this consumer its own image, safe to annotate.
                cv2.putText(
                    frame.image,
                    f"Frame {last_id} | Preview FPS: {fps:.1f} | q: exit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(window, frame.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("Webcam preview failed")
        exit_code = 1
    finally:
        try:
            if not worker.stop():
                logging.error("Camera read is still blocked; shutdown is waiting for the backend")
                exit_code = 1
            elif worker.error is not None and exit_code == 0:
                logging.error("Camera worker error: %s", worker.error)
                exit_code = 1
        finally:
            cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
