"""Optional laptop display. Reads existing snapshots; owns no camera or detector."""

import os
import threading
import time

import numpy as np

from homesense.application import ApplicationState
from homesense.perception import FreshnessSettings
from homesense.session import SessionStatus


class LaptopPreview:
    """Pump OpenCV on the main thread, at most 15 renders/s and a 1 ms key wait.

    Live and processed frames are separate: boxes always belong to the exact
    processed image. Slow inference must not freeze the live camera panel or
    draw old coordinates on a newer frame. No rendering occurs in a worker.
    """

    title = "HomeSense: live camera | detections (q: quit)"

    def __init__(self, *, cv=None, clock=time.monotonic):
        if cv is None:
            if os.name == 'posix' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                raise RuntimeError('--preview requires a graphical desktop; omit it for headless operation')
            import cv2 as cv
        self.cv, self.clock = cv, clock
        self._opened = False
        self._next_render = 0.0
        self.render_seconds = 0.0

    def _panel(self, frame, label, detections=(), detail=''):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        if frame is not None:
            # Buffer/result readers already give private images. Only drawing
            # boxes needs another copy; resize/text-on-canvas do not mutate input.
            image = frame.image.copy() if detections else frame.image
            for detection in detections:
                x1, y1, x2, y2 = map(round, detection.bbox)
                self.cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.cv.putText(image, f'{detection.class_name} {detection.confidence:.2f}',
                                (x1, max(18, y1 - 5)), self.cv.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            h, w = image.shape[:2]
            scale = min(640 / w, 420 / h)
            w, h = max(1, round(w * scale)), max(1, round(h * scale))
            canvas[60:60 + h, :w] = self.cv.resize(image, (w, h))
        self.cv.putText(canvas, label, (10, 25), self.cv.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
        if detail:
            self.cv.putText(canvas, detail, (10, 50), self.cv.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        return canvas

    def update(self, snapshot, coordinator=None):
        """Return False on q/window close; never wait for a frame or prediction."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Preview must run on the application main thread')
        now = self.clock()
        if now < self._next_render:
            return True
        self._next_render = now + 1 / 15
        if not self._opened:
            self.cv.namedWindow(self.title, self.cv.WINDOW_NORMAL)
            self._opened = True
        session = snapshot.session
        active = (snapshot.state is ApplicationState.SEARCHING and session is not None
                  and session.status is SessionStatus.ACTIVE and now < session.deadline)
        frame = coordinator.latest_frame() if active and coordinator is not None else None
        target = session.target.value if active else snapshot.state.value
        batch = snapshot.batch if active else None
        if batch is not None and (batch.session_id != session.session_id
                or not 0 <= now - batch.frame.captured_at < FreshnessSettings().max_result_age):
            batch = None
        if snapshot.errors:
            detail = f'ERROR {snapshot.errors[0].stage}: {snapshot.errors[0].exception}'
        elif snapshot.state is ApplicationState.STOPPING:
            detail = 'Stopping search; waiting for resource cleanup.'
        elif snapshot.state is ApplicationState.FAILED:
            detail = 'Search failed; check terminal diagnostics.'
        elif snapshot.state is ApplicationState.CLOSED:
            detail = 'Application closed; camera stopped.'
        elif not active:
            detail = 'Camera OFF. After feedback, say: Find my phone.'
        elif frame is None:
            detail = 'Search active; waiting for the first camera frame.'
        else:
            age = max(0.0, now - frame.captured_at)
            detail = f'Camera frame age: {age:.2f}s' + (' (STALE)' if age >= FreshnessSettings().max_frame_age else '')
            if snapshot.scan is not None and now < snapshot.scan.valid_until:
                detail += f'; {snapshot.scan.phase.value} {snapshot.scan.scan.progress:.0%}'
        live = self._panel(frame, f'Target: {target}; camera frame {frame.frame_id if frame else "-"}', detail=detail)
        processed = self._panel(batch.frame if batch else None,
                                f'YOLO frame {batch.frame.frame_id}; age {now - batch.frame.captured_at:.2f}s'
                                if batch else ('Waiting for fresh inference' if active else 'No active inference'),
                                batch.detections if batch else (),
                                detail=(f'Motion: {snapshot.motion.kind.value}; support {snapshot.motion.support_fraction:.0%}'
                                        if snapshot.motion is not None else
                                        (f'Outcome: {snapshot.outcome.value}; {snapshot.search_reason or ""}' if snapshot.outcome else '')))
        self.cv.imshow(self.title, np.concatenate((live, processed), axis=1))
        keep_running = (self.cv.waitKey(1) & 0xFF != ord('q')
                        and self.cv.getWindowProperty(self.title, self.cv.WND_PROP_VISIBLE) >= 1)
        self.render_seconds = max(0.0, self.clock() - now)
        return keep_running

    def close(self):
        if self._opened:
            self.cv.destroyWindow(self.title)
            self._opened = False
