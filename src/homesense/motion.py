"""Conservative apparent camera motion from existing frames; no pose/coverage claim."""

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from uuid import UUID

import numpy as np

from homesense.camera import Frame, LatestFrameBuffer
from homesense.config import positive
from homesense.session import SearchSession, SessionManager
from homesense import diagnostic_logging as diagnostic


class MotionKind(str, Enum):
    LEFT = 'left'
    RIGHT = 'right'
    STATIONARY = 'stationary'
    UNKNOWN = 'unknown'
    TOO_FAST = 'too_fast'


@dataclass(frozen=True)
class MotionSettings:
    width: int = 320
    sample_period: float = .15
    max_age: float = .5
    max_pair_gap: float = .5
    min_features: int = 24
    min_inlier_fraction: float = .75
    min_grid_cells: int = 6
    max_step: float = .15
    max_speed: float = .7
    stationary_speed: float = .015
    min_sharpness: float = 15.0
    max_features: int = 140
    max_track_error: float = 25.0
    max_reverse_error: float = 1.0
    ransac_error: float = 1.5
    max_scale_change: float = .035
    max_roll_degrees: float = 2.0

    def __post_init__(self):
        for name in ('sample_period', 'max_age', 'max_pair_gap', 'max_step', 'max_speed', 'stationary_speed',
                     'min_sharpness', 'max_track_error', 'max_reverse_error', 'ransac_error', 'max_scale_change', 'max_roll_degrees'):
            positive(name, getattr(self, name))
        for name, lower in (('width', 80), ('min_features', 8), ('min_grid_cells', 4), ('max_features', 8)):
            if type(getattr(self, name)) is not int or getattr(self, name) < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        if not .5 <= self.min_inlier_fraction <= 1 or self.min_grid_cells > 12:
            raise ValueError('invalid motion consensus settings')
        if self.stationary_speed >= self.max_speed:
            raise ValueError('stationary speed must be below maximum speed')
        if self.min_features > self.max_features:
            raise ValueError('minimum features exceeds feature budget')


@dataclass(frozen=True)
class MotionEvidence:
    session_id: UUID
    frame_id: int
    captured_at: float
    kind: MotionKind
    epoch: int
    offset: float
    step: float = 0.0
    reason: str = ''
    inliers: int = 0
    features: int = 0
    processing_seconds: float = 0.0
    pair_seconds: float = 0.0

    @property
    def reliable(self):
        return self.kind in (MotionKind.LEFT, MotionKind.RIGHT, MotionKind.STATIONARY)

    @property
    def support_fraction(self):
        """Geometric consensus, not a calibrated probability of camera rotation."""
        return self.inliers/self.features if self.features else 0.0


class CameraMotionEstimator:
    """Sparse forward/backward flow with distributed peripheral consensus.

    Signed offset is in image widths, not yaw degrees. Static scene motion has
    the opposite sign to apparent camera movement. Long/unverified gaps break
    continuity; callers must not sum independent motion chunks or elapsed time.
    A foreground occupying the whole image remains fundamentally ambiguous.
    A rejected pair retains its last accepted reference for at most max_pair_gap;
    recovery must track back to that image, never sum motion across an unseen gap.
    """

    def __init__(self, settings=MotionSettings()):
        import cv2
        self.cv, self.settings = cv2, settings
        self._previous = None
        self._epoch, self._offset = 0, 0.0
        self._session_id = None
        self._pair_seconds = 0.0
        self._diagnostic_started = time.monotonic()
        self._diagnostic_geometry = {}

    def _result(self, frame, sid, kind, *, step=0.0, reason='', inliers=0, features=0):
        previous_offset = self._offset
        if kind is MotionKind.TOO_FAST:
            self._previous = None
            self._epoch += 1
            self._offset = 0.0
        elif kind is not MotionKind.UNKNOWN:
            self._offset += step
        if diagnostic.enabled():
            diagnostic.event('motion', 'motion_evidence', sid, important=False,
                frame_id=frame.frame_id, captured_at=frame.captured_at, direction=kind.value,
                reliable=kind in (MotionKind.LEFT, MotionKind.RIGHT, MotionKind.STATIONARY),
                reason=reason or 'accepted', tracked_features=features, inliers=inliers,
                support=inliers/features if features else 0., epoch=self._epoch,
                previous_offset=previous_offset, offset=self._offset, credited_step=step,
                pair_seconds=self._pair_seconds,
                processing_seconds=time.monotonic()-self._diagnostic_started, **self._diagnostic_geometry)
        return MotionEvidence(sid, frame.frame_id, frame.captured_at, kind, self._epoch,
                              self._offset, step, reason, inliers, features, pair_seconds=self._pair_seconds)

    def observe(self, frame: Frame, session_id: UUID) -> MotionEvidence:
        cv, settings = self.cv, self.settings
        if diagnostic.enabled():
            self._diagnostic_started = time.monotonic()
            self._diagnostic_geometry = dict(min_features=settings.min_features,
                min_support=settings.min_inlier_fraction, max_pair_gap=settings.max_pair_gap,
                max_step=settings.max_step, max_speed=settings.max_speed,
                displacement_floor=settings.stationary_speed*settings.sample_period)
        image = frame.image
        self._pair_seconds = 0.0
        if self._session_id != session_id:
            self._session_id = session_id
            self._previous = None
            self._offset = 0.0
            self._epoch += 1
        h, w = image.shape[:2]
        if min(h, w) < 16:
            self._previous = None
            self._offset = 0.0
            self._epoch += 1
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='image too small')
        width = min(w, settings.width)
        gray = cv.resize(cv.cvtColor(image, cv.COLOR_BGR2GRAY), (width, max(16, round(h * width / w))),
                         interpolation=cv.INTER_AREA)
        previous = self._previous
        restart_reason = 'initial frame'
        if previous is not None:
            self._pair_seconds = frame.captured_at-previous[1]
            if diagnostic.enabled():
                self._diagnostic_geometry.update(reference_frame_id=previous[2], reference_captured_at=previous[1])
        if previous is not None and (not 0 < frame.captured_at-previous[1] <= settings.max_pair_gap
                                     or frame.frame_id <= previous[2] or gray.shape != previous[0].shape):
            self._previous = previous = None
            self._offset = 0.0
            self._epoch += 1
            restart_reason = 'discontinuous frames'
        sharpness = cv.Laplacian(gray, cv.CV_32F).var()
        if diagnostic.enabled():
            self._diagnostic_geometry.update(sharpness=float(sharpness), min_sharpness=settings.min_sharpness)
        if sharpness < settings.min_sharpness:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='blur or low texture')
        if previous is None:
            self._previous = (gray, frame.captured_at, frame.frame_id)
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason=restart_reason)
        old, timestamp, frame_id = previous
        dt = frame.captured_at - timestamp
        h, w = gray.shape
        mask = np.full_like(gray, 255)
        mask[round(h*.25):round(h*.75), round(w*.25):round(w*.75)] = 0
        mask[:5] = mask[-5:] = 0
        mask[:, :5] = mask[:, -5:] = 0
        # A global corner budget can be monopolized by one high-contrast object.
        # Select within each cell; all tracks still face the same geometric tests.
        points = []
        budget, extra = divmod(settings.max_features, 12)
        for row in range(3):
            for col in range(4):
                x0, x1, y0, y1 = col*w//4, (col+1)*w//4, row*h//3, (row+1)*h//3
                count = budget + int(row*4+col < extra)
                if not count:
                    continue
                found = cv.goodFeaturesToTrack(old[y0:y1, x0:x1], maxCorners=count,
                    qualityLevel=.02, minDistance=7, mask=mask[y0:y1, x0:x1])
                if found is not None:
                    points.append(found + np.float32([x0, y0]))
        if diagnostic.enabled():
            self._diagnostic_geometry['selected_features'] = sum(len(p) for p in points)
        if not points or sum(len(p) for p in points) < settings.min_features:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='insufficient background features')
        points = np.concatenate(points)
        if diagnostic.enabled():
            self._diagnostic_geometry['selected_features'] = len(points)
        options = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 20, .03))
        moved, forward, error = cv.calcOpticalFlowPyrLK(old, gray, points, None, **options)
        if moved is None:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='flow failed')
        back, backward, _ = cv.calcOpticalFlowPyrLK(gray, old, moved, None, **options)
        if back is None:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='reverse flow failed')
        valid = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                 & (np.linalg.norm(back-points, axis=2).ravel() < settings.max_reverse_error)
                 & (error.ravel() < settings.max_track_error) & np.isfinite(moved).all(axis=(1, 2)))
        source, destination = points[valid].reshape(-1, 2), moved[valid].reshape(-1, 2)
        count = len(source)
        if count < settings.min_features:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='inconsistent feature tracks', features=count)
        # Camera panning is projective: a similarity transform wrongly rejects
        # valid peripheral tracks at ordinary yaw angles. Keep the same RANSAC
        # residual, inlier fraction and spatial support with the appropriate model.
        matrix, inlier_mask = cv.findHomography(source, destination, cv.RANSAC,
            settings.ransac_error, maxIters=100, confidence=.99)
        if matrix is None or inlier_mask is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-9:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason='no scene consensus', features=count)
        support = source[inlier_mask.ravel().astype(bool)]
        inliers = len(support)
        cells = {(min(3, int(x*4/w)), min(2, int(y*3/h))) for x, y in support}
        distributed = (inliers >= settings.min_features and len(cells) >= settings.min_grid_cells
                       and np.ptp(support[:, 0]) >= .55*w and np.ptp(support[:, 1]) >= .5*h
                       and support[:, 0].min() < .2*w and support[:, 0].max() > .8*w
                       and support[:, 1].min() < .2*h and support[:, 1].max() > .8*h)
        if diagnostic.enabled():
            self._diagnostic_geometry.update(grid_cells=len(cells), min_grid_cells=settings.min_grid_cells,
                spatial_support_accepted=bool(distributed), transform=matrix.tolist(),
                inlier_bounds=[float(support[:, 0].min()), float(support[:, 1].min()),
                               float(support[:, 0].max()), float(support[:, 1].max())] if inliers else None,
                max_scale_change=settings.max_scale_change, max_roll_degrees=settings.max_roll_degrees,
                forward_backward_limit=settings.max_reverse_error, track_error_limit=settings.max_track_error,
                ransac_error=settings.ransac_error)
            center_projection = matrix @ np.array([w/2, h/2, 1.])
            if abs(center_projection[2]) > 1e-9:
                self._diagnostic_geometry['candidate_scene_displacement'] = (
                    center_projection[:2]/center_projection[2]-[w/2, h/2]).tolist()
        if not distributed or inliers / count < settings.min_inlier_fraction:
            return self._result(frame, session_id, MotionKind.UNKNOWN,
                                reason='insufficient spatial support' if not distributed else 'conflicting scene motion',
                                inliers=inliers, features=count)
        center = np.array([w/2, h/2])
        probes = np.float32([center, center+[1, 0], center+[0, 1]]).reshape(1, -1, 2)
        projected = cv.perspectiveTransform(probes, matrix)[0]
        jacobian = (projected[1:]-projected[0]).T
        scales = np.linalg.svd(jacobian, compute_uv=False)
        roll = abs(math.degrees(math.atan2(jacobian[1, 0]-jacobian[0, 1], np.trace(jacobian))))
        dx, dy = projected[0] - center
        if diagnostic.enabled():
            self._diagnostic_geometry.update(scene_dx_pixels=float(dx), scene_dy_pixels=float(dy),
                estimated_camera_step=-float(dx)/w, estimated_speed=abs(float(dx)/w)/dt,
                roll_degrees=float(roll), local_scales=scales.tolist(),
                grid_cells=len(cells), frame_width=w)
        # Reject folds/poles, scale/shear and roll, not normal perspective yaw.
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        denominators = corners @ (matrix[2] / matrix[2, 2])
        reason = ''
        if (not np.isfinite(projected).all() or np.any(denominators < .75) or
                np.any(denominators > 1.25) or np.linalg.det(jacobian) <= 0):
            reason = 'unstable projective transform'
        elif np.any(abs(scales-1) > settings.max_scale_change):
            reason = 'scale or shear change'
        elif roll > settings.max_roll_degrees:
            reason = 'camera roll'
        elif abs(dy) > max(.02*h, .5*abs(dx)):
            reason = 'vertical-dominated motion'
        if reason:
            return self._result(frame, session_id, MotionKind.UNKNOWN, reason=reason,
                                inliers=inliers, features=count)
        step = -float(dx)/w  # Camera left -> scene right -> negative camera offset.
        speed = abs(step)/dt
        if abs(step) > settings.max_step or speed > settings.max_speed:
            return self._result(frame, session_id, MotionKind.TOO_FAST, reason='motion exceeds reliable scan speed',
                                inliers=inliers, features=count)
        # Keep the original nominal per-sample displacement deadband. Hold a
        # verified reference briefly so slow motion can exceed that noise floor;
        # do not discard a real small step at every sample forever.
        stationary = abs(step) < settings.stationary_speed*settings.sample_period
        if not stationary or dt >= max(settings.sample_period, settings.max_pair_gap-settings.sample_period):
            self._previous = (gray, frame.captured_at, frame.frame_id)
        kind = MotionKind.STATIONARY if stationary else (MotionKind.LEFT if step < 0 else MotionKind.RIGHT)
        return self._result(frame, session_id, kind, step=0.0 if kind is MotionKind.STATIONARY else step,
                            inliers=inliers, features=count)


class CameraMotionWorker:
    """Independent, sampled consumer of the same latest-frame slot as inference."""

    def __init__(self, buffer: LatestFrameBuffer, sessions: SessionManager, session: SearchSession,
                 settings=MotionSettings(), *, clock=time.monotonic):
        self.buffer, self.sessions, self.session = buffer, sessions, session
        self.settings, self.clock = settings, clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._latest = None
        self._error = None
        self._finished = threading.Event()

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def finished(self):
        return self._finished.is_set()

    def start(self) -> None:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError('motion worker is single-use')
        thread = threading.Thread(target=self._run, name='camera-motion')
        try:
            thread.start()
        except Exception as error:
            self._error = error
            self._stop.set()
            self._finished.set()
            raise
        self._thread = thread

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        with self._lock:
            self._latest = None
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def latest(self) -> MotionEvidence | None:
        with self._lock:
            value = self._latest
        if (value is not None and not self._stop.is_set() and self.sessions.is_eligible(self.session.session_id)
                and 0 <= self.clock()-value.captured_at < self.settings.max_age):
            return value
        return None

    def _run(self):
        try:
            estimator = CameraMotionEstimator(self.settings)
            last_id, next_sample = 0, self.clock()
            while not self._stop.is_set() and self.sessions.is_eligible(self.session.session_id):
                if self._stop.wait(max(0.0, next_sample-self.clock())):
                    break
                frame = self.buffer.wait_for_newer(last_id, timeout=.1)
                if self.buffer.closed:
                    break
                if frame is None:
                    continue
                last_id = frame.frame_id
                next_sample = self.clock()+self.settings.sample_period
                if frame.captured_at < self.session.started_at or not 0 <= self.clock()-frame.captured_at < self.settings.max_age:
                    continue
                started = time.monotonic()
                value = replace(estimator.observe(frame, self.session.session_id), processing_seconds=time.monotonic()-started)
                if (self.sessions.is_eligible(self.session.session_id) and not self._stop.is_set()
                        and 0 <= self.clock()-frame.captured_at < self.settings.max_age):
                    with self._lock:
                        self._latest = value
                else:
                    diagnostic.event('motion', 'motion_handoff_rejected', self.session.session_id,
                        frame_id=frame.frame_id, reason='session_stopped_or_frame_stale',
                        frame_age=self.clock()-frame.captured_at, max_age=self.settings.max_age)
        except Exception as error:
            with self._lock:
                self._error = error
        finally:
            with self._lock:
                self._latest = None
            self._finished.set()
