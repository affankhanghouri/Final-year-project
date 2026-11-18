"""Periodic process/worker measurements; no audio, camera or inference ownership."""

import logging
import time
import resource
from homesense.diagnostic_logging import event


class RuntimeMetrics:
    """CPU uses process seconds / wall seconds; 100% means one busy logical core.

    Rates are observed publication counts, not GUI refresh or claimed hardware
    capability. Sampling never opens a device and retains no images/audio.
    """

    def __init__(self, interval=5.0, *, clock=time.monotonic, cpu_clock=time.process_time):
        self.interval, self.clock, self.cpu_clock = interval, clock, cpu_clock
        self._at, self._cpu = clock(), cpu_clock()
        self._session = None
        self._capture = (0, 0.0)

    def sample(self, runtime, preview=None):
        now = self.clock()
        if now - self._at < self.interval:
            return None
        cpu = self.cpu_clock()
        snapshot = runtime.interaction.snapshot
        session = snapshot.session
        sid = session.session_id if session else None
        if sid != self._session:
            self._session = sid
            self._capture = (0, session.started_at if session else now)
        app = runtime.interaction.coordinator
        frame = app.latest_frame() if app is not None else None
        fps = 0.0
        if frame is not None:
            previous_id, previous_time = self._capture
            elapsed = frame.captured_at - previous_time
            fps = (frame.frame_id - previous_id) / elapsed if elapsed > 0 else 0.0
            self._capture = (frame.frame_id, frame.captured_at)
        speech = runtime.speech.stats
        stats = snapshot.stats
        values = dict(cpu_percent=100 * max(0.0, cpu - self._cpu) / (now - self._at),
                      memory_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      camera_fps=fps, inference_fps=stats.processed_fps,
                      inference_ms=1000 * stats.last_inference_seconds,
                      capture_to_result_ms=1000 * stats.last_result_age,
                      rejected_results=stats.rejected_results, stale_frames=stats.stale_frames,
                      tts_pending=speech.pending, tts_started=speech.started,
                      tts_cancelled=speech.cancelled, tts_replaced=speech.replaced,
                      tts_queue_ms=1000 * speech.queue_delay_seconds,
                      tts_playback_ms=1000 * speech.playback_seconds,
                      preview_ms=1000 * preview.render_seconds if preview else 0.0)
        if snapshot.motion is not None:
            values['motion_ms'] = 1000*snapshot.motion.processing_seconds
            values['motion_support'] = snapshot.motion.support_fraction
            values['motion_reliable'] = int(snapshot.motion.reliable)
            values['motion_direction'] = snapshot.motion.kind.value
            values['motion_reason'] = snapshot.motion.reason or 'accepted'
            values['motion_tracks'] = snapshot.motion.features
            values['motion_inliers'] = snapshot.motion.inliers
            values['motion_pair_ms'] = 1000*snapshot.motion.pair_seconds
        if snapshot.scan is not None:
            values['scan_progress'] = snapshot.scan.scan.progress
            values['scan_idle_s'] = snapshot.scan.scan.inactive_seconds
            values['scan_effort_s'] = snapshot.scan.scan.leg_seconds
            values['scan_response_s'] = snapshot.scan.scan.response_seconds
            values['scan_uncertain_s'] = snapshot.scan.scan.uncertain_seconds
            values['scan_retry'] = snapshot.scan.scan.retry
            values['scan_decision'] = snapshot.scan.reason or snapshot.scan.status.value
            values['scan_phase'] = snapshot.scan.phase.value
            values['target_recovery_s'] = snapshot.scan.recovery_seconds
            values['target_recovery_id'] = snapshot.scan.recovery_id
            if snapshot.motion is None:
                values.update(motion_direction='unknown', motion_reliable=0, motion_reason='no fresh motion')
        if snapshot.last_target is not None:
            values['last_target_age_s'] = max(0., now-snapshot.last_target.captured_at)
        self._at, self._cpu = now, cpu
        formatted = ' '.join(f'{k}={v!r}' if isinstance(v, str) else f'{k}={v:.2f}' for k, v in values.items())
        logging.getLogger(__name__).info('Performance: %s', formatted)
        event('metrics', 'performance', sid, important=False, **values)
        return values
