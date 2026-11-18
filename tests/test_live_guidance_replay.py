"""Replay consented diagnostic metadata, never physical imagery or hardware."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
from homesense.application import ApplicationSnapshot, ApplicationState, SearchOutcome
from homesense.audio.speech_output import SpeechValidity
from homesense.camera import Frame
from homesense.detector import Detection
from homesense.guidance import GuidanceProcessor
from homesense.interaction.feedback_manager import FeedbackManager
from homesense.motion import MotionEvidence, MotionKind
from homesense.perception import DetectionBatch
from homesense.search_strategy import SearchStrategyController, SearchPhase
from homesense.session import SessionManager
from homesense.stability import StabilityProcessor
from test_voice_interaction import Clock

TRACE = Path(__file__).parent/'fixtures/live_phone_20261118.json'


def replay():
    """Actual stability/guidance/controller/feedback on original published observations.

    Retains input timestamps, classes, confidences, boxes and motion metadata.
    It does not reproduce native scheduling or infer image motion from metadata.
    """
    trace = json.loads(TRACE.read_text())
    clock = Clock()
    sessions = SessionManager(clock=clock)
    session = sessions.start('phone', trace['timeout_seconds'])
    stability = StabilityProcessor(sessions, clock=clock)
    guidance = GuidanceProcessor(sessions, clock=clock)
    controller = SearchStrategyController(session)
    messages = []
    output = Mock()
    output.submit.side_effect = lambda message: messages.append(message) or True
    feedback = FeedbackManager(output, SpeechValidity(clock=clock), clock=clock)
    image = np.zeros(trace['image_shape'], dtype=np.uint8)
    results = []
    history = None
    for item in trace['samples']:
        clock.now = session.started_at+item['at']
        frame = Frame(item['frame_id'], session.started_at+item['captured_at'], image)
        # Numeric IDs are immaterial here: canonical names are original log data.
        ds = tuple(Detection(-1, d['class_name'], d['confidence'], tuple(d['box'])) for d in item['detections'])
        batch = DetectionBatch(session.session_id, frame, clock(), 0, ds)
        stability.observe(batch)
        view = stability.snapshot()
        guidance.update(batch, view)
        guide = guidance.latest()
        m = item['motion']
        motion = (MotionEvidence(session.session_id, m['frame_id'], clock()-m['age'],
                  MotionKind(m['kind']), m['epoch'], m['offset']) if m['kind'] else None)
        scan = controller.update(session, batch, view, guide, motion, clock())
        if scan.status.value == 'target_visible':
            history = guide
        snapshot = ApplicationSnapshot(ApplicationState.SEARCHING, session, None, None, (), False,
            batch, view, guide, motion=motion, scan=scan, last_target=history)
        feedback.on_snapshot(snapshot)
        results.append((item['at'], scan, snapshot))
    clock.now = session.deadline+.0074
    terminal = ApplicationSnapshot(ApplicationState.IDLE, sessions.current, SearchOutcome.TIMED_OUT,
        session.session_id, (), True, last_target=history)
    feedback.on_snapshot(terminal)
    return results, messages, terminal


class LoggedPhoneReplayTests(unittest.TestCase):
    def test_original_acquisitions_reproduced_and_timeout_remembers_fresh_guidance(self):
        results, messages, terminal = replay()
        acquisitions = [at for i, (at, scan, _) in enumerate(results)
                        if scan.phase is SearchPhase.TARGET_ACQUIRED
                        and (i == 0 or results[i-1][1].phase is not SearchPhase.TARGET_ACQUIRED)]
        self.assertEqual(len(acquisitions), 2)
        self.assertAlmostEqual(acquisitions[0], 33.7446, delta=.01)
        self.assertAlmostEqual(acquisitions[1], 54.002, delta=.01)
        self.assertEqual(results[-1][2].guidance.direction.value, 'right')
        self.assertEqual(results[-1][2].batch.frame.frame_id, 1352)
        self.assertIsNotNone(terminal.last_target)
        self.assertIn('located your phone during this search', messages[-1].text)
        self.assertFalse(any("couldn't locate" in m.text for m in messages))
        self.assertIsNone(terminal.guidance)

    def test_log_loss_no_longer_restarts_left_scan_using_original_observation_clock(self):
        results, messages, _ = replay()
        # The original hardware log restarted left at 41.714 s. Corrected replay
        # freezes scanning and gives one recovery opportunity instead.
        during_loss = [(at, scan) for at, scan, _ in results if 40 < at < 50]
        self.assertTrue(any(scan.phase is SearchPhase.TARGET_LOST for _, scan in during_loss))
        self.assertFalse(any(scan.phase is SearchPhase.SCAN_LEFT for _, scan in during_loss))
        self.assertEqual(sum('lost sight' in m.text for m in messages), 1)
        self.assertEqual(sum('Phone on your right.' == m.text for m in messages), 2)
        restarts = [at for i, (at, scan, _) in enumerate(results)
                    if i and scan.phase is SearchPhase.SCAN_LEFT
                    and results[i-1][1].phase is SearchPhase.OBSERVE_CURRENT_VIEW and at > 34]
        self.assertTrue(all(at >= 52 for at in restarts))
