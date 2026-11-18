"""Bounded diagnostics and real module integration using simulated hardware only."""

from dataclasses import replace
import json
import logging
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock
from uuid import uuid4

import numpy as np

from homesense import diagnostic_logging as diagnostic
from homesense.diagnostic_logging import DiagnosticRun
from homesense.audio.speech_output import SpeechWorker, SpeechValidity, SpeechMessage
from homesense.audio.input import PlaybackGate
from homesense.camera import Frame
from homesense.motion import CameraMotionEstimator, MotionKind
from homesense.search_strategy import ScanSettings, SearchStrategyController, SearchPhase
from test_search_strategy import StrategyFixture
from test_guided_recovery import furnished_view, pan_view


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class LogFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = DiagnosticRun(self.directory.name).start(test='fake hardware')
        self.addCleanup(self.run.close)

    def finish(self):
        self.assertTrue(self.run.close())
        return records(self.run.path)


class DiagnosticWriterTests(LogFixture):
    def test_envelope_session_correlation_order_and_file_preservation(self):
        sid = uuid4()
        diagnostic.event('test', 'one', sid, target='phone')
        diagnostic.event('test', 'two', sid, important=False, direction='left')
        result = self.finish()
        selected = [r for r in result if r['component'] == 'test']
        self.assertEqual([r['event'] for r in selected], ['one', 'two'])
        self.assertTrue(all(r['session_id'] == str(sid) for r in selected))
        self.assertTrue(all(k in selected[0] for k in ('timestamp', 'elapsed_seconds', 'run_id', 'severity', 'thread')))
        self.assertEqual([r['sequence'] for r in result], sorted(r['sequence'] for r in result))
        content = self.run.path.read_bytes()
        second = DiagnosticRun(self.directory.name).start()
        self.assertTrue(second.close())
        self.assertNotEqual(self.run.path, second.path)
        self.assertEqual(self.run.path.read_bytes(), content)

    def test_important_events_flush_before_close_and_close_is_repeatable(self):
        diagnostic.event('test', 'important')
        deadline = time.monotonic()+1
        while 'important' not in self.run.path.read_text():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.002)
        self.assertTrue(self.run.close())
        self.assertTrue(self.run.close())
        self.assertEqual(records(self.run.path)[-1]['event'], 'run_finished')
        self.assertFalse(diagnostic.enabled())

    def test_no_images_audio_or_unbounded_native_objects_retained(self):
        diagnostic.event('test', 'metadata_guard', image=np.ones((10, 10)), audio=b'private samples',
                         many=list(range(1000)), huge='x'*10000)
        row = next(r for r in self.finish() if r['event'] == 'metadata_guard')
        self.assertEqual(row['data']['image'], '<omitted ndarray>')
        self.assertNotIn('private samples', json.dumps(row))
        self.assertEqual(len(row['data']['many']), 32)
        self.assertEqual(len(row['data']['huge']), 2048)

    def test_python_errors_are_preserved(self):
        with self.assertLogs('homesense.test', level='ERROR'):
            # assertLogs temporarily replaces that logger's handlers/propagation;
            # use root for the independent diagnostic bridge.
            logging.getLogger('homesense.test').error('device detail')
        logging.getLogger().error('diagnostic error detail')
        row = next(r for r in self.finish() if r['event'] == 'log')
        self.assertEqual(row['severity'], 'ERROR')
        self.assertIn('error detail', row['data']['message'])

    def test_slow_disk_does_not_block_producers_and_loss_is_reported(self):
        self.finish()
        run = DiagnosticRun(self.directory.name, capacity=8, reserve=2)
        entered, release = threading.Event(), threading.Event()
        original = run._write
        def blocked(record):
            entered.set()
            release.wait(2)
            original(record)
        run._write = blocked
        run.start()
        self.addCleanup(run.close)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        for index in range(1000):
            run.emit('test', 'sample', important=False, index=index)
        run.emit('test', 'critical_transition')
        self.assertLess(time.monotonic()-started, 1)
        self.assertLessEqual(run.queued, 8)
        release.set()
        self.assertTrue(run.close())
        result = records(run.path)
        self.assertTrue(any(r['event'] == 'critical_transition' for r in result))
        self.assertTrue(any(r['event'] == 'logging_dropped' for r in result))
        self.assertGreater(result[-1]['data']['dropped']['telemetry'], 0)

    def test_write_failure_is_visible_and_does_not_raise_in_worker(self):
        self.finish()
        run = DiagnosticRun(self.directory.name)
        run._write = Mock(side_effect=OSError('disk full'))
        run.start()
        self.addCleanup(run.close)
        run.emit('test', 'survives')
        self.assertFalse(run.close())
        self.assertIn('disk full', run.error)

    def test_critical_overload_is_counted_and_close_timeout_is_explicit(self):
        self.finish()
        run = DiagnosticRun(self.directory.name, capacity=4, reserve=1)
        entered, release = threading.Event(), threading.Event()
        original = run._write
        def blocked(record):
            entered.set()
            release.wait(2)
            original(record)
        run._write = blocked
        run.start()
        self.addCleanup(run.close)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(1))
        for _ in range(20):
            run.emit('test', 'transition')
        self.assertLessEqual(run.queued, 4)
        self.assertFalse(run.close(timeout=.001))
        release.set()
        self.assertTrue(run.close())
        self.assertGreater(records(run.path)[-1]['data']['dropped']['critical'], 0)


class ModuleDiagnosticTests(LogFixture):
    def test_target_recovery_and_truthful_timeout_are_retained_in_diagnostics(self):
        from integration.test_guided_search import GuidedJourneyTests
        for name in ('test_loss_recovery_interrupts_real_speech_then_reacquires_without_rescanning',
                     'test_visible_at_deadline_ends_truthfully_and_releases_camera'):
            result = unittest.TestResult()
            GuidedJourneyTests(name).run(result)
            self.assertTrue(result.wasSuccessful(), result.errors+result.failures)
        result = self.finish()
        decisions = [r['data'] for r in result if r['event'] == 'search_decision']
        self.assertTrue(any(d['phase'] == 'target_lost' and d['last_target_at'] is not None
                            and d['target_recovery_wall_limit'] == 12 for d in decisions))
        summaries = records(self.run.summary_path)
        timeout = next(r['data'] for r in summaries if r['data']['outcome'] == 'timed_out')
        self.assertTrue(timeout['ever_acquired'])
        self.assertEqual(timeout['last_target_direction'], 'left')
        self.assertIsNotNone(timeout['last_target_at'])
        self.assertTrue(any('located your phone during this search' in r['data'].get('text', '')
                            for r in result if r['event'] == 'speech_started'))

    def test_vosk_quiet_setting_is_diagnostic_only_and_does_not_change_grammar(self):
        from homesense.audio.speech_recognition import VoskRecognizer
        model = Path(self.directory.name)/'model'
        (model/'am').mkdir(parents=True)
        (model/'am'/'final.mdl').touch()
        vosk = Mock()
        with patch.dict('sys.modules', {'vosk': vosk}):
            VoskRecognizer(model, grammar=('find my phone', '[unk]'))
            vosk.SetLogLevel.assert_called_once_with(-1)
            self.assertEqual(json.loads(vosk.KaldiRecognizer.call_args.args[2]), ['find my phone', '[unk]'])
            self.finish()
            vosk.reset_mock()
            VoskRecognizer(model)
            vosk.SetLogLevel.assert_not_called()

    def test_native_motion_rejection_and_correct_direction_values(self):
        estimator, sid = CameraMotionEstimator(), uuid4()
        image = furnished_view()
        estimator.observe(Frame(1, 100, image), sid)
        estimator.observe(Frame(2, 100.15, pan_view(image, 2)), sid)
        estimator.observe(Frame(3, 100.30, np.zeros_like(image)), sid)
        values = [r['data'] for r in self.finish() if r['event'] == 'motion_evidence']
        self.assertEqual(values[1]['direction'], 'left')
        self.assertGreaterEqual(values[1]['support'], .75)
        self.assertLess(values[1]['estimated_camera_step'], 0)
        self.assertGreater(values[1]['scene_dx_pixels'], 0)
        self.assertEqual(values[1]['reference_frame_id'], 1)
        self.assertEqual(values[2]['reason'], 'blur or low texture')
        self.assertFalse(values[2]['reliable'])
        self.assertEqual(values[2]['credited_step'], 0)

    def test_speech_queue_replacement_rejection_and_error_are_correlated(self):
        sid = uuid4()
        validity = SpeechValidity()
        validity.update(0, 'left', time.monotonic()+10)
        backend = Mock()
        backend.speak.side_effect = RuntimeError('speaker unavailable')
        speech = SpeechWorker(backend, validity, PlaybackGate(0), listening_gap=0)
        self.addCleanup(speech.stop)
        old = SpeechMessage('old', 0, 'left', session_id=sid)
        speech.submit(old)
        speech.submit(SpeechMessage('new', 0, 'left', session_id=sid))
        speech.submit(SpeechMessage('obsolete', -1, session_id=sid))
        speech.start()
        deadline = time.monotonic()+1
        while speech.error is None:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.002)
        speech.stop()
        values = [r for r in self.finish() if r['component'] == 'speech']
        kinds = {r['event'] for r in values}
        self.assertTrue({'speech_queued', 'speech_replaced', 'speech_rejected', 'speech_started', 'speech_failed', 'speech_error'} <= kinds)
        self.assertTrue(all(r['session_id'] == str(sid) for r in values))
        self.assertTrue(any('playback_seconds' in r['data'] for r in values))

    def test_simulated_preview_headless_recovery_timeout_replacement_and_cleanup(self):
        from integration.test_guided_search import GuidedJourneyTests
        for name in ('test_voice_left_right_exhaustion_spoken_result_cleanup_and_new_search',
                     'test_live_failure_recovers_after_delayed_user_movement_then_finishes_both_legs',
                     'test_search_timeout_remains_distinct_from_exhaustion'):
            result = unittest.TestResult()
            GuidedJourneyTests(name).run(result)
            self.assertTrue(result.wasSuccessful(), result.errors+result.failures)
        result = self.finish()
        kinds = {r['event'] for r in result}
        self.assertTrue({'camera_ready', 'camera_released', 'parser_decision', 'search_decision',
                         'session_summary', 'resource_cleanup'} <= kinds)
        summaries = records(self.run.summary_path)
        self.assertTrue(any(r['data']['outcome'] == 'timed_out' for r in summaries))
        successful_scan = next(r for r in summaries if r['data']['reason'] == 'local_limit')
        self.assertTrue(successful_scan['data']['progress']['left']['completed'])
        self.assertTrue(successful_scan['data']['progress']['right']['completed'])
        self.assertTrue(any(r['data']['recovery_attempts'] for r in summaries))
        self.assertTrue(all(r['session_id'] for r in summaries))
        self.assertTrue(all(r['data']['cleanup_complete'] for r in summaries if r['data']['finalized_at_run_shutdown']))

    def test_active_replacement_and_blocked_cleanup_keep_correct_session_outcomes(self):
        from homesense.application import ApplicationCoordinator
        from integration.test_guided_search import Scene, GuidedWorkerCleanupTests
        scene = Scene()
        app = ApplicationCoordinator(scene, source_factory=scene.source, shutdown_timeout=.3)
        try:
            first = app.start_search('phone', 10).session.session_id
            second = app.start_search('wallet', 10).session.session_id
        finally:
            app.close()
        result = unittest.TestResult()
        GuidedWorkerCleanupTests('test_blocked_motion_refuses_replacement_and_preserves_cleanup_ownership').run(result)
        self.assertTrue(result.wasSuccessful(), result.errors+result.failures)
        self.finish()
        summaries = records(self.run.summary_path)
        self.assertTrue(any(r['session_id'] == str(first) and r['data']['outcome'] == 'replaced' for r in summaries))
        self.assertFalse(any(r['session_id'] == str(second) and r['data']['outcome'] == 'replaced' for r in summaries))
        self.assertTrue(any(r['data']['outcome'] == 'failed' and r['data']['errors'] for r in summaries))


class StrategyDiagnosticTests(StrategyFixture):
    def test_speech_pause_retry_and_timeout_inputs_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = DiagnosticRun(directory).start()
            try:
                self.settings = ScanSettings(initial_observe_seconds=.4)
                self.strategy = SearchStrategyController(self.session, self.settings)
                self.begin()
                snap = self.factory.make()
                self.clock.now += .1
                self.strategy.update(self.session, None, None, None, None, self.clock(), instruction_busy=True)
                for _ in range(45):
                    self.clock.now += .25
                    self.update(kind=MotionKind.UNKNOWN)
                self.clock.now = self.session.deadline
                self.strategy.update(self.session, None, None, None, None, self.clock())
            finally:
                run.close()
            decisions = [r['data'] for r in records(run.path) if r['event'] == 'search_decision']
            self.assertTrue(any(d['timer_pause_reason'] == 'speech_pending_or_playing' and not d['timers_running'] for d in decisions))
            self.assertTrue(any(d['status'] == 'recovering' and d['retry'] == 1 for d in decisions))
            self.assertEqual(decisions[-1]['remaining_session_seconds'], 0)
            self.assertEqual(decisions[-1]['phase'], 'inactive')


class DiagnosticCLITests(unittest.TestCase):
    def test_flag_wires_logging_for_preview_and_headless_without_hardware(self):
        from scripts import run_homesense
        for preview in (False, True):
            runtime = Mock(failed=False)
            runtime.step.side_effect = KeyboardInterrupt
            fake = Mock()
            fake.start.return_value = fake
            args = ['run_homesense.py', '--diagnostics', '--text', '--console-feedback']
            if preview:
                args.append('--preview')
            with patch('sys.argv', args), patch.object(run_homesense, 'DiagnosticRun', return_value=fake), \
                 patch.object(run_homesense, 'compose', return_value=runtime), \
                 patch('homesense.preview.LaptopPreview') as window:
                self.assertEqual(run_homesense.main(), 0)
                fake.start.assert_called_once()
                fake.close.assert_called_once_with(exit_code=0)
                self.assertEqual(window.call_count, int(preview))
