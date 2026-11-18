"""Opt-in, bounded metadata logging. Producers never serialize or touch files."""

from collections import Counter, deque
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
from uuid import UUID, uuid4
from itertools import islice

_active = None


def enabled():
    return _active is not None


def event(component, kind, session_id=None, *, important=True, severity='INFO', **data):
    sink = _active
    if sink is not None:
        sink.emit(component, kind, session_id, important=important, severity=severity, **data)


def _metadata(value, depth=0):
    """Copy a bounded JSON-safe payload; never retain image/PCM/native objects."""
    if depth > 5:
        return '<depth limit>'
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:2048]
    if isinstance(value, (UUID, Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, dict):
        return {str(k)[:80]: _metadata(v, depth+1) for k, v in islice(value.items(), 64)}
    if isinstance(value, (tuple, list)):
        return [_metadata(v, depth+1) for v in value[:32]]
    return '<omitted '+type(value).__name__+'>'


class _LogBridge(logging.Handler):
    def __init__(self, sink):
        super().__init__(logging.INFO)
        self.sink = sink

    def emit(self, record):
        # Native Vosk stderr is intentionally not redirected process-wide.
        self.sink.emit(record.name, 'log', severity=record.levelname,
                       important=record.levelno >= logging.WARNING,
                       message=record.getMessage(),
                       source_file=record.pathname, source_line=record.lineno,
                       exception=str(record.exc_info[1]) if record.exc_info else None)


class DiagnosticRun:
    """One writer, two bounded lanes; critical events reserve queue capacity.

    Ordering follows enqueue sequence across both lanes. Overload drops telemetry
    first; even critical loss is explicitly counted. Disk failure cannot kill an
    application worker. close() drains with a bounded wait and reports failure.
    """

    def __init__(self, directory='outputs/logs', *, capacity=1024, reserve=128,
                 clock=time.monotonic):
        if capacity < 4 or not 1 <= reserve < capacity:
            raise ValueError('invalid diagnostic queue bounds')
        self.clock, self.started = clock, clock()
        self.run_id = uuid4().hex
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime('homesense-%Y%m%dT%H%M%S.%fZ-')+self.run_id[:12]
        self.path = directory/(stem+'.jsonl')
        self.summary_path = directory/(stem+'.sessions.jsonl')
        self._stream = self.path.open('x', encoding='utf-8', buffering=1)
        try:
            self._summaries = self.summary_path.open('x', encoding='utf-8', buffering=1)
        except Exception:
            self._stream.close()
            raise
        self.capacity, self.reserve = capacity, reserve
        self._critical, self._telemetry = deque(), deque()
        self._condition = threading.Condition()
        self._stopping = False
        self._sequence = 0
        self._drops = Counter()
        self._total_drops = Counter()
        self.error = None
        self._sessions = {}
        self._bridge = _LogBridge(self)
        self._thread = threading.Thread(target=self._run, name='diagnostic-writer', daemon=True)
        self._close_requested = False
        self._started = False

    def start(self, **settings):
        global _active
        if self._started or self._close_requested:
            raise RuntimeError('diagnostic run is single-use')
        if _active is not None:
            self._stream.close()
            self._summaries.close()
            raise RuntimeError('a diagnostic run is already active')
        try:
            self._thread.start()
        except Exception:
            self._stream.close()
            self._summaries.close()
            raise
        self._started = True
        _active = self
        logging.getLogger().addHandler(self._bridge)
        self.emit('application', 'run_started', settings=settings, path=str(self.path))
        return self

    def emit(self, component, kind, session_id=None, *, important=True, severity='INFO', **data):
        data = _metadata(data)
        with self._condition:
            if self._stopping or self.error:
                return False
            lane = self._critical if important else self._telemetry
            if not important and (len(self._telemetry) >= self.capacity-self.reserve or
                                  len(self._critical)+len(self._telemetry) >= self.capacity):
                if self._telemetry:
                    self._telemetry.popleft()
                else:
                    self._drop('telemetry')
                    return False
                self._drop('telemetry')
            elif len(self._critical)+len(self._telemetry) >= self.capacity:
                victim = self._telemetry if self._telemetry else self._critical
                self._drop('telemetry' if self._telemetry else 'critical')
                victim.popleft()
            self._sequence += 1
            lane.append(dict(sequence=self._sequence, timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=max(0., self.clock()-self.started), run_id=self.run_id,
                session_id=str(session_id) if session_id else None, component=component,
                event=kind, severity=severity, thread=threading.current_thread().name, data=data))
            self._condition.notify()
        return True

    def _drop(self, lane):
        self._drops[lane] += 1
        self._total_drops[lane] += 1

    @property
    def queued(self):
        with self._condition:
            return len(self._critical)+len(self._telemetry)

    def _write(self, record):
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')

    def _aggregate(self, record):
        sid, kind, data = record['session_id'], record['event'], record['data']
        if kind == 'run_finished':
            for saved_sid, summary in self._sessions.items():
                if 'ending' in summary:
                    self._write_summary(dict(record, session_id=saved_sid), summary, finalized=True)
        if sid is None:
            return
        if sid not in self._sessions:
            if len(self._sessions) >= 16:
                # Summaries are incremental snapshots, not retained run histories.
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[sid] = dict(target=None, phases=[], phase_transitions=0,
                motion_samples=0, reliable_motion_samples=0, motion_rejections={},
                progress={leg: dict(max_observed=0., last_credited=0., completed=False, attempted=False)
                          for leg in ('left', 'right')},
                retries=0, recovery_attempts=0, target_result_samples=0, max_target_confidence=None,
                speech_events={}, recent_speech=[], errors=[], performance=None)
        summary = self._sessions[sid]
        if kind == 'session_started':
            summary['target'] = data.get('target')
        elif kind == 'motion_evidence':
            summary['motion_samples'] += 1
            summary['reliable_motion_samples'] += int(data.get('reliable', False))
            if not data.get('reliable'):
                reason = data.get('reason') or 'unknown'
                counts = summary['motion_rejections']
                reason = reason if reason in counts or len(counts) < 24 else 'other'
                counts[reason] = counts.get(reason, 0)+1
        elif kind == 'search_decision':
            phase = data.get('phase')
            if not summary['phases'] or summary['phases'][-1] != phase:
                summary['phase_transitions'] += 1
                summary['phases'] = (summary['phases']+[phase])[-32:]
            direction = data.get('direction')
            if direction:
                progress = summary['progress'].setdefault(direction, dict(max_observed=0., last_credited=0., completed=False))
                progress['attempted'] = True
                progress['max_observed'] = max(progress['max_observed'], data['progress'])
                progress['last_credited'] = data['progress']
                progress['completed'] |= data.get('leg_completed', False)
            for leg in ('left', 'right'):
                if data.get('completed_'+leg) or data.get('input_leg_direction') == leg:
                    progress = summary['progress'].setdefault(leg, dict(max_observed=0., last_credited=0., completed=False))
                    progress['completed'] |= data.get('completed_'+leg, False)
                    if data.get('input_leg_direction') == leg:
                        progress['max_observed'] = max(progress['max_observed'], data.get('input_leg_progress', 0.))
            summary['retries'] = max(summary['retries'], data.get('retry', 0))
            if data.get('status') == 'recovering' and data.get('previous_status') != 'recovering':
                summary['recovery_attempts'] += 1
        elif kind == 'detection_sample':
            detections = data.get('detections', [])
            if detections:
                summary['target_result_samples'] += 1
                maximum = max(d['confidence'] for d in detections)
                summary['max_target_confidence'] = max(summary['max_target_confidence'] or 0., maximum)
        elif kind.startswith('speech_'):
            counts = summary['speech_events']
            counts[kind] = counts.get(kind, 0)+1
            if kind in ('speech_started', 'speech_finished', 'speech_cancelled', 'speech_failed'):
                summary['recent_speech'] = (summary['recent_speech']+[dict(event=kind, **data)])[-6:]
        elif kind == 'performance':
            summary['performance'] = data
        if record['severity'] in ('ERROR', 'CRITICAL'):
            summary['errors'] = (summary['errors']+[data])[-8:]
        if kind == 'session_finished':
            summary['ending'] = data
        if 'ending' in summary and (kind in ('session_finished', 'speech_finished', 'speech_cancelled', 'speech_failed')
                                   or record['severity'] in ('ERROR', 'CRITICAL')):
            self._write_summary(record, summary)

    def _write_summary(self, record, summary, *, finalized=False):
        summary['summary_revision'] = summary.get('summary_revision', 0)+1
        payload = {k: v for k, v in summary.items() if k != 'ending'}
        payload.update(summary['ending'])
        payload['errors'] = (summary.get('errors', [])+summary['ending'].get('errors', []))[-16:]
        payload.update(observation_scope='software metadata; not physical movement or exhaustive coverage',
                       phase_history_truncated=summary['phase_transitions'] > len(summary['phases']),
                       log_drops_so_far=dict(self._total_drops), finalized_at_run_shutdown=finalized)
        output = dict(record, event='session_summary', data=payload)
        self._write(output)
        self._summaries.write(json.dumps(output, ensure_ascii=False, allow_nan=False)+'\n')

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._critical or self._telemetry)
                    drops, self._drops = self._drops, Counter()
                    if not self._critical and not self._telemetry:
                        if self._stopping:
                            break
                        continue
                    lane = (self._critical if not self._telemetry or self._critical and
                            self._critical[0]['sequence'] < self._telemetry[0]['sequence'] else self._telemetry)
                    record = lane.popleft()
                if drops:
                    self._write(dict(record, event='logging_dropped', component='diagnostics', severity='WARNING',
                                     session_id=None, data=dict(drops)))
                if record['event'] == 'run_finished':
                    self._aggregate(record)
                    self._write(record)
                else:
                    self._write(record)
                    self._aggregate(record)
            self._stream.flush()
            self._summaries.flush()
        except Exception as error:
            self.error = str(error)
            print('HomeSense diagnostic writer failed: '+str(error), file=sys.__stderr__, flush=True)
        finally:
            self._stream.close()
            self._summaries.close()

    def close(self, *, exit_code=0, timeout=3.0):
        global _active
        if not self._started:
            self._stream.close()
            self._summaries.close()
            self._close_requested = True
            return True
        if not self._close_requested:
            self._close_requested = True
            self.emit('application', 'run_finished', exit_code=exit_code, dropped=dict(self._total_drops))
            logging.getLogger().removeHandler(self._bridge)
            if _active is self:
                _active = None
            with self._condition:
                self._stopping = True
                self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            print('HomeSense diagnostic flush timed out: '+str(self.path), file=sys.__stderr__, flush=True)
        return not self._thread.is_alive() and self.error is None
