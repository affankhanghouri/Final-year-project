"""Offline Phase 3 experiment. Replays existing WAVs; never opens audio or cameras.

Run from the repository root with the editable package installed. No experiment
policy in this file is imported by the production runtime.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import time

from homesense.audio.comparison import load_manifest, normalize, pcm_stats, word_errors
from homesense.audio.events import VoiceEvent, VoiceEventKind
from homesense.audio.wake_word import WakePhraseGate
from homesense.commands.models import Intent
from homesense.commands.parser import parse_command

if __package__:
    from .speech_phase2_investigation import EXPECTED, grammar_for, replay, role
else:
    from speech_phase2_investigation import EXPECTED, grammar_for, replay, role


def wake_normalize(text):
    """Only alias a complete heyagent token; no fuzzy or substring matching."""
    return re.sub(r"\bheyagent\b", "hey agent", normalize(text))


def starts_wake(text):
    value = wake_normalize(text)
    return value == 'hey agent' or value.startswith('hey agent ')


def command_result(text):
    """Preserve unknown markers and all extra words; never cherry-pick fragments."""
    command = parse_command(text)
    accepted = command.intent is not Intent.UNKNOWN and '[unk]' not in text
    return {'intent': command.intent.value, 'target': command.target.value if command.target else None,
            'accepted': accepted}


class SequenceProbe:
    """Camera-free experimental wrapper around the actual wake gate and parser.

    Final command transcripts are passed unchanged. A boundary rule rejects
    embedded/negated wake phrases. Partial hypotheses can arm listening but may
    never dispatch an action; the existing gate rechecks the final hypothesis.
    The action list is a dry-run sink, not an ApplicationCoordinator.
    """

    def __init__(self, timeout=10.0):
        self.gate = WakePhraseGate(timeout=timeout)
        self.actions = []

    def accept(self, text, final, now):
        value = wake_normalize(text)
        embedded = re.search(r'\bhey agent\b', value) and not starts_wake(value)
        if '[unk]' in text or embedded:
            if final:
                was_listening = self.gate.deadline is not None
                self.gate.reset()
                if was_listening:
                    return [VoiceEvent(VoiceEventKind.TIMEOUT, occurred_at=now)]
            return []
        events = self.gate.accept(value, final, now)
        for event in events:
            if event.kind is VoiceEventKind.TRANSCRIPT:
                result = command_result(event.text)
                if result['accepted']:
                    self.actions.append(result)
        return events


# Text-only contract probes, independent of acoustic recognition quality.
TEXT_CASES = (
    ('Hey Agent, find my phone', ('find_object', 'phone')),
    ('heyagent find my phone', ('find_object', 'phone')),
    ('HEY, AGENT! find my phone', ('find_object', 'phone')),
    ('hey-agent find my phone', ('find_object', 'phone')),
    ('hey agents find my phone', None), ('hey Adrian find my phone', None),
    ('hey Aiden find my phone', None), ('hey urgent find my phone', None),
    ('hey engine find my phone', None), ('hi agent find my phone', None),
    ('heyagentphone', None), ('theyagent find my phone', None),
    ('do not hey agent find my phone', None), ('say hey agent find my phone', None),
    ('', None), ('mobile', None), ('find my phone', None),
    ('hey agent find my phone or keys', None), ('hey agent find my glasses', None),
    ('hey agent find', None), ('hey agent find my', None),
    ('hey agent do not cancel', None), ("hey agent don't stop searching", None),
    ('hey agent i have not found it', None), ('hey agent cancel search now maybe', None),
    ('hey agent locate my wallet [unk]', None),
    ('hey agent cancel search cancel search', None),
    ('hey agent repeat the direction', None), ('hey agent help me', None),
    ('hey agent repeat direction', ('repeat_guidance', None)),
    ('hey agent help', ('help', None)),
    ('hey agent please find my mobile', ('find_object', 'phone')),
    ('hey agent where are my keys', ('find_object', 'keys')),
    ("hey agent I've got it", ('confirm_found', None)),
)


def text_cases():
    rows = []
    for text, expected in TEXT_CASES:
        probe = SequenceProbe()
        probe.accept(text, True, 0)
        actual = [(a['intent'], a['target']) for a in probe.actions]
        rows.append({'text': text, 'expected': expected, 'actions': actual,
                     'passed': actual == ([expected] if expected else [])})
    return rows


def score_commands(rows, *, bare_only=False):
    selected = [r for r in rows if not bare_only or not starts_wake(r['reference'])]
    positives = [r for r in selected if role(r['reference']) == 'command']
    negatives = [r for r in selected if role(r['reference']) != 'command']
    def expected(row):
        return EXPECTED[normalize(row['reference'])]
    def correct(row):
        result = command_result(row['transcript'])
        return result['accepted'] and (result['intent'], result['target']) == expected(row)
    return {'positive_clips': len(positives),
            'exact_transcripts': sum(normalize(r['reference']) == normalize(r['transcript']) for r in positives),
            'word_errors': sum(word_errors(r['reference'], r['transcript']) for r in positives),
            'reference_words': sum(len(normalize(r['reference']).split()) for r in positives),
            'correct_intents': sum(correct(r) for r in positives),
            'rejected_positive_clips': sum(not command_result(r['transcript'])['accepted'] for r in positives),
            'wrong_accepted_intents': sum(command_result(r['transcript'])['accepted'] and not correct(r) for r in positives),
            'negative_clips': len(negatives),
            'false_command_accepts': sum(command_result(r['transcript'])['accepted'] for r in negatives),
            'silence_nonempty': sum(role(r['reference']) == 'silence' and bool(r['transcript']) for r in negatives)}


def score_wake(rows):
    positives = [r for r in rows if starts_wake(r['reference'])]
    negatives = [r for r in rows if not starts_wake(r['reference'])]
    def any_match(row):
        return any(starts_wake(s['text']) for s in row['segments'])
    def anywhere(row):
        return bool(re.search(r'\bhey agent\b', wake_normalize(row['transcript'])))
    return {'positive_clips': len(positives), 'negative_clips': len(negatives),
            'final_prefix_true': sum(starts_wake(r['transcript']) for r in positives),
            'final_prefix_false': sum(starts_wake(r['transcript']) for r in negatives),
            'partial_or_final_prefix_true': sum(any_match(r) for r in positives),
            'partial_or_final_prefix_false': sum(any_match(r) for r in negatives),
            'final_anywhere_true': sum(anywhere(r) for r in positives),
            'final_anywhere_false': sum(anywhere(r) for r in negatives)}


def sequence_rows(wake_rows, command_rows):
    """Two-pass buffered-clip simulation, not measured live handover latency.

    Independent decoders get identical complete PCM. Only a final wake prefix
    opens a window. Full command output is then submitted, including any unknown
    token from the wake prefix. No reference-driven cutting or suffix cleanup.
    """
    if [(r['file'], r['sha256']) for r in wake_rows] != [(r['file'], r['sha256']) for r in command_rows]:
        raise ValueError('Sequence inputs must be the identical ordered audio clips')
    result = []
    for wake, command in zip(wake_rows, command_rows):
        probe = SequenceProbe()
        activated = starts_wake(wake['transcript'])
        if activated:
            probe.accept('hey agent', True, 0)
            probe.accept(command['transcript'], True, 0)
        expected = EXPECTED.get(normalize(wake['reference']))
        result.append({'file': wake['file'], 'reference': wake['reference'],
                       'wake_text': wake['transcript'], 'command_text': command['transcript'],
                       'activated': activated, 'actions': probe.actions,
                       'correct': bool(expected and any((a['intent'], a['target']) == expected for a in probe.actions))})
    return result


def energy_sweep(rows, clips):
    """Descriptive in-sample tradeoff only; never selects an operating cutoff."""
    energies = {c.path.name: pcm_stats(c.pcm)['rms_normalized'] for c in clips}
    results = []
    for threshold in (0, .005, .01, .015, .03, .05):
        kept = [r for r in rows if energies[r['file']] >= threshold]
        scores = score_commands(kept, bare_only=True)
        results.append({'exploratory_rms_minimum': threshold, 'kept_clips': len(kept),
                        'correct_bare_commands_retained': scores['correct_intents'],
                        'nonempty_silence_retained': scores['silence_nonempty'],
                        'false_commands_retained': scores['false_command_accepts']})
    return results


def confidence_sweep(rows):
    """Use measured word fields only; alternative path scores are not probabilities."""
    results = []
    for threshold in (0, .5, .7, .9, 1.0):
        kept = [r for r in rows if r['evaluation']['minimum_word_confidence'] is not None
                and r['evaluation']['minimum_word_confidence'] >= threshold]
        scores = score_commands(kept, bare_only=True)
        results.append({'exploratory_word_confidence_minimum': threshold,
                        'correct_bare_commands_retained': scores['correct_intents'],
                        'nonempty_silence_retained': scores['silence_nonempty'],
                        'false_commands_retained': scores['false_command_accepts']})
    return results


def mixed_sequence_rows(rows):
    """Replay actual partial/final events through the text wrapper, with offsets."""
    output = []
    for row in rows:
        probe = SequenceProbe()
        events = []
        for segment in row['segments']:
            events.extend(probe.accept(segment['text'], segment['final'], segment['offset_seconds']))
        output.append({'file': row['file'], 'reference': row['reference'],
                       'actions': probe.actions,
                       'listening_offsets': [e.occurred_at for e in events if e.kind is VoiceEventKind.LISTENING],
                       'transcript_offsets': [e.occurred_at for e in events if e.kind is VoiceEventKind.TRANSCRIPT]})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/speech/vosk-model-small-en-us-0.15'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output path; existing evidence will not be overwritten')
    _, clips = load_manifest(args.manifest)
    for clip in clips:
        role(clip.reference)
    import vosk
    vosk.SetLogLevel(-1)
    started = time.monotonic()
    model = vosk.Model(str(args.model))
    report = {'scope': 'Existing audio replay, batch decoder, dry-run intents, no hardware or runtime',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'model_load_seconds': time.monotonic() - started,
              'model': str(args.model), 'profiles': {}, 'text_cases': text_cases(),
              'heyagent_in_model_vocabulary': model.vosk_model_find_word('heyagent') >= 0,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__).resolve(), Path(__file__).with_name('speech_phase2_investigation.py'),
                   Path('src/homesense/audio/comparison.py'), Path('src/homesense/audio/wake_word.py'),
                   Path('src/homesense/commands/parser.py'))}}
    for name, grammar_name, alternatives in (('wake', 'wake', 0), ('commands', 'commands', 0),
                                              ('commands-nbest', 'commands', 3),
                                              ('mixed', 'mixed', 0), ('mixed-nbest', 'mixed', 3)):
        grammar = grammar_for(grammar_name)
        rows = [replay(model, clip, grammar, alternatives) for clip in clips]
        report['profiles'][name] = {
            'grammar': grammar, 'grammar_sha256': hashlib.sha256(json.dumps(grammar).encode()).hexdigest(),
            'max_alternatives': alternatives, 'words': True,
            'missing_vocabulary': [w for w in sorted(set(' '.join(grammar).split())) if model.vosk_model_find_word(w) < 0],
            'rows': rows, 'bare_command_scores': score_commands(rows, bare_only=True),
            'wake_scores': score_wake(rows),
            'median_decode_seconds': statistics.median(r['decode_wall_seconds'] for r in rows),
            'maximum_decode_seconds': max(r['decode_wall_seconds'] for r in rows),
            'median_cpu_seconds': statistics.median(r['decode_cpu_seconds'] for r in rows)}
        print(name, json.dumps({k: v for k, v in report['profiles'][name].items() if k not in ('rows', 'grammar')}), flush=True)
    report['sequence'] = sequence_rows(report['profiles']['wake']['rows'], report['profiles']['commands']['rows'])
    report['mixed_event_sequence'] = mixed_sequence_rows(report['profiles']['mixed']['rows'])
    report['exploratory_energy_sweep'] = energy_sweep(report['profiles']['commands']['rows'], clips)
    report['exploratory_confidence_sweep'] = confidence_sweep(report['profiles']['commands']['rows'])
    report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['elapsed_seconds'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        os.chmod(args.output, 0o600)
        json.dump(report, output, indent=2)
    print('Evidence:', args.output)


if __name__ == '__main__':
    main()
