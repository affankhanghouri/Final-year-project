"""Isolated Phase 2 replay; no microphone, runtime integration or production edits."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import statistics
import time

from homesense.audio.comparison import grammar_phrases, load_manifest, normalize, word_errors
from homesense.audio.events import VoiceEventKind
from homesense.audio.wake_word import WakePhraseGate
from homesense.commands.models import Intent
from homesense.commands.parser import parse_command

PROFILES = ('mixed', 'commands', 'wake', 'mixed-no-unk', 'mixed-nbest')
# Product intent labels, independent of parser support: UNKNOWN is not a success.
EXPECTED = {
    'hey agent find my phone': ('find_object', 'phone'),
    'find my keys': ('find_object', 'keys'),
    'locate my wallet': ('find_object', 'wallet'),
    'cancel search': ('cancel_search', None),
    'repeat the direction': ('repeat_guidance', None),
    'i found it': ('confirm_found', None),
    'hey agent help me': ('help', None),
}


def role(reference):
    value = normalize(reference)
    if value in EXPECTED:
        return 'command'
    if value == 'please close the window':
        return 'unrelated'
    if not value:
        return 'silence'
    raise ValueError(f'Unlabelled reference: {reference!r}; add explicit ground truth before scoring')


def grammar_for(profile):
    original = grammar_phrases()
    if profile in ('mixed', 'mixed-nbest'):
        return original
    if profile == 'commands':
        return tuple(p for p in original if not p.startswith('hey agent'))
    if profile == 'wake':
        return ('hey agent', '[unk]')
    if profile == 'mixed-no-unk':
        return tuple(p for p in original if p != '[unk]')
    raise ValueError('Unknown experimental profile')


def contains_wake(text):
    return bool(re.search(r'\bhey agent\b', normalize(text)))


def parsed(text):
    value = re.sub(r'^hey agent\s*', '', normalize(text))
    command = parse_command(value)
    return {'intent': command.intent.value, 'target': command.target.value if command.target else None,
            'accepted': command.intent is not Intent.UNKNOWN}


def evaluate(reference, transcript, segments):
    label = role(reference)
    expected = EXPECTED.get(normalize(reference))
    actual = parsed(transcript)
    words = len(normalize(reference).split())
    finals = [s for s in segments if s['final']]
    all_words = [w for segment in finals for w in segment['result'].get('result', [])]
    confidences = [w['conf'] for w in all_words if 'conf' in w]
    gate = WakePhraseGate()
    events = [event for segment in segments
              for event in gate.accept(segment['text'], segment['final'], segment['offset_seconds'])]
    delivered = [event.text for event in events if event.kind is VoiceEventKind.TRANSCRIPT]
    return {'role': label, 'exact': normalize(reference) == normalize(transcript),
            'word_errors': word_errors(reference, transcript), 'reference_words': words,
            'expected_wake': contains_wake(reference), 'final_wake': contains_wake(transcript),
            'partial_or_final_wake': any(event.kind is VoiceEventKind.LISTENING for event in events),
            'delivered_commands': delivered, 'parsed': actual,
            'intended_command_correct': bool(expected and (actual['intent'], actual['target']) == expected),
            'delivered_intended_command': bool(expected and any((parsed(text)['intent'], parsed(text)['target']) == expected for text in delivered)),
            'minimum_word_confidence': min(confidences) if confidences else None,
            'mean_word_confidence': statistics.mean(confidences) if confidences else None,
            'contains_unknown': '[unk]' in transcript,
            'exact_grammar_phrase': transcript in grammar_phrases()}


def summarize(rows):
    commands = [r for r in rows if r['evaluation']['role'] == 'command']
    silence = [r for r in rows if r['evaluation']['role'] == 'silence']
    unrelated = [r for r in rows if r['evaluation']['role'] == 'unrelated']
    positives = [r for r in rows if r['evaluation']['expected_wake']]
    negatives = [r for r in rows if not r['evaluation']['expected_wake']]
    def count(group, field):
        return sum(bool(r['evaluation'][field]) for r in group)
    return {'commands': len(commands), 'command_exact': count(commands, 'exact'),
            'command_word_errors': sum(r['evaluation']['word_errors'] for r in commands),
            'command_reference_words': sum(r['evaluation']['reference_words'] for r in commands),
            'command_intent_target_correct': count(commands, 'intended_command_correct'),
            'wake_positive_clips': len(positives), 'wake_final_true_positives': count(positives, 'final_wake'),
            'wake_final_false_positives': count(negatives, 'final_wake'), 'wake_negative_clips': len(negatives),
            'wake_any_true_positives': count(positives, 'partial_or_final_wake'),
            'wake_any_false_positives': count(negatives, 'partial_or_final_wake'),
            'wake_positive_delivered_correct_command': count(positives, 'delivered_intended_command'),
            'silence_clips': len(silence), 'silence_nonempty': sum(bool(r['transcript']) for r in silence),
            'silence_parser_accepts': sum(r['evaluation']['parsed']['accepted'] for r in silence),
            'unrelated_clips': len(unrelated), 'unrelated_parser_accepts': sum(r['evaluation']['parsed']['accepted'] for r in unrelated),
            'negative_delivered_commands': count([*silence, *unrelated], 'delivered_commands'),
            'median_replay_wall_seconds': statistics.median(r['decode_wall_seconds'] for r in rows)}


def rejection_sweep(rows):
    """Exploratory only: no threshold selection or claim of held-out validation."""
    result = []
    for threshold in (0, .5, .7, .8, .9, .95, 1.0):
        accepted = [r for r in rows if r['evaluation']['minimum_word_confidence'] is not None
                    and r['evaluation']['minimum_word_confidence'] >= threshold]
        strict = [r for r in accepted if r['evaluation']['exact_grammar_phrase']
                  and not r['evaluation']['contains_unknown'] and r['evaluation']['parsed']['accepted']]
        result.append({'threshold': threshold,
            'confidence_only_nonempty_silence': sum(r['evaluation']['role'] == 'silence' and bool(r['transcript']) for r in accepted),
            'confidence_only_parser_accepted_negatives': sum(r['evaluation']['role'] != 'command' and r['evaluation']['parsed']['accepted'] for r in accepted),
            'confidence_only_exact_commands': sum(r['evaluation']['role'] == 'command' and r['evaluation']['exact'] for r in accepted),
            'strict_correct_commands': sum(r['evaluation']['intended_command_correct'] for r in strict),
            'strict_negative_commands': sum(r['evaluation']['role'] != 'command' for r in strict)})
    return result


def native_result(result, offset, final):
    alternatives = result.get('alternatives')
    best = alternatives[0] if alternatives else result
    return {'text': best.get('text' if final else 'partial', ''), 'offset_seconds': offset,
            'final': final, 'result': result}


def replay(model, clip, grammar, alternatives):
    import vosk
    started, cpu = time.monotonic(), time.process_time()
    recognizer = vosk.KaldiRecognizer(model, 16000, json.dumps(grammar))
    recognizer.SetWords(True)
    if alternatives:
        recognizer.SetMaxAlternatives(alternatives)
    segments = []
    last_partial = None
    for offset in range(0, len(clip.pcm), 3200):
        final = bool(recognizer.AcceptWaveform(clip.pcm[offset:offset + 3200]))
        result = json.loads(recognizer.Result() if final else recognizer.PartialResult())
        item = native_result(result, min(offset + 3200, len(clip.pcm)) / 32000, final)
        if final or item['text'] != last_partial:
            segments.append(item)
        last_partial = None if final else item['text']
    result = json.loads(recognizer.FinalResult())
    tail = native_result(result, len(clip.pcm) / 32000, True)
    tail['eof_flush'] = True
    segments.append(tail)
    transcript = ' '.join(s['text'] for s in segments if s['final'] and s['text'])
    return {'file': clip.path.name, 'reference': clip.reference, 'sha256': clip.sha256,
            'transcript': transcript, 'segments': segments,
            'decode_wall_seconds': time.monotonic() - started, 'decode_cpu_seconds': time.process_time() - cpu,
            'evaluation': evaluate(clip.reference, transcript, segments)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--profile', choices=PROFILES, default='mixed')
    parser.add_argument('--model', type=Path, default=Path('models/speech/vosk-model-small-en-us-0.15'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; choose a new report path')
    metadata, clips = load_manifest(args.manifest)
    # Validate labels before invoking native code. No new physical recording here.
    for clip in clips:
        role(clip.reference)
    grammar = grammar_for(args.profile)
    import vosk
    vosk.SetLogLevel(-1)
    started = time.monotonic()
    model = vosk.Model(str(args.model))
    report = {'profile': args.profile, 'grammar': grammar,
              'grammar_sha256': hashlib.sha256(json.dumps(grammar).encode()).hexdigest(),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'model_load_seconds': time.monotonic() - started,
              'missing_vocabulary': [word for word in sorted(set(' '.join(grammar).split())) if model.vosk_model_find_word(word) < 0],
              'scope': 'Replay of existing clips; one model load, new decoder per clip, no real-time pacing or active YOLO',
              'rows': []}
    for clip in clips:
        row = replay(model, clip, grammar, 3 if args.profile == 'mixed-nbest' else 0)
        report['rows'].append(row)
        print(f"{args.profile}: {clip.path.name}: {row['transcript']!r}", flush=True)
    report['summary'] = summarize(report['rows'])
    report['exploratory_rejection_sweep'] = rejection_sweep(report['rows'])
    report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        os.chmod(args.output, 0o600)
        json.dump(report, output, indent=2)
    print(json.dumps(report['summary'], indent=2))
    print('Report:', args.output)


if __name__ == '__main__':
    main()
