"""Replay existing consented WAVs through production STT; no microphone/camera.

This finite development regression measures raw native utterance delivery. It
does not synthesize feedback or model live TTS suppression between utterances.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

from homesense.audio.command_gate import CommandGate
from homesense.audio.comparison import load_manifest, normalize, word_errors
from homesense.audio.events import VoiceEventKind
from homesense.audio.speech_recognition import VoskCommandRecognizer
from homesense.commands.models import Intent
from homesense.commands.parser import command_grammar, parse_command


def command_record(text):
    command = parse_command(text)
    return {'intent': command.intent.value, 'target': command.target.value if command.target else None,
            'accepted': command.intent is not Intent.UNKNOWN}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--model', type=Path, default=Path('models/speech/vosk-model-small-en-us-0.15'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output path; existing evidence is preserved')
    _, clips = load_manifest(args.manifest)
    import vosk
    vosk.SetLogLevel(-1)
    started = time.monotonic()
    recognizer = VoskCommandRecognizer(args.model)
    load_seconds = time.monotonic() - started
    rows = []
    for clip in clips:
        recognizer.reset()
        gate = CommandGate()
        segments, commands, feed_times = [], [], []
        started, cpu = time.monotonic(), time.process_time()
        def consume(update, offset, *, eof=False):
            if update.final and update.text:
                segments.append({'text': update.text, 'utterance_id': update.utterance_id,
                                 'offset': offset, 'eof_flush': eof})
            for event in gate.accept(update.text, update.final, offset, update.utterance_id):
                if event.kind is VoiceEventKind.TRANSCRIPT:
                    result = command_record(event.text)
                    if result['accepted']:
                        commands.append(result)
        for offset in range(0, len(clip.pcm), 3200):
            before = time.monotonic()
            update = recognizer.feed(clip.pcm[offset:offset + 3200])
            feed_times.append(time.monotonic() - before)
            consume(update, min(offset + 3200, len(clip.pcm)) / 32000)
        consume(recognizer.finish(), len(clip.pcm) / 32000, eof=True)
        text = ' '.join(s['text'] for s in segments)
        expected = command_record(clip.reference)
        correct = [c for c in commands if (c['intent'], c['target']) == (expected['intent'], expected['target'])]
        rows.append({'file': clip.path.name, 'sha256': clip.sha256, 'reference': clip.reference,
                     'transcript': text, 'expected': expected, 'segments': segments, 'commands': commands,
                     'exact': normalize(text) == normalize(clip.reference),
                     'word_errors': word_errors(clip.reference, text),
                     'one_correct_action': expected['accepted'] and len(commands) == len(correct) == 1,
                     'any_correct_action': expected['accepted'] and bool(correct),
                     'maximum_feed_seconds': max(feed_times),
                     'decode_wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu})
        print(clip.path.name, repr(text), commands, flush=True)
    positives = [r for r in rows if r['expected']['accepted']]
    negatives = [r for r in rows if not r['expected']['accepted']]
    speech = [r for r in rows if r['reference']]
    report = {'scope': 'Development replay; resident production recognizer; native endpoints plus explicit EOF; no live TTS',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'grammar': command_grammar(), 'model_load_seconds': load_seconds, 'rows': rows,
              'summary': {'command_clips': len(positives), 'exact_command_transcripts': sum(r['exact'] for r in positives),
                  'one_correct_action': sum(r['one_correct_action'] for r in positives),
                  'any_correct_action': sum(r['any_correct_action'] for r in positives),
                  'multiple_action_clips': sum(len(r['commands']) > 1 for r in positives),
                  'negative_clips': len(negatives), 'false_action_clips': sum(bool(r['commands']) for r in negatives),
                  'speech_clips': len(speech), 'exact_speech_transcripts': sum(r['exact'] for r in speech),
                  'speech_word_errors': sum(r['word_errors'] for r in speech),
                  'speech_reference_words': sum(len(normalize(r['reference']).split()) for r in speech),
                  'median_decode_seconds': statistics.median(r['decode_wall_seconds'] for r in rows),
                  'maximum_decode_seconds': max(r['decode_wall_seconds'] for r in rows),
                  'maximum_feed_seconds': max(r['maximum_feed_seconds'] for r in rows),
                  'cpu_seconds': sum(r['cpu_seconds'] for r in rows),
                  'audio_seconds': sum(len(c.pcm) / 32000 for c in clips)},
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
