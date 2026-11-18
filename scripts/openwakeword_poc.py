"""Isolated Hey Agent experiment: synthetic training and offline WAV replay only.

No production imports, network calls, microphone access or application actions.
The classifier follows openWakeWord's manual notebook's two-layer dense example;
the small local dataset is an experiment, not an official pretrained wake model.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time
import wave

RATE = 16000
CHUNK = 1280
WINDOW = 16
SEED = 20261117
ROOT = Path('outputs/openwakeword-poc')
NEGATIVES = (
    'Hey Adrian', 'Hey Aiden', 'Hey agents', 'Hey urgent', 'Hey engine', 'Hi agent',
    'Hey Jason', 'Hey again', 'An agent', 'Hello agent', 'Agent', 'Hey',
    'Find my phone', 'Find my keys', 'Locate my wallet', 'Cancel search',
    'Repeat the direction', 'I found it', 'Help me', 'Please close the window',
    'Do not cancel', 'We need an urgent answer', 'The agent is here', 'Hey Jacob',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2)


def load_pcm(path):
    import numpy as np
    with wave.open(str(path), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (RATE, 1, 2, 'NONE'):
            raise ValueError('Expected uncompressed 16 kHz mono PCM16 WAV')
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').copy()


def chunks(pcm):
    """Retain the short EOF tail, zero-padding only the final 80 ms block."""
    import numpy as np
    for start in range(0, len(pcm), CHUNK):
        block = pcm[start:start + CHUNK]
        yield np.pad(block, (0, CHUNK - len(block))), min(start + CHUNK, len(pcm)) / RATE


def activation_summary(scores, threshold=.5):
    """Count rising edges; adjacent high frames are one activation episode."""
    active = False
    times = []
    for offset, score in scores:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        current = score >= threshold
        if current and not active:
            times.append(offset)
        active = current
    return {'peak_score': max((s for _, s in scores), default=0),
            'activated': bool(times), 'activation_episodes': len(times), 'activation_offsets': times}


def features(args):
    """Generate only synthetic audio; no HomeSense recordings enter training."""
    import numpy as np
    from scipy.signal import resample_poly
    from piper import PiperVoice, SynthesisConfig
    from openwakeword.utils import AudioFeatures

    if (args.root / 'synthetic.json').exists():
        raise FileExistsError('Synthetic evidence already exists; choose another root')
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    assets = args.root / 'assets'
    voice = PiperVoice.load(str(assets / 'en_US-libritts_r-medium.onnx'))
    extractor = AudioFeatures(inference_framework='onnx', ncpu=1,
                             melspec_model_path=str(assets / 'melspectrogram.onnx'),
                             embedding_model_path=str(assets / 'embedding_model.onnx'))
    audio_dir = args.root / 'synthetic'
    audio_dir.mkdir()
    rows = []
    tensors = {'train': [], 'validation': []}
    labels = {'train': [], 'validation': []}
    started = time.monotonic()
    for split, positive_count, negative_count, speaker_start in (('train', 64, 128, 0), ('validation', 16, 32, 64)):
        for index in range(positive_count + negative_count):
            positive = index < positive_count
            text = 'Hey Agent' if positive else NEGATIVES[(index - positive_count) % len(NEGATIVES)]
            speaker = speaker_start + index % positive_count
            config = SynthesisConfig(speaker_id=speaker, length_scale=float(rng.uniform(.85, 1.15)))
            rendered = list(voice.synthesize(text, config))
            audio = np.concatenate([chunk.audio_int16_array for chunk in rendered])
            source_rate = rendered[0].sample_rate
            audio = resample_poly(audio.astype(np.float32), RATE // math.gcd(RATE, source_rate),
                                  source_rate // math.gcd(RATE, source_rate))
            audible = np.flatnonzero(np.abs(audio) > 100)
            if len(audible) == 0:
                raise ValueError('Empty synthetic speech')
            audio = audio[audible[0]:audible[-1] + 1]
            onset = float(rng.uniform(.7, 1.3))
            end = onset + len(audio) / RATE
            samples = max(5 * RATE, int((end + .8) * RATE))
            pcm = rng.normal(0, rng.uniform(10, 100), samples)
            begin = int(onset * RATE)
            pcm[begin:begin + len(audio)] += audio * rng.uniform(.4, 1.0)
            # A simple synthetic echo; no claim of room/acoustic diversity.
            delay = int(.06 * RATE)
            pcm[delay:] += .12 * pcm[:-delay]
            pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
            path = audio_dir / f'{split}-{index:03d}.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                wav.writeframes(pcm.astype('<i2').tobytes())
            extractor.reset()
            selected = []
            for block, offset in chunks(pcm):
                extractor(block)
                # Fixed locations, selected from synthetic timing, never human labels.
                positive_window = positive and end <= offset < end + .25
                negative_window = not positive and int(round(offset / .08)) % 10 == 0
                background = positive and abs(offset - .56) < .001
                if positive_window or negative_window or background:
                    tensors[split].append(extractor.get_features(WINDOW)[0].copy())
                    labels[split].append(float(positive_window))
                    selected.append({'offset': offset, 'label': int(positive_window)})
            rows.append({'file': str(path.relative_to(args.root)), 'split': split, 'text': text,
                         'wake': positive, 'speaker': speaker, 'onset': onset, 'end': end,
                         'sha256': digest(path), 'selected_windows': selected})
            if index % 16 == 0:
                print(split, index, text, flush=True)
    for split in tensors:
        np.savez_compressed(args.root / f'{split}.npz', X=np.asarray(tensors[split], dtype=np.float32),
                            y=np.asarray(labels[split], dtype=np.float32)[:, None])
    write_json(args.root / 'synthetic.json', {'seed': SEED, 'rows': rows,
                'elapsed_seconds': time.monotonic() - started,
                'training_windows': len(labels['train']), 'validation_windows': len(labels['validation']),
                'voice_sha256': digest(assets / 'en_US-libritts_r-medium.onnx')})


def train(args):
    """Use already-installed PyTorch; export weights without adding dependencies."""
    import numpy as np
    import torch
    from torch import nn
    if (args.root / 'head-weights.npz').exists():
        raise FileExistsError('Training evidence already exists')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    data = np.load(args.root / 'train.npz')
    X, y = torch.from_numpy(data['X']), torch.from_numpy(data['y'])
    network = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW * 96, 32), nn.LayerNorm(32), nn.ReLU(),
                            nn.Linear(32, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = torch.optim.Adam(network.parameters(), lr=.001)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=128, shuffle=True)
    losses = []
    started = time.monotonic()
    for _ in range(30):
        loss_sum = 0
        for inputs, expected in loader:
            optimizer.zero_grad()
            weights = torch.where(expected == 1, .1, 1.)
            loss = torch.nn.functional.binary_cross_entropy(network(inputs), expected, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        losses.append(loss_sum / len(loader))
    np.savez(args.root / 'head-weights.npz', **{k: v.detach().numpy() for k, v in network.state_dict().items()})
    with torch.no_grad():
        np.savez(args.root / 'export-check.npz', X=data['X'][:16], y=network(X[:16]).numpy())
    write_json(args.root / 'training.json', {'seed': SEED, 'epochs': 30, 'hidden_units': 32,
               'feature_frames': WINDOW, 'positive_loss_weight': .1, 'negative_loss_weight': 1.,
               'losses': losses, 'elapsed_seconds': time.monotonic() - started,
               'torch': torch.__version__, 'train_features_sha256': digest(args.root / 'train.npz'),
               'note': 'Fixed small manual-notebook-style training; no human evaluation data used'})


def export(args):
    """Serialize dense/LayerNorm weights with ONNX; verify against PyTorch output."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import onnxruntime as ort
    if args.model.exists():
        raise FileExistsError('Model exists; will not overwrite')
    weights = np.load(args.root / 'head-weights.npz')
    nodes = [helper.make_node('Flatten', ['input'], ['flat'], axis=1)]
    initializers = []
    previous = 'flat'
    for index, layer in enumerate((1, 4, 7)):
        prefix = f'layer{index}'
        for name, data in (('W', weights[f'{layer}.weight'].T), ('b', weights[f'{layer}.bias'])):
            initializers.append(numpy_helper.from_array(data, f'{prefix}_{name}'))
        nodes.extend([helper.make_node('MatMul', [previous, f'{prefix}_W'], [prefix + '_mm']),
                      helper.make_node('Add', [prefix + '_mm', f'{prefix}_b'], [prefix + '_bias'])])
        if layer != 7:
            for name, data in (('scale', weights[f'{layer + 1}.weight']), ('offset', weights[f'{layer + 1}.bias'])):
                initializers.append(numpy_helper.from_array(data, prefix + '_' + name))
            nodes.extend([helper.make_node('LayerNormalization', [prefix + '_bias', prefix + '_scale', prefix + '_offset'],
                                           [prefix + '_norm'], axis=-1, epsilon=1e-5),
                          helper.make_node('Relu', [prefix + '_norm'], [prefix + '_out'])])
            previous = prefix + '_out'
        else:
            nodes.append(helper.make_node('Sigmoid', [prefix + '_bias'], ['output']))
    graph = helper.make_graph(nodes, 'Hey Agent small synthetic prototype',
                [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, WINDOW, 96])],
                [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.model))
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    check = np.load(args.root / 'export-check.npz')
    actual = np.concatenate([session.run(None, {'input': x[None]})[0] for x in check['X']])
    error = float(np.max(np.abs(actual - check['y'])))
    if error > 1e-5:
        raise ValueError(f'ONNX export disagrees with PyTorch: {error}')
    write_json(args.root / 'export.json', {'max_absolute_error': error, 'model_sha256': digest(args.model)})


def replay(model, pcm):
    model.reset()
    started, cpu = time.monotonic(), time.process_time()
    scores, durations = [], []
    for block, offset in chunks(pcm):
        before = time.monotonic()
        predictions = model.predict(block)
        durations.append(time.monotonic() - before)
        scores.append((offset, float(next(iter(predictions.values())))))
    return {'scores': scores, 'decode_wall_seconds': time.monotonic() - started,
            'decode_cpu_seconds': time.process_time() - cpu,
            'block_wall_seconds': durations, **activation_summary(scores)}


def summarize(rows):
    positives = [r for r in rows if r['expected_wake']]
    negatives = [r for r in rows if not r['expected_wake']]
    return {'positive_clips': len(positives), 'detections': sum(r['activated'] for r in positives),
            'misses': sum(not r['activated'] for r in positives), 'negative_clips': len(negatives),
            'false_positive_clips': sum(r['activated'] for r in negatives),
            'false_activation_episodes': sum(r['activation_episodes'] for r in negatives),
            'negative_audio_seconds': sum(r['audio_seconds'] for r in negatives)}


def evaluate(args):
    import numpy as np
    from openwakeword.model import Model
    np.random.seed(SEED)
    assets = args.root / 'assets'
    started = time.monotonic()
    model = Model(wakeword_models=[str(args.model)], inference_framework='onnx',
                  melspec_model_path=str(assets / 'melspectrogram.onnx'),
                  embedding_model_path=str(assets / 'embedding_model.onnx'), ncpu=1)
    load_seconds = time.monotonic() - started
    manifest = json.loads(args.manifest.read_text())
    # Original recording format is inspected/validated; no label-derived cropping.
    rows = []
    for clip in manifest['clips']:
        path = args.manifest.parent / clip['file']
        if not path.resolve().is_relative_to(args.manifest.parent.resolve()):
            raise ValueError('Manifest path escapes its directory')
        if digest(path) != clip['sha256']:
            raise ValueError('Recording hash mismatch')
        pcm = load_pcm(path)
        text = clip['reference'].lower()
        expected = text.startswith('hey agent')
        result = replay(model, pcm)
        result.update(file=path.name, reference=clip['reference'], expected_wake=expected,
                      sha256=clip['sha256'], audio_seconds=len(pcm) / RATE)
        rows.append(result)
        print(path.name, expected, result['peak_score'], result['activated'], flush=True)
    synthetic = json.loads((args.root / 'synthetic.json').read_text())
    validation = []
    for clip in synthetic['rows']:
        if clip['split'] != 'validation':
            continue
        path = args.root / clip['file']
        if digest(path) != clip['sha256']:
            raise ValueError('Synthetic validation hash mismatch')
        pcm = load_pcm(path)
        validation.append(dict(replay(model, pcm), file=clip['file'], reference=clip['text'],
                               expected_wake=clip['wake'], audio_seconds=len(pcm) / RATE))
    block_times = [d for r in rows for d in r['block_wall_seconds']]
    write_json(args.output, {'phrase': 'Hey Agent', 'nominal_threshold': .5,
            'policy': 'One >=0.5 frame activates; adjacent high frames form one episode; no threshold tuning',
            'model_sha256': digest(args.model), 'manifest_sha256': digest(args.manifest),
            'model_load_seconds': load_seconds, 'rows': rows, 'summary': summarize(rows),
            'synthetic_validation': validation, 'synthetic_summary': summarize(validation),
            'median_clip_decode_seconds': statistics.median(r['decode_wall_seconds'] for r in rows),
            'median_block_seconds': statistics.median(block_times), 'p95_block_seconds': float(np.percentile(block_times, 95)),
            'max_block_seconds': max(block_times),
            'real_time_factor': sum(r['decode_wall_seconds'] for r in rows) / sum(r['audio_seconds'] for r in rows),
            'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'script_sha256': digest(__file__)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('features', 'train', 'export', 'evaluate'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--model', type=Path, default=Path('models/speech/openwakeword-poc/hey_agent.onnx'))
    parser.add_argument('--manifest', type=Path, default=Path('recordings/asr-device12-trial1/manifest.json'))
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation.json')
    args = parser.parse_args(argv)
    if args.action == 'evaluate' and args.output.exists():
        parser.error('Output exists; choose a new path')
    {'features': features, 'train': train, 'export': export, 'evaluate': evaluate}[args.action](args)


if __name__ == '__main__':
    main()
