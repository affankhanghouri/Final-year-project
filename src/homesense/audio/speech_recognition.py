"""Replaceable streaming speech-to-text interface and local Vosk implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Recognition:
    text: str
    final: bool
    utterance_id: int | None = None


class Recognizer(Protocol):
    def feed(self, pcm: bytes) -> Recognition: ...
    def reset(self) -> None: ...


class VoskRecognizer:
    def __init__(self, model_path: Path, sample_rate: int = 16000, silence_seconds: float = 0.6, silence_rms: float = 0.015,
                 *, software_endpoint: bool = True, grammar: tuple[str, ...] | None = None):
        path = Path(model_path).expanduser()
        if not (path / "am" / "final.mdl").is_file():
            raise FileNotFoundError(f"Local Vosk model missing at {path}; see docs/audio-setup.md")
        import vosk
        from homesense import diagnostic_logging as diagnostic
        if diagnostic.enabled():
            vosk.SetLogLevel(-1)  # Kaldi warnings/errors remain; suppress informational grammar dumps.
            diagnostic.event('recognizer', 'recognizer_loading', model_path=str(path), sample_rate=sample_rate,
                             grammar_entries=len(grammar) if grammar else 0)
        self._vosk = vosk
        self._model = vosk.Model(str(path))  # Explicit path: never use automatic model download.
        self._rate = sample_rate
        self._silence = silence_seconds
        self._rms = silence_rms
        # Legacy diagnostics retain their RMS endpoint; production disables it.
        self._software_endpoint = software_endpoint
        self._grammar = grammar
        self.reset()
        diagnostic.event('recognizer', 'recognizer_ready', sample_rate=sample_rate,
                         grammar_entries=len(grammar) if grammar else 0, software_endpoint=software_endpoint)

    def reset(self):
        import json
        args = () if self._grammar is None else (json.dumps(self._grammar),)
        self._recognizer = self._vosk.KaldiRecognizer(self._model, self._rate, *args)
        self._silence_elapsed = 0.0
        self._heard_text = False

    def feed(self, pcm: bytes) -> Recognition:
        import json
        from array import array
        final = self._recognizer.AcceptWaveform(pcm)
        result = json.loads(self._recognizer.Result() if final else self._recognizer.PartialResult())
        best = result.get("alternatives", [result])[0]
        text = best.get("text" if final else "partial", "")
        self._heard_text |= bool(text)
        samples = array("h", pcm)
        rms = (sum(v * v for v in samples) / len(samples)) ** 0.5 / 32768 if samples else 0.0
        self._silence_elapsed = self._silence_elapsed + len(pcm) / (2 * self._rate) if rms < self._rms else 0.0
        if self._software_endpoint and not final and self._heard_text and self._silence_elapsed >= self._silence:
            text = json.loads(self._recognizer.FinalResult()).get("text", "")
            self._recognizer.Reset()
            final = True
        if final:
            self._silence_elapsed = 0.0
            self._heard_text = False
        return Recognition(text, bool(final))

    def finish(self) -> Recognition:
        """Flush the tail at a known recording boundary, for offline replay only."""
        import json
        result = json.loads(self._recognizer.FinalResult())
        text = result.get('alternatives', [result])[0].get('text', '')
        self._recognizer.Reset()
        self._silence_elapsed = 0.0
        self._heard_text = False
        return Recognition(text, True)


def resolve_command(primary: str, fallback: str) -> str:
    """Fallback may clarify/veto; it never authorizes a new supported action.

    Unrestricted decoding is poor on this model. Treat explicit contradictory
    intent, unsupported requests and negation as a veto; an unrecognized fallback
    is not confirmation of a constrained transcript or a new source of commands.
    No confidence threshold or fuzzy/subsequence command acceptance is used.
    """
    import re
    from homesense.commands.parser import parse_command, words
    from homesense.commands.models import Intent, CommandProblem
    constrained, open_command = parse_command(primary), parse_command(fallback)
    if re.search(r"\b(?:not|never|don t|haven t|cannot|can t)\b", words(fallback)):
        return "[unk]"
    if open_command.problem in (CommandProblem.UNSUPPORTED, CommandProblem.AMBIGUOUS):
        return fallback
    if constrained.intent is not Intent.UNKNOWN and open_command.intent is not Intent.UNKNOWN:
        if (constrained.intent, constrained.target) != (open_command.intent, open_command.target):
            return "[unk]"
    return primary


class VoskCommandRecognizer(VoskRecognizer):
    """Resident constrained decoder with native endpoints and bounded local veto.

    Both decoders share one acoustic model. Only this audio worker runs inference.
    PCM is retained in RAM for at most max_seconds and is never saved. The second
    decode runs once at each nonempty native endpoint;
    malformed/fallback-only supported requests never become executable commands.
    """

    def __init__(self, model_path: Path, sample_rate=16000, *, max_seconds=10.0):
        from homesense.commands.parser import command_grammar
        self._serial = 0
        self._limit = int(sample_rate * 2 * max_seconds)
        self._audio = bytearray()
        self._partial_seen = False
        self._discarding = False
        super().__init__(model_path, sample_rate, software_endpoint=False, grammar=command_grammar())
        self._fallback = self._vosk.KaldiRecognizer(self._model, sample_rate)

    def reset(self):
        self._audio.clear()
        if hasattr(self, '_recognizer'):
            self._recognizer.Reset()
        else:
            super().reset()
            self._recognizer.SetMaxAlternatives(3)
        if hasattr(self, '_fallback'):
            self._fallback.Reset()
        self._partial_seen = False
        self._discarding = False
        self._silence_elapsed = 0.0
        self._heard_text = False

    def _open_transcript(self):
        import json
        self._fallback.Reset()
        finals = []
        for start in range(0, len(self._audio), 3200):
            if self._fallback.AcceptWaveform(bytes(self._audio[start:start + 3200])):
                finals.append(json.loads(self._fallback.Result()).get('text', ''))
        finals.append(json.loads(self._fallback.FinalResult()).get('text', ''))
        return ' '.join(text for text in finals if text)

    def _final(self, text):
        if text:
            fallback = self._open_transcript()
            import logging
            selected = resolve_command(text, fallback)
            logging.getLogger(__name__).debug('Constrained final=%r; open-vocabulary veto=%r; delivered=%r', text, fallback, selected)
            text = selected
        self._audio.clear()
        self._partial_seen = False
        self._serial += 1
        return Recognition(text, True, self._serial)

    def discard(self):
        """Discard an expired utterance until its native boundary, not its suffix."""
        self._audio.clear()
        self._discarding = True

    def feed(self, pcm: bytes) -> Recognition:
        if len(pcm) % 2:
            raise ValueError('Expected PCM16 byte pairs')
        if self._discarding:
            if self._recognizer.AcceptWaveform(pcm):
                self._recognizer.Result()
                self.reset()
            return Recognition('', False)
        self._audio.extend(pcm)
        if len(self._audio) > self._limit:
            # Idle silence must not grow memory; an overlong request is discarded
            # in full rather than acting on a truncated suffix after resetting.
            self.discard()
            self._serial += 1
            return Recognition('', True, self._serial)
        result = super().feed(pcm)
        self._partial_seen |= bool(result.text)
        if result.final:
            return self._final(result.text)
        if not self._partial_seen:
            # Bounded pre-roll keeps quiet idle time from consuming the command
            # budget, without an RMS/VAD rule that could cut off quiet speech.
            del self._audio[:-self._rate * 2]
        return result

    def finish(self) -> Recognition:
        """Explicit finite-file boundary for diagnostics, never used by live audio."""
        return self._final(super().finish().text)
