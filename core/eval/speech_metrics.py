"""Intelligibility and artifact metrics on rendered speech, engine-agnostic.

Whisper transcribes the audio; CER/WER compare it with the script's spoken
text; word timestamps plus frame energy find silences that land inside a word
or mid-sentence (F5's mid-word gaps and stalls).
"""

import contextlib
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from core.script_gen.linter import _normalize_apostrophes, _strip_tags

_CLOSERS = "\"'\u2019\u201d)\\]"
_SENTENCE_END = re.compile(rf"[.!?\u2026][{_CLOSERS}]*$")
_CLAUSE_END = re.compile(rf"[.!?,;:\u2026\u2014\u2013][{_CLOSERS}]*$")


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript: ...


@contextlib.contextmanager
def _torchcodec_disabled():
    """Make transformers' ASR pipeline treat torchcodec as unavailable.

    The pipeline's preprocess does `import torchcodec` whenever
    `is_torchcodec_available()` is true. torchcodec 0.10 is linked against
    ffmpeg 4 (libavutil.56) but Homebrew ships ffmpeg 8.1, so that import
    raises OSError. We pass raw numpy audio, so no decoding is needed.
    Scoped to the pipeline call only; torchcodec stays installed for f5-tts.
    """
    from transformers.pipelines import automatic_speech_recognition as asr

    original = asr.is_torchcodec_available
    asr.is_torchcodec_available = lambda: False
    try:
        yield
    finally:
        asr.is_torchcodec_available = original


def _words_from_chunks(chunks: list[dict], duration: float) -> list[Word]:
    """Build Words; a missing end falls back to the next start, then the duration."""
    valid = [c for c in chunks if c.get("timestamp") and c["timestamp"][0] is not None]
    words: list[Word] = []
    for i, c in enumerate(valid):
        start, end = c["timestamp"][0], c["timestamp"][1]
        if end is None:
            end = valid[i + 1]["timestamp"][0] if i + 1 < len(valid) else duration
        words.append(Word(c["text"].strip(), float(start), float(max(end, start))))
    return words


# Whisper generate() kwargs. "language" skips language detection (which can
# drift on near-silent input); "condition_on_prev_tokens" False stops each
# window being conditioned on the previous text, the main driver of repetition
# loops. no_repeat_ngram_size is deliberately not set: it would hide genuine
# repeats from the synthesiser, which this metric exists to surface.
_GENERATE_KWARGS = {"language": "english", "condition_on_prev_tokens": False}

SPLIT_SILENCE_SEC = 0.8
REGION_PAD_SEC = 0.15
MAX_REGION_SEC = 28.0
MIN_REGION_SEC = 0.3  # shorter spans are noise blips; Whisper hallucinates on them
_FRAME_SEC = 0.01


class WhisperTranscriber:
    """transformers ASR pipeline with word timestamps; loaded lazily."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id
        self._pipe = None

    def _load(self):
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        """Transcribe speech regions separately, then stitch text and word times.

        Feeding long audio with long silences to Whisper makes it hallucinate
        ("Thank you", "Bye-bye" loops) and duplicate phrases at long-form chunk
        overlaps, so each region is a short pass with no pipeline chunking.
        """
        from scipy.signal import resample_poly

        if self._pipe is None:
            self._load()
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        texts: list[str] = []
        words: list[Word] = []
        for start, end in speech_regions(mono, sr):
            piece = mono[int(start * sr) : int(end * sr)]
            if len(piece) == 0:
                continue
            audio16 = resample_poly(piece.astype(np.float64), 16000, sr).astype(np.float32)
            with _torchcodec_disabled():
                result = self._pipe(
                    {"raw": audio16, "sampling_rate": 16000},
                    return_timestamps="word",
                    generate_kwargs=_GENERATE_KWARGS,
                )
            text = result["text"].strip()
            if text:
                texts.append(text)
            for w in _words_from_chunks(result.get("chunks", []), len(piece) / float(sr)):
                words.append(Word(w.text, w.start + start, w.end + start))
        return Transcript(" ".join(texts), words)

    def unload(self) -> None:
        import gc

        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _spell_number(match: re.Match) -> str:
    from num2words import num2words

    return num2words(int(match.group(0))).replace("-", " ")


def normalize_text(text: str) -> str:
    """Lowercase, tags removed, digits spelled, punctuation dropped."""
    text = _normalize_apostrophes(_strip_tags(text)).lower()
    text = re.sub(r"\d+", _spell_number, text)
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    from rapidfuzz.distance import Levenshtein

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    from rapidfuzz.distance import Levenshtein

    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def _frame_rms_db(audio: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)


def _silent_frames(audio: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    rms_db = _frame_rms_db(audio, sr, frame_ms)
    if len(rms_db) == 0:
        return np.zeros(0, dtype=bool)
    threshold = max(-50.0, float(np.max(rms_db)) - 40.0)
    return rms_db < threshold


def _silent_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of each run of True in mask."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _split_long(rms_db: np.ndarray, a: int, b: int, max_frames: int) -> list[tuple[int, int]]:
    """Recursively cut [a, b) at the quietest frame in its middle half."""
    if b - a <= max_frames:
        return [(a, b)]
    lo, hi = a + (b - a) // 4, a + 3 * (b - a) // 4
    mid = (a + b) / 2.0
    cut = min(range(lo, hi), key=lambda i: (rms_db[i], abs(i - mid)))
    return _split_long(rms_db, a, cut, max_frames) + _split_long(rms_db, cut, b, max_frames)


def speech_regions(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """(start_sec, end_sec) spans of speech, split at silences >= SPLIT_SILENCE_SEC.

    Each span is padded by REGION_PAD_SEC (clamped to the audio) and spans
    longer than MAX_REGION_SEC are cut at their quietest middle frame.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms_db = _frame_rms_db(mono, sr)
    n = len(rms_db)
    if n == 0:
        return []
    silent = _silent_frames(mono, sr)
    split_frames = int(round(SPLIT_SILENCE_SEC / _FRAME_SEC))
    pad = int(round(REGION_PAD_SEC / _FRAME_SEC))
    total = len(mono) / float(sr)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for s0, e0 in _silent_runs(silent):
        if e0 - s0 >= split_frames:
            if s0 > cursor:
                spans.append((cursor, s0))
            cursor = e0
    if cursor < n:
        spans.append((cursor, n))
    regions: list[tuple[float, float]] = []
    for a, b in spans:
        if not (~silent[a:b]).any() or (b - a) * _FRAME_SEC < MIN_REGION_SEC:
            continue
        a, b = max(0, a - pad), min(n, b + pad)
        for x, y in _split_long(rms_db, a, b, int(MAX_REGION_SEC / _FRAME_SEC)):
            regions.append((x * _FRAME_SEC, min(y * _FRAME_SEC, total)))
    return regions


def gap_breakdown(
    audio: np.ndarray, sr: int, words: list[Word], *, min_gap_ms: float = 250.0
) -> tuple[int, int]:
    """(gaps inside a word span, gaps between words of one clause).

    Silence touching a word span's edge is ignored (Whisper spans are loose and
    absorb intended pauses); between words, the frames over the gap must
    themselves be silent for min_gap_ms. Between-word pairs whose timestamp gap
    is >= SPLIT_SILENCE_SEC straddle a transcription-region boundary and are
    not counted; long stalls still show in CER and rate spread.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    silent = _silent_frames(mono, sr)
    frames_needed = int(min_gap_ms / 10.0)
    in_word = between = 0
    for word in words:
        a, b = int(word.start * 100), int(word.end * 100)
        span = silent[a:b]
        if any(e - s >= frames_needed and s > 0 and e < len(span) for s, e in _silent_runs(span)):
            in_word += 1
    for prev, nxt in zip(words, words[1:]):
        if _CLAUSE_END.search(prev.text):
            continue
        if nxt.start - prev.end >= SPLIT_SILENCE_SEC:
            continue
        gap = silent[int(prev.end * 100) : int(nxt.start * 100)]
        if any(e - s >= frames_needed for s, e in _silent_runs(gap)):
            between += 1
    return in_word, between


def internal_gaps(audio: np.ndarray, sr: int, words: list[Word], *, min_gap_ms: float = 250.0) -> int:
    """Total of gap_breakdown(): silences inside a word or mid-clause between words."""
    return sum(gap_breakdown(audio, sr, words, min_gap_ms=min_gap_ms))


def rate_spread(words: list[Word]) -> dict:
    """Median and IQR of seconds-per-character across Whisper sentences."""
    rates: list[float] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if _SENTENCE_END.search(word.text):
            chars = sum(len(re.sub(r"[^A-Za-z']", "", w.text)) for w in current)
            if chars:
                rates.append((current[-1].end - current[0].start) / chars)
            current = []
    if not rates:
        return {"sec_per_char_p50": None, "sec_per_char_iqr": None}
    q1, q2, q3 = np.percentile(rates, [25, 50, 75])
    return {"sec_per_char_p50": round(float(q2), 4), "sec_per_char_iqr": round(float(q3 - q1), 4)}


def speech_metrics(audio: np.ndarray, sr: int, expected_text: str, *, transcriber: Transcriber) -> dict:
    transcript = transcriber.transcribe(audio, sr)
    in_word, between = gap_breakdown(audio, sr, transcript.words)
    return {
        "cer": round(cer(expected_text, transcript.text), 4),
        "wer": round(wer(expected_text, transcript.text), 4),
        "internal_gaps": in_word + between,
        "gaps_in_word": in_word,
        "gaps_between_words": between,
        "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in transcript.words],
        **rate_spread(transcript.words),
        "transcript": transcript.text,
    }
