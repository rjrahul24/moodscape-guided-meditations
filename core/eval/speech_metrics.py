"""Intelligibility and artifact metrics on rendered speech, engine-agnostic.

Whisper transcribes the audio; CER/WER compare it with the script's spoken
text; word timestamps plus frame energy find silences that land inside a word
or mid-sentence (F5's mid-word gaps and stalls).
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from core.script_gen.linter import _normalize_apostrophes, _strip_tags

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
_CLAUSE_END = re.compile(r"[.!?,;:][\"')\]]*$")


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
        from scipy.signal import resample_poly

        if self._pipe is None:
            self._load()
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        audio16 = resample_poly(mono.astype(np.float64), 16000, sr).astype(np.float32)
        result = self._pipe(
            {"raw": audio16, "sampling_rate": 16000},
            return_timestamps="word",
            chunk_length_s=30,
            generate_kwargs={"language": "english"},
        )
        words = [
            Word(c["text"].strip(), float(c["timestamp"][0]), float(c["timestamp"][1] or c["timestamp"][0]))
            for c in result.get("chunks", [])
            if c.get("timestamp") and c["timestamp"][0] is not None
        ]
        return Transcript(result["text"].strip(), words)

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


def _silent_frames(audio: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: n * hop].reshape(n, hop)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)
    threshold = max(-50.0, float(np.max(rms_db)) - 40.0)
    return rms_db < threshold


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def internal_gaps(audio: np.ndarray, sr: int, words: list[Word], *, min_gap_ms: float = 250.0) -> int:
    """Silences > min_gap_ms inside a word, or between words of one clause."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    silent = _silent_frames(mono, sr)
    frames_needed = int(min_gap_ms / 10.0)
    count = 0
    for word in words:
        a, b = int(word.start * 100), int(word.end * 100)
        if _longest_run(silent[a:b]) >= frames_needed:
            count += 1
    for prev, nxt in zip(words, words[1:]):
        if not _CLAUSE_END.search(prev.text) and (nxt.start - prev.end) * 1000.0 > min_gap_ms:
            count += 1
    return count


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
    return {
        "cer": round(cer(expected_text, transcript.text), 4),
        "wer": round(wer(expected_text, transcript.text), 4),
        "internal_gaps": internal_gaps(audio, sr, transcript.words),
        **rate_spread(transcript.words),
        "transcript": transcript.text,
    }
