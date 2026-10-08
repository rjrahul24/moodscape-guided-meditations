"""Engine-agnostic per-chunk speech QA (gaps, truncation, rate outliers, CER).

Used to pick which TTS chunks to re-render. Imports nothing engine-specific.
"""

from __future__ import annotations

import numpy as np

from core.eval.speech_metrics import cer, gap_breakdown, normalize_text

INTERNAL_GAP, TRUNCATED, RATE_OUTLIER, HIGH_CER = "INTERNAL_GAP", "TRUNCATED", "RATE_OUTLIER", "HIGH_CER"
MAX_CER = 0.20
SHORT_CHARS = 12
_TRUNC_FRACTION = 0.35
_RATE_LO, _RATE_HI = 0.6, 1.7


def _nonspace_chars(text: str) -> int:
    return len("".join(text.split()))


def sec_per_char(audio: np.ndarray, sr: int, text: str) -> float:
    chars = _nonspace_chars(text)
    if chars == 0:
        return 0.0
    return (len(audio) / float(sr)) / chars


def chunk_signal_checks(
    audio: np.ndarray, sr: int, text: str, *, median_sec_per_char: float | None
) -> list[str]:
    """Signal-only checks (TRUNCATED). Gaps are judged by transcript_checks().

    Raw silence is not a defect: F5's natural comma/sentence pauses run
    0.6-1.7 s and every chunk opens with a quiet ref-pad lead-in, which a
    silence rule misread as "speech, gap, speech" on ~65% of real chunks.
    """
    codes: list[str] = []
    chars = _nonspace_chars(text)
    if (
        median_sec_per_char is not None
        and chars > SHORT_CHARS
        and len(audio) / float(sr) < _TRUNC_FRACTION * chars * median_sec_per_char
    ):
        codes.append(TRUNCATED)
    return codes


def rate_outliers(rates: list[tuple[float, int]]) -> set[int]:
    eligible = [r for r, c in rates if c > SHORT_CHARS]
    if len(eligible) < 3:
        return set()
    median = float(np.median(eligible))
    return {
        i
        for i, (r, c) in enumerate(rates)
        if c > SHORT_CHARS and (r < _RATE_LO * median or r > _RATE_HI * median)
    }


def transcript_checks(audio: np.ndarray, sr: int, text: str, transcriber) -> tuple[list[str], float]:
    """All transcript-based codes from ONE transcription: HIGH_CER and INTERNAL_GAP.

    INTERNAL_GAP uses Whisper word timings: silence inside a word span, or
    between two words of one clause (not after a comma/sentence end).
    """
    transcript = transcriber.transcribe(audio, sr)
    value = cer(normalize_text(text), normalize_text(transcript.text))
    codes: list[str] = []
    if value > MAX_CER:
        codes.append(HIGH_CER)
    if sum(gap_breakdown(audio, sr, transcript.words)) > 0:
        codes.append(INTERNAL_GAP)
    return codes, value


def score_candidate(codes: list[str], cer_value: float, rate_dev: float) -> tuple[int, float, float]:
    """Lower is better; compare tuples."""
    return (len(codes), cer_value, rate_dev)
