"""Engine-agnostic per-chunk speech QA (gaps, truncation, rate outliers, CER).

Used to pick which TTS chunks to re-render. Imports nothing engine-specific.
"""

from __future__ import annotations

import numpy as np

from core.eval.speech_metrics import _silent_frames, _silent_runs, cer, normalize_text

INTERNAL_GAP, TRUNCATED, RATE_OUTLIER, HIGH_CER = "INTERNAL_GAP", "TRUNCATED", "RATE_OUTLIER", "HIGH_CER"
MAX_CER = 0.20
SHORT_CHARS = 12
_GAP_SEC = 0.6
_FRAME_MS = 10.0
_TRUNC_FRACTION = 0.35
_RATE_LO, _RATE_HI = 0.6, 1.7


def _nonspace_chars(text: str) -> int:
    return len("".join(text.split()))


def sec_per_char(audio: np.ndarray, sr: int, text: str) -> float:
    chars = _nonspace_chars(text)
    if chars == 0:
        return 0.0
    return (len(audio) / float(sr)) / chars


def _has_internal_gap(audio: np.ndarray, sr: int) -> bool:
    mask = _silent_frames(audio, sr, _FRAME_MS)
    min_frames = int(round(_GAP_SEC * 1000.0 / _FRAME_MS))
    for start, end in _silent_runs(mask):
        if end - start >= min_frames and start > 0 and end < len(mask):
            return True
    return False


def chunk_signal_checks(
    audio: np.ndarray, sr: int, text: str, *, median_sec_per_char: float | None
) -> list[str]:
    codes: list[str] = []
    if _has_internal_gap(audio, sr):
        codes.append(INTERNAL_GAP)
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


def transcript_check(audio: np.ndarray, sr: int, text: str, transcriber) -> tuple[str | None, float]:
    transcript = transcriber.transcribe(audio, sr)
    value = cer(normalize_text(text), normalize_text(transcript.text))
    return (HIGH_CER if value > MAX_CER else None), value


def score_candidate(codes: list[str], cer_value: float, rate_dev: float) -> tuple[int, float, float]:
    """Lower is better; compare tuples."""
    return (len(codes), cer_value, rate_dev)
