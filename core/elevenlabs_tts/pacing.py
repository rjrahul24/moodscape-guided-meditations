"""Timestamp-driven pacing for Eleven v4 (opt-in experiment, pure DSP, no SDK).

v4 ignores ``voice_settings.speed`` and speaks a multi-sentence chunk as one
continuous breath, so there are no internal gaps to stretch. With pacing on,
the engine renders through the timestamps endpoint and this module (1) applies
a gentle formant-preserving time-stretch and (2) inserts silence after sentence
ends and clause marks, located from the character alignment. Off by default:
native v4 delivery (~130 WPM) is the user's choice.
"""

from __future__ import annotations

import os

import numpy as np

SENTENCE_MARKS = frozenset(".!?…")
CLAUSE_MARKS = frozenset(",;:—")
_CLOSERS = frozenset("\"'”’)")
_FADE_MS = 5.0


def pacing_enabled() -> bool:
    return os.environ.get("MOODSCAPE_ELEVENLABS_PACING", "0").strip() != "0"


def _env_clamped(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    return min(hi, max(lo, value))


def pacing_params() -> tuple[float, float, float]:
    """(stretch, sentence_sec, clause_sec) from the environment, clamped."""
    return (
        _env_clamped("MOODSCAPE_ELEVENLABS_STRETCH", 0.90, 0.75, 1.0),
        _env_clamped("MOODSCAPE_ELEVENLABS_SENTENCE_PAUSE_SEC", 0.9, 0.0, 3.0),
        _env_clamped("MOODSCAPE_ELEVENLABS_CLAUSE_PAUSE_SEC", 0.35, 0.0, 2.0),
    )


def pause_points(alignment: dict | None, sentence_sec: float,
                 clause_sec: float) -> list[tuple[float, float]]:
    """(time_sec, gap_sec) after each sentence end / clause mark followed by
    whitespace. Characters inside ``[...]`` are skipped. A point needs a later
    spoken character, so none lands at/after the chunk's final character."""
    if not alignment:
        return []
    chars = list(alignment.get("characters") or [])
    starts = list(alignment.get("starts") or [])
    ends = list(alignment.get("ends") or [])
    n = min(len(chars), len(starts), len(ends))
    points: list[tuple[float, float]] = []
    depth = 0
    for i in range(n):
        ch = chars[i]
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        if ch in SENTENCE_MARKS:
            gap = sentence_sec
        elif ch in CLAUSE_MARKS:
            gap = clause_sec
        else:
            continue
        j = i + 1
        while j < n and chars[j] in _CLOSERS:
            j += 1
        if j >= n or not chars[j].isspace():
            continue
        k = j
        while k < n and chars[k].isspace():
            k += 1
        if k >= n or chars[k] == "[":
            continue  # trailing whitespace or a tag: end-of-chunk spacing is the script's job
        if gap <= 0:
            continue
        mark_end = float(ends[j - 1])
        next_start = float(starts[k])
        t = 0.5 * (mark_end + max(mark_end, next_start))
        points.append((t, float(gap)))
    return points


def _stretch(y: np.ndarray, sr: int, stretch: float) -> np.ndarray:
    from pedalboard import time_stretch

    out = time_stretch(
        y.astype(np.float32, copy=False), sr, stretch_factor=float(stretch),
        high_quality=True, transient_mode="smooth", transient_detector="compound",
        retain_phase_continuity=True, use_long_fft_window=True, preserve_formants=True,
    )
    return np.asarray(out, dtype=np.float32).reshape(-1)


def apply_pacing(y: np.ndarray, sr: int, alignment: dict | None, *, stretch: float,
                 sentence_sec: float, clause_sec: float) -> np.ndarray:
    """Stretch (if != 1.0), then insert silence at each pause point with ~5 ms
    cosine fades either side of the cut. Returns float32."""
    y = np.asarray(y, dtype=np.float32)
    ratio = 1.0
    if stretch != 1.0 and y.size:
        n_in = y.size
        y = _stretch(y, sr, stretch)
        ratio = y.size / n_in
    points = pause_points(alignment, sentence_sec, clause_sec)
    if not points or y.size == 0:
        return y if y.dtype == np.float32 else y.astype(np.float32)

    fade = max(1, int(round(sr * _FADE_MS / 1000.0)))
    cuts = sorted((min(y.size, max(0, int(round(t * ratio * sr)))), int(round(g * sr)))
                  for t, g in points)
    y = y.copy()
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32)))  # 1 → 0
    for k, _ in cuts:
        a = max(0, k - fade)
        y[a:k] *= ramp[fade - (k - a):]
        b = min(y.size, k + fade)
        y[k:b] *= ramp[::-1][: b - k]
    parts: list[np.ndarray] = []
    prev = 0
    for k, n in cuts:
        parts.append(y[prev:k])
        parts.append(np.zeros(n, dtype=np.float32))
        prev = k
    parts.append(y[prev:])
    return np.concatenate(parts).astype(np.float32, copy=False)
