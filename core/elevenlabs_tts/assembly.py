"""Pure DSP for assembling ElevenLabs chunks into one 48 kHz voice track."""

from __future__ import annotations

import os
import warnings

import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def chunk_lufs_target() -> float | None:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "-21").strip().lower()
    return None if raw == "off" else float(raw)


def normalize_chunk_lufs(y: np.ndarray, sr: int, target_lufs: float,
                         gate_lufs: float = -50.0) -> np.ndarray:
    """Match each chunk's integrated loudness; passthrough (same object) when
    the chunk is < 400 ms or effectively silent, so near-silence is never boosted."""
    if y.size < int(sr * 0.4):
        return y
    import pyloudnorm as pyln

    measured = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float32)))
    if not np.isfinite(measured) or measured < gate_lufs:
        return y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = pyln.normalize.loudness(y.astype(np.float32), measured, target_lufs)
    return out.astype(np.float32, copy=False)


def activity_mask(y: np.ndarray, sr: int, frame_ms: float = 20.0,
                  rel_db: float = -35.0, hangover_ms: float = 150.0) -> np.ndarray:
    """True where speech is active: frame RMS within rel_db of the chunk's loudest
    frame, extended forward by hangover_ms. Gaps inside a chunk read inactive so
    the music bed can rise there."""
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=bool)
    hop = max(1, int(sr * frame_ms / 1000.0))
    frames = -(-n // hop)
    padded = np.zeros(frames * hop, dtype=np.float32)
    padded[:n] = y
    rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    peak = float(db.max())
    if peak < -60.0:
        return np.zeros(n, dtype=bool)
    active = db > peak + rel_db
    hang = int(np.ceil(hangover_ms / frame_ms))
    if hang > 0:
        active = np.convolve(active.astype(np.int32), np.ones(hang + 1, np.int32))[:frames] > 0
    return np.repeat(active, hop)[:n]


def crossfade_concat(pieces: list[np.ndarray], masks: list[np.ndarray], sr: int,
                     xfade_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Join pieces with equal-power crossfades; masks overlap with OR so
    len(mask) == len(audio) always."""
    xn = max(1, int(sr * xfade_ms / 1000.0))
    out = pieces[0].astype(np.float32, copy=True)
    mask = masks[0].astype(bool, copy=True)
    for y, m in zip(pieces[1:], masks[1:]):
        y = y.astype(np.float32, copy=False)
        n = min(xn, len(out), len(y))
        if n <= 1:
            out = np.concatenate([out, y])
            mask = np.concatenate([mask, m])
            continue
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        cross = out[-n:] * np.cos(t * np.pi / 2) + y[:n] * np.sin(t * np.pi / 2)
        out = np.concatenate([out[:-n], cross, y[n:]])
        mask = np.concatenate([mask[:-n], mask[-n:] | m[:n], m[n:]])
    return out.astype(np.float32, copy=False), mask
