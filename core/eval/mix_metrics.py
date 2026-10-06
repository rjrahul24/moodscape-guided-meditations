"""Loudness, loudness range, true peak and duration of a rendered file.

Mono files are also measured as dual-mono (the channel duplicated), which is
how stereo players reproduce them: +3.01 LU over the mono measurement. That
gap is the loudness-convention bug R5 fixes, so both numbers are recorded.
"""

import math

import numpy as np


def _finite(value: float) -> float | None:
    return round(float(value), 2) if value is not None and math.isfinite(value) else None


def _as_2d(audio: np.ndarray) -> np.ndarray:
    return audio.reshape(-1, 1) if audio.ndim == 1 else audio


def true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float | None:
    """4x-oversampled true peak in dBTP; None for digital silence."""
    from scipy.signal import resample_poly

    x = _as_2d(audio).astype(np.float64)
    up = resample_poly(x, oversample, 1, axis=0)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    if peak <= 0.0:
        return None
    return round(20.0 * math.log10(peak), 2)


def loudness(audio: np.ndarray, sr: int) -> dict:
    """Integrated LUFS (as is and as dual-mono) and loudness range (LU)."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    x = _as_2d(audio).astype(np.float64)
    measured = x[:, 0] if x.shape[1] == 1 else x
    dual = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x

    def _safe(fn, data):
        try:
            return _finite(fn(data))
        except Exception:
            return None

    return {
        "integrated_lufs": _safe(meter.integrated_loudness, measured),
        "integrated_lufs_dual_mono": _safe(meter.integrated_loudness, dual),
        "lra": _safe(meter.loudness_range, measured),
    }


def mix_metrics(path: str, *, estimated_sec: float | None = None) -> dict:
    """All mix-level metrics for one audio file."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    actual_sec = audio.shape[0] / sr
    metrics = {
        "channels": int(audio.shape[1]),
        "sample_rate": int(sr),
        "actual_sec": round(actual_sec, 3),
        **loudness(audio, sr),
        "true_peak_dbtp": true_peak_dbtp(audio, sr),
        "duration_ratio": round(actual_sec / estimated_sec, 4) if estimated_sec else None,
    }
    return metrics
