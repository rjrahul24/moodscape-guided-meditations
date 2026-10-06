"""Output conditioning for Qwen3-TTS chunks, plus the (shared) voice chain.

Qwen's codec-decoded output is close to Chatterbox's in character, so the
mastering engine and voice FX chain reuse Chatterbox's dry chain.
"""

import numpy as np

from core.chatterbox_tts.postprocessor import (
    ChatterboxMasteringEngine as Qwen3MasteringEngine,
    build_chatterbox_voice_chain as build_qwen3_voice_chain,
)

TARGET_SR = 24000
_TRIM_DBFS = -45.0
_FADE_MS = 5.0


def condition_chunk(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 24 kHz, trim edge silence, apply short edge fades."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != TARGET_SR:
        from core.audio_processor import resample_highly_accurate
        x = resample_highly_accurate(x, sr, TARGET_SR).astype(np.float32)
    if x.size == 0:
        return x
    loud = np.flatnonzero(np.abs(x) > 10 ** (_TRIM_DBFS / 20.0))
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = x[loud[0]: loud[-1] + 1].copy()
    fade = min(len(x) // 2, int(TARGET_SR * _FADE_MS / 1000.0))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Optional pacing arm (rate < 1 slows). Identity at 1.0."""
    if rate == 1.0:
        return audio
    import librosa
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate).astype(np.float32)


__all__ = ["Qwen3MasteringEngine", "build_qwen3_voice_chain", "condition_chunk", "time_stretch"]
