"""Mono-to-stereo upmixing for meditation music.

Applies the Haas effect to create a pseudo-stereo field from mono music.
Voice stays mono (center-panned); only the music track gets width.

The Haas effect works by adding a short delay (15-18ms) to one channel.
The ear perceives the delayed channel as coming from a different direction,
creating a sense of width without phase cancellation on mono playback.
"""

import numpy as np


def haas_stereo(
    audio_mono: np.ndarray,
    sample_rate: int = 48000,
    delay_ms: float = 16.0,
    right_gain: float = 0.92,
) -> np.ndarray:
    """Create pseudo-stereo via Haas effect.

    Args:
        audio_mono: 1D mono float32 array.
        sample_rate: Sample rate in Hz.
        delay_ms: Delay for right channel in milliseconds (15-18ms optimal).
        right_gain: Amplitude multiplier for delayed channel (0.90-0.95).

    Returns:
        Stereo float32 array of shape (2, samples).
    """
    if audio_mono.ndim != 1:
        raise ValueError(f"Expected 1D mono array, got shape {audio_mono.shape}")

    delay_samples = int(delay_ms * sample_rate / 1000.0)

    left = audio_mono.copy()
    right = np.zeros_like(audio_mono)

    # Right channel = delayed copy at reduced amplitude
    if delay_samples < len(audio_mono):
        right[delay_samples:] = audio_mono[:-delay_samples] * right_gain

    return np.stack([left, right], axis=0).astype(np.float32)


def center_pan_voice(
    voice_mono: np.ndarray,
) -> np.ndarray:
    """Duplicate mono voice to both channels for center panning.

    Args:
        voice_mono: 1D mono float32 array.

    Returns:
        Stereo float32 array of shape (2, samples) with identical L/R.
    """
    if voice_mono.ndim != 1:
        raise ValueError(f"Expected 1D mono array, got shape {voice_mono.shape}")

    return np.stack([voice_mono, voice_mono], axis=0).astype(np.float32)
