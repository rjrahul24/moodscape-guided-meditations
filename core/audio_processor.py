"""Audio FX chains using Spotify's Pedalboard."""

import numpy as np
from pedalboard import (
    Compressor,
    HighShelfFilter,
    Limiter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)


def make_voice_chain(reverb_amount: float = 0.15) -> Pedalboard:
    """FX chain for the narration voice: warmth, compression, subtle reverb."""
    return Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=300, gain_db=2.0),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        Reverb(room_size=0.3, damping=0.7, wet_level=reverb_amount, dry_level=1.0 - reverb_amount),
        Limiter(threshold_db=-1.0),
    ])


def make_music_chain() -> Pedalboard:
    """FX chain for background music: warm low end, tamed highs."""
    return Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=1.5),
        HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-3.0),
        Limiter(threshold_db=-1.0),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering limiter."""
    return Pedalboard([
        Limiter(threshold_db=-0.5),
    ])


def apply_fx(
    audio: np.ndarray,
    chain: Pedalboard,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply a Pedalboard FX chain to mono audio.

    Args:
        audio: 1D float32 numpy array (mono).
        chain: Pedalboard instance.
        sample_rate: Audio sample rate in Hz.

    Returns:
        1D float32 numpy array (mono), same length as input.
    """
    # Pedalboard expects shape (channels, samples)
    audio_2d = audio.astype(np.float32).reshape(1, -1)
    processed = chain(audio_2d, sample_rate)
    return processed.squeeze(0).astype(np.float32)
