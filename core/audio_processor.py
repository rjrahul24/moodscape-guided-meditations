"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import numpy as np
from pedalboard import (
    Compressor,
    LowpassFilter,
    NoiseGate,
    PeakFilter,
    Limiter,
    Pedalboard,
    Reverb,
)


def make_voice_chain(reverb_amount: float = 0.15) -> Pedalboard:
    """FX chain for narration: compression → reverb → limit.

    Warmth / EQ is handled by MasteringEngine.master_vocals() at 44.1 kHz,
    so this chain focuses only on dynamics and spatial effects.

    Args:
        reverb_amount: Reverb wet level (0.0 = dry, 0.5 = very wet).
                       Exposed as a Gradio slider (default 0.15).
    """
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        NoiseGate(threshold_db=-40, ratio=10.0, attack_ms=1.0, release_ms=50),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=10, release_ms=100),
        Reverb(
            room_size=0.3,
            damping=0.7,
            wet_level=reverb_amount,
            dry_level=1.0 - reverb_amount,
        ),
        Limiter(threshold_db=-1.0),
    ])


def make_music_chain() -> Pedalboard:
    """FX chain for MusicGen output: warm low end → tamed highs → limit.

    The HighShelfFilter is critical — it tames MusicGen's 'digital shimmer'
    in the 8kHz+ range and creates spectral space for the voice.
    """
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=1500, gain_db=-2.5, q=0.5),  # vocal pocket
        LowpassFilter(cutoff_frequency_hz=10000),                    # gentle HF rolloff
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
    """Apply a Pedalboard FX chain to a mono audio array.

    Handles all the shape manipulation internally.
    Input and output are both 1D float32 arrays.
    """
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    audio_2d = audio.reshape(1, -1)
    processed = chain(audio_2d, sample_rate)
    result = processed.squeeze(0)
    result = result[:len(audio)]  # Trim reverb tail
    return np.clip(result, -1.0, 1.0).astype(np.float32)
