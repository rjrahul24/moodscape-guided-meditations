"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import numpy as np
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    LowpassFilter,
    NoiseGate,
    PeakFilter,
    Limiter,
    Pedalboard,
    Reverb,
)


def make_voice_chain(reverb_amount: float = 0.09) -> Pedalboard:
    """FX chain for narration: noise gate → highpass → compression → reverb → limit.

    The HighpassFilter at 80 Hz removes sub-bass rumble from TTS output.
    Warmth / presence EQ is handled by MasteringEngine.master_vocals()
    at 44.1 kHz, so this chain focuses on cleanup, dynamics, and spatial
    effects.

    Args:
        reverb_amount: Reverb wet level (0.0 = dry, 0.5 = very wet).
                       Exposed as a Gradio slider (default 0.09).
    """
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        # Strict noise gate: mutes TTS static in silences *before* compression
        # and LUFS normalization can amplify it.
        NoiseGate(threshold_db=-35, ratio=20.0, attack_ms=1.0, release_ms=50),
        # Remove sub-bass rumble and plosives from TTS output
        HighpassFilter(cutoff_frequency_hz=80.0),
        # Gentle compression to keep the guiding voice perfectly steady
        Compressor(threshold_db=-19.0, ratio=3.5, attack_ms=10.0, release_ms=200.0),
        # Subtle reverb so the voice sounds like it is in a physical room
        Reverb(
            room_size=0.17,
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
        LowpassFilter(cutoff_frequency_hz=3000.0),                    # gentle HF rolloff
        Limiter(threshold_db=-1.0),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering chain: gain → glue compressor → brickwall limiter.

    The Compressor at 2:1 ratio with slow attack/release acts as a soft-knee
    "glue" — gently reining in peaks from music swells without the harsh,
    pumping sound of aggressive limiting alone.
    """
    return Pedalboard([
        Gain(gain_db=-3.0),
        # Bus compressor: gentle 2:1 glue before the brickwall limiter
        Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=30.0, release_ms=300.0),
        Limiter(threshold_db=-0.1),
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


def resample_to_44100(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to the 44.1kHz studio standard.
    
    Uses torchaudio for high-quality, efficient resampling. Required before
    applying any Pedalboard FX or mixing Kokoro (24kHz) and MusicGen (32kHz).
    """
    import torch
    import torchaudio.functional as F
    
    if orig_sr == 44100:
        return audio
        
    tensor_audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = F.resample(
        tensor_audio, orig_sr, 44100,
        lowpass_filter_width=64,
        rolloff=0.9475,
    )
    return resampled.squeeze(0).numpy()


def upsample_audio(
    audio: np.ndarray,
    from_sr: int = 24000,
    to_sr: int = 48000,
) -> np.ndarray:
    """Upsample audio for higher-fidelity output.

    Uses torchaudio's Kaiser-windowed sinc interpolation for clean,
    alias-free sample-rate conversion. Only apply at the final export
    stage, not during intermediate processing.
    """
    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = F.resample(
        tensor, from_sr, to_sr,
        lowpass_filter_width=64,
        rolloff=0.9475,
    )
    return resampled.squeeze(0).numpy().astype(np.float32)
