"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import numpy as np
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LowpassFilter,
    LowShelfFilter,
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
    """FX chain for MusicGen output — spectral shaping for 24 kHz source material.

    Super-resolution strategy:
    MusicGen outputs at 32 kHz native, downsampled to 24 kHz in the engine
    (Nyquist: 12 kHz), then resampled to 44.1 kHz. The 12–22 kHz band in the
    final file is mathematically empty; the resampler's anti-alias filter
    (rolloff=0.9475) creates a natural taper from ~11.4 kHz to 12 kHz.

    The previous HighShelfFilter at 10 kHz was acting almost entirely in the
    dead zone above 12 kHz and barely touching the actual artifact zone. The
    real MusicGen artefact region — grainy autoregressive-decoding shimmer —
    sits at 8–10 kHz (just below the processing ceiling).

    Chain:
      1. 300 Hz low-end warmth: pads sound fuller without low-mid muddiness.
      2. 1800 Hz vocal-pocket notch: creates spectral space for narration to
         sit above the music during mixed sessions.
      3. 5500 Hz clarity/air presence (+0.8 dB, broad Q=0.6): psychoacoustic
         "super-resolution" — a gentle boost in the upper-presence range adds
         perceived openness and compensates for the absent 12–22 kHz octave
         that the listener's ear expects from high-fidelity content. Applied
         below the artifact zone so it adds clarity without amplifying shimmer.
      4. 8000 Hz HF shelf (-3 dB): targets the actual MusicGen artifact zone.
         Previously at 10 kHz this filter was largely inactive (above Nyquist
         of the 24 kHz intermediate); moved to 8 kHz it now attenuates the
         8–12 kHz region where autoregressive decoding leaves grainy noise.
         ACE-Step's chain already uses this same 8 kHz placement.
      5. Limiter at -1 dBFS.
    """
    return Pedalboard([
        PeakFilter(cutoff_frequency_hz=300, gain_db=2.0, q=0.7),      # Low-end warmth
        PeakFilter(cutoff_frequency_hz=1800, gain_db=-3.0, q=0.6),    # Vocal pocket notch
        PeakFilter(cutoff_frequency_hz=5500, gain_db=0.8, q=0.6),     # Clarity/air presence
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-3.0),    # HF artefact suppression
        Limiter(threshold_db=-1.0),
    ])


def make_acestep_music_chain() -> Pedalboard:
    """FX chain tailored for ACE-Step 1.5 output for meditation use.

    ACE-Step's DiT decoder has a different spectral profile than MusicGen:
    - It can leave sub-bass energy (below 60 Hz) from diffusion noise
    - Its output is already warm/mid-present — boosting 300 Hz makes it muddy
    - High-frequency diffusion artefacts sit around 8–12 kHz rather than 10 kHz+
    - Dynamic range is wider, requiring gentler, slower compression

    Chain:
      1. Sub-bass HPF at 60 Hz — removes inaudible rumble from diffusion noise
         that wastes headroom and causes the limiter to trigger early.
      2. Mud notch at 200 Hz (-2 dB) — ACE-Step ambient pads tend to
         accumulate muddiness in the 180–220 Hz range. A narrow notch
         keeps warmth while improving clarity.
      3. Upper-mid softening at 4 kHz (-1.5 dB) — tones down any slight
         'edge' or 'presence' that makes ambient music sound aggressive.
      4. Gentle HF shelf at 8 kHz (-3 dB) — smooth rolloff of the top-end
         where diffusion artefacts cluster, without dulling the mids.
      5. Very slow compression (2:1, 500ms release) — gentle 'glue' that
         tames occasional dynamic spikes from the LM planner without
         creating the pumping artefact that fast release causes on drones.
      6. Brickwall limiter at -0.5 dBFS — leaves tiny additional headroom
         vs the standard -1.0 limit for extra safety.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60.0),                        # Remove diffusion sub-bass
        PeakFilter(cutoff_frequency_hz=200, gain_db=-2.0, q=1.0),        # Mud notch
        PeakFilter(cutoff_frequency_hz=4000, gain_db=-1.5, q=0.8),       # Upper-mid edge softening
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-3.0),       # Gentle HF smoothing
        Compressor(
            threshold_db=-18.0,
            ratio=2.0,
            attack_ms=80.0,      # Slow attack: let transients through naturally
            release_ms=500.0,    # Slow release: no pumping on slow ambient pads
        ),
        Limiter(threshold_db=-0.5),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering chain: subsonic HPF → gain → glue compressor → brickwall limiter.

    The HighpassFilter at 35 Hz removes inaudible subsonic rumble from MusicGen
    output that would otherwise consume digital headroom, causing the limiter
    to trigger earlier and produce a quieter or more compressed final mix.

    The Compressor at 2:1 ratio with slow attack/release acts as a soft-knee
    "glue" — gently reining in peaks from music swells without the harsh,
    pumping sound of aggressive limiting alone.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=35.0),   # Remove inaudible MusicGen rumble
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
    """Apply a Pedalboard FX chain to an audio array.

    Supports both 1D (mono) and 2D (stereo) float32 arrays.
    """
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    is_1d = audio.ndim == 1
    
    if is_1d:
        audio_2d = audio.reshape(1, -1)
    else:
        audio_2d = audio
        
    processed = chain(audio_2d, sample_rate)
    
    if is_1d:
        result = processed.squeeze(0)
    else:
        result = processed
        
    result = result[..., :audio.shape[-1]]  # Trim reverb tail
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def resample_to_44100(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to the 44.1kHz studio standard.
    
    Uses torchaudio for high-quality, efficient resampling. Required before
    applying any Pedalboard FX or mixing Kokoro (24kHz) and MusicGen (32kHz).
    Supports 1D and 2D arrays.
    """
    import torch
    import torchaudio.functional as F
    
    if orig_sr == 44100:
        return audio
        
    tensor_audio = torch.from_numpy(audio.astype(np.float32))
    is_1d = tensor_audio.ndim == 1
    if is_1d:
        tensor_audio = tensor_audio.unsqueeze(0)
        
    resampled = F.resample(
        tensor_audio, orig_sr, 44100,
        lowpass_filter_width=64,
        rolloff=0.9475,
    )
    
    if is_1d:
        return resampled.squeeze(0).numpy()
    return resampled.numpy()


def upsample_audio(
    audio: np.ndarray,
    from_sr: int = 24000,
    to_sr: int = 48000,
) -> np.ndarray:
    """Upsample audio for higher-fidelity output.

    Uses torchaudio's Kaiser-windowed sinc interpolation for clean,
    alias-free sample-rate conversion.
    Supports 1D and 2D arrays.
    """
    import torch
    import torchaudio.functional as F

    tensor_audio = torch.from_numpy(audio.astype(np.float32))
    is_1d = tensor_audio.ndim == 1
    if is_1d:
        tensor_audio = tensor_audio.unsqueeze(0)
        
    resampled = F.resample(
        tensor_audio, from_sr, to_sr,
        lowpass_filter_width=64,
        rolloff=0.9475,
    )
    
    if is_1d:
        return resampled.squeeze(0).numpy().astype(np.float32)
    return resampled.numpy().astype(np.float32)
