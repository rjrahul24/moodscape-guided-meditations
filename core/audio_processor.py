"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import numpy as np
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LowShelfFilter,
    PeakFilter,
    Limiter,
    Pedalboard,
    LowpassFilter,
)


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
        PeakFilter(cutoff_frequency_hz=5500, gain_db=0.8, q=0.6),     # Clarity/air presence
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-3.0),    # HF artefact suppression
        Limiter(threshold_db=-1.0),
    ])


def make_acestep_music_chain() -> Pedalboard:
    """FX chain tailored for ACE-Step 1.5 clean VAE output for meditation use.

    Tuned for the clean postprocess chain (no tanh/kernel pre-filtering) and
    the -14 LUFS pre-mix target (Task 5):
      1. Sub-bass HPF at 60 Hz — removes diffusion noise rumble (30–60 Hz band)
      2. Low-shelf warmth at 200 Hz (+2.0 dB) — enveloping warmth; conservative
         increment over the former +1.5 dB to avoid low-mid muddiness at the
         hotter -14 LUFS premix level
      3. Upper-mid softening at 4 kHz (-1.5 dB) — keeps music behind narration
      4. Gentle HF shelf at 10 kHz (-1.0 dB) — smooths the digital edge
      5. Glue compressor — threshold -20 dB / 2.5:1 / 80ms attack / 800ms release.
         Lower threshold (-20 dB) ensures engagement on quieter ambient passages;
         2.5:1 ratio gives tighter macro-dynamic control without squashing pads;
         800ms release is slower and more meditative than the former 500ms.
      6. Limiter at -0.5 dBFS
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60.0),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
        HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=-1.0),
        Compressor(
            threshold_db=-20.0,
            ratio=2.5,
            attack_ms=80.0,
            release_ms=800.0,
        ),
        Limiter(threshold_db=-0.5),
    ])


def make_lyria_music_chain() -> Pedalboard:
    """FX chain tailored for Lyria RealTime output at 48 kHz.

    Lyria's diffusion model produces stereo audio that has been averaged to
    mono by the engine.  Its spectral profile differs from both MusicGen and
    ACE-Step:
    - The 48 kHz native rate means content extends above 12 kHz (unlike
      MusicGen's 32→24 kHz pipeline, whose Nyquist sits at 12 kHz).
    - Lyria tends to be brighter and more harmonically dense than ACE-Step's
      ambient output, so upper-mid softening is more aggressive.
    - Sub-bass rumble is present but not as severe as ACE-Step diffusion noise.

    Chain:
      1. Sub-bass HPF at 60 Hz — removes inaudible energy from the API stream
         that wastes mix headroom.
      2. Mud notch at 250 Hz (-1.5 dB) — controls warmth buildup common in
         Lyria's harmonic-rich textures.
      3. Upper-mid edge at 4500 Hz (-2.0 dB, Q=0.7) — tones down any 'presence'
         that makes the ambient bed sound too forward against the narration.
      4. High shelf at 9000 Hz (-2.5 dB) — gentle rolloff of Lyria's extended
         high-frequency content to maintain a meditative warmth.
      5. Slow glue compressor (2:1, 500ms release) — same gentle compression as
         the ACE-Step chain; prevents dynamic spikes without audible pumping.
      6. Limiter at -0.5 dBFS — leaves extra headroom before the master chain.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=60.0),                         # Sub-bass removal
        PeakFilter(cutoff_frequency_hz=250, gain_db=-1.5, q=0.8),         # Mud notch
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=-2.5),        # HF warmth rolloff
        Compressor(
            threshold_db=-18.0,
            ratio=2.0,
            attack_ms=80.0,    # Slow attack: preserve transients
            release_ms=500.0,  # Slow release: no pumping on sustained pads
        ),
        Limiter(threshold_db=-0.5),
    ])


def make_vocal_pocket_chain() -> Pedalboard:
    """Consolidated EQ chain to carve a spectral 'lane' for human speech.
    
    Human speech intelligibility resides primarily in the 1-3 kHz range.
    By carving this pocket in the music track, the voice sits naturally
    on top without needing excessive volume boosts.
    
    Chain:
      1. HighpassFilter at 30 Hz: Remove sub-bass rumble.
      2. 300 Hz Peak (-3 dB, Q=0.8): Clear low-mid mud buildup.
      3. 1000 Hz Peak (-2 dB, Q=0.7): Open space for vowel fundamentals.
      4. 3000 Hz Peak (-4 dB, Q=1.0): Presence pocket for sibilance/consonants.
      5. LowpassFilter at 12000 Hz: Tame harsh highs and digital artifacts.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        PeakFilter(cutoff_frequency_hz=300, gain_db=-3, q=0.8),
        PeakFilter(cutoff_frequency_hz=1000, gain_db=-2, q=0.7),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=-4, q=1.0),
        LowpassFilter(cutoff_frequency_hz=12000),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering chain: subsonic HPF → glue compression → gain → peak limiter.
    
    Tuned for gentle 'glue' and peak control rather than loudness maximization:
      1. HPF at 30 Hz: Remove subsonic MusicGen rubble.
      2. Compressor (-24 dB threshold, 1.5:1 ratio): Subtle glue to bind voice 
         and music together. 30ms attack preserves transients.
      3. Gain (+1 dB): Makeup gain for the compressor.
      4. Limiter at -1.5 dBFS: Ensures safe true peak levels for lossy codecs.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        Compressor(
            threshold_db=-24.0,
            ratio=1.5,
            attack_ms=30.0,
            release_ms=300.0,
        ),
        Gain(gain_db=1.0),
        Limiter(threshold_db=-1.5, release_ms=200.0),
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
