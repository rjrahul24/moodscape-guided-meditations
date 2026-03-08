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


def make_voice_chain(reverb_amount: float = 0.08) -> Pedalboard:
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
        # Noise gate: mutes inter-chunk silence & residual TTS static before
        # compression amplifies it.  Threshold raised to -42 dB (was -35) so
        # soft phonemes like breathy /h/, /f/, and unvoiced trailing stops
        # are NOT gated — gating these produces the "clipped / robotic" quality
        # on consonants.  Ratio kept high (20:1) to still close firmly on true
        # silence and near-silence.
        NoiseGate(threshold_db=-42, ratio=20.0, attack_ms=2.0, release_ms=80),
        # Remove sub-bass rumble and plosives from TTS output (80-100Hz range)
        HighpassFilter(cutoff_frequency_hz=90.0),
        # Soft compression to gently glue dynamics without pumping or lifting the noise floor.
        # Threshold: -20dB, Ratio: 2:1 — lighter than before to avoid pulling up inter-word
        # static. Attack: 5ms lets transients breathe; Release: 100ms prevents pumping.
        Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=5.0, release_ms=100.0),
        # High-shelf de-harshener at 5 kHz, -4.5 dB (pre-reverb).
        # Targeting the broader "presence/sizzle" band (5–9 kHz) where Kokoro's ISTFTNet
        # vocoder concentrates harshness. Applying HERE (before reverb) treats the artifact
        # at its source. The existing 9 kHz LowpassFilter above handles the Nyquist
        # brick-wall artifact; this shelf covers the lower harshness range.
        HighShelfFilter(cutoff_frequency_hz=5000, gain_db=-4.5),
        # Low-pass filter (de-essing) above 9 kHz to tame metallic TTS hallucinations
        LowpassFilter(cutoff_frequency_hz=9000.0),
        # Subtle chamber/studio reverb so the voice sounds like it is perfectly recorded in a physical room
        # Room size 0.15 (15%), Damping 0.6 to retain some vocal air
        Reverb(
            room_size=0.15,
            damping=0.6,
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
        PeakFilter(cutoff_frequency_hz=1500, gain_db=-5.0, q=0.3),    # Vocal pocket notch (wider, deeper carve)
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
        PeakFilter(cutoff_frequency_hz=4500, gain_db=-2.0, q=0.7),        # Upper-mid softening
        HighShelfFilter(cutoff_frequency_hz=9000.0, gain_db=-2.5),        # HF warmth rolloff
        Compressor(
            threshold_db=-18.0,
            ratio=2.0,
            attack_ms=80.0,    # Slow attack: preserve transients
            release_ms=500.0,  # Slow release: no pumping on sustained pads
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
        Limiter(threshold_db=-1.0),
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
