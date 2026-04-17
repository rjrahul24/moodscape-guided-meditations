"""Audio FX chains using Spotify's Pedalboard — MoodScape."""

import os

import numpy as np
from pedalboard import (
    Compressor,
    Convolution,
    HighpassFilter,
    HighShelfFilter,
    LowShelfFilter,
    NoiseGate,
    PeakFilter,
    Limiter,
    Pedalboard,
    LowpassFilter,
)
import librosa

# ---------------------------------------------------------------------------
# Impulse Response (IR) catalog for convolution reverb
# ---------------------------------------------------------------------------
_IR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "impulse_responses")

IR_CATALOG = {
    "warm_studio": {
        "path": os.path.join(_IR_DIR, "warm_studio.wav"),
        "label": "Warm Studio (intimate, short decay)",
    },
    "wooden_hall": {
        "path": os.path.join(_IR_DIR, "wooden_hall.wav"),
        "label": "Wooden Hall (natural warmth, medium space)",
    },
    "stone_chapel": {
        "path": os.path.join(_IR_DIR, "stone_chapel.wav"),
        "label": "Stone Chapel (ethereal, long decay)",
    },
}
DEFAULT_IR = "warm_studio"


def make_heartmula_music_chain() -> Pedalboard:
    """FX chain tailored for HeartMuLa / HeartCodec output at 48 kHz.

    HeartMuLa/HeartCodec produces 48 kHz stereo output (downmixed to mono
    in the engine).  Tuned for studio-grade meditation quality with the
    following design principles:

    - Suppress codec quantization noise (HeartCodec 12.5 Hz token rate)
    - Fletcher-Munson bass compensation for low-volume listening (~50-60 dBSPL)
    - Clarity cuts in low-mid and upper-mid (cuts are more transparent than boosts)
    - Add grounding warmth via dual low shelves (drone-heavy output benefits)
    - LPF at 14 kHz — HeartCodec's 12.5 Hz frame rate produces limited useful
      content above 8 kHz; gentle LPF removes HF codec artifacts
    - Slow compressor dynamics for a meditative, non-pumping envelope

    Chain:
      1. NoiseGate (-52 dB, 2:1) — suppress codec quantization noise.
         With lower codec_guidance_scale (1.25), output is smoother and
         cleaner, so the gate threshold can be higher than -55 dB.
      2. HighpassFilter at 60 Hz — removes low-frequency codec noise.
      3. LowShelfFilter at 100 Hz (+1.5 dB) — grounding warmth for
         drone-heavy content.
      4. LowShelfFilter at 150 Hz (+2.0 dB) — Fletcher-Munson bass
         compensation.  At 50-60 dBSPL (typical meditation listening),
         ISO 226 contours show bass perception drops ~20 dB relative to
         midrange.  This shelf restores perceived low-end warmth.
      5. PeakFilter at 220 Hz (-1.0 dB, Q=0.7) — clarity cut; removes
         low-mid mud from HeartCodec harmonic output.
      6. PeakFilter at 4000 Hz (-2.0 dB, Q=0.5) — tame upper-mid
         brightness; transparent presence notch.  Wide Q (0.5) for a
         gentle, unhearable cut that keeps music behind narration.
      7. HighShelfFilter at 9500 Hz (-2.0 dB) — warmer HF rolloff for
         headphone listening during meditation.
      8. LowpassFilter at 14000 Hz — remove HF codec artifacts above the
         useful spectral range for ambient music.
      9. Compressor (-20 dB, 2:1, 100ms attack, 900ms release) — slow,
         meditative dynamics; prevents pumping on sustained pads.
     10. Convolution reverb (stone_chapel IR, 15% wet) — meditation
          spaciousness.  Cathedral/hall IRs add immersive depth that
          masks subtle codec artifacts.
     11. Compressor (-20 dB, 2:1, 100ms attack, 900ms release) — slow,
          meditative dynamics; prevents pumping on sustained pads.
     12. Limiter at -0.5 dBFS.
    """
    # Build chain with convolution reverb if IR file exists
    ir_path = IR_CATALOG.get("stone_chapel", {}).get("path", "")
    plugins = [
        NoiseGate(
            threshold_db=-52.0,
            ratio=2.0,
            attack_ms=5.0,
            release_ms=200.0,
        ),
        HighpassFilter(cutoff_frequency_hz=60.0),
        LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=1.5),
        LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=2.0),
        PeakFilter(cutoff_frequency_hz=220, gain_db=-1.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=4000, gain_db=-2.0, q=0.5),
        HighShelfFilter(cutoff_frequency_hz=9500.0, gain_db=-2.0),
        LowpassFilter(cutoff_frequency_hz=14000.0),
    ]
    # Add convolution reverb for meditation spaciousness (if IR available)
    if os.path.isfile(ir_path):
        plugins.append(Convolution(ir_path, mix=0.15))
    plugins.extend([
        Compressor(
            threshold_db=-20.0,
            ratio=2.0,
            attack_ms=100.0,
            release_ms=900.0,
        ),
        Limiter(threshold_db=-0.5),
    ])
    return Pedalboard(plugins)


def make_acestep_music_chain() -> Pedalboard:
    """Minimal FX chain for ACE-Step 1.5 — preserves natural VAE output quality.

    ACE-Step's VAE produces clean audio that doesn't need heavy processing.
    This chain applies only essential corrections:

      1. Noise gate (-55 dB) — catches quiet diffusion residual noise.
      2. Sub-bass HPF at 60 Hz — removes diffusion rumble.
      3. Low-shelf warmth at 200 Hz (+1.5 dB) — subtle bass presence.
      4. Mild presence cut at 3 kHz (-1.5 dB) — creates space for voice;
         vocal pocket adds -1.5 dB more = -3 dB combined.
      5. Ultrasonic LPF at 16 kHz — removes diffusion noise above audible range.
      6. Gentle glue compressor — 2:1 / 80ms attack / 800ms release.
      7. Limiter at -0.5 dBFS.
    """
    plugins = [
        NoiseGate(
            threshold_db=-55.0,
            ratio=2.0,
            attack_ms=15.0,
            release_ms=100.0,
        ),
        HighpassFilter(cutoff_frequency_hz=60.0),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
        PeakFilter(cutoff_frequency_hz=2500, gain_db=-2.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=4500, gain_db=-2.0, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=1.5),
        LowpassFilter(cutoff_frequency_hz=16000.0),
        Compressor(
            threshold_db=-20.0,
            ratio=2.0,
            attack_ms=80.0,
            release_ms=800.0,
        ),
    ]
    # Add warm_studio convolution reverb for spaciousness and to mask diffusion artifacts.
    # 8% wet is subtle — primarily adds depth and warmth rather than audible room effect.
    ir_path = IR_CATALOG.get("warm_studio", {}).get("path", "")
    if os.path.isfile(ir_path):
        plugins.append(Convolution(ir_path, mix=0.08))
    plugins.append(Limiter(threshold_db=-1.0))
    return Pedalboard(plugins)


def make_lyria_music_chain() -> Pedalboard:
    """FX chain tailored for Lyria RealTime output at 48 kHz.

    Lyria's diffusion model produces stereo audio that has been averaged to
    mono by the engine.  Its spectral profile differs from both MusicGen and
    ACE-Step:
    - The 48 kHz native rate means content extends above 12 kHz, with full
      bandwidth up to ~22 kHz.
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
        PeakFilter(cutoff_frequency_hz=4500, gain_db=-2.0, q=0.7),        # Upper-mid presence control
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
    """Frequency-aware EQ to carve a spectral lane for voice intelligibility.

    Research-spec vocal pocket: deeper cuts in the voice presence range
    (800-3000 Hz) create clear space for the narrator without needing
    aggressive ducking. The ear is most sensitive at 1-3 kHz (Fletcher-
    Munson curves), so even modest cuts here dramatically improve
    perceived voice clarity.

    Chain:
      1. HighpassFilter at 30 Hz: Remove sub-bass rumble.
      2. 350 Hz Peak (-3 dB, Q=0.7): Voice body/warmth range — wider Q
         creates a gradual scoop across 200-500 Hz.
      3. 1500 Hz Peak (-3.5 dB, Q=0.7): Primary vocal presence range.
         Research: -3 to -4 dB @ 800-3kHz. This is the single most
         effective cut for making voice punch through music.
      4. 3000 Hz Peak (-2 dB, Q=0.8): Upper harmonics of speech
         consonants. Prevents sibilance masking.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        PeakFilter(cutoff_frequency_hz=350, gain_db=-3.0, q=0.7),
        PeakFilter(cutoff_frequency_hz=1500, gain_db=-3.5, q=0.7),
        PeakFilter(cutoff_frequency_hz=3000, gain_db=-2.0, q=0.8),
    ])


def make_master_chain() -> Pedalboard:
    """Final mastering chain: subsonic HPF → bus compressor → peak limiter.

    Chain (per audio-opt research):
      1. HPF 30 Hz — removes subsonic rumble below the audible range.
      2. Compressor 1.5:1 @ -22 dB, 40ms/300ms — gentle 'glue' compression
         that bonds voice and music into a cohesive mix (~1-2 dB GR max).
         At 1.5:1 with a -22 dB threshold, this is inaudible as a compressor
         but adds perceived cohesion between TTS and music stems.
      3. Limiter -1.0 dBTP, 400ms release — true peak safety margin for
         lossy codec encoding. Research: -1.0 dBTP for meditation content.
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        Compressor(threshold_db=-22.0, ratio=1.5, attack_ms=40.0, release_ms=300.0),
        Limiter(threshold_db=-1.0, release_ms=400.0),
    ])


def reduce_music_noise(
    audio: np.ndarray,
    sample_rate: int = 48000,
    prop_decrease: float = 0.7,
    n_fft: int = 512,
) -> np.ndarray:
    """Stationary noise reduction for music engine artifacts.

    Research-spec parameters: prop_decrease=0.7 for aggressive gating,
    n_fft=512 for finer spectral resolution. Neural music generators
    produce consistent artifact patterns that spectral gating handles well.

    Must be applied BEFORE the EQ chain so that artifact energy is removed
    before the compressor amplifies quiet passages.
    """
    import noisereduce as nr

    if audio.shape[-1] < int(sample_rate * 0.5):
        return audio  # Too short for reliable noise profile estimation

    try:
        cleaned = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=prop_decrease,
            n_fft=n_fft,
            freq_mask_smooth_hz=300,
        )
        return cleaned.astype(np.float32)
    except Exception:
        return audio


def apply_tape_saturation(
    audio: np.ndarray,
    drive: float = 0.3,
    bias: float = 0.15,
) -> np.ndarray:
    """Asymmetric soft clipping with even-order harmonic bias.

    Transforms clinical AI output into warmer, more natural-sounding audio
    by adding subtle harmonic distortion weighted toward even-order harmonics
    (2nd, 4th — octave relationships that the ear perceives as richness).

    Args:
        audio: float32 array.
        drive: Saturation amount (0.0–1.0). 0.3 is subtle warmth.
        bias: DC offset bias for even-order harmonics (0.0–0.5).
    """
    peak_in = np.abs(audio).max()
    if peak_in < 1e-8:
        return audio

    x = audio * (1.0 + drive)
    saturated = np.tanh(x + bias) - np.tanh(np.float32(bias))

    # Preserve original peak level
    peak_out = np.abs(saturated).max()
    if peak_out > 1e-8:
        saturated = saturated * (peak_in / peak_out)

    return saturated.astype(np.float32)


def add_organic_noise_floor(
    audio: np.ndarray,
    sample_rate: int = 48000,
    noise_db: float = -58.0,
    lpf_hz: float = 8000.0,
) -> np.ndarray:
    """Add shaped pink noise floor for analog warmth.

    AI-generated audio has an unnaturally clean noise floor that sounds
    clinical.  Adding pink noise at -58 dB (subliminally present, consciously
    inaudible) emulates the character of analog recording equipment.
    Real 1/4" tape measures ~-65 dB RMS; -58 dB is slightly warmer.
    """
    from scipy.signal import butter, sosfilt

    n_samples = audio.shape[-1]
    rng = np.random.default_rng(42)

    # Generate pink noise via cumulative sum of white noise (1/f approximation)
    white = rng.standard_normal(n_samples).astype(np.float32)
    pink = np.cumsum(white)
    pink = pink - np.mean(pink)
    peak = np.abs(pink).max()
    if peak > 1e-6:
        pink = pink / peak

    # LPF to remove harsh high frequencies
    sos = butter(4, lpf_hz, btype="low", fs=sample_rate, output="sos")
    pink = sosfilt(sos, pink).astype(np.float32)

    # Re-normalize after filtering
    peak = np.abs(pink).max()
    if peak > 1e-6:
        pink = pink / peak

    # Scale to target dB relative to audio peak
    audio_peak = np.abs(audio).max()
    if audio_peak < 1e-8:
        return audio
    noise_level = audio_peak * (10.0 ** (noise_db / 20.0))
    pink = pink * noise_level

    return (audio + pink).astype(np.float32)


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
    applying any Pedalboard FX or mixing audio at different sample rates.
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


def resample_highly_accurate(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Standard sinc interpolation for high-fidelity resampling.

    Uses librosa's 'soxr_vhq' which offers mathematically zero artifacts,
    retaining the original signal integrity without adding frequency content.
    Ideal for upsampling 24kHz F5-TTS output to 48kHz studio standards for
    clean mixing with background music.
    """
    if orig_sr == target_sr:
        return audio
    
    # librosa.resample works on float32/64 numpy arrays.
    # 'soxr_vhq' is the Very High Quality sinc interpolation.
    resampled = librosa.resample(
        audio.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="soxr_vhq",
    )
    return resampled.astype(np.float32)


def upsample_audio(
    audio: np.ndarray,
    from_sr: int = 24000,
    to_sr: int = 48000,
    high_accuracy: bool = False,
) -> np.ndarray:
    """Upsample audio for higher-fidelity output.

    Defaults to torchaudio's Kaiser-windowed sinc interpolation.
    If high_accuracy=True, uses librosa's soxr_vhq (standard sinc interpolation)
    for artifact-free upsampling, as recommended for F5-TTS meditation speech.
    
    Supports 1D and 2D arrays.
    """
    if high_accuracy:
        if audio.ndim == 1:
            return resample_highly_accurate(audio, from_sr, to_sr)
        else:
            # Handle multi-channel by resampling each channel
            resampled_channels = [
                resample_highly_accurate(audio[i], from_sr, to_sr)
                for i in range(audio.shape[0])
            ]
            return np.stack(resampled_channels).astype(np.float32)

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
