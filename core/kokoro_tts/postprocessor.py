"""Kokoro TTS postprocessing pipeline — artifact removal, normalization, and FX.

Three processing stages tailored to Kokoro-82M's specific output characteristics:

  Stage 1 — Per-chunk cleanup (called on each raw audio slice from KPipeline):
    - DC offset removal (subtract mean)
    - Hard-clip guard (clamp to [-1, 1])
    - Trailing silence trim (RMS windowing)
    - Repetition-loop detection (spectral flatness / Wiener entropy)
    - RMS normalization to -23 dBFS (EBU R128 speech reference)

  Stage 2 — Segment assembly:
    - 300ms cosine-squared crossfade between chunks (smooth blend through silence)
    - 25ms pre-roll silence + 100ms fade-in (masks cold-start prosody drift)
    - 50ms fade-out at segment boundary

  Stage 3 — Unified voice chain (single-pass Pedalboard FX):
    Professional signal flow: cleanup → EQ → dynamics → space → protection
    - NoiseGate → HPF → warmth shelf → mud cut → compressor → presence →
      HF de-harsh → convolution reverb → Nyquist LPF → limiter
"""

import logging

import numpy as np
import noisereduce as nr
from pedalboard import (
    Compressor,
    HighpassFilter,
    Convolution,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    NoiseGate,
    PeakFilter,
    Pedalboard,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# ── Crossfade constants ──────────────────────────────────────────────────
# 300ms at 24 kHz = 7200 samples. Meditation audio has 0.8–1.2s inter-sentence
# pauses, so crossfade regions overlap silence — no intelligibility risk.
# 300ms eliminates any residual spectral discontinuity at chunk seams.
CROSSFADE_SAMPLES = int(0.300 * SAMPLE_RATE)

# ── Segment-level fade constants ─────────────────────────────────────────
# Masks Kokoro's "cold start" prosody drift in the first ~100ms of each call
PRE_ROLL_SEC = 0.025
FADE_IN_SEC = 0.100
FADE_OUT_SEC = 0.050

# ── Silence detection thresholds ─────────────────────────────────────────
_SILENCE_THRESHOLD_LINEAR = 10 ** (-45.0 / 20)  # -45 dBFS ≈ 0.00562
_SILENCE_WINDOW_SAMPLES = int(0.020 * SAMPLE_RATE)  # 20ms windows
_MIN_TAIL_SAMPLES = int(0.020 * SAMPLE_RATE)

# ── Spectral flatness (repetition-loop detection) ────────────────────────
_FLATNESS_THRESHOLD = 0.85
_FLATNESS_MIN_TAIL_SAMPLES = int(0.200 * SAMPLE_RATE)  # 200ms
_FLATNESS_FFT_SIZE = 2048


# =====================================================================
# Stage 1: Per-chunk cleanup
# =====================================================================

def _spectral_flatness(frame: np.ndarray) -> float:
    """Wiener entropy (spectral flatness) of a short audio frame.

    Returns a value in [0, 1]:
      ~0 = tonal / speech-like
      ~1 = noise-like / flat spectrum (possible repetition artifact)
    """
    win = np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(frame * win, n=_FLATNESS_FFT_SIZE))
    spectrum = np.where(spectrum < 1e-10, 1e-10, spectrum)
    log_mean = np.mean(np.log(spectrum))
    arithmetic_mean = np.mean(spectrum)
    if arithmetic_mean < 1e-10:
        return 0.0
    return float(np.exp(log_mean) / arithmetic_mean)


def trim_tts_artifacts(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Strip trailing silence and repetition-loop artifacts from a TTS chunk.

    Conservative by design: only modifies audio when conditions are clearly
    anomalous. Normal narration output is returned unchanged.

    Steps:
      1. Hard-clip guard — clamp any out-of-range samples.
      2. Trailing silence trim — strip near-zero RMS windows from the tail,
         preserving a minimum 20ms tail for crossfades.
      3. Spectral flatness check — if the trimmed tail (≥200ms) has very
         high spectral flatness (noise-like), trim that region too.
    """
    if len(audio) == 0:
        return audio

    audio = audio.astype(np.float32)
    audio -= np.mean(audio)  # Remove DC offset before clipping — prevents clicks at chunk boundaries
    audio = np.clip(audio, -1.0, 1.0)
    original_len = len(audio)

    # Trailing silence trim
    keep_end = len(audio)
    win = _SILENCE_WINDOW_SAMPLES
    while keep_end - win >= _MIN_TAIL_SAMPLES:
        window = audio[keep_end - win : keep_end]
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < _SILENCE_THRESHOLD_LINEAR:
            keep_end -= win
        else:
            break

    keep_end = max(keep_end, _MIN_TAIL_SAMPLES)
    silence_trimmed = original_len - keep_end
    if silence_trimmed > 0:
        audio = audio[:keep_end]
        logger.debug("[TrimArtifacts] trimmed %dms trailing silence",
                     int(silence_trimmed / sr * 1000))

    # Spectral flatness check
    tail_samples = min(_FLATNESS_MIN_TAIL_SAMPLES * 4, len(audio))
    if tail_samples >= _FLATNESS_MIN_TAIL_SAMPLES:
        tail = audio[-tail_samples:]
        flatness = _spectral_flatness(tail)
        if flatness > _FLATNESS_THRESHOLD:
            trim_end = len(audio) - tail_samples
            trim_end = max(trim_end, _MIN_TAIL_SAMPLES)
            loop_trimmed = len(audio) - trim_end
            audio = audio[:trim_end]
            logger.debug("[TrimArtifacts] trimmed %dms flat-spectrum repetition tail (flatness=%.3f)",
                         int(loop_trimmed / sr * 1000), flatness)

    return audio


def normalize_chunk_rms(
    chunk: np.ndarray,
    target_db: float = -23.0,
    min_rms_db: float = -60.0,
) -> np.ndarray:
    """RMS-normalize a single TTS chunk to a consistent loudness level.

    Called on every audio slice yielded by the Kokoro generator before
    crossfading. Prevents volume jumps at crossfade boundaries caused by
    energy-mismatched slices (stressed phrases vs. soft trailing clauses).

    Args:
        chunk:      1-D float32 audio at SAMPLE_RATE.
        target_db:  Target RMS in dBFS (-23 dBFS = EBU R128 speech reference).
        min_rms_db: Chunks quieter than this are returned unchanged.
    """
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 10 ** (min_rms_db / 20):
        return chunk
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    gain = np.clip(gain, 10 ** (-12 / 20), 10 ** (12 / 20))
    return np.clip(chunk * gain, -1.0, 1.0).astype(np.float32)


def process_chunk(audio: np.ndarray) -> np.ndarray:
    """Full per-chunk cleanup: artifact trim → RMS normalize.

    Single entry point for Stage 1 processing on each raw audio slice
    from KPipeline before crossfading.
    """
    cleaned = trim_tts_artifacts(audio.astype(np.float32), sr=SAMPLE_RATE)
    return normalize_chunk_rms(cleaned)


# =====================================================================
# Stage 2: Segment assembly (crossfade + edge fades)
# =====================================================================

def crossfade_chunks(chunks: list[np.ndarray], crossfade_samples: int = CROSSFADE_SAMPLES) -> np.ndarray:
    """Stitch audio chunks with cosine-squared crossfade at boundaries.

    Uses equal-power cosine-squared curves so that energy is preserved
    through the crossfade region.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for chunk in chunks[1:]:
        # Clamp to the shortest safe overlap so very short chunks still get a fade
        overlap = min(crossfade_samples, len(result) // 2, len(chunk) // 2)
        if overlap < 2:
            result = np.concatenate([result, chunk])
            continue

        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)).astype(np.float32) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)).astype(np.float32) ** 2

        blended = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
        result = np.concatenate([result[:-overlap], blended, chunk[overlap:]])

    return result


def apply_segment_fades(speech_audio: np.ndarray) -> np.ndarray:
    """Apply pre-roll silence + cosine fade-in/out to mask cold-start drift.

    The 80ms fade-in masks Kokoro's cold-start prosody drift — the first
    ~80ms of each fresh inference call can have a slightly different
    pitch/energy baseline. A cosine curve keeps the transition smooth.
    The 25ms pre-roll ensures the ramp starts from true zero.
    """
    pre_roll = np.zeros(int(PRE_ROLL_SEC * SAMPLE_RATE), dtype=np.float32)
    speech_audio = np.concatenate([pre_roll, speech_audio])

    fade_in_samples = int(FADE_IN_SEC * SAMPLE_RATE)
    fade_out_samples = int(FADE_OUT_SEC * SAMPLE_RATE)

    if len(speech_audio) > fade_in_samples + fade_out_samples:
        t_in = np.linspace(np.pi, 2 * np.pi, fade_in_samples)
        f_in = ((1 - np.cos(t_in)) / 2).astype(np.float32)
        t_out = np.linspace(0, np.pi, fade_out_samples)
        f_out = ((1 + np.cos(t_out)) / 2).astype(np.float32)
        speech_audio[:fade_in_samples] *= f_in
        speech_audio[-fade_out_samples:] *= f_out

    return speech_audio


# =====================================================================
# Stage 2b: Spectral gating noise reduction
# =====================================================================

def reduce_synthesis_noise(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    prop_decrease: float = 0.6,
    n_std_thresh: float = 2.0,
) -> np.ndarray:
    """Remove low-level synthesis hiss via stationary spectral gating.

    Lightweight alternative to the neural denoiser (resemble-enhance), which
    is disabled on Apple Silicon due to instability. Uses noisereduce's
    stationary mode — assumes a consistent noise profile across the signal,
    which matches Kokoro's ISTFTNet vocoder hiss characteristics.

    Conservative defaults (prop_decrease=0.6, n_std_thresh=2.0) preserve
    soft consonants (/h/, /f/, breathy phonemes) while still reducing the
    noise floor by ~4–6 dB. The freq_mask_smooth_hz=500 parameter smooths
    the spectral gate to prevent sharp on/off ringing ("musical noise").

    Args:
        audio:         1-D float32 audio at sr.
        sr:            Sample rate (default 24 kHz).
        prop_decrease: How much to reduce detected noise (0.0–1.0).
                       Lower = more conservative. 0.6 is safe for speech.
        n_std_thresh:  Number of standard deviations above noise mean to
                       consider as "signal". Higher = more conservative.
    """
    if len(audio) < sr * 0.1:  # skip very short clips (<100ms)
        return audio

    try:
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh,
            freq_mask_smooth_hz=500,
        )
        return denoised.astype(np.float32)
    except Exception as e:
        logger.warning("Spectral gating failed: %s — returning original audio", e)
        return audio


# =====================================================================
# Stage 2b+: Room-tone pause generation
# =====================================================================


def generate_room_tone(
    duration_sec: float,
    sr: int = SAMPLE_RATE,
    level_db: float = -55.0,
) -> np.ndarray:
    """Generate low-level bandpass-filtered noise for natural-sounding pauses.

    Dead silence between sentences sounds unnatural — professional meditation
    audio maintains room-tone continuity. This generates barely perceptible
    noise (default -55 dBFS) bandpass-filtered to 100–800 Hz with cosine
    fade-in/out to avoid clicks.

    Args:
        duration_sec: Pause duration in seconds.
        sr: Sample rate (default 24 kHz).
        level_db: Noise floor level in dBFS (-55 = barely perceptible).

    Returns:
        float32 noise array at the specified sample rate.
    """
    n_samples = int(duration_sec * sr)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    # Generate white noise at target level
    level_linear = 10 ** (level_db / 20.0)
    noise = np.random.randn(n_samples).astype(np.float32) * level_linear

    # Bandpass filter 100–800 Hz (voice frequency range for room ambience)
    from scipy.signal import butter, sosfilt
    nyquist = sr / 2.0
    low = 100.0 / nyquist
    high = min(800.0 / nyquist, 0.99)
    sos = butter(2, [low, high], btype='band', output='sos')
    noise = sosfilt(sos, noise).astype(np.float32)

    # Re-normalize to target level after filtering
    rms = float(np.sqrt(np.mean(noise ** 2)))
    if rms > 1e-10:
        noise *= level_linear / rms

    # Cosine fade-in/out (20ms each) to avoid clicks
    fade_samples = min(int(0.020 * sr), n_samples // 2)
    if fade_samples > 0:
        fade_in = (0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_samples)))).astype(np.float32)
        fade_out = (0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_samples)))).astype(np.float32)
        noise[:fade_samples] *= fade_in
        noise[-fade_samples:] *= fade_out

    return noise


# =====================================================================
# Stage 2c: Pitch humanization & formant warmth (pyworld)
# =====================================================================

_PYWORLD_AVAILABLE = None  # lazy check


def _check_pyworld() -> bool:
    """Lazy-check pyworld availability. Warns once if missing."""
    global _PYWORLD_AVAILABLE
    if _PYWORLD_AVAILABLE is None:
        try:
            import pyworld  # noqa: F401
            _PYWORLD_AVAILABLE = True
        except ImportError:
            _PYWORLD_AVAILABLE = False
            logger.warning(
                "pyworld not installed — pitch humanization and formant warmth "
                "disabled. Install with: pip install 'pyworld>=0.3.4'"
            )
    return _PYWORLD_AVAILABLE


def humanize_voice(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    drift_hz: float = 0.5,
    drift_cents: float = 6.0,
    vibrato_hz: float = 5.0,
    vibrato_cents: float = 3.0,
    jitter_cents: float = 2.0,
    formant_shift: float = 0.97,
) -> np.ndarray:
    """Add micro-pitch variation and formant warmth in a single pyworld pass.

    Natural speech contains three layers of pitch variation that TTS lacks:
    slow drift (~0.5 Hz, ±6 cents from vocal fold tension changes), subtle
    vibrato (~5 Hz, ±3 cents present in sustained vowels), and random
    micro-jitter (±2 cents from neural noise). Combined modulation stays
    under ±15 cents to avoid a trembling quality.

    Formant warmth lowers formants by 3% (shift_ratio=0.97), simulating a
    slightly larger vocal tract for perceived warmth. Keep under 5% to
    preserve vowel identity.

    Both transforms share a single pyworld analysis/resynthesis pass for
    efficiency.

    Args:
        audio: 1-D float32 audio at sr.
        sr: Sample rate (default 24 kHz).
        drift_hz: Slow drift frequency in Hz.
        drift_cents: Slow drift amplitude in cents.
        vibrato_hz: Vibrato frequency in Hz.
        vibrato_cents: Vibrato amplitude in cents.
        jitter_cents: Random jitter amplitude in cents.
        formant_shift: Spectral envelope warp ratio (0.97 = 3% lower formants).

    Returns:
        Humanized audio at the same sample rate. Returns input unchanged if
        pyworld is unavailable or audio is too short.
    """
    if not _check_pyworld():
        return audio
    if len(audio) < sr * 0.5:  # skip clips < 500ms
        return audio

    import pyworld as pw
    from scipy.ndimage import gaussian_filter1d

    audio_f64 = audio.astype(np.float64)

    # ── pyworld analysis (single pass) ──
    f0, t = pw.harvest(audio_f64, sr)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)
    ap = pw.d4c(audio_f64, f0, t, sr)

    # ── Pitch humanization ──
    voiced = f0 > 0
    n = len(f0)
    frame_period = pw.default_frame_period / 1000.0  # ms → sec
    t_frames = np.arange(n) * frame_period

    # Layer 1: Slow drift (simulates vocal fold tension variation)
    drift = drift_cents * np.sin(2 * np.pi * drift_hz * t_frames)

    # Layer 2: Subtle vibrato with rate variation
    vib_rate = vibrato_hz + 0.5 * np.sin(2 * np.pi * 0.1 * t_frames)
    vib_phase = np.cumsum(vib_rate * frame_period) * 2 * np.pi
    vibrato = vibrato_cents * np.sin(vib_phase)

    # Layer 3: Random micro-jitter (Gaussian-smoothed)
    jitter = gaussian_filter1d(np.random.randn(n) * jitter_cents, sigma=3)

    total_cents = (drift + vibrato + jitter) * voiced
    f0_mod = np.where(voiced, f0 * 2 ** (total_cents / 1200.0), f0)

    # ── Formant warmth (spectral envelope warp) ──
    sp_warped = np.zeros_like(sp)
    for i in range(sp.shape[1]):
        src_idx = int(i * formant_shift)
        if src_idx < sp.shape[1]:
            sp_warped[:, i] = sp[:, src_idx]
        else:
            sp_warped[:, i] = sp[:, -1]

    # ── Resynthesize (single pass) ──
    result = pw.synthesize(f0_mod, sp_warped, ap, sr)

    # Match original length (pyworld can shift by a few samples)
    if len(result) > len(audio):
        result = result[:len(audio)]
    elif len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))

    return result.astype(np.float32)


# =====================================================================
# Stage 3: Pedalboard FX chains (Kokoro-specific)
# =====================================================================

def build_voice_chain(reverb_amount: float = 0.08, ir_name: str = "warm_studio") -> Pedalboard:
    """Unified Kokoro voice chain — single-pass processing with professional signal flow.

    Replaces the previous two-chain architecture (master_vocals + voice FX) which
    caused over-processing: double HF shelving (-5 dB cumulative at 7-8 kHz),
    double 300 Hz cuts (-3.5 dB), and four cascading dynamic processors.

    Signal flow (cleanup → EQ → dynamics → space → protection):
      1. NoiseGate -42 dB: mutes inter-chunk silence without gating soft
         phonemes (breathy /h/, /f/, unvoiced trailing stops)
      2. HPF 80 Hz: removes sub-bass rumble and plosive energy
      3. LowShelf +2.0 dB @ 200 Hz: warmth / proximity effect
      4. Peak -2 dB @ 350 Hz (Q=1.0): single mud cut for ISTFTNet boxy
         resonance (replaces dual cuts at 300 Hz and 400 Hz)
      5. Compressor 2:1 @ -18 dB: gentle glue compression
      6. Peak +1.0 dB @ 3 kHz (Q=0.6): broad presence for intelligibility
      7. HiShelf -3.0 dB @ 7.5 kHz: single de-harsh shelf for vocoder
         artifacts (replaces dual shelves at 7 kHz + 8 kHz = -5 dB)
      8. Convolution reverb: real IR for natural room presence
      9. LPF 9.5 kHz: Nyquist masking AFTER reverb (so reverb tails are
         filtered too — previous chain applied this BEFORE reverb)
     10. Limiter -1 dBFS: single peak limiter
    """
    from core.audio_processor import IR_CATALOG, DEFAULT_IR

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]

    return Pedalboard([
        # ── Cleanup ──
        NoiseGate(threshold_db=-42, ratio=20.0, attack_ms=2.0, release_ms=80),
        HighpassFilter(cutoff_frequency_hz=80.0),
        # ── Tone shaping ──
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
        PeakFilter(cutoff_frequency_hz=350, gain_db=-2.0, q=1.0),
        # ── Dynamics: gentle glue compression ──
        Compressor(threshold_db=-18, ratio=2.0, attack_ms=10.0, release_ms=100.0),
        # ── Presence & de-harsh ──
        PeakFilter(cutoff_frequency_hz=3000, gain_db=1.0, q=0.6),
        HighShelfFilter(cutoff_frequency_hz=7500, gain_db=-3.0),
        # ── Space ──
        Convolution(
            impulse_response_filename=ir_path,
            mix=reverb_amount,
        ),
        # ── Protection ──
        LowpassFilter(cutoff_frequency_hz=9500),
        Limiter(threshold_db=-1.0),
    ])


def apply_fx(
    audio: np.ndarray,
    chain: Pedalboard,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply a Pedalboard FX chain to an audio array (1D or 2D float32)."""
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
    result = result[..., :audio.shape[-1]]
    return np.clip(result, -1.0, 1.0).astype(np.float32)
