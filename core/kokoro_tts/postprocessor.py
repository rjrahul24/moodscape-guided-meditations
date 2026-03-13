"""Kokoro TTS postprocessing pipeline — artifact removal, normalization, and FX.

Three processing stages tailored to Kokoro-82M's specific output characteristics:

  Stage 1 — Per-chunk cleanup (called on each raw audio slice from KPipeline):
    - Hard-clip guard (clamp to [-1, 1])
    - Trailing silence trim (RMS windowing)
    - Repetition-loop detection (spectral flatness / Wiener entropy)
    - RMS normalization to -23 dBFS (EBU R128 speech reference)

  Stage 2 — Segment assembly:
    - 22ms cosine-squared crossfade between chunks (preserves consonant onsets)
    - 25ms pre-roll silence + 100ms fade-in (masks cold-start prosody drift)
    - 50ms fade-out at segment boundary

  Stage 3 — Pedalboard FX chains (Kokoro-specific signal processing):
    - Phase A: Neural denoising (resemble-enhance, pre-reverb at 24 kHz)
    - Voice chain: noise gate → highpass → compression → HF de-harsh → reverb → limiter
    - Phase B: Mastering EQ at 44.1 kHz (warmth, mud cut, presence, de-esser,
      10.5 kHz lowpass for Kokoro's 12 kHz Nyquist masking, limiter)
"""

import logging

import numpy as np
import noisereduce as nr
from pedalboard import (
    Compressor,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    NoiseGate,
    PeakFilter,
    Pedalboard,
    Reverb,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

# ── Crossfade constants ──────────────────────────────────────────────────
# 22ms at 24 kHz = 528 samples. Shorter than before to reduce inter-chunk
# phoneme blending at word boundaries — improves consonant intelligibility.
CROSSFADE_SAMPLES = int(0.022 * SAMPLE_RATE)

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

    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
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
        if len(result) < crossfade_samples or len(chunk) < crossfade_samples:
            result = np.concatenate([result, chunk])
            continue

        overlap = crossfade_samples
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
    n_std_thresh: float = 1.5,
) -> np.ndarray:
    """Remove low-level synthesis hiss via stationary spectral gating.

    Lightweight alternative to the neural denoiser (resemble-enhance), which
    is disabled on Apple Silicon due to instability. Uses noisereduce's
    stationary mode — assumes a consistent noise profile across the signal,
    which matches Kokoro's ISTFTNet vocoder hiss characteristics.

    Conservative defaults (prop_decrease=0.6, n_std_thresh=1.5) preserve
    soft consonants (/h/, /f/, breathy phonemes) while still reducing the
    noise floor by ~6 dB.

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
        )
        return denoised.astype(np.float32)
    except Exception as e:
        logger.warning("Spectral gating failed: %s — returning original audio", e)
        return audio


# =====================================================================
# Stage 3: Pedalboard FX chains (Kokoro-specific)
# =====================================================================

def build_voice_chain(reverb_amount: float = 0.08) -> Pedalboard:
    """Kokoro-tailored voice FX chain: noise gate → highpass → compression → reverb → limit.

    Tuned for Kokoro's ISTFTNet vocoder characteristics:
    - Noise gate at -42 dB: mutes inter-chunk silence without gating soft
      phonemes (breathy /h/, /f/, unvoiced trailing stops)
    - Highpass at 90 Hz: removes sub-bass rumble and plosive energy
    - Compressor 2.5:1 at -18 dB: glues the track without pumping; lower ratio
      preserves consonant transients for clearer pronunciation
    - High-shelf -5.5 dB at 6.5 kHz: more aggressive de-harshening of Kokoro's
      "radio static" vocoder artifact (6–9 kHz), applied pre-reverb
    - Lowpass at 8 kHz: harder digital ceiling removes metallic TTS hallucinations
    - Reverb: subtle chamber/studio for natural room presence
    - Limiter: brickwall at -1 dBFS
    """
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    return Pedalboard([
        NoiseGate(threshold_db=-42, ratio=20.0, attack_ms=2.0, release_ms=80),
        HighpassFilter(cutoff_frequency_hz=90.0),
        Compressor(threshold_db=-18.0, ratio=2.5, attack_ms=2.0, release_ms=80.0),
        HighShelfFilter(cutoff_frequency_hz=6500, gain_db=-5.5),
        LowpassFilter(cutoff_frequency_hz=8000.0),
        Reverb(
            room_size=0.15,
            damping=0.6,
            wet_level=reverb_amount,
            dry_level=1.0 - reverb_amount,
        ),
        Limiter(threshold_db=-1.0),
    ])


def build_master_chain(sr: int = 44100) -> Pedalboard:
    """Phase B mastering chain tailored to Kokoro's 24 kHz → 44.1 kHz output.

    Signal chain:
      1. Highpass 80 Hz — remove DC offset & sub-bass rumble
      2. Warmth +2 dB @ 200 Hz low-shelf — "proximity effect" warmth
      3. Surgical -1.5 dB @ 400 Hz, +1.5 dB @ 3.5 kHz, +1.2 dB @ 2.5 kHz —
         mud cut, presence, and consonant intelligibility boost
      4. De-Esser: boost → compress → cut at 7 kHz — tame sibilance
         ±4 dB (reduced from ±6 to avoid triggering on every sibilant)
      5. Lowpass 9.5 kHz — psychoacoustic "super-resolution" smoothing:
         Kokoro TTS runs at 24 kHz native (Nyquist: 12 kHz), upsampled to
         44.1 kHz. This creates a smooth, analog-sounding rolloff in the
         10.5–12 kHz region, masking the abrupt digital brick-wall at the
         12 kHz Nyquist boundary. Sibilance (7 kHz) is already handled
         by the de-esser above.
      6. Limiter -0.5 dB — prevent overs on final output
    """
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
        PeakFilter(cutoff_frequency_hz=400, gain_db=-1.5, q=1.0),
        PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5, q=0.8),
        PeakFilter(cutoff_frequency_hz=2500, gain_db=1.2, q=1.2),
        PeakFilter(cutoff_frequency_hz=7000, gain_db=4.0, q=2.0),
        Compressor(threshold_db=-20, ratio=3.0, attack_ms=1, release_ms=50),
        PeakFilter(cutoff_frequency_hz=7000, gain_db=-4.0, q=2.0),
        LowpassFilter(cutoff_frequency_hz=9500),
        Limiter(threshold_db=-0.5),
    ])


class KokoroMasteringEngine:
    """Two-phase mastering engine tailored to Kokoro TTS output.

    Phase A — restore_vocals(): AI neural denoising at native 24 kHz (pre-reverb)
    Phase B — master_vocals(): EQ / de-ess / limiting at 44.1 kHz (post-mix)
    """

    def __init__(self, device: str = "mps", sample_rate: int = SAMPLE_RATE):
        self.device = device
        self.sample_rate = sample_rate
        self.enhancer_fn = None
        self._master_chain: Pedalboard | None = None
        self._master_chain_sr: int | None = None

    def restore_vocals(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Phase A: AI denoising via resemble-enhance. Run on dry audio BEFORE reverb."""
        import torch

        if sr is None:
            sr = self.sample_rate

        if self.enhancer_fn is None:
            return audio

        try:
            tensor_wav = torch.tensor(audio, dtype=torch.float32)
            with torch.no_grad():
                restored, _ = self.enhancer_fn(
                    tensor_wav, sr, device=self.device,
                    nfe=32, solver="euler", lambd=0.5, tau=0.5,
                )
            return restored.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error("resemble-enhance failed: %s — falling back to unprocessed audio", e)
            return audio

    def master_vocals(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Phase B: Kokoro-tailored mastering EQ, de-ess, and limit the final mix.

        Should be called AFTER resampling to 44.1 kHz. Uses Kokoro-specific
        10.5 kHz lowpass to mask the 12 kHz Nyquist brick-wall.
        """
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = build_master_chain(sr)
            self._master_chain_sr = sr

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        result = processed.squeeze(0)
        return np.clip(result, -1.0, 1.0).astype(np.float32)


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
