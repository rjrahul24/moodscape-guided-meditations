"""F5-TTS postprocessing — crossfade assembly and vocal mastering chain.

F5-TTS uses the Vocos vocoder, which produces a cleaner signal than Kokoro's
ISTFTNet. No neural denoising or spectral gating is needed. The mastering
chain uses a 12 kHz lowpass to preserve Vocos's broader native bandwidth
(vs. Kokoro's 9.5 kHz cap which masks its 12 kHz Nyquist artefact), with a
10 kHz air shelf for breathiness and intimacy.
"""

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt
from pedalboard import (
    Compressor,
    Convolution,
    HighpassFilter,
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
CROSSFADE_SAMPLES = int(0.150 * SAMPLE_RATE)  # 150 ms overlap-add cosine crossfade


# ---------------------------------------------------------------------------
# Microprosody — phrase-final pitch declination + breathiness (research B4)
# ---------------------------------------------------------------------------

_PYWORLD_AVAILABLE = None


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
                "pyworld not installed — F5 microprosody disabled. "
                "Install with: pip install 'pyworld>=0.3.4'"
            )
    return _PYWORLD_AVAILABLE


def _apply_f0_declination(
    f0: np.ndarray,
    frame_period_s: float,
    tail_ms: float = 600.0,
    decline_cents: float = 120.0,
) -> np.ndarray:
    """Glide the final ``tail_ms`` of the pitch contour downward.

    Multiplies an exponential decay (1.0 → ``2^(-decline_cents/1200)``) onto the
    last ``tail_ms`` of ``f0`` so each phrase ends on a downward pitch trajectory
    — the natural breath-group declination that signals relaxation. Only voiced
    frames (``f0 > 0``) are moved; the head of the contour is untouched.
    """
    out = f0.astype(np.float64).copy()
    n = out.shape[0]
    if n == 0 or decline_cents <= 0.0:
        return out
    tail_frames = int(round((tail_ms / 1000.0) / max(frame_period_s, 1e-6)))
    tail_frames = max(1, min(tail_frames, n))
    end_factor = 2.0 ** (-decline_cents / 1200.0)
    ramp = np.exp(np.linspace(0.0, np.log(end_factor), tail_frames))
    decay = np.ones(n)
    decay[-tail_frames:] = ramp
    voiced = out > 0
    out[voiced] = out[voiced] * decay[voiced]
    return out


def _widen_pitch(f0: np.ndarray, scale: float = 1.15) -> np.ndarray:
    """Widen the pitch range about its voiced mean for gentler rises and falls.

    For voiced frames (``f0 > 0``), ``f0 = mean + (f0 - mean) * scale``; unvoiced
    frames stay 0 and results are clamped positive. ``scale`` 1.0 is a no-op;
    1.1–1.3 adds expressive contour without sounding warbly (research Issue 3,
    subtopic 4).
    """
    out = f0.astype(np.float64).copy()
    if scale == 1.0:
        return out
    voiced = out > 0
    if not np.any(voiced):
        return out
    mean = float(out[voiced].mean())
    out[voiced] = np.maximum(mean + (out[voiced] - mean) * scale, 1.0)
    return out


def _warp_formants(sp: np.ndarray, shift: float = 0.98) -> np.ndarray:
    """Warp the spectral-envelope columns to move formants (``shift`` < 1 = warmer).

    Mirrors the column-warp convention in
    ``core/kokoro_tts/postprocessor.py::humanize_voice`` so F5 warmth matches the
    accepted Kokoro behaviour. ``shift`` 1.0 is identity; 0.98 lowers formants ~2%
    for a larger, warmer vocal-tract feel without changing pitch.
    """
    if shift == 1.0:
        return sp
    n_bins = sp.shape[1]
    warped = np.zeros_like(sp)
    for i in range(n_bins):
        src = int(i * shift)
        warped[:, i] = sp[:, src] if src < n_bins else sp[:, -1]
    return warped


def _apply_amplitude_taper(
    audio: np.ndarray,
    sr: int,
    taper_ms: float = 300.0,
    floor: float = 0.6,
) -> np.ndarray:
    """Cosine-ramp the final ``taper_ms`` of amplitude down to ``floor``.

    Adds "trailing softness" at each phrase end (the gentle fade a human guide
    gives the last word). ``taper_ms`` 0 or ``floor`` ≥ 1.0 is a no-op. Operates
    on the last axis, so mono and stereo both work.
    """
    out = audio.astype(np.float32).copy()
    if taper_ms <= 0.0 or floor >= 1.0:
        return out
    n = out.shape[-1]
    k = min(int(taper_ms / 1000.0 * sr), n)
    if k <= 0:
        return out
    t = np.linspace(0.0, np.pi, k, dtype=np.float64)
    ramp = (floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(t))).astype(np.float32)
    out[..., -k:] = out[..., -k:] * ramp
    return out


def _match_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    """Scale ``audio`` so its absolute peak equals ``target_peak``.

    WORLD analysis/resynthesis is not gain-preserving (it can add ~1–2 dB),
    which would make the voice hotter than it came in and risk intermediate
    clipping. Re-matching the input peak keeps the microprosody pass a pure
    contour/timbre change, not a level change. Silence is returned unchanged.
    """
    out = audio.astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out = out * (float(target_peak) / peak)
    return out.astype(np.float32)


def apply_microprosody(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    decline_cents: float = 120.0,
    tail_ms: float = 600.0,
    ap_scale: float = 1.05,
    pitch_scale: float = 1.15,
    formant_shift: float = 0.98,
    taper_ms: float = 300.0,
    taper_floor: float = 0.6,
) -> np.ndarray:
    """Phrase-final pitch declination + breathiness via one pyworld pass.

    Decomposes the chunk into pitch / spectral-envelope / aperiodicity, glides
    the final ``tail_ms`` of the f0 contour downward by ``decline_cents`` (the
    relaxation cue human meditation guides use), scales aperiodicity by
    ``ap_scale`` for a breathier, more intimate tone, and resynthesizes.

    Returns the input unchanged if pyworld is unavailable or the clip is too
    short. WORLD resynthesis can colour Vocos's already-clean output, so this is
    OFF by default and opted into per session via MOODSCAPE_F5_MICROPROSODY.
    Research Issue 3.3 / 3.4.
    """
    if not _check_pyworld():
        return audio
    if len(audio) < int(sr * 0.5):  # skip clips < 500 ms
        return audio

    import pyworld as pw

    audio_f64 = audio.astype(np.float64)
    f0, t = pw.harvest(audio_f64, sr)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)
    ap = pw.d4c(audio_f64, f0, t, sr)

    frame_period_s = pw.default_frame_period / 1000.0
    # Pitch: widen about the mean (expressive contour), then glide the tail down.
    f0_mod = _widen_pitch(f0, pitch_scale)
    f0_mod = _apply_f0_declination(f0_mod, frame_period_s, tail_ms, decline_cents)
    # Timbre: lower formants slightly for warmth; lift aperiodicity for breath.
    sp_mod = _warp_formants(sp, formant_shift)
    ap_mod = np.clip(ap * float(ap_scale), 0.0, 1.0)

    result = pw.synthesize(f0_mod, sp_mod, ap_mod, sr)
    if len(result) > len(audio):
        result = result[: len(audio)]
    elif len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    # Re-match the input peak so the WORLD pass doesn't inject gain, then apply
    # the phrase-final amplitude taper for trailing softness.
    in_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    result = _match_peak(result.astype(np.float32), in_peak)
    result = _apply_amplitude_taper(result, sr, taper_ms, taper_floor)
    return result.astype(np.float32)


def crossfade_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Stitch audio chunks with a 300 ms equal-power cosine crossfade at each boundary.

    Uses a cos/sin equal-power crossfade (cos²+sin²=1) which maintains constant
    perceived loudness throughout the transition, eliminating the amplitude dip
    that linear crossfades produce at the midpoint.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].copy().astype(np.float32)

    result = chunks[0].copy().astype(np.float32)
    for c in chunks[1:]:
        c = c.astype(np.float32)
        fade = min(CROSSFADE_SAMPLES, len(result), len(c))
        if fade == 0:
            result = np.concatenate([result, c])
            continue
        _t = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
        fade_out = np.cos(_t)   # 1.0 → 0.0
        fade_in  = np.sin(_t)   # 0.0 → 1.0
        overlap = result[-fade:] * fade_out + c[:fade] * fade_in
        result = np.concatenate([result[:-fade], overlap, c[fade:]])
    return result


def split_band_deess(
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Two-stage dynamic split-band de-esser using scipy for crossover and Pedalboard for compression.

    Stage 1: Split-band at 6-7 kHz, ratio 3:1, fast attack.
    Stage 2: Narrow band at 10-12 kHz, ratio 2:1, gentle compression.
    """
    def _deess_band(audio_in, center_freq, bandwidth, threshold_db, ratio):
        nyquist = sr / 2.0
        low = (center_freq - bandwidth / 2.0) / nyquist
        high = (center_freq + bandwidth / 2.0) / nyquist
        # Keep band edges strictly inside (0, 1) — at low sample rates (e.g. 24 kHz)
        # the upper sibilance band can reach Nyquist, which butter() rejects.
        low = max(low, 1e-4)
        high = min(high, 1.0 - 1e-4)
        if high <= low:
            return audio_in.astype(np.float32)
        sos = butter(4, [low, high], btype="band", output="sos")

        sibilant_band = sosfiltfilt(sos, audio_in)
        non_sibilant = audio_in - sibilant_band

        comp = Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=0.1,  # fast attack
            release_ms=10.0,
        )
        s_2d = sibilant_band.reshape(1, -1) if sibilant_band.ndim == 1 else sibilant_band
        compressed_sibilant = comp(s_2d, sr).squeeze(0)

        return (non_sibilant + compressed_sibilant).astype(np.float32)

    # Stage 1: 6-7 kHz (male/female sibilance)
    audio = _deess_band(audio, 6500.0, 1000.0, -25.0, 3.0)
    # Stage 2: 10-12 kHz (airy sibilance)
    audio = _deess_band(audio, 11000.0, 2000.0, -20.0, 2.0)
    
    return audio


class F5MasteringEngine:
    """Two-phase mastering engine for F5-TTS / Vocos output.

    Mirrors the interface of KokoroMasteringEngine so the pipeline can swap
    engines without branching in the mastering code:

        mastering_engine = F5MasteringEngine(sample_rate=SAMPLE_RATE)
        # ... upsample voice_audio to mix_sr ...
        voice_audio = mastering_engine.master_vocals(voice_audio, sr=mix_sr)

    Phase A — restore_vocals(): stub (Vocos output is pre-clean, no neural
        denoising needed).
    Phase B — master_vocals(): EQ / de-ess / limiting at the mix sample rate.
        The Pedalboard chain is cached per sample rate to avoid rebuilding on
        every call.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._master_chain: Pedalboard | None = None
        self._master_chain_sr: int | None = None

    def restore_vocals(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Phase A stub — Vocos output is already clean, no denoising needed."""
        return audio

    def master_vocals(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Phase B: EQ, de-ess, and limit at the mix sample rate.

        Signal chain (tuned for F5-TTS / Vocos meditation narration):

            Phase A — split_band_deess(): dynamic sibilance control
            Tape saturation: subtle harmonic warmth (drive=1.08)
            Phase B — EQ / dynamics:

            NoiseGate(-45 dB, 2:1)          — catch diffusion residual noise
            HighpassFilter(80 Hz)           — remove sub-bass rumble
            PeakFilter(300 Hz, -2 dB)       — anti-boxiness (cut low-mid mud)
            LowShelfFilter(200 Hz, +2 dB)   — add warmth
            PeakFilter(3.0 kHz, -2.0 dB)    — subtractive cut: metallic resonance
            HighShelfFilter(7.5 kHz, -3.0 dB) — de-harsh shelf (steep spectral tilt)
            HighShelfFilter(10 kHz, +1.0 dB) — air shelf for breathiness/intimacy
            LowpassFilter(12 kHz)           — preserve Vocos native bandwidth
            Compressor(-20 dB, 2.5:1)       — gentle, meditation-paced leveling
            Limiter(-1.5 dB)                — safe, transparent ceiling

        The chain is rebuilt only when the sample rate changes between calls.
        """
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = Pedalboard([
                NoiseGate(threshold_db=-45, ratio=2.0, attack_ms=5, release_ms=250),
                HighpassFilter(cutoff_frequency_hz=60),
                PeakFilter(cutoff_frequency_hz=400, gain_db=-2.5, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=135, gain_db=3.0),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=2.5, q=1.0),
                HighShelfFilter(cutoff_frequency_hz=7500, gain_db=-3.0),
                HighShelfFilter(cutoff_frequency_hz=10000, gain_db=1.5),
                LowpassFilter(cutoff_frequency_hz=12000),
                Compressor(threshold_db=-25, ratio=2.0, attack_ms=15, release_ms=200),
                Limiter(threshold_db=-2.0, release_ms=80),
            ])
            self._master_chain_sr = sr

        # Phase A: Dynamic De-Essing (Preprocessing)
        audio = split_band_deess(audio, sr)

        # Subtle tape saturation — adds 2nd/3rd harmonics for perceived warmth
        # without audible distortion. Mixed at ~15% wet to retain clarity.
        saturated = np.tanh(audio * 1.08) / 1.08
        audio = (audio * 0.85 + saturated * 0.15)

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)


def build_f5_voice_chain(reverb_amount: float = 0.15, ir_name: str = "warm_studio") -> Pedalboard:
    """F5-TTS / Vocos voice FX chain: convolution reverb (with Abbey Road EQ) + limiter.

    Implements the "Abbey Road trick" by placing the reverb in a parallel Mix block
    with an HPF at 300 Hz and LPF at 6 kHz on the wet return, keeping the dry
    voice completely clear of reverb mud.
    """
    from core.audio_processor import IR_CATALOG, DEFAULT_IR
    from pedalboard import Mix, Gain

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]
    
    dry_gain = 1.0 - reverb_amount
    wet_gain = reverb_amount
    
    dry_db = 20 * np.log10(max(dry_gain, 1e-5))
    wet_db = 20 * np.log10(max(wet_gain, 1e-5))

    return Pedalboard([
        Mix([
            Gain(gain_db=dry_db),
            Pedalboard([
                Convolution(
                    impulse_response_filename=ir_path,
                    mix=1.0,
                ),
                HighpassFilter(cutoff_frequency_hz=300),
                LowpassFilter(cutoff_frequency_hz=6000),
                Gain(gain_db=wet_db),
            ])
        ]),
        Limiter(threshold_db=-1.0),
    ])
