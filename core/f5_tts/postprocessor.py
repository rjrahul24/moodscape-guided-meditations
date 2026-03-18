"""F5-TTS postprocessing — crossfade assembly and vocal mastering chain.

F5-TTS uses the Vocos vocoder, which produces a cleaner signal than Kokoro's
ISTFTNet. No neural denoising or spectral gating is needed. The mastering
chain raises the lowpass ceiling to 10.5 kHz to preserve Vocos's broader native
bandwidth (vs. Kokoro's 9.5 kHz cap which masks its 12 kHz Nyquist artefact).
"""

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

SAMPLE_RATE = 24000
CROSSFADE_SAMPLES = int(0.300 * SAMPLE_RATE)  # 300 ms equal-power cosine crossfade


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
    center_freq: float = 6000.0,
    bandwidth: float = 4000.0,
    threshold_db: float = -18.0,
    ratio: float = 4.0,
) -> np.ndarray:
    """Dynamic split-band de-esser using scipy for crossover and Pedalboard for compression.

    Isolates the sibilant band (typically 4-8 kHz), compresses it aggressively
    with a fast attack/release, and sums it back with the untouched non-sibilant signal.
    This preserves high-end brilliance while taming harsh 's' and 't' transients.
    """
    # 1. Isolate the sibilant band using a 4th-order Butterworth bandpass (SOS for stability)
    nyquist = sr / 2.0
    low = (center_freq - bandwidth / 2.0) / nyquist
    high = (center_freq + bandwidth / 2.0) / nyquist
    sos = butter(4, [low, high], btype="band", output="sos")

    sibilant_band = sosfiltfilt(sos, audio)
    non_sibilant = audio - sibilant_band

    # 2. Compress the sibilant band
    # Pedalboard expects (channels, samples)
    comp = Compressor(
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=0.5,
        release_ms=10.0,
    )
    s_2d = sibilant_band.reshape(1, -1) if sibilant_band.ndim == 1 else sibilant_band
    compressed_sibilant = comp(s_2d, sr).squeeze(0)

    # 3. Recombine
    return (non_sibilant + compressed_sibilant).astype(np.float32)


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

        Signal chain (tuned for Vocos — lowpass raised to 10.5 kHz to preserve
        broader native bandwidth, no algorithmic reverb since convolution
        reverb is applied downstream in build_f5_voice_chain):

            Phase A — split_band_deess(): dynamic sibilance control (preprocessing)
            Phase B — master_vocals(): EQ / dynamics at mix sample rate.

        Signal chain (tuned for F5-TTS / Vocos meditation narration):

            NoiseGate(-45 dB, 2:1)         — catch diffusion residual noise
            HighpassFilter(80 Hz)          — remove sub-bass rumble
            PeakFilter(300 Hz, -2 dB)      — anti-boxiness (cut low-mid mud)
            LowShelfFilter(200 Hz, +2 dB)  — add warmth
            PeakFilter(3.2 kHz, +1.5 dB)   — presence / intelligibility
            HighShelfFilter(8 kHz, -1.0)   — tame brightness / diffusion hiss
            LowpassFilter(10.5 kHz)        — remove diffusion artifacts above useful bandwidth
            Compressor(-20 dB, 2.5:1)      — gentle, transparent leveling
            Limiter(-1.5 dB)               — safe, transparent ceiling

        The chain is rebuilt only when the sample rate changes between calls.
        """
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = Pedalboard([
                NoiseGate(threshold_db=-45, ratio=2.0, attack_ms=5, release_ms=250),
                HighpassFilter(cutoff_frequency_hz=80),
                PeakFilter(cutoff_frequency_hz=300, gain_db=-2.0, q=1.5),
                LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
                PeakFilter(cutoff_frequency_hz=3200, gain_db=1.5, q=0.8),
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.0),
                LowpassFilter(cutoff_frequency_hz=10500),
                Compressor(threshold_db=-20, ratio=2.5, attack_ms=15, release_ms=100),
                Limiter(threshold_db=-1.5, release_ms=80),
            ])
            self._master_chain_sr = sr

        # Phase A: Dynamic De-Essing (Preprocessing)
        audio = split_band_deess(audio, sr)

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)


def build_f5_voice_chain(reverb_amount: float = 0.15, ir_name: str = "warm_studio") -> Pedalboard:
    """F5-TTS / Vocos voice FX chain: convolution reverb + limiter only.

    F5MasteringEngine.master_vocals() already handles EQ, de-essing, and
    dynamic control upstream. This chain adds the user-controlled convolution
    reverb (real IR for natural room presence) and a safety limiter — nothing
    else. Deliberately omits the noise gate, 8 kHz lowpass, and high-shelf
    present in Kokoro's build_voice_chain, which are tuned for ISTFTNet
    artifacts and would damage Vocos's 10.5 kHz native bandwidth.
    """
    from core.audio_processor import IR_CATALOG, DEFAULT_IR

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]

    return Pedalboard([
        Convolution(
            impulse_response_filename=ir_path,
            mix=reverb_amount,
        ),
        Limiter(threshold_db=-1.0),
    ])
