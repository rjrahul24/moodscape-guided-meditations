"""IndexTTS-2 postprocessing — vocal mastering chain and voice FX.

IndexTTS-2 uses the BigVGANv2 vocoder, which produces high-quality waveforms
at 24 kHz with slightly different spectral characteristics than Kokoro's
ISTFTNet or F5-TTS's Vocos:

  - Cleaner bass reproduction (lower HPF safe at 70 Hz vs 80 Hz)
  - Different metallic resonance peak (2.8 kHz vs F5's 3.0 kHz)
  - ~12 kHz intrinsic bandwidth ceiling (BigVGANv2 trained on [0, 12 kHz] mel
    per Lee et al. ICLR 2023); our 13 kHz LPF is a safety net, not a sculpt

The mastering chain is tuned specifically for autoregressive TTS output,
which can exhibit occasional repetition artifacts and metallic resonances
different from diffusion-based models like F5-TTS.
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


def split_band_deess(
    audio: np.ndarray,
    sr: int,
    center_freq: float = 6500.0,
    bandwidth: float = 3500.0,
    threshold_db: float = -22.0,
    ratio: float = 3.5,
) -> np.ndarray:
    """Dynamic split-band de-esser tuned for IndexTTS-2 / BigVGANv2 output.

    Centred at 6.5 kHz — the snake-activation aliasing artifacts inherent to
    BigVGANv2 (Lee et al., ICLR 2023) ring closer to 7 kHz than the raw
    sibilance peak. Threshold is slightly more relaxed (-22 dB vs F5's -20 dB)
    since BigVGANv2 produces smoother transients.
    """
    nyquist = sr / 2.0
    low = max((center_freq - bandwidth / 2.0) / nyquist, 0.01)
    high = min((center_freq + bandwidth / 2.0) / nyquist, 0.99)
    sos = butter(4, [low, high], btype="band", output="sos")

    sibilant_band = sosfiltfilt(sos, audio)
    non_sibilant = audio - sibilant_band

    comp = Compressor(
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=0.5,
        release_ms=12.0,
    )
    s_2d = sibilant_band.reshape(1, -1) if sibilant_band.ndim == 1 else sibilant_band
    compressed_sibilant = comp(s_2d, sr).squeeze(0)

    return (non_sibilant + compressed_sibilant).astype(np.float32)


class IndexTTSMasteringEngine:
    """Two-phase mastering engine for IndexTTS-2 / BigVGANv2 output.

    Mirrors the interface of F5MasteringEngine so the pipeline can swap
    engines without branching in the mastering code:

        mastering_engine = IndexTTSMasteringEngine(sample_rate=SAMPLE_RATE)
        voice_audio = mastering_engine.master_vocals(voice_audio, sr=mix_sr)

    Phase A — restore_vocals(): stub (BigVGANv2 output is clean).
    Phase B — master_vocals(): EQ / de-ess / limiting at the mix sample rate.

    Signal chain (tuned for IndexTTS-2 / BigVGANv2 meditation narration):

        split_band_deess()              — dynamic sibilance control (6.5 kHz center)
        Tape saturation (drive=1.08)    — subtle harmonic warmth
        NoiseGate(-48 dB, 2:1)          — gentle threshold preserves natural breath
        HighpassFilter(70 Hz)           — remove sub-bass rumble
        PeakFilter(350 Hz, -2.0 dB)    — anti-boxiness (BigVGANv2 low-mid resonance)
        LowShelfFilter(180 Hz, +1.5 dB) — warmth / proximity effect
        PeakFilter(2.8 kHz, -1.5 dB)   — reduce autoregressive metallic resonance
        HighShelfFilter(8 kHz, -3.0 dB) — deeper taming of snake-activation ringing
        LowpassFilter(13 kHz)           — safety net above BigVGANv2 useful bandwidth
        Compressor(-18 dB, 2:1)         — preserve dynamic range for intimacy
        Limiter(-1.5 dB)                — safe ceiling
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._master_chain: Pedalboard | None = None
        self._master_chain_sr: int | None = None

    def restore_vocals(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Phase A stub — BigVGANv2 output is already clean, no denoising needed."""
        return audio

    def master_vocals(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Phase B: EQ, de-ess, and limit at the mix sample rate."""
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = Pedalboard([
                NoiseGate(threshold_db=-48, ratio=2.0, attack_ms=5, release_ms=250),
                HighpassFilter(cutoff_frequency_hz=70),
                PeakFilter(cutoff_frequency_hz=350, gain_db=-2.0, q=1.2),
                LowShelfFilter(cutoff_frequency_hz=180, gain_db=1.5),
                PeakFilter(cutoff_frequency_hz=2800, gain_db=-1.5, q=0.8),
                # Softened from -3.0 dB / 13 kHz: with Rubber Band pacing (no
                # phase-vocoder smear) the aggressive HF taming dulled the
                # voice more than it hid artifacts. The split-band de-esser
                # upstream already handles snake-activation sibilance.
                HighShelfFilter(cutoff_frequency_hz=8000, gain_db=-1.5),
                LowpassFilter(cutoff_frequency_hz=15000),
                Compressor(threshold_db=-18, ratio=2.0, attack_ms=15, release_ms=200),
                Limiter(threshold_db=-1.5, release_ms=80),
            ])
            self._master_chain_sr = sr

        # Phase A: Dynamic De-Essing
        audio = split_band_deess(audio, sr)

        # Subtle tape saturation — adds 2nd/3rd harmonics for perceived warmth
        audio = np.tanh(audio * 1.08) / 1.08

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)


def build_index_voice_chain(reverb_amount: float = 0.15, ir_name: str = "warm_studio") -> Pedalboard:
    """IndexTTS-2 voice FX chain: convolution reverb + limiter only.

    IndexTTSMasteringEngine.master_vocals() handles EQ, de-essing, and
    dynamic control upstream. This chain adds the user-controlled convolution
    reverb (real IR for natural room presence) and a safety limiter — same
    pattern as F5-TTS's build_f5_voice_chain.
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
