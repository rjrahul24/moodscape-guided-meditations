"""F5-TTS postprocessing — crossfade assembly and vocal mastering chain.

F5-TTS uses the Vocos vocoder, which produces a cleaner signal than Kokoro's
ISTFTNet. No neural denoising or spectral gating is needed. The mastering
chain uses a 12 kHz lowpass to preserve Vocos's broader native bandwidth
(vs. Kokoro's 9.5 kHz cap which masks its 12 kHz Nyquist artefact), with a
10 kHz air shelf for breathiness and intimacy.
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
CROSSFADE_SAMPLES = int(0.150 * SAMPLE_RATE)  # 150 ms overlap-add cosine crossfade


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
