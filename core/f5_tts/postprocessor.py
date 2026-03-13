"""F5-TTS postprocessing — crossfade assembly and vocal mastering chain.

F5-TTS uses the Vocos vocoder, which produces a cleaner signal than Kokoro's
ISTFTNet. No neural denoising or spectral gating is needed. The mastering
chain raises the lowpass ceiling to 13 kHz to preserve Vocos's broader native
bandwidth (vs. Kokoro's 9.5 kHz cap which masks its 12 kHz Nyquist artefact).
"""

import numpy as np
from pedalboard import (
    Compressor,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    PeakFilter,
    Pedalboard,
)

SAMPLE_RATE = 24000
CROSSFADE_SAMPLES = int(0.020 * SAMPLE_RATE)  # 20 ms linear crossfade


def crossfade_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Stitch audio chunks with a 20 ms linear crossfade at each boundary.

    Uses a linear (rather than cosine-squared) crossfade because F5-TTS
    chunks are already segmented at clean sentence boundaries, so the
    simpler fade is sufficient and avoids amplitude pumping.
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
        fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        overlap = result[-fade:] * fade_out + c[:fade] * fade_in
        result = np.concatenate([result[:-fade], overlap, c[fade:]])
    return result


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

        Signal chain (identical to KokoroMasteringEngine except lowpass is
        raised to 13 kHz to preserve Vocos's broader native bandwidth):

            HighpassFilter(80 Hz)          — remove sub-bass rumble
            LowShelfFilter(200 Hz, +2 dB)  — add warmth
            PeakFilter(400 Hz, -1.5 dB)    — cut low-mid mud
            PeakFilter(3.5 kHz, +1.5 dB)   — presence / intelligibility
            PeakFilter(7 kHz, +4 dB)       — air / clarity boost
            Compressor(-20 dB, 3:1)        — gentle dynamic control
            PeakFilter(7 kHz, -4 dB)       — de-ess (cancel the air boost)
            LowpassFilter(13 kHz)          — remove ultrasonic content
            Limiter(-0.5 dB)               — hard ceiling

        The chain is rebuilt only when the sample rate changes between calls.
        """
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=80),
                LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
                PeakFilter(cutoff_frequency_hz=400, gain_db=-1.5, q=1.0),
                PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5, q=0.8),
                PeakFilter(cutoff_frequency_hz=7000, gain_db=4.0, q=2.0),
                Compressor(threshold_db=-20, ratio=3.0, attack_ms=1, release_ms=50),
                PeakFilter(cutoff_frequency_hz=7000, gain_db=-4.0, q=2.0),
                LowpassFilter(cutoff_frequency_hz=13000),
                Limiter(threshold_db=-0.5),
            ])
            self._master_chain_sr = sr

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)
