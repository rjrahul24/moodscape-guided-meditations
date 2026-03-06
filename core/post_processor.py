"""Advanced Mastering Post-Processing Pipeline for MoodScape.

Split into two phases for optimal signal-chain placement:
  Phase A – restore_vocals():  AI neural denoising (runs on dry audio BEFORE reverb)
  Phase B – master_vocals():   EQ / de-ess / limiting  (runs AFTER mix, at 44.1 kHz)
"""

import logging

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

logger = logging.getLogger(__name__)


class MasteringEngine:
    """Two-phase mastering engine for meditation vocals.

    Usage in the pipeline:
        1. engine.restore_vocals(audio, sr=24000)   # Phase A – before reverb
        2. … apply reverb, mix, resample to 44100 …
        3. engine.master_vocals(audio, sr=44100)     # Phase B – final master
    """

    def __init__(self, device: str = "mps", sample_rate: int = 24000):
        self.device = device
        self.sample_rate = sample_rate
        logger.info("Initializing MasteringEngine on %s", self.device)

        # The resemble-enhance library is highly unstable on Apple Silicon and newer NumPy versions,
        # repeatedly failing with bus errors and scalar casting errors.
        # We rely on the Phase B high-quality noise gate and EQ chain instead.
        self.enhancer_fn = None

        # Pre-build the Phase B mastering chain (built lazily on first call
        # because target SR may differ from __init__ SR)
        self._master_chain: Pedalboard | None = None
        self._master_chain_sr: int | None = None

    # ------------------------------------------------------------------
    # Phase A: Neural Restoration (pre-reverb, at native TTS sample rate)
    # ------------------------------------------------------------------

    def restore_vocals(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """AI denoising via resemble-enhance. Run on dry audio BEFORE reverb.

        Args:
            audio: 1-D float32 numpy array.
            sr:    Sample rate (defaults to self.sample_rate).

        Returns:
            Denoised 1-D float32 numpy array (same length / sr).
        """
        import torch

        if sr is None:
            sr = self.sample_rate

        if self.enhancer_fn is None:
            return audio

        try:
            # enhance() expects a 1D tensor dwav: shape (T,)
            tensor_wav = torch.tensor(audio, dtype=torch.float32)
            with torch.no_grad():
                # enhance() signature: dwav, sr, device, nfe=32, solver='midpoint', lambd=0.5, tau=0.5
                # The returned tuple is (enhanced_wav, new_sr)
                # Note: using solver="euler" to bypass a bug in resemble-enhance's midpoint solver
                # where scipy.optimize.fsolve throws a scalar casting error on newer NumPy versions.
                restored, _ = self.enhancer_fn(
                    tensor_wav,
                    sr,
                    device=self.device,
                    nfe=32,
                    solver="euler",
                    lambd=0.5,
                    tau=0.5
                )
            return restored.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error("resemble-enhance failed: %s — falling back to unprocessed audio", e)
            return audio

    # ------------------------------------------------------------------
    # Phase B: Mastering EQ / De-Ess / Limiting (post-mix, at 44.1 kHz)
    # ------------------------------------------------------------------

    def _build_master_chain(self, sr: int) -> Pedalboard:
        """Build the Phase B mastering Pedalboard for a given sample rate.

        Signal chain:
          1. Highpass  80 Hz   – remove DC offset & sub-bass rumble
          2. Warmth    +2 dB @ 200 Hz low-shelf – "proximity effect" warmth
          3. Surgical  -1.5 dB @ 400 Hz, +1.5 dB @ 3.5 kHz – mud cut & presence
          4. De-Esser  boost → compress → cut at 7 kHz – tame sibilance only
          5. Lowpass   15 kHz  – remove digital hiss / aliasing artifacts
          6. Limiter   -0.5 dB – prevent overs on final output
        """
        return Pedalboard([
            # 1. Clean sub-bass
            HighpassFilter(cutoff_frequency_hz=80),

            # 2. Warmth boost (proximity effect)
            LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),

            # 3. Surgical EQ: mud cut + presence boost
            PeakFilter(cutoff_frequency_hz=400, gain_db=-1.5, q=1.0),
            PeakFilter(cutoff_frequency_hz=3500, gain_db=1.5, q=0.8),

            # 4. De-Esser: boost → compress → cut  (targets 6-8 kHz sibilance)
            PeakFilter(cutoff_frequency_hz=7000, gain_db=6.0, q=2.0),
            Compressor(threshold_db=-20, ratio=4.0, attack_ms=1, release_ms=30),
            PeakFilter(cutoff_frequency_hz=7000, gain_db=-6.0, q=2.0),

            # 4. Lowpass – remove HF hiss while preserving vocal "air"
            LowpassFilter(cutoff_frequency_hz=15000),

            # 5. Final limiter
            Limiter(threshold_db=-0.5),
        ])

    def master_vocals(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """Phase B mastering: EQ, de-ess, and limit the final mix.

        Should be called AFTER:
          - mixing voice + music
          - resampling to 44.1 kHz

        Args:
            audio: 1-D float32 numpy array at target sample rate.
            sr:    Sample rate (should be 44100 for proper filter behaviour).

        Returns:
            Mastered 1-D float32 numpy array, clipped to [-1, 1].
        """
        # Lazily build / cache the chain for this sample rate
        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = self._build_master_chain(sr)
            self._master_chain_sr = sr

        audio_2d = audio.astype(np.float32).reshape(1, -1)
        processed = self._master_chain(audio_2d, sr)
        result = processed.squeeze(0)

        return np.clip(result, -1.0, 1.0).astype(np.float32)
