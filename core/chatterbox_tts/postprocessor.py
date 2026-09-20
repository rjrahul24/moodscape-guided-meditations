"""Chatterbox TTS postprocessing — studio mastering and Abbey Road filtered space.

Chatterbox uses flow matching diffusion into an S3Gen neural vocoder, producing
rich acoustic timbre. Unlike Kokoro (which needs heavy mud cutting and ISTFTNet de-essing),
Chatterbox benefits from transparent EQ, smooth dynamic leveling, and an
Abbey Road filtered convolution reverb bus that keeps the voice clear of echo and mud.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from pedalboard import (
    Compressor,
    Convolution,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    Mix,
    PeakFilter,
    Pedalboard,
)

logger = logging.getLogger("moodscape.chatterbox_postprocessor")

SAMPLE_RATE = 24000


class ChatterboxMasteringEngine:
    """Mastering engine for Chatterbox TTS output at the mix sample rate (48 kHz).

    Applies transparent, meditation-tailored signal flow:
      1. HPF 70 Hz: strip sub-bass rumble without thinning vocal body.
      2. PeakFilter 350 Hz (-1.5 dB, Q=1.0): clean up lower-mid boxiness.
      3. LowShelf 150 Hz (+1.2 dB): gentle chest warmth.
      4. HighShelf 10 kHz (+1.0 dB): silky air and intimacy.
      5. LowpassFilter 12 kHz: smooth, relaxing top-end rolloff.
      6. Compressor (-24 dB, 1.8:1, attack 20ms, release 250ms): gentle leveling
         that does NOT pump up quiet silence or breath tails.
      7. Limiter (-1.5 dB): transparent ceiling protection.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._master_chain: Pedalboard | None = None
        self._master_chain_sr: int | None = None

    def master_vocals(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Master Chatterbox voice audio at target sample rate."""
        if audio.size == 0:
            return audio.astype(np.float32)

        if self._master_chain is None or self._master_chain_sr != sr:
            self._master_chain = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=70.0),
                PeakFilter(cutoff_frequency_hz=350.0, gain_db=-1.5, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=1.2),
                HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.0),
                LowpassFilter(cutoff_frequency_hz=12000.0),
                Compressor(threshold_db=-24.0, ratio=1.8, attack_ms=20.0, release_ms=250.0),
                Limiter(threshold_db=-1.5),
            ])
            self._master_chain_sr = sr

        audio_2d = audio.astype(np.float32).reshape(1, -1) if audio.ndim == 1 else audio.astype(np.float32)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)


def build_chatterbox_voice_chain(
    reverb_amount: float = 0.05,
    ir_name: str = "warm_studio",
) -> Pedalboard:
    """Chatterbox voice FX chain: Abbey Road filtered convolution reverb + limiter.

    Applies the Abbey Road trick:
      - The convolution reverb sits strictly on a parallel wet path.
      - A HighpassFilter (300 Hz) and LowpassFilter (6000 Hz) filter the wet return,
        preventing low-end rumble and high-frequency flutter from creating an echo chamber.
      - When reverb_amount == 0.0 (or for critical vocal evaluation), the signal is 100% dry.
    """
    from core.audio_processor import DEFAULT_IR, IR_CATALOG

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))

    # If completely dry, return transparent limiter only
    if reverb_amount <= 0.001:
        return Pedalboard([
            Limiter(threshold_db=-1.0),
        ])

    ir_entry = IR_CATALOG.get(ir_name, IR_CATALOG.get(DEFAULT_IR))
    ir_path = ir_entry["path"] if ir_entry else ""

    dry_gain = 1.0 - reverb_amount
    wet_gain = reverb_amount

    dry_db = 20.0 * np.log10(max(dry_gain, 1e-5))
    wet_db = 20.0 * np.log10(max(wet_gain, 1e-5))

    wet_chain: list[Any] = []
    if os.path.isfile(ir_path):
        wet_chain.append(Convolution(impulse_response_filename=ir_path, mix=1.0))
    wet_chain.extend([
        HighpassFilter(cutoff_frequency_hz=300.0),
        LowpassFilter(cutoff_frequency_hz=6000.0),
        Gain(gain_db=wet_db),
    ])

    return Pedalboard([
        Mix([
            Gain(gain_db=dry_db),
            Pedalboard(wet_chain),
        ]),
        Limiter(threshold_db=-1.0),
    ])
