"""Chatterbox TTS postprocessing — studio vocal mastering and Abbey Road acoustic space.

Calibrated to match ElevenLabs-grade broadcast intimacy and clarity:
  1. Studio NoiseGate (-48 dBFS) eliminates low-level vocoder hiss in pauses.
  2. HighpassFilter (75 Hz) strips sub-bass mic rumble without thinning vocal body.
  3. PeakFilter (350 Hz, -2.0 dB) cleans boxy resonance.
  4. LowShelfFilter (160 Hz, +2.0 dB) provides close-mic chest warmth.
  5. HighShelfFilter (10.5 kHz, +1.5 dB) adds silky breath intimacy.
  6. LowpassFilter (14 kHz) smooths supra-audible vocoder hash.
  7. Compressor (-22 dB, 2.0:1, attack 15ms, release 180ms) levels dynamics naturally.
  8. NO Pedalboard Limiter (avoids pedalboard 0.9.23's +4.75 dB sub-threshold static bug).
  9. Default 100% DRY voice chain (matching ElevenLabs intimate studio delivery).
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
    LowpassFilter,
    LowShelfFilter,
    Mix,
    NoiseGate,
    PeakFilter,
    Pedalboard,
)

logger = logging.getLogger("moodscape.chatterbox_postprocessor")

SAMPLE_RATE = 24000


class ChatterboxMasteringEngine:
    """Studio mastering engine for Chatterbox TTS output at the mix sample rate (48 kHz).

    Delivers an ElevenLabs-style close-mic studio delivery:
      - NoiseGate: silences inter-phrase vocoder hiss.
      - EQ: removes boxiness, enhances vocal chest warmth and silky air.
      - Optical compression: gentle leveling without pumping quiet breath tails.
      - Peak protection via np.clip (true-peak limiting handled at export via mixer.true_peak_limit).
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
                NoiseGate(threshold_db=-48.0, ratio=3.0, attack_ms=2.0, release_ms=150.0),
                HighpassFilter(cutoff_frequency_hz=75.0),
                PeakFilter(cutoff_frequency_hz=350.0, gain_db=-2.0, q=1.0),
                LowShelfFilter(cutoff_frequency_hz=160.0, gain_db=2.0),
                HighShelfFilter(cutoff_frequency_hz=10500.0, gain_db=1.5),
                LowpassFilter(cutoff_frequency_hz=14000.0),
                Compressor(threshold_db=-22.0, ratio=2.0, attack_ms=15.0, release_ms=180.0),
            ])
            self._master_chain_sr = sr

        audio_2d = audio.astype(np.float32).reshape(1, -1) if audio.ndim == 1 else audio.astype(np.float32)
        processed = self._master_chain(audio_2d, sr)
        return np.clip(processed.squeeze(0), -1.0, 1.0).astype(np.float32)


def build_chatterbox_voice_chain(
    reverb_amount: float = 0.0,
    ir_name: str = "warm_studio",
) -> Pedalboard:
    """Chatterbox voice FX chain: 100% dry studio vocal by default, or Abbey Road reverb.

    ElevenLabs vocal delivery is renowned for its bone-dry, intimate studio presence.
    When reverb_amount == 0.0 (default), returns a transparent pass-through chain
    without reverb or buggy pedalboard limiters.

    When reverb is explicitly requested (>0.0), applies the Abbey Road trick:
      - Wet convolution reverb is bandpassed between 300 Hz and 6000 Hz.
      - Wet return is padded by -6 dB to prevent room convolution from overwhelming
        the direct voice into an echo chamber.
    """
    from core.audio_processor import DEFAULT_IR, IR_CATALOG

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))

    # If dry (ElevenLabs studio default), return transparent chain
    if reverb_amount <= 0.001:
        return Pedalboard([])

    ir_entry = IR_CATALOG.get(ir_name, IR_CATALOG.get(DEFAULT_IR))
    ir_path = ir_entry["path"] if ir_entry else ""

    dry_gain = 1.0 - reverb_amount
    wet_gain = reverb_amount

    dry_db = float(20.0 * np.log10(max(dry_gain, 1e-5)))
    # Attenuate wet return by -18 dB so convolution energy can never overwhelm
    # the direct vocal into an echo chamber.
    wet_db = float(20.0 * np.log10(max(wet_gain, 1e-5))) - 18.0

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
    ])
