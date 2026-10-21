"""Light, clean post-processing for ElevenLabs voice.

ElevenLabs output is studio-clean: no DeepFilterNet, no mastering engine, no
saturation. Just a gentle de-ess, an HPF, and optional room reverb.
"""

from __future__ import annotations

import os

import numpy as np
from pedalboard import HighpassFilter, Pedalboard


def deess_elevenlabs(audio: np.ndarray, sr: int) -> np.ndarray:
    if os.environ.get("MOODSCAPE_ELEVENLABS_DEESS", "1") == "0":
        return audio
    from core.f5_tts.postprocessor import split_band_deess
    return split_band_deess(audio, sr)


def build_elevenlabs_voice_chain(reverb_amount: float = 0.0,
                                 ir_name: str = "warm_studio") -> Pedalboard:
    plugins = [HighpassFilter(cutoff_frequency_hz=70)]
    if reverb_amount > 0:
        from core.f5_tts.postprocessor import parallel_reverb_mix
        plugins.append(parallel_reverb_mix(reverb_amount, ir_name))
    return Pedalboard(plugins)
