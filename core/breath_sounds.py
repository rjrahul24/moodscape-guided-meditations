"""Shared breath sound loader for TTS engines.

Loads WAV samples from assets/breath_sounds/ and caches them at the target
sample rate.  Falls back to silence if sample files are missing.
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("moodscape.breath_sounds")

_BREATH_DIR = Path(__file__).resolve().parent.parent / "assets" / "breath_sounds"

# Cache keyed by (subtype, target_sr)
_CACHE: dict[tuple[str, int], np.ndarray] = {}

# Cosine fade applied to each loaded sample to avoid clicks
_FADE_SEC = 0.075  # 75ms


def load_breath(subtype: str, target_sr: int = 24000) -> np.ndarray:
    """Load and return a breath sound as mono float32 at *target_sr*.

    Args:
        subtype: One of "breath", "inhale", "exhale".
        target_sr: Desired sample rate (default 24 kHz).

    Returns:
        numpy float32 array.  Falls back to 1.2s of silence if the
        WAV file is missing.
    """
    key = (subtype, target_sr)
    if key in _CACHE:
        return _CACHE[key].copy()

    filename = f"{subtype}.wav"
    path = _BREATH_DIR / filename

    if not path.exists():
        logger.warning("Breath sample not found: %s — inserting silence", path)
        silence = np.zeros(int(1.2 * target_sr), dtype=np.float32)
        _CACHE[key] = silence
        return silence.copy()

    audio, file_sr = sf.read(str(path), dtype="float32")

    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != target_sr:
        try:
            import torchaudio.functional as F
            import torch

            tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled = F.resample(tensor, file_sr, target_sr)
            audio = resampled.squeeze(0).numpy()
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_sr / file_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # Apply cosine fade-in/fade-out
    fade_n = int(_FADE_SEC * target_sr)
    if len(audio) > 2 * fade_n:
        t = np.linspace(0, np.pi / 2, fade_n, dtype=np.float32)
        audio[:fade_n] *= np.sin(t)
        audio[-fade_n:] *= np.cos(t)

    audio = audio.astype(np.float32)
    _CACHE[key] = audio
    return audio.copy()
