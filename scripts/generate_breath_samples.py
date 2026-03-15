"""Generate synthetic breath sound samples for meditation paralinguistic cues.

Creates three WAV files at 24 kHz mono float32:
  - inhale.wav  (1.5s) — rising amplitude, bandpass 100-800 Hz
  - exhale.wav  (1.8s) — falling amplitude, bandpass 80-600 Hz
  - breath.wav  (1.2s) — bell-curve amplitude, bandpass 100-700 Hz

Usage:
    python scripts/generate_breath_samples.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 24000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "breath_sounds"


def _bandpass(signal: np.ndarray, low: float, high: float, sr: int) -> np.ndarray:
    """Apply a 4th-order Butterworth bandpass filter."""
    sos = butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def _generate_inhale(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Inhale: 1.5s, rising amplitude, bandpass 100-800 Hz."""
    duration = 1.5
    n = int(duration * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)

    noise = np.random.randn(n).astype(np.float32)
    filtered = _bandpass(noise, 100, 800, sr)

    # Rising envelope with gentle attack and soft peak
    envelope = np.sin(t * np.pi / 2) ** 1.5
    # Taper the very end slightly
    fade_out_n = int(0.1 * sr)
    envelope[-fade_out_n:] *= np.linspace(1, 0.3, fade_out_n, dtype=np.float32)

    audio = filtered * envelope * 0.35
    return audio


def _generate_exhale(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Exhale: 1.8s, falling amplitude, bandpass 80-600 Hz, longer tail."""
    duration = 1.8
    n = int(duration * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)

    noise = np.random.randn(n).astype(np.float32)
    filtered = _bandpass(noise, 80, 600, sr)

    # Quick onset, slow exponential decay
    envelope = np.exp(-2.5 * t) * (1 - np.exp(-15 * t))

    audio = filtered * envelope * 0.30
    return audio


def _generate_breath(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Breath: 1.2s, bell-curve amplitude, bandpass 100-700 Hz."""
    duration = 1.2
    n = int(duration * sr)
    t = np.linspace(0, 1, n, dtype=np.float32)

    noise = np.random.randn(n).astype(np.float32)
    filtered = _bandpass(noise, 100, 700, sr)

    # Symmetric bell curve (Gaussian-ish)
    center = 0.45  # slightly front-loaded
    envelope = np.exp(-((t - center) ** 2) / (2 * 0.15**2))

    audio = filtered * envelope * 0.30
    return audio


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)  # reproducible samples

    samples = {
        "inhale.wav": _generate_inhale(),
        "exhale.wav": _generate_exhale(),
        "breath.wav": _generate_breath(),
    }

    for name, audio in samples.items():
        path = OUTPUT_DIR / name
        sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")
        print(f"  wrote {path}  ({len(audio) / SAMPLE_RATE:.1f}s, peak={np.max(np.abs(audio)):.3f})")

    print("Done.")


if __name__ == "__main__":
    main()
