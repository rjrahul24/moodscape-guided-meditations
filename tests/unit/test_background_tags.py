"""Tests for automatic background-music tagging.

Uses synthetic signals written to temporary WAVs -- no audio fixtures, no
network, and deterministic across machines.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.background_tags import extract_features

SR = 22050


def _write(path: Path, y: np.ndarray) -> str:
    sf.write(str(path), y.astype(np.float32), SR)
    return str(path)


def _pure_tone(seconds: float = 90.0, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * freq * t)


def _white_noise(seconds: float = 90.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return 0.3 * rng.standard_normal(int(SR * seconds))


class ExtractFeaturesTest(unittest.TestCase):
    def test_returns_all_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "tone.wav", _pure_tone())
            features = extract_features(path)
        self.assertEqual(
            set(features),
            {"centroid", "flatness", "flux", "onset_rate", "dynamics", "percussive"},
        )
        for key, value in features.items():
            self.assertIsInstance(value, float, msg=key)

    def test_pure_tone_is_tonal_and_noise_is_flat(self):
        """Spectral flatness must separate a sine from white noise."""
        with tempfile.TemporaryDirectory() as tmp:
            tone = extract_features(_write(Path(tmp) / "t.wav", _pure_tone()))
            noise = extract_features(_write(Path(tmp) / "n.wav", _white_noise()))
        self.assertLess(tone["flatness"], 1.0)
        self.assertGreater(noise["flatness"], tone["flatness"] * 10)

    def test_pure_tone_has_low_flux(self):
        """A sustained sine has almost no spectral change over time."""
        with tempfile.TemporaryDirectory() as tmp:
            tone = extract_features(_write(Path(tmp) / "t.wav", _pure_tone()))
        self.assertLess(tone["flux"], 1.0)


if __name__ == "__main__":
    unittest.main()
