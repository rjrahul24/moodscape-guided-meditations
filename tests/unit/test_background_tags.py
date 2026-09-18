"""Tests for automatic background-music tagging.

Uses synthetic signals written to temporary WAVs -- no audio fixtures, no
network, and deterministic across machines.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.background_tags import (
    DECLARED_VOCAB,
    MEASURED_VOCAB,
    extract_features,
    tags_from_features,
)

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


def _features(**overrides) -> dict:
    """A mid-range track that trips no threshold, plus overrides."""
    base = {
        "centroid": 850.0,
        "flatness": 0.5,
        "flux": 1.5,
        "onset_rate": 3.0,
        "dynamics": 2.5,
        "percussive": 0.02,
    }
    base.update(overrides)
    return base


class TagsFromFeaturesTest(unittest.TestCase):
    def test_mid_range_track_gets_only_a_brightness_tag(self):
        """Brightness always emits exactly one tag; nothing else should fire.

        Every track has a brightness, so 'dark'/'warm'/'bright' is a partition
        rather than a threshold pair. The other five features only tag
        extremes, so a mid-range track trips none of them.
        """
        self.assertEqual(tags_from_features(_features()), ["warm"])

    def test_brightness_bands(self):
        self.assertIn("dark", tags_from_features(_features(centroid=300.0)))
        self.assertIn("bright", tags_from_features(_features(centroid=1300.0)))
        self.assertIn("warm", tags_from_features(_features(centroid=800.0)))

    def test_low_flux_is_a_drone(self):
        self.assertIn("drone", tags_from_features(_features(flux=0.3)))

    def test_high_flux_is_evolving(self):
        self.assertIn("evolving", tags_from_features(_features(flux=2.2)))

    def test_near_silent_drone_is_never_labelled_busy(self):
        """The onset detector fires on noise floor in very quiet drones.

        One real track reports 5.47 onsets/s with flux 0.29 and zero detected
        beats. Without gating 'busy'/'sparse' on flux, the calmest beds in the
        library get labelled the busiest.
        """
        tags = tags_from_features(_features(flux=0.29, onset_rate=5.47))
        self.assertIn("drone", tags)
        self.assertNotIn("busy", tags)
        self.assertNotIn("sparse", tags)

    def test_onset_bands_apply_when_flux_is_high_enough(self):
        self.assertIn("sparse", tags_from_features(_features(flux=1.5, onset_rate=0.9)))
        self.assertIn("busy", tags_from_features(_features(flux=1.5, onset_rate=7.0)))

    def test_tonal_and_textured(self):
        self.assertIn("tonal", tags_from_features(_features(flatness=0.02)))
        self.assertIn("textured", tags_from_features(_features(flatness=4.6)))

    def test_struck_and_sustained(self):
        self.assertIn("struck", tags_from_features(_features(percussive=0.047)))
        self.assertIn("sustained", tags_from_features(_features(percussive=0.006)))

    def test_steady_and_dynamic(self):
        self.assertIn("steady", tags_from_features(_features(dynamics=1.7)))
        self.assertIn("dynamic", tags_from_features(_features(dynamics=4.4)))

    def test_every_emitted_tag_is_in_the_vocabulary(self):
        extremes = [
            _features(centroid=100.0, flux=0.2, flatness=0.0, percussive=0.001, dynamics=1.0),
            _features(centroid=3000.0, flux=5.0, flatness=9.0, percussive=0.5, dynamics=9.0, onset_rate=12.0),
        ]
        for features in extremes:
            for tag in tags_from_features(features):
                self.assertIn(tag, MEASURED_VOCAB)

    def test_vocabularies_do_not_overlap(self):
        self.assertEqual(MEASURED_VOCAB & DECLARED_VOCAB, frozenset())


if __name__ == "__main__":
    unittest.main()
