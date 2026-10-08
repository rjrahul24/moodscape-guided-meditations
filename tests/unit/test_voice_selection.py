# tests/unit/test_voice_selection.py
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.voice_selection import (
    DEFAULT_VOICE, FAMILY_PROFILE, VoiceTraits, measure_voice, profile_scores, voice_for_family, voice_traits,
)

# Measured 2026-10-07 from assets/speakers (see the R1 spec).
TRAITS = {
    "Brittney": VoiceTraits(109, 185, 175, 1412),
    "Clara": VoiceTraits(106, 175, 75, 1615),
    "Delilah": VoiceTraits(97, 177, 139, 1303),
    "Eryn": VoiceTraits(106, 191, 86, 1864),
}


class TestDefaultVoice(unittest.TestCase):
    """Per-family choice lost the R1 blind A/B (2026-10-08), so it is opt-in."""

    def setUp(self):
        self._env = os.environ.pop("MOODSCAPE_VOICE_BY_FAMILY", None)

    def tearDown(self):
        os.environ.pop("MOODSCAPE_VOICE_BY_FAMILY", None)
        if self._env is not None:
            os.environ["MOODSCAPE_VOICE_BY_FAMILY"] = self._env

    def test_default_voice_is_brittney(self):
        self.assertEqual(DEFAULT_VOICE, "Brittney")

    def test_flag_off_returns_default_for_every_family(self):
        for family in FAMILY_PROFILE:
            self.assertEqual(voice_for_family(family, traits=TRAITS, overrides_path=Path("/nonexistent")),
                             "Brittney", family)

    def test_flag_off_does_not_measure_clips(self):
        from unittest.mock import patch
        with patch("core.voice_selection.voice_traits", side_effect=AssertionError("measured")):
            self.assertEqual(voice_for_family("Sleep & Rest", overrides_path=Path("/nonexistent")), "Brittney")

    def test_flag_off_override_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice_map.toml"
            p.write_text('[families]\n"Sleep & Rest" = "Delilah"\n')
            self.assertEqual(voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=p), "Delilah")
            self.assertEqual(voice_for_family("Breathwork", traits=TRAITS, overrides_path=p), "Brittney")

    def test_flag_zero_is_off(self):
        os.environ["MOODSCAPE_VOICE_BY_FAMILY"] = "0"
        self.assertEqual(voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=Path("/nonexistent")),
                         "Brittney")


class TestMapping(unittest.TestCase):
    """Measured per-family choice, enabled with MOODSCAPE_VOICE_BY_FAMILY=1."""

    def setUp(self):
        self._env = os.environ.get("MOODSCAPE_VOICE_BY_FAMILY")
        os.environ["MOODSCAPE_VOICE_BY_FAMILY"] = "1"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("MOODSCAPE_VOICE_BY_FAMILY", None)
        else:
            os.environ["MOODSCAPE_VOICE_BY_FAMILY"] = self._env

    def test_expected_assignment_for_todays_voices(self):
        expected = {"Sleep & Rest": "Delilah", "Stress & Anxiety": "Delilah",
                    "Morning & Energy": "Eryn", "Body & Movement": "Eryn",
                    "Emotional": "Brittney", "Presence": "Brittney",
                    "Focus & Work": "Clara", "Breathwork": "Clara"}
        for family, voice in expected.items():
            self.assertEqual(voice_for_family(family, traits=TRAITS, overrides_path=Path("/nonexistent")), voice, family)

    def test_every_shipped_family_has_a_profile(self):
        from core.genres import load_all_packs
        families = {p.family for p in load_all_packs().values()}
        self.assertTrue(families <= set(FAMILY_PROFILE), families - set(FAMILY_PROFILE))

    def test_unknown_family_uses_steady(self):
        self.assertEqual(voice_for_family("Nope", traits=TRAITS, overrides_path=Path("/nonexistent")), "Clara")

    def test_single_voice_pool(self):
        self.assertEqual(voice_for_family("Sleep & Rest", traits={"Solo": TRAITS["Eryn"]},
                                          overrides_path=Path("/nonexistent")), "Solo")

    def test_ties_break_alphabetically(self):
        same = {"B": VoiceTraits(100, 180, 100, 1500), "A": VoiceTraits(100, 180, 100, 1500)}
        self.assertEqual(voice_for_family("Sleep & Rest", traits=same, overrides_path=Path("/nonexistent")), "A")

    def test_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice_map.toml"
            p.write_text('[families]\n"Sleep & Rest" = "Eryn"\n')
            self.assertEqual(voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=p), "Eryn")

    def test_override_unknown_slug_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice_map.toml"
            p.write_text('[families]\n"Sleep & Rest" = "Nobody"\n')
            with self.assertRaisesRegex(ValueError, "Nobody"):
                voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=p)

    def test_scores_are_relative(self):
        s = profile_scores(TRAITS)
        self.assertGreater(s["Delilah"]["calm"], s["Eryn"]["calm"])
        self.assertGreater(s["Brittney"]["expressive"], s["Clara"]["expressive"])


class TestMeasurement(unittest.TestCase):
    def _tone(self, path, freq, seconds=3.0, sr=24000):
        t = np.arange(int(sr * seconds)) / sr
        sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)

    def test_measure_pitch_and_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "v.wav"
            self._tone(wav, 200.0)
            tr = measure_voice(wav, "one two three four five six")
            self.assertAlmostEqual(tr.f0_median, 200.0, delta=10.0)
            self.assertAlmostEqual(tr.wpm, 6 / 3.0 * 60, delta=1.0)

    def test_cache_hits_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            wav = d / "A.wav"
            self._tone(wav, 200.0)
            reg = {"A": {"default": {"audio": wav, "transcript": "one two three"}}}
            cache = d / "traits.json"
            first = voice_traits(reg, cache_path=cache)
            data = json.loads(cache.read_text())
            data["A"]["traits"]["f0_median"] = 999.0  # poison the cache
            cache.write_text(json.dumps(data))
            self.assertEqual(voice_traits(reg, cache_path=cache)["A"].f0_median, 999.0)  # cache hit
            os.utime(wav, (wav.stat().st_atime, wav.stat().st_mtime + 10))
            self.assertAlmostEqual(voice_traits(reg, cache_path=cache)["A"].f0_median, first["A"].f0_median, delta=1.0)

    def test_unreadable_clip_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bad = d / "Bad.wav"
            bad.write_bytes(b"not audio")
            good = d / "Good.wav"
            self._tone(good, 180.0)
            reg = {"Bad": {"default": {"audio": bad, "transcript": "x"}},
                   "Good": {"default": {"audio": good, "transcript": "one two"}}}
            self.assertEqual(set(voice_traits(reg, cache_path=d / "c.json")), {"Good"})


if __name__ == "__main__":
    unittest.main()
