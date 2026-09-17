"""Tests for content profiles and content-type-aware preprocessing.

Guards two things:
  1. The meditation path is unchanged — prepare_segments without content_type, with
     content_type="meditation", and the meditation profile all match the old behaviour.
  2. Sleep stories shorten paragraph-break pauses (the "fewer/shorter pauses" change).
"""

import unittest

from core.content_profiles import (
    CONTENT_PROFILES,
    DEFAULT_CONTENT_TYPE,
    get_profile,
    normalize_content_type,
)
from core.kokoro_tts.preprocessor import (
    parse_script as kokoro_parse,
    prepare_segments as kokoro_prepare,
    _PARAGRAPH_PAUSE_SEC as KOKORO_PARA_DEFAULT,
)
from core.f5_tts.preprocessor import (
    parse_script as f5_parse,
    prepare_segments as f5_prepare,
    _PARAGRAPH_PAUSE_SEC as F5_PARA_DEFAULT,
)


class TestContentProfiles(unittest.TestCase):
    def test_default_is_meditation(self):
        self.assertEqual(DEFAULT_CONTENT_TYPE, "meditation")

    def test_get_profile_unknown_falls_back_to_meditation(self):
        self.assertIs(get_profile("nonsense"), CONTENT_PROFILES["meditation"])
        self.assertIs(get_profile(None), CONTENT_PROFILES["meditation"])

    def test_get_profile_returns_requested(self):
        self.assertIs(get_profile("sleep_story"), CONTENT_PROFILES["sleep_story"])

    def test_normalize_accepts_keys_and_labels(self):
        self.assertEqual(normalize_content_type("meditation"), "meditation")
        self.assertEqual(normalize_content_type("sleep_story"), "sleep_story")
        self.assertEqual(normalize_content_type("Guided Meditation"), "meditation")
        self.assertEqual(normalize_content_type("Sleep Story"), "sleep_story")
        self.assertEqual(normalize_content_type("  sleep story  "), "sleep_story")
        self.assertEqual(normalize_content_type(""), "meditation")
        self.assertEqual(normalize_content_type(None), "meditation")
        self.assertEqual(normalize_content_type("???"), "meditation")

    def test_meditation_profile_matches_pipeline_defaults(self):
        # These mirror MeditationPipeline.generate() / app.py defaults. If a default
        # changes, this test forces the profile to be updated in lock-step.
        med = get_profile("meditation")
        self.assertEqual(med["speed"], 0.90)
        self.assertEqual(med["duck_amount_db"], -16.0)
        self.assertEqual(med["reverb_amount"], 0.15)
        self.assertEqual(med["kokoro_paragraph_pause_sec"], KOKORO_PARA_DEFAULT)
        self.assertEqual(med["f5_paragraph_pause_sec"], F5_PARA_DEFAULT)
        self.assertIsNone(med["bed"])

    def test_sleep_profile_is_softer_and_shorter(self):
        sleep = get_profile("sleep_story")
        med = get_profile("meditation")
        # Shorter paragraph pauses than meditation.
        self.assertLess(sleep["kokoro_paragraph_pause_sec"], med["kokoro_paragraph_pause_sec"])
        self.assertLess(sleep["f5_paragraph_pause_sec"], med["f5_paragraph_pause_sec"])
        # Shallower duck, longer fade-out, slightly slower.
        self.assertGreater(sleep["duck_amount_db"], med["duck_amount_db"])  # -11 > -16
        self.assertGreater(sleep["fade_out_sec"], med["fade_out_sec"])
        self.assertLessEqual(sleep["speed"], med["speed"])
        # Sleep bed overrides exist and do NOT collide with the duck-depth slider.
        self.assertIsNotNone(sleep["bed"])
        self.assertNotIn("duck_depth_db", sleep["bed"]["duck_kwargs"])


class TestMeditationPathUnchanged(unittest.TestCase):
    SCRIPT = "Breathe in.\n\nNow breathe out. [pause:3s] Rest here."

    def test_kokoro_default_matches_explicit_meditation(self):
        implicit = kokoro_prepare(self.SCRIPT)
        explicit = kokoro_prepare(self.SCRIPT, content_type="meditation")
        self.assertEqual(implicit, explicit)

    def test_f5_default_matches_explicit_meditation(self):
        implicit = f5_prepare(self.SCRIPT)
        explicit = f5_prepare(self.SCRIPT, content_type="meditation")
        self.assertEqual(implicit, explicit)

    def test_kokoro_parse_paragraph_default_unchanged(self):
        segs = kokoro_parse("One.\n\nTwo.")
        pause = next(s for s in segs if s["type"] == "pause")
        self.assertEqual(pause["duration_sec"], 6.5)

    def test_f5_parse_paragraph_default_unchanged(self):
        segs = f5_parse("One.\n\nTwo.")
        pause = next(s for s in segs if s["type"] == "pause")
        self.assertEqual(pause["duration_sec"], 3.0)


class TestSleepStoryShortensPauses(unittest.TestCase):
    def test_kokoro_sleep_paragraph_pause_is_shorter(self):
        segs = kokoro_prepare("One.\n\nTwo.", content_type="sleep_story")
        pause = next(s for s in segs if s["type"] == "pause")
        self.assertEqual(pause["duration_sec"], 2.5)

    def test_f5_sleep_paragraph_pause_is_shorter(self):
        segs = f5_prepare("One.\n\nTwo.", content_type="sleep_story")
        pause = next(s for s in segs if s["type"] == "pause")
        self.assertEqual(pause["duration_sec"], 1.5)

    def test_explicit_pauses_unaffected_by_content_type(self):
        # Author-placed [pause:Xs] must be honoured exactly regardless of mode.
        med = kokoro_prepare("A. [pause:4s] B.", content_type="meditation")
        sleep = kokoro_prepare("A. [pause:4s] B.", content_type="sleep_story")
        med_pause = next(s for s in med if s["type"] == "pause")["duration_sec"]
        sleep_pause = next(s for s in sleep if s["type"] == "pause")["duration_sec"]
        self.assertEqual(med_pause, 4.0)
        self.assertEqual(sleep_pause, 4.0)


if __name__ == "__main__":
    unittest.main()
