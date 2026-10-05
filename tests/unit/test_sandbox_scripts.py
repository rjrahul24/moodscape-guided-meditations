"""Tests for sandbox benchmark scripts."""

import unittest

from core.sandbox_scripts import (
    MEDITATION_5MIN_SCRIPT,
    SLEEP_STORY_5MIN_SCRIPT,
    get_benchmark_script,
    get_script_metadata,
)
from core.kokoro_tts.preprocessor import prepare_segments as kokoro_prepare
from core.f5_tts.preprocessor import prepare_segments as f5_prepare


class TestSandboxScripts(unittest.TestCase):
    def test_meditation_script_word_count(self):
        words = MEDITATION_5MIN_SCRIPT.split()
        # Guided meditation should be ~220-320 spoken words so speech + pauses = ~5 min
        self.assertGreater(len(words), 200)
        self.assertLess(len(words), 350)

    def test_sleep_story_script_word_count(self):
        words = SLEEP_STORY_5MIN_SCRIPT.split()
        # Sleep story should be ~400-550 words for continuous 5-min bedtime narration
        self.assertGreater(len(words), 400)
        self.assertLess(len(words), 600)

    def test_meditation_script_contains_markers(self):
        self.assertIn("[pause:", MEDITATION_5MIN_SCRIPT)
        self.assertIn("[breath]", MEDITATION_5MIN_SCRIPT)

    def test_sleep_story_script_contains_markers(self):
        self.assertIn("[pause:", SLEEP_STORY_5MIN_SCRIPT)

    def test_kokoro_preprocessor_parses_both_scripts(self):
        med_segments = kokoro_prepare(MEDITATION_5MIN_SCRIPT, content_type="meditation")
        self.assertGreater(len(med_segments), 10)
        speech_segs = [s for s in med_segments if s["type"] == "speech"]
        pause_segs = [s for s in med_segments if s["type"] == "pause"]
        self.assertGreater(len(speech_segs), 5)
        self.assertGreater(len(pause_segs), 5)

        story_segments = kokoro_prepare(SLEEP_STORY_5MIN_SCRIPT, content_type="sleep_story")
        self.assertGreater(len(story_segments), 10)

    def test_f5_preprocessor_parses_both_scripts(self):
        med_segments = f5_prepare(MEDITATION_5MIN_SCRIPT, content_type="meditation")
        self.assertGreater(len(med_segments), 10)

        story_segments = f5_prepare(SLEEP_STORY_5MIN_SCRIPT, content_type="sleep_story")
        self.assertGreater(len(story_segments), 10)

    def test_get_benchmark_script_normalization(self):
        self.assertEqual(get_benchmark_script("meditation"), MEDITATION_5MIN_SCRIPT)
        self.assertEqual(get_benchmark_script("Guided Meditation"), MEDITATION_5MIN_SCRIPT)
        self.assertEqual(get_benchmark_script("sleep_story"), SLEEP_STORY_5MIN_SCRIPT)
        self.assertEqual(get_benchmark_script("Sleep Story"), SLEEP_STORY_5MIN_SCRIPT)

    def test_get_script_metadata(self):
        med_meta = get_script_metadata("meditation")
        self.assertEqual(med_meta["content_type"], "meditation")
        self.assertEqual(med_meta["target_duration_sec"], 300)

        story_meta = get_script_metadata("sleep_story")
        self.assertEqual(story_meta["content_type"], "sleep_story")
        self.assertEqual(story_meta["target_duration_sec"], 300)
