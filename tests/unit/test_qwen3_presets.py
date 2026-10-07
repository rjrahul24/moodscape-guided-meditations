import os
import unittest
from unittest.mock import patch

from core.qwen3_tts import presets


class TestQwen3Presets(unittest.TestCase):
    def test_every_mode_has_a_repo(self):
        self.assertEqual(set(presets.MODEL_REPOS), set(presets.MODES))

    def test_env_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in ("MOODSCAPE_QWEN3_MODE", "MOODSCAPE_QWEN3_TEMPERATURE",
                        "MOODSCAPE_QWEN3_CHUNK_CHARS", "MOODSCAPE_QWEN3_STRETCH"):
                os.environ.pop(key, None)
            self.assertEqual(presets.mode_from_env(), "clone")
            self.assertEqual(presets.temperature_from_env(), 0.7)
            self.assertEqual(presets.chunk_chars_from_env(), 200)
            self.assertEqual(presets.stretch_from_env(), 1.0)

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": "turbo"}):
            with self.assertRaisesRegex(ValueError, "turbo"):
                presets.mode_from_env()

    def test_instruct_none_in_clone_mode(self):
        self.assertIsNone(presets.instruct_for("clone", "meditation"))

    def test_instruct_by_content_type(self):
        os.environ.pop("MOODSCAPE_QWEN3_INSTRUCT", None)
        self.assertIn("slowly", presets.instruct_for("custom", "meditation").lower())
        self.assertIn("bedtime", presets.instruct_for("custom", "sleep_story").lower())
        self.assertIn("narrator", presets.instruct_for("design", "meditation").lower())

    def test_instruct_env_override(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_INSTRUCT": "Whisper it."}):
            self.assertEqual(presets.instruct_for("custom", "meditation"), "Whisper it.")

    def test_unknown_content_type_falls_back_to_meditation(self):
        os.environ.pop("MOODSCAPE_QWEN3_INSTRUCT", None)
        self.assertEqual(presets.instruct_for("custom", "guided_meditation"),
                         presets.INSTRUCT_PRESETS["meditation"])

    def test_speaker_from_env_defaults_when_unset_empty_or_blank(self):
        for val in (None, "", "   "):
            env = {} if val is None else {"MOODSCAPE_QWEN3_SPEAKER": val}
            with patch.dict(os.environ, env):
                if val is None:
                    os.environ.pop("MOODSCAPE_QWEN3_SPEAKER", None)
                self.assertEqual(presets.speaker_from_env(), presets.DEFAULT_SPEAKER)
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_SPEAKER": " ryan "}):
            self.assertEqual(presets.speaker_from_env(), "ryan")


if __name__ == "__main__":
    unittest.main()
