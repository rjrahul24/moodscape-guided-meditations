"""Tests for prompt assembly from on-disk guides."""

import tempfile
import unittest
from pathlib import Path

from core.script_gen.rules import (
    GUIDES_DIR,
    build_generator_system_prompt,
    build_judge_system_prompt,
    load_guide,
    load_safety_rules,
)


class TestRuleLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for content_type in ("meditation", "sleep_story"):
            for engine in ("f5", "kokoro"):
                (self.dir / f"vocal_{content_type}_{engine}_instructions.md").write_text(
                    f"GUIDE {content_type} {engine}"
                )
        (self.dir / "content_safety_rules.md").write_text("SAFETY RULES")

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_the_matching_guide(self):
        text = load_guide("f5", "meditation", guides_dir=self.dir)
        self.assertEqual(text, "GUIDE meditation f5")

    def test_loads_sleep_story_guide(self):
        text = load_guide("kokoro", "sleep_story", guides_dir=self.dir)
        self.assertEqual(text, "GUIDE sleep_story kokoro")

    def test_missing_guide_raises_with_a_helpful_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_guide("f5", "haiku", guides_dir=self.dir)
        self.assertIn("vocal_haiku_f5_instructions.md", str(ctx.exception))

    def test_loads_safety_rules(self):
        self.assertEqual(load_safety_rules(guides_dir=self.dir), "SAFETY RULES")

    def test_generator_prompt_includes_guide_and_safety(self):
        prompt = build_generator_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("GUIDE meditation f5", prompt)
        self.assertIn("SAFETY RULES", prompt)

    def test_generator_prompt_states_the_duration_window(self):
        prompt = build_generator_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("5", prompt)
        self.assertIn("7", prompt)

    def test_judge_prompt_includes_guide_and_safety(self):
        prompt = build_judge_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("GUIDE meditation f5", prompt)
        self.assertIn("SAFETY RULES", prompt)

    def test_judge_prompt_demands_the_delimited_output_format(self):
        prompt = build_judge_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("<script>", prompt)
        self.assertIn("<changelog>", prompt)

    def test_guides_dir_default_points_at_prompting_guides(self):
        self.assertEqual(GUIDES_DIR.name, "prompting_guides")

    def test_real_guides_exist_on_disk(self):
        # Guards against a rename breaking prompt assembly silently.
        for content_type in ("meditation", "sleep_story"):
            for engine in ("f5", "kokoro"):
                self.assertTrue(
                    (GUIDES_DIR / f"vocal_{content_type}_{engine}_instructions.md").is_file()
                )

    def test_real_safety_rules_exist_on_disk(self):
        self.assertTrue((GUIDES_DIR / "content_safety_rules.md").is_file())


if __name__ == "__main__":
    unittest.main()
