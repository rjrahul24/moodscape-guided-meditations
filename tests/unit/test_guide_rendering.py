import tempfile
import unittest
from pathlib import Path

from core.script_gen.guide_context import guide_context, render_guide
from core.script_gen.rules import (
    build_generator_system_prompt, build_judge_system_prompt, build_planner_system_prompt, load_guide,
)

COMBOS = [(e, c) for e in ("f5", "kokoro") for c in ("meditation", "sleep_story")]
STALE = ["300 char", "300-char", "0.88", "100–130", "100-130", "1.5s in Sleep Story", "1.5s in sleep story"]


class TestRenderGuide(unittest.TestCase):
    def test_renders_known_placeholder(self):
        self.assertEqual(render_guide("max {{WPM}} wpm", {"WPM": "85"}), "max 85 wpm")

    def test_unknown_placeholder_raises_before_llm(self):
        with self.assertRaisesRegex(KeyError, "CHUNK_CHAR"):
            render_guide("limit {{CHUNK_CHAR}}", {"CHUNK_CHARS": "250"})

    def test_unknown_placeholder_in_guide_file_raises_at_prompt_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "vocal_meditation_f5_instructions.md").write_text("bad {{NOPE}}")
            (d / "content_safety_rules.md").write_text("rules")
            with self.assertRaisesRegex(KeyError, "NOPE"):
                build_generator_system_prompt("f5", "meditation", 360, 600, guides_dir=d)


class TestContext(unittest.TestCase):
    def test_f5_values_come_from_code(self):
        from core.content_profiles import get_profile
        from core.f5_tts.preprocessor import MAX_CHUNK_CHARS
        from core.script_gen.duration import DEFAULT_WPM
        ctx = guide_context("f5", "sleep_story")
        self.assertEqual(ctx["CHUNK_CHARS"], str(MAX_CHUNK_CHARS))
        self.assertEqual(ctx["SPEED"], f"{get_profile('sleep_story')['speed']:.2f}")
        self.assertEqual(ctx["PARA_PAUSE"], f"{get_profile('sleep_story')['f5_paragraph_pause_sec']:g}")
        self.assertEqual(ctx["WPM"], f"{DEFAULT_WPM['f5']:.0f}")
        self.assertNotIn("CHUNK_TOKENS", ctx)

    def test_kokoro_uses_tokens(self):
        ctx = guide_context("kokoro", "meditation")
        self.assertIn("CHUNK_TOKENS", ctx)
        self.assertNotIn("CHUNK_CHARS", ctx)


class TestShippedGuides(unittest.TestCase):
    def test_every_guide_renders_clean_and_current(self):
        for engine, ct in COMBOS:
            text = load_guide(engine, ct)
            ctx = guide_context(engine, ct)
            self.assertNotIn("{{", text, (engine, ct))
            for key in ("WPM", "PARA_PAUSE", "SPEED"):
                self.assertIn(ctx[key], text, (engine, ct, key))
            for stale in STALE:
                self.assertNotIn(stale, text, (engine, ct, stale))

    def test_full_prompts_build(self):
        for engine, ct in COMBOS:
            build_generator_system_prompt(engine, ct, 360, 600)
            build_judge_system_prompt(engine, ct, 360, 600)
        build_planner_system_prompt("meditation", 360, 600)

    def test_example_scripts_pass_the_linter_safety_and_format(self):
        """Every fenced example script in the F5 guides must itself obey the rules."""
        import re
        from core.script_gen.linter import FATAL, check_format, check_safety
        for engine, ct in [("f5", "meditation"), ("f5", "sleep_story")]:
            text = load_guide(engine, ct)
            examples = [b for b in re.findall(r"```\n(.*?)```", text, re.S) if "[pause:" in b]
            self.assertTrue(examples, (engine, ct))
            for block in examples:
                fatal = [v.code for v in check_format(block) + check_safety(block) if v.severity == FATAL]
                self.assertEqual(fatal, [], (engine, ct, block[:80]))
                self.assertNotIn("...", block)
