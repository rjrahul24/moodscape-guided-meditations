"""Tests for the two-pass generator/judge flow, against a fake engine."""

import unittest

from core.script_gen.engine import FakeScriptEngine
from core.script_gen.generator import draft
from core.script_gen.judge import parse_judge_response, repair, review
from core.script_gen.linter import FATAL, Violation


class TestGenerator(unittest.TestCase):
    def test_returns_the_engine_output(self):
        engine = FakeScriptEngine(["Breathe in."])
        self.assertEqual(draft(engine, "I feel anxious", "SYS"), "Breathe in.")

    def test_passes_the_system_prompt_through(self):
        engine = FakeScriptEngine(["out"])
        draft(engine, "I feel anxious", "SYSTEM")
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")

    def test_user_message_contains_the_prompt(self):
        engine = FakeScriptEngine(["out"])
        draft(engine, "I feel anxious", "SYS")
        self.assertIn("I feel anxious", engine.calls[0]["user"])

    def test_strips_surrounding_whitespace(self):
        engine = FakeScriptEngine(["\n\n  Breathe in.  \n\n"])
        self.assertEqual(draft(engine, "p", "s"), "Breathe in.")

    def test_strips_a_fenced_code_block(self):
        engine = FakeScriptEngine(["```\nBreathe in.\n```"])
        self.assertEqual(draft(engine, "p", "s"), "Breathe in.")


class TestJudgeParsing(unittest.TestCase):
    def test_extracts_script_and_changelog(self):
        raw = "<script>\nBreathe in.\n</script>\n<changelog>\n- fixed a tag\n</changelog>"
        script, changelog = parse_judge_response(raw)
        self.assertEqual(script, "Breathe in.")
        self.assertIn("fixed a tag", changelog)

    def test_missing_changelog_yields_empty_string(self):
        script, changelog = parse_judge_response("<script>Breathe in.</script>")
        self.assertEqual(script, "Breathe in.")
        self.assertEqual(changelog, "")

    def test_missing_delimiters_falls_back_to_whole_output(self):
        # A model that ignores the format must not fail the run.
        script, changelog = parse_judge_response("Breathe in.")
        self.assertEqual(script, "Breathe in.")
        self.assertEqual(changelog, "")

    def test_tolerates_surrounding_prose(self):
        raw = "Sure!\n<script>\nBreathe in.\n</script>\nHope that helps."
        script, _ = parse_judge_response(raw)
        self.assertEqual(script, "Breathe in.")

    def test_tolerates_attributes_on_the_script_tag(self):
        raw = (
            '<script lang="en">\nBreathe deeply.\n</script>\n'
            "<changelog>\n- fixed pacing\n</changelog>"
        )
        script, changelog = parse_judge_response(raw)
        self.assertEqual(script, "Breathe deeply.")
        self.assertIn("fixed pacing", changelog)

    def test_unmatchable_script_tag_does_not_leak_the_changelog(self):
        # The opening tag never closes, so <script> can't be located
        # structurally. The fallback must not let the changelog block (or
        # its own tag text) leak into what the TTS engine reads aloud.
        raw = (
            "<script\nBreathe deeply and relax.\n"
            "<changelog>\n- fixed pacing\n</changelog>"
        )
        script, _ = parse_judge_response(raw)
        self.assertNotIn("fixed pacing", script)
        self.assertNotIn("changelog", script.lower())
        self.assertNotIn("<script", script.lower())


class TestReview(unittest.TestCase):
    def test_returns_revised_script_and_changelog(self):
        engine = FakeScriptEngine(
            ["<script>Revised.</script><changelog>- tightened</changelog>"]
        )
        script, changelog = review(engine, "Draft.", "SYS")
        self.assertEqual(script, "Revised.")
        self.assertIn("tightened", changelog)

    def test_draft_is_included_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Revised.</script>"])
        review(engine, "THE DRAFT", "SYS")
        self.assertIn("THE DRAFT", engine.calls[0]["user"])


class TestRepair(unittest.TestCase):
    def test_violations_appear_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [
            Violation(code="MARKER_MALFORMED", severity=FATAL, message="bad tag")
        ]
        repair(engine, "Broken.", violations, "SYS")
        user = engine.calls[0]["user"]
        self.assertIn("MARKER_MALFORMED", user)
        self.assertIn("bad tag", user)

    def test_current_script_appears_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [Violation(code="X", severity=FATAL, message="m")]
        repair(engine, "THE BROKEN SCRIPT", violations, "SYS")
        self.assertIn("THE BROKEN SCRIPT", engine.calls[0]["user"])

    def test_returns_the_repaired_script(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [Violation(code="X", severity=FATAL, message="m")]
        script, _ = repair(engine, "Broken.", violations, "SYS")
        self.assertEqual(script, "Fixed.")


if __name__ == "__main__":
    unittest.main()
