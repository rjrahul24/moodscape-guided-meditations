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

    def test_matched_script_keeps_a_literal_changelog_word_intact(self):
        # When <script> is located structurally, the returned script must be
        # exactly match.group(1) -- NOT run through the fallback path's
        # _STRAY_TAG/_CHANGELOG_BLOCK cleanup, which exists only to sanitize
        # the no-match case. A regression that applied that cleanup
        # unconditionally would mangle a script whose body legitimately uses
        # the word "changelog" in ordinary prose.
        raw = (
            "<script>\nKeep a mental changelog of small moments today.\n</script>\n"
            "<changelog>\n- tightened pacing\n</changelog>"
        )
        script, changelog = parse_judge_response(raw)
        self.assertEqual(
            script, "Keep a mental changelog of small moments today."
        )
        self.assertIn("tightened pacing", changelog)

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


class TestEchoedProblemsBlock(unittest.TestCase):
    """The repair prompt wraps the linter's findings in <problems>. A weak
    model often echoes that block back. It must never survive into the script:
    the linter's own messages name tags and show marker examples, so an echoed
    problem list re-triggers UNKNOWN_TAG / MARKER_MALFORMED / ANGLE_TAG on the
    text that reports them, and every repair round then adds more violations
    than it fixes. Observed with ollama:llama3.2:3b, the default engine.
    """

    ECHO = (
        "<problems>\n"
        "1. [UNKNOWN_TAG/fatal] Unknown tag '[UNK]'. Only [pause:Xs], "
        "[breath], [inhale] and [exhale] are supported.\n"
        "2. [MARKER_MALFORMED/fatal] Malformed pause marker '[pause:1s]'. "
        "Use exactly [pause:Xs], e.g. [pause:4s].\n"
        "</problems>\n"
    )

    def test_echoed_block_is_removed_on_the_fallback_path(self):
        script, _ = parse_judge_response(self.ECHO + "Breathe in.[pause:4s]")
        self.assertEqual(script, "Breathe in.[pause:4s]")

    def test_echoed_block_is_removed_when_a_script_block_is_present(self):
        raw = self.ECHO + "<script>Breathe out.[pause:4s]</script>"
        script, _ = parse_judge_response(raw)
        self.assertEqual(script, "Breathe out.[pause:4s]")

    def test_echoed_block_does_not_reach_the_linter(self):
        from core.script_gen.linter import check, fatal_violations

        script, _ = parse_judge_response(self.ECHO + "Breathe in.[pause:4s]")
        self.assertEqual(fatal_violations(check(script)), [])

    def test_unclosed_problems_tag_is_still_stripped(self):
        script, _ = parse_judge_response("<problems>\nBreathe in.[pause:4s]")
        self.assertNotIn("<problems>", script)
        self.assertIn("Breathe in.[pause:4s]", script)

    def test_script_prose_after_the_block_is_preserved_verbatim(self):
        body = "Settle in.[pause:3s]\n\n[breath]\n\nLet go.[pause:5s]"
        script, _ = parse_judge_response(self.ECHO + body)
        self.assertEqual(script, body)


if __name__ == "__main__":
    unittest.main()
