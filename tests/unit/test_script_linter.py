"""Tests for the deterministic script linter.

The linter is what makes a weaker or cheaper LLM viable: format errors are
caught in code and repaired, requiring no model judgment.
"""

import unittest

from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    Violation,
    check_format,
)


def codes(violations):
    return {v.code for v in violations}


class TestFormatChecks(unittest.TestCase):
    def test_clean_script_has_no_violations(self):
        script = "Settle in and let your shoulders drop.\n\n[pause:4s]\n\nBreathe out slowly."
        self.assertEqual(check_format(script), [])

    def test_malformed_pause_marker_is_fatal(self):
        violations = check_format("Breathe in. [pause:4] Breathe out.")
        self.assertIn("MARKER_MALFORMED", codes(violations))
        self.assertTrue(any(v.severity == FATAL for v in violations))

    def test_pause_out_of_bounds_is_fatal(self):
        violations = check_format("Rest here. [pause:900s] Now return.")
        self.assertIn("PAUSE_OUT_OF_RANGE", codes(violations))

    def test_unknown_tag_is_fatal(self):
        violations = check_format("Settle in. [whisper] Let go.")
        self.assertIn("UNKNOWN_TAG", codes(violations))

    def test_known_tags_are_accepted(self):
        script = "Settle in. [breath] Let go. [inhale] And out. [exhale]"
        self.assertEqual(check_format(script), [])

    def test_markdown_is_fatal(self):
        violations = check_format("## Opening\n\nBreathe in.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_bold_markdown_is_fatal(self):
        violations = check_format("Now **really** let go.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_emoji_is_fatal(self):
        violations = check_format("Breathe in and smile \U0001F60A")
        self.assertIn("EMOJI_PRESENT", codes(violations))

    def test_shouting_caps_is_advisory(self):
        violations = check_format("Now RELAX completely.")
        self.assertIn("ALL_CAPS", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY for v in violations if v.code == "ALL_CAPS")
        )

    def test_short_acronyms_are_allowed(self):
        self.assertEqual(check_format("Rest for a moment. OK."), [])

    def test_long_sentence_is_advisory(self):
        long_sentence = " ".join(["word"] * 40) + "."
        violations = check_format(long_sentence)
        self.assertIn("SENTENCE_TOO_LONG", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY for v in violations if v.code == "SENTENCE_TOO_LONG")
        )

    def test_violation_is_frozen(self):
        v = Violation(code="X", severity=FATAL, message="m")
        with self.assertRaises(Exception):
            v.code = "Y"

    def test_numbered_list_is_fatal(self):
        violations = check_format("Now follow these steps:\n1. Breathe in.\n2. Breathe out.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_numbered_list_with_paren_is_fatal(self):
        violations = check_format("1) First breathe\n2) Then relax")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_star_emoji_is_fatal(self):
        violations = check_format("You are doing great ⭐")
        self.assertIn("EMOJI_PRESENT", codes(violations))

    def test_number_word_at_line_start_is_not_list(self):
        script = "5 minutes from now, you will feel the floor beneath you."
        self.assertEqual(check_format(script), [])


if __name__ == "__main__":
    unittest.main()
