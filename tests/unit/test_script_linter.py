"""Tests for the deterministic script linter.

The linter is what makes a weaker or cheaper LLM viable: format errors are
caught in code and repaired, requiring no model judgment.
"""

import unittest

from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    Violation,
    check,
    check_format,
    check_safety,
    fatal_violations,
    format_for_repair,
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


class TestSafetyChecks(unittest.TestCase):
    def test_clean_script_passes(self):
        script = "If it feels right, you might let your eyes close."
        self.assertEqual(check_safety(script), [])

    def test_clinical_claim_is_fatal(self):
        violations = check_safety("This meditation will cure your anxiety.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))
        self.assertTrue(all(v.severity == FATAL for v in violations))

    def test_therapy_replacement_is_fatal(self):
        violations = check_safety("This replaces therapy for most people.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_outcome_promise_is_fatal(self):
        violations = check_safety("By the end you will be completely calm.")
        self.assertIn("OUTCOME_PROMISE", codes(violations))

    def test_invalidating_imperative_is_fatal(self):
        violations = check_safety("Don't feel anxious about it.")
        self.assertIn("INVALIDATING", codes(violations))

    def test_curly_apostrophe_does_not_evade_the_block(self):
        # LLMs emit U+2019 constantly; a safety block must not be defeated
        # by a typographic quote.
        violations = check_safety("Don't feel anxious about it.")
        self.assertIn("INVALIDATING", codes(violations))

    def test_extended_breath_hold_is_fatal(self):
        violations = check_safety("Hold your breath for 20 seconds.")
        self.assertIn("BREATH_HOLD", codes(violations))

    def test_short_breath_hold_is_allowed(self):
        self.assertEqual(check_safety("Hold your breath for 3 seconds."), [])

    def test_dissociation_imagery_is_fatal(self):
        violations = check_safety("Now leave your body behind.")
        self.assertIn("DISSOCIATION", codes(violations))

    def test_matching_is_case_insensitive(self):
        violations = check_safety("This Will Cure Your Depression.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    # FIX 1: CLINICAL_CLAIM verb inflections and false positive guards
    def test_clinical_claim_catches_healing_gerund(self):
        violations = check_safety(
            "This recording is designed for healing your anxiety over time."
        )
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_clinical_claim_catches_treated_past_tense(self):
        violations = check_safety("This has treated my depression for years.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_clinical_claim_catches_curing_gerund(self):
        violations = check_safety(
            "By curing your anxiety, this meditation helps you sleep."
        )
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_clinical_claim_guard_treat_yourself_kindly(self):
        # "Treat yourself" is warm guidance, not a clinical claim
        violations = check_safety(
            "Treat yourself kindly, especially when anxiety creeps in."
        )
        self.assertEqual(violations, [])

    # FIX 2: OUTCOME_PROMISE guard for grounding language
    def test_outcome_promise_guard_fully_present(self):
        # "Fully present" is grounding language, not an outcome guarantee
        violations = check_safety(
            "Notice how you will feel fully present in your body."
        )
        self.assertEqual(violations, [])

    # FIX 3: INVALIDATING with intervening adverb
    def test_invalidating_catches_dont_ever_feel(self):
        violations = check_safety("Don't ever feel ashamed of needing rest.")
        self.assertIn("INVALIDATING", codes(violations))

    def test_invalidating_guard_dont_rush_and_feel(self):
        # "Don't rush and feel" is benign guidance, not invalidation
        violations = check_safety(
            "Don't rush and feel the floor beneath you."
        )
        self.assertEqual(violations, [])

    # FIX 4: BREATH_HOLD with spelled-out numbers
    def test_breath_hold_catches_spelled_out_twenty(self):
        violations = check_safety("Hold your breath for twenty seconds.")
        self.assertIn("BREATH_HOLD", codes(violations))

    def test_breath_hold_allows_spelled_out_three(self):
        violations = check_safety("Hold your breath for three seconds.")
        self.assertEqual(violations, [])

    # FIX 5: BREATH_HOLD with qualifying clause
    def test_breath_hold_catches_with_qualifying_clause(self):
        violations = check_safety(
            "Hold your breath gently, without straining, for about 20 seconds."
        )
        self.assertIn("BREATH_HOLD", codes(violations))

    # FIX 6: DISSOCIATION drift imagery
    def test_dissociation_catches_drift_outside_body(self):
        violations = check_safety(
            "Let yourself drift outside your body, watching from above."
        )
        self.assertIn("DISSOCIATION", codes(violations))

    def test_dissociation_guard_tension_floats_away(self):
        # "Tension floats away" is benign, "tension" is not "body"
        violations = check_safety(
            "Let the tension float away with each exhale."
        )
        self.assertEqual(violations, [])


class TestCombinedCheck(unittest.TestCase):
    def test_duration_below_window_is_advisory(self):
        violations = check(
            "Breathe in.", estimated_sec=100.0,
            target_min_sec=300.0, target_max_sec=420.0,
        )
        self.assertIn("DURATION_OUT_OF_WINDOW", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY
                for v in violations if v.code == "DURATION_OUT_OF_WINDOW")
        )

    def test_duration_inside_window_is_clean(self):
        violations = check("Breathe in.", estimated_sec=360.0)
        self.assertNotIn("DURATION_OUT_OF_WINDOW", codes(violations))

    def test_duration_skipped_when_not_supplied(self):
        violations = check("Breathe in.")
        self.assertNotIn("DURATION_OUT_OF_WINDOW", codes(violations))

    def test_check_combines_format_and_safety(self):
        violations = check("## Title\n\nThis will cure your anxiety.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_fatal_violations_filters(self):
        violations = check("Now RELAX. This will cure your anxiety.")
        fatal = fatal_violations(violations)
        self.assertTrue(fatal)
        self.assertTrue(all(v.severity == FATAL for v in fatal))
        self.assertNotIn("ALL_CAPS", {v.code for v in fatal})

    def test_format_for_repair_lists_every_violation(self):
        violations = check("## Title\n\nThis will cure your anxiety.")
        text = format_for_repair(violations)
        self.assertIn("MARKDOWN_PRESENT", text)
        self.assertIn("CLINICAL_CLAIM", text)

    def test_format_for_repair_is_empty_when_clean(self):
        self.assertEqual(format_for_repair([]), "")


if __name__ == "__main__":
    unittest.main()
