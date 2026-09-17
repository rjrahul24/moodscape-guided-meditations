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
        violations = check_format("You are doing great \u2b50")
        self.assertIn("EMOJI_PRESENT", codes(violations))

    def test_number_word_at_line_start_is_not_list(self):
        script = "5 minutes from now, you will feel the floor beneath you."
        self.assertEqual(check_format(script), [])

    def test_stray_open_ssml_tag_is_fatal(self):
        violations = check_format("Settle in. <emphasis>Breathe out.</emphasis>")
        self.assertIn("ANGLE_TAG", codes(violations))
        self.assertTrue(
            all(v.severity == FATAL for v in violations if v.code == "ANGLE_TAG")
        )

    def test_self_closing_ssml_tag_is_fatal(self):
        violations = check_format('Rest here. <break time="2s"/> Now return.')
        self.assertIn("ANGLE_TAG", codes(violations))

    def test_literal_less_than_in_prose_is_not_a_tag(self):
        # Pins the non-match: ordinary prose using "less than" in words, and
        # a literal "<" character not followed by a letter (so it cannot be
        # confused with the start of a tag), must not be flagged.
        script = (
            "This should take less than five minutes of your day. "
            "Aim for a rate of <10 breaths per minute."
        )
        self.assertNotIn("ANGLE_TAG", codes(check_format(script)))


class TestKokoroChunkLengthCheck(unittest.TestCase):
    """CHUNK_TOO_LONG is a Kokoro-specific backstop.

    merge_sentences_to_chunks() flushes a chunk before it would exceed
    MAX_CHUNK_TOKENS, so a multi-sentence chunk can never end up over the
    limit -- only a single sentence that is *itself* already ~115+ words can
    produce an oversized chunk, and a sentence that long has already tripped
    the far cheaper SENTENCE_TOO_LONG check (25-word threshold). These tests
    pin exactly that: the check exists, it is engine-gated, and it never
    fires without SENTENCE_TOO_LONG also firing.
    """

    # 120 words with no internal punctuation, so split_into_sentences()
    # and the linter's own sentence splitter both see it as a single
    # sentence. estimate_tokens(120 words) ~= 156 tokens, over the
    # 150-token MAX_CHUNK_TOKENS limit.
    OVERSIZED_CHUNK_SCRIPT = (" ".join(["breathe slowly and let go"] * 24) + ".")

    def test_oversized_chunk_flags_under_kokoro(self):
        violations = check_format(self.OVERSIZED_CHUNK_SCRIPT, engine="kokoro")
        self.assertIn("CHUNK_TOO_LONG", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY for v in violations if v.code == "CHUNK_TOO_LONG")
        )

    def test_oversized_chunk_also_trips_sentence_too_long(self):
        # Confirms the redundancy claim: the script that triggers
        # CHUNK_TOO_LONG must also already be caught by SENTENCE_TOO_LONG.
        violations = check_format(self.OVERSIZED_CHUNK_SCRIPT, engine="kokoro")
        self.assertIn("SENTENCE_TOO_LONG", codes(violations))

    def test_oversized_chunk_is_not_flagged_for_f5(self):
        violations = check_format(self.OVERSIZED_CHUNK_SCRIPT, engine="f5")
        self.assertNotIn("CHUNK_TOO_LONG", codes(violations))

    def test_oversized_chunk_is_not_flagged_by_default(self):
        violations = check_format(self.OVERSIZED_CHUNK_SCRIPT)
        self.assertNotIn("CHUNK_TOO_LONG", codes(violations))

    def test_ordinary_meditation_script_flags_nothing_under_kokoro(self):
        script = (
            "Settle in and let your shoulders drop.\n\n"
            "[pause:4s]\n\n"
            "Breathe out slowly, and notice the weight of your hands."
        )
        violations = check_format(script, engine="kokoro")
        self.assertNotIn("CHUNK_TOO_LONG", codes(violations))

    def test_check_threads_engine_through_to_check_format(self):
        violations = check(self.OVERSIZED_CHUNK_SCRIPT, engine="kokoro")
        self.assertIn("CHUNK_TOO_LONG", codes(violations))

    def test_check_default_engine_omits_chunk_check(self):
        violations = check(self.OVERSIZED_CHUNK_SCRIPT)
        self.assertNotIn("CHUNK_TOO_LONG", codes(violations))


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
        # by a typographic quote. Every form is built from an EXPLICIT
        # \uXXXX escape, never a literal curly character typed into this
        # file -- a literal character is exactly what caused the original
        # bug (it was silently normalized to ASCII on the way to disk,
        # turning the old test byte-identical to the plain-ASCII test above
        # it, so it passed at every commit while the safety block was off).
        forms = [
            ("U+2019 RIGHT SINGLE QUOTATION MARK", "\u2019"),
            ("U+2018 LEFT SINGLE QUOTATION MARK", "\u2018"),
            ("U+02BC MODIFIER LETTER APOSTROPHE", "\u02bc"),
            ("U+00B4 ACUTE ACCENT", "\u00b4"),
            ("U+0060 GRAVE ACCENT", "`"),
        ]
        for label, ch in forms:
            with self.subTest(label):
                violations = check_safety(f"Don{ch}t feel anxious about it.")
                self.assertIn("INVALIDATING", codes(violations))

    def test_apostrophe_widening_does_not_over_match_clean_prose(self):
        # The widened character set must not turn ordinary prose (which may
        # itself contain any of these characters, e.g. in a quotation or a
        # name) into a false positive.
        script = (
            "She said, \u2018just breathe\u2019 and then quoted a `poem` "
            "with an \u00b4accent\u00b4 mark."
        )
        self.assertEqual(check_safety(script), [])

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

    # FIX A: Regression guards for bare verbs (Critical)
    def test_clinical_claim_catches_bare_treat_verb(self):
        # Regression: "treat" without inflection must be caught
        violations = check_safety("This will treat your anxiety.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_clinical_claim_catches_bare_heal_verb(self):
        # Regression: "heal" without inflection must be caught
        violations = check_safety("This will heal your anxiety.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_clinical_claim_catches_treat_without_auxiliary(self):
        # Regression: "treats" (3rd person) must be caught
        violations = check_safety("This treats anxiety.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    # FIX B: Revert grief/stress/the additions
    def test_clinical_claim_guard_grief_observation(self):
        # Grief is a content type; don't block observations about it
        violations = check_safety(
            "Healing the deep grief that you carry is part of being human."
        )
        self.assertEqual(violations, [])

    # FIX D: Narrow dissociation verbs
    def test_dissociation_guard_rise_tension_release(self):
        # "Rise" in tension-release context is benign, not dissociation
        violations = check_safety(
            "Let the warmth rise up out of your body as you relax."
        )
        self.assertEqual(violations, [])

    # FIX C: Restore bounded breath-hold budget
    def test_breath_hold_guard_unrelated_number_in_runon(self):
        # Unbounded pattern would match "20" from unrelated context
        violations = check_safety(
            "Hold your breath, and think about how, at 7 years old, you used to "
            "play in the yard until sunset, then count 20 seconds of pure stillness."
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
