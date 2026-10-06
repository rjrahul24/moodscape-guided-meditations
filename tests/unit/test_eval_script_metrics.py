# tests/unit/test_eval_script_metrics.py
import unittest

from core.eval.script_metrics import script_metrics

SCRIPT = (
    "Find a comfortable position. [pause:4s] Let your shoulders soften gently.\n\n"
    "[breath]\n\n"
    "Notice the weight of your hands. [pause:6s]"
)


class TestScriptMetrics(unittest.TestCase):
    def test_counts(self):
        m = script_metrics(SCRIPT)
        self.assertEqual(m["sentences"], 3)
        self.assertEqual(m["words"], 15)
        self.assertEqual(m["max_sentence_words"], 6)
        self.assertEqual(m["breath_count"], 1)
        self.assertEqual(m["question_count"], 0)

    def test_pause_seconds_match_preprocessor(self):
        from core.f5_tts.preprocessor import prepare_segments
        expected = sum(
            s["duration_sec"] for s in prepare_segments(SCRIPT, content_type="meditation")
            if s["type"] == "pause"
        )
        self.assertAlmostEqual(script_metrics(SCRIPT)["pause_seconds"], expected)

    def test_pause_ratio_is_fraction_of_estimate(self):
        m = script_metrics(SCRIPT)
        self.assertAlmostEqual(m["pause_ratio_est"], m["pause_seconds"] / m["estimated_sec"])

    def test_lint_codes_split_by_severity(self):
        m = script_metrics("**Bold** text. [pause:3]")
        self.assertIn("MARKDOWN_PRESENT", m["lint_fatal"])
        self.assertIn("MARKER_MALFORMED", m["lint_fatal"])

    def test_slop_reported(self):
        m = script_metrics(SCRIPT)
        self.assertGreater(m["slop_density"], 0.0)
        self.assertIn("gently", m["slop_top"])

    def test_edit_ratio_none_without_draft(self):
        self.assertIsNone(script_metrics(SCRIPT)["edit_ratio"])

    def test_edit_ratio_zero_for_identical_and_positive_for_changed(self):
        self.assertEqual(script_metrics(SCRIPT, draft=SCRIPT)["edit_ratio"], 0.0)
        changed = script_metrics(SCRIPT, draft="Something entirely different here.")
        self.assertGreater(changed["edit_ratio"], 0.5)

    def test_non_f5_cloning_engine_uses_f5_grammar(self):
        m = script_metrics(SCRIPT, engine="qwen3")
        self.assertEqual(m["breath_count"], 1)


if __name__ == "__main__":
    unittest.main()
