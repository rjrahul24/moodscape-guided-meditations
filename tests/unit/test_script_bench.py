"""Tests for the model benchmark harness, using fake engines."""

import unittest

from core.auto_generate import AutoConfig
from core.bench import (
    BENCH_PROMPTS,
    BenchRow,
    format_bench_table,
    run_bench,
)
from core.script_gen.engine import FakeScriptEngine

CLEAN = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:5s]\n\n"
    "Notice the weight of your hands."
)
UNSAFE = "This will cure your anxiety.\n\n[pause:5s]\n\nRest."


def judged(script):
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


class TestBench(unittest.TestCase):
    def setUp(self):
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_prompt_set_covers_several_moods(self):
        self.assertGreaterEqual(len(BENCH_PROMPTS), 5)

    def test_passing_model_is_recorded_as_passed(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("fake:gen", "fake:judge")],
            prompts=["I feel anxious"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].passed)

    def test_failing_model_is_recorded_with_the_error(self):
        def factory(spec):
            return FakeScriptEngine([UNSAFE, judged(UNSAFE)] * 40)

        rows = run_bench(
            [("fake:gen", "fake:judge")],
            prompts=["I feel anxious"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertFalse(rows[0].passed)
        self.assertIn("CLINICAL_CLAIM", rows[0].error)

    def test_one_row_per_pairing_and_prompt(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("a:1", "b:1"), ("c:1", "d:1")],
            prompts=["p1", "p2"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 4)

    def test_row_records_timing_and_duration(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("a:1", "b:1")],
            prompts=["p"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertGreaterEqual(rows[0].elapsed_sec, 0.0)
        self.assertGreater(rows[0].estimated_sec, 0.0)

    def test_table_has_a_header_and_a_row_per_result(self):
        row = BenchRow(
            generator="a:1",
            judge="b:1",
            prompt="p",
            passed=True,
            estimated_sec=330.0,
            repairs_used=0,
            elapsed_sec=1.5,
            advisories=0,
            error="",
        )
        table = format_bench_table([row])
        self.assertIn("| Generator |", table)
        self.assertIn("a:1", table)

    def test_table_handles_no_rows(self):
        self.assertIn("| Generator |", format_bench_table([]))

    def test_malformed_spec_is_isolated_and_later_pairings_still_run(self):
        def factory(spec):
            if ":" not in spec:
                raise ValueError(f"Malformed engine spec {spec!r}")
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("badspec", "alsobad"), ("a:1", "b:1")],
            prompts=["p"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0].passed)
        self.assertIn("Malformed engine spec", rows[0].error)
        self.assertTrue(rows[1].passed)

    def test_mixed_run_isolates_the_malformed_pairing(self):
        def factory(spec):
            if ":" not in spec:
                raise ValueError(f"Malformed engine spec {spec!r}")
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("a:1", "b:1"), ("badspec", "alsobad"), ("c:1", "d:1")],
            prompts=["p"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].passed)
        self.assertFalse(rows[1].passed)
        self.assertTrue(rows[2].passed)

    def test_failed_row_has_no_fabricated_repairs_used(self):
        def factory(spec):
            return FakeScriptEngine([UNSAFE, judged(UNSAFE)] * 40)

        rows = run_bench(
            [("fake:gen", "fake:judge")],
            prompts=["I feel anxious"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertFalse(rows[0].passed)
        self.assertIsNone(rows[0].repairs_used)
        table = format_bench_table(rows)
        self.assertIn("| fake:gen | fake:judge |", table)
        self.assertIn("| - |", table)


if __name__ == "__main__":
    unittest.main()
