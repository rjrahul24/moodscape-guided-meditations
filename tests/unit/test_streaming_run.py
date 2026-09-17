"""Tests for UI-independent streaming orchestration.

Never imports app.py: that module loads torch and Gradio and registers an
atexit hard-exit hook.
"""

import unittest

from core.auto_generate import AutoResult, ScriptGenerationError
from core.streaming_run import ProgressUpdate, StreamingRun


def make_result():
    return AutoResult(
        audio_path="/out/m.wav",
        script_path="/out/m.script.txt",
        meta_path="/out/m.meta.json",
        script="Breathe in.",
        changelog="- none",
        background="Healing Forest — 23:12",
        background_path="/assets/backgrounds/healing_forest.wav",
        violations=[],
        estimated_sec=330.0,
    )


class TestStreamingRun(unittest.TestCase):
    def test_yields_progress_updates(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            progress_cb(0.5, "halfway")
            return make_result()

        run = StreamingRun("p", runner=runner)
        updates = list(run)
        self.assertTrue(any(isinstance(u, ProgressUpdate) for u in updates))
        self.assertIn("halfway", [u.message for u in updates])

    def test_result_is_available_after_iteration(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            return make_result()

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertEqual(run.result.audio_path, "/out/m.wav")
        self.assertIsNone(run.error)
        self.assertFalse(run.invalid_input)

    def test_script_generation_error_is_captured_not_raised(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            raise ScriptGenerationError("CLINICAL_CLAIM survived repairs")

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertIsNone(run.result)
        self.assertEqual(run.error, "CLINICAL_CLAIM survived repairs")
        self.assertFalse(run.invalid_input)

    def test_runtime_error_is_captured_without_class_prefix(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            raise RuntimeError("ollama unreachable")

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertEqual(run.error, "ollama unreachable")
        self.assertFalse(run.invalid_input)

    def test_unexpected_error_is_captured_with_its_type(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            raise ValueError("bad thing")

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertEqual(run.error, "ValueError: bad thing")

    def test_prompt_is_forwarded(self):
        seen = {}

        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            seen["prompt"] = prompt
            return make_result()

        list(StreamingRun("I feel anxious", runner=runner))
        self.assertEqual(seen["prompt"], "I feel anxious")

    def test_blank_prompt_errors_without_running(self):
        called = []

        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            called.append(True)
            return make_result()

        run = StreamingRun("   ", runner=runner)
        list(run)
        self.assertEqual(called, [])
        self.assertEqual(run.error, "Enter a prompt first.")
        self.assertTrue(run.invalid_input)

    def test_empty_prompt_sets_invalid_input(self):
        run = StreamingRun("", runner=lambda *a, **k: make_result())
        list(run)
        self.assertEqual(run.error, "Enter a prompt first.")
        self.assertTrue(run.invalid_input)

    def test_iteration_terminates_even_with_no_progress_calls(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            return make_result()

        self.assertIsInstance(list(StreamingRun("p", runner=runner)), list)


if __name__ == "__main__":
    unittest.main()
