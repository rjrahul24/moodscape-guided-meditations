"""Tests for the Auto-Generate tab builder.

Never imports app.py: that module loads torch and registers an atexit
hard-exit hook, so importing it from a test is not viable.
"""

import unittest
from unittest.mock import patch

import gradio as gr

from core.auto_tab import auto_generate_handler, build_auto_tab


class TestBuildAutoTab(unittest.TestCase):
    def test_builds_inside_a_blocks_context(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertIsInstance(components, dict)

    def test_exposes_the_components_the_handler_needs(self):
        with gr.Blocks():
            components = build_auto_tab()
        for key in (
            "prompt", "content_type", "tts_engine", "target_min", "target_max",
            "generator", "judge", "button", "audio", "status", "script", "changelog",
        ):
            self.assertIn(key, components)

    def test_defaults_to_the_golden_path(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertEqual(components["tts_engine"].value, "f5")
        self.assertEqual(components["content_type"].value, "meditation")

    def test_duration_defaults_are_five_to_seven_minutes(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertEqual(components["target_min"].value, 5)
        self.assertEqual(components["target_max"].value, 7)


class _FakeEmptyErrorRun:
    """Stands in for StreamingRun: iterates to nothing, then reports a
    failure whose message is the empty string -- the case that used to slip
    past a `if run.error:` truthiness check straight into an AttributeError
    on `run.result` (still None)."""

    def __init__(self, prompt, config=None, **kwargs):
        self.error = ""
        self.result = None
        self.invalid_input = False

    def __iter__(self):
        return iter([])


class TestAutoGenerateHandlerErrorGuard(unittest.TestCase):
    def test_empty_error_message_reports_failure_not_attributeerror(self):
        with patch("core.auto_tab.StreamingRun", _FakeEmptyErrorRun):
            outputs = list(
                auto_generate_handler(
                    "I feel anxious",
                    "meditation",
                    "f5",
                    5,
                    7,
                    "ollama:llama3.2:3b",
                    "ollama:llama3.2:3b",
                )
            )
        # The final yielded status must be the "Failed: ..." message, not an
        # AttributeError raised from touching run.result.background.
        self.assertEqual(outputs[-1][3], "Failed: ")

    def test_invalid_input_with_empty_error_still_uses_invalid_message(self):
        class FakeInvalidInputRun(_FakeEmptyErrorRun):
            def __init__(self, prompt, config=None, **kwargs):
                super().__init__(prompt, config=config, **kwargs)
                self.error = ""
                self.invalid_input = True

        with patch("core.auto_tab.StreamingRun", FakeInvalidInputRun):
            outputs = list(
                auto_generate_handler(
                    "", "meditation", "f5", 5, 7,
                    "ollama:llama3.2:3b", "ollama:llama3.2:3b",
                )
            )
        self.assertEqual(outputs[-1][3], "")


if __name__ == "__main__":
    unittest.main()
