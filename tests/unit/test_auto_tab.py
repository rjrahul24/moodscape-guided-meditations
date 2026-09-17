"""Tests for the Auto-Generate tab builder.

Never imports app.py: that module loads torch and registers an atexit
hard-exit hook, so importing it from a test is not viable.
"""

import unittest

import gradio as gr

from core.auto_tab import build_auto_tab


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


if __name__ == "__main__":
    unittest.main()
