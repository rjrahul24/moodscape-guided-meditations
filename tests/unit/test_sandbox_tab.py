"""Tests for the TTS Sandbox tab builder and handlers.

Follows the same pattern as test_auto_tab.py: tests the tab builder
inside a gr.Blocks() context without importing app.py directly.
"""

import unittest
import gradio as gr

from core.sandbox_tab import (
    build_sandbox_tab,
    format_promoted_models_html,
    handle_discard_model,
    render_qa_scorecard,
    render_sandbox_status,
)
from core.sandbox_scripts import MEDITATION_5MIN_SCRIPT, SLEEP_STORY_5MIN_SCRIPT


class TestBuildSandboxTab(unittest.TestCase):
    def test_builds_inside_a_blocks_context(self):
        with gr.Blocks():
            components = build_sandbox_tab()
        self.assertIsInstance(components, dict)

    def test_exposes_essential_components(self):
        with gr.Blocks():
            components = build_sandbox_tab()

        required_keys = (
            "benchmark_script_choice",
            "reset_script_btn",
            "content_type_dropdown",
            "generation_mode",
            "script_input",
            "generate_btn",
            "audio_output",
            "vocal_stem_output",
            "music_stem_output",
            "status_display",
            "promote_btn",
            "discard_btn",
            "promotion_status",
            "scorecard_display",
            "model_source_radio",
            "engine_dropdown",
            "speed_slider",
            "duck_slider",
            "reverb_slider",
            "stems_checkbox",
        )
        for key in required_keys:
            self.assertIn(key, components, f"Missing component key: {key}")

    def test_default_values(self):
        with gr.Blocks():
            components = build_sandbox_tab()

        self.assertEqual(components["benchmark_script_choice"].value, "🧘 5-Min Guided Meditation")
        self.assertEqual(components["generation_mode"].value, "Vocals Only")
        self.assertTrue(components["stems_checkbox"].value)
        self.assertEqual(components["script_input"].value, MEDITATION_5MIN_SCRIPT)

    def test_render_status_html(self):
        html_ready = render_sandbox_status("Ready", 0.0, "Ready to test")
        self.assertIn("status-shell", html_ready)
        self.assertIn("Ready to test", html_ready)

        html_done = render_sandbox_status("Complete", 1.0, elapsed=65)
        self.assertIn("status-complete", html_done)
        self.assertIn("1:05", html_done)

    def test_render_qa_scorecard_placeholder(self):
        html = render_qa_scorecard(None)
        self.assertIn("Audio QA Scorecard", html)
        self.assertIn("Generate audio to run quality metrics", html)

    def test_format_promoted_models_html(self):
        html = format_promoted_models_html()
        self.assertIsInstance(html, str)

    def test_discard_handler(self):
        audio, v_stem, m_stem, msg, status, scorecard = handle_discard_model()
        self.assertIsNone(audio)
        self.assertIsNone(v_stem)
        self.assertIsNone(m_stem)
        self.assertIn("discarded", msg)
