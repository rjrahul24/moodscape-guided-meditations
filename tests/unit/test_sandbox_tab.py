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
            "generation_mode",
            "script_input",
            "music_prompt",
            "generate_btn",
            "audio_output",
            "status_display",
            "engine_dropdown",
            "voice_dropdown",
            "speed_slider",
            "f5_wpm_slider",
            "f5_cfg_slider",
            "df_wet_slider",
            "duck_slider",
            "reverb_slider",
            "reverb_ir_dropdown",
            "fade_in_slider",
            "fade_out_slider",
            "uploaded_music",
            "refresh_backgrounds_btn",
        )
        for key in required_keys:
            self.assertIn(key, components, f"Missing component key: {key}")

        # Ensure removed clutter components are not present
        removed_keys = (
            "vocal_stem_output",
            "music_stem_output",
            "promote_btn",
            "discard_btn",
            "promotion_status",
            "scorecard_display",
            "model_source_radio",
            "stems_checkbox",
        )
        for key in removed_keys:
            self.assertNotIn(key, components, f"Component should have been removed: {key}")

    def test_default_values(self):
        with gr.Blocks():
            components = build_sandbox_tab()

        self.assertEqual(components["benchmark_script_choice"].value, "🧘 5-Min Guided Meditation")
        self.assertEqual(components["generation_mode"].value, "Vocals Only")
        self.assertEqual(components["script_input"].value, MEDITATION_5MIN_SCRIPT)
        self.assertEqual(components["engine_dropdown"].value, "chatterbox")

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
        audio, msg, status = handle_discard_model()
        self.assertIsNone(audio)
        self.assertIn("discarded", msg)
