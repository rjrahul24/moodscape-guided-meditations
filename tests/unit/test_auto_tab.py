"""Tests for the Auto-Generate tab builder.

Never imports app.py: that module loads torch and registers an atexit
hard-exit hook, so importing it from a test is not viable.
"""

import os
import unittest
from unittest.mock import patch

import gradio as gr

from core.auto_tab import auto_generate_handler, build_auto_tab


class GenreControlsTest(unittest.TestCase):
    def test_dropdown_choices_cover_every_pack_and_name_the_family(self):
        from core.auto_tab import genre_dropdown_choices

        choices = genre_dropdown_choices()
        self.assertEqual(len(choices), 46)
        labels = [label for label, _slug in choices]
        self.assertTrue(any("Sleep & Rest — Fall Asleep" == l for l in labels))
        self.assertEqual(len(set(labels)), len(labels))

    def test_every_choice_value_is_a_loadable_slug(self):
        from core.auto_tab import genre_dropdown_choices
        from core.genres import load_pack

        for _label, slug in genre_dropdown_choices():
            load_pack(slug)

    def test_the_tab_exposes_the_genre_band_and_steer_controls(self):
        from core.auto_tab import build_auto_tab

        with gr.Blocks():
            components = build_auto_tab()
        for key in ("genre", "band", "steer"):
            self.assertIn(key, components)

    def test_changing_genre_prefills_the_content_type(self):
        from core.auto_tab import content_type_for_genre

        self.assertEqual(content_type_for_genre("fall_asleep"), "sleep_story")
        self.assertEqual(content_type_for_genre("grief_and_loss"), "meditation")

    def test_band_choices_labels_match_duration_bands(self):
        from core.auto_tab import BAND_CHOICES
        from core.auto_generate import DURATION_BANDS

        band_dict = {slug: label for label, slug in BAND_CHOICES}
        for band_slug, (min_sec, max_sec) in DURATION_BANDS.items():
            min_min = min_sec / 60
            max_min = max_sec / 60
            expected_label = f"{min_min:.0f}–{max_min:.0f} min"
            self.assertEqual(
                band_dict[band_slug], expected_label,
                f"Band '{band_slug}' label does not match its duration range"
            )


class TestBuildAutoTab(unittest.TestCase):
    def test_builds_inside_a_blocks_context(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertIsInstance(components, dict)

    def test_exposes_the_components_the_handler_needs(self):
        with gr.Blocks():
            components = build_auto_tab()
        for key in (
            "genre", "band", "steer", "content_type", "tts_engine",
            "planner", "generator", "judge", "button", "audio", "status", "script", "changelog",
        ):
            self.assertIn(key, components)

    def test_defaults_to_the_golden_path(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertEqual(components["tts_engine"].value, "f5")
        self.assertEqual(components["content_type"].value, "meditation")

    def test_band_defaults_to_medium(self):
        with gr.Blocks():
            components = build_auto_tab()
        self.assertEqual(components["band"].value, "medium")


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
    """auto_generate_handler writes MOODSCAPE_SCRIPT_{PLANNER,GENERATOR,JUDGE}
    into os.environ as a side effect of wiring the UI's model textboxes
    through to build_engine(). Every call here must run inside
    patch.dict(os.environ, ...) so that write is rolled back afterwards --
    otherwise 'ollama:llama3.2:3b' leaks into the process environment and
    silently shadows AutoConfig.from_env()'s defaults for every test that
    runs later in the same process (e.g. test_auto_generate.py's
    engine-sharing guards, which depend on DEFAULT_PLANNER/_GENERATOR being
    read back unset)."""

    def test_empty_error_message_reports_failure_not_attributeerror(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch("core.auto_tab.StreamingRun", _FakeEmptyErrorRun):
                outputs = list(
                    auto_generate_handler(
                        "stress_relief",
                        "medium",
                        "",
                        "meditation",
                        "f5",
                        "ollama:llama3.2:3b",
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

        with patch.dict(os.environ, {}, clear=False):
            with patch("core.auto_tab.StreamingRun", FakeInvalidInputRun):
                outputs = list(
                    auto_generate_handler(
                        "stress_relief", "medium", "",
                        "meditation", "f5",
                        "ollama:llama3.2:3b", "ollama:llama3.2:3b", "ollama:llama3.2:3b",
                    )
                )
        self.assertEqual(outputs[-1][3], "")

    def test_steer_none_does_not_raise_attribute_error(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch("core.auto_tab.StreamingRun", _FakeEmptyErrorRun):
                outputs = list(
                    auto_generate_handler(
                        "stress_relief", "medium", None,
                        "meditation", "f5",
                        "ollama:llama3.2:3b", "ollama:llama3.2:3b", "ollama:llama3.2:3b",
                    )
                )
        self.assertEqual(outputs[-1][3], "Failed: ")


if __name__ == "__main__":
    unittest.main()
