"""Tests for the planner: genre pack + angle -> prose creative brief."""

import unittest

from core.genres import Angle, GenrePack
from core.script_gen.engine import FakeScriptEngine
from core.script_gen.planner import plan

PACK = GenrePack(
    slug="grief_and_loss",
    label="Grief & Loss",
    family="Emotional",
    content_type="meditation",
    music_tags=("warm", "sparse"),
    pause_ratio=0.34,
    technique="RAIN, held loosely. Never resolve the grief.",
    arc=("arrival", "the body's weather", "one memory", "kindness", "return"),
    safety="No stage models. Do not imply closure.",
    banned=("time heals", "move on"),
    angles=(
        Angle(name="the empty chair", imagery=("a chair by a window", "cold tea")),
        Angle(name="tidal", imagery=("a shoreline at dusk", "wet sand")),
    ),
)

BRIEF = "Write a grief meditation built around a chair by a window."


class PlanTest(unittest.TestCase):
    def _plan(self, **kwargs):
        engine = FakeScriptEngine([BRIEF])
        result = plan(
            engine,
            PACK,
            PACK.angles[0],
            system="SYSTEM",
            target_min_sec=360.0,
            target_max_sec=600.0,
            **kwargs,
        )
        return engine, result

    def test_returns_the_models_brief_verbatim(self):
        _engine, brief = self._plan()
        self.assertEqual(brief, BRIEF)

    def test_strips_a_markdown_fence(self):
        engine = FakeScriptEngine([f"```\n{BRIEF}\n```"])
        brief = plan(
            engine, PACK, PACK.angles[0], system="S",
            target_min_sec=360.0, target_max_sec=600.0,
        )
        self.assertEqual(brief, BRIEF)

    def test_the_prompt_carries_the_pack_and_the_chosen_angle(self):
        engine, _brief = self._plan()
        user = engine.calls[0]["user"]
        self.assertIn("Grief & Loss", user)
        self.assertIn("RAIN, held loosely", user)
        self.assertIn("the empty chair", user)
        self.assertIn("a chair by a window", user)
        self.assertIn("the body's weather", user)

    def test_the_prompt_carries_the_banned_phrases(self):
        engine, _brief = self._plan()
        self.assertIn("time heals", engine.calls[0]["user"])

    def test_the_prompt_does_not_leak_the_other_angle(self):
        """One angle per run, or the writer blends them into mush."""
        engine, _brief = self._plan()
        self.assertNotIn("tidal", engine.calls[0]["user"])
        self.assertNotIn("wet sand", engine.calls[0]["user"])

    def test_the_avoid_list_appears_when_given(self):
        engine, _brief = self._plan(avoid=["copper staircase", "a distant train"])
        self.assertIn("copper staircase", engine.calls[0]["user"])

    def test_no_avoid_section_when_the_list_is_empty(self):
        engine, _brief = self._plan()
        self.assertNotIn("Do not reuse", engine.calls[0]["user"])

    def test_steer_text_is_included(self):
        engine, _brief = self._plan(steer="for a night shift worker")
        self.assertIn("night shift worker", engine.calls[0]["user"])

    def test_the_pause_budget_is_expressed_as_a_percentage(self):
        engine, _brief = self._plan()
        user = engine.calls[0]["user"]
        # 34% of the 360-600s window is roughly 122-204 seconds of silence.
        self.assertIn("34%", user)

    def test_the_system_prompt_is_passed_through(self):
        engine, _brief = self._plan()
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")

    def test_empty_imagery_omits_the_header(self):
        """Imagery is guarded; dangling headers must never appear."""
        empty_angle = Angle(name="bare", imagery=())
        engine = FakeScriptEngine([BRIEF])
        plan(
            engine, PACK, empty_angle, system="S",
            target_min_sec=360.0, target_max_sec=600.0,
        )
        self.assertNotIn("Imagery to build on", engine.calls[0]["user"])


if __name__ == "__main__":
    unittest.main()
