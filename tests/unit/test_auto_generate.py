"""Tests for the auto-generation orchestrator.

Uses fake engines and a stub pipeline — no model, no network, no audio.
"""

import json
import tempfile
import unittest
from pathlib import Path

from core.auto_generate import (
    AutoConfig,
    ScriptGenerationError,
    generate_script,
    run,
)
from core.script_gen.engine import FakeScriptEngine

CLEAN_SCRIPT = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:5s]\n\n"
    "If it feels right, you might let your eyes close.\n\n"
    "[pause:5s]\n\n"
    "Notice the weight of your hands."
)

UNSAFE_SCRIPT = "This meditation will cure your anxiety.\n\n[pause:5s]\n\nRest now."

BROKEN_SCRIPT = "Breathe in. [pause:4] Breathe out."


def judged(script: str, changelog: str = "- none") -> str:
    return f"<script>\n{script}\n</script>\n<changelog>\n{changelog}\n</changelog>"


class StubPipeline:
    """Stands in for MeditationPipeline; records the kwargs it was called with."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        wav = self.out_dir / "meditation.wav"
        wav.write_bytes(b"RIFF")
        return str(wav), "ok"


class TestScriptGeneration(unittest.TestCase):
    def setUp(self):
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_clean_script_passes_first_time(self):
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
            config=self.config,
        )
        self.assertEqual(outcome.script, CLEAN_SCRIPT)
        self.assertEqual(outcome.repairs_used, 0)

    def test_judge_revision_is_what_gets_used(self):
        revised = CLEAN_SCRIPT + "\n\nAnd rest."
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(revised, "- added a closing")]),
            config=self.config,
        )
        self.assertEqual(outcome.script, revised)
        self.assertIn("added a closing", outcome.changelog)

    def test_broken_script_is_repaired_and_accepted(self):
        judge = FakeScriptEngine([judged(BROKEN_SCRIPT), judged(CLEAN_SCRIPT)])
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([BROKEN_SCRIPT]),
            judge_engine=judge,
            config=self.config,
        )
        self.assertEqual(outcome.script, CLEAN_SCRIPT)
        self.assertEqual(outcome.repairs_used, 1)

    def test_persistent_safety_violation_raises(self):
        judge = FakeScriptEngine([judged(UNSAFE_SCRIPT)])
        with self.assertRaises(ScriptGenerationError) as ctx:
            generate_script(
                "I feel anxious",
                generator_engine=FakeScriptEngine([UNSAFE_SCRIPT]),
                judge_engine=judge,
                config=self.config,
            )
        self.assertIn("CLINICAL_CLAIM", str(ctx.exception))

    def test_repair_budget_is_respected(self):
        judge = FakeScriptEngine([judged(BROKEN_SCRIPT)])
        config = AutoConfig(
            target_min_sec=1.0, target_max_sec=100000.0, max_repairs=2
        )
        with self.assertRaises(ScriptGenerationError):
            generate_script(
                "p",
                generator_engine=FakeScriptEngine([BROKEN_SCRIPT]),
                judge_engine=judge,
                config=config,
            )
        # 1 review + 2 repairs = 3 judge calls.
        self.assertEqual(len(judge.calls), 3)

    def test_advisory_violation_does_not_raise(self):
        # Duration far outside the window is advisory only.
        config = AutoConfig(target_min_sec=100000.0, target_max_sec=200000.0)
        outcome = generate_script(
            "p",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)] * 5),
            config=config,
        )
        self.assertTrue(
            any(v.code == "DURATION_OUT_OF_WINDOW" for v in outcome.violations)
        )


FAKE_SCAN = lambda: [("Healing Forest — 23:12", "/bg/forest.mp3")]


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            background_scan=FAKE_SCAN,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, pipeline):
        return run(
            "I feel anxious",
            config=self.config,
            pipeline=pipeline,
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )

    def test_returns_the_pipeline_audio_path(self):
        pipeline = StubPipeline(self.dir)
        result = self._run(pipeline)
        self.assertTrue(Path(result.audio_path).is_file())

    def test_writes_script_and_meta_siblings(self):
        result = self._run(StubPipeline(self.dir))
        self.assertTrue(Path(result.script_path).is_file())
        self.assertTrue(Path(result.meta_path).is_file())
        self.assertEqual(
            Path(result.script_path).parent, Path(result.audio_path).parent
        )

    def test_script_file_holds_the_final_script(self):
        result = self._run(StubPipeline(self.dir))
        self.assertEqual(Path(result.script_path).read_text(), CLEAN_SCRIPT)

    def test_meta_records_prompt_models_and_background(self):
        result = self._run(StubPipeline(self.dir))
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["prompt"], "I feel anxious")
        self.assertIn("generator", meta)
        self.assertIn("judge", meta)
        self.assertEqual(meta["background"], "Healing Forest — 23:12")
        self.assertEqual(meta["background_path"], "/bg/forest.mp3")

    def test_pipeline_receives_the_script_and_background(self):
        pipeline = StubPipeline(self.dir)
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["script"], CLEAN_SCRIPT)
        self.assertEqual(call["uploaded_music_path"], "/bg/forest.mp3")

    def test_pipeline_uses_the_golden_path_defaults(self):
        pipeline = StubPipeline(self.dir)
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["tts_engine"], "f5")
        self.assertEqual(call["music_model"], "upload")

    def test_artifacts_are_written_when_script_generation_fails(self):
        pipeline = StubPipeline(self.dir)
        self.config.failure_dir = self.dir / "failures"
        with self.assertRaises(ScriptGenerationError):
            run(
                "I feel anxious",
                config=self.config,
                pipeline=pipeline,
                generator_engine=FakeScriptEngine([UNSAFE_SCRIPT]),
                judge_engine=FakeScriptEngine([judged(UNSAFE_SCRIPT)] * 5),
            )
        # Nothing rendered, but the failed script is on disk to read.
        self.assertEqual(pipeline.calls, [])
        failures = list(self.config.failure_dir.glob("*.script.txt"))
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
