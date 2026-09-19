"""Tests for the auto-generation orchestrator.

Uses fake engines and a stub pipeline — no model, no network, no audio.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.auto_generate import (
    RECENT_BACKGROUNDS_LIMIT,
    AutoConfig,
    ScriptGenerationError,
    _measure_actual_duration_sec,
    generate_script,
    run,
)
from core.script_gen.duration import estimate_duration_sec
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


class RealAudioStubPipeline:
    """Like StubPipeline, but writes a real (silent) WAV of a known duration.

    Exercises the actual_sec/estimate_ratio calibration math end-to-end
    without any TTS model or rendering -- just a synthetic silent clip
    written via soundfile, the same pattern other unit tests use.
    """

    def __init__(self, out_dir: Path, duration_sec: float):
        self.out_dir = out_dir
        self.duration_sec = duration_sec
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        import numpy as np
        import soundfile as sf

        self.calls.append(kwargs)
        wav = self.out_dir / "meditation.wav"
        sample_rate = 24000
        sf.write(
            wav,
            np.zeros(int(sample_rate * self.duration_sec), dtype="float32"),
            sample_rate,
        )
        return str(wav), "ok"


class TestScriptGeneration(unittest.TestCase):
    def setUp(self):
        # A per-test failure_dir: a fatal-violation test below writes failure
        # artifacts, and without an override that lands in the shared
        # <tempdir>/moodscape_failures every test/run uses by default.
        self._tmp = tempfile.TemporaryDirectory()
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            failure_dir=Path(self._tmp.name) / "failures",
        )

    def tearDown(self):
        self._tmp.cleanup()

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
            target_min_sec=1.0,
            target_max_sec=100000.0,
            max_repairs=2,
            failure_dir=self.config.failure_dir,
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

class EngineUnloadOnEveryExitPathTest(unittest.TestCase):
    """Every engine actually passed in must be unloaded no matter which
    stage raises -- planning, drafting, reviewing, or a repair. An 18 GB
    model left resident because an earlier stage never got the chance to
    unload it is exactly the swap / MPS-bus-error condition this exists to
    prevent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            corpus_dir=Path(self._tmp.name) / "corpus",
            failure_dir=Path(self._tmp.name) / "failures",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_planner_failure_unloads_planner_generator_and_judge(self):
        from types import SimpleNamespace

        pack = SimpleNamespace(label="Grief", banned=[])
        angle = SimpleNamespace(name="the empty chair")
        planner = FakeScriptEngine(["unused"])
        generator = FakeScriptEngine(["unused"])
        judge = FakeScriptEngine(["unused"])

        with patch("core.auto_generate.plan", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                generate_script(
                    "p",
                    generator_engine=generator,
                    judge_engine=judge,
                    planner_engine=planner,
                    pack=pack,
                    angle=angle,
                    config=self.config,
                )

        self.assertEqual(planner.unload_calls, 1)
        self.assertEqual(generator.unload_calls, 1)
        self.assertEqual(judge.unload_calls, 1)

    def test_draft_failure_unloads_generator_and_judge(self):
        generator = FakeScriptEngine(["unused"])
        judge = FakeScriptEngine(["unused"])

        with patch("core.auto_generate.draft", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                generate_script(
                    "p",
                    generator_engine=generator,
                    judge_engine=judge,
                    config=self.config,
                )

        self.assertEqual(generator.unload_calls, 1)
        self.assertEqual(judge.unload_calls, 1)

    def test_review_failure_unloads_generator_and_judge(self):
        generator = FakeScriptEngine([CLEAN_SCRIPT])
        judge = FakeScriptEngine(["unused"])

        with patch("core.auto_generate.review", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                generate_script(
                    "p",
                    generator_engine=generator,
                    judge_engine=judge,
                    config=self.config,
                )

        self.assertEqual(generator.unload_calls, 1)
        self.assertEqual(judge.unload_calls, 1)

    def test_repair_failure_unloads_generator_and_judge(self):
        generator = FakeScriptEngine([BROKEN_SCRIPT])
        judge = FakeScriptEngine([judged(BROKEN_SCRIPT)])

        with patch("core.auto_generate.repair", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                generate_script(
                    "p",
                    generator_engine=generator,
                    judge_engine=judge,
                    config=self.config,
                )

        self.assertEqual(generator.unload_calls, 1)
        self.assertEqual(judge.unload_calls, 1)

    def test_shared_planner_and_generator_is_unloaded_once_on_draft_failure(self):
        shared = FakeScriptEngine(["a brief"])
        judge = FakeScriptEngine(["unused"])
        from types import SimpleNamespace

        pack = SimpleNamespace(label="Grief", banned=[])
        angle = SimpleNamespace(name="the empty chair")

        with patch("core.auto_generate.plan", return_value="a brief"):
            with patch("core.auto_generate.draft", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    generate_script(
                        "p",
                        generator_engine=shared,
                        judge_engine=judge,
                        planner_engine=shared,
                        pack=pack,
                        angle=angle,
                        config=self.config,
                    )

        # Exactly once, not twice, even though the shared engine plays both
        # the planner and generator roles.
        self.assertEqual(shared.unload_calls, 1)
        self.assertEqual(judge.unload_calls, 1)


FAKE_SCAN = lambda: [("Healing Forest — 23:12", "/bg/forest.mp3")]


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            background_scan=FAKE_SCAN,
            # Per-test failure_dir so a failing test never writes into the
            # shared <tempdir>/moodscape_failures.
            failure_dir=self.dir / "failures",
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

    def test_result_exposes_the_exact_background_path(self):
        # pick_background's exclusion matching is exact string comparison on
        # the path, so a caller building up recent_backgrounds across runs
        # needs this exact string on AutoResult, not just the human label.
        pipeline = StubPipeline(self.dir)
        result = self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(result.background_path, call["uploaded_music_path"])

    def test_run_succeeds_with_stub_audio_and_records_null_actual_sec(self):
        # StubPipeline writes b"RIFF" -- not a real audio file. Reading its
        # duration must fail, but that failure must never fail a job that
        # already produced "audio": run() still succeeds, and actual_sec is
        # recorded as null rather than raising.
        pipeline = StubPipeline(self.dir)
        result = self._run(pipeline)
        self.assertTrue(Path(result.audio_path).is_file())
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertIsNone(meta["actual_sec"])

    def test_meta_metadata_shape_includes_duration_calibration_keys(self):
        result = self._run(StubPipeline(self.dir))
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertIn("estimated_sec", meta)
        self.assertIn("actual_sec", meta)
        self.assertIn("estimate_ratio", meta)
        # estimate_ratio can't be computed without a real actual_sec.
        self.assertIsNone(meta["estimate_ratio"])

    def test_run_records_actual_sec_and_ratio_for_a_real_render(self):
        estimated = estimate_duration_sec(
            CLEAN_SCRIPT,
            engine=self.config.tts_engine,
            content_type=self.config.content_type,
        )
        pipeline = RealAudioStubPipeline(self.dir, duration_sec=estimated * 2)
        result = self._run(pipeline)
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertAlmostEqual(meta["actual_sec"], estimated * 2, delta=0.05)
        self.assertAlmostEqual(meta["estimate_ratio"], 2.0, places=2)

    def test_pipeline_uses_the_golden_path_defaults(self):
        pipeline = StubPipeline(self.dir)
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["tts_engine"], "f5")
        self.assertEqual(call["music_model"], "upload")
        self.assertEqual(call["speed"], 0.80)
        self.assertEqual(call["duck_amount_db"], -16.0)
        self.assertEqual(call["reverb_amount"], 0.15)

    def test_pipeline_receives_sleep_story_profile_settings(self):
        pipeline = StubPipeline(self.dir)
        self.config.content_type = "sleep_story"
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["speed"], 0.75)
        self.assertEqual(call["duck_amount_db"], -11.0)
        self.assertEqual(call["reverb_amount"], 0.18)

    def test_pipeline_receives_kokoro_voice_default(self):
        pipeline = StubPipeline(self.dir)
        self.config.tts_engine = "kokoro"
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["voice"], "balanced_calm")

    def test_explicit_speed_and_voice_override_defaults(self):
        pipeline = StubPipeline(self.dir)
        self.config.speed = 0.72
        self.config.voice = "deep_rest"
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["speed"], 0.72)
        self.assertEqual(call["voice"], "deep_rest")

    def test_artifacts_are_written_when_script_generation_fails(self):
        pipeline = StubPipeline(self.dir)
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

        # The draft (pre-judge) must also survive — without it you can't tell
        # what the generator produced versus what the judge mutated.
        drafts = list(self.config.failure_dir.glob("*.draft.txt"))
        self.assertTrue(drafts)
        self.assertEqual(drafts[0].read_text(), UNSAFE_SCRIPT)

        # The changelog of every review/repair attempt must be in meta.json —
        # not just the surviving violations — so a reader can see what was
        # already tried, not only the final broken state.
        metas = list(self.config.failure_dir.glob("*.meta.json"))
        self.assertTrue(metas)
        meta = json.loads(metas[0].read_text())
        self.assertIn("changelog", meta)
        self.assertTrue(meta["changelog"].strip())


FAKE_SCAN_TWO_TRACKS = lambda: [
    ("Track One", "/bg/one.mp3"),
    ("Track Two", "/bg/two.mp3"),
]


class TestRecentBackgrounds(unittest.TestCase):
    """recent_backgrounds has a producer (run()) as well as a consumer.

    Only has effect when a caller reuses one AutoConfig across run() calls
    (batch/scripted use) -- these tests do exactly that, sharing self.config.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            background_scan=FAKE_SCAN_TWO_TRACKS,
            failure_dir=self.dir / "failures",
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

    def test_successive_runs_sharing_a_config_avoid_the_same_track(self):
        pipeline = StubPipeline(self.dir)
        first = self._run(pipeline)
        second = self._run(pipeline)
        # Pool has exactly two tracks; after run 1 excludes its own pick,
        # run 2 has only the other track left as a candidate.
        self.assertNotEqual(first.background_path, second.background_path)

    def test_recent_backgrounds_is_capped_at_the_limit(self):
        pipeline = StubPipeline(self.dir)
        for _ in range(RECENT_BACKGROUNDS_LIMIT + 3):
            self._run(pipeline)
        self.assertLessEqual(
            len(self.config.recent_backgrounds), RECENT_BACKGROUNDS_LIMIT
        )


class TestMeasureActualDuration(unittest.TestCase):
    """_measure_actual_duration_sec must never raise -- it degrades to None
    on anything unreadable, since a calibration nicety must never fail a
    job that already produced audio."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_duration_from_a_real_wav_file(self):
        import numpy as np
        import soundfile as sf

        path = self.dir / "clip.wav"
        sample_rate = 24000
        sf.write(path, np.zeros(sample_rate * 2, dtype="float32"), sample_rate)
        self.assertAlmostEqual(
            _measure_actual_duration_sec(str(path)), 2.0, places=2
        )

    def test_returns_none_for_a_non_audio_file(self):
        # Mirrors StubPipeline's b"RIFF" placeholder in the tests above.
        path = self.dir / "not_audio.wav"
        path.write_bytes(b"RIFF")
        self.assertIsNone(_measure_actual_duration_sec(str(path)))

    def test_returns_none_for_a_missing_file(self):
        self.assertIsNone(
            _measure_actual_duration_sec(str(self.dir / "missing.wav"))
        )

    def test_returns_none_when_soundfile_is_not_importable(self):
        # Regression guard: `import soundfile as sf` used to sit outside the
        # try/except, so an ImportError would propagate out of run() BEFORE
        # meta.json and .script.txt are written -- losing every artifact
        # after a successful render. The import now lives inside the same
        # try/except that already swallows read failures.
        import sys

        with patch.dict(sys.modules, {"soundfile": None}):
            self.assertIsNone(
                _measure_actual_duration_sec(str(self.dir / "clip.wav"))
            )


class TestAutoConfigFromEnv(unittest.TestCase):
    """A malformed env var must be reported, not silently defaulted or let
    escape as a bare ValueError the caller's except ScriptGenerationError
    would not catch."""

    def test_malformed_max_repairs_raises_script_generation_error(self):
        with patch.dict(
            "os.environ", {"MOODSCAPE_SCRIPT_MAX_REPAIRS": "abc"}, clear=False
        ):
            with self.assertRaises(ScriptGenerationError) as ctx:
                AutoConfig.from_env()
        self.assertIn("MOODSCAPE_SCRIPT_MAX_REPAIRS", str(ctx.exception))

    def test_malformed_target_min_sec_raises_script_generation_error(self):
        with patch.dict(
            "os.environ", {"MOODSCAPE_TARGET_MIN_SEC": "not-a-number"}, clear=False
        ):
            with self.assertRaises(ScriptGenerationError) as ctx:
                AutoConfig.from_env()
        self.assertIn("MOODSCAPE_TARGET_MIN_SEC", str(ctx.exception))

    def test_malformed_target_max_sec_raises_script_generation_error(self):
        with patch.dict(
            "os.environ", {"MOODSCAPE_TARGET_MAX_SEC": "not-a-number"}, clear=False
        ):
            with self.assertRaises(ScriptGenerationError) as ctx:
                AutoConfig.from_env()
        self.assertIn("MOODSCAPE_TARGET_MAX_SEC", str(ctx.exception))


class FromGenrePrecedenceTest(unittest.TestCase):
    """from_genre() must not bypass from_env() -- it is the only path the UI
    has, and MOODSCAPE_ORIGINALITY=0 is documented as a kill switch.

    Precedence, low to high: dataclass default -> environment -> pack ->
    explicit override.
    """

    def _pack(self):
        from core.genres import load_pack

        return load_pack("grief_and_loss")

    def test_default_level_matches_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOODSCAPE_ORIGINALITY", None)
            os.environ.pop("MOODSCAPE_SCRIPT_MAX_REPAIRS", None)
            config = AutoConfig.from_genre(self._pack(), band="short")
        self.assertTrue(config.originality)
        self.assertEqual(config.max_repairs, 2)

    def test_environment_level_is_honoured(self):
        """The bug: MOODSCAPE_ORIGINALITY=0 previously did nothing on the
        genre path because from_genre() built AutoConfig directly."""
        with patch.dict(
            os.environ,
            {"MOODSCAPE_ORIGINALITY": "0", "MOODSCAPE_SCRIPT_MAX_REPAIRS": "5"},
        ):
            config = AutoConfig.from_genre(self._pack(), band="short")
        self.assertFalse(config.originality)
        self.assertEqual(config.max_repairs, 5)

    def test_pack_level_beats_environment(self):
        """The duration band's seconds must win over a stray env override --
        the band is what the UI's Length radio actually controls."""
        with patch.dict(
            os.environ,
            {"MOODSCAPE_TARGET_MIN_SEC": "999", "MOODSCAPE_TARGET_MAX_SEC": "1000"},
        ):
            config = AutoConfig.from_genre(self._pack(), band="short")
        self.assertEqual(config.target_min_sec, 180.0)
        self.assertEqual(config.target_max_sec, 360.0)
        self.assertEqual(config.content_type, self._pack().content_type)
        self.assertEqual(config.genre, "grief_and_loss")

    def test_explicit_override_beats_pack(self):
        with patch.dict(os.environ, {}, clear=False):
            config = AutoConfig.from_genre(
                self._pack(), band="short", content_type="sleep_story",
                target_min_sec=42.0,
            )
        self.assertEqual(config.content_type, "sleep_story")
        self.assertEqual(config.target_min_sec, 42.0)


class OriginalityIntegrationTest(unittest.TestCase):
    """The core requirement: the same script twice must be caught."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, **overrides):
        values = {
            "corpus_dir": self.dir / "corpus",
            "background_scan": lambda: [("Bed — 10:00", "/bg/a.mp3")],
            "genre": "sleep",
        }
        values.update(overrides)
        return AutoConfig(**values)

    def _run_once(self, script: str, config):
        pipeline = StubPipeline(self.dir)
        return run(
            "a prompt",
            config=config,
            pipeline=pipeline,
            generator_engine=FakeScriptEngine([script]),
            judge_engine=FakeScriptEngine([judged(script)]),
        )

    def test_a_successful_run_is_added_to_the_corpus(self):
        from core.originality import load_corpus

        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)
        entries = load_corpus(genre="sleep", corpus_dir=config.corpus_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, CLEAN_SCRIPT)

    def test_regenerating_an_identical_script_is_fatal(self):
        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)

        with self.assertRaises(ScriptGenerationError) as ctx:
            self._run_once(CLEAN_SCRIPT, config)

        self.assertIn("PASSAGE_LIFTED", str(ctx.exception))

    def test_originality_can_be_switched_off(self):
        config = self._config(originality=False)
        self._run_once(CLEAN_SCRIPT, config)
        result = self._run_once(CLEAN_SCRIPT, config)
        self.assertTrue(result.audio_path)

    def test_metadata_records_the_similarity_score(self):
        config = self._config()
        result = self._run_once(CLEAN_SCRIPT, config)
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertIn("originality_cosine", meta)
        self.assertIn("originality_shared_span", meta)
        self.assertEqual(meta["genre"], "sleep")

    def test_a_different_script_in_the_same_genre_passes(self):
        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)
        other = (
            "Let the day set itself down for a moment.\n\n"
            "[pause:5s]\n\n"
            "Copper light moves slowly along the far wall.\n\n"
            "[pause:5s]\n\n"
            "Nothing here needs deciding tonight."
        )
        result = self._run_once(other, config)
        self.assertTrue(result.audio_path)


class OriginalityEnvTest(unittest.TestCase):
    def test_default_is_on(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOODSCAPE_ORIGINALITY", None)
            self.assertTrue(AutoConfig.from_env().originality)

    def test_zero_disables_it(self):
        with patch.dict(os.environ, {"MOODSCAPE_ORIGINALITY": "0"}):
            self.assertFalse(AutoConfig.from_env().originality)

    def test_a_typo_is_reported_not_silently_defaulted(self):
        with patch.dict(os.environ, {"MOODSCAPE_ORIGINALITY": "maybe"}):
            with self.assertRaises(ScriptGenerationError):
                AutoConfig.from_env()


class GenrePathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, genre="grief_and_loss", band="medium", steer="", **kw):
        from core.auto_generate import DURATION_BANDS
        from core.genres import load_pack

        pack = load_pack(genre)
        config = AutoConfig.from_genre(
            pack,
            band=band,
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        self.planner = FakeScriptEngine(["A brief about a chair by a window."])
        self.generator = FakeScriptEngine([CLEAN_SCRIPT])
        self.judge = FakeScriptEngine([judged(CLEAN_SCRIPT)])
        self.pipeline = StubPipeline(self.dir)
        return run(
            "",
            genre=genre,
            steer=steer,
            config=config,
            pipeline=self.pipeline,
            planner_engine=self.planner,
            generator_engine=self.generator,
            judge_engine=self.judge,
            **kw,
        ), config

    def test_duration_bands_cover_the_three_ui_options(self):
        from core.auto_generate import DURATION_BANDS

        self.assertEqual(DURATION_BANDS["short"], (180.0, 360.0))
        self.assertEqual(DURATION_BANDS["medium"], (360.0, 600.0))
        self.assertEqual(DURATION_BANDS["long"], (600.0, 900.0))

    def test_from_genre_copies_the_packs_deterministic_fields(self):
        from core.genres import load_pack

        config = AutoConfig.from_genre(load_pack("fall_asleep"), band="short")
        self.assertEqual(config.content_type, "sleep_story")
        self.assertEqual(config.genre, "fall_asleep")
        self.assertEqual(config.target_min_sec, 180.0)
        self.assertEqual(config.target_max_sec, 360.0)

    def test_the_planners_brief_becomes_the_writers_prompt(self):
        _result, _config = self._run()
        self.assertIn(
            "A brief about a chair by a window.", self.generator.calls[0]["user"]
        )

    def test_the_chosen_angle_is_recorded(self):
        from core.genres import load_pack

        result, _config = self._run()
        names = {a.name for a in load_pack("grief_and_loss").angles}
        self.assertIn(result.angle, names)

    def test_steer_text_reaches_the_planner(self):
        _result, _config = self._run(steer="after a long hospital week")
        self.assertIn("hospital week", self.planner.calls[0]["user"])

    def test_the_packs_music_tags_reach_the_background_picker(self):
        seen = {}

        def fake_pick(**kwargs):
            seen.update(kwargs)
            return ("Bed — 10:00", "/bg/a.mp3")

        with patch("core.auto_generate.pick_background", fake_pick):
            self._run()
        self.assertEqual(tuple(seen["prefer_tags"]), ("warm", "sparse"))

    def test_every_engine_is_preflighted_and_unloaded(self):
        self._run()
        self.assertEqual(self.judge.preflight_calls, 1)
        self.assertGreaterEqual(self.generator.unload_calls, 1)
        self.assertGreaterEqual(self.judge.unload_calls, 1)

    def test_a_shared_planner_and_writer_engine_is_not_unloaded_between_them(self):
        """The default config uses one model for both stages; unloading
        between them would pay an 18 GB reload for nothing."""
        shared = FakeScriptEngine(["A brief.", CLEAN_SCRIPT])
        from core.genres import load_pack

        config = AutoConfig.from_genre(
            load_pack("grief_and_loss"),
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        run(
            "", genre="grief_and_loss", config=config,
            pipeline=StubPipeline(self.dir),
            planner_engine=shared, generator_engine=shared,
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )
        self.assertEqual(shared.unload_calls, 1)

    def test_default_construction_shares_one_engine_for_matching_specs(self):
        """run()'s own (non-injected) construction path must reuse one
        engine for planner + writer when their specs match -- the default
        config's whole reason to exist. build_engine() does no caching, so
        building both separately would hand generate_script() two distinct
        objects even for identical specs, silently defeating the `is not`
        unload check (always true) and paying a full model reload between
        planning and writing on every real run."""
        from core.auto_generate import DEFAULT_GENERATOR, DEFAULT_PLANNER
        from core.genres import load_pack

        self.assertEqual(DEFAULT_PLANNER, DEFAULT_GENERATOR)  # the case under test

        built: dict[str, list] = {}

        def fake_build_engine(spec):
            engine = FakeScriptEngine(["A generated brief.", CLEAN_SCRIPT])
            built.setdefault(spec, []).append(engine)
            return engine

        config = AutoConfig.from_genre(
            load_pack("grief_and_loss"),
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        with patch("core.auto_generate.build_engine", fake_build_engine):
            result = run(
                "",
                genre="grief_and_loss",
                config=config,
                pipeline=StubPipeline(self.dir),
                judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
            )

        self.assertTrue(result.audio_path)
        # Exactly one engine built for the shared spec -- not one per role.
        self.assertEqual(len(built[DEFAULT_GENERATOR]), 1)
        shared_engine = built[DEFAULT_GENERATOR][0]
        # One unload after writing; none between planning and writing.
        self.assertEqual(shared_engine.unload_calls, 1)

    def test_default_construction_builds_two_engines_for_differing_specs(self):
        """When the planner and generator specs differ, run() must build two
        distinct engines and unload the planner separately -- the other half
        of the branch the shared-spec test above does not exercise."""
        from core.auto_generate import DEFAULT_GENERATOR
        from core.genres import load_pack

        other_spec = "ollama:some-other-model:7b"
        built: dict[str, list] = {}

        def fake_build_engine(spec):
            engine = FakeScriptEngine(["A generated brief.", CLEAN_SCRIPT])
            built.setdefault(spec, []).append(engine)
            return engine

        config = AutoConfig.from_genre(
            load_pack("grief_and_loss"),
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        with patch.dict(os.environ, {"MOODSCAPE_SCRIPT_PLANNER": other_spec}):
            with patch("core.auto_generate.build_engine", fake_build_engine):
                result = run(
                    "",
                    genre="grief_and_loss",
                    config=config,
                    pipeline=StubPipeline(self.dir),
                    judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
                )

        self.assertTrue(result.audio_path)
        self.assertEqual(len(built[other_spec]), 1)
        self.assertEqual(len(built[DEFAULT_GENERATOR]), 1)
        planner_engine = built[other_spec][0]
        generator_engine = built[DEFAULT_GENERATOR][0]
        self.assertIsNot(planner_engine, generator_engine)
        self.assertEqual(planner_engine.unload_calls, 1)

    def test_the_brief_and_genre_land_in_the_metadata(self):
        result, _config = self._run()
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["genre"], "grief_and_loss")
        self.assertIn("brief", meta)
        self.assertTrue(meta["angle"])

    def test_genre_argument_sets_config_genre_on_a_bare_config(self):
        """Regression: a caller doing
        run("", genre="grief_and_loss", config=AutoConfig(...)) -- i.e. a
        config NOT built via from_genre() -- must still get config.genre
        set. Angle rotation reads the `genre` ARGUMENT, but comparison, the
        avoid-list and the corpus write all read config.genre; before this
        fix a bare AutoConfig() left genre="" and the script silently filed
        under the empty-string partition with no music tags."""
        config = AutoConfig(
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        result = run(
            "",
            genre="grief_and_loss",
            config=config,
            pipeline=StubPipeline(self.dir),
            planner_engine=FakeScriptEngine(["A brief about a chair."]),
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )
        self.assertEqual(config.genre, "grief_and_loss")
        self.assertEqual(result.genre, "grief_and_loss")

        from core.originality import load_corpus

        entries = load_corpus(genre="grief_and_loss", corpus_dir=config.corpus_dir)
        self.assertEqual(len(entries), 1)

    def test_recently_used_angles_are_avoided(self):
        from core.genres import load_pack
        from core.originality import add_to_corpus

        pack = load_pack("grief_and_loss")
        corpus = self.dir / "corpus"
        for angle in list(pack.angles)[:-1]:
            add_to_corpus("x", genre="grief_and_loss", angle=angle.name,
                          corpus_dir=corpus)
        result, _config = self._run()
        self.assertEqual(result.angle, pack.angles[-1].name)

    def test_the_prompt_path_still_works_unchanged(self):
        """Backward compatibility: no genre, positional prompt, as before."""
        result = run(
            "I feel anxious.",
            config=AutoConfig(
                corpus_dir=self.dir / "corpus",
                background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
            ),
            pipeline=StubPipeline(self.dir),
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )
        self.assertTrue(result.audio_path)
        self.assertEqual(result.genre, "")


if __name__ == "__main__":
    unittest.main()
