"""Genre -> finished meditation, end to end with a stub pipeline.

No model, no network, no audio rendering, so this always runs -- unlike
test_auto_generate_e2e.py, which drives the real pipeline behind
MOODSCAPE_E2E=1.
"""

import json
import tempfile
import unittest
from pathlib import Path


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


REALISTIC_SCRIPT = (
    "Let the day set itself down for a moment.\n\n"
    "[pause:6s]\n\n"
    "Copper light moves slowly along the far wall, and nothing here needs "
    "deciding tonight.\n\n"
    "[pause:8s]\n\n"
    "Feel the weight of your hands where they rest.\n\n"
    "[pause:6s]\n\n"
    "When you are ready, let the room come back.\n"
)


def judged(script: str) -> str:
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


class GenreEndToEndTest(unittest.TestCase):
    """Genre -> audio with a stub pipeline. No model, no network, no render."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, genre="stress_relief"):
        from core.auto_generate import AutoConfig
        from core.genres import load_pack

        return AutoConfig.from_genre(
            load_pack(genre),
            band="medium",
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )

    def test_a_genre_run_produces_audio_script_and_metadata(self):
        from core.auto_generate import run
        from core.script_gen.engine import FakeScriptEngine

        config = self._config()
        result = run(
            "",
            genre="stress_relief",
            config=config,
            pipeline=StubPipeline(self.dir),
            planner_engine=FakeScriptEngine(["A brief."]),
            generator_engine=FakeScriptEngine([REALISTIC_SCRIPT]),
            judge_engine=FakeScriptEngine([
                f"<script>\n{REALISTIC_SCRIPT}\n</script>\n<changelog>\n- none\n</changelog>"
            ]),
        )
        self.assertTrue(Path(result.audio_path).is_file())
        self.assertTrue(Path(result.script_path).is_file())
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["genre"], "stress_relief")
        self.assertTrue(meta["angle"])

    def test_the_same_genre_twice_with_the_same_text_is_rejected(self):
        """The requirement, end to end: no two meditations may be the same."""
        from core.auto_generate import ScriptGenerationError, run
        from core.script_gen.engine import FakeScriptEngine

        config = self._config()

        def once():
            return run(
                "",
                genre="stress_relief",
                config=config,
                pipeline=StubPipeline(self.dir),
                planner_engine=FakeScriptEngine(["A brief."]),
                generator_engine=FakeScriptEngine([REALISTIC_SCRIPT]),
                judge_engine=FakeScriptEngine([
                    f"<script>\n{REALISTIC_SCRIPT}\n</script>\n<changelog>\n- none\n</changelog>"
                ]),
            )

        once()
        with self.assertRaises(ScriptGenerationError) as ctx:
            once()
        self.assertIn("PASSAGE_LIFTED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
