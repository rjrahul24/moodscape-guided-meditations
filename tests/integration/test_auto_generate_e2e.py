"""End-to-end: fake script models, real audio pipeline.

Slow — renders actual audio. Run with:
    MOODSCAPE_E2E=1 .venv/bin/python -m pytest \
        tests/integration/test_auto_generate_e2e.py -v

Everything else stubs the pipeline, so this is the only test that proves the
orchestrator's kwargs actually satisfy MeditationPipeline.generate().
"""

import json
import os
import unittest
from pathlib import Path

import soundfile as sf

from core.auto_generate import AutoConfig, run
from core.script_gen.engine import FakeScriptEngine

E2E = os.environ.get("MOODSCAPE_E2E") == "1"

# Short on purpose: a full 5-7 minute render would make this unusable.
SHORT_SCRIPT = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:3s]\n\n"
    "Notice the weight of your hands.\n\n"
    "[pause:3s]\n\n"
    "And when you are ready, let your eyes open."
)

# Realistic length (198 words excluding [pause:Xs] markers / 17 sentences),
# used ONLY for the duration estimate test. Fixed per-chunk overhead (F5
# reference-audio padding, leading/trailing silence) is roughly constant per
# chunk regardless of script length, so it dominates a short script like
# SHORT_SCRIPT and swamps the per-word speaking rate the estimator is
# actually trying to measure. On a 22-word script measured separately, the
# observed ratio was 2.14 (implied ~37 WPM) purely from that overhead
# amortizing over too few words; that measurement is not this script and is
# not reproduced here. A script in this length range is long enough for
# per-word rate to dominate over fixed overhead, so it is the only length
# that can actually validate DEFAULT_WPM.
REALISTIC_SCRIPT = (
    "Find a comfortable position, either sitting or lying down.\n\n"
    "[pause:4s]\n\n"
    "Allow your eyes to close gently, or soften your gaze toward the floor.\n\n"
    "[pause:4s]\n\n"
    "Begin to notice the natural rhythm of your breath, without trying to change it.\n\n"
    "[pause:4s]\n\n"
    "Feel the air moving in through your nose, cool and light.\n\n"
    "[pause:4s]\n\n"
    "And feel it leaving again, a little warmer, a little slower.\n\n"
    "[pause:4s]\n\n"
    "With each exhale, let your shoulders soften a little more.\n\n"
    "[pause:4s]\n\n"
    "Let your jaw unclench, and let your forehead smooth out.\n\n"
    "[pause:4s]\n\n"
    "Bring your attention to the points of contact between your body and the surface beneath you.\n\n"
    "[pause:4s]\n\n"
    "Notice where you feel supported, held, and at ease.\n\n"
    "[pause:4s]\n\n"
    "If your mind wanders, that is completely normal.\n\n"
    "[pause:4s]\n\n"
    "Simply notice where it went, and gently guide your attention back to the breath.\n\n"
    "[pause:4s]\n\n"
    "There is nowhere else you need to be right now, and nothing else you need to do.\n\n"
    "[pause:4s]\n\n"
    "Just this breath, and the next one after it.\n\n"
    "[pause:4s]\n\n"
    "Notice any lingering tension in your legs, and let it drain away with the next exhale.\n\n"
    "[pause:4s]\n\n"
    "Let a sense of quiet settle over you, steady and unhurried.\n\n"
    "[pause:4s]\n\n"
    "When you feel ready, begin to deepen your breath.\n\n"
    "[pause:4s]\n\n"
    "Wiggle your fingers and toes, and slowly let your eyes open."
)


def judged(script):
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


@unittest.skipUnless(E2E, "set MOODSCAPE_E2E=1 to run (renders real audio)")
class TestAutoGenerateEndToEnd(unittest.TestCase):
    def setUp(self):
        # Wide window: this deliberately short script is nowhere near 5 minutes.
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_produces_a_playable_wav_and_its_siblings(self):
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )

        audio = Path(result.audio_path)
        self.assertTrue(audio.is_file())
        self.assertGreater(audio.stat().st_size, 1000)

        info = sf.info(str(audio))
        self.assertGreater(info.duration, 5.0)

        self.assertTrue(Path(result.script_path).is_file())
        self.assertTrue(Path(result.meta_path).is_file())

    def test_metadata_records_the_real_run(self):
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["tts_engine"], "f5")
        self.assertTrue(meta["background"])
        self.assertTrue(Path(meta["background_path"]).is_file())

    def test_duration_estimate_is_within_thirty_percent_of_actual(self):
        """Guards DEFAULT_WPM against drift.

        Loose on purpose: F5 renders are not deterministic, and this exists to
        catch a badly wrong constant, not to pin an exact number.

        Uses REALISTIC_SCRIPT, not SHORT_SCRIPT: a short script is dominated
        by fixed per-chunk overhead (reference-audio padding, leading/
        trailing silence), not by the per-word speaking rate this test
        exists to validate. Measured ratio on a 22-word script was 2.14
        (implied ~37 WPM) from overhead alone, telling us nothing about
        DEFAULT_WPM. A ~200-250 word script amortizes that overhead over
        enough words that the per-word rate dominates instead.
        """
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([REALISTIC_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(REALISTIC_SCRIPT)]),
        )
        actual = sf.info(result.audio_path).duration
        ratio = actual / result.estimated_sec
        self.assertGreater(ratio, 0.7, f"estimate far too long: {ratio:.2f}")
        self.assertLess(ratio, 1.3, f"estimate far too short: {ratio:.2f}")

    def test_kokoro_path_also_renders(self):
        config = AutoConfig(
            target_min_sec=1.0, target_max_sec=100000.0, tts_engine="kokoro"
        )
        result = run(
            "I feel anxious and need to unwind.",
            config=config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )
        self.assertTrue(Path(result.audio_path).is_file())


if __name__ == "__main__":
    unittest.main()
