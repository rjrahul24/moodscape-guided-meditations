import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.eval.render_only import load_source_cases

ROOT = Path(__file__).resolve().parent.parent.parent


def _case(root, name, genre="body_scan", script=True, bg="/bed.mp3"):
    d = root / name
    d.mkdir(parents=True)
    (d / "a.meta.json").write_text(json.dumps({"genre": genre, "voice": "Brittney",
                                                "background": "Bed", "background_path": bg,
                                                "estimated_sec": 300}))
    if script:
        (d / "a.script.txt").write_text("Hello. [pause:2s]")
    (d / "a.wav").write_bytes(b"")
    return d


class TestLoadSourceCases(unittest.TestCase):
    def test_reads_cases_and_seed_from_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base" / "eval"
            _case(root, "body_scan-short")
            (root.parent / "results.json").write_text(json.dumps(
                [{"case": "body_scan-short", "genre": "body_scan", "band": "short", "seed": 1600, "ok": True}]))
            cases, skipped = load_source_cases(root)
            self.assertEqual([(c.case, c.seed, c.band, c.voice) for c in cases],
                             [("body_scan-short", 1600, "short", "Brittney")])
            self.assertEqual(skipped, [])

    def test_case_without_script_is_skipped_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base" / "eval"
            _case(root, "x-short", script=False)
            cases, skipped = load_source_cases(root)
            self.assertEqual(cases, [])
            self.assertTrue(any("x-short" in m for m in skipped))

    def test_without_results_json_band_unknown_seed_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base" / "eval"
            _case(root, "y-short")
            cases, _ = load_source_cases(root)
            self.assertEqual((cases[0].seed, cases[0].band), (None, "unknown"))


class _FakePipeline:
    last_stems = None
    last_speech_qa = None

    def generate(self, **kw):
        import soundfile as sf
        import numpy as np
        out = Path(tempfile.mkdtemp()) / "out.wav"
        sf.write(str(out), np.zeros(4800, dtype="float32"), 48000)
        return str(out), "ok"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_genres_ro", ROOT / "scripts" / "eval_genres.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestScriptRenderOnly(unittest.TestCase):
    def test_missing_background_row_and_render_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "base" / "eval"
            bed = tmp / "bed.mp3"
            bed.write_bytes(b"x")
            _case(src, "good-short", bg=str(bed))
            _case(src, "bad-short", bg=str(tmp / "nope.mp3"))
            out = tmp / "new"
            mod = _load_script()
            argv = ["eval_genres.py", "--scripts-from", str(src), "--out", str(out), "--name", "arm"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch("core.pipeline.MeditationPipeline", _FakePipeline):
                mod.main()
            rows = {r["case"]: r for r in json.loads((out / "results.json").read_text())}
            self.assertFalse(rows["bad-short"]["ok"])
            self.assertIn("background missing", rows["bad-short"]["error"])
            self.assertTrue(rows["good-short"]["ok"], rows["good-short"])
            metas = list((out / "arm" / "good-short").glob("*.meta.json"))
            self.assertEqual(len(metas), 1)
            self.assertIn("rendered_from", json.loads(metas[0].read_text()))
            self.assertIn("rendered_from", rows["good-short"])
            self.assertFalse((out / "corpus_snapshot").exists())


class TestScriptNormalPath(unittest.TestCase):
    def test_normal_row_fields_unchanged(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gen = tmp / "gen"
            gen.mkdir()
            audio, script, meta = gen / "a.wav", gen / "a.script.txt", gen / "a.meta.json"
            for f in (audio, script, meta):
                f.write_text("x")
            fake = SimpleNamespace(audio_path=str(audio), script_path=str(script), meta_path=str(meta),
                                   voice="v", angle="ang", estimated_sec=12.34, originality=0.5,
                                   violations=[], script="s", draft_script="")
            corpus = tmp / "live"
            corpus.mkdir()
            out = tmp / "out"
            mod = _load_script()
            argv = ["eval_genres.py", "--genres", "body_scan", "--band", "short",
                    "--configs", "c=w|j", "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(mod, "CORPUS_DIR", corpus), \
                    mock.patch.object(mod, "run", return_value=fake), \
                    mock.patch("core.pipeline.MeditationPipeline", _FakePipeline):
                mod.main()
            row = json.loads((out / "results.json").read_text())[0]
            for key in ("config", "genre", "case", "band", "seed", "ok", "voice", "seconds",
                        "angle", "estimated_sec", "originality", "advisories", "audio"):
                self.assertIn(key, row)
            self.assertTrue(row["ok"])
            self.assertEqual(row["estimated_sec"], 12.3)
            self.assertTrue((out / "c" / "body_scan-0" / "a.wav").is_file())
            self.assertFalse((out / "c" / "body_scan-0" / ".corpus").exists())


if __name__ == "__main__":
    unittest.main()
