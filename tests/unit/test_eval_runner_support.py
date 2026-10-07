import json
import tempfile
import unittest
from pathlib import Path

from core.eval.runner_support import (
    fresh_job_corpus,
    prepare_snapshot,
    snapshot_corpus,
    stale_case_dirs,
    write_results,
)


def _corpus(root: Path) -> Path:
    (root / "scripts").mkdir(parents=True)
    (root / "index.json").write_text("[1]")
    (root / "scripts" / "a.txt").write_text("hello")
    return root


class TestSnapshot(unittest.TestCase):
    def test_copies_live_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            live = _corpus(tmp / "live")
            (tmp / "out").mkdir()
            snap = prepare_snapshot(tmp / "out", live)
            self.assertEqual((snap / "scripts" / "a.txt").read_text(), "hello")
            self.assertEqual(snap, tmp / "out" / "corpus_snapshot")

    def test_missing_live_corpus_gives_empty_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            snap = snapshot_corpus(tmp / "nope", tmp / "snap")
            self.assertTrue(snap.is_dir())
            self.assertEqual(list(snap.iterdir()), [])

    def test_corpus_from_uses_earlier_snapshot_not_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "base").mkdir()
            (tmp / "new").mkdir()
            prepare_snapshot(tmp / "base", _corpus(tmp / "live"))
            (tmp / "live" / "scripts" / "later.txt").write_text("added after baseline")
            snap = prepare_snapshot(tmp / "new", tmp / "live", corpus_from=tmp / "base")
            self.assertTrue((snap / "scripts" / "a.txt").exists())
            self.assertFalse((snap / "scripts" / "later.txt").exists())

    def test_corpus_from_without_snapshot_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                prepare_snapshot(tmp / "out", tmp / "live", corpus_from=tmp / "empty")

    def test_existing_snapshot_is_reused_without_corpus_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "out").mkdir()
            prepare_snapshot(tmp / "out", _corpus(tmp / "live"))
            (tmp / "live" / "scripts" / "later.txt").write_text("x")
            snap = prepare_snapshot(tmp / "out", tmp / "live")
            self.assertFalse((snap / "scripts" / "later.txt").exists())


class TestJobCorpus(unittest.TestCase):
    def test_each_job_gets_a_fresh_copy_and_snapshot_stays_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            snap = _corpus(tmp / "snap")
            job = tmp / "cfg" / "case"
            job.mkdir(parents=True)
            first = fresh_job_corpus(snap, job)
            (first / "scripts" / "appended.txt").write_text("job wrote this")
            second = fresh_job_corpus(snap, job)
            self.assertEqual(first, second)
            self.assertFalse((second / "scripts" / "appended.txt").exists())
            self.assertFalse((snap / "scripts" / "appended.txt").exists())


class TestHygiene(unittest.TestCase):
    def test_stale_case_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertEqual(stale_case_dirs(tmp / "none"), [])
            (tmp / "cfg" / "b").mkdir(parents=True)
            (tmp / "cfg" / "a").mkdir()
            (tmp / "cfg" / "file.txt").write_text("x")
            self.assertEqual(stale_case_dirs(tmp / "cfg"), ["a", "b"])

    def test_write_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_results(Path(tmp), [{"ok": True}], "| md |\n")
            self.assertEqual(json.loads((Path(tmp) / "results.json").read_text()), [{"ok": True}])
            self.assertEqual((Path(tmp) / "summary.md").read_text(), "| md |\n")


if __name__ == "__main__":
    unittest.main()
