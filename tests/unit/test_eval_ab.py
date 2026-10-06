# tests/unit/test_eval_ab.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.eval.ab import build_pairs, cut_excerpts, find_case_audio, level_match, score, sign_test_p
from core.eval.mix_metrics import loudness

SR = 8000


def _noise(seconds: float, amp: float, seed: int) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _make_run(root: Path, cases: dict[str, float]) -> Path:
    for case, amp in cases.items():
        d = root / case
        d.mkdir(parents=True)
        sf.write(d / "med_1.wav", _noise(300, amp, hash(case) % 1000), SR)
        (d / "med_1.meta.json").write_text("{}")
        sf.write(d / "narration_stem.wav", _noise(1, amp, 1), SR)  # must be ignored
    return root


class TestExcerpts(unittest.TestCase):
    def test_three_positions(self):
        parts = cut_excerpts(_noise(300, 0.1, 0), SR, excerpt_sec=75)
        self.assertEqual([p[0] for p in parts], ["start", "middle", "end"])
        self.assertTrue(all(len(p[1]) == 75 * SR and not p[2] for p in parts))

    def test_short_file_excerpt_truncated(self):
        parts = cut_excerpts(_noise(60, 0.1, 0), SR, excerpt_sec=75)
        self.assertTrue(all(p[2] for p in parts))
        self.assertTrue(all(0 < len(p[1]) <= 60 * SR for p in parts))

    def test_level_match(self):
        matched = level_match(_noise(20, 0.3, 0), SR, target_lufs=-20.0)
        self.assertAlmostEqual(loudness(matched, SR)["integrated_lufs_dual_mono"], -20.0, delta=0.1)


class TestPairs(unittest.TestCase):
    def test_find_case_audio_ignores_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp) / "a", {"body_scan-short": 0.1})
            self.assertEqual(find_case_audio(run)["body_scan-short"].name, "med_1.wav")

    def test_missing_case_raises_with_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.1})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.1})
            with self.assertRaisesRegex(ValueError, "y-short"):
                build_pairs(a, b, Path(tmp) / "ab")

    def test_build_pairs_is_blind_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.2})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.3, "y-short": 0.05})
            n = build_pairs(a, b, Path(tmp) / "ab1", seed=7, excerpt_sec=30)
            build_pairs(a, b, Path(tmp) / "ab2", seed=7, excerpt_sec=30)
            self.assertEqual(n, 6)
            key1 = json.loads((Path(tmp) / "ab1" / "key.json").read_text())
            key2 = json.loads((Path(tmp) / "ab2" / "key.json").read_text())
            self.assertEqual(key1, key2)
            self.assertEqual({k["A"] for k in key1["pairs"].values()} | {k["B"] for k in key1["pairs"].values()},
                             {"baseline", "challenger"})
            with open(Path(tmp) / "ab1" / "sheet.csv") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, ["pair", "choice", "reason"])  # no arm/case leakage


class TestScore(unittest.TestCase):
    def test_sign_test_thresholds(self):
        self.assertLess(sign_test_p(17, 24), 0.05)
        self.assertGreater(sign_test_p(16, 24), 0.05)

    def _scored(self, choices: list[str]) -> dict:
        tmp = Path(tempfile.mkdtemp())
        pairs = {f"{i:02d}": {"A": "challenger", "B": "baseline", "case": "c", "position": "start"}
                 for i in range(len(choices))}
        (tmp / "key.json").write_text(json.dumps({"pairs": pairs}))
        with open(tmp / "sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "choice", "reason"])
            for i, c in enumerate(choices):
                w.writerow([f"{i:02d}", c, ""])
        return score(tmp)

    def test_ship(self):
        result = self._scored(["A"] * 17 + ["B"] * 7)
        self.assertEqual(result["verdict"], "SHIP")
        self.assertEqual(result["challenger_wins"], 17)

    def test_ties_excluded_and_inconclusive(self):
        result = self._scored(["A"] * 10 + ["B"] * 8 + ["same"] * 6)
        self.assertEqual(result["ties"], 6)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")

    def test_no_ship_when_baseline_clearly_wins(self):
        self.assertEqual(self._scored(["B"] * 18 + ["A"] * 6)["verdict"], "NO-SHIP")


if __name__ == "__main__":
    unittest.main()
