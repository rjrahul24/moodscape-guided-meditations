import unittest
from unittest.mock import patch

from core.eval.report import compute_run_metrics, summarize


class TestComputeRunMetrics(unittest.TestCase):
    def test_group_failure_is_recorded_not_raised(self):
        with patch("core.eval.report.mix_metrics", side_effect=RuntimeError("bad wav")):
            m = compute_run_metrics(
                script="Rest here. [pause:2s]", draft=None, engine="f5",
                content_type="meditation", audio_path="/nonexistent.wav",
                voice_stem_path=None, estimated_sec=10.0, transcriber=None,
            )
        self.assertEqual(m["mix"], {"error": "RuntimeError: bad wav"})
        self.assertIn("words", m["script"])
        self.assertEqual(m["speech"], {"skipped": "no transcriber"})


class TestSummarize(unittest.TestCase):
    def test_medians_per_config_and_pass_rate(self):
        rows = [
            {"config": "base", "ok": True, "metrics": {"speech_input": "voice_stem",
                "script": {"slop_density": 1.0, "edit_ratio": 0.2},
                "speech": {"cer": 0.02, "internal_gaps": 1},
                "mix": {"integrated_lufs_dual_mono": -13.0, "duration_ratio": 1.1}}},
            {"config": "base", "ok": True, "metrics": {"speech_input": "voice_stem",
                "script": {"slop_density": 3.0, "edit_ratio": 0.4},
                "speech": {"cer": 0.04, "internal_gaps": 3},
                "mix": {"integrated_lufs_dual_mono": -13.0, "duration_ratio": 0.9}}},
            {"config": "base", "ok": False, "error": "boom"},
        ]
        md = summarize(rows)
        self.assertIn("| base | 2/3 | 2/2 |", md)
        self.assertIn("2.00 (n=2)", md)   # slop median
        self.assertIn("0.030 (n=2)", md)  # CER median
        self.assertIn("0.10 (n=2)", md)   # median |1 - duration ratio|

    def test_speech_columns_use_only_stem_rows(self):
        def row(cer, source):
            return {"config": "c", "ok": True, "metrics": {
                "speech_input": source, "speech": {"cer": cer, "internal_gaps": 0},
                "script": {"slop_density": 1.0}}}
        md = summarize([row(0.01, "voice_stem"), row(0.50, "mix"), row(0.03, "voice_stem")])
        self.assertIn("| c | 3/3 | 2/3 |", md)
        self.assertIn("0.020 (n=2)", md)  # median of stem rows only
        self.assertIn("1.00 (n=3)", md)   # script metric counts every passed row
        self.assertNotIn("0.500", md)

    def test_missing_metrics_render_as_dash(self):
        md = summarize([{"config": "x", "ok": True, "metrics": {}}])
        self.assertIn("| x | 1/1 | 0/1 | – |", md)


if __name__ == "__main__":
    unittest.main()
