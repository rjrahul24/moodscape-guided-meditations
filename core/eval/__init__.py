"""Evaluation harness: script, speech and mix metrics plus blind A/B tooling.

Every module imports without torch or Gradio; heavy dependencies are imported
inside the functions that need them.
"""

from core.eval.ab import build_pairs, cut_excerpts, find_case_audio, level_match, score, sign_test_p
from core.eval.mix_metrics import loudness, mix_metrics, true_peak_dbtp
from core.eval.report import compute_run_metrics, summarize
from core.eval.script_metrics import script_metrics
from core.eval.slop import SlopHit, slop_density, slop_hits
from core.eval.speech_metrics import (
    Transcript,
    Transcriber,
    WhisperTranscriber,
    Word,
    cer,
    internal_gaps,
    normalize_text,
    rate_spread,
    speech_metrics,
    wer,
)
from core.eval.testset import DEFAULT_TESTSET, TestCase, load_testset

__all__ = [
    "SlopHit",
    "Transcriber",
    "Transcript",
    "WhisperTranscriber",
    "Word",
    "cer",
    "internal_gaps",
    "normalize_text",
    "rate_spread",
    "loudness",
    "mix_metrics",
    "script_metrics",
    "slop_density",
    "slop_hits",
    "speech_metrics",
    "true_peak_dbtp",
    "wer",
    "DEFAULT_TESTSET",
    "TestCase",
    "load_testset",
    "compute_run_metrics",
    "summarize",
    "build_pairs",
    "cut_excerpts",
    "find_case_audio",
    "level_match",
    "score",
    "sign_test_p",
]
