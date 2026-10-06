"""Evaluation harness: script, speech and mix metrics plus blind A/B tooling.

Every module imports without torch or Gradio; heavy dependencies are imported
inside the functions that need them.
"""

from core.eval.mix_metrics import loudness, mix_metrics, true_peak_dbtp
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
]
