# core/eval/script_metrics.py
"""Deterministic text metrics for a generated script."""

import difflib
import re
from collections import Counter

from core.eval.slop import slop_density, slop_hits
from core.script_gen.linter import ADVISORY, FATAL, _normalize_apostrophes, _strip_tags

_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _segments(script: str, engine: str, content_type: str) -> list[dict]:
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import prepare_segments
    else:  # f5, chatterbox, qwen3 all share the F5 script grammar
        from core.f5_tts.preprocessor import prepare_segments
    return prepare_segments(script, content_type=content_type)


def _estimate(script: str, engine: str, content_type: str) -> float:
    from core.script_gen.duration import estimate_duration_sec
    duration_engine = "kokoro" if engine == "kokoro" else "f5"
    return estimate_duration_sec(script, engine=duration_engine, content_type=content_type)


def script_metrics(
    script: str,
    *,
    draft: str | None = None,
    engine: str = "f5",
    content_type: str = "meditation",
) -> dict:
    """Counts, silence budget, lint codes, slop and draft-vs-final edit ratio."""
    from core.script_gen.linter import check

    prose = _normalize_apostrophes(_strip_tags(script))
    sentences = [s for s in _SENTENCE_SPLIT.split(prose.strip()) if _WORD.search(s)]
    sentence_words = [len(_WORD.findall(s)) for s in sentences]

    segments = _segments(script, engine, content_type)
    pause_seconds = float(sum(s["duration_sec"] for s in segments if s["type"] == "pause"))
    breath_count = sum(1 for s in segments if s["type"] == "breath")
    estimated_sec = float(_estimate(script, engine, content_type))

    violations = check(script)
    hits = slop_hits(script)

    edit_ratio = None
    if draft is not None:
        matcher = difflib.SequenceMatcher(None, draft.split(), script.split())
        edit_ratio = round(1.0 - matcher.ratio(), 4)

    return {
        "words": len(_WORD.findall(prose)),
        "sentences": len(sentences),
        "mean_sentence_words": (sum(sentence_words) / len(sentence_words)) if sentence_words else 0.0,
        "max_sentence_words": max(sentence_words, default=0),
        "question_count": prose.count("?"),
        "pause_seconds": pause_seconds,
        "breath_count": breath_count,
        "estimated_sec": estimated_sec,
        "pause_ratio_est": (pause_seconds / estimated_sec) if estimated_sec else 0.0,
        "lint_fatal": [v.code for v in violations if v.severity == FATAL],
        "lint_advisory": [v.code for v in violations if v.severity == ADVISORY],
        "slop_density": round(slop_density(script), 3),
        "slop_top": [term for term, _n in Counter(h.term for h in hits).most_common(5)],
        "edit_ratio": edit_ratio,
    }
