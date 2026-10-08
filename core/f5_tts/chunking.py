"""Breath-group chunker sized to F5-TTS's own text budget.

Upstream ``infer_process`` silently re-splits any chunk longer than
``max_chars = len(ref_text_bytes) / ref_dur * (22 - ref_dur) * speed`` and
renders the pieces with a fixed 0.15 s crossfade. We split first, at natural
breath points and under a safety-scaled budget, so upstream never re-splits,
and give each group a punctuation-dependent pause. Pure functions, no model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

SAFETY = 0.9
_MIN_BUDGET = 20

_SENTENCE_END = ".!?…"
_CLAUSE_END = ";:"
_COMMA_END = ",—–"
_CLOSERS = "\"'’”)]"

_GAP_DEFAULTS = (
    ("MOODSCAPE_F5_GAP_SENTENCE", 0.45),
    ("MOODSCAPE_F5_GAP_CLAUSE", 0.35),
    ("MOODSCAPE_F5_GAP_COMMA", 0.20),
)


@dataclass(frozen=True)
class BreathGroup:
    text: str
    gap_after_sec: float | None  # None = last group of the segment


def f5_char_budget(ref_text: str, ref_duration_sec: float, speed: float) -> int:
    """Byte budget mirroring upstream's ``max_chars``, scaled by SAFETY."""
    raw = len(ref_text.encode("utf-8")) / ref_duration_sec * (22 - ref_duration_sec) * speed * SAFETY
    return max(_MIN_BUDGET, int(raw))


def _gap_value(env_name: str, default: float, environ) -> float:
    env = os.environ if environ is None else environ
    try:
        return float(env[env_name])
    except (KeyError, ValueError):
        return default


def _final_punct(text: str) -> str:
    stripped = text.rstrip().rstrip(_CLOSERS).rstrip()
    return stripped[-1] if stripped else ""


def gap_for(text: str, environ=None) -> float:
    """Pause (seconds) after a group, from its final punctuation."""
    ch = _final_punct(text)
    if ch and ch in _SENTENCE_END:
        idx = 0
    elif ch and ch in _CLAUSE_END:
        idx = 1
    elif ch and ch in _COMMA_END:
        idx = 2
    else:
        return 0.0
    name, default = _GAP_DEFAULTS[idx]
    return _gap_value(name, default, environ)


def _ends_unit(word: str) -> bool:
    ch = _final_punct(word)
    return bool(ch) and ch in (_SENTENCE_END + _CLAUSE_END + _COMMA_END)


def _units(text: str) -> list[list[str]]:
    """Split into units (lists of words); each ends at a clause/sentence end.

    Operating on whitespace-split words means a dash only ends a unit when it
    is followed by whitespace (it is then the last char of its word).
    """
    units: list[list[str]] = []
    cur: list[str] = []
    for word in text.split():
        cur.append(word)
        if _ends_unit(word):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)
    return units


def _size(s: str) -> int:
    return len(s.encode("utf-8"))


def breath_groups(text: str, budget: int, environ=None) -> list[BreathGroup]:
    """Greedily pack clause units into groups of at most ``budget`` bytes.

    A unit over budget is split at spaces; a single over-budget word is
    emitted whole (never dropped, never looped on).
    """
    pieces: list[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur:
            pieces.append(cur)
            cur = ""

    for words in _units(text):
        unit = " ".join(words)
        if cur and _size(cur + " " + unit) <= budget:
            cur += " " + unit
            continue
        flush()
        if _size(unit) <= budget:
            cur = unit
            continue
        # Over-budget unit: pack word by word.
        for word in words:
            if cur and _size(cur + " " + word) <= budget:
                cur += " " + word
            else:
                flush()
                cur = word
    flush()

    groups = [BreathGroup(p, gap_for(p, environ)) for p in pieces]
    if groups:
        groups[-1] = BreathGroup(groups[-1].text, None)
    return groups
