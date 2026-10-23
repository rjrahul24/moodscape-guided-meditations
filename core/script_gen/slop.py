"""Meditation-writing slop lexicon and density metric.

Lives in script_gen; shared by the linter (ADVISORY checks) and the eval
harness (core.eval.slop re-exports it).

Seeded from the Oct 2026 research report (artifact d). R1 refines the list
from MoodScape's own output; keep it short and targeted (Antislop: plain
banning degrades past ~2,000 patterns).
"""

import re
from dataclasses import dataclass

SLOP_HARD: tuple[str, ...] = (
    "tapestry", "symphony of", "dance of", "embark", "delve", "testament to",
    "realm", "ethereal", "kaleidoscope", "whisper of the wind", "kiss of the sun",
    "wash over you", "melt away", "every cell of your body", "healing light",
    "golden light flowing", "pure white light", "inner sanctuary", "oasis of calm",
    "universe holds you", "vibrant energy", "deeper and deeper and deeper",
)

SLOP_RATE_LIMITED: tuple[str, ...] = (
    "gently", "softly", "slowly", "simply", "allow yourself", "invite you to",
    "notice how", "let go", "release", "sink", "drift", "serene", "tranquil",
    "peaceful", "journey", "embrace", "present moment", "stillness within",
)

_SOFT_ADJ = (
    "warm|soft|gentle|calm|quiet|peaceful|serene|tranquil|still|safe|light|"
    "heavy|deep|slow|smooth|tender|loose|easy"
)

SLOP_PATTERNS: dict[str, str] = {
    "more_and_more": r"\bmore and more\b",
    "not_x_its_y": r"\bit's not [^.,;]{1,40}, it's\b",
    "triple_soft_adjectives": (
        rf"\b(?:{_SOFT_ADJ}), (?:{_SOFT_ADJ}),? (?:and )?(?:{_SOFT_ADJ})\b"
    ),
}

_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class SlopHit:
    term: str
    tier: str  # "hard" | "rate" | "pattern"


def _prose(text: str) -> str:
    # Lazy import: linter.check_slop imports this module lazily too, but keeping
    # this lazy means neither module needs the other at import time.
    from core.script_gen.linter import _normalize_apostrophes, strip_tags

    return _normalize_apostrophes(strip_tags(text)).lower()


def _phrase_regex(phrase: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(phrase) + r"\b")


def slop_hits(text: str) -> list[SlopHit]:
    """Every lexicon hit in the tag-stripped, apostrophe-normalised prose."""
    prose = _prose(text)
    hits: list[SlopHit] = []
    for tier, phrases in (("hard", SLOP_HARD), ("rate", SLOP_RATE_LIMITED)):
        for phrase in phrases:
            count = len(_phrase_regex(phrase).findall(prose))
            hits.extend(SlopHit(phrase, tier) for _ in range(count))
    for name, pattern in SLOP_PATTERNS.items():
        count = len(re.findall(pattern, prose))
        hits.extend(SlopHit(name, "pattern") for _ in range(count))
    return hits


def slop_density(text: str) -> float:
    """Lexicon hits per 150 words of prose (0.0 for empty text)."""
    words = len(_WORD.findall(_prose(text)))
    if words == 0:
        return 0.0
    return len(slop_hits(text)) * 150.0 / words


def slop_guidance(top_n: int = 20) -> str:
    """One-line writer guidance listing the worst offenders (hard tier first)."""
    items = list(SLOP_HARD) + [p for p in SLOP_RATE_LIMITED if p not in SLOP_HARD]
    return (
        "Prefer concrete, specific nouns and verbs. Use gently, softly or slowly "
        "at most once per 150 words. Avoid: " + ", ".join(items[:top_n])
    )
