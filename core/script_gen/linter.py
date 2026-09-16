"""Deterministic validation of generated scripts.

Three check families: format (this module's check_format), safety hard-blocks
(check_safety), and duration (check). Violations are returned as structured
objects so they can be fed back to the judge as a *targeted* repair
instruction rather than a vague "try again".

Severity matters. Treating every violation as fatal would make a weaker model
unusable; treating none as fatal would let a safety failure reach audio.
"""

import re
from dataclasses import dataclass

FATAL = "fatal"
ADVISORY = "advisory"

# Bounds for [pause:Xs]. Below 0.5s the marker is pointless; above 60s it is
# almost certainly a model error rather than an intentional silence.
MIN_PAUSE_SEC = 0.5
MAX_PAUSE_SEC = 60.0

# The guide's ideal band is 8-20 words. Allow headroom before complaining.
MAX_SENTENCE_WORDS = 25

KNOWN_BARE_TAGS = {"breath", "inhale", "exhale"}

_PAUSE_TAG = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]")
_ANY_TAG = re.compile(r"\[([^\]]*)\]")
_MARKDOWN = re.compile(r"(^\s{0,3}#{1,6}\s)|(\*\*)|(^\s*[-*+]\s+)|(^\s*\d+[.)]\s+)", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
)
_ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Violation:
    """One problem found in a script.

    Args:
        code: Stable machine-readable identifier, e.g. "MARKER_MALFORMED".
        severity: FATAL (do not render) or ADVISORY (render, log a warning).
        message: Human- and model-readable description used for repair.
        span: Optional (start, end) character offsets into the script.
    """

    code: str
    severity: str
    message: str
    span: tuple[int, int] | None = None


def _strip_tags(script: str) -> str:
    """Remove all bracket tags so prose checks do not trip over them."""
    return _ANY_TAG.sub(" ", script)


def check_format(script: str) -> list[Violation]:
    """Validate the engine contract: tags, markup, and sentence shape."""
    violations: list[Violation] = []

    for match in _ANY_TAG.finditer(script):
        inner = match.group(1).strip()
        span = (match.start(), match.end())
        if inner in KNOWN_BARE_TAGS:
            continue
        if inner.startswith("pause:"):
            if not _PAUSE_TAG.fullmatch(match.group(0)):
                violations.append(
                    Violation(
                        code="MARKER_MALFORMED",
                        severity=FATAL,
                        message=(
                            f"Malformed pause marker {match.group(0)!r}. "
                            "Use exactly [pause:Xs], e.g. [pause:4s] or [pause:2.5s]."
                        ),
                        span=span,
                    )
                )
                continue
            seconds = float(_PAUSE_TAG.fullmatch(match.group(0)).group(1))
            if not (MIN_PAUSE_SEC <= seconds <= MAX_PAUSE_SEC):
                violations.append(
                    Violation(
                        code="PAUSE_OUT_OF_RANGE",
                        severity=FATAL,
                        message=(
                            f"Pause of {seconds}s is outside the allowed range "
                            f"{MIN_PAUSE_SEC}-{MAX_PAUSE_SEC}s."
                        ),
                        span=span,
                    )
                )
            continue
        violations.append(
            Violation(
                code="UNKNOWN_TAG",
                severity=FATAL,
                message=(
                    f"Unknown tag {match.group(0)!r}. Only [pause:Xs], "
                    "[breath], [inhale] and [exhale] are supported."
                ),
                span=span,
            )
        )

    if _MARKDOWN.search(script):
        violations.append(
            Violation(
                code="MARKDOWN_PRESENT",
                severity=FATAL,
                message=(
                    "Script contains markdown (headings, bold, or bullets). "
                    "Output plain prose only — the TTS engine reads markup aloud."
                ),
            )
        )

    if _EMOJI.search(script):
        violations.append(
            Violation(
                code="EMOJI_PRESENT",
                severity=FATAL,
                message="Script contains emoji. Use plain text only.",
            )
        )

    prose = _strip_tags(script)

    for match in _ALL_CAPS.finditer(prose):
        violations.append(
            Violation(
                code="ALL_CAPS",
                severity=ADVISORY,
                message=(
                    f"{match.group(0)!r} is in capitals. Use lower case — "
                    "capitals change the engine's pronunciation."
                ),
            )
        )

    for sentence in _SENTENCE_SPLIT.split(prose):
        words = sentence.split()
        if len(words) > MAX_SENTENCE_WORDS:
            violations.append(
                Violation(
                    code="SENTENCE_TOO_LONG",
                    severity=ADVISORY,
                    message=(
                        f"Sentence is {len(words)} words; keep sentences to "
                        "8-20 words so the pacing stays calm."
                    ),
                )
            )

    return violations
