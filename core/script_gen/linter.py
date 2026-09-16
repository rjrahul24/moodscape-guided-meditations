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


# --- Safety hard-blocks -------------------------------------------------
#
# These are the mental-health guardrails. Each entry is
# (code, compiled pattern, message). Patterns are matched case-insensitively
# against the tag-stripped prose.

MAX_BREATH_HOLD_SEC = 7

_SAFETY_RULES: list[tuple[str, re.Pattern, str]] = [
    (
        "CLINICAL_CLAIM",
        re.compile(
            r"\b(cure[sd]?|heal[s]?|treat[s]?|diagnos\w+)\b[^.?!]{0,40}"
            r"\b(anxiety|depression|trauma|ptsd|insomnia|illness|condition)\b"
            r"|\breplaces?\s+(therapy|medication|treatment)\b",
            re.IGNORECASE,
        ),
        "Clinical claim. This is not treatment and must not present itself as "
        "therapy or a cure. Describe the practice, never a medical outcome.",
    ),
    (
        "OUTCOME_PROMISE",
        re.compile(
            r"\byou\s+will\s+(be|feel)\s+(completely|totally|entirely|fully)\b"
            r"|\bthis\s+will\s+(eliminate|remove|erase|banish)\b"
            r"|\bguarantee[sd]?\b",
            re.IGNORECASE,
        ),
        "Promises an outcome. A listener who does not feel that way will read "
        "it as their own failure. Use invitational phrasing instead.",
    ),
    (
        "INVALIDATING",
        re.compile(
            r"\b(don'?t|do not|stop)\s+feel\w*\b"
            r"|\bthere'?s\s+nothing\s+wrong\s+with\s+you\b"
            r"|\byou\s+shouldn'?t\s+(feel|be)\b",
            re.IGNORECASE,
        ),
        "Invalidating instruction. Telling a distressed listener not to feel "
        "something dismisses their experience. Acknowledge, do not override.",
    ),
    (
        "DISSOCIATION",
        re.compile(
            r"\bleave\s+your\s+body\b"
            r"|\bfloat\s+away\s+from\s+your\s*self\b"
            r"|\byou\s+are\s+not\s+your\s+body\b"
            r"|\bdetach\s+from\s+your\s+body\b",
            re.IGNORECASE,
        ),
        "Dissociation-adjacent imagery, which is contraindicated for trauma "
        "survivors. Keep the listener grounded in the body and the room.",
    ),
]

_BREATH_HOLD = re.compile(
    r"\bhold\s+(?:your\s+)?breath\b[^.?!]{0,30}?(\d+)", re.IGNORECASE
)


def _normalize_apostrophes(text: str) -> str:
    """Fold typographic apostrophes to ASCII before pattern matching.

    LLMs routinely emit U+2019 (') rather than "'". Without this, "Don't
    feel anxious" slips past the INVALIDATING pattern — a safety hard-block
    silently defeated by a curly quote.
    """
    return text.replace("'", "'").replace("ʼ", "'")


def check_safety(script: str) -> list[Violation]:
    """Apply mental-health content hard-blocks. All findings are FATAL."""
    prose = _normalize_apostrophes(_strip_tags(script))
    violations: list[Violation] = []

    for code, pattern, message in _SAFETY_RULES:
        for match in pattern.finditer(prose):
            violations.append(
                Violation(
                    code=code,
                    severity=FATAL,
                    message=f"{message} Found: {match.group(0)!r}.",
                    span=(match.start(), match.end()),
                )
            )

    for match in _BREATH_HOLD.finditer(prose):
        seconds = int(match.group(1))
        if seconds > MAX_BREATH_HOLD_SEC:
            violations.append(
                Violation(
                    code="BREATH_HOLD",
                    severity=FATAL,
                    message=(
                        f"Instructs a {seconds}-second breath hold. Holds over "
                        f"{MAX_BREATH_HOLD_SEC}s are a real risk for listeners "
                        "with panic disorder or asthma. Shorten it or remove it."
                    ),
                    span=(match.start(), match.end()),
                )
            )

    return violations


def check(
    script: str,
    *,
    estimated_sec: float | None = None,
    target_min_sec: float = 300.0,
    target_max_sec: float = 420.0,
) -> list[Violation]:
    """Run every check family. Duration is skipped when no estimate is given."""
    violations = check_format(script) + check_safety(script)

    if estimated_sec is not None and not (
        target_min_sec <= estimated_sec <= target_max_sec
    ):
        violations.append(
            Violation(
                code="DURATION_OUT_OF_WINDOW",
                severity=ADVISORY,
                message=(
                    f"Estimated runtime is {estimated_sec:.0f}s, outside the "
                    f"target {target_min_sec:.0f}-{target_max_sec:.0f}s. "
                    "Add or remove content and pauses to land in the window."
                ),
            )
        )

    return violations


def fatal_violations(violations: list[Violation]) -> list[Violation]:
    """Return only the violations that must block a render."""
    return [v for v in violations if v.severity == FATAL]


def format_for_repair(violations: list[Violation]) -> str:
    """Render violations as a numbered instruction block for the judge.

    Targeted repair beats a vague "try again" — the judge is told exactly what
    is wrong and why.
    """
    if not violations:
        return ""
    lines = [
        f"{i}. [{v.code}/{v.severity}] {v.message}"
        for i, v in enumerate(violations, start=1)
    ]
    return "\n".join(lines)
