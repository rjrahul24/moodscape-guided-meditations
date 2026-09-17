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
# By the time check_format() runs on a judge's output, parse_judge_response()
# has already stripped the judge's own <script>/<changelog> protocol tags, so
# any surviving <...> is genuinely stray markup (e.g. SSML like
# <break time="2s"/> or <emphasis>) that the TTS engine would speak aloud.
_ANGLE_TAG = re.compile(r"</?\s*[A-Za-z][^>\n]*/?>")


@dataclass(frozen=True)
class Violation:
    """One problem found in a script.

    Args:
        code: Stable machine-readable identifier, e.g. "MARKER_MALFORMED".
        severity: FATAL (do not render) or ADVISORY (render, log a warning).
        message: Human- and model-readable description used for repair.
        span: Optional (start, end) character offsets into the script.
            check_safety() always sets this to None: its offsets are computed
            against tag-stripped, apostrophe-normalized prose, not the
            original script, so they would not point at the right place in
            the source text. Nothing currently reads span, but a future
            reader should not be misled into treating a safety violation's
            span as a script offset.
    """

    code: str
    severity: str
    message: str
    span: tuple[int, int] | None = None


def _strip_tags(script: str) -> str:
    """Remove all bracket tags so prose checks do not trip over them."""
    return _ANY_TAG.sub(" ", script)


def check_format(script: str, engine: str | None = None) -> list[Violation]:
    """Validate the engine contract: tags, markup, and sentence shape.

    Args:
        engine: When "kokoro", also runs the Kokoro-specific chunk-length
            backstop (see `_check_kokoro_chunk_length`). Any other value,
            including the default None, runs exactly the engine-agnostic
            checks below — unchanged from before this parameter existed.
    """
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

    for match in _ANGLE_TAG.finditer(script):
        violations.append(
            Violation(
                code="ANGLE_TAG",
                severity=FATAL,
                message=(
                    f"Angle-bracket markup {match.group(0)!r} found. Only the "
                    "square-bracket tags [pause:Xs], [breath], [inhale] and "
                    "[exhale] are supported — SSML or HTML-style tags are not "
                    "stripped before synthesis and would be read aloud."
                ),
                span=(match.start(), match.end()),
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

    if engine == "kokoro":
        violations.extend(_check_kokoro_chunk_length(script))

    return violations


def _check_kokoro_chunk_length(script: str) -> list[Violation]:
    """Backstop: flag any chunk Kokoro's own chunker would actually produce
    that exceeds its MAX_CHUNK_TOKENS sweet spot.

    core.kokoro_tts.preprocessor.merge_sentences_to_chunks() merges
    consecutive sentences up to MAX_CHUNK_TOKENS (150 tokens, ~115 words),
    flushing the chunk before it would exceed that limit. Because of that
    flush-before-exceed behaviour, a multi-sentence chunk can never end up
    over MAX_CHUNK_TOKENS — the only way a produced chunk exceeds it is for
    a single sentence to already be over ~115 words on its own, and any
    sentence that long has already tripped the far cheaper SENTENCE_TOO_LONG
    check above (its threshold is 25 words, roughly a third of the size).

    In other words: SENTENCE_TOO_LONG is the stricter gate in every case
    this function can reach. This check exists as a backstop for a script
    that would somehow slip past it, not as an independent detector — do
    not expect it to fire on its own in practice.
    """
    from core.kokoro_tts.preprocessor import (
        MAX_CHUNK_TOKENS,
        estimate_tokens,
        merge_sentences_to_chunks,
        parse_script,
        split_into_sentences,
    )

    violations: list[Violation] = []
    for segment in parse_script(script):
        if segment["type"] != "speech":
            continue
        sentences = split_into_sentences(segment["text"])
        for chunk in merge_sentences_to_chunks(sentences):
            tokens = estimate_tokens(chunk)
            if tokens > MAX_CHUNK_TOKENS:
                violations.append(
                    Violation(
                        code="CHUNK_TOO_LONG",
                        severity=ADVISORY,
                        message=(
                            f"Kokoro chunk is ~{tokens} tokens, above the "
                            f"{MAX_CHUNK_TOKENS}-token limit. Kokoro rushes "
                            "chunks above ~150 tokens, flattening prosody "
                            "and pacing."
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

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}

_SAFETY_RULES: list[tuple[str, re.Pattern, str]] = [
    (
        "CLINICAL_CLAIM",
        re.compile(
            r"\b(?:cure[sd]?|curing|heal(?:s|ed|ing)?|treat(?:s|ed|ing)?|diagnos\w+)\s+(?:your|my|his|her|their)\b\s+(?:\w+\s+){0,2}?\b(?:anxiety|depression|trauma|ptsd|insomnia|illness|condition)\b"
            r"|\b(?:cure[sd]?|curing|heal(?:s|ed|ing)?|treat(?:s|ed|ing)?)\s+\b(?:anxiety|depression|trauma|ptsd|insomnia|illness|condition)\b"
            r"|\breplaces?\s+(?:therapy|medication|treatment)\b",
            re.IGNORECASE,
        ),
        "Clinical claim. This is not treatment and must not present itself as "
        "therapy or a cure. Describe the practice, never a medical outcome.",
    ),
    (
        "OUTCOME_PROMISE",
        re.compile(
            r"\byou\s+will\s+(?:be|feel)\s+(?:completely|totally|entirely|fully)\s+(?!present\b|here\b|aware\b|awake\b|alive\b|grounded\b)"
            r"|\bthis\s+will\s+(?:eliminate|remove|erase|banish)\b"
            r"|\bguarantee[sd]?\b",
            re.IGNORECASE,
        ),
        "Promises an outcome. A listener who does not feel that way will read "
        "it as their own failure. Use invitational phrasing instead.",
    ),
    (
        "INVALIDATING",
        re.compile(
            r"\b(?:don'?t|do not|stop)\b(?:\s+\w+){0,1}\s+feel\w*\b"
            r"|\bthere'?s\s+nothing\s+wrong\s+with\s+you\b"
            r"|\byou\s+shouldn'?t\s+(?:feel|be)\b",
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
            r"|\bdetach\s+from\s+your\s+body\b"
            r"|\b(?:drift|float)\s+(?:up\s+)?(?:out(?:side)?(?:\s+of)?|away\s+from)\s+your\s+body\b",
            re.IGNORECASE,
        ),
        "Dissociation-adjacent imagery, which is contraindicated for trauma "
        "survivors. Keep the listener grounded in the body and the room.",
    ),
]

_BREATH_HOLD = re.compile(
    r"\bhold\s+(?:your\s+)?breath\b[^.?!]{0,80}?(?:(\d+)|(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty))\s+seconds?",
    re.IGNORECASE
)


def _normalize_apostrophes(text: str) -> str:
    """Fold typographic apostrophes to ASCII before pattern matching.

    LLMs routinely emit U+2019 (RIGHT SINGLE QUOTATION MARK) rather than
    U+0027 (APOSTROPHE). Without this, "Don\u2019t feel anxious" slips past
    the INVALIDATING pattern — a safety hard-block silently defeated by a
    curly quote.

    IMPORTANT: every replacement below MUST use an explicit \\uXXXX escape,
    never a literal curly character typed into this file. A previous version
    of this function used a literal right single quote as BOTH the search and
    replacement character (i.e. a no-op "'" -> "'"), because an editor/tool
    in the save path silently normalized the literal curly character to
    ASCII on the way to disk. That turned this whole function into a no-op
    and disabled the INVALIDATING hard-block for the common case (LLMs emit
    U+2019 by default). Explicit escapes cannot be silently re-normalized.
    """
    return (
        text.replace("\u2019", "'")  # right single quotation mark
        .replace("\u2018", "'")  # left single quotation mark
        .replace("\u02bc", "'")  # modifier letter apostrophe
        .replace("\u00b4", "'")  # acute accent
        .replace("\u0060", "'")  # grave accent
    )


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
                    # span=None: offsets here are into the stripped/normalized
                    # `prose`, not the original script — see Violation.span.
                    span=None,
                )
            )

    for match in _BREATH_HOLD.finditer(prose):
        # Group 1 is digit number, Group 2 is word number
        if match.group(1):
            seconds = int(match.group(1))
        elif match.group(2):
            word_num = match.group(2).lower()
            seconds = _NUMBER_WORDS.get(word_num)
            if seconds is None:
                continue
        else:
            continue

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
                    # span=None: see Violation.span docstring.
                    span=None,
                )
            )

    return violations


def check(
    script: str,
    *,
    engine: str | None = None,
    estimated_sec: float | None = None,
    target_min_sec: float = 300.0,
    target_max_sec: float = 420.0,
) -> list[Violation]:
    """Run every check family. Duration is skipped when no estimate is given.

    Args:
        engine: Forwarded to check_format() — "kokoro" enables the
            Kokoro-specific chunk-length backstop. Default None preserves
            prior behaviour exactly.
    """
    violations = check_format(script, engine=engine) + check_safety(script)

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
