"""LLM content check: a backstop for sensitive wording the regexes miss.

The deterministic SENSITIVE_TOPIC rule in linter.py blocks the known classes
(inner-child imagery, medical disclaimers, ...). This pass asks the already
loaded judge model to flag *novel* phrasings in the same spirit. It is a
classifier, not a writer: it returns quotes, and the repair loop in
core/auto_generate.py decides what to do with them.

Policy (spec Part K): findings are FATAL-for-repair within the shared repair
budget, then ADVISORY once the budget is spent, so a model false positive can
never stop production. Every finding is verified against the script: a quote
the model invented is dropped.
"""

import json
import logging
import re

from core.script_gen.engine import ScriptEngine
from core.script_gen.linter import Violation, _strip_tags

logger = logging.getLogger(__name__)

CONTENT_CODE = "SENSITIVE_CONTENT"

# Pack slugs containing any of these may talk about loss, death and grief.
_GRIEF_MARKERS = ("grief", "loss")

CONTENT_SYSTEM_PROMPT = (
    "You review guided-meditation and sleep-story scripts for a general "
    "wellness audience. Find every passage that mentions or pictures any of "
    "the following:\n"
    "1. illness, disease, symptoms, a medical or physical condition, a "
    "diagnosis, medication, surgery, hospitals or doctors;\n"
    "2. pregnancy;\n"
    "3. a child, children, a baby, or the listener's younger self or inner "
    "child;\n"
    "4. death, dying or bereavement (unless the request says the genre may "
    "mention loss);\n"
    "5. self-harm or suicide;\n"
    "6. weight, body shape or dieting;\n"
    "7. anything else a general wellness audience could find alienating.\n\n"
    "Ordinary body-awareness language (breath, heart, shoulders, a hand on "
    "the chest) is fine and must not be flagged.\n\n"
    "Reply with ONLY a JSON array and nothing else. Each item is "
    '{"quote": "<exact words copied from the script>", "reason": "<short '
    'reason>"}. Copy each quote verbatim from the script, a few words up to '
    "one sentence. If nothing needs flagging, reply with [] exactly."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WS = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalise(text: str) -> str:
    return _WS.sub(" ", text.translate(_APOSTROPHES)).strip().lower()


def _is_grief_genre(genre: str) -> bool:
    slug = (genre or "").lower()
    return any(marker in slug for marker in _GRIEF_MARKERS)


def _build_user(prose: str, genre: str) -> str:
    lines = [f"genre: {genre}"]
    if _is_grief_genre(genre):
        lines.append(
            "This genre is about grief: the script may mention loss, death "
            "and bereavement, so do not flag those. Flag everything else."
        )
    lines += ["", "<script>", prose.strip(), "</script>"]
    return "\n".join(lines)


def _parse_array(raw: str) -> list | None:
    """Return the first JSON array in ``raw``, or None if there is none."""
    text = _THINK_BLOCK.sub("", raw or "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, list):
            return value
    return None


def _verified(items: list, prose_norm: str) -> list[dict]:
    kept: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        quote = item.get("quote")
        if not isinstance(quote, str):
            continue
        quote = quote.strip().strip("\"'“”‘’").strip()
        if not quote:
            continue
        if _normalise(quote) not in prose_norm:
            logger.info("Content check: dropped finding not in script: %r", quote)
            continue
        reason = item.get("reason")
        kept.append({"quote": quote, "reason": reason.strip() if isinstance(reason, str) else ""})
    return kept


def content_findings(
    engine: ScriptEngine,
    script: str,
    *,
    genre: str,
    max_tokens: int = 800,
    sampling=None,
) -> list[dict] | None:
    """Ask ``engine`` for sensitive passages in ``script``.

    Returns:
        ``[{"quote", "reason"}]`` with every quote verified present in the
        tag-stripped script (case- and whitespace-insensitive); ``[]`` when
        clean; ``None`` when the check could not run (two malformed replies
        or an engine error). ``None`` is logged here; the caller skips.
    """
    prose = _strip_tags(script)
    prose_norm = _normalise(prose)
    user = _build_user(prose, genre)

    for attempt in (1, 2):
        try:
            raw = engine.complete(
                CONTENT_SYSTEM_PROMPT, user, max_tokens=max_tokens, sampling=sampling
            )
        except Exception as exc:  # noqa: BLE001 -- the check must never fail a job
            logger.warning("Content check skipped: engine error: %s", exc)
            return None
        items = _parse_array(raw)
        if items is not None:
            return _verified(items, prose_norm)
        logger.info("Content check: reply %d was not a JSON array", attempt)

    logger.warning("Content check skipped: two malformed (non-JSON-array) replies")
    return None


def verify_findings(findings: list[dict], script: str) -> list[dict]:
    """Keep only findings whose quote is present in ``script`` (tags stripped)."""
    return _verified(findings, _normalise(_strip_tags(script)))


def findings_to_violations(findings: list[dict], *, severity: str) -> list[Violation]:
    """One SENSITIVE_CONTENT violation per finding, at ``severity``."""
    return [
        Violation(
            code=CONTENT_CODE,
            severity=severity,
            message=(
                f"Sensitive wording: \"{f['quote']}\""
                + (f" ({f['reason']})" if f.get("reason") else "")
                + ". Rewrite this passage so it no longer mentions or pictures "
                "this; keep the meaning and pacing of the session."
            ),
        )
        for f in findings
    ]
