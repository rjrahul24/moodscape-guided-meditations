"""Pass 2: an independent model reviews the draft and returns a revision.

The judge revises rather than scores. A score does not help a fire-and-forget
pipeline; a corrected script does.

The same module handles targeted repair, which is the same operation with the
linter's findings supplied as the instruction.
"""

import re

from core.script_gen.engine import ScriptEngine
from core.script_gen.generator import strip_wrapper
from core.script_gen.linter import Violation, format_for_repair

# \b after the tag name matters: without it, "<scriptfoo>" would match. The
# open tag tolerates attributes (models occasionally add lang="en" etc.); the
# closing tag tolerates whitespace ("</ script >").
_SCRIPT_BLOCK = re.compile(
    r"<script\b[^>]*>\s*(.*?)\s*</\s*script\s*>", re.DOTALL | re.IGNORECASE
)
_CHANGELOG_BLOCK = re.compile(
    r"<changelog\b[^>]*>\s*(.*?)\s*</\s*changelog\s*>", re.DOTALL | re.IGNORECASE
)
# Catches a leftover script/changelog tag that didn't pair up into a full
# block above (e.g. an opening tag with no matching close). [^>\n]* is
# deliberately bounded to a single line so this can never swallow real script
# prose sitting on the next line; >? tolerates a tag that never closes at all.
_STRAY_TAG = re.compile(r"</?\s*(?:script|changelog)\b[^>\n]*>?", re.IGNORECASE)


def parse_judge_response(raw: str) -> tuple[str, str]:
    """Split the judge's response into (script, changelog).

    Degrades gracefully: a model that ignores the delimiters still yields a
    usable script rather than failing the run. When no <script> block can be
    located, the fallback is the raw text with any complete <changelog> block
    and any stray script/changelog tags removed first — otherwise a malformed
    tag could leak the judge's changelog straight into audio that gets read
    aloud.
    """
    script_match = _SCRIPT_BLOCK.search(raw)
    changelog_match = _CHANGELOG_BLOCK.search(raw)
    changelog = changelog_match.group(1).strip() if changelog_match else ""

    if script_match:
        script = script_match.group(1)
    else:
        script = _CHANGELOG_BLOCK.sub("", raw)
        script = _STRAY_TAG.sub("", script)

    return strip_wrapper(script), changelog


def review(
    engine: ScriptEngine,
    draft_script: str,
    system: str,
    *,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Review a draft and return (revised_script, changelog)."""
    user = (
        "Review this draft script and return a corrected version.\n\n"
        "<draft>\n"
        f"{draft_script}\n"
        "</draft>"
    )
    return parse_judge_response(engine.complete(system, user, max_tokens=max_tokens))


def repair(
    engine: ScriptEngine,
    script: str,
    violations: list[Violation],
    system: str,
    *,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Fix specific, named violations. Returns (repaired_script, changelog)."""
    user = (
        "An automated checker found the following problems in this script. "
        "Fix every one of them and return the corrected script. Change nothing "
        "else.\n\n"
        "<problems>\n"
        f"{format_for_repair(violations)}\n"
        "</problems>\n\n"
        "<script>\n"
        f"{script}\n"
        "</script>"
    )
    return parse_judge_response(engine.complete(system, user, max_tokens=max_tokens))
