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

_SCRIPT_BLOCK = re.compile(r"<script>\s*(.*?)\s*</script>", re.DOTALL | re.IGNORECASE)
_CHANGELOG_BLOCK = re.compile(
    r"<changelog>\s*(.*?)\s*</changelog>", re.DOTALL | re.IGNORECASE
)


def parse_judge_response(raw: str) -> tuple[str, str]:
    """Split the judge's response into (script, changelog).

    Degrades gracefully: a model that ignores the delimiters still yields a
    usable script rather than failing the run.
    """
    script_match = _SCRIPT_BLOCK.search(raw)
    changelog_match = _CHANGELOG_BLOCK.search(raw)

    script = script_match.group(1) if script_match else raw
    changelog = changelog_match.group(1).strip() if changelog_match else ""

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
