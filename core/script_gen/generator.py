"""Pass 1: turn a natural-language prompt into a draft script."""

import re

from core.script_gen.engine import ScriptEngine

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_wrapper(text: str) -> str:
    """Remove a surrounding markdown code fence and outer whitespace.

    Models wrap output in fences despite instructions not to; the fence would
    otherwise be read aloud by the TTS engine.
    """
    match = _FENCE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def draft(
    engine: ScriptEngine,
    prompt: str,
    system: str,
    *,
    max_tokens: int = 4096,
) -> str:
    """Generate a draft script from the user's natural-language prompt."""
    user = (
        "Write a complete script for this request:\n\n"
        f"{prompt}\n\n"
        "Output only the script."
    )
    return strip_wrapper(engine.complete(system, user, max_tokens=max_tokens))
