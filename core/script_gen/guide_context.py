"""Values the prompting guides quote, taken from the code that enforces them.

Guides are templates: {{NAME}} is replaced at load time, so a guide can never
again tell the writer a chunk limit, speed or speaking rate the renderer does
not use.
"""

import re

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")


def guide_context(engine: str, content_type: str) -> dict[str, str]:
    from core.content_profiles import get_profile
    from core.script_gen import linter
    from core.script_gen.duration import DEFAULT_WPM
    from core.script_gen.slop import slop_guidance

    profile = get_profile(content_type)
    ctx = {
        "SPEED": f"{profile['speed']:.2f}",
        "PARA_PAUSE": f"{profile[f'{engine}_paragraph_pause_sec']:g}",
        "WPM": f"{DEFAULT_WPM[engine]:.0f}",
        "PAUSE_MIN": f"{linter.MIN_PAUSE_SEC:g}",
        "PAUSE_MAX": f"{linter.MAX_PAUSE_SEC:g}",
        "MAX_HOLD": str(linter.MAX_BREATH_HOLD_SEC),
        "SENT_IDEAL_MIN": str(linter.IDEAL_SENTENCE_MIN_WORDS),
        "SENT_IDEAL_MAX": str(linter.IDEAL_SENTENCE_MAX_WORDS),
        "SENT_MAX": str(linter.MAX_SENTENCE_WORDS),
        "SLOP_GUIDANCE": slop_guidance(20),
    }
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import MAX_CHUNK_TOKENS
        ctx["CHUNK_TOKENS"] = str(MAX_CHUNK_TOKENS)
    else:
        from core.f5_tts.preprocessor import MAX_CHUNK_CHARS
        ctx["CHUNK_CHARS"] = str(MAX_CHUNK_CHARS)
    return ctx


def render_guide(text: str, context: dict[str, str]) -> str:
    def _sub(match: re.Match) -> str:
        name = match.group(1)
        if name not in context:
            raise KeyError(f"Unknown guide placeholder {{{{{name}}}}}; known: {sorted(context)}")
        return context[name]

    rendered = _PLACEHOLDER.sub(_sub, text)
    if "{{" in rendered:
        raise ValueError("Unrendered '{{' left in guide after substitution.")
    return rendered
