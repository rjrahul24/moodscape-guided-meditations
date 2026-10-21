"""Delivery direction: the voice-quality instruction prefixed to every chunk.

Eleven v4 has no speed/style settings, so pacing and tone come from an inline
bracketed direction. It must describe VOICE QUALITY only — never a sound
(breaths, sighs, laughs): no model renders those naturally (user rule).
"""

from __future__ import annotations

import os
import re

DEFAULT_DIRECTION = "[soft, slow, calm voice]"
MAX_DIRECTION_CHARS = 80

_SOUND_WORDS = (
    "breath", "breaths", "breathe", "breathes", "breathing", "breathy",
    "inhale", "inhales", "inhaling", "exhale", "exhales", "exhaling",
    "sigh", "sighs", "sighing", "gasp", "gasps", "gasping",
    "laugh", "laughs", "laughing", "chuckle", "chuckles", "giggle", "giggles",
    "yawn", "yawns", "yawning", "hum", "hums", "humming",
    "cough", "coughs", "sniff", "sniffs", "gulp", "gulps", "swallow", "swallows",
    "whimper", "whimpers", "sob", "sobs", "sobbing", "cry", "cries", "crying",
)
_SOUND_RE = re.compile(r"\b(" + "|".join(_SOUND_WORDS) + r")\b", re.IGNORECASE)


def sanitize_direction(text: str | None) -> str:
    """Normalize to a single ``[...]`` group; reject sound words and overlong text."""
    inner = re.sub(r"[\[\]]", " ", text or "")
    inner = re.sub(r"\s+", " ", inner).strip(" ,")
    if not inner:
        return DEFAULT_DIRECTION
    match = _SOUND_RE.search(inner)
    if match:
        raise ValueError(
            "Delivery direction must describe voice quality, not sounds "
            f"(found {match.group(0)!r}). Example: {DEFAULT_DIRECTION}"
        )
    result = f"[{inner}]"
    if len(result) > MAX_DIRECTION_CHARS:
        raise ValueError(f"Delivery direction is {len(result)} chars; max is {MAX_DIRECTION_CHARS}.")
    return result


def default_direction() -> str:
    return sanitize_direction(os.environ.get("MOODSCAPE_ELEVENLABS_DIRECTION") or DEFAULT_DIRECTION)
