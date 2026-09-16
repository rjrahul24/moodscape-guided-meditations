"""Script generation: prompt -> validated meditation script.

Two LLM passes (generator, then an independent judge that revises) bounded by
a deterministic linter and duration estimator that cost no tokens.
"""

from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    Violation,
    check,
    check_format,
    check_safety,
    fatal_violations,
    format_for_repair,
)

__all__ = [
    "ADVISORY",
    "FATAL",
    "Violation",
    "check",
    "check_format",
    "check_safety",
    "fatal_violations",
    "format_for_repair",
]
