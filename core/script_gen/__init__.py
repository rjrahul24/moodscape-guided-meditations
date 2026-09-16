"""Script generation: prompt -> validated meditation script.

Two LLM passes (generator, then an independent judge that revises) bounded by
a deterministic linter and duration estimator that cost no tokens.
"""

from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    Violation,
    check_format,
)

__all__ = [
    "ADVISORY",
    "FATAL",
    "Violation",
    "check_format",
]
