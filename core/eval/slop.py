"""Re-export: the slop lexicon lives in core.script_gen.slop (shared with the linter)."""

from core.script_gen.slop import (  # noqa: F401
    SLOP_HARD, SLOP_PATTERNS, SLOP_RATE_LIMITED, SlopHit, slop_density, slop_hits,
)
