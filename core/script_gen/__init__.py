"""Script generation: prompt -> validated meditation script.

Two LLM passes (generator, then an independent judge that revises) bounded by
a deterministic linter and duration estimator that cost no tokens.
"""

from core.script_gen.linter import (
    ADVISORY,
    ADVISORY_COSINE,
    FATAL,
    FATAL_COSINE,
    Violation,
    check,
    check_banned_phrases,
    check_format,
    check_originality,
    check_safety,
    fatal_violations,
    format_for_repair,
)
from core.script_gen.duration import (
    BREATH_SEC,
    DEFAULT_WPM,
    estimate_duration_sec,
    log_estimate_accuracy,
)
from core.script_gen.rules import (
    build_generator_system_prompt,
    build_judge_system_prompt,
    load_guide,
    load_safety_rules,
)
from core.script_gen.engine import (
    FakeScriptEngine,
    ScriptEngine,
    build_engine,
    parse_engine_spec,
)
from core.script_gen.generator import draft, strip_wrapper
from core.script_gen.judge import parse_judge_response, repair, review

__all__ = [
    "ADVISORY",
    "ADVISORY_COSINE",
    "FATAL",
    "FATAL_COSINE",
    "Violation",
    "check",
    "check_banned_phrases",
    "check_format",
    "check_originality",
    "check_safety",
    "fatal_violations",
    "format_for_repair",
    "BREATH_SEC",
    "DEFAULT_WPM",
    "estimate_duration_sec",
    "log_estimate_accuracy",
    "build_generator_system_prompt",
    "build_judge_system_prompt",
    "load_guide",
    "load_safety_rules",
    "FakeScriptEngine",
    "ScriptEngine",
    "build_engine",
    "parse_engine_spec",
    "draft",
    "strip_wrapper",
    "parse_judge_response",
    "repair",
    "review",
]
