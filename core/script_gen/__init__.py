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

__all__ = [
    "ADVISORY",
    "FATAL",
    "Violation",
    "check",
    "check_format",
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
]
