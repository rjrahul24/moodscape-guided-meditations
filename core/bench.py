"""Benchmark candidate model pairings on the script-generation task.

Answers "which model" with evidence rather than argument: fixed prompts, the
same linter, and the scripts themselves to read.
"""

import time
from dataclasses import dataclass

from core.auto_generate import AutoConfig, ScriptGenerationError, generate_script
from core.script_gen.engine import build_engine

# Deliberately spans the real emotional range the app serves, including the
# harder cases (grief, overwhelm) where safety rules matter most.
BENCH_PROMPTS: list[str] = [
    "I'm feeling anxious and my chest is tight.",
    "I can't sleep. My mind won't stop racing.",
    "I'm grieving someone I lost and I don't know what to do with it.",
    "I'm overwhelmed at work and I have ten minutes.",
    "I feel numb and disconnected from everything.",
    "I'm angry and I don't want to be.",
    "I want to feel grounded before a difficult conversation.",
    "I'm exhausted but wired and I need to come down.",
    "I keep worrying about things I can't control.",
    "I just want a few minutes of quiet.",
]


@dataclass
class BenchRow:
    """One (pairing, prompt) result."""

    generator: str
    judge: str
    prompt: str
    passed: bool
    estimated_sec: float
    repairs_used: int
    elapsed_sec: float
    advisories: int
    error: str


def run_bench(
    pairings: list[tuple[str, str]],
    *,
    prompts: list[str] | None = None,
    config: AutoConfig | None = None,
    engine_factory=None,
) -> list[BenchRow]:
    """Run every prompt through every pairing.

    Args:
        pairings: (generator_spec, judge_spec) tuples.
        prompts: Defaults to BENCH_PROMPTS.
        config: Defaults to AutoConfig.from_env().
        engine_factory: Injected for tests; defaults to build_engine.

    Returns:
        One BenchRow per (pairing, prompt).
    """
    prompts = prompts if prompts is not None else BENCH_PROMPTS
    config = config or AutoConfig.from_env()
    factory = engine_factory or build_engine

    rows: list[BenchRow] = []
    for generator_spec, judge_spec in pairings:
        for prompt in prompts:
            started = time.monotonic()
            try:
                outcome = generate_script(
                    prompt,
                    generator_engine=factory(generator_spec),
                    judge_engine=factory(judge_spec),
                    config=config,
                )
                rows.append(
                    BenchRow(
                        generator=generator_spec,
                        judge=judge_spec,
                        prompt=prompt,
                        passed=True,
                        estimated_sec=outcome.estimated_sec,
                        repairs_used=outcome.repairs_used,
                        elapsed_sec=time.monotonic() - started,
                        advisories=len(outcome.violations),
                        error="",
                    )
                )
            except (ScriptGenerationError, RuntimeError) as exc:
                rows.append(
                    BenchRow(
                        generator=generator_spec,
                        judge=judge_spec,
                        prompt=prompt,
                        passed=False,
                        estimated_sec=0.0,
                        repairs_used=config.max_repairs,
                        elapsed_sec=time.monotonic() - started,
                        advisories=0,
                        error=str(exc),
                    )
                )
    return rows


def format_bench_table(rows: list[BenchRow]) -> str:
    """Render results as a markdown table."""
    lines = [
        "| Generator | Judge | Prompt | Pass | Est. min | Repairs | Sec | Advisories |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.generator} | {row.judge} | {row.prompt[:40]} | "
            f"{'yes' if row.passed else 'NO'} | {row.estimated_sec / 60:.1f} | "
            f"{row.repairs_used} | {row.elapsed_sec:.1f} | {row.advisories} |"
        )
    return "\n".join(lines)
