"""Orchestrate prompt -> script -> music -> rendered meditation.

This is the only module the Auto-Generate UI tab calls. It never modifies the
audio path: MeditationPipeline.generate() is invoked exactly as the manual tab
invokes it.
"""

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.background_picker import pick_background
from core.script_gen.duration import estimate_duration_sec
from core.script_gen.engine import ScriptEngine, build_engine
from core.script_gen.generator import draft
from core.script_gen.judge import repair, review
from core.script_gen.linter import Violation, check, fatal_violations
from core.script_gen.rules import (
    build_generator_system_prompt,
    build_judge_system_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_GENERATOR = "ollama:llama3.2:3b"
DEFAULT_JUDGE = "ollama:llama3.2:3b"


class ScriptGenerationError(RuntimeError):
    """Raised when a script cannot be made safe or well-formed in budget."""


@dataclass
class AutoConfig:
    """Everything the auto path needs that is not the prompt itself."""

    content_type: str = "meditation"
    tts_engine: str = "f5"
    target_min_sec: float = 300.0
    target_max_sec: float = 420.0
    max_repairs: int = 2
    max_tokens: int = 4096
    # Zero-arg callable returning [(label, path), ...]; None uses the real
    # scan_backgrounds. Injected in tests.
    background_scan: object | None = None
    recent_backgrounds: list[str] = field(default_factory=list)
    failure_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "moodscape_failures"
    )

    @classmethod
    def from_env(cls, **overrides) -> "AutoConfig":
        """Build a config from environment variables, with explicit overrides."""
        values = {
            "target_min_sec": float(os.environ.get("MOODSCAPE_TARGET_MIN_SEC", 300.0)),
            "target_max_sec": float(os.environ.get("MOODSCAPE_TARGET_MAX_SEC", 420.0)),
            "max_repairs": int(os.environ.get("MOODSCAPE_SCRIPT_MAX_REPAIRS", 2)),
        }
        values.update(overrides)
        return cls(**values)


@dataclass
class ScriptOutcome:
    """Result of the two-pass script generation."""

    script: str
    draft_script: str
    changelog: str
    violations: list[Violation]
    estimated_sec: float
    repairs_used: int


@dataclass
class AutoResult:
    """Result of a full auto-generation run."""

    audio_path: str
    script_path: str
    meta_path: str
    script: str
    changelog: str
    background: str
    violations: list[Violation]
    estimated_sec: float


def _violation_dicts(violations: list[Violation]) -> list[dict]:
    return [
        {"code": v.code, "severity": v.severity, "message": v.message}
        for v in violations
    ]


def _write_failure_artifacts(
    config: AutoConfig, prompt: str, script: str, violations: list[Violation]
) -> Path:
    """Persist a failed script so it can be read and debugged."""
    config.failure_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = config.failure_dir / f"failed-{stamp}"
    base.with_suffix(".script.txt").write_text(script, encoding="utf-8")
    base.with_suffix(".meta.json").write_text(
        json.dumps(
            {"prompt": prompt, "violations": _violation_dicts(violations)}, indent=2
        ),
        encoding="utf-8",
    )
    return base


def generate_script(
    prompt: str,
    *,
    generator_engine: ScriptEngine,
    judge_engine: ScriptEngine,
    config: AutoConfig,
    progress_cb=None,
) -> ScriptOutcome:
    """Run generator -> judge -> lint -> bounded repair.

    Raises:
        ScriptGenerationError: If fatal violations survive the repair budget.
    """
    gen_system = build_generator_system_prompt(
        config.tts_engine,
        config.content_type,
        config.target_min_sec,
        config.target_max_sec,
    )
    judge_system = build_judge_system_prompt(
        config.tts_engine,
        config.content_type,
        config.target_min_sec,
        config.target_max_sec,
    )

    if progress_cb:
        progress_cb(0.05, "Writing draft script")
    draft_script = draft(
        generator_engine, prompt, gen_system, max_tokens=config.max_tokens
    )

    if progress_cb:
        progress_cb(0.12, "Reviewing script")
    script, changelog = review(
        judge_engine, draft_script, judge_system, max_tokens=config.max_tokens
    )

    repairs_used = 0
    while True:
        estimated_sec = estimate_duration_sec(
            script,
            engine=config.tts_engine,
            content_type=config.content_type,
        )
        violations = check(
            script,
            estimated_sec=estimated_sec,
            target_min_sec=config.target_min_sec,
            target_max_sec=config.target_max_sec,
        )
        fatal = fatal_violations(violations)

        if not fatal:
            return ScriptOutcome(
                script=script,
                draft_script=draft_script,
                changelog=changelog,
                violations=violations,
                estimated_sec=estimated_sec,
                repairs_used=repairs_used,
            )

        if repairs_used >= config.max_repairs:
            codes = ", ".join(sorted({v.code for v in fatal}))
            path = _write_failure_artifacts(config, prompt, script, violations)
            raise ScriptGenerationError(
                f"Script still has fatal problems after {repairs_used} repair "
                f"attempts: {codes}. Script saved to {path}.script.txt for review."
            )

        repairs_used += 1
        if progress_cb:
            progress_cb(0.15, f"Repairing script (attempt {repairs_used})")
        script, repair_log = repair(
            judge_engine, script, fatal, judge_system, max_tokens=config.max_tokens
        )
        changelog = f"{changelog}\n{repair_log}".strip()


def run(
    prompt: str,
    *,
    config: AutoConfig | None = None,
    pipeline=None,
    generator_engine: ScriptEngine | None = None,
    judge_engine: ScriptEngine | None = None,
    rng: random.Random | None = None,
    progress_cb=None,
    **pipeline_kwargs,
) -> AutoResult:
    """Prompt in, finished meditation out.

    Args:
        prompt: The user's natural-language request.
        config: Overrides; defaults come from the environment.
        pipeline: Injected for tests. Defaults to a real MeditationPipeline.
        generator_engine / judge_engine: Injected for tests. Default to the
            engines named by MOODSCAPE_SCRIPT_GENERATOR / _JUDGE.
        rng: Seeded Random for reproducible background selection.
        progress_cb: Called with (fraction, message).
        **pipeline_kwargs: Forwarded verbatim to MeditationPipeline.generate().

    Returns:
        AutoResult with paths to the audio, script, and metadata.
    """
    config = config or AutoConfig.from_env()

    if generator_engine is None:
        generator_engine = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR)
        )
    if judge_engine is None:
        judge_engine = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_JUDGE", DEFAULT_JUDGE)
        )

    outcome = generate_script(
        prompt,
        generator_engine=generator_engine,
        judge_engine=judge_engine,
        config=config,
        progress_cb=progress_cb,
    )

    for violation in outcome.violations:
        logger.warning("script advisory %s: %s", violation.code, violation.message)

    if progress_cb:
        progress_cb(0.20, "Choosing background music")
    background_label, background_path = pick_background(
        scan=config.background_scan,
        exclude=config.recent_backgrounds,
        rng=rng,
    )

    if pipeline is None:
        from core.pipeline import MeditationPipeline
        pipeline = MeditationPipeline()

    audio_path, _status = pipeline.generate(
        script=outcome.script,
        music_prompt="",
        content_type=config.content_type,
        tts_engine=config.tts_engine,
        music_model="upload",
        uploaded_music_path=background_path,
        progress_cb=progress_cb,
        **pipeline_kwargs,
    )

    audio = Path(audio_path)
    script_path = audio.with_suffix(".script.txt")
    meta_path = audio.with_suffix(".meta.json")

    script_path.write_text(outcome.script, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "content_type": config.content_type,
                "tts_engine": config.tts_engine,
                "generator": generator_engine.name,
                "judge": judge_engine.name,
                "background": background_label,
                "background_path": background_path,
                "changelog": outcome.changelog,
                "violations": _violation_dicts(outcome.violations),
                "estimated_sec": outcome.estimated_sec,
                "repairs_used": outcome.repairs_used,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return AutoResult(
        audio_path=str(audio),
        script_path=str(script_path),
        meta_path=str(meta_path),
        script=outcome.script,
        changelog=outcome.changelog,
        background=background_label,
        violations=outcome.violations,
        estimated_sec=outcome.estimated_sec,
    )
