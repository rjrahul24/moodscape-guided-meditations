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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.background_picker import pick_background
from core.genres import load_pack, pick_angle
from core.originality import add_to_corpus, assess, avoid_terms, load_corpus, recent_angles
from core.script_gen.duration import estimate_duration_sec, log_estimate_accuracy
from core.script_gen.engine import ScriptEngine, build_engine
from core.script_gen.generator import draft
from core.script_gen.judge import repair, review
from core.script_gen.linter import (
    Violation,
    check,
    check_banned_phrases,
    check_originality,
    fatal_violations,
)
from core.script_gen.planner import plan
from core.script_gen.rules import (
    build_generator_system_prompt,
    build_judge_system_prompt,
    build_planner_system_prompt,
)

logger = logging.getLogger(__name__)

# Benchmarked 2026-09-17 against EQ-Bench Creative Writing v3 and Judgemark v4.
# qwen3.8:27b has the best slop score of any model that fits a 32 GB M1 Max
# (1.7) and plans as well as it writes, so one load covers both stages.
# gemma4:31b scores 72.31 on Judgemark -- the best local judge by five points
# -- which is a different skill from writing well. See the spec's section 6.
DEFAULT_PLANNER = "ollama:qwen3.8:27b"
DEFAULT_GENERATOR = "ollama:qwen3.8:27b"
DEFAULT_JUDGE = "ollama:gemma4:31b"

# The three UI length options, in seconds.
DURATION_BANDS: dict[str, tuple[float, float]] = {
    "short": (180.0, 360.0),
    "medium": (360.0, 600.0),
    "long": (600.0, 900.0),
}

# Cap on how many recently-used background paths AutoConfig.recent_backgrounds
# remembers. Bounded so a config reused across a long batch doesn't grow an
# ever-larger exclude list.
RECENT_BACKGROUNDS_LIMIT = 5


class ScriptGenerationError(RuntimeError):
    """Raised when a script cannot be made safe or well-formed in budget."""


def _parse_env_float(name: str, default: float) -> float:
    """Parse an env var as float, or raise ScriptGenerationError naming it."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ScriptGenerationError(
            f"Environment variable {name}={raw!r} is not a valid number."
        ) from exc


def _parse_env_int(name: str, default: int) -> int:
    """Parse an env var as int, or raise ScriptGenerationError naming it."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ScriptGenerationError(
            f"Environment variable {name}={raw!r} is not a valid integer."
        ) from exc


def _parse_env_bool(name: str, default: bool) -> bool:
    """Parse an env var as a flag. '0', 'false', 'no' and '' are false.

    Mirrors _parse_env_float/_parse_env_int: a typo'd value is reported, not
    silently treated as the default, because a silently-disabled originality
    check is exactly the failure nobody notices.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"0", "false", "no", ""}:
        return False
    if lowered in {"1", "true", "yes"}:
        return True
    raise ScriptGenerationError(
        f"Environment variable {name}={raw!r} is not a valid boolean. "
        "Use 1 or 0."
    )


@dataclass
class AutoConfig:
    """Everything the auto path needs that is not the prompt itself."""

    content_type: str = "meditation"
    # Genre partitions the originality corpus. The prompt-driven path leaves
    # this empty, which is its own partition -- prompt runs share no genre
    # with each other and should only be compared among themselves.
    genre: str = ""
    angle: str = ""
    # Tags from the genre pack, forwarded to pick_background's prefer_tags so
    # a running meditation does not land on a sleep drone. Empty for the
    # prompt-driven path, which has no pack to draw tags from.
    music_tags: tuple[str, ...] = ()
    originality: bool = True
    # None uses core.originality.CORPUS_DIR. Injected in tests.
    corpus_dir: Path | None = None
    tts_engine: str = "f5"
    target_min_sec: float = 300.0
    target_max_sec: float = 420.0
    max_repairs: int = 2
    max_tokens: int = 4096
    # Zero-arg callable returning [(label, path), ...]; None uses the real
    # scan_backgrounds. Injected in tests.
    background_scan: object | None = None
    # Paths of recently-used background tracks to avoid repeating; run()
    # appends the chosen background_path here after a successful render and
    # caps the list at RECENT_BACKGROUNDS_LIMIT.
    #
    # IMPORTANT: this only has any effect when the CALLER REUSES ONE
    # AutoConfig instance across multiple run() calls — e.g. a batch script
    # or scripts/generate.py looping over prompts with one config object.
    # The Gradio Auto-Generate tab (core/auto_tab.py) builds a brand-new
    # AutoConfig on every button click, so recent_backgrounds is always
    # empty there by construction — the exclude-recent behaviour is
    # currently inert in the UI, not just untested.
    recent_backgrounds: list[str] = field(default_factory=list)
    failure_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "moodscape_failures"
    )

    @classmethod
    def from_env(cls, **overrides) -> "AutoConfig":
        """Build a config from environment variables, with explicit overrides.

        Raises:
            ScriptGenerationError: If an environment variable is set but is
                not parseable as the expected type. A typo'd config is
                reported, not silently replaced with the default.
        """
        values = {
            "target_min_sec": _parse_env_float(
                "MOODSCAPE_TARGET_MIN_SEC", 300.0
            ),
            "target_max_sec": _parse_env_float(
                "MOODSCAPE_TARGET_MAX_SEC", 420.0
            ),
            "max_repairs": _parse_env_int("MOODSCAPE_SCRIPT_MAX_REPAIRS", 2),
            "originality": _parse_env_bool("MOODSCAPE_ORIGINALITY", True),
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_genre(cls, pack, *, band: str = "medium", **overrides) -> "AutoConfig":
        """Build a config from a genre pack and a duration band.

        Copies the pack's deterministic fields explicitly. run() never infers
        content_type at run time -- the caller owns it, so a UI that lets the
        user override the pack's choice and a script that does not both behave
        predictably.
        """
        if band not in DURATION_BANDS:
            raise ScriptGenerationError(
                f"Unknown duration band {band!r}. Expected one of "
                f"{sorted(DURATION_BANDS)}."
            )
        target_min_sec, target_max_sec = DURATION_BANDS[band]
        values = {
            "genre": pack.slug,
            "content_type": pack.content_type,
            "music_tags": pack.music_tags,
            "target_min_sec": target_min_sec,
            "target_max_sec": target_max_sec,
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
    originality_cosine: float = 0.0
    originality_shared_span: int = 0
    brief: str = ""


@dataclass
class AutoResult:
    """Result of a full auto-generation run."""

    audio_path: str
    script_path: str
    meta_path: str
    script: str
    changelog: str
    background: str
    background_path: str
    violations: list[Violation]
    estimated_sec: float
    originality: float = 0.0
    brief: str = ""
    genre: str = ""
    angle: str = ""


def _violation_dicts(violations: list[Violation]) -> list[dict]:
    return [
        {"code": v.code, "severity": v.severity, "message": v.message}
        for v in violations
    ]


def _write_failure_artifacts(
    config: AutoConfig,
    prompt: str,
    script: str,
    violations: list[Violation],
    *,
    draft_script: str,
    changelog: str,
) -> Path:
    """Persist a failed run's full trajectory so it can be read and debugged.

    Writes the still-fatal script, the original draft (before any judge
    revision or repair), and a meta.json carrying the changelog of every
    review/repair attempt plus the surviving violations — the draft and the
    changelog are what let a reader see what was tried, not just the final
    broken state.
    """
    config.failure_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond precision so two failures in the same wall-clock second
    # (routine in a fast unit-test suite, and possible in production under
    # concurrent auto-generation) don't silently collide and overwrite one
    # another's artifacts.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = config.failure_dir / f"failed-{stamp}"
    base.with_suffix(".script.txt").write_text(script, encoding="utf-8")
    base.with_suffix(".draft.txt").write_text(draft_script, encoding="utf-8")
    base.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "changelog": changelog,
                "violations": _violation_dicts(violations),
            },
            indent=2,
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
    planner_engine: ScriptEngine | None = None,
    pack=None,
    angle=None,
    steer: str = "",
    progress_cb=None,
) -> ScriptOutcome:
    """Run [planner ->] generator -> judge -> lint -> bounded repair.

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

    brief = ""
    if pack is not None and angle is not None and planner_engine is not None:
        if progress_cb:
            progress_cb(0.02, f"Planning a {pack.label} session")
        planner_system = build_planner_system_prompt(
            config.content_type, config.target_min_sec, config.target_max_sec
        )
        avoid = avoid_terms(
            load_corpus(genre=config.genre, limit=5, corpus_dir=config.corpus_dir),
            [e.text for e in load_corpus(limit=500, corpus_dir=config.corpus_dir)],
        )
        brief = plan(
            planner_engine,
            pack,
            angle,
            system=planner_system,
            target_min_sec=config.target_min_sec,
            target_max_sec=config.target_max_sec,
            avoid=avoid,
            steer=steer,
        )
        prompt = brief
        # Only unload if the writer is a different model. The default config
        # uses one model for both stages, where unloading would pay a full
        # 18 GB reload to save nothing.
        if planner_engine is not generator_engine:
            planner_engine.unload()

    if progress_cb:
        progress_cb(0.05, "Writing draft script")
    draft_script = draft(
        generator_engine, prompt, gen_system, max_tokens=config.max_tokens
    )
    generator_engine.unload()

    if progress_cb:
        progress_cb(0.12, "Reviewing script")
    script, changelog = review(
        judge_engine, draft_script, judge_system, max_tokens=config.max_tokens
    )

    repairs_used = 0
    try:
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
            if pack is not None:
                violations = violations + check_banned_phrases(script, pack.banned)

            report = None
            if config.originality:
                # IDF over EVERY genre (that is what learns generic meditation
                # vocabulary); comparison within this genre only (that is
                # where collisions happen). See core/originality.py.
                same_genre = load_corpus(
                    genre=config.genre, limit=100, corpus_dir=config.corpus_dir
                )
                all_texts = [
                    entry.text
                    for entry in load_corpus(limit=500, corpus_dir=config.corpus_dir)
                ]
                report = assess(
                    script, compare_against=same_genre, idf_texts=all_texts
                )
                violations = violations + check_originality(report)

            fatal = fatal_violations(violations)

            if not fatal:
                return ScriptOutcome(
                    script=script,
                    draft_script=draft_script,
                    changelog=changelog,
                    violations=violations,
                    estimated_sec=estimated_sec,
                    repairs_used=repairs_used,
                    originality_cosine=report.max_cosine if report else 0.0,
                    originality_shared_span=report.shared_span if report else 0,
                    brief=brief,
                )

            if repairs_used >= config.max_repairs:
                codes = ", ".join(sorted({v.code for v in fatal}))
                path = _write_failure_artifacts(
                    config,
                    prompt,
                    script,
                    violations,
                    draft_script=draft_script,
                    changelog=changelog,
                )
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
    finally:
        judge_engine.unload()


def _measure_actual_duration_sec(audio_path: str) -> float | None:
    """Read the rendered file's actual duration in seconds, or None.

    Used to close the calibration loop: log_estimate_accuracy() compares
    this against the pre-render estimate so DEFAULT_WPM can eventually be
    tuned from real data instead of guesswork (see docs/auto_generation/
    README.md :: "Calibrating DEFAULT_WPM").

    A calibration nicety must never fail a job that already produced audio
    -- the render succeeded, and that is what the user cares about -- so any
    read failure (unreadable file, or in tests a stub pipeline that writes a
    non-audio placeholder) is logged at DEBUG and swallowed rather than
    raised.
    """
    try:
        import soundfile as sf

        return float(sf.info(audio_path).duration)
    except Exception:
        logger.debug(
            "could not read actual duration from %s", audio_path, exc_info=True
        )
        return None


def run(
    prompt: str = "",
    *,
    genre: str | None = None,
    steer: str = "",
    config: AutoConfig | None = None,
    pipeline=None,
    planner_engine: ScriptEngine | None = None,
    generator_engine: ScriptEngine | None = None,
    judge_engine: ScriptEngine | None = None,
    rng: random.Random | None = None,
    progress_cb=None,
    **pipeline_kwargs,
) -> AutoResult:
    """Prompt in, finished meditation out.

    Args:
        prompt: The user's natural-language request. Ignored on the genre
            path, where the planner's brief becomes the writer's prompt.
        genre: A genre pack slug. When set, a planner stage turns the pack
            and a chosen angle into a creative brief before writing. config
            never has content_type inferred from this at run time -- the
            caller owns content_type, via AutoConfig.from_genre() or the UI.
        steer: Free-text listener steering, forwarded to the planner.
        config: Overrides; defaults come from the environment.
        pipeline: Injected for tests. Defaults to a real MeditationPipeline.
        planner_engine / generator_engine / judge_engine: Injected for
            tests. Default to the engines named by MOODSCAPE_SCRIPT_PLANNER /
            _GENERATOR / _JUDGE.
        rng: Seeded Random for reproducible background/angle selection.
        progress_cb: Called with (fraction, message).
        **pipeline_kwargs: Forwarded verbatim to MeditationPipeline.generate().

    Returns:
        AutoResult with paths to the audio, script, and metadata.
    """
    config = config or AutoConfig.from_env()

    pack, angle = None, None
    if genre:
        pack = load_pack(genre)
        angle = pick_angle(
            pack,
            recent=recent_angles(genre, limit=3, corpus_dir=config.corpus_dir),
            rng=rng,
        )
        config.angle = angle.name

    if generator_engine is None:
        generator_engine = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR)
        )

    # Reuse the generator's engine for the planner when both specs name the
    # same model -- the default config does, and build_engine() does no
    # caching of its own, so two separate calls would hand generate_script()
    # two distinct objects even though they load identical weights. That
    # would make the `planner_engine is not generator_engine` unload check
    # always true, defeating the whole point: the planner would be unloaded
    # only to have the writer immediately reload the same 18 GB model.
    if genre and planner_engine is None:
        planner_spec = os.environ.get("MOODSCAPE_SCRIPT_PLANNER", DEFAULT_PLANNER)
        generator_spec = os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR)
        planner_engine = (
            generator_engine
            if planner_spec == generator_spec
            else build_engine(planner_spec)
        )

    if judge_engine is None:
        judge_engine = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_JUDGE", DEFAULT_JUDGE)
        )

    # Fail in seconds rather than five minutes in.
    for engine in (planner_engine, generator_engine, judge_engine):
        if engine is not None:
            engine.preflight()

    outcome = generate_script(
        prompt,
        pack=pack,
        angle=angle,
        steer=steer,
        planner_engine=planner_engine,
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
        prefer_tags=config.music_tags,
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

    if config.originality:
        # Recorded only after a successful render: a script that never became
        # audio should not constrain future runs.
        add_to_corpus(
            outcome.script,
            genre=config.genre,
            angle=config.angle,
            corpus_dir=config.corpus_dir,
        )

    # Record this background as recently used so a caller that reuses this
    # AutoConfig across multiple run() calls (batch/scripted use) gets
    # variety. See the recent_backgrounds field docstring for the caveat
    # that a fresh-config-per-call caller (the Gradio tab) never benefits.
    config.recent_backgrounds.append(background_path)
    del config.recent_backgrounds[:-RECENT_BACKGROUNDS_LIMIT]

    actual_sec = _measure_actual_duration_sec(audio_path)
    estimate_ratio = None
    if actual_sec is not None:
        log_estimate_accuracy(outcome.estimated_sec, actual_sec, config.tts_engine)
        if outcome.estimated_sec:
            estimate_ratio = actual_sec / outcome.estimated_sec

    audio = Path(audio_path)
    script_path = audio.with_suffix(".script.txt")
    meta_path = audio.with_suffix(".meta.json")

    script_path.write_text(outcome.script, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "content_type": config.content_type,
                "genre": config.genre,
                "angle": config.angle,
                "tts_engine": config.tts_engine,
                "generator": generator_engine.name,
                "judge": judge_engine.name,
                "background": background_label,
                "background_path": background_path,
                "changelog": outcome.changelog,
                "violations": _violation_dicts(outcome.violations),
                "estimated_sec": outcome.estimated_sec,
                "actual_sec": actual_sec,
                "estimate_ratio": estimate_ratio,
                "repairs_used": outcome.repairs_used,
                "originality_cosine": outcome.originality_cosine,
                "originality_shared_span": outcome.originality_shared_span,
                "brief": outcome.brief,
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
        background_path=background_path,
        violations=outcome.violations,
        estimated_sec=outcome.estimated_sec,
        originality=outcome.originality_cosine,
        brief=outcome.brief,
        genre=config.genre,
        angle=config.angle,
    )
