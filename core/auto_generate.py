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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from core.background_picker import pick_background
from core.genres import load_pack, pick_angle
from core.originality import add_to_corpus, assess, avoid_terms, load_corpus, recent_angles
from core.script_gen.content_check import (
    content_findings,
    findings_to_violations,
    verify_findings,
)
from core.script_gen.duration import (
    PAUSE_SCALE_BOUNDS,
    estimate_duration_sec,
    log_estimate_accuracy,
    scale_pauses,
    total_pause_sec,
)
from core.script_gen.engine import ScriptEngine, build_engine
from core.script_gen.generator import draft
from core.script_gen.judge import repair, review
from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    STYLE_CODES,
    Violation,
    check,
    check_originality,
    check_pack_rules,
    fatal_violations,
)
from core.script_gen.linter import strip_tags
from core.script_gen.planner import plan
from core.script_gen.rules import (
    build_generator_system_prompt,
    build_judge_system_prompt,
    build_planner_system_prompt,
)
from core.script_gen.sampling import sampling_for

logger = logging.getLogger(__name__)

# Engines whose voice is a reference-clip slug (Kokoro uses `voice` instead).
_VOICE_SLUG_ENGINES = ("f5", "chatterbox", "qwen3")

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

# Duration control (kill-switch MOODSCAPE_DURATION_CONTROL=0). A script whose
# estimate is more than this fraction outside the target window gets one
# targeted length rewrite from the judge; anything still outside the window
# afterwards is nudged in by scaling pauses within PAUSE_SCALE_BOUNDS. Drift
# itself stays ADVISORY: neither step can fail a job.
LENGTH_REPAIR_TOLERANCE = 0.15

# LLM content check (kill-switch MOODSCAPE_CONTENT_CHECK=0). Hard cap on judge
# calls per script generation: normally at most two (one that may trigger a
# repair, one that only reports); the third covers a fallback that restores an
# earlier, never-checked script.
CONTENT_CHECK_MAX_CALLS = 3


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
    speed: float | None = None
    voice: str | None = None
    # F5/Chatterbox/Qwen3 reference voice; set per genre family by from_genre.
    voice_slug: str | None = None
    # ElevenLabs voice id (provider = ElevenLabs); set by from_genre.
    elevenlabs_voice_id: str | None = None
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

        Precedence, low to high: dataclass default -> environment
        (MOODSCAPE_ORIGINALITY, _SCRIPT_MAX_REPAIRS, _TARGET_MIN_SEC/_MAX_SEC)
        -> the pack's deterministic fields -> the caller's explicit overrides.

        This starts from from_env() rather than constructing AutoConfig
        directly, so the genre path -- the only path the UI has -- still
        honours MOODSCAPE_ORIGINALITY=0 and friends instead of silently
        ignoring them. run() never infers content_type at run time -- the
        caller owns it, so a UI that lets the user override the pack's choice
        and a script that does not both behave predictably.
        """
        if band not in DURATION_BANDS:
            raise ScriptGenerationError(
                f"Unknown duration band {band!r}. Expected one of "
                f"{sorted(DURATION_BANDS)}."
            )
        target_min_sec, target_max_sec = DURATION_BANDS[band]
        config = cls.from_env()
        pack_values = {
            "genre": pack.slug,
            "content_type": pack.content_type,
            "music_tags": pack.music_tags,
            "target_min_sec": target_min_sec,
            "target_max_sec": target_max_sec,
        }
        engine = overrides.get("tts_engine", config.tts_engine)
        if engine in _VOICE_SLUG_ENGINES and "voice_slug" not in overrides:
            try:
                from core.voice_selection import voice_for_family
                pack_values["voice_slug"] = voice_for_family(pack.family)
            except ValueError:
                raise
            except Exception as exc:
                logger.warning("voice selection failed for %s (%s); using engine default voice", pack.family, exc)
        if engine == "elevenlabs" and "elevenlabs_voice_id" not in overrides:
            from core.elevenlabs_tts.voices import voice_for_family as el_voice_for_family
            pack_values["elevenlabs_voice_id"] = el_voice_for_family(pack.family)
        config = replace(config, **pack_values)
        if overrides:
            config = replace(config, **overrides)
        return config


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
    sampling: dict = field(default_factory=dict)
    # {"length_repair": bool, "length_repair_reverted": bool,
    # "pause_scale": float}; pause_scale is 1.0 when no pause scaling was
    # applied; length_repair_reverted is True when the budget-spent fallback
    # discarded the length rewrite.
    duration_control: dict = field(default_factory=dict)
    # {"ran": bool, "findings_first": int, "findings_final": int}. ran is True
    # when at least one LLM content check returned a usable answer;
    # findings_final counts the SENSITIVE_CONTENT advisories returned.
    content_check: dict = field(default_factory=dict)


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
    draft_script: str = ""
    voice: str = ""


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


def _duration_control_enabled() -> bool:
    return os.environ.get("MOODSCAPE_DURATION_CONTROL", "1") != "0"


def _content_check_enabled() -> bool:
    return os.environ.get("MOODSCAPE_CONTENT_CHECK", "1") != "0"


def _length_repair_violation(
    script: str, estimated_sec: float, min_sec: float, max_sec: float
) -> Violation:
    """An ADVISORY violation asking the judge to rewrite to the window's midpoint."""
    target_mid = (min_sec + max_sec) / 2
    speech_words = len(strip_tags(script).split())
    pauses = total_pause_sec(script)
    if pauses >= target_mid:
        # Pauses alone fill the target, so the speech-only formula would ask for
        # "about 1 word". Scale the whole script instead: cutting a passage
        # removes its pauses too.
        target_words = max(1, round(speech_words * target_mid / max(1.0, estimated_sec)))
        how = "cut whole passages together with their pauses"
    else:
        # Scale SPEECH only: pauses are not words, so a pause-heavy script must not
        # have its word count scaled by the pause-inclusive time ratio.
        target_words = round(
            speech_words * (target_mid - pauses) / max(1.0, estimated_sec - pauses)
        )
        how = "keep the same [pause] marker style"
    return Violation(
        code="DURATION_OUT_OF_WINDOW",
        severity=ADVISORY,
        message=(
            f"Estimated runtime {estimated_sec:.0f}s; target {min_sec:.0f}-{max_sec:.0f}s. "
            f"Rewrite to about {target_words} words of speech ({how}), "
            f"aiming for {target_mid:.0f}s."
        ),
    )


def _pause_scale_factor(
    script: str, estimated_sec: float, min_sec: float, max_sec: float
) -> float | None:
    """Pause factor that moves the estimate to the nearest window edge.

    Only explicit [pause:Xs] seconds scale; everything else (speech, breath
    cues, paragraph pauses) is held fixed. Returns None when there is nothing
    to scale. The result is clamped to PAUSE_SCALE_BOUNDS.
    """
    pauses = total_pause_sec(script)
    if pauses <= 0:
        return None
    fixed = estimated_sec - pauses
    target = max_sec if estimated_sec > max_sec else min_sec
    lo, hi = PAUSE_SCALE_BOUNDS
    return round(min(hi, max(lo, (target - fixed) / pauses)), 3)


def generate_script(
    prompt: str,
    *,
    generator_engine: ScriptEngine,
    judge_engine: ScriptEngine,
    config: AutoConfig,
    planner_engine: ScriptEngine | None = None,
    pack=None,
    angle=None,
    steer: str | None = "",
    progress_cb=None,
) -> ScriptOutcome:
    steer = steer or ""
    """Run [planner ->] generator -> judge -> lint -> bounded repair.

    Whichever stage raises -- planning, drafting, reviewing, or a repair --
    every engine that was constructed for this run is unloaded on the way
    out. An 18 GB model left resident while a later stage never runs is
    exactly the swap / MPS-bus-error condition this machinery exists to
    prevent. Unloading is deduplicated by identity, so a shared engine
    (planner sharing the generator's model, or the generator sharing the
    judge's -- see run()'s spec-matching reuse) is unloaded exactly once,
    and never early: it is not unloaded between two stages that share it.

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

    unloaded_ids: set[int] = set()

    def _unload_once(engine: ScriptEngine | None) -> None:
        if engine is not None and id(engine) not in unloaded_ids:
            engine.unload()
            unloaded_ids.add(id(engine))

    brief = ""
    try:
        stage_sampling = {
            "planner": sampling_for("planner", planner_engine.name) if planner_engine is not None else None,
            "writer": sampling_for("writer", generator_engine.name),
            "judge": sampling_for("judge", judge_engine.name),
            "repair": sampling_for("repair", judge_engine.name),
        }
    except ValueError as exc:
        raise ScriptGenerationError(str(exc)) from exc

    try:
        if pack is not None and angle is not None and planner_engine is not None:
            if progress_cb:
                progress_cb(0.02, f"Planning a {pack.label} session")
            planner_system = build_planner_system_prompt(
                config.content_type,
                config.target_min_sec,
                config.target_max_sec,
                engine=config.tts_engine,
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
                sampling=stage_sampling["planner"],
            )
            prompt = brief
            # Only unload if the writer is a different model. The default
            # config uses one model for both stages, where unloading would
            # pay a full 18 GB reload to save nothing.
            if planner_engine is not generator_engine:
                _unload_once(planner_engine)

        if progress_cb:
            progress_cb(0.05, "Writing draft script")
        draft_script = draft(
            generator_engine,
            prompt,
            gen_system,
            max_tokens=config.max_tokens,
            sampling=stage_sampling["writer"],
        )
        # Same reasoning as the planner/generator share above, mirrored for
        # generator/judge: don't evict a model that review() is about to use
        # again immediately.
        if generator_engine is not judge_engine:
            _unload_once(generator_engine)

        if progress_cb:
            progress_cb(0.12, "Reviewing script")
        script, changelog = review(
            judge_engine,
            draft_script,
            judge_system,
            max_tokens=config.max_tokens,
            sampling=stage_sampling["judge"],
        )

        def _lint(text: str):
            """Estimate + every check, so violations always match ``text``."""
            est = estimate_duration_sec(
                text,
                engine=config.tts_engine,
                content_type=config.content_type,
            )
            found = check(
                text,
                estimated_sec=est,
                target_min_sec=config.target_min_sec,
                target_max_sec=config.target_max_sec,
            )
            if pack is not None:
                found = found + check_pack_rules(text, pack)

            rep = None
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
                rep = assess(
                    text, compare_against=same_genre, idf_texts=all_texts
                )
                found = found + check_originality(rep)
            return est, found, rep

        duration_control = _duration_control_enabled()
        min_sec, max_sec = config.target_min_sec, config.target_max_sec
        length_repair = False
        length_repair_reverted = False
        pause_scale = 1.0
        pause_scale_tried = False

        # Fatal-free snapshots (kind, script, changelog) taken just before a
        # repair that was not fixing a fatal problem: the length rewrite
        # ("length") and the content repair ("content"). If that repair -- or
        # a repair after it -- leaves fatal problems once the budget is spent,
        # the most recent snapshot is restored. Neither duration drift nor a
        # model's opinion of the wording may fail a job.
        fallbacks: list[tuple[str, str, str]] = []

        # LLM content check (spec Part K). Results are cached by exact script
        # text, so re-linting the same script (or restoring a snapshot that
        # was already checked) costs no judge call; calls are capped at
        # CONTENT_CHECK_MAX_CALLS.
        content_check = _content_check_enabled()
        content_genre = config.genre or (getattr(pack, "slug", "") if pack is not None else "")
        content_cache: dict[str, list[dict] | None] = {}
        content_calls = 0
        content_repaired = False
        content_ran = False
        content_first: int | None = None
        content_last: list[dict] = []
        content_advisories: list[Violation] = []

        def _content_findings_for(text: str) -> list[dict]:
            """Findings for ``text``; [] when clean or the check was skipped.

            When the check cannot run on ``text`` (malformed replies, engine
            error, or the call cap), the last usable findings whose quotes
            are still present in ``text`` are carried forward: a quote that
            survived a repair is still a finding.
            """
            nonlocal content_calls, content_ran, content_first, content_last
            if text in content_cache:
                found = content_cache[text]
            elif content_calls >= CONTENT_CHECK_MAX_CALLS:
                logger.warning(
                    "Content check: call cap (%d) reached; carrying earlier findings forward",
                    CONTENT_CHECK_MAX_CALLS,
                )
                found = None
            else:
                content_calls += 1
                if progress_cb:
                    progress_cb(0.16, "Checking script wording")
                found = content_findings(
                    judge_engine,
                    text,
                    genre=content_genre,
                    label=getattr(pack, "label", "") or "",
                    technique=getattr(pack, "technique", "") or "",
                    sampling=stage_sampling["judge"],
                )
                content_cache[text] = found
                if found is not None:
                    content_ran = True
                    if content_first is None:
                        content_first = len(found)
                    content_last = found
            if found is None:
                return verify_findings(content_last, text)
            return found

        repairs_used = 0
        while True:
            estimated_sec, violations, report = _lint(script)
            fatal = fatal_violations(violations)

            if not fatal:
                # Post-lint steps, in order: length repair, content check,
                # pause scaling. Each either loops back for a fresh lint
                # (``continue``) or falls through to the next.
                if duration_control:
                    # Step 1: one targeted length repair when far outside the window.
                    far_outside = not (
                        min_sec * (1 - LENGTH_REPAIR_TOLERANCE)
                        <= estimated_sec
                        <= max_sec * (1 + LENGTH_REPAIR_TOLERANCE)
                    )
                    if (
                        far_outside
                        and not length_repair
                        and repairs_used < config.max_repairs
                        and estimated_sec > 0
                    ):
                        fallbacks.append(("length", script, changelog))
                        length_repair = True
                        repairs_used += 1
                        if progress_cb:
                            progress_cb(0.15, "Adjusting script length")
                        script, repair_log = repair(
                            judge_engine,
                            script,
                            [_length_repair_violation(script, estimated_sec, min_sec, max_sec)]
                            + [v for v in violations if v.code in STYLE_CODES],
                            judge_system,
                            max_tokens=config.max_tokens,
                            sampling=stage_sampling["repair"],
                        )
                        changelog = f"{changelog}\n{repair_log}".strip()
                        continue

                # Step 2: LLM content check, after the length rewrite so the
                # rewrite is checked too, and before pause scaling so the
                # content-repaired script is the one that gets scaled. One
                # repair at most (FATAL-for-repair within the shared budget);
                # anything found after that, or once the budget is spent, is
                # returned as ADVISORY so a false positive never stops a job.
                content_advisories = []
                if content_check:
                    found = _content_findings_for(script)
                    if found and not content_repaired and repairs_used < config.max_repairs:
                        fallbacks.append(("content", script, changelog))
                        content_repaired = True
                        repairs_used += 1
                        if progress_cb:
                            progress_cb(0.17, "Repairing sensitive wording")
                        script, repair_log = repair(
                            judge_engine,
                            script,
                            findings_to_violations(found, severity=FATAL)
                            + [v for v in violations if v.code in STYLE_CODES],
                            judge_system,
                            max_tokens=config.max_tokens,
                            sampling=stage_sampling["repair"],
                        )
                        changelog = f"{changelog}\n{repair_log}".strip()
                        continue
                    content_advisories = findings_to_violations(found, severity=ADVISORY)

                if duration_control:
                    # Step 3: bounded pause scaling toward the nearest edge.
                    # Applied at most once, in place, and only kept if the
                    # scaled script lints fatal-free (e.g. a 55s pause scaled
                    # past the 60s marker limit is rejected, not repaired).
                    if not pause_scale_tried and not (min_sec <= estimated_sec <= max_sec):
                        pause_scale_tried = True
                        factor = _pause_scale_factor(script, estimated_sec, min_sec, max_sec)
                        scaled = scale_pauses(script, factor) if factor is not None else script
                        if scaled != script:
                            s_est, s_violations, s_report = _lint(scaled)
                            if not fatal_violations(s_violations):
                                script = scaled
                                estimated_sec, violations, report = s_est, s_violations, s_report
                                pause_scale = factor
                                logger.info(
                                    "Duration control: scaled pauses x%.3f -> estimate %.0fs",
                                    factor,
                                    estimated_sec,
                                )
                            else:
                                logger.warning(
                                    "Duration control: pause scaling x%.3f introduced fatal "
                                    "violations; keeping unscaled script",
                                    factor,
                                )

                return ScriptOutcome(
                    script=script,
                    draft_script=draft_script,
                    changelog=changelog,
                    violations=violations + content_advisories,
                    estimated_sec=estimated_sec,
                    repairs_used=repairs_used,
                    originality_cosine=report.max_cosine if report else 0.0,
                    originality_shared_span=report.shared_span if report else 0,
                    brief=brief,
                    sampling={
                        k: v.as_dict() for k, v in stage_sampling.items() if v is not None
                    },
                    duration_control={
                        "enabled": duration_control,
                        "length_repair": length_repair,
                        "length_repair_reverted": length_repair_reverted,
                        "pause_scale": pause_scale,
                    },
                    content_check={
                        "enabled": content_check,
                        "ran": content_ran,
                        "findings_first": content_first or 0,
                        "findings_final": len(content_advisories),
                    },
                )

            if repairs_used >= config.max_repairs and fallbacks:
                # A length rewrite or content repair (or a repair after it)
                # left fatal problems and the budget is spent: fall back to the
                # most recent fatal-free script from before it. Duration drift
                # and the LLM's content opinion are advisory; they never fail
                # a job. The restored script goes through the post-lint steps
                # again, so the content check covers it too.
                kind, script, changelog = fallbacks.pop()
                logger.warning(
                    "%s repair left fatal violations (%s); reverting to the pre-repair script",
                    "Duration control: length" if kind == "length" else "Content check: wording",
                    ", ".join(sorted({v.code for v in fatal})),
                )
                if kind == "length":
                    length_repair_reverted = True
                    note = "duration control: length repair reverted"
                else:
                    note = "content check: wording repair reverted"
                changelog = f"{changelog}\n- {note}".strip()
                continue

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
                judge_engine,
                script,
                fatal
                + [v for v in violations if v.code in STYLE_CODES],
                judge_system,
                max_tokens=config.max_tokens,
                sampling=stage_sampling["repair"],
            )
            changelog = f"{changelog}\n{repair_log}".strip()
    finally:
        # Covers every early exit above (plan/draft/review raising) as well
        # as the normal return and the fatal-after-budget raise. Deduped by
        # identity, so a shared planner/generator or generator/judge engine
        # is unloaded exactly once, not zero and not twice.
        _unload_once(planner_engine)
        _unload_once(generator_engine)
        _unload_once(judge_engine)


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


@dataclass
class RenderResult:
    """Paths and measurements from rendering a script to audio."""

    audio_path: str
    script_path: str
    meta_path: str
    actual_sec: float | None
    voice: str  # slug or ""


def tts_preflight(config: AutoConfig, client_factory=None) -> None:
    """Fail in seconds, before any LLM stage, if the TTS provider cannot render."""
    if config.tts_engine != "elevenlabs":
        return
    from core.elevenlabs_tts.client import ElevenLabsClient, ElevenLabsConfigError
    from core.elevenlabs_tts.direction import default_direction
    from core.elevenlabs_tts.voices import default_voice_id
    try:
        default_direction()
        client = (client_factory or ElevenLabsClient)()
        voice_id = config.elevenlabs_voice_id or default_voice_id()
        if not client.voice_exists(voice_id):
            raise ScriptGenerationError(
                f"ElevenLabs voice {voice_id!r} is not in this account — add it from the "
                "ElevenLabs Voice Library (Add to my voices), or change assets/elevenlabs/voices.toml."
            )
    except (ElevenLabsConfigError, ValueError) as exc:
        raise ScriptGenerationError(str(exc)) from exc


def render_script(
    script: str,
    *,
    config: AutoConfig,
    background_label: str,
    background_path: str,
    pipeline=None,
    progress_cb=None,
    meta: dict | None = None,
    **pipeline_kwargs,
) -> RenderResult:
    """Render an already-written script with a fixed background.

    No LLM stage runs, so a saved script can be re-rendered (same script,
    same background, same seed) and compared on audio alone. ``meta`` is the
    caller's metadata; this function merges in the keys it owns.
    """
    meta = dict(meta or {})
    if pipeline is None:
        from core.pipeline import MeditationPipeline
        pipeline = MeditationPipeline()

    from core.content_profiles import get_profile
    profile = get_profile(config.content_type)
    speed = config.speed if config.speed is not None else profile["speed"]

    generate_kwargs = {
        "speed": speed,
        "duck_amount_db": profile["duck_amount_db"],
        "reverb_amount": profile["reverb_amount"],
        "fade_in_sec": profile["fade_in_sec"],
        "fade_out_sec": profile["fade_out_sec"],
    }
    if config.voice is not None:
        generate_kwargs["voice"] = config.voice
    elif config.tts_engine == "kokoro":
        generate_kwargs["voice"] = "balanced_calm"

    voice_slug = config.voice_slug if config.tts_engine in _VOICE_SLUG_ENGINES else None
    if voice_slug:
        generate_kwargs["f5_voice_slug"] = voice_slug

    el_voice = None
    est_chars = 0
    if config.tts_engine == "elevenlabs":
        from core.elevenlabs_tts.preprocessor import estimate_billed_chars, prepare_segments
        from core.elevenlabs_tts.voices import default_voice_id
        el_voice = config.elevenlabs_voice_id or default_voice_id()
        generate_kwargs["elevenlabs_voice_id"] = el_voice
        est_chars = estimate_billed_chars(prepare_segments(script, content_type=config.content_type))
        logger.info("ElevenLabs: ~%d characters for this render (before cache)", est_chars)
        if progress_cb is not None:
            progress_cb(0.0, f"ElevenLabs: ~{est_chars:,} characters (cached chunks are free)")

    generate_kwargs.update(pipeline_kwargs)

    audio_path, _status = pipeline.generate(
        script=script,
        music_prompt="",
        content_type=config.content_type,
        tts_engine=config.tts_engine,
        music_model="upload",
        uploaded_music_path=background_path,
        progress_cb=progress_cb,
        **generate_kwargs,
    )

    actual_sec = _measure_actual_duration_sec(audio_path)
    estimate_ratio = None
    estimated_sec = meta.get("estimated_sec")
    if actual_sec is not None:
        if estimated_sec:
            log_estimate_accuracy(estimated_sec, actual_sec, config.tts_engine)
            estimate_ratio = actual_sec / estimated_sec

    audio = Path(audio_path)
    script_path = audio.with_suffix(".script.txt")
    meta_path = audio.with_suffix(".meta.json")

    meta.update(
        {
            "content_type": config.content_type,
            "genre": config.genre,
            "tts_engine": config.tts_engine,
            "voice": el_voice or voice_slug,
            "background": background_label,
            "background_path": background_path,
            "actual_sec": actual_sec,
            "estimate_ratio": estimate_ratio,
        }
    )
    speech_qa = getattr(pipeline, "last_speech_qa", None)
    if isinstance(speech_qa, dict) and speech_qa:
        meta["speech_qa"] = speech_qa
    tts_cache_hit = getattr(pipeline, "last_tts_cache_hit", None)
    if isinstance(tts_cache_hit, bool):
        meta["tts_cache_hit"] = tts_cache_hit
    if el_voice is not None:
        manifest = getattr(pipeline, "last_tts_manifest", None)
        meta["elevenlabs"] = {**(manifest if isinstance(manifest, dict) else {}),
                              "estimated_chars": est_chars}

    script_path.write_text(script, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return RenderResult(
        audio_path=str(audio),
        script_path=str(script_path),
        meta_path=str(meta_path),
        actual_sec=actual_sec,
        voice=el_voice or voice_slug or "",
    )


def run(
    prompt: str = "",
    *,
    genre: str | None = None,
    steer: str | None = "",
    config: AutoConfig | None = None,
    pipeline=None,
    planner_engine: ScriptEngine | None = None,
    generator_engine: ScriptEngine | None = None,
    judge_engine: ScriptEngine | None = None,
    rng: random.Random | None = None,
    progress_cb=None,
    **pipeline_kwargs,
) -> AutoResult:
    steer = steer or ""
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
    tts_preflight(config)

    pack, angle = None, None
    if genre:
        pack = load_pack(genre)
        angle = pick_angle(
            pack,
            recent=recent_angles(genre, limit=3, corpus_dir=config.corpus_dir),
            rng=rng,
        )
        config.angle = angle.name
        # config.genre is the single source of truth read by comparison, the
        # avoid-list and the corpus write below -- angle rotation above reads
        # the `genre` ARGUMENT instead, so without this a caller passing a
        # genre string alongside a config that doesn't already carry it (e.g.
        # a bare AutoConfig()) would file the script under the empty-string
        # partition and get no music tags, silently.
        config.genre = genre

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

    meta = {
        "prompt": prompt,
        "angle": config.angle,
        "generator": generator_engine.name,
        "judge": judge_engine.name,
        "changelog": outcome.changelog,
        "violations": _violation_dicts(outcome.violations),
        "estimated_sec": outcome.estimated_sec,
        "repairs_used": outcome.repairs_used,
        "originality_cosine": outcome.originality_cosine,
        "originality_shared_span": outcome.originality_shared_span,
        "brief": outcome.brief,
        "sampling": outcome.sampling,
        "duration_control": outcome.duration_control,
        "content_check": outcome.content_check,
    }
    rendered = render_script(
        outcome.script,
        config=config,
        background_label=background_label,
        background_path=background_path,
        pipeline=pipeline,
        progress_cb=progress_cb,
        meta=meta,
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

    return AutoResult(
        audio_path=rendered.audio_path,
        script_path=rendered.script_path,
        meta_path=rendered.meta_path,
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
        draft_script=outcome.draft_script,
        voice=rendered.voice,
    )
