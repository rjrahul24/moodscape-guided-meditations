# Automated Meditation Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a natural-language prompt ("I'm feeling anxious, I need a relaxing meditation") into a finished meditation WAV, with no human step in between.

**Architecture:** A new `core/script_gen/` package generates the script in two LLM passes — a generator, then an *independent* judge that revises rather than scores — bounded by a deterministic linter and duration estimator that cost no tokens. A pluggable `ScriptEngine` ABC (mirroring the existing `SpeechEngine` ABC) puts local, hosted open-weight, and frontier models behind one interface, so the model is a config string. An orchestrator picks a random background instrumental and calls the existing `MeditationPipeline.generate()` unchanged.

**Tech Stack:** Python 3.11, `httpx` (OpenAI-compatible endpoints), `anthropic` SDK, Gradio, `unittest` (run under pytest), existing Kokoro/F5 preprocessors.

**Spec:** [docs/superpowers/specs/2026-09-16-automated-meditation-generation-design.md](../specs/2026-09-16-automated-meditation-generation-design.md)

## Global Constraints

- **Branch:** `dev-automate` (already created, off `dev`). Do not commit to `dev` or `main`.
- **Never modify** `core/pipeline.py`, `core/mixer.py`, or any audio-path module. The auto path *calls* `MeditationPipeline.generate()` exactly as the manual tab does.
- **Every unit test runs with no model and no network call.** Any test needing a real model goes in `tests/integration/` and is marked slow.
- **Test style:** `unittest.TestCase` classes ending with `if __name__ == "__main__": unittest.main()`, matching `tests/unit/test_content_profiles.py`. Run under pytest.
- **Conventional commits:** `feat:` `fix:` `refactor:` `docs:` `test:` `chore:`.
- **Imports:** `from core.module import Thing` (relative to project root).
- **Every subpackage needs `__init__.py` with explicit public exports.**
- **Target duration window:** 300–420 seconds (5–7 min), configurable.
- **Repair budget:** 2 attempts, from `MOODSCAPE_SCRIPT_MAX_REPAIRS`.
- **Segment dict contract** (from the preprocessors): `{"type": "speech", "text": str, "voice": str|None}` and `{"type": "pause", "duration_sec": float}`.
- **Known engine constants:** `core/kokoro_tts/engine.py` defines `INTER_SENTENCE_PAUSE_SEC = 0.8` and `ELLIPSIS_PAUSE_SEC = 1.2`. Import them; do not redefine.
- **`prepare_segments` signature** is `prepare_segments(script: str, content_type: str = "meditation")` in both `core/kokoro_tts/preprocessor.py` and `core/f5_tts/preprocessor.py`.

## Phasing

| Phase | Tasks | Deliverable |
|---|---|---|
| **1 — Deterministic core** | 1–5 | Linter, duration estimator, background picker, rules loader. No LLM anywhere. Fully tested offline. |
| **2 — Model interface** | 6–9 | `ScriptEngine` ABC, adapters, generator + judge + repair loop. Tested against a fake engine. |
| **3 — Orchestration** | 10–12 | Orchestrator, streaming runner, Gradio tab, benchmark harness. |
| **4 — Delivery** | 13–15 | End-to-end integration tests, full documentation, review and push. |

Phase 1 is independently valuable: the linter and estimator are usable on hand-written scripts the moment they exist.

**Sequencing note.** A separate branch is correcting the "36 GB" RAM figure to 32 GB in `CLAUDE.md` and `docs/ARCHITECTURE.md` — the same two files Task 14 edits. Task 14 Step 2 and Task 15 Step 4 both check whether that fix has landed and rebase if so. Do not fix the RAM number in this branch.

---

# Phase 1 — Deterministic core

### Task 1: Script linter — format checks

**Files:**
- Create: `core/script_gen/__init__.py`
- Create: `core/script_gen/linter.py`
- Test: `tests/unit/test_script_linter.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Violation` (frozen dataclass with `code: str`, `severity: str`, `message: str`, `span: tuple[int,int] | None`), constants `FATAL = "fatal"` and `ADVISORY = "advisory"`, and `check_format(script: str) -> list[Violation]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_script_linter.py`:

```python
"""Tests for the deterministic script linter.

The linter is what makes a weaker or cheaper LLM viable: format errors are
caught in code and repaired, requiring no model judgment.
"""

import unittest

from core.script_gen.linter import (
    ADVISORY,
    FATAL,
    Violation,
    check_format,
)


def codes(violations):
    return {v.code for v in violations}


class TestFormatChecks(unittest.TestCase):
    def test_clean_script_has_no_violations(self):
        script = "Settle in and let your shoulders drop.\n\n[pause:4s]\n\nBreathe out slowly."
        self.assertEqual(check_format(script), [])

    def test_malformed_pause_marker_is_fatal(self):
        violations = check_format("Breathe in. [pause:4] Breathe out.")
        self.assertIn("MARKER_MALFORMED", codes(violations))
        self.assertTrue(any(v.severity == FATAL for v in violations))

    def test_pause_out_of_bounds_is_fatal(self):
        violations = check_format("Rest here. [pause:900s] Now return.")
        self.assertIn("PAUSE_OUT_OF_RANGE", codes(violations))

    def test_unknown_tag_is_fatal(self):
        violations = check_format("Settle in. [whisper] Let go.")
        self.assertIn("UNKNOWN_TAG", codes(violations))

    def test_known_tags_are_accepted(self):
        script = "Settle in. [breath] Let go. [inhale] And out. [exhale]"
        self.assertEqual(check_format(script), [])

    def test_markdown_is_fatal(self):
        violations = check_format("## Opening\n\nBreathe in.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_bold_markdown_is_fatal(self):
        violations = check_format("Now **really** let go.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))

    def test_emoji_is_fatal(self):
        violations = check_format("Breathe in and smile \U0001F60A")
        self.assertIn("EMOJI_PRESENT", codes(violations))

    def test_shouting_caps_is_advisory(self):
        violations = check_format("Now RELAX completely.")
        self.assertIn("ALL_CAPS", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY for v in violations if v.code == "ALL_CAPS")
        )

    def test_short_acronyms_are_allowed(self):
        self.assertEqual(check_format("Rest for a moment. OK."), [])

    def test_long_sentence_is_advisory(self):
        long_sentence = " ".join(["word"] * 40) + "."
        violations = check_format(long_sentence)
        self.assertIn("SENTENCE_TOO_LONG", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY for v in violations if v.code == "SENTENCE_TOO_LONG")
        )

    def test_violation_is_frozen(self):
        v = Violation(code="X", severity=FATAL, message="m")
        with self.assertRaises(Exception):
            v.code = "Y"


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen'`

- [ ] **Step 3: Create the package marker**

Create `core/script_gen/__init__.py`:

```python
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
```

- [ ] **Step 4: Write minimal implementation**

Create `core/script_gen/linter.py`:

```python
"""Deterministic validation of generated scripts.

Three check families: format (this module's check_format), safety hard-blocks
(check_safety), and duration (check). Violations are returned as structured
objects so they can be fed back to the judge as a *targeted* repair
instruction rather than a vague "try again".

Severity matters. Treating every violation as fatal would make a weaker model
unusable; treating none as fatal would let a safety failure reach audio.
"""

import re
from dataclasses import dataclass

FATAL = "fatal"
ADVISORY = "advisory"

# Bounds for [pause:Xs]. Below 0.5s the marker is pointless; above 60s it is
# almost certainly a model error rather than an intentional silence.
MIN_PAUSE_SEC = 0.5
MAX_PAUSE_SEC = 60.0

# The guide's ideal band is 8-20 words. Allow headroom before complaining.
MAX_SENTENCE_WORDS = 25

KNOWN_BARE_TAGS = {"breath", "inhale", "exhale"}

_PAUSE_TAG = re.compile(r"\[pause:(\d+(?:\.\d+)?)s\]")
_ANY_TAG = re.compile(r"\[([^\]]*)\]")
_MARKDOWN = re.compile(r"(^\s{0,3}#{1,6}\s)|(\*\*)|(^\s*[-*+]\s+)", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Violation:
    """One problem found in a script.

    Args:
        code: Stable machine-readable identifier, e.g. "MARKER_MALFORMED".
        severity: FATAL (do not render) or ADVISORY (render, log a warning).
        message: Human- and model-readable description used for repair.
        span: Optional (start, end) character offsets into the script.
    """

    code: str
    severity: str
    message: str
    span: tuple[int, int] | None = None


def _strip_tags(script: str) -> str:
    """Remove all bracket tags so prose checks do not trip over them."""
    return _ANY_TAG.sub(" ", script)


def check_format(script: str) -> list[Violation]:
    """Validate the engine contract: tags, markup, and sentence shape."""
    violations: list[Violation] = []

    for match in _ANY_TAG.finditer(script):
        inner = match.group(1).strip()
        span = (match.start(), match.end())
        if inner in KNOWN_BARE_TAGS:
            continue
        if inner.startswith("pause:"):
            if not _PAUSE_TAG.fullmatch(match.group(0)):
                violations.append(
                    Violation(
                        code="MARKER_MALFORMED",
                        severity=FATAL,
                        message=(
                            f"Malformed pause marker {match.group(0)!r}. "
                            "Use exactly [pause:Xs], e.g. [pause:4s] or [pause:2.5s]."
                        ),
                        span=span,
                    )
                )
                continue
            seconds = float(_PAUSE_TAG.fullmatch(match.group(0)).group(1))
            if not (MIN_PAUSE_SEC <= seconds <= MAX_PAUSE_SEC):
                violations.append(
                    Violation(
                        code="PAUSE_OUT_OF_RANGE",
                        severity=FATAL,
                        message=(
                            f"Pause of {seconds}s is outside the allowed range "
                            f"{MIN_PAUSE_SEC}-{MAX_PAUSE_SEC}s."
                        ),
                        span=span,
                    )
                )
            continue
        violations.append(
            Violation(
                code="UNKNOWN_TAG",
                severity=FATAL,
                message=(
                    f"Unknown tag {match.group(0)!r}. Only [pause:Xs], "
                    "[breath], [inhale] and [exhale] are supported."
                ),
                span=span,
            )
        )

    if _MARKDOWN.search(script):
        violations.append(
            Violation(
                code="MARKDOWN_PRESENT",
                severity=FATAL,
                message=(
                    "Script contains markdown (headings, bold, or bullets). "
                    "Output plain prose only — the TTS engine reads markup aloud."
                ),
            )
        )

    if _EMOJI.search(script):
        violations.append(
            Violation(
                code="EMOJI_PRESENT",
                severity=FATAL,
                message="Script contains emoji. Use plain text only.",
            )
        )

    prose = _strip_tags(script)

    for match in _ALL_CAPS.finditer(prose):
        violations.append(
            Violation(
                code="ALL_CAPS",
                severity=ADVISORY,
                message=(
                    f"{match.group(0)!r} is in capitals. Use lower case — "
                    "capitals change the engine's pronunciation."
                ),
            )
        )

    for sentence in _SENTENCE_SPLIT.split(prose):
        words = sentence.split()
        if len(words) > MAX_SENTENCE_WORDS:
            violations.append(
                Violation(
                    code="SENTENCE_TOO_LONG",
                    severity=ADVISORY,
                    message=(
                        f"Sentence is {len(words)} words; keep sentences to "
                        "8-20 words so the pacing stays calm."
                    ),
                )
            )

    return violations
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v`
Expected: PASS — 11 tests

- [ ] **Step 6: Commit**

```bash
git add core/script_gen/__init__.py core/script_gen/linter.py tests/unit/test_script_linter.py
git commit -m "feat(script_gen): add deterministic format linter"
```

---

### Task 2: Script linter — safety hard-blocks

**Files:**
- Modify: `core/script_gen/linter.py` (append; do not alter Task 1 code)
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_script_linter.py` (append a new TestCase class)

**Interfaces:**
- Consumes: `Violation`, `FATAL`, `ADVISORY` from Task 1.
- Produces: `check_safety(script: str) -> list[Violation]`; `check(script: str, *, estimated_sec: float | None = None, target_min_sec: float = 300.0, target_max_sec: float = 420.0) -> list[Violation]`; `fatal_violations(violations: list[Violation]) -> list[Violation]`; `format_for_repair(violations: list[Violation]) -> str`.

**Why these specific rules:** this is meditation content aimed at people who are anxious, sleepless, or grieving. The failure modes are not cosmetic. Outcome promises set a listener up to read their own experience as failure; invalidating imperatives tell a distressed person their feelings are wrong; extended breath-holds are a genuine physical risk for people with panic disorder or asthma; dissociation-adjacent imagery is actively contraindicated for trauma survivors.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_script_linter.py`, above the `if __name__` block, and add `check, check_safety, fatal_violations, format_for_repair` to the existing import from `core.script_gen.linter`:

```python
class TestSafetyChecks(unittest.TestCase):
    def test_clean_script_passes(self):
        script = "If it feels right, you might let your eyes close."
        self.assertEqual(check_safety(script), [])

    def test_clinical_claim_is_fatal(self):
        violations = check_safety("This meditation will cure your anxiety.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))
        self.assertTrue(all(v.severity == FATAL for v in violations))

    def test_therapy_replacement_is_fatal(self):
        violations = check_safety("This replaces therapy for most people.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_outcome_promise_is_fatal(self):
        violations = check_safety("By the end you will be completely calm.")
        self.assertIn("OUTCOME_PROMISE", codes(violations))

    def test_invalidating_imperative_is_fatal(self):
        violations = check_safety("Don't feel anxious about it.")
        self.assertIn("INVALIDATING", codes(violations))

    def test_curly_apostrophe_does_not_evade_the_block(self):
        # LLMs emit U+2019 constantly; a safety block must not be defeated
        # by a typographic quote.
        violations = check_safety("Don’t feel anxious about it.")
        self.assertIn("INVALIDATING", codes(violations))

    def test_extended_breath_hold_is_fatal(self):
        violations = check_safety("Hold your breath for 20 seconds.")
        self.assertIn("BREATH_HOLD", codes(violations))

    def test_short_breath_hold_is_allowed(self):
        self.assertEqual(check_safety("Hold your breath for 3 seconds."), [])

    def test_dissociation_imagery_is_fatal(self):
        violations = check_safety("Now leave your body behind.")
        self.assertIn("DISSOCIATION", codes(violations))

    def test_matching_is_case_insensitive(self):
        violations = check_safety("This Will Cure Your Depression.")
        self.assertIn("CLINICAL_CLAIM", codes(violations))


class TestCombinedCheck(unittest.TestCase):
    def test_duration_below_window_is_advisory(self):
        violations = check(
            "Breathe in.", estimated_sec=100.0,
            target_min_sec=300.0, target_max_sec=420.0,
        )
        self.assertIn("DURATION_OUT_OF_WINDOW", codes(violations))
        self.assertTrue(
            all(v.severity == ADVISORY
                for v in violations if v.code == "DURATION_OUT_OF_WINDOW")
        )

    def test_duration_inside_window_is_clean(self):
        violations = check("Breathe in.", estimated_sec=360.0)
        self.assertNotIn("DURATION_OUT_OF_WINDOW", codes(violations))

    def test_duration_skipped_when_not_supplied(self):
        violations = check("Breathe in.")
        self.assertNotIn("DURATION_OUT_OF_WINDOW", codes(violations))

    def test_check_combines_format_and_safety(self):
        violations = check("## Title\n\nThis will cure your anxiety.")
        self.assertIn("MARKDOWN_PRESENT", codes(violations))
        self.assertIn("CLINICAL_CLAIM", codes(violations))

    def test_fatal_violations_filters(self):
        violations = check("Now RELAX. This will cure your anxiety.")
        fatal = fatal_violations(violations)
        self.assertTrue(fatal)
        self.assertTrue(all(v.severity == FATAL for v in fatal))
        self.assertNotIn("ALL_CAPS", {v.code for v in fatal})

    def test_format_for_repair_lists_every_violation(self):
        violations = check("## Title\n\nThis will cure your anxiety.")
        text = format_for_repair(violations)
        self.assertIn("MARKDOWN_PRESENT", text)
        self.assertIn("CLINICAL_CLAIM", text)

    def test_format_for_repair_is_empty_when_clean(self):
        self.assertEqual(format_for_repair([]), "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_safety'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/script_gen/linter.py`:

```python
# --- Safety hard-blocks -------------------------------------------------
#
# These are the mental-health guardrails. Each entry is
# (code, compiled pattern, message). Patterns are matched case-insensitively
# against the tag-stripped prose.

MAX_BREATH_HOLD_SEC = 7

_SAFETY_RULES: list[tuple[str, re.Pattern, str]] = [
    (
        "CLINICAL_CLAIM",
        re.compile(
            r"\b(cure[sd]?|heal[s]?|treat[s]?|diagnos\w+)\b[^.?!]{0,40}"
            r"\b(anxiety|depression|trauma|ptsd|insomnia|illness|condition)\b"
            r"|\breplaces?\s+(therapy|medication|treatment)\b",
            re.IGNORECASE,
        ),
        "Clinical claim. This is not treatment and must not present itself as "
        "therapy or a cure. Describe the practice, never a medical outcome.",
    ),
    (
        "OUTCOME_PROMISE",
        re.compile(
            r"\byou\s+will\s+(be|feel)\s+(completely|totally|entirely|fully)\b"
            r"|\bthis\s+will\s+(eliminate|remove|erase|banish)\b"
            r"|\bguarantee[sd]?\b",
            re.IGNORECASE,
        ),
        "Promises an outcome. A listener who does not feel that way will read "
        "it as their own failure. Use invitational phrasing instead.",
    ),
    (
        "INVALIDATING",
        re.compile(
            r"\b(don'?t|do not|stop)\s+feel\w*\b"
            r"|\bthere'?s\s+nothing\s+wrong\s+with\s+you\b"
            r"|\byou\s+shouldn'?t\s+(feel|be)\b",
            re.IGNORECASE,
        ),
        "Invalidating instruction. Telling a distressed listener not to feel "
        "something dismisses their experience. Acknowledge, do not override.",
    ),
    (
        "DISSOCIATION",
        re.compile(
            r"\bleave\s+your\s+body\b"
            r"|\bfloat\s+away\s+from\s+your\s*self\b"
            r"|\byou\s+are\s+not\s+your\s+body\b"
            r"|\bdetach\s+from\s+your\s+body\b",
            re.IGNORECASE,
        ),
        "Dissociation-adjacent imagery, which is contraindicated for trauma "
        "survivors. Keep the listener grounded in the body and the room.",
    ),
]

_BREATH_HOLD = re.compile(
    r"\bhold\s+(?:your\s+)?breath\b[^.?!]{0,30}?(\d+)", re.IGNORECASE
)


def _normalize_apostrophes(text: str) -> str:
    """Fold typographic apostrophes to ASCII before pattern matching.

    LLMs routinely emit U+2019 (') rather than "'". Without this, "Don't
    feel anxious" slips past the INVALIDATING pattern — a safety hard-block
    silently defeated by a curly quote.
    """
    return text.replace("’", "'").replace("ʼ", "'")


def check_safety(script: str) -> list[Violation]:
    """Apply mental-health content hard-blocks. All findings are FATAL."""
    prose = _normalize_apostrophes(_strip_tags(script))
    violations: list[Violation] = []

    for code, pattern, message in _SAFETY_RULES:
        for match in pattern.finditer(prose):
            violations.append(
                Violation(
                    code=code,
                    severity=FATAL,
                    message=f"{message} Found: {match.group(0)!r}.",
                    span=(match.start(), match.end()),
                )
            )

    for match in _BREATH_HOLD.finditer(prose):
        seconds = int(match.group(1))
        if seconds > MAX_BREATH_HOLD_SEC:
            violations.append(
                Violation(
                    code="BREATH_HOLD",
                    severity=FATAL,
                    message=(
                        f"Instructs a {seconds}-second breath hold. Holds over "
                        f"{MAX_BREATH_HOLD_SEC}s are a real risk for listeners "
                        "with panic disorder or asthma. Shorten it or remove it."
                    ),
                    span=(match.start(), match.end()),
                )
            )

    return violations


def check(
    script: str,
    *,
    estimated_sec: float | None = None,
    target_min_sec: float = 300.0,
    target_max_sec: float = 420.0,
) -> list[Violation]:
    """Run every check family. Duration is skipped when no estimate is given."""
    violations = check_format(script) + check_safety(script)

    if estimated_sec is not None and not (
        target_min_sec <= estimated_sec <= target_max_sec
    ):
        violations.append(
            Violation(
                code="DURATION_OUT_OF_WINDOW",
                severity=ADVISORY,
                message=(
                    f"Estimated runtime is {estimated_sec:.0f}s, outside the "
                    f"target {target_min_sec:.0f}-{target_max_sec:.0f}s. "
                    "Add or remove content and pauses to land in the window."
                ),
            )
        )

    return violations


def fatal_violations(violations: list[Violation]) -> list[Violation]:
    """Return only the violations that must block a render."""
    return [v for v in violations if v.severity == FATAL]


def format_for_repair(violations: list[Violation]) -> str:
    """Render violations as a numbered instruction block for the judge.

    Targeted repair beats a vague "try again" — the judge is told exactly what
    is wrong and why.
    """
    if not violations:
        return ""
    lines = [
        f"{i}. [{v.code}/{v.severity}] {v.message}"
        for i, v in enumerate(violations, start=1)
    ]
    return "\n".join(lines)
```

Then replace the contents of `core/script_gen/__init__.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v`
Expected: PASS — 26 tests

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/linter.py core/script_gen/__init__.py tests/unit/test_script_linter.py
git commit -m "feat(script_gen): add mental-health safety hard-blocks to linter"
```

---

### Task 3: Duration estimator

**Files:**
- Create: `core/script_gen/duration.py`
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_duration_estimate.py`

**Interfaces:**
- Consumes: `prepare_segments` from both preprocessors; `INTER_SENTENCE_PAUSE_SEC` and `ELLIPSIS_PAUSE_SEC` from `core/kokoro_tts/engine.py`.
- Produces: `DEFAULT_WPM: dict[str, float]`; `estimate_duration_sec(script: str, *, engine: str = "f5", content_type: str = "meditation", wpm: float | None = None) -> float`; `log_estimate_accuracy(estimated_sec: float, actual_sec: float, engine: str) -> str`.

**Key correctness note:** fades are deliberately **not** added. `apply_fades` shapes amplitude over audio that already exists — a 1.5s fade-in does not make the file 1.5s longer. Adding them would bias every estimate long and trigger needless repair loops.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_duration_estimate.py`:

```python
"""Tests for script runtime estimation.

Pure arithmetic over the engine's own segment parse — no model, no audio.
"""

import unittest

from core.script_gen.duration import (
    DEFAULT_WPM,
    estimate_duration_sec,
    log_estimate_accuracy,
)


class TestDurationEstimate(unittest.TestCase):
    def test_explicit_pauses_are_summed(self):
        # Speech between the pauses is required: the preprocessor MERGES
        # adjacent pauses and keeps only the longest, so "[pause:10s]
        # [pause:20s]" back-to-back yields 20s, not 30s.
        script = "One.\n\n[pause:10s]\n\nTwo.\n\n[pause:20s]\n\nThree."
        estimate = estimate_duration_sec(script, engine="f5")
        self.assertGreaterEqual(estimate, 30.0)

    def test_adjacent_pauses_merge_to_the_longest(self):
        # Guards the merge behaviour itself, so the estimator can never drift
        # into naive addition.
        merged = estimate_duration_sec("[pause:10s]\n\n[pause:20s]", engine="f5")
        self.assertLess(merged, 30.0)

    def test_speech_scales_with_word_count(self):
        short = estimate_duration_sec("one two three four five.", engine="f5")
        longer = estimate_duration_sec(
            " ".join(["word"] * 100) + ".", engine="f5"
        )
        self.assertGreater(longer, short)

    def test_wpm_override_is_honoured(self):
        script = " ".join(["word"] * 100) + "."
        fast = estimate_duration_sec(script, engine="f5", wpm=200.0)
        slow = estimate_duration_sec(script, engine="f5", wpm=50.0)
        self.assertGreater(slow, fast)

    def test_hundred_words_at_hundred_wpm_is_about_a_minute(self):
        script = " ".join(["word"] * 100) + "."
        estimate = estimate_duration_sec(script, engine="f5", wpm=100.0)
        # 60s of speech, plus no pauses and a single sentence (no gaps).
        self.assertAlmostEqual(estimate, 60.0, delta=1.0)

    def test_inter_sentence_gaps_are_counted(self):
        # Both are exactly 8 words; only the sentence count differs, so the
        # delta is purely the three inter-sentence gaps (3 x 0.8s).
        one = estimate_duration_sec(
            "word word word word word word word word.", engine="f5", wpm=100.0
        )
        four = estimate_duration_sec(
            "word word. word word. word word. word word.", engine="f5", wpm=100.0
        )
        self.assertGreater(four, one + 1.5)

    def test_both_engines_supported(self):
        script = "Breathe in and let go.\n\n[pause:5s]\n\nBreathe out."
        self.assertGreater(estimate_duration_sec(script, engine="f5"), 0)
        self.assertGreater(estimate_duration_sec(script, engine="kokoro"), 0)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            estimate_duration_sec("Breathe in.", engine="nope")

    def test_empty_script_is_zero(self):
        self.assertEqual(estimate_duration_sec("", engine="f5"), 0.0)

    def test_default_wpm_defined_for_both_engines(self):
        self.assertIn("f5", DEFAULT_WPM)
        self.assertIn("kokoro", DEFAULT_WPM)

    def test_sleep_story_uses_shorter_paragraph_pauses(self):
        script = "Once there was a lantern.\n\nIt glowed softly."
        meditation = estimate_duration_sec(
            script, engine="f5", content_type="meditation"
        )
        sleep_story = estimate_duration_sec(
            script, engine="f5", content_type="sleep_story"
        )
        self.assertLess(sleep_story, meditation)


class TestAccuracyLogging(unittest.TestCase):
    def test_log_reports_signed_error(self):
        line = log_estimate_accuracy(300.0, 330.0, "f5")
        self.assertIn("f5", line)
        self.assertIn("300", line)
        self.assertIn("330", line)

    def test_log_handles_zero_actual(self):
        line = log_estimate_accuracy(300.0, 0.0, "f5")
        self.assertIsInstance(line, str)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_duration_estimate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.duration'`

- [ ] **Step 3: Write minimal implementation**

Create `core/script_gen/duration.py`:

```python
"""Estimate a script's spoken runtime without rendering it.

Pauses are summed exactly from the engine's own parse. Speech is estimated as
word_count / wpm * 60 — the same formula core/f5_tts/engine.py:463 uses when
fixed pacing is enabled. Inter-sentence room-tone gaps are added because the
engines insert them.

Fades are deliberately NOT added: apply_fades shapes amplitude over audio that
already exists, so they do not extend runtime.
"""

import logging
import re

from core.kokoro_tts.engine import ELLIPSIS_PAUSE_SEC, INTER_SENTENCE_PAUSE_SEC

logger = logging.getLogger(__name__)

# Measured speaking rates. F5 at speed 0.88 runs ~95-100 WPM per
# docs/prompting_guides/vocal_meditation_f5_instructions.md. These are the
# calibration constants: log_estimate_accuracy() exists to refine them from
# real renders rather than leaving them a guess.
DEFAULT_WPM: dict[str, float] = {
    "f5": 97.0,
    "kokoro": 105.0,
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _load_prepare_segments(engine: str):
    """Return the engine's prepare_segments, or raise for an unknown engine."""
    if engine == "f5":
        from core.f5_tts.preprocessor import prepare_segments
        return prepare_segments
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import prepare_segments
        return prepare_segments
    raise ValueError(
        f"Unknown engine {engine!r}. Expected 'f5' or 'kokoro'."
    )


def _speech_seconds(text: str, wpm: float) -> float:
    """Words at the target rate, plus the gaps the engine puts between sentences."""
    words = len(text.split())
    if words == 0:
        return 0.0

    speech = words / wpm * 60.0

    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    gaps = 0.0
    # A gap follows every sentence except the last one in this segment.
    for sentence in sentences[:-1]:
        gaps += (
            ELLIPSIS_PAUSE_SEC
            if sentence.rstrip().endswith("...")
            else INTER_SENTENCE_PAUSE_SEC
        )

    return speech + gaps


def estimate_duration_sec(
    script: str,
    *,
    engine: str = "f5",
    content_type: str = "meditation",
    wpm: float | None = None,
) -> float:
    """Estimate total spoken runtime in seconds.

    Args:
        script: Raw script text with [pause:Xs] markers.
        engine: "f5" or "kokoro" — selects the preprocessor and default WPM.
        content_type: "meditation" or "sleep_story". Changes paragraph-break
            pause length via the content profile.
        wpm: Override the engine's default speaking rate.

    Returns:
        Estimated duration in seconds. 0.0 for an empty script.
    """
    if not script.strip():
        return 0.0

    prepare_segments = _load_prepare_segments(engine)
    rate = wpm if wpm is not None else DEFAULT_WPM[engine]

    segments = prepare_segments(script, content_type=content_type)

    total = 0.0
    for segment in segments:
        if segment["type"] == "pause":
            total += float(segment["duration_sec"])
        elif segment["type"] == "speech":
            total += _speech_seconds(segment["text"], rate)

    return total


def log_estimate_accuracy(
    estimated_sec: float, actual_sec: float, engine: str
) -> str:
    """Log estimate-versus-actual so DEFAULT_WPM can be calibrated from data.

    Returns the log line so callers can also write it into run metadata.
    """
    error = actual_sec - estimated_sec
    ratio = (actual_sec / estimated_sec) if estimated_sec else float("nan")
    line = (
        f"duration[{engine}] estimated={estimated_sec:.1f}s "
        f"actual={actual_sec:.1f}s error={error:+.1f}s ratio={ratio:.3f}"
    )
    logger.info(line)
    return line
```

Add to `core/script_gen/__init__.py` — extend the import block and `__all__`:

```python
from core.script_gen.duration import (
    DEFAULT_WPM,
    estimate_duration_sec,
    log_estimate_accuracy,
)
```

and add `"DEFAULT_WPM"`, `"estimate_duration_sec"`, `"log_estimate_accuracy"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_duration_estimate.py -v`
Expected: PASS — 12 tests

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/duration.py core/script_gen/__init__.py tests/unit/test_duration_estimate.py
git commit -m "feat(script_gen): add script duration estimator"
```

---

### Task 4: Background picker

**Files:**
- Create: `core/background_picker.py`
- Test: `tests/unit/test_background_picker.py`

**Interfaces:**
- Consumes: `scan_backgrounds` from `core.upload_music` (already exists).
- Produces: `pick_background(*, scan=None, exclude: Sequence[str] = (), rng: random.Random | None = None) -> tuple[str, str]` returning `(label, path)`.

**Do not reimplement directory scanning.** `core/upload_music/background_library.py` already provides `scan_backgrounds() -> list[tuple[str, str]]`, returning `("Healing Forest — 23:12", "/abs/path.mp3")` pairs sorted by label. It handles seven audio formats and skips unreadable files with a logged warning. This task reuses it and adds only the random-choice-with-exclusions logic on top. The `scan` parameter exists so tests can inject a fake and stay fast — `scan_backgrounds` reads every file's header via `soundfile.info`, which needs real audio.

`assets/backgrounds/` currently holds 20 `.mp3` files. The exclude-recent behaviour stops a batch of generations landing on the same instrumental repeatedly.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_background_picker.py`:

```python
"""Tests for random background-instrumental selection.

Builds on core.upload_music.scan_backgrounds; a fake scan keeps these fast,
since the real scanner reads audio headers from disk.
"""

import random
import unittest

from core.background_picker import pick_background

FAKE_TRACKS = [
    ("Autumn Sky — 12:00", "/bg/autumn.mp3"),
    ("Healing Forest — 23:12", "/bg/forest.mp3"),
    ("Piano Dreamcloud — 18:40", "/bg/piano.mp3"),
]


def fake_scan(tracks=FAKE_TRACKS):
    return lambda: list(tracks)


class TestBackgroundPicker(unittest.TestCase):
    def test_returns_a_label_and_path_pair(self):
        label, path = pick_background(scan=fake_scan())
        self.assertIn((label, path), FAKE_TRACKS)

    def test_is_reproducible_under_a_seed(self):
        first = pick_background(scan=fake_scan(), rng=random.Random(42))
        second = pick_background(scan=fake_scan(), rng=random.Random(42))
        self.assertEqual(first, second)

    def test_exclude_is_honoured(self):
        _label, path = pick_background(
            scan=fake_scan(), exclude=["/bg/autumn.mp3", "/bg/forest.mp3"]
        )
        self.assertEqual(path, "/bg/piano.mp3")

    def test_excluding_everything_falls_back_to_full_pool(self):
        _label, path = pick_background(
            scan=fake_scan(), exclude=[p for _, p in FAKE_TRACKS]
        )
        self.assertIn(path, [p for _, p in FAKE_TRACKS])

    def test_empty_library_raises_with_a_helpful_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            pick_background(scan=lambda: [])
        self.assertIn("backgrounds", str(ctx.exception).lower())

    def test_defaults_to_the_real_scanner(self):
        # The real library has 20 tracks committed; this guards the wiring.
        label, path = pick_background()
        self.assertTrue(label)
        self.assertTrue(path.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_background_picker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.background_picker'`

- [ ] **Step 3: Write minimal implementation**

Create `core/background_picker.py`:

```python
"""Pick a royalty-free background instrumental at random.

Used by the auto-generation path, which has no UI for choosing a track.

Track discovery is delegated to core.upload_music.scan_backgrounds — the
canonical scanner, which already handles every supported format and skips
unreadable files. This module adds only the random choice and the
exclude-recent behaviour that stops a batch landing on the same instrumental
repeatedly.
"""

import random
from collections.abc import Sequence

from core.upload_music import BACKGROUNDS_DIR, scan_backgrounds


def pick_background(
    *,
    scan=None,
    exclude: Sequence[str] = (),
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Choose one background instrumental at random.

    Args:
        scan: Zero-arg callable returning [(label, path), ...]. Defaults to
            scan_backgrounds. Injected in tests to avoid reading real audio.
        exclude: Paths of recently used tracks to avoid. If excluding them
            would leave nothing, the full pool is used instead — variety is a
            preference, not a reason to fail a job.
        rng: Inject a seeded Random for reproducible selection.

    Returns:
        (label, path) — the label is human-readable, e.g.
        "Healing Forest — 23:12", and goes into the run metadata.

    Raises:
        FileNotFoundError: If the library holds no usable tracks.
    """
    scanner = scan if scan is not None else scan_backgrounds
    pool = scanner()

    if not pool:
        raise FileNotFoundError(
            f"No background instrumentals found in {BACKGROUNDS_DIR}. "
            "Add royalty-free audio files there before auto-generating."
        )

    excluded = set(exclude)
    candidates = [entry for entry in pool if entry[1] not in excluded] or pool

    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_background_picker.py -v`
Expected: PASS — 6 tests

- [ ] **Step 5: Commit**

```bash
git add core/background_picker.py tests/unit/test_background_picker.py
git commit -m "feat(background): add random royalty-free instrumental picker"
```

---

### Task 5: Safety rules document and prompt assembly

**Files:**
- Create: `docs/prompting_guides/content_safety_rules.md`
- Create: `core/script_gen/rules.py`
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_script_rules.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `GUIDES_DIR: Path`; `load_guide(engine: str, content_type: str, guides_dir: Path | None = None) -> str`; `load_safety_rules(guides_dir: Path | None = None) -> str`; `build_generator_system_prompt(engine: str, content_type: str, target_min_sec: float, target_max_sec: float, guides_dir: Path | None = None) -> str`; `build_judge_system_prompt(engine: str, content_type: str, target_min_sec: float, target_max_sec: float, guides_dir: Path | None = None) -> str`.

The guide is read **from disk at call time**, so edits to `docs/prompting_guides/` take effect on the next generation with no code change.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_script_rules.py`:

```python
"""Tests for prompt assembly from on-disk guides."""

import tempfile
import unittest
from pathlib import Path

from core.script_gen.rules import (
    GUIDES_DIR,
    build_generator_system_prompt,
    build_judge_system_prompt,
    load_guide,
    load_safety_rules,
)


class TestRuleLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        for content_type in ("meditation", "sleep_story"):
            for engine in ("f5", "kokoro"):
                (self.dir / f"vocal_{content_type}_{engine}_instructions.md").write_text(
                    f"GUIDE {content_type} {engine}"
                )
        (self.dir / "content_safety_rules.md").write_text("SAFETY RULES")

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_the_matching_guide(self):
        text = load_guide("f5", "meditation", guides_dir=self.dir)
        self.assertEqual(text, "GUIDE meditation f5")

    def test_loads_sleep_story_guide(self):
        text = load_guide("kokoro", "sleep_story", guides_dir=self.dir)
        self.assertEqual(text, "GUIDE sleep_story kokoro")

    def test_missing_guide_raises_with_a_helpful_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_guide("f5", "haiku", guides_dir=self.dir)
        self.assertIn("vocal_haiku_f5_instructions.md", str(ctx.exception))

    def test_loads_safety_rules(self):
        self.assertEqual(load_safety_rules(guides_dir=self.dir), "SAFETY RULES")

    def test_generator_prompt_includes_guide_and_safety(self):
        prompt = build_generator_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("GUIDE meditation f5", prompt)
        self.assertIn("SAFETY RULES", prompt)

    def test_generator_prompt_states_the_duration_window(self):
        prompt = build_generator_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("5", prompt)
        self.assertIn("7", prompt)

    def test_judge_prompt_includes_guide_and_safety(self):
        prompt = build_judge_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("GUIDE meditation f5", prompt)
        self.assertIn("SAFETY RULES", prompt)

    def test_judge_prompt_demands_the_delimited_output_format(self):
        prompt = build_judge_system_prompt(
            "f5", "meditation", 300.0, 420.0, guides_dir=self.dir
        )
        self.assertIn("<script>", prompt)
        self.assertIn("<changelog>", prompt)

    def test_guides_dir_default_points_at_prompting_guides(self):
        self.assertEqual(GUIDES_DIR.name, "prompting_guides")

    def test_real_guides_exist_on_disk(self):
        # Guards against a rename breaking prompt assembly silently.
        for content_type in ("meditation", "sleep_story"):
            for engine in ("f5", "kokoro"):
                self.assertTrue(
                    (GUIDES_DIR / f"vocal_{content_type}_{engine}_instructions.md").is_file()
                )

    def test_real_safety_rules_exist_on_disk(self):
        self.assertTrue((GUIDES_DIR / "content_safety_rules.md").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.rules'`

- [ ] **Step 3: Write the safety rules document**

Create `docs/prompting_guides/content_safety_rules.md`:

```markdown
# Content Safety Rules — Meditation and Sleep Story Scripts

These rules apply to every generated script, in addition to the engine-specific
formatting guide. They exist because this content is written for people who are
anxious, sleepless, grieving, or overwhelmed. The failure modes below are not
stylistic preferences.

A deterministic linter enforces the hard blocks in code. These rules cover both
those and the judgment calls the linter cannot make.

## Hard blocks — never produce these

1. **No clinical claims.** Never say the practice cures, heals, treats, or
   diagnoses anything, and never position it as a replacement for therapy or
   medication. Describe the practice, never a medical outcome.

2. **No outcome promises.** Never write "you will be completely calm", "this
   will eliminate your stress", or any guarantee about how the listener will
   feel. A listener who does not feel that way reads it as their own failure.

3. **No invalidating instructions.** Never write "don't feel anxious", "stop
   feeling that", or "there's nothing wrong with you". Telling a distressed
   person their feelings are wrong dismisses their experience. Acknowledge
   what is present; do not override it.

4. **No extended breath holds.** Never instruct a hold longer than about seven
   seconds. Long holds are a genuine physical risk for listeners with panic
   disorder or asthma.

5. **No dissociation-adjacent imagery.** Never write "leave your body", "float
   away from yourself", or "you are not your body". This is actively
   contraindicated for trauma survivors. Keep the listener grounded in the
   body and the room.

## Required qualities — always produce these

6. **Invitational, never imperative.** Prefer "you might", "if it feels right",
   "when you're ready", "allow", "see if". Avoid "you must", "you have to",
   "now do".

7. **Permission to opt out, stated early.** Eyes may stay open. Posture may be
   adjusted at any time. Any instruction may be skipped. Say so once near the
   opening rather than assuming compliance.

8. **Choice and control throughout.** Never force eye closing. Never require a
   body scan of a specific area — offer it and allow the listener to move on.
   Trauma-informed practice means the listener stays in charge.

9. **Present-tense sensory grounding over abstraction.** "Feel the weight of
   your hands" lands better than "consider the nature of impermanence".

10. **Acknowledge difficulty without dwelling.** If the prompt names a hard
    feeling, name it once with warmth, then offer somewhere to rest attention.
    Do not analyse it, explain it, or promise to remove it.

## Tone

Warm, unhurried, plain. Short sentences. Concrete images. No jargon, no
Sanskrit unless the prompt asks for it, no spiritual claims, no instructing the
listener about what their experience means.
```

- [ ] **Step 4: Write minimal implementation**

Create `core/script_gen/rules.py`:

```python
"""Assemble system prompts from the on-disk prompting guides.

Guides are read at call time, not import time, so edits to
docs/prompting_guides/ take effect on the next generation with no code change
and no restart.
"""

from pathlib import Path

GUIDES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "prompting_guides"
)

GUIDE_TEMPLATE = "vocal_{content_type}_{engine}_instructions.md"
SAFETY_RULES_FILENAME = "content_safety_rules.md"


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompting guide not found: {path}. Expected it at {path.name} "
            f"inside {path.parent}."
        )
    return path.read_text(encoding="utf-8").strip()


def load_guide(
    engine: str, content_type: str, guides_dir: Path | None = None
) -> str:
    """Load the engine- and content-type-specific formatting guide."""
    root = guides_dir if guides_dir is not None else GUIDES_DIR
    filename = GUIDE_TEMPLATE.format(content_type=content_type, engine=engine)
    return _read(root / filename)


def load_safety_rules(guides_dir: Path | None = None) -> str:
    """Load the mental-health content safety rules shared by both passes."""
    root = guides_dir if guides_dir is not None else GUIDES_DIR
    return _read(root / SAFETY_RULES_FILENAME)


def _duration_clause(target_min_sec: float, target_max_sec: float) -> str:
    return (
        f"The finished audio must run between {target_min_sec / 60:.0f} and "
        f"{target_max_sec / 60:.0f} minutes. Spoken delivery is roughly 95-100 "
        "words per minute and explicit [pause:Xs] markers add their full "
        "duration, so budget both words and silence deliberately."
    )


def build_generator_system_prompt(
    engine: str,
    content_type: str,
    target_min_sec: float,
    target_max_sec: float,
    guides_dir: Path | None = None,
) -> str:
    """System prompt for pass 1 — writing the draft script."""
    return "\n\n".join(
        [
            "You write production-ready scripts for a text-to-speech "
            "meditation pipeline. Your output is consumed directly by a TTS "
            "engine, so it must obey the formatting contract exactly.",
            _duration_clause(target_min_sec, target_max_sec),
            "# Formatting guide\n\n" + load_guide(engine, content_type, guides_dir),
            "# Content safety rules\n\n" + load_safety_rules(guides_dir),
            "Output the script and nothing else. No preamble, no explanation, "
            "no markdown, no surrounding quotes.",
        ]
    )


def build_judge_system_prompt(
    engine: str,
    content_type: str,
    target_min_sec: float,
    target_max_sec: float,
    guides_dir: Path | None = None,
) -> str:
    """System prompt for pass 2 — independent review that revises."""
    return "\n\n".join(
        [
            "You are an independent reviewer of scripts written for a "
            "text-to-speech meditation pipeline. You did not write the draft "
            "you are given. Your job is to return a corrected script, not a "
            "score and not a critique.",
            "Read the draft against the formatting guide and the safety rules "
            "below. Fix every violation you find. Preserve what works — do not "
            "rewrite wholesale when a targeted edit will do.",
            _duration_clause(target_min_sec, target_max_sec),
            "# Formatting guide\n\n" + load_guide(engine, content_type, guides_dir),
            "# Content safety rules\n\n" + load_safety_rules(guides_dir),
            "Respond in exactly this format and nothing else:\n\n"
            "<script>\n"
            "the full corrected script\n"
            "</script>\n"
            "<changelog>\n"
            "- one line per change you made, or 'no changes needed'\n"
            "</changelog>",
        ]
    )
```

Add to `core/script_gen/__init__.py` — extend the import block and `__all__`:

```python
from core.script_gen.rules import (
    build_generator_system_prompt,
    build_judge_system_prompt,
    load_guide,
    load_safety_rules,
)
```

and add `"build_generator_system_prompt"`, `"build_judge_system_prompt"`, `"load_guide"`, `"load_safety_rules"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_script_rules.py -v`
Expected: PASS — 11 tests

- [ ] **Step 6: Run the whole unit suite — Phase 1 gate**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS, including the pre-existing tests. Nothing in Phase 1 touches the audio path.

- [ ] **Step 7: Commit**

```bash
git add docs/prompting_guides/content_safety_rules.md core/script_gen/rules.py core/script_gen/__init__.py tests/unit/test_script_rules.py
git commit -m "feat(script_gen): add safety rules doc and prompt assembly"
```

---

# Phase 2 — Model interface

### Task 6: ScriptEngine ABC and provider registry

**Files:**
- Create: `core/script_gen/engine.py`
- Create: `core/script_gen/adapters/__init__.py`
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_script_engine.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ScriptEngine` ABC with abstract `complete(self, system: str, user: str, *, max_tokens: int = 4096, temperature: float = 1.0) -> str` and abstract property `name -> str`; `PROVIDER_BASE_URLS: dict[str, str]`; `PROVIDER_KEY_ENV: dict[str, str]`; `parse_engine_spec(spec: str) -> tuple[str, str]`; `build_engine(spec: str) -> ScriptEngine`; `FakeScriptEngine` (test double, shipped in the module so both tests and the bench can use it).

Specs look like `ollama:qwen3:30b`, `openrouter:meta-llama/llama-3.3-70b`, `anthropic:claude-opus-5`. Split on the **first** colon only — model names contain colons.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_script_engine.py`:

```python
"""Tests for the ScriptEngine interface and provider registry.

No network calls: build_engine is exercised for construction and error paths
only, and behaviour is tested through FakeScriptEngine.
"""

import unittest

from core.script_gen.engine import (
    PROVIDER_BASE_URLS,
    FakeScriptEngine,
    ScriptEngine,
    build_engine,
    parse_engine_spec,
)


class TestSpecParsing(unittest.TestCase):
    def test_splits_provider_from_model(self):
        self.assertEqual(parse_engine_spec("ollama:llama3.2"), ("ollama", "llama3.2"))

    def test_splits_on_first_colon_only(self):
        # Ollama tags contain colons; they belong to the model name.
        self.assertEqual(
            parse_engine_spec("ollama:qwen3:30b"), ("ollama", "qwen3:30b")
        )

    def test_slashes_in_model_names_survive(self):
        self.assertEqual(
            parse_engine_spec("openrouter:meta-llama/llama-3.3-70b"),
            ("openrouter", "meta-llama/llama-3.3-70b"),
        )

    def test_missing_colon_raises(self):
        with self.assertRaises(ValueError):
            parse_engine_spec("ollama")

    def test_empty_model_raises(self):
        with self.assertRaises(ValueError):
            parse_engine_spec("ollama:")

    def test_unknown_provider_raises_listing_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            build_engine("nosuchprovider:model")
        self.assertIn("ollama", str(ctx.exception))


class TestRegistry(unittest.TestCase):
    def test_openai_compatible_providers_registered(self):
        for provider in ("ollama", "openrouter", "together", "fireworks", "groq"):
            self.assertIn(provider, PROVIDER_BASE_URLS)

    def test_ollama_points_at_localhost(self):
        self.assertIn("localhost", PROVIDER_BASE_URLS["ollama"])

    def test_every_base_url_ends_with_v1(self):
        for url in PROVIDER_BASE_URLS.values():
            self.assertTrue(url.endswith("/v1"), url)


class TestFakeEngine(unittest.TestCase):
    def test_is_a_script_engine(self):
        self.assertIsInstance(FakeScriptEngine(["out"]), ScriptEngine)

    def test_returns_queued_responses_in_order(self):
        engine = FakeScriptEngine(["first", "second"])
        self.assertEqual(engine.complete("sys", "usr"), "first")
        self.assertEqual(engine.complete("sys", "usr"), "second")

    def test_repeats_the_last_response_when_exhausted(self):
        engine = FakeScriptEngine(["only"])
        engine.complete("sys", "usr")
        self.assertEqual(engine.complete("sys", "usr"), "only")

    def test_records_calls_for_assertions(self):
        engine = FakeScriptEngine(["out"])
        engine.complete("SYSTEM", "USER")
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")
        self.assertEqual(engine.calls[0]["user"], "USER")

    def test_has_a_name(self):
        self.assertTrue(FakeScriptEngine(["out"]).name)

    def test_empty_response_list_raises(self):
        with self.assertRaises(ValueError):
            FakeScriptEngine([])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.engine'`

- [ ] **Step 3: Write minimal implementation**

Create `core/script_gen/adapters/__init__.py`:

```python
"""Concrete ScriptEngine adapters."""
```

Create `core/script_gen/engine.py`:

```python
"""ScriptEngine ABC and the provider registry.

Mirrors core/speech_engine.py: one narrow interface, several concrete
engines, and the rest of the system stays engine-agnostic.

Ollama, OpenRouter, Together, Fireworks and Groq all speak the
OpenAI-compatible /v1/chat/completions protocol, so one adapter with a
configurable base_url covers local *and* every hosted open-weight provider.
Only Claude needs its own adapter.
"""

from abc import ABC, abstractmethod

# Every entry here is OpenAI-compatible and served by OpenAICompatEngine.
PROVIDER_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Ollama runs locally and needs no key, so it is absent here by design.
PROVIDER_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class ScriptEngine(ABC):
    """Interface every script-generation backend must implement.

    Deliberately narrow: one text-in, text-out call. Generator, judge and
    repair are all the same operation with different prompts.
    """

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """Run one completion and return the assistant's text.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Ceiling on generated tokens.
            temperature: Sampling temperature.

        Returns:
            The assistant's response as plain text.

        Raises:
            RuntimeError: On an unrecoverable backend failure, with a message
                naming the provider and how to fix it.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'ollama:qwen3:30b'."""


class FakeScriptEngine(ScriptEngine):
    """Test double returning canned responses. No network, no model.

    Lets the generator/judge/repair loop be tested exhaustively without
    spending a token.
    """

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("FakeScriptEngine needs at least one response.")
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    @property
    def name(self) -> str:
        return "fake:canned"


def parse_engine_spec(spec: str) -> tuple[str, str]:
    """Split a 'provider:model' spec.

    Splits on the FIRST colon only — Ollama model tags such as 'qwen3:30b'
    contain colons that belong to the model name.

    Raises:
        ValueError: If the spec has no colon or an empty half.
    """
    if ":" not in spec:
        raise ValueError(
            f"Malformed engine spec {spec!r}. Expected 'provider:model', "
            "e.g. 'ollama:qwen3:30b' or 'anthropic:claude-opus-5'."
        )
    provider, model = spec.split(":", 1)
    provider, model = provider.strip(), model.strip()
    if not provider or not model:
        raise ValueError(
            f"Malformed engine spec {spec!r}. Both provider and model are required."
        )
    return provider, model


def build_engine(spec: str) -> ScriptEngine:
    """Construct the engine named by a 'provider:model' spec."""
    provider, model = parse_engine_spec(spec)

    if provider == "anthropic":
        from core.script_gen.adapters.anthropic_api import AnthropicEngine
        return AnthropicEngine(model)

    if provider in PROVIDER_BASE_URLS:
        from core.script_gen.adapters.openai_compat import OpenAICompatEngine
        return OpenAICompatEngine(
            provider=provider,
            model=model,
            base_url=PROVIDER_BASE_URLS[provider],
            api_key_env=PROVIDER_KEY_ENV.get(provider),
        )

    known = ", ".join(sorted([*PROVIDER_BASE_URLS, "anthropic"]))
    raise ValueError(
        f"Unknown provider {provider!r} in spec {spec!r}. Known providers: {known}."
    )
```

Add to `core/script_gen/__init__.py` — extend the import block and `__all__`:

```python
from core.script_gen.engine import (
    FakeScriptEngine,
    ScriptEngine,
    build_engine,
    parse_engine_spec,
)
```

and add `"FakeScriptEngine"`, `"ScriptEngine"`, `"build_engine"`, `"parse_engine_spec"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_script_engine.py -v`
Expected: PASS — 15 tests

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/engine.py core/script_gen/adapters/__init__.py core/script_gen/__init__.py tests/unit/test_script_engine.py
git commit -m "feat(script_gen): add ScriptEngine ABC and provider registry"
```

---

### Task 7: OpenAI-compatible adapter

**Files:**
- Create: `core/script_gen/adapters/openai_compat.py`
- **Do NOT modify `requirements.txt`.** See the dependency note below.
- Test: `tests/unit/test_openai_compat_adapter.py`

**Interfaces:**
- Consumes: `ScriptEngine` from Task 6.
- Produces: `OpenAICompatEngine(provider: str, model: str, base_url: str, api_key_env: str | None = None, timeout: float = 300.0, transport: httpx.BaseTransport | None = None)`.

The `transport` parameter exists so tests can inject `httpx.MockTransport` and exercise the real request-building and response-parsing code without a network. A 300s default timeout accommodates a local model generating slowly.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_openai_compat_adapter.py`:

```python
"""Tests for the OpenAI-compatible adapter (Ollama + hosted open-weight providers).

Uses httpx.MockTransport so the real request-building and response-parsing
paths run with no network.
"""

import json
import unittest

import httpx

from core.script_gen.adapters.openai_compat import OpenAICompatEngine


def ok_transport(captured: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "GENERATED SCRIPT"}}]},
        )

    return httpx.MockTransport(handler)


def status_transport(code: int, body: str = "boom") -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: httpx.Response(code, text=body))


class TestOpenAICompatEngine(unittest.TestCase):
    def build(self, transport, **kwargs):
        return OpenAICompatEngine(
            provider="ollama",
            model="qwen3:30b",
            base_url="http://localhost:11434/v1",
            transport=transport,
            **kwargs,
        )

    def test_returns_the_message_content(self):
        engine = self.build(ok_transport([]))
        self.assertEqual(engine.complete("sys", "usr"), "GENERATED SCRIPT")

    def test_posts_to_chat_completions(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertEqual(
            str(captured[0].url), "http://localhost:11434/v1/chat/completions"
        )

    def test_sends_system_and_user_messages(self):
        captured = []
        self.build(ok_transport(captured)).complete("SYSTEM", "USER")
        payload = json.loads(captured[0].content)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "USER"})

    def test_sends_the_model_name(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertEqual(json.loads(captured[0].content)["model"], "qwen3:30b")

    def test_forwards_max_tokens_and_temperature(self):
        captured = []
        self.build(ok_transport(captured)).complete(
            "sys", "usr", max_tokens=1234, temperature=0.4
        )
        payload = json.loads(captured[0].content)
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["temperature"], 0.4)

    def test_no_auth_header_without_a_key_env(self):
        captured = []
        self.build(ok_transport(captured)).complete("sys", "usr")
        self.assertNotIn("authorization", captured[0].headers)

    def test_name_is_provider_and_model(self):
        self.assertEqual(self.build(ok_transport([])).name, "ollama:qwen3:30b")

    def test_connection_error_names_the_provider_and_the_fix(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)

        engine = self.build(httpx.MockTransport(refuse))
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        message = str(ctx.exception)
        self.assertIn("ollama", message.lower())
        self.assertIn("11434", message)

    def test_http_error_is_wrapped_with_the_status(self):
        engine = self.build(status_transport(500))
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIn("500", str(ctx.exception))

    def test_missing_api_key_raises_naming_the_env_var(self):
        with self.assertRaises(RuntimeError) as ctx:
            OpenAICompatEngine(
                provider="openrouter",
                model="some/model",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="MOODSCAPE_TEST_ABSENT_KEY",
                transport=ok_transport([]),
            ).complete("sys", "usr")
        self.assertIn("MOODSCAPE_TEST_ABSENT_KEY", str(ctx.exception))

    def test_malformed_response_raises(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"unexpected": True})
        )
        with self.assertRaises(RuntimeError):
            self.build(transport).complete("sys", "usr")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_openai_compat_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.adapters.openai_compat'`

- [ ] **Step 3: Confirm httpx is importable (do NOT edit requirements.txt)**

Run: `.venv/bin/python -c "import httpx; print(httpx.__version__)"`
Expected: a version prints (0.28.1 at time of writing — it is already installed transitively).

**Dependency note:** `httpx` is NOT declared in `requirements.txt`, and this task deliberately does not add it. That file currently holds unrelated uncommitted work, so `git add requirements.txt` would sweep a third party's changes into this branch. The declaration is deferred to a single controller-owned step; see Task 15 Step 2b.

- [ ] **Step 4: Write minimal implementation**

Create `core/script_gen/adapters/openai_compat.py`:

```python
"""Adapter for every OpenAI-compatible /v1/chat/completions endpoint.

One adapter covers local Ollama and the hosted open-weight providers
(OpenRouter, Together, Fireworks, Groq) because they all speak the same
protocol — only the base URL and the API-key env var differ.

Uses httpx directly rather than adding another SDK dependency.
"""

import os

import httpx

from core.script_gen.engine import ScriptEngine

# Generous default: a local 32B model on Apple Silicon can take minutes for a
# few thousand tokens.
DEFAULT_TIMEOUT_SEC = 300.0


class OpenAICompatEngine(ScriptEngine):
    """Talk to any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        transport: httpx.BaseTransport | None = None,
    ):
        self._provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return f"{self._provider}:{self._model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key_env is None:
            return headers
        key = os.environ.get(self._api_key_env)
        if not key:
            raise RuntimeError(
                f"{self._provider} needs an API key but {self._api_key_env} is "
                f"not set. Add {self._api_key_env}=... to your .env."
            )
        headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = self._headers()

        try:
            with httpx.Client(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            hint = (
                "Is `ollama serve` running?"
                if self._provider == "ollama"
                else "Check network access and the provider's status."
            )
            raise RuntimeError(
                f"Cannot reach {self._provider} at {url}. {hint}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"{self._provider} timed out after {self._timeout:.0f}s at {url}. "
                "A large local model may need a longer timeout."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"{self._provider} returned HTTP {exc.response.status_code} "
                f"for model {self._model!r}: {exc.response.text[:400]}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"{self._provider} returned an unexpected response shape: "
                f"{str(data)[:400]}"
            ) from exc
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_openai_compat_adapter.py -v`
Expected: PASS — 11 tests

- [ ] **Step 6: Commit**

```bash
git add core/script_gen/adapters/openai_compat.py tests/unit/test_openai_compat_adapter.py
git commit -m "feat(script_gen): add OpenAI-compatible adapter for local and hosted models"
```

---

### Task 8: Anthropic adapter

**Files:**
- Create: `core/script_gen/adapters/anthropic_api.py`
- **Do NOT modify `requirements.txt`.** See the dependency note below.
- Test: `tests/unit/test_anthropic_adapter.py`

**Interfaces:**
- Consumes: `ScriptEngine` from Task 6.
- Produces: `AnthropicEngine(model: str, api_key_env: str = "ANTHROPIC_API_KEY", client: object | None = None)`.

The `client` parameter allows a stub in tests. Uses `claude-opus-5` semantics: adaptive thinking is the default, `budget_tokens` is rejected by current models, and streaming is used because `max_tokens` is large enough to risk an HTTP timeout otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_anthropic_adapter.py`:

```python
"""Tests for the Anthropic adapter, using a stub client (no network)."""

import unittest

from core.script_gen.adapters.anthropic_api import AnthropicEngine


class StubBlock:
    def __init__(self, text, type_="text"):
        self.text = text
        self.type = type_


class StubMessage:
    def __init__(self, blocks):
        self.content = blocks


class StubStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class StubMessages:
    def __init__(self, message, recorder):
        self._message = message
        self._recorder = recorder

    def stream(self, **kwargs):
        self._recorder.append(kwargs)
        return StubStream(self._message)


class StubClient:
    def __init__(self, blocks):
        self.calls: list[dict] = []
        self.messages = StubMessages(StubMessage(blocks), self.calls)


class TestAnthropicEngine(unittest.TestCase):
    def test_returns_concatenated_text_blocks(self):
        client = StubClient([StubBlock("Hello "), StubBlock("world")])
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.complete("sys", "usr"), "Hello world")

    def test_ignores_thinking_blocks(self):
        client = StubClient(
            [StubBlock("reasoning", type_="thinking"), StubBlock("answer")]
        )
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.complete("sys", "usr"), "answer")

    def test_passes_system_as_top_level_param(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("SYSTEM", "USER")
        self.assertEqual(client.calls[0]["system"], "SYSTEM")

    def test_passes_user_message(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("SYSTEM", "USER")
        self.assertEqual(
            client.calls[0]["messages"], [{"role": "user", "content": "USER"}]
        )

    def test_does_not_send_budget_tokens(self):
        # budget_tokens is rejected with a 400 on current models.
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete("s", "u")
        thinking = client.calls[0].get("thinking", {})
        self.assertNotIn("budget_tokens", thinking)

    def test_forwards_max_tokens(self):
        client = StubClient([StubBlock("out")])
        AnthropicEngine("claude-opus-5", client=client).complete(
            "s", "u", max_tokens=9000
        )
        self.assertEqual(client.calls[0]["max_tokens"], 9000)

    def test_name_is_prefixed(self):
        client = StubClient([StubBlock("out")])
        engine = AnthropicEngine("claude-opus-5", client=client)
        self.assertEqual(engine.name, "anthropic:claude-opus-5")

    def test_missing_key_raises_naming_the_env_var(self):
        engine = AnthropicEngine(
            "claude-opus-5", api_key_env="MOODSCAPE_TEST_ABSENT_KEY"
        )
        with self.assertRaises(RuntimeError) as ctx:
            engine.complete("sys", "usr")
        self.assertIn("MOODSCAPE_TEST_ABSENT_KEY", str(ctx.exception))

    def test_empty_response_raises(self):
        engine = AnthropicEngine("claude-opus-5", client=StubClient([]))
        with self.assertRaises(RuntimeError):
            engine.complete("sys", "usr")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_anthropic_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.adapters.anthropic_api'`

- [ ] **Step 3: Do NOT install or declare `anthropic`**

**Dependency note:** the `anthropic` package is NOT required for this task's tests — every test injects a stub client, and `_get_client()` raises on the missing-API-key check *before* it ever reaches the lazy `import anthropic`. Do not install it and do not add it to `requirements.txt`: that file holds unrelated uncommitted work, so staging it would sweep a third party's changes into this branch. The declaration is deferred to a single controller-owned step; see Task 15 Step 2b.

Verify the import is lazy by confirming the module imports without the package present:
Run: `.venv/bin/python -c "from core.script_gen.adapters.anthropic_api import AnthropicEngine; print('ok')"`
Expected: prints `ok` (after Step 4 creates the file).

- [ ] **Step 4: Write minimal implementation**

Create `core/script_gen/adapters/anthropic_api.py`:

```python
"""Adapter for the Anthropic Messages API.

Separate from openai_compat because the wire protocol differs. Uses the
official anthropic SDK.

Current-model notes: adaptive thinking is the default and budget_tokens is
rejected with a 400, so it is never sent. Streaming is used because large
max_tokens values risk an HTTP timeout on a non-streaming request.
"""

import os

from core.script_gen.engine import ScriptEngine


class AnthropicEngine(ScriptEngine):
    """Talk to Claude via the Messages API."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        client: object | None = None,
    ):
        self._model = model
        self._api_key_env = api_key_env
        self._client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not os.environ.get(self._api_key_env):
            raise RuntimeError(
                f"Claude needs an API key but {self._api_key_env} is not set. "
                f"Add {self._api_key_env}=... to your .env."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The anthropic package is not installed. "
                "Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        client = self._get_client()

        # temperature is deliberately not forwarded: sampling parameters are
        # rejected on current thinking-enabled models.
        try:
            with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic request failed for model {self._model!r}: {exc}"
            ) from exc

        text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        if not text:
            raise RuntimeError(
                f"Anthropic returned no text content for model {self._model!r}."
            )
        return text
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_anthropic_adapter.py -v`
Expected: PASS — 9 tests

- [ ] **Step 6: Commit**

```bash
git add core/script_gen/adapters/anthropic_api.py tests/unit/test_anthropic_adapter.py
git commit -m "feat(script_gen): add Anthropic adapter"
```

---

### Task 9: Generator, judge, and the bounded repair loop

**Files:**
- Create: `core/script_gen/generator.py`
- Create: `core/script_gen/judge.py`
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_generator_judge.py`

**Interfaces:**
- Consumes: `ScriptEngine`, `FakeScriptEngine` (Task 6); `Violation`, `format_for_repair` (Tasks 1–2).
- Produces: `draft(engine: ScriptEngine, prompt: str, system: str, *, max_tokens: int = 4096) -> str`; `parse_judge_response(raw: str) -> tuple[str, str]`; `review(engine: ScriptEngine, draft_script: str, system: str, *, max_tokens: int = 4096) -> tuple[str, str]`; `repair(engine: ScriptEngine, script: str, violations: list[Violation], system: str, *, max_tokens: int = 4096) -> tuple[str, str]`.

`parse_judge_response` must degrade gracefully: a model that ignores the delimiters still produces a usable script rather than failing the run.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_generator_judge.py`:

```python
"""Tests for the two-pass generator/judge flow, against a fake engine."""

import unittest

from core.script_gen.engine import FakeScriptEngine
from core.script_gen.generator import draft
from core.script_gen.judge import parse_judge_response, repair, review
from core.script_gen.linter import FATAL, Violation


class TestGenerator(unittest.TestCase):
    def test_returns_the_engine_output(self):
        engine = FakeScriptEngine(["Breathe in."])
        self.assertEqual(draft(engine, "I feel anxious", "SYS"), "Breathe in.")

    def test_passes_the_system_prompt_through(self):
        engine = FakeScriptEngine(["out"])
        draft(engine, "I feel anxious", "SYSTEM")
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")

    def test_user_message_contains_the_prompt(self):
        engine = FakeScriptEngine(["out"])
        draft(engine, "I feel anxious", "SYS")
        self.assertIn("I feel anxious", engine.calls[0]["user"])

    def test_strips_surrounding_whitespace(self):
        engine = FakeScriptEngine(["\n\n  Breathe in.  \n\n"])
        self.assertEqual(draft(engine, "p", "s"), "Breathe in.")

    def test_strips_a_fenced_code_block(self):
        engine = FakeScriptEngine(["```\nBreathe in.\n```"])
        self.assertEqual(draft(engine, "p", "s"), "Breathe in.")


class TestJudgeParsing(unittest.TestCase):
    def test_extracts_script_and_changelog(self):
        raw = "<script>\nBreathe in.\n</script>\n<changelog>\n- fixed a tag\n</changelog>"
        script, changelog = parse_judge_response(raw)
        self.assertEqual(script, "Breathe in.")
        self.assertIn("fixed a tag", changelog)

    def test_missing_changelog_yields_empty_string(self):
        script, changelog = parse_judge_response("<script>Breathe in.</script>")
        self.assertEqual(script, "Breathe in.")
        self.assertEqual(changelog, "")

    def test_missing_delimiters_falls_back_to_whole_output(self):
        # A model that ignores the format must not fail the run.
        script, changelog = parse_judge_response("Breathe in.")
        self.assertEqual(script, "Breathe in.")
        self.assertEqual(changelog, "")

    def test_tolerates_surrounding_prose(self):
        raw = "Sure!\n<script>\nBreathe in.\n</script>\nHope that helps."
        script, _ = parse_judge_response(raw)
        self.assertEqual(script, "Breathe in.")


class TestReview(unittest.TestCase):
    def test_returns_revised_script_and_changelog(self):
        engine = FakeScriptEngine(
            ["<script>Revised.</script><changelog>- tightened</changelog>"]
        )
        script, changelog = review(engine, "Draft.", "SYS")
        self.assertEqual(script, "Revised.")
        self.assertIn("tightened", changelog)

    def test_draft_is_included_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Revised.</script>"])
        review(engine, "THE DRAFT", "SYS")
        self.assertIn("THE DRAFT", engine.calls[0]["user"])


class TestRepair(unittest.TestCase):
    def test_violations_appear_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [
            Violation(code="MARKER_MALFORMED", severity=FATAL, message="bad tag")
        ]
        repair(engine, "Broken.", violations, "SYS")
        user = engine.calls[0]["user"]
        self.assertIn("MARKER_MALFORMED", user)
        self.assertIn("bad tag", user)

    def test_current_script_appears_in_the_user_message(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [Violation(code="X", severity=FATAL, message="m")]
        repair(engine, "THE BROKEN SCRIPT", violations, "SYS")
        self.assertIn("THE BROKEN SCRIPT", engine.calls[0]["user"])

    def test_returns_the_repaired_script(self):
        engine = FakeScriptEngine(["<script>Fixed.</script>"])
        violations = [Violation(code="X", severity=FATAL, message="m")]
        script, _ = repair(engine, "Broken.", violations, "SYS")
        self.assertEqual(script, "Fixed.")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_generator_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.generator'`

- [ ] **Step 3: Write the generator**

Create `core/script_gen/generator.py`:

```python
"""Pass 1: turn a natural-language prompt into a draft script."""

import re

from core.script_gen.engine import ScriptEngine

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_wrapper(text: str) -> str:
    """Remove a surrounding markdown code fence and outer whitespace.

    Models wrap output in fences despite instructions not to; the fence would
    otherwise be read aloud by the TTS engine.
    """
    match = _FENCE.match(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def draft(
    engine: ScriptEngine,
    prompt: str,
    system: str,
    *,
    max_tokens: int = 4096,
) -> str:
    """Generate a draft script from the user's natural-language prompt."""
    user = (
        "Write a complete script for this request:\n\n"
        f"{prompt}\n\n"
        "Output only the script."
    )
    return strip_wrapper(engine.complete(system, user, max_tokens=max_tokens))
```

- [ ] **Step 4: Write the judge**

Create `core/script_gen/judge.py`:

```python
"""Pass 2: an independent model reviews the draft and returns a revision.

The judge revises rather than scores. A score does not help a fire-and-forget
pipeline; a corrected script does.

The same module handles targeted repair, which is the same operation with the
linter's findings supplied as the instruction.
"""

import re

from core.script_gen.engine import ScriptEngine
from core.script_gen.generator import strip_wrapper
from core.script_gen.linter import Violation, format_for_repair

_SCRIPT_BLOCK = re.compile(r"<script>\s*(.*?)\s*</script>", re.DOTALL | re.IGNORECASE)
_CHANGELOG_BLOCK = re.compile(
    r"<changelog>\s*(.*?)\s*</changelog>", re.DOTALL | re.IGNORECASE
)


def parse_judge_response(raw: str) -> tuple[str, str]:
    """Split the judge's response into (script, changelog).

    Degrades gracefully: a model that ignores the delimiters still yields a
    usable script rather than failing the run.
    """
    script_match = _SCRIPT_BLOCK.search(raw)
    changelog_match = _CHANGELOG_BLOCK.search(raw)

    script = script_match.group(1) if script_match else raw
    changelog = changelog_match.group(1).strip() if changelog_match else ""

    return strip_wrapper(script), changelog


def review(
    engine: ScriptEngine,
    draft_script: str,
    system: str,
    *,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Review a draft and return (revised_script, changelog)."""
    user = (
        "Review this draft script and return a corrected version.\n\n"
        "<draft>\n"
        f"{draft_script}\n"
        "</draft>"
    )
    return parse_judge_response(engine.complete(system, user, max_tokens=max_tokens))


def repair(
    engine: ScriptEngine,
    script: str,
    violations: list[Violation],
    system: str,
    *,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    """Fix specific, named violations. Returns (repaired_script, changelog)."""
    user = (
        "An automated checker found the following problems in this script. "
        "Fix every one of them and return the corrected script. Change nothing "
        "else.\n\n"
        "<problems>\n"
        f"{format_for_repair(violations)}\n"
        "</problems>\n\n"
        "<script>\n"
        f"{script}\n"
        "</script>"
    )
    return parse_judge_response(engine.complete(system, user, max_tokens=max_tokens))
```

Add to `core/script_gen/__init__.py` — extend the import block and `__all__`:

```python
from core.script_gen.generator import draft, strip_wrapper
from core.script_gen.judge import parse_judge_response, repair, review
```

and add `"draft"`, `"strip_wrapper"`, `"parse_judge_response"`, `"repair"`, `"review"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_generator_judge.py -v`
Expected: PASS — 14 tests

- [ ] **Step 6: Run the whole unit suite — Phase 2 gate**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/script_gen/generator.py core/script_gen/judge.py core/script_gen/__init__.py tests/unit/test_generator_judge.py
git commit -m "feat(script_gen): add generator, judge, and targeted repair"
```

---

# Phase 3 — Orchestration

### Task 10: Orchestrator

**Files:**
- Create: `core/auto_generate.py`
- Test: `tests/unit/test_auto_generate.py`

**Interfaces:**
- Consumes: everything from Phases 1–2, plus `MeditationPipeline` (called, never modified).
- Produces: `AutoConfig` dataclass; `AutoResult` dataclass; `generate_script(prompt, *, generator_engine, judge_engine, config) -> ScriptOutcome`; `run(prompt, *, config=None, pipeline=None, generator_engine=None, judge_engine=None, rng=None, progress_cb=None) -> AutoResult`; `ScriptGenerationError`.

The severity split from the spec lives here: fatal violations surviving the repair budget abort before rendering; advisory ones are carried into metadata and the render proceeds.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auto_generate.py`:

```python
"""Tests for the auto-generation orchestrator.

Uses fake engines and a stub pipeline — no model, no network, no audio.
"""

import json
import tempfile
import unittest
from pathlib import Path

from core.auto_generate import (
    AutoConfig,
    ScriptGenerationError,
    generate_script,
    run,
)
from core.script_gen.engine import FakeScriptEngine

CLEAN_SCRIPT = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:5s]\n\n"
    "If it feels right, you might let your eyes close.\n\n"
    "[pause:5s]\n\n"
    "Notice the weight of your hands."
)

UNSAFE_SCRIPT = "This meditation will cure your anxiety.\n\n[pause:5s]\n\nRest now."

BROKEN_SCRIPT = "Breathe in. [pause:4] Breathe out."


def judged(script: str, changelog: str = "- none") -> str:
    return f"<script>\n{script}\n</script>\n<changelog>\n{changelog}\n</changelog>"


class StubPipeline:
    """Stands in for MeditationPipeline; records the kwargs it was called with."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        wav = self.out_dir / "meditation.wav"
        wav.write_bytes(b"RIFF")
        return str(wav), "ok"


class TestScriptGeneration(unittest.TestCase):
    def setUp(self):
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_clean_script_passes_first_time(self):
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
            config=self.config,
        )
        self.assertEqual(outcome.script, CLEAN_SCRIPT)
        self.assertEqual(outcome.repairs_used, 0)

    def test_judge_revision_is_what_gets_used(self):
        revised = CLEAN_SCRIPT + "\n\nAnd rest."
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(revised, "- added a closing")]),
            config=self.config,
        )
        self.assertEqual(outcome.script, revised)
        self.assertIn("added a closing", outcome.changelog)

    def test_broken_script_is_repaired_and_accepted(self):
        judge = FakeScriptEngine([judged(BROKEN_SCRIPT), judged(CLEAN_SCRIPT)])
        outcome = generate_script(
            "I feel anxious",
            generator_engine=FakeScriptEngine([BROKEN_SCRIPT]),
            judge_engine=judge,
            config=self.config,
        )
        self.assertEqual(outcome.script, CLEAN_SCRIPT)
        self.assertEqual(outcome.repairs_used, 1)

    def test_persistent_safety_violation_raises(self):
        judge = FakeScriptEngine([judged(UNSAFE_SCRIPT)])
        with self.assertRaises(ScriptGenerationError) as ctx:
            generate_script(
                "I feel anxious",
                generator_engine=FakeScriptEngine([UNSAFE_SCRIPT]),
                judge_engine=judge,
                config=self.config,
            )
        self.assertIn("CLINICAL_CLAIM", str(ctx.exception))

    def test_repair_budget_is_respected(self):
        judge = FakeScriptEngine([judged(BROKEN_SCRIPT)])
        config = AutoConfig(
            target_min_sec=1.0, target_max_sec=100000.0, max_repairs=2
        )
        with self.assertRaises(ScriptGenerationError):
            generate_script(
                "p",
                generator_engine=FakeScriptEngine([BROKEN_SCRIPT]),
                judge_engine=judge,
                config=config,
            )
        # 1 review + 2 repairs = 3 judge calls.
        self.assertEqual(len(judge.calls), 3)

    def test_advisory_violation_does_not_raise(self):
        # Duration far outside the window is advisory only.
        config = AutoConfig(target_min_sec=100000.0, target_max_sec=200000.0)
        outcome = generate_script(
            "p",
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)] * 5),
            config=config,
        )
        self.assertTrue(
            any(v.code == "DURATION_OUT_OF_WINDOW" for v in outcome.violations)
        )


FAKE_SCAN = lambda: [("Healing Forest — 23:12", "/bg/forest.mp3")]


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.config = AutoConfig(
            target_min_sec=1.0,
            target_max_sec=100000.0,
            background_scan=FAKE_SCAN,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, pipeline):
        return run(
            "I feel anxious",
            config=self.config,
            pipeline=pipeline,
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )

    def test_returns_the_pipeline_audio_path(self):
        pipeline = StubPipeline(self.dir)
        result = self._run(pipeline)
        self.assertTrue(Path(result.audio_path).is_file())

    def test_writes_script_and_meta_siblings(self):
        result = self._run(StubPipeline(self.dir))
        self.assertTrue(Path(result.script_path).is_file())
        self.assertTrue(Path(result.meta_path).is_file())
        self.assertEqual(
            Path(result.script_path).parent, Path(result.audio_path).parent
        )

    def test_script_file_holds_the_final_script(self):
        result = self._run(StubPipeline(self.dir))
        self.assertEqual(Path(result.script_path).read_text(), CLEAN_SCRIPT)

    def test_meta_records_prompt_models_and_background(self):
        result = self._run(StubPipeline(self.dir))
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["prompt"], "I feel anxious")
        self.assertIn("generator", meta)
        self.assertIn("judge", meta)
        self.assertEqual(meta["background"], "Healing Forest — 23:12")
        self.assertEqual(meta["background_path"], "/bg/forest.mp3")

    def test_pipeline_receives_the_script_and_background(self):
        pipeline = StubPipeline(self.dir)
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["script"], CLEAN_SCRIPT)
        self.assertEqual(call["uploaded_music_path"], "/bg/forest.mp3")

    def test_pipeline_uses_the_golden_path_defaults(self):
        pipeline = StubPipeline(self.dir)
        self._run(pipeline)
        call = pipeline.calls[0]
        self.assertEqual(call["tts_engine"], "f5")
        self.assertEqual(call["music_model"], "upload")

    def test_artifacts_are_written_when_script_generation_fails(self):
        pipeline = StubPipeline(self.dir)
        self.config.failure_dir = self.dir / "failures"
        with self.assertRaises(ScriptGenerationError):
            run(
                "I feel anxious",
                config=self.config,
                pipeline=pipeline,
                generator_engine=FakeScriptEngine([UNSAFE_SCRIPT]),
                judge_engine=FakeScriptEngine([judged(UNSAFE_SCRIPT)] * 5),
            )
        # Nothing rendered, but the failed script is on disk to read.
        self.assertEqual(pipeline.calls, [])
        failures = list(self.config.failure_dir.glob("*.script.txt"))
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_auto_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.auto_generate'`

- [ ] **Step 3: Write minimal implementation**

Create `core/auto_generate.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_auto_generate.py -v`
Expected: PASS — 13 tests

- [ ] **Step 5: Commit**

```bash
git add core/auto_generate.py tests/unit/test_auto_generate.py
git commit -m "feat(auto): add prompt-to-meditation orchestrator"
```

---

### Task 11: Gradio Auto-Generate tab

**Files:**
- Create: `core/streaming_run.py`
- Modify: `app.py` (add a tab; do not alter the existing manual tab)
- Test: `tests/unit/test_streaming_run.py`

**Interfaces:**
- Consumes: `run`, `AutoConfig`, `ScriptGenerationError`, `AutoResult` from Task 10.
- Produces: `ProgressUpdate` dataclass (`fraction: float`, `message: str`); `StreamingRun(prompt, *, config=None, runner=None, **kwargs)` — iterating it yields `ProgressUpdate`s while work happens on a background thread, after which `.result` holds an `AutoResult` or `.error` holds a message string.

**Why the extra module:** `app.py` cannot be imported in a test — it pulls in torch and Gradio and registers `atexit.register(lambda: os._exit(0))`, which would hijack pytest's exit. Keeping the thread-and-queue logic in `core/` puts it under test and leaves the Gradio handler a thin formatting shim. The pattern itself is copied from the manual handler at `app.py:278`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_streaming_run.py`:

```python
"""Tests for UI-independent streaming orchestration.

Never imports app.py: that module loads torch and Gradio and registers an
atexit hard-exit hook.
"""

import unittest

from core.auto_generate import AutoResult, ScriptGenerationError
from core.streaming_run import ProgressUpdate, StreamingRun


def make_result():
    return AutoResult(
        audio_path="/out/m.wav",
        script_path="/out/m.script.txt",
        meta_path="/out/m.meta.json",
        script="Breathe in.",
        changelog="- none",
        background="Healing Forest — 23:12",
        violations=[],
        estimated_sec=330.0,
    )


class TestStreamingRun(unittest.TestCase):
    def test_yields_progress_updates(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            progress_cb(0.5, "halfway")
            return make_result()

        run = StreamingRun("p", runner=runner)
        updates = list(run)
        self.assertTrue(any(isinstance(u, ProgressUpdate) for u in updates))
        self.assertIn("halfway", [u.message for u in updates])

    def test_result_is_available_after_iteration(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            return make_result()

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertEqual(run.result.audio_path, "/out/m.wav")
        self.assertIsNone(run.error)

    def test_script_generation_error_is_captured_not_raised(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            raise ScriptGenerationError("CLINICAL_CLAIM survived repairs")

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertIsNone(run.result)
        self.assertIn("CLINICAL_CLAIM", run.error)

    def test_unexpected_error_is_captured_with_its_type(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            raise ValueError("bad thing")

        run = StreamingRun("p", runner=runner)
        list(run)
        self.assertIn("ValueError", run.error)
        self.assertIn("bad thing", run.error)

    def test_prompt_is_forwarded(self):
        seen = {}

        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            seen["prompt"] = prompt
            return make_result()

        list(StreamingRun("I feel anxious", runner=runner))
        self.assertEqual(seen["prompt"], "I feel anxious")

    def test_blank_prompt_errors_without_running(self):
        called = []

        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            called.append(True)
            return make_result()

        run = StreamingRun("   ", runner=runner)
        list(run)
        self.assertEqual(called, [])
        self.assertIn("prompt", run.error.lower())

    def test_iteration_terminates_even_with_no_progress_calls(self):
        def runner(prompt, *, config=None, progress_cb=None, **kwargs):
            return make_result()

        self.assertIsInstance(list(StreamingRun("p", runner=runner)), list)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_streaming_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.streaming_run'`

- [ ] **Step 3: Write minimal implementation**

Create `core/streaming_run.py`:

```python
"""Run an auto-generation on a background thread, streaming progress.

Lives in core/ rather than app.py so it can be unit-tested: app.py loads
torch and Gradio and registers an atexit hard-exit hook, so importing it from
a test is not viable.

The thread-and-queue pattern mirrors the manual handler in app.py.
"""

import queue
import threading
from dataclasses import dataclass

from core.auto_generate import AutoResult, ScriptGenerationError
from core.auto_generate import run as default_run


@dataclass(frozen=True)
class ProgressUpdate:
    """One progress tick from the pipeline."""

    fraction: float
    message: str


class StreamingRun:
    """Iterate for progress; read .result or .error when iteration ends."""

    def __init__(self, prompt: str, *, config=None, runner=None, **kwargs):
        self._prompt = prompt
        self._config = config
        self._runner = runner if runner is not None else default_run
        self._kwargs = kwargs
        self.result: AutoResult | None = None
        self.error: str | None = None

    def __iter__(self):
        if not self._prompt or not self._prompt.strip():
            self.error = "Enter a prompt first."
            return

        updates: queue.Queue = queue.Queue()

        def progress_cb(fraction, message):
            updates.put(ProgressUpdate(fraction=fraction, message=message))

        def worker():
            try:
                self.result = self._runner(
                    self._prompt,
                    config=self._config,
                    progress_cb=progress_cb,
                    **self._kwargs,
                )
            except ScriptGenerationError as exc:
                self.error = str(exc)
            except Exception as exc:  # adapter or pipeline failure
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = updates.get()
            if item is None:
                break
            yield item

        thread.join()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_streaming_run.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Read the existing tab structure**

Run: `grep -n "gr.Tab\|with gr.Blocks\|def run_pipeline\|update_queue" app.py`
Read the surrounding 40 lines of each hit. The new tab must follow the same construction and the same progress-streaming generator pattern.

- [ ] **Step 6: Add the imports**

Near the other `core` imports in `app.py`:

```python
from core.auto_generate import AutoConfig
from core.streaming_run import StreamingRun
```

- [ ] **Step 7: Add the handler**

A thin shim: build the config, iterate for progress, format the outcome. All
the real logic is in `core/streaming_run.py`, under test.

```python
def auto_generate_handler(
    prompt,
    content_type,
    tts_engine,
    target_min_min,
    target_max_min,
    generator_spec,
    judge_spec,
):
    """Prompt -> finished meditation, streaming progress to the UI."""
    os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = generator_spec
    os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge_spec

    config = AutoConfig(
        content_type=content_type,
        tts_engine=tts_engine,
        target_min_sec=float(target_min_min) * 60.0,
        target_max_sec=float(target_max_min) * 60.0,
    )

    run = StreamingRun(prompt, config=config)
    for update in run:
        yield None, "", "", update.message

    if run.error:
        yield None, "", "", f"Failed: {run.error}"
        return

    result = run.result
    status = (
        f"Done. Background: {result.background}. "
        f"Estimated {result.estimated_sec / 60:.1f} min."
    )
    advisories = "\n".join(f"- [{v.code}] {v.message}" for v in result.violations)
    if advisories:
        status = f"{status}\n\nAdvisories:\n{advisories}"

    yield result.audio_path, result.script, result.changelog, status
```

- [ ] **Step 8: Add the tab**

Inside the existing `gr.Blocks` context, after the current tab:

```python
    with gr.Tab("Auto-Generate"):
        gr.Markdown(
            "Describe how you feel. A script is written, independently "
            "reviewed, checked, and rendered with a random background track — "
            "no further input needed."
        )
        auto_prompt = gr.Textbox(
            label="What do you need?",
            placeholder="I'm feeling anxious. I need a relaxing meditation.",
            lines=3,
        )
        with gr.Row():
            auto_content_type = gr.Dropdown(
                choices=["meditation", "sleep_story"],
                value="meditation",
                label="Content Type",
            )
            auto_tts_engine = gr.Dropdown(
                choices=["f5", "kokoro"], value="f5", label="Voice Engine"
            )
        with gr.Row():
            auto_min = gr.Slider(1, 20, value=5, step=1, label="Min minutes")
            auto_max = gr.Slider(1, 30, value=7, step=1, label="Max minutes")
        with gr.Row():
            auto_generator = gr.Textbox(
                label="Generator model",
                value=os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", "ollama:llama3.2:3b"),
            )
            auto_judge = gr.Textbox(
                label="Judge model",
                value=os.environ.get("MOODSCAPE_SCRIPT_JUDGE", "ollama:llama3.2:3b"),
            )
        auto_button = gr.Button("Generate", variant="primary")
        auto_audio = gr.Audio(label="Result", type="filepath")
        auto_status = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Accordion("Script", open=False):
            auto_script = gr.Textbox(label="Final script", lines=20, interactive=False)
        with gr.Accordion("Judge changelog", open=False):
            auto_changelog = gr.Textbox(label="Changes", lines=8, interactive=False)

        auto_button.click(
            fn=auto_generate_handler,
            inputs=[
                auto_prompt,
                auto_content_type,
                auto_tts_engine,
                auto_min,
                auto_max,
                auto_generator,
                auto_judge,
            ],
            outputs=[auto_audio, auto_script, auto_changelog, auto_status],
        )
```

- [ ] **Step 9: Verify the app still starts**

Run: `.venv/bin/python -c "import app"`
Expected: no exception. (Do not launch the server in this step — importing proves the layout is syntactically valid and the handler resolves.)

- [ ] **Step 10: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS

- [ ] **Step 11: Manual smoke test**

Run: `ollama pull llama3.2:3b` (if not already present), then `.venv/bin/python app.py`
Open http://localhost:7860, select the Auto-Generate tab, enter "I'm feeling anxious and need to unwind", and click Generate.

Expected: progress messages stream; either a rendered WAV appears with the script and changelog populated, or a clear failure message naming the violation codes. `llama3.2:3b` is small enough that the script may well fail the linter — that is a *successful* test of the repair-and-fail path, not a bug.

- [ ] **Step 12: Commit**

```bash
git add core/streaming_run.py tests/unit/test_streaming_run.py app.py
git commit -m "feat(ui): add Auto-Generate tab and testable streaming runner"
```

---

### Task 12: Benchmark harness

**Files:**
- Create: `core/script_gen/bench.py`
- Create: `scripts/bench_script_models.py`
- Modify: `core/script_gen/__init__.py` (extend exports)
- Test: `tests/unit/test_script_bench.py`

**Interfaces:**
- Consumes: `generate_script`, `AutoConfig`, `ScriptGenerationError` (Task 10); `build_engine`, `FakeScriptEngine` (Task 6).
- Produces: `BENCH_PROMPTS: list[str]`; `BenchRow` dataclass; `run_bench(pairings: list[tuple[str, str]], *, prompts=None, config=None, engine_factory=None) -> list[BenchRow]`; `format_bench_table(rows: list[BenchRow]) -> str`.

This is the tool that answers "which model" with evidence instead of argument.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_script_bench.py`:

```python
"""Tests for the model benchmark harness, using fake engines."""

import unittest

from core.auto_generate import AutoConfig
from core.script_gen.bench import (
    BENCH_PROMPTS,
    BenchRow,
    format_bench_table,
    run_bench,
)
from core.script_gen.engine import FakeScriptEngine

CLEAN = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:5s]\n\n"
    "Notice the weight of your hands."
)
UNSAFE = "This will cure your anxiety.\n\n[pause:5s]\n\nRest."


def judged(script):
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


class TestBench(unittest.TestCase):
    def setUp(self):
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_prompt_set_covers_several_moods(self):
        self.assertGreaterEqual(len(BENCH_PROMPTS), 5)

    def test_passing_model_is_recorded_as_passed(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("fake:gen", "fake:judge")],
            prompts=["I feel anxious"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].passed)

    def test_failing_model_is_recorded_with_the_error(self):
        def factory(spec):
            return FakeScriptEngine([UNSAFE, judged(UNSAFE)] * 40)

        rows = run_bench(
            [("fake:gen", "fake:judge")],
            prompts=["I feel anxious"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertFalse(rows[0].passed)
        self.assertIn("CLINICAL_CLAIM", rows[0].error)

    def test_one_row_per_pairing_and_prompt(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("a:1", "b:1"), ("c:1", "d:1")],
            prompts=["p1", "p2"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertEqual(len(rows), 4)

    def test_row_records_timing_and_duration(self):
        def factory(spec):
            return FakeScriptEngine([CLEAN, judged(CLEAN)] * 40)

        rows = run_bench(
            [("a:1", "b:1")],
            prompts=["p"],
            config=self.config,
            engine_factory=factory,
        )
        self.assertGreaterEqual(rows[0].elapsed_sec, 0.0)
        self.assertGreater(rows[0].estimated_sec, 0.0)

    def test_table_has_a_header_and_a_row_per_result(self):
        row = BenchRow(
            generator="a:1",
            judge="b:1",
            prompt="p",
            passed=True,
            estimated_sec=330.0,
            repairs_used=0,
            elapsed_sec=1.5,
            advisories=0,
            error="",
        )
        table = format_bench_table([row])
        self.assertIn("| Generator |", table)
        self.assertIn("a:1", table)

    def test_table_handles_no_rows(self):
        self.assertIn("| Generator |", format_bench_table([]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_script_bench.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.script_gen.bench'`

- [ ] **Step 3: Write minimal implementation**

Create `core/script_gen/bench.py`:

```python
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
```

Create `scripts/bench_script_models.py`:

```python
"""CLI: benchmark script-generation model pairings.

Usage:
    python scripts/bench_script_models.py \
        --pair ollama:qwen3:30b ollama:gemma3:27b \
        --pair anthropic:claude-opus-5 anthropic:claude-sonnet-5 \
        --out bench_results.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_generate import AutoConfig  # noqa: E402
from core.script_gen.bench import (  # noqa: E402
    BENCH_PROMPTS,
    format_bench_table,
    run_bench,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("GENERATOR", "JUDGE"),
        required=True,
        help="A generator and judge spec, e.g. --pair ollama:qwen3:30b anthropic:claude-opus-5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(BENCH_PROMPTS),
        help="Use only the first N benchmark prompts.",
    )
    parser.add_argument("--out", type=Path, default=Path("bench_results.md"))
    args = parser.parse_args()

    pairings = [tuple(p) for p in args.pair]
    rows = run_bench(
        pairings,
        prompts=BENCH_PROMPTS[: args.limit],
        config=AutoConfig.from_env(),
    )

    table = format_bench_table(rows)
    args.out.write_text(table + "\n", encoding="utf-8")
    print(table)

    failures = [r for r in rows if not r.passed]
    print(f"\n{len(rows) - len(failures)}/{len(rows)} passed.")
    for row in failures:
        print(f"  FAIL {row.generator} -> {row.judge}: {row.error[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `core/script_gen/__init__.py` — extend the import block and `__all__`:

```python
from core.script_gen.bench import (
    BENCH_PROMPTS,
    BenchRow,
    format_bench_table,
    run_bench,
)
```

and add `"BENCH_PROMPTS"`, `"BenchRow"`, `"format_bench_table"`, `"run_bench"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_script_bench.py -v`
Expected: PASS — 7 tests

- [ ] **Step 5: Add the opt-in live integration test**

Create `tests/integration/test_script_gen_live.py`:

```python
"""Opt-in live test: hits a real configured model.

Excluded from the default run. Enable with:
    MOODSCAPE_LIVE_SCRIPT_TEST=1 .venv/bin/python -m pytest \
        tests/integration/test_script_gen_live.py -v

Requires MOODSCAPE_SCRIPT_GENERATOR and MOODSCAPE_SCRIPT_JUDGE to name
reachable models (e.g. a running `ollama serve`).
"""

import os
import unittest

from core.auto_generate import AutoConfig, generate_script
from core.script_gen.engine import build_engine

LIVE = os.environ.get("MOODSCAPE_LIVE_SCRIPT_TEST") == "1"


@unittest.skipUnless(LIVE, "set MOODSCAPE_LIVE_SCRIPT_TEST=1 to run")
class TestLiveScriptGeneration(unittest.TestCase):
    def test_configured_models_produce_a_passing_script(self):
        generator = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", "ollama:llama3.2:3b")
        )
        judge = build_engine(
            os.environ.get("MOODSCAPE_SCRIPT_JUDGE", "ollama:llama3.2:3b")
        )
        outcome = generate_script(
            "I'm feeling anxious and need to unwind.",
            generator_engine=generator,
            judge_engine=judge,
            config=AutoConfig(),
        )
        self.assertTrue(outcome.script.strip())
        self.assertLessEqual(outcome.repairs_used, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Verify it skips by default**

Run: `.venv/bin/python -m pytest tests/integration/test_script_gen_live.py -v`
Expected: SKIPPED — the default run must never depend on a model being up.

- [ ] **Step 7: Run the whole unit suite — Phase 3 gate**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 8: Commit**

```bash
git add core/script_gen/bench.py scripts/bench_script_models.py core/script_gen/__init__.py tests/unit/test_script_bench.py tests/integration/test_script_gen_live.py
git commit -m "feat(script_gen): add model benchmark harness and live integration test"
```

---

---

# Phase 4 — Integration, documentation, delivery

### Task 13: End-to-end integration tests

**Files:**
- Create: `tests/integration/test_auto_generate_e2e.py`
- Test: itself

**Interfaces:**
- Consumes: `run`, `AutoConfig` (Task 10); `FakeScriptEngine` (Task 6); the real `MeditationPipeline`.
- Produces: nothing importable.

These are **slow** — they run a real render through the real pipeline, which takes minutes on an M1 Max. That is the point: every other test stubs the pipeline, so nothing yet proves the orchestrator's kwargs actually satisfy `MeditationPipeline.generate()`. A signature drift would otherwise only surface in the UI.

Script generation is faked so the test is deterministic and needs no model; the audio path is real.

- [ ] **Step 1: Check how existing integration tests are gated**

Run: `sed -n '1,40p' tests/integration/test_integration_modes.py`
Match whatever skip/marker convention is already there. If none exists, use the env-var gate below.

- [ ] **Step 2: Write the test**

Create `tests/integration/test_auto_generate_e2e.py`:

```python
"""End-to-end: fake script models, real audio pipeline.

Slow — renders actual audio. Run with:
    MOODSCAPE_E2E=1 .venv/bin/python -m pytest \
        tests/integration/test_auto_generate_e2e.py -v

Everything else stubs the pipeline, so this is the only test that proves the
orchestrator's kwargs actually satisfy MeditationPipeline.generate().
"""

import json
import os
import unittest
from pathlib import Path

import soundfile as sf

from core.auto_generate import AutoConfig, run
from core.script_gen.engine import FakeScriptEngine

E2E = os.environ.get("MOODSCAPE_E2E") == "1"

# Short on purpose: a full 5-7 minute render would make this unusable.
SHORT_SCRIPT = (
    "Settle in and let your shoulders drop.\n\n"
    "[pause:3s]\n\n"
    "Notice the weight of your hands.\n\n"
    "[pause:3s]\n\n"
    "And when you are ready, let your eyes open."
)


def judged(script):
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


@unittest.skipUnless(E2E, "set MOODSCAPE_E2E=1 to run (renders real audio)")
class TestAutoGenerateEndToEnd(unittest.TestCase):
    def setUp(self):
        # Wide window: this deliberately short script is nowhere near 5 minutes.
        self.config = AutoConfig(target_min_sec=1.0, target_max_sec=100000.0)

    def test_produces_a_playable_wav_and_its_siblings(self):
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )

        audio = Path(result.audio_path)
        self.assertTrue(audio.is_file())
        self.assertGreater(audio.stat().st_size, 1000)

        info = sf.info(str(audio))
        self.assertGreater(info.duration, 5.0)

        self.assertTrue(Path(result.script_path).is_file())
        self.assertTrue(Path(result.meta_path).is_file())

    def test_metadata_records_the_real_run(self):
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["tts_engine"], "f5")
        self.assertTrue(meta["background"])
        self.assertTrue(Path(meta["background_path"]).is_file())

    def test_duration_estimate_is_within_thirty_percent_of_actual(self):
        """Guards DEFAULT_WPM against drift.

        Loose on purpose: F5 renders are not deterministic, and this exists to
        catch a badly wrong constant, not to pin an exact number.
        """
        result = run(
            "I feel anxious and need to unwind.",
            config=self.config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )
        actual = sf.info(result.audio_path).duration
        ratio = actual / result.estimated_sec
        self.assertGreater(ratio, 0.7, f"estimate far too long: {ratio:.2f}")
        self.assertLess(ratio, 1.3, f"estimate far too short: {ratio:.2f}")

    def test_kokoro_path_also_renders(self):
        config = AutoConfig(
            target_min_sec=1.0, target_max_sec=100000.0, tts_engine="kokoro"
        )
        result = run(
            "I feel anxious and need to unwind.",
            config=config,
            generator_engine=FakeScriptEngine([SHORT_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(SHORT_SCRIPT)]),
        )
        self.assertTrue(Path(result.audio_path).is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Verify it skips by default**

Run: `.venv/bin/python -m pytest tests/integration/test_auto_generate_e2e.py -v`
Expected: SKIPPED — 4 tests.

- [ ] **Step 4: Run it for real, once**

Run: `MOODSCAPE_E2E=1 .venv/bin/python -m pytest tests/integration/test_auto_generate_e2e.py -v`
Expected: PASS. Takes several minutes — it renders four meditations.

If `test_duration_estimate_is_within_thirty_percent_of_actual` fails, that is a **real finding**, not a flaky test: it means `DEFAULT_WPM` in `core/script_gen/duration.py` is wrong. Record the ratio from the failure, adjust the constant, and re-run.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_auto_generate_e2e.py
git commit -m "test(auto): add end-to-end integration tests through the real pipeline"
```

---

### Task 14: Documentation

**Files:**
- Create: `docs/auto_generation/README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/COMPONENT_REGISTRY.md`
- Modify: `docs/TASK_ROUTING.md`
- Modify: `docs/GOTCHAS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the finished implementation.
- Produces: no code.

**Merge-conflict warning — read before starting.** A separate branch is correcting the "36 GB" figure to 32 GB in `CLAUDE.md` (lines 3 and 61) and `docs/ARCHITECTURE.md` (line 318). **Before editing either file, run `git fetch origin && git log --oneline origin/dev -5` to see whether that fix has landed.** If it has, rebase `dev-automate` onto `origin/dev` first. If it has not, do not touch those specific lines, and do not "helpfully" fix the RAM number here — leave it to the other branch so the two changes stay separable.

- [ ] **Step 1: Write the subsystem guide**

Create `docs/auto_generation/README.md` covering, with real content and no placeholders:

- **What it does** — prompt in, finished meditation out, no human step.
- **The four layers** — prose rules (LLM-enforced) vs linter (code-enforced), generator, judge, validation. Explain *why* the split exists: format rules should never cost a token, and a deterministic backstop is what makes a weaker or cheaper model viable.
- **Configuration table** — every env var with its default:
  `MOODSCAPE_SCRIPT_GENERATOR`, `MOODSCAPE_SCRIPT_JUDGE`,
  `MOODSCAPE_SCRIPT_MAX_REPAIRS`, `MOODSCAPE_TARGET_MIN_SEC`,
  `MOODSCAPE_TARGET_MAX_SEC`, plus the per-provider key vars from
  `PROVIDER_KEY_ENV`.
- **Model spec format** — `provider:model`, the five OpenAI-compatible providers plus `anthropic`, and the first-colon-only parsing rule with `ollama:qwen3:30b` as the worked example.
- **The failure severity table** — copied from the spec, since this is the load-bearing design decision. Fatal blocks the render; advisory warns and proceeds.
- **Violation code reference** — every code the linter can emit, what triggers it, and its severity.
- **How to choose a model** — the `scripts/bench_script_models.py` workflow with a runnable command.
- **Calibrating `DEFAULT_WPM`** — what `log_estimate_accuracy` writes and how to act on it.

- [ ] **Step 2: Update CLAUDE.md**

Add to the Folder Map (respecting the RAM-fix warning above):

```
│   ├── script_gen/                    # prompt → validated script (generator + judge + linter)
│   ├── auto_generate.py               # orchestrator: script → music → pipeline
│   ├── background_picker.py           # random pick from assets/backgrounds/
│   └── streaming_run.py               # threaded progress streaming for the UI
```

Add a new section after "Pipeline Flow":

```markdown
## Auto-Generation Flow (`core/auto_generate.py :: run()`)

Prompt in, finished meditation out, with no human step.

1. **Assemble prompts** → `script_gen/rules.py` reads the engine- and
   content-type-specific guide from `docs/prompting_guides/` at call time,
   plus `content_safety_rules.md`
2. **Draft** → generator model (`MOODSCAPE_SCRIPT_GENERATOR`)
3. **Review** → an *independent* judge model (`MOODSCAPE_SCRIPT_JUDGE`)
   returns a revised script plus a changelog — it revises, it does not score
4. **Validate** → `script_gen/linter.py` (format + mental-health safety) and
   `script_gen/duration.py` (runtime estimate, no rendering)
5. **Repair** → fatal violations go back to the judge as targeted
   instructions, bounded by `MOODSCAPE_SCRIPT_MAX_REPAIRS` (default 2)
6. **Pick music** → `background_picker.pick_background()` reuses
   `upload_music.scan_backgrounds()`, excluding recently used tracks
7. **Render** → `MeditationPipeline.generate()`, unchanged, on the golden path
   (F5 + uploaded background)
8. **Persist** → `<name>.wav`, `<name>.script.txt`, `<name>.meta.json` as siblings

Full detail: [docs/auto_generation/README.md](docs/auto_generation/README.md).
```

Add to Top Gotchas:

```markdown
- **Fatal vs advisory violations** → `script_gen/linter.py` fails the job for
  safety hard-blocks and malformed markers, but renders anyway (with a warning)
  for duration drift and style issues. Treating every violation as fatal makes
  a weaker local model unusable; treating none as fatal lets a safety failure
  reach audio. Do not flatten this distinction.
```

- [ ] **Step 3: Update docs/ARCHITECTURE.md**

Add an "Auto-Generation Subsystem" section with the data-flow diagram from the
spec, the module table, the failure-severity table, and a note that
`MeditationPipeline` is called and never modified. Respect the RAM-fix warning.

- [ ] **Step 4: Update docs/COMPONENT_REGISTRY.md**

One row per new module: `script_gen/{engine,rules,generator,judge,linter,duration,bench}.py`, `script_gen/adapters/{openai_compat,anthropic_api}.py`, `auto_generate.py`, `background_picker.py`, `streaming_run.py` — each with its public API and one-line responsibility.

- [ ] **Step 5: Update docs/TASK_ROUTING.md**

Add rows: "Change how scripts are written" → `docs/prompting_guides/` + `script_gen/rules.py`. "Add a safety rule" → `content_safety_rules.md` **and** `script_gen/linter.py` **and** a case in `tests/unit/test_script_linter.py`. "Add a model provider" → `script_gen/engine.py` registry + an adapter. "Tune duration accuracy" → `script_gen/duration.py::DEFAULT_WPM`.

- [ ] **Step 6: Update docs/GOTCHAS.md**

Add: the fatal/advisory split; first-colon-only spec parsing (`ollama:qwen3:30b`); guides read at call time so edits apply without restart; `app.py` cannot be imported in tests because of the `atexit` hard-exit hook, which is why `core/streaming_run.py` exists; fades are excluded from duration estimates because `apply_fades` does not extend runtime.

- [ ] **Step 7: Update README.md**

A short "Auto-Generate" section: what it does, the minimum `.env` needed, and a pointer to `docs/auto_generation/README.md`.

- [ ] **Step 8: Verify every referenced path exists**

```bash
grep -oE '\[[^]]+\]\(([^)]+)\)' docs/auto_generation/README.md CLAUDE.md \
  | grep -oE '\(([^)]+)\)' | tr -d '()' | grep -v '^http' \
  | while read -r p; do [ -e "$p" ] || echo "BROKEN: $p"; done
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add docs/auto_generation/README.md CLAUDE.md docs/ARCHITECTURE.md docs/COMPONENT_REGISTRY.md docs/TASK_ROUTING.md docs/GOTCHAS.md README.md
git commit -m "docs: document the auto-generation subsystem"
```

---

### Task 15: Review history and push

**Files:** none — this task only inspects and publishes.

**Interfaces:** none.

The per-task commits already form a logical sequence. This task verifies that, then publishes the branch.

- [ ] **Step 1: Confirm the working tree is clean**

Run: `git status --short`

Expected: only the pre-existing sleep-story changes (`CLAUDE.md`, `app.py`, `core/mixer.py`, `core/pipeline.py`, the preprocessors, `requirements.txt`, `scripts/generate.py`, `core/content_profiles.py`, the sleep-story guides, `tests/unit/test_content_profiles.py`). **Those are not ours — do not commit them.** If anything from this plan is uncommitted, commit it to its own task's commit first.

- [ ] **Step 2: Review the commit sequence**

Run: `git log --oneline dev..dev-automate`

Expected: roughly seventeen commits, each a conventional commit naming one deliverable. Read them as a story: does each message say what changed and why? If any is vague ("fix stuff", "wip"), reword it with `git rebase -i dev` before pushing. Nothing has been published yet, so history is still safe to edit.

- [ ] **Step 2b: Declare the new runtime dependencies**

Tasks 7 and 8 deliberately did not touch `requirements.txt` because it held unrelated uncommitted work. Check whether that is still true:

Run: `git status --short requirements.txt`

- **If it is now clean** (the other work was committed or stashed): append `httpx` and `anthropic` to `requirements.txt`, then `git add requirements.txt` and commit as `chore(deps): declare httpx and anthropic for script generation`.
- **If it is still dirty**: do NOT stage it. Leave the declaration to the user and say so explicitly in the final report — the two lines needed are `httpx` and `anthropic`. Note it in the PR body as a known follow-up.

Either way, `httpx` is already installed transitively so nothing breaks at runtime today; `anthropic` is only needed if a Claude model is actually configured.

- [ ] **Step 3: Confirm the full suite is green**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 4: Check whether the RAM fix has landed**

Run: `git fetch origin && git log --oneline origin/dev -5`

If the 32 GB correction is on `origin/dev`, rebase before pushing:
`git rebase origin/dev` and resolve any `CLAUDE.md` / `ARCHITECTURE.md` conflicts by keeping **both** changes — the RAM number from their commit, the auto-generation sections from ours.

- [ ] **Step 5: Push the branch**

```bash
git push -u origin dev-automate
```

This creates `dev-automate` on the remote; it does not exist upstream yet.

- [ ] **Step 6: Open a pull request into `dev`**

```bash
gh pr create --base dev --head dev-automate \
  --title "feat: automated end-to-end meditation generation" \
  --body "$(cat <<'BODY'
Turns a natural-language prompt into a finished meditation with no human step.

## What this adds
- `core/script_gen/` — two-pass generation (generator, then an independent judge that revises) bounded by a deterministic linter and duration estimator
- Pluggable `ScriptEngine` ABC — local (Ollama), hosted open-weight (OpenRouter/Together/Fireworks/Groq) and Claude behind one interface; the model is a config string
- `core/auto_generate.py` — orchestrator: script -> random background -> existing pipeline
- Auto-Generate tab in the Gradio UI
- `scripts/bench_script_models.py` — benchmark harness to choose models with evidence

## What this does not change
`core/pipeline.py`, `core/mixer.py` and the audio path are untouched. The auto
path calls `MeditationPipeline.generate()` exactly as the manual tab does.

## Design decisions
- **Fatal vs advisory violations.** Safety hard-blocks and malformed markers
  fail the job; duration drift and style issues render with a warning. This is
  what makes a weaker local model usable without letting a safety failure reach
  audio.
- **Model choice is deferred to measurement.** The bench harness answers it
  with data rather than an assertion in a design doc.

## Testing
Unit tests run with no model and no network. Integration tests are opt-in:
`MOODSCAPE_E2E=1` renders real audio, `MOODSCAPE_LIVE_SCRIPT_TEST=1` hits a
real model.

Spec: `docs/superpowers/specs/2026-09-16-automated-meditation-generation-design.md`
Plan: `docs/superpowers/plans/2026-09-16-automated-meditation-generation.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 7: Report the PR URL**

Print the URL `gh pr create` returned. Do **not** merge it and do **not** enable auto-merge — review is the user's call.

---

## After the plan

Once all twelve tasks are done, the model question is answered empirically rather than by assertion:

```bash
ollama pull <candidate-a>
ollama pull <candidate-b>
.venv/bin/python scripts/bench_script_models.py \
    --pair ollama:<candidate-a> ollama:<candidate-b> \
    --pair anthropic:claude-opus-5 anthropic:claude-sonnet-5 \
    --limit 3 --out bench_results.md
```

Read the scripts, compare lint pass rates and durations, then set the winning pair
in `.env` as `MOODSCAPE_SCRIPT_GENERATOR` and `MOODSCAPE_SCRIPT_JUDGE`.

Deferred by design, recorded here so it is not lost:

- **`mlx_local.py` adapter.** Only worth building if the bench shows a local model
  worth running in-process rather than through Ollama. `mlx-lm` is installed but
  undeclared in `requirements.txt` (a leftover from the ACE-Step removal), so
  declaring it is a prerequisite.
- **WPM calibration.** `log_estimate_accuracy` writes estimate-versus-actual on
  every render. After a dozen real generations, adjust `DEFAULT_WPM`.
- **Safety rules growth.** `content_safety_rules.md` and the linter's
  `_SAFETY_RULES` are expected to grow as real failure modes appear. Every
  addition gets a test case in `tests/unit/test_script_linter.py`.
