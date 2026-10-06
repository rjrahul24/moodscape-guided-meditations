# R0 Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evaluation harness for the auto-generation pipeline. It computes script, speech and mix metrics for every run of a fixed test set, and exports and scores blind A/B listening pairs, so every later quality change is judged on evidence.

**Architecture:**
- A new torch-free-at-import `core/eval/` package holds small single-purpose modules: `slop`, `script_metrics`, `mix_metrics`, `speech_metrics`, `testset`, `report` and `ab`.
- `scripts/eval_genres.py` gains opt-in flags that call those modules after each run. Without the flags, its behaviour is byte-for-byte the same.
- Two new CLI scripts wrap `core/eval/ab.py` for blind listening.

**Tech Stack:** Python 3.11, numpy, scipy (`resample_poly`), soundfile, pyloudnorm 0.2.0 (`Meter.integrated_loudness`, `Meter.loudness_range`), rapidfuzz (new), num2words (installed), transformers pipeline with `openai/whisper-large-v3-turbo` (already in the HF cache), unittest-style tests run by pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-r0-eval-harness-design.md`

## Global Constraints

- R0 is read-only with respect to the audio path. Do not modify `core/pipeline.py`, `core/mixer.py`, or any FX chain.
- The only allowed change to `core/auto_generate.py` is an additive `draft_script: str = ""` field on `AutoResult`, populated from `outcome.draft_script`. This is needed for the draft-vs-final edit ratio. No behaviour changes.
- Every `core/eval/*` module must import without torch or Gradio. Heavy imports go inside functions.
- A metric failure must never fail a run row. Record `{"error": "<Type>: <msg>"}` for that metric group instead.
- Excerpts are level-matched to −20 LUFS measured as dual-mono. Positions are 10% / 50% / 85%. The default excerpt length is 75 s.
- Ship rule: the challenger wins ≥ 60% of non-tie pairs **and** the one-sided sign test gives p < 0.05.
- Tests follow repo style: `unittest.TestCase` classes in `tests/unit/test_eval_*.py`, run with `.venv/bin/python -m pytest`.
- Commits use conventional-commit messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit on the current branch. Never push.
- `var/` is gitignored. Baseline runs go under `var/eval/`.

## Review Focus

1. **Mono vs stereo input files**: every audio metric must accept mono (1-D after read) and 2-channel files without shape errors. *Pinned in Task 3* (`test_stereo_file_metrics`).
2. **Silent or near-silent audio**, such as a failed render: `integrated_loudness` returns `-inf`. Metrics must report `None`, not crash or emit `-Infinity` into JSON. *Pinned in Task 3* (`test_silent_file_yields_none_loudness`).
3. **Scripts with digits, tags and curly apostrophes**: CER must not be inflated by `[pause:3s]`, "4" vs "four", or `’` vs `'`. *Pinned in Task 4* (`test_normalize_handles_tags_digits_apostrophes`).
4. **Run directories with a missing case on one side**: `ab_export` must refuse and name the missing cases, never silently shrink. *Pinned in Task 7* (`test_missing_case_raises_with_names`).
5. **Files shorter than one excerpt** (short band, a failed partial render): cut what exists and flag `truncated`. Never index past the end. *Pinned in Task 7* (`test_short_file_excerpt_truncated`).

---

### Task 1: Slop lexicon and density

**Files:**
- Create: `core/eval/__init__.py`
- Create: `core/eval/slop.py`
- Test: `tests/unit/test_eval_slop.py`

**Interfaces:**
- Consumes: `core.script_gen.linter._strip_tags(script) -> str` and `core.script_gen.linter._normalize_apostrophes(text) -> str` (both exist).
- Produces:
  - `SLOP_HARD: tuple[str, ...]`, `SLOP_RATE_LIMITED: tuple[str, ...]`, `SLOP_PATTERNS: dict[str, str]`.
  - `@dataclass(frozen=True) class SlopHit: term: str; tier: str  # "hard" | "rate" | "pattern"`.
  - `slop_hits(text: str) -> list[SlopHit]`.
  - `slop_density(text: str) -> float` (hits per 150 words; 0.0 for empty text).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_slop.py
import unittest

from core.eval.slop import SlopHit, slop_density, slop_hits


class TestSlopHits(unittest.TestCase):
    def test_hard_phrase_found_case_insensitive(self):
        hits = slop_hits("Let the warmth WASH OVER YOU like a tide.")
        self.assertIn(SlopHit("wash over you", "hard"), hits)

    def test_rate_limited_word_boundary(self):
        hits = slop_hits("Breathe gently. The gentleman waits.")
        terms = [h.term for h in hits]
        self.assertEqual(terms.count("gently"), 1)  # "gentleman" must not match

    def test_tags_are_ignored(self):
        self.assertEqual(slop_hits("[pause:3s] [breath]"), [])

    def test_curly_apostrophe_pattern(self):
        hits = slop_hits("It’s not a task, it’s a rest.")
        self.assertIn("not_x_its_y", [h.term for h in hits])

    def test_triple_soft_adjectives(self):
        hits = slop_hits("A warm, soft, gentle light.")
        self.assertIn("triple_soft_adjectives", [h.term for h in hits])

    def test_five_senses_list_is_not_triple_adjective(self):
        hits = slop_hits("Notice what you see, hear, and feel.")
        self.assertNotIn("triple_soft_adjectives", [h.term for h in hits])


class TestSlopDensity(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(slop_density(""), 0.0)

    def test_density_per_150_words(self):
        text = "gently " + "word " * 149  # 150 words, 1 hit
        self.assertAlmostEqual(slop_density(text), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_slop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval'`

- [ ] **Step 3: Write the implementation**

```python
# core/eval/__init__.py
"""Evaluation harness: script, speech and mix metrics plus blind A/B tooling.

Every module imports without torch or Gradio; heavy dependencies are imported
inside the functions that need them.
"""

from core.eval.slop import SlopHit, slop_density, slop_hits

__all__ = ["SlopHit", "slop_density", "slop_hits"]
```

```python
# core/eval/slop.py
"""Meditation-writing slop lexicon and density metric.

Seeded from the Oct 2026 research report (artifact d). R1 refines the list
from MoodScape's own output; keep it short and targeted (Antislop: plain
banning degrades past ~2,000 patterns).
"""

import re
from dataclasses import dataclass

from core.script_gen.linter import _normalize_apostrophes, _strip_tags

SLOP_HARD: tuple[str, ...] = (
    "tapestry", "symphony of", "dance of", "embark", "delve", "testament to",
    "realm", "ethereal", "kaleidoscope", "whisper of the wind", "kiss of the sun",
    "wash over you", "melt away", "every cell of your body", "healing light",
    "golden light flowing", "pure white light", "inner sanctuary", "oasis of calm",
    "universe holds you", "vibrant energy", "deeper and deeper and deeper",
)

SLOP_RATE_LIMITED: tuple[str, ...] = (
    "gently", "softly", "slowly", "simply", "allow yourself", "invite you to",
    "notice how", "let go", "release", "sink", "drift", "serene", "tranquil",
    "peaceful", "journey", "embrace", "present moment", "stillness within",
)

_SOFT_ADJ = (
    "warm|soft|gentle|calm|quiet|peaceful|serene|tranquil|still|safe|light|"
    "heavy|deep|slow|smooth|tender|loose|easy"
)

SLOP_PATTERNS: dict[str, str] = {
    "more_and_more": r"\bmore and more\b",
    "not_x_its_y": r"\bit's not [^.,;]{1,40}, it's\b",
    "triple_soft_adjectives": (
        rf"\b(?:{_SOFT_ADJ}), (?:{_SOFT_ADJ}),? (?:and )?(?:{_SOFT_ADJ})\b"
    ),
}

_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class SlopHit:
    term: str
    tier: str  # "hard" | "rate" | "pattern"


def _prose(text: str) -> str:
    return _normalize_apostrophes(_strip_tags(text)).lower()


def _phrase_regex(phrase: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(phrase) + r"\b")


def slop_hits(text: str) -> list[SlopHit]:
    """Every lexicon hit in the tag-stripped, apostrophe-normalised prose."""
    prose = _prose(text)
    hits: list[SlopHit] = []
    for tier, phrases in (("hard", SLOP_HARD), ("rate", SLOP_RATE_LIMITED)):
        for phrase in phrases:
            count = len(_phrase_regex(phrase).findall(prose))
            hits.extend(SlopHit(phrase, tier) for _ in range(count))
    for name, pattern in SLOP_PATTERNS.items():
        count = len(re.findall(pattern, prose))
        hits.extend(SlopHit(name, "pattern") for _ in range(count))
    return hits


def slop_density(text: str) -> float:
    """Lexicon hits per 150 words of prose (0.0 for empty text)."""
    words = len(_WORD.findall(_prose(text)))
    if words == 0:
        return 0.0
    return len(slop_hits(text)) * 150.0 / words
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_slop.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add core/eval/__init__.py core/eval/slop.py tests/unit/test_eval_slop.py
git commit -m "feat(eval): add meditation slop lexicon and density metric

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Script metrics

**Files:**
- Create: `core/eval/script_metrics.py`
- Modify: `core/eval/__init__.py` (add export)
- Test: `tests/unit/test_eval_script_metrics.py`

**Interfaces:**
- Consumes:
  - `slop_hits` and `slop_density` from Task 1.
  - `core.script_gen.linter.check(script) -> list[Violation]` (`Violation.code`, `.severity` in `{"FATAL","ADVISORY"}`).
  - `core.script_gen.duration.estimate_duration_sec(script, *, engine, content_type) -> float`.
  - `core.f5_tts.preprocessor.prepare_segments(script, content_type)` and `core.kokoro_tts.preprocessor.prepare_segments(script, content_type)`. Segments are dicts: `{"type":"pause","duration_sec":float}`, `{"type":"breath","subtype":str}`, `{"type":"speech","text":str}`.
- Produces: `script_metrics(script: str, *, draft: str | None = None, engine: str = "f5", content_type: str = "meditation") -> dict` with keys `words, sentences, mean_sentence_words, max_sentence_words, question_count, pause_seconds, breath_count, estimated_sec, pause_ratio_est, lint_fatal, lint_advisory, slop_density, slop_top, edit_ratio`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_script_metrics.py
import unittest

from core.eval.script_metrics import script_metrics

SCRIPT = (
    "Find a comfortable position. [pause:4s] Let your shoulders soften gently.\n\n"
    "[breath]\n\n"
    "Notice the weight of your hands. [pause:6s]"
)


class TestScriptMetrics(unittest.TestCase):
    def test_counts(self):
        m = script_metrics(SCRIPT)
        self.assertEqual(m["sentences"], 3)
        self.assertEqual(m["words"], 15)
        self.assertEqual(m["max_sentence_words"], 6)
        self.assertEqual(m["breath_count"], 1)
        self.assertEqual(m["question_count"], 0)

    def test_pause_seconds_match_preprocessor(self):
        from core.f5_tts.preprocessor import prepare_segments
        expected = sum(
            s["duration_sec"] for s in prepare_segments(SCRIPT, content_type="meditation")
            if s["type"] == "pause"
        )
        self.assertAlmostEqual(script_metrics(SCRIPT)["pause_seconds"], expected)

    def test_pause_ratio_is_fraction_of_estimate(self):
        m = script_metrics(SCRIPT)
        self.assertAlmostEqual(m["pause_ratio_est"], m["pause_seconds"] / m["estimated_sec"])

    def test_lint_codes_split_by_severity(self):
        m = script_metrics("**Bold** text. [pause:3]")
        self.assertIn("MARKDOWN_PRESENT", m["lint_fatal"])
        self.assertIn("MARKER_MALFORMED", m["lint_fatal"])

    def test_slop_reported(self):
        m = script_metrics(SCRIPT)
        self.assertGreater(m["slop_density"], 0.0)
        self.assertIn("gently", m["slop_top"])

    def test_edit_ratio_none_without_draft(self):
        self.assertIsNone(script_metrics(SCRIPT)["edit_ratio"])

    def test_edit_ratio_zero_for_identical_and_positive_for_changed(self):
        self.assertEqual(script_metrics(SCRIPT, draft=SCRIPT)["edit_ratio"], 0.0)
        changed = script_metrics(SCRIPT, draft="Something entirely different here.")
        self.assertGreater(changed["edit_ratio"], 0.5)

    def test_non_f5_cloning_engine_uses_f5_grammar(self):
        m = script_metrics(SCRIPT, engine="qwen3")
        self.assertEqual(m["breath_count"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_script_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.script_metrics'`

- [ ] **Step 3: Write the implementation**

```python
# core/eval/script_metrics.py
"""Deterministic text metrics for a generated script."""

import difflib
import re
from collections import Counter

from core.eval.slop import slop_density, slop_hits
from core.script_gen.linter import _normalize_apostrophes, _strip_tags

_WORD = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _segments(script: str, engine: str, content_type: str) -> list[dict]:
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import prepare_segments
    else:  # f5, chatterbox, qwen3 all share the F5 script grammar
        from core.f5_tts.preprocessor import prepare_segments
    return prepare_segments(script, content_type=content_type)


def _estimate(script: str, engine: str, content_type: str) -> float:
    from core.script_gen.duration import estimate_duration_sec
    duration_engine = "kokoro" if engine == "kokoro" else "f5"
    return estimate_duration_sec(script, engine=duration_engine, content_type=content_type)


def script_metrics(
    script: str,
    *,
    draft: str | None = None,
    engine: str = "f5",
    content_type: str = "meditation",
) -> dict:
    """Counts, silence budget, lint codes, slop and draft-vs-final edit ratio."""
    from core.script_gen.linter import check

    prose = _normalize_apostrophes(_strip_tags(script))
    sentences = [s for s in _SENTENCE_SPLIT.split(prose.strip()) if _WORD.search(s)]
    sentence_words = [len(_WORD.findall(s)) for s in sentences]

    segments = _segments(script, engine, content_type)
    pause_seconds = float(sum(s["duration_sec"] for s in segments if s["type"] == "pause"))
    breath_count = sum(1 for s in segments if s["type"] == "breath")
    estimated_sec = float(_estimate(script, engine, content_type))

    violations = check(script)
    hits = slop_hits(script)

    edit_ratio = None
    if draft is not None:
        matcher = difflib.SequenceMatcher(None, draft.split(), script.split())
        edit_ratio = round(1.0 - matcher.ratio(), 4)

    return {
        "words": len(_WORD.findall(prose)),
        "sentences": len(sentences),
        "mean_sentence_words": (sum(sentence_words) / len(sentence_words)) if sentence_words else 0.0,
        "max_sentence_words": max(sentence_words, default=0),
        "question_count": prose.count("?"),
        "pause_seconds": pause_seconds,
        "breath_count": breath_count,
        "estimated_sec": estimated_sec,
        "pause_ratio_est": (pause_seconds / estimated_sec) if estimated_sec else 0.0,
        "lint_fatal": [v.code for v in violations if v.severity == "FATAL"],
        "lint_advisory": [v.code for v in violations if v.severity == "ADVISORY"],
        "slop_density": round(slop_density(script), 3),
        "slop_top": [term for term, _n in Counter(h.term for h in hits).most_common(5)],
        "edit_ratio": edit_ratio,
    }
```

Add to `core/eval/__init__.py`: `from core.eval.script_metrics import script_metrics` and `"script_metrics"` in `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_script_metrics.py -v`
Expected: PASS (8 tests). If `test_counts` disagrees on `words`, count the words in `SCRIPT` by hand ("Find a comfortable position" 4 + "Let your shoulders soften gently" 5 + "Notice the weight of your hands" 6 = 15) and fix the regex, not the test.

- [ ] **Step 5: Commit**

```bash
git add core/eval/script_metrics.py core/eval/__init__.py tests/unit/test_eval_script_metrics.py
git commit -m "feat(eval): add deterministic script metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Mix metrics (loudness, LRA, true peak, duration)

**Files:**
- Create: `core/eval/mix_metrics.py`
- Modify: `core/eval/__init__.py` (add export)
- Test: `tests/unit/test_eval_mix_metrics.py`

**Interfaces:**
- Consumes: `soundfile`, `pyloudnorm.Meter`, `scipy.signal.resample_poly`.
- Produces:
  - `true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float | None` (audio shape `(n,)` or `(n, ch)`).
  - `loudness(audio, sr) -> dict` with keys `integrated_lufs, integrated_lufs_dual_mono, lra`.
  - `mix_metrics(path: str, *, estimated_sec: float | None = None) -> dict` with keys `channels, sample_rate, actual_sec, integrated_lufs, integrated_lufs_dual_mono, lra, true_peak_dbtp, duration_ratio`.
  - Non-finite loudness values become `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_mix_metrics.py
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from core.eval.mix_metrics import loudness, mix_metrics, true_peak_dbtp

SR = 48000


def _pink(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(SR * seconds))
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1 / SR)
    spectrum[1:] /= np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum, n=len(white))
    return (0.1 * pink / np.max(np.abs(pink))).astype(np.float32)


class TestLoudness(unittest.TestCase):
    def test_dual_mono_is_about_3db_louder(self):
        m = loudness(_pink(10.0), SR)
        self.assertAlmostEqual(m["integrated_lufs_dual_mono"] - m["integrated_lufs"], 3.01, delta=0.05)

    def test_silent_file_yields_none_loudness(self):
        m = loudness(np.zeros(SR * 5, dtype=np.float32), SR)
        self.assertIsNone(m["integrated_lufs"])
        self.assertIsNone(m["integrated_lufs_dual_mono"])


class TestTruePeak(unittest.TestCase):
    def test_intersample_peak_exceeds_sample_peak(self):
        n = np.arange(SR)
        x = np.sin(2 * np.pi * (SR / 4) * n / SR + np.pi / 4).astype(np.float32)  # samples at ±0.707
        self.assertGreater(true_peak_dbtp(x, SR), 20 * np.log10(np.max(np.abs(x))) + 2.0)

    def test_silence_is_none(self):
        self.assertIsNone(true_peak_dbtp(np.zeros(1000, dtype=np.float32), SR))


class TestMixMetricsFile(unittest.TestCase):
    def _write(self, audio: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, SR)
        self.addCleanup(os.remove, path)
        return path

    def test_mono_file_metrics(self):
        m = mix_metrics(self._write(_pink(10.0)), estimated_sec=8.0)
        self.assertEqual(m["channels"], 1)
        self.assertAlmostEqual(m["actual_sec"], 10.0, places=2)
        self.assertAlmostEqual(m["duration_ratio"], 1.25, places=2)
        self.assertIsNotNone(m["lra"])

    def test_stereo_file_metrics(self):
        mono = _pink(10.0)
        m = mix_metrics(self._write(np.stack([mono, mono], axis=1)))
        self.assertEqual(m["channels"], 2)
        self.assertAlmostEqual(m["integrated_lufs"], m["integrated_lufs_dual_mono"], places=3)
        self.assertIsNone(m["duration_ratio"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_mix_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.mix_metrics'`

- [ ] **Step 3: Write the implementation**

```python
# core/eval/mix_metrics.py
"""Loudness, loudness range, true peak and duration of a rendered file.

Mono files are also measured as dual-mono (the channel duplicated), which is
how stereo players reproduce them: +3.01 LU over the mono measurement. That
gap is the loudness-convention bug R5 fixes, so both numbers are recorded.
"""

import math

import numpy as np


def _finite(value: float) -> float | None:
    return round(float(value), 2) if value is not None and math.isfinite(value) else None


def _as_2d(audio: np.ndarray) -> np.ndarray:
    return audio.reshape(-1, 1) if audio.ndim == 1 else audio


def true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float | None:
    """4x-oversampled true peak in dBTP; None for digital silence."""
    from scipy.signal import resample_poly

    x = _as_2d(audio).astype(np.float64)
    up = resample_poly(x, oversample, 1, axis=0)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    if peak <= 0.0:
        return None
    return round(20.0 * math.log10(peak), 2)


def loudness(audio: np.ndarray, sr: int) -> dict:
    """Integrated LUFS (as is and as dual-mono) and loudness range (LU)."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    x = _as_2d(audio).astype(np.float64)
    measured = x[:, 0] if x.shape[1] == 1 else x
    dual = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x

    def _safe(fn, data):
        try:
            return _finite(fn(data))
        except Exception:
            return None

    return {
        "integrated_lufs": _safe(meter.integrated_loudness, measured),
        "integrated_lufs_dual_mono": _safe(meter.integrated_loudness, dual),
        "lra": _safe(meter.loudness_range, measured),
    }


def mix_metrics(path: str, *, estimated_sec: float | None = None) -> dict:
    """All mix-level metrics for one audio file."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    actual_sec = audio.shape[0] / sr
    metrics = {
        "channels": int(audio.shape[1]),
        "sample_rate": int(sr),
        "actual_sec": round(actual_sec, 3),
        **loudness(audio, sr),
        "true_peak_dbtp": true_peak_dbtp(audio, sr),
        "duration_ratio": round(actual_sec / estimated_sec, 4) if estimated_sec else None,
    }
    return metrics
```

Add `loudness, mix_metrics, true_peak_dbtp` to `core/eval/__init__.py` exports.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_mix_metrics.py -v`
Expected: PASS (6 tests). If `loudness_range` raises on 10 s of audio (it needs ≥ 3 s short-term windows), the `_safe` wrapper returns `None`. In that case `test_mono_file_metrics` will fail on `lra`. Lengthen the fixture to 20 s rather than weakening the assertion.

- [ ] **Step 5: Commit**

```bash
git add core/eval/mix_metrics.py core/eval/__init__.py tests/unit/test_eval_mix_metrics.py
git commit -m "feat(eval): add loudness, LRA, true-peak and duration metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Speech metrics (Whisper CER/WER, internal gaps, rate spread)

**Files:**
- Create: `core/eval/speech_metrics.py`
- Modify: `requirements.txt` (add `rapidfuzz>=3.9`), `core/eval/__init__.py` (exports)
- Test: `tests/unit/test_eval_speech_metrics.py`
- Test (integration, gated): `tests/integration/test_eval_whisper_live.py`

**Interfaces:**
- Consumes: `core.script_gen.linter._strip_tags`, `_normalize_apostrophes`; `rapidfuzz.distance.Levenshtein`; `num2words`.
- Produces:
  - `@dataclass class Word: text: str; start: float; end: float`
  - `@dataclass class Transcript: text: str; words: list[Word]`
  - `class Transcriber(Protocol): def transcribe(self, audio: np.ndarray, sr: int) -> Transcript`
  - `class WhisperTranscriber` with `__init__(model_id="openai/whisper-large-v3-turbo")`, `transcribe(audio, sr)` and `unload()`
  - `normalize_text(text: str) -> str`
  - `cer(reference: str, hypothesis: str) -> float`, `wer(reference: str, hypothesis: str) -> float`
  - `internal_gaps(audio, sr, words, *, min_gap_ms=250.0) -> int`
  - `rate_spread(words) -> dict` with keys `sec_per_char_p50, sec_per_char_iqr`
  - `speech_metrics(audio, sr, expected_text, *, transcriber: Transcriber) -> dict` with keys `cer, wer, internal_gaps, sec_per_char_p50, sec_per_char_iqr, transcript`

- [ ] **Step 1: Add the dependency and install it**

Append `rapidfuzz>=3.9` to `requirements.txt` (keep alphabetical order if the file is sorted). Then run:

Run: `.venv/bin/pip install "rapidfuzz>=3.9"`
Expected: `Successfully installed rapidfuzz-...` (or "already satisfied").

- [ ] **Step 2: Write the failing unit test**

```python
# tests/unit/test_eval_speech_metrics.py
import unittest

import numpy as np

from core.eval.speech_metrics import (
    Transcript,
    Word,
    cer,
    internal_gaps,
    normalize_text,
    rate_spread,
    speech_metrics,
    wer,
)

SR = 16000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeTranscriber:
    def __init__(self, transcript: Transcript):
        self._t = transcript

    def transcribe(self, audio, sr):
        return self._t


class TestNormalize(unittest.TestCase):
    def test_normalize_handles_tags_digits_apostrophes(self):
        self.assertEqual(
            normalize_text("Breathe in for 4. [pause:3s] Don’t RUSH!"),
            "breathe in for four don't rush",
        )


class TestErrorRates(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(cer("let it be", "let it be"), 0.0)
        self.assertEqual(wer("let it be", "let it be"), 0.0)

    def test_one_word_dropped(self):
        self.assertAlmostEqual(wer("rest your hands now", "rest your now"), 0.25)

    def test_empty_reference(self):
        self.assertEqual(cer("", ""), 0.0)
        self.assertEqual(cer("", "extra"), 1.0)


class TestInternalGaps(unittest.TestCase):
    def test_planted_gap_inside_word_is_counted(self):
        audio = np.concatenate([_tone(0.5), np.zeros(int(SR * 0.4), np.float32), _tone(0.5)])
        words = [Word("relax", 0.0, 1.4)]
        self.assertEqual(internal_gaps(audio, SR, words), 1)

    def test_no_gap_in_continuous_speech(self):
        words = [Word("relax", 0.0, 1.0)]
        self.assertEqual(internal_gaps(_tone(1.0), SR, words), 0)

    def test_long_gap_between_words_mid_sentence_is_counted(self):
        words = [Word("rest", 0.0, 0.4), Word("here", 1.0, 1.4)]
        self.assertEqual(internal_gaps(_tone(1.4), SR, words), 1)

    def test_gap_after_sentence_end_is_intended(self):
        words = [Word("rest.", 0.0, 0.4), Word("Now", 3.0, 3.4)]
        self.assertEqual(internal_gaps(_tone(3.4), SR, words), 0)


class TestRateSpread(unittest.TestCase):
    def test_uniform_rate_has_zero_iqr(self):
        words = [Word("abc.", 0.0, 0.3), Word("abc.", 1.0, 1.3), Word("abc.", 2.0, 2.3)]
        spread = rate_spread(words)
        self.assertAlmostEqual(spread["sec_per_char_p50"], 0.1)
        self.assertAlmostEqual(spread["sec_per_char_iqr"], 0.0)

    def test_no_words(self):
        self.assertEqual(rate_spread([]), {"sec_per_char_p50": None, "sec_per_char_iqr": None})


class TestSpeechMetrics(unittest.TestCase):
    def test_uses_injected_transcriber(self):
        t = Transcript("Rest your hands.", [Word("Rest", 0.0, 0.3), Word("your", 0.3, 0.6), Word("hands.", 0.6, 1.0)])
        m = speech_metrics(_tone(1.0), SR, "Rest your hands. [pause:2s]", transcriber=FakeTranscriber(t))
        self.assertEqual(m["cer"], 0.0)
        self.assertEqual(m["internal_gaps"], 0)
        self.assertEqual(m["transcript"], "Rest your hands.")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_speech_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.speech_metrics'`

- [ ] **Step 4: Write the implementation**

```python
# core/eval/speech_metrics.py
"""Intelligibility and artifact metrics on rendered speech, engine-agnostic.

Whisper transcribes the audio; CER/WER compare it with the script's spoken
text; word timestamps plus frame energy find silences that land inside a word
or mid-sentence (F5's mid-word gaps and stalls).
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from core.script_gen.linter import _normalize_apostrophes, _strip_tags

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
_CLAUSE_END = re.compile(r"[.!?,;:][\"')\]]*$")


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript: ...


class WhisperTranscriber:
    """transformers ASR pipeline with word timestamps; loaded lazily."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id
        self._pipe = None

    def _load(self):
        import torch
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=device,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> Transcript:
        from scipy.signal import resample_poly

        if self._pipe is None:
            self._load()
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        audio16 = resample_poly(mono.astype(np.float64), 16000, sr).astype(np.float32)
        result = self._pipe(
            {"raw": audio16, "sampling_rate": 16000},
            return_timestamps="word",
            chunk_length_s=30,
            generate_kwargs={"language": "english"},
        )
        words = [
            Word(c["text"].strip(), float(c["timestamp"][0]), float(c["timestamp"][1] or c["timestamp"][0]))
            for c in result.get("chunks", [])
            if c.get("timestamp") and c["timestamp"][0] is not None
        ]
        return Transcript(result["text"].strip(), words)

    def unload(self) -> None:
        import gc

        self._pipe = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _spell_number(match: re.Match) -> str:
    from num2words import num2words

    return num2words(int(match.group(0))).replace("-", " ")


def normalize_text(text: str) -> str:
    """Lowercase, tags removed, digits spelled, punctuation dropped."""
    text = _normalize_apostrophes(_strip_tags(text)).lower()
    text = re.sub(r"\d+", _spell_number, text)
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    from rapidfuzz.distance import Levenshtein

    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    from rapidfuzz.distance import Levenshtein

    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def _silent_frames(audio: np.ndarray, sr: int, frame_ms: float = 10.0) -> np.ndarray:
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: n * hop].reshape(n, hop)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)
    threshold = max(-50.0, float(np.max(rms_db)) - 40.0)
    return rms_db < threshold


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def internal_gaps(audio: np.ndarray, sr: int, words: list[Word], *, min_gap_ms: float = 250.0) -> int:
    """Silences > min_gap_ms inside a word, or between words of one clause."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    silent = _silent_frames(mono, sr)
    frames_needed = int(min_gap_ms / 10.0)
    count = 0
    for word in words:
        a, b = int(word.start * 100), int(word.end * 100)
        if _longest_run(silent[a:b]) >= frames_needed:
            count += 1
    for prev, nxt in zip(words, words[1:]):
        if not _CLAUSE_END.search(prev.text) and (nxt.start - prev.end) * 1000.0 > min_gap_ms:
            count += 1
    return count


def rate_spread(words: list[Word]) -> dict:
    """Median and IQR of seconds-per-character across Whisper sentences."""
    rates: list[float] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if _SENTENCE_END.search(word.text):
            chars = sum(len(re.sub(r"[^A-Za-z']", "", w.text)) for w in current)
            if chars:
                rates.append((current[-1].end - current[0].start) / chars)
            current = []
    if not rates:
        return {"sec_per_char_p50": None, "sec_per_char_iqr": None}
    q1, q2, q3 = np.percentile(rates, [25, 50, 75])
    return {"sec_per_char_p50": round(float(q2), 4), "sec_per_char_iqr": round(float(q3 - q1), 4)}


def speech_metrics(audio: np.ndarray, sr: int, expected_text: str, *, transcriber: Transcriber) -> dict:
    transcript = transcriber.transcribe(audio, sr)
    return {
        "cer": round(cer(expected_text, transcript.text), 4),
        "wer": round(wer(expected_text, transcript.text), 4),
        "internal_gaps": internal_gaps(audio, sr, transcript.words),
        **rate_spread(transcript.words),
        "transcript": transcript.text,
    }
```

Add `Transcript, Transcriber, WhisperTranscriber, Word, cer, wer, normalize_text, internal_gaps, rate_spread, speech_metrics` to `core/eval/__init__.py`.

- [ ] **Step 5: Run the unit test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_speech_metrics.py -v`
Expected: PASS (11 tests). `internal_gaps` uses 10 ms frames, so word times map to frame index `int(t * 100)`. Keep `frame_ms` at 10 or update that mapping too.

- [ ] **Step 6: Write the gated live integration test**

```python
# tests/integration/test_eval_whisper_live.py
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("MOODSCAPE_EVAL_LIVE") == "1", "set MOODSCAPE_EVAL_LIVE=1 to run")
class TestWhisperLive(unittest.TestCase):
    def test_real_whisper_on_tts_fixture(self):
        from core.eval.speech_metrics import WhisperTranscriber, speech_metrics
        from core.kokoro_tts.engine import KokoroEngine
        from core.kokoro_tts.preprocessor import prepare_segments

        text = "Find a comfortable position and let your shoulders soften."
        engine = KokoroEngine()
        engine.load_model()
        audio, _ = engine.synthesize(prepare_segments(text), voice="balanced_calm", speed=0.9)
        engine.unload_model()
        transcriber = WhisperTranscriber()
        metrics = speech_metrics(audio, 24000, text, transcriber=transcriber)
        transcriber.unload()
        self.assertLess(metrics["cer"], 0.10, metrics["transcript"])
        self.assertTrue(np.isfinite(metrics["cer"]))
```

Run: `MOODSCAPE_EVAL_LIVE=1 .venv/bin/python -m pytest tests/integration/test_eval_whisper_live.py -v`
Expected: PASS (first run downloads nothing; the model is in the HF cache). If `KokoroEngine.synthesize` needs a different signature, read `core/kokoro_tts/engine.py` and adapt the call, keeping the assertion.

- [ ] **Step 7: Commit**

```bash
git add core/eval/speech_metrics.py core/eval/__init__.py requirements.txt tests/unit/test_eval_speech_metrics.py tests/integration/test_eval_whisper_live.py
git commit -m "feat(eval): add Whisper CER/WER, internal-gap and rate-spread speech metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Fixed test set

**Files:**
- Create: `eval/testset.toml`
- Create: `core/eval/testset.py`
- Modify: `core/eval/__init__.py` (exports)
- Test: `tests/unit/test_eval_testset.py`

**Interfaces:**
- Consumes: `core.auto_generate.DURATION_BANDS` (keys `short`, `medium`, `long`) and `core.genres.load_pack(slug)` (raises on an unknown slug).
- Produces:
  - `@dataclass(frozen=True) class TestCase: id: str; genre: str; band: str; seed: int`.
  - `DEFAULT_TESTSET: Path` (repo `eval/testset.toml`).
  - `load_testset(path: Path | None = None) -> list[TestCase]`. Ids are `f"{genre}-{band}"`; the seed is `base_seed + band index`.

- [ ] **Step 1: Write the test set file**

```toml
# eval/testset.toml — fixed evaluation set (R0). 12 genres x 2 bands = 24 jobs.
# Seeds fix angle choice, background choice and the TTS base seed. F5 is still
# nondeterministic (up to ~4 LU short-term drift), so always compare runs made
# in the same session, never against a stored number.

bands = ["short", "medium"]

[[cases]]
genre = "anxiety_relief"
seed = 1100

[[cases]]
genre = "panic_sos"
seed = 1200

[[cases]]
genre = "yoga_nidra"
seed = 1300

[[cases]]
genre = "racing_mind_at_night"
seed = 1400

[[cases]]
genre = "deep_sleep"
seed = 1500

[[cases]]
genre = "fall_asleep"
seed = 1600

[[cases]]
genre = "box_breathing"
seed = 1700

[[cases]]
genre = "coherent_breathing"
seed = 1800

[[cases]]
genre = "body_scan"
seed = 1900

[[cases]]
genre = "workout_cool_down"
seed = 2000

[[cases]]
genre = "self_compassion"
seed = 2100

[[cases]]
genre = "grief_and_loss"
seed = 2200
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_eval_testset.py
import tempfile
import unittest
from pathlib import Path

from core.eval.testset import DEFAULT_TESTSET, TestCase, load_testset


class TestLoadTestset(unittest.TestCase):
    def test_default_testset_has_24_unique_cases(self):
        cases = load_testset()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({c.id for c in cases}), 24)
        self.assertTrue(DEFAULT_TESTSET.is_file())

    def test_case_fields(self):
        first = load_testset()[0]
        self.assertEqual(first, TestCase("anxiety_relief-short", "anxiety_relief", "short", 1100))

    def test_seed_offset_per_band(self):
        cases = {c.id: c for c in load_testset()}
        self.assertEqual(cases["anxiety_relief-medium"].seed, 1101)

    def test_unknown_band_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.toml"
            path.write_text('bands = ["huge"]\n[[cases]]\ngenre = "body_scan"\nseed = 1\n')
            with self.assertRaisesRegex(ValueError, "huge"):
                load_testset(path)

    def test_unknown_genre_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.toml"
            path.write_text('bands = ["short"]\n[[cases]]\ngenre = "no_such_genre"\nseed = 1\n')
            with self.assertRaises(Exception):
                load_testset(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_testset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.testset'`

- [ ] **Step 4: Write the implementation**

```python
# core/eval/testset.py
"""The fixed evaluation test set (eval/testset.toml)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TESTSET = Path(__file__).resolve().parent.parent.parent / "eval" / "testset.toml"


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # stop pytest collecting this dataclass as a test class

    id: str
    genre: str
    band: str
    seed: int


def load_testset(path: Path | None = None) -> list[TestCase]:
    """Expand genres x bands into cases; validate bands and genre packs."""
    from core.auto_generate import DURATION_BANDS
    from core.genres import load_pack

    data = tomllib.loads(Path(path or DEFAULT_TESTSET).read_text(encoding="utf-8"))
    bands = list(data.get("bands", []))
    unknown = [b for b in bands if b not in DURATION_BANDS]
    if unknown:
        raise ValueError(f"Unknown duration band(s) {unknown}; expected {sorted(DURATION_BANDS)}.")
    cases: list[TestCase] = []
    for entry in data.get("cases", []):
        genre = entry["genre"]
        load_pack(genre)  # raises for an unknown or malformed pack
        for offset, band in enumerate(bands):
            cases.append(TestCase(f"{genre}-{band}", genre, band, int(entry["seed"]) + offset))
    return cases
```

Add `DEFAULT_TESTSET, TestCase, load_testset` to `core/eval/__init__.py`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_testset.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add eval/testset.toml core/eval/testset.py core/eval/__init__.py tests/unit/test_eval_testset.py
git commit -m "feat(eval): add fixed 24-job evaluation test set

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Runner integration and summary report

**Files:**
- Create: `core/eval/report.py`
- Modify: `core/auto_generate.py` (additive `draft_script` field on `AutoResult`, set in `run()`)
- Modify: `scripts/eval_genres.py`
- Modify: `core/eval/__init__.py` (exports)
- Test: `tests/unit/test_eval_report.py`
- Test: `tests/unit/test_auto_generate.py` (one added assertion)

**Interfaces:**
- Consumes:
  - `script_metrics`, `mix_metrics`, `speech_metrics`, `WhisperTranscriber` and `load_testset` from Tasks 2–5.
  - `core.auto_generate.run(prompt, *, genre, config, pipeline, rng, **pipeline_kwargs) -> AutoResult`. `pipeline_kwargs` are forwarded to `MeditationPipeline.generate()`, which accepts `seed: int | None` and `do_export_stems: bool`. After `generate()`, `pipeline.last_stems` is `{"voice": path, "music": path}` or `None`.
  - `AutoConfig.from_genre(pack, band=..., **overrides)`, with `tts_engine` as an override.
- Produces:
  - `AutoResult.draft_script: str` (default `""`).
  - `compute_run_metrics(*, script, draft, engine, content_type, audio_path, voice_stem_path, estimated_sec, transcriber) -> dict` with keys `script`, `mix`, `speech` (each a dict, or `{"error": ...}`) and `speech_input` (`"voice_stem"` or `"mix"`).
  - `summarize(rows: list[dict]) -> str` (markdown).
  - `scripts/eval_genres.py` flags `--testset PATH`, `--metrics`, `--export-stems`, `--engine NAME`. It writes `metrics.json` per run, plus `results.json` and `summary.md` at the out root.

- [ ] **Step 1: Write the failing report test**

```python
# tests/unit/test_eval_report.py
import unittest
from unittest.mock import patch

from core.eval.report import compute_run_metrics, summarize


class TestComputeRunMetrics(unittest.TestCase):
    def test_group_failure_is_recorded_not_raised(self):
        with patch("core.eval.report.mix_metrics", side_effect=RuntimeError("bad wav")):
            m = compute_run_metrics(
                script="Rest here. [pause:2s]", draft=None, engine="f5",
                content_type="meditation", audio_path="/nonexistent.wav",
                voice_stem_path=None, estimated_sec=10.0, transcriber=None,
            )
        self.assertEqual(m["mix"], {"error": "RuntimeError: bad wav"})
        self.assertIn("words", m["script"])
        self.assertEqual(m["speech"], {"skipped": "no transcriber"})


class TestSummarize(unittest.TestCase):
    def test_medians_per_config_and_pass_rate(self):
        rows = [
            {"config": "base", "ok": True, "metrics": {
                "script": {"slop_density": 1.0, "edit_ratio": 0.2},
                "speech": {"cer": 0.02, "internal_gaps": 1},
                "mix": {"integrated_lufs_dual_mono": -13.0, "duration_ratio": 1.1}}},
            {"config": "base", "ok": True, "metrics": {
                "script": {"slop_density": 3.0, "edit_ratio": 0.4},
                "speech": {"cer": 0.04, "internal_gaps": 3},
                "mix": {"integrated_lufs_dual_mono": -13.0, "duration_ratio": 0.9}}},
            {"config": "base", "ok": False, "error": "boom"},
        ]
        md = summarize(rows)
        self.assertIn("| base | 2/3 |", md)
        self.assertIn("2.00", md)   # slop median
        self.assertIn("0.030", md)  # CER median
        self.assertIn("0.10", md)   # median |1 - duration ratio|

    def test_missing_metrics_render_as_dash(self):
        md = summarize([{"config": "x", "ok": True, "metrics": {}}])
        self.assertIn("| x | 1/1 | – |", md)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.report'`

- [ ] **Step 3: Write `core/eval/report.py`**

```python
# core/eval/report.py
"""Per-run metric collection and the markdown summary table."""

import statistics

from core.eval.mix_metrics import mix_metrics
from core.eval.script_metrics import script_metrics
from core.eval.speech_metrics import speech_metrics


def _guard(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # a metric must never fail a run row
        return {"error": f"{type(exc).__name__}: {exc}"}


def _speech(audio_path: str, script: str, transcriber) -> dict:
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")
    return speech_metrics(audio, sr, script, transcriber=transcriber)


def compute_run_metrics(
    *,
    script: str,
    draft: str | None,
    engine: str,
    content_type: str,
    audio_path: str,
    voice_stem_path: str | None,
    estimated_sec: float | None,
    transcriber,
) -> dict:
    speech_source = voice_stem_path or audio_path
    if transcriber is None:
        speech = {"skipped": "no transcriber"}
    else:
        speech = _guard(_speech, speech_source, script, transcriber)
    return {
        "script": _guard(script_metrics, script, draft=draft, engine=engine, content_type=content_type),
        "mix": _guard(mix_metrics, audio_path, estimated_sec=estimated_sec),
        "speech": speech,
        "speech_input": "voice_stem" if voice_stem_path else "mix",
    }


_COLUMNS = (
    ("slop/150w", "script", "slop_density", "{:.2f}"),
    ("edit ratio", "script", "edit_ratio", "{:.2f}"),
    ("CER", "speech", "cer", "{:.3f}"),
    ("gaps", "speech", "internal_gaps", "{:.1f}"),
    ("|1-dur ratio|", "mix", "duration_ratio", "{:.2f}"),
    ("LUFS (dual-mono)", "mix", "integrated_lufs_dual_mono", "{:.1f}"),
)


def _value(row: dict, group: str, key: str):
    value = (row.get("metrics") or {}).get(group, {}).get(key)
    if value is None or not isinstance(value, (int, float)):
        return None
    return abs(1.0 - value) if key == "duration_ratio" else float(value)


def summarize(rows: list[dict]) -> str:
    """Markdown table: one line per config with pass rate and metric medians."""
    configs = list(dict.fromkeys(row["config"] for row in rows))
    header = "| config | pass | " + " | ".join(c[0] for c in _COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(_COLUMNS) + 2)]
    for config in configs:
        mine = [r for r in rows if r["config"] == config]
        passed = [r for r in mine if r.get("ok")]
        cells = []
        for _label, group, key, fmt in _COLUMNS:
            values = [v for v in (_value(r, group, key) for r in passed) if v is not None]
            cells.append(fmt.format(statistics.median(values)) if values else "–")
        lines.append(f"| {config} | {len(passed)}/{len(mine)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
```

Add `compute_run_metrics, summarize` to `core/eval/__init__.py`.

- [ ] **Step 4: Run the report test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add `draft_script` to `AutoResult` (additive)**

In `core/auto_generate.py`, add a field to the `AutoResult` dataclass after `angle: str = ""`:

```python
    draft_script: str = ""
```

In `run()`'s final `return AutoResult(...)`, add the argument:

```python
        draft_script=outcome.draft_script,
```

Add one assertion to the existing successful-run test in `tests/unit/test_auto_generate.py`, the one asserting `Path(result.script_path).read_text() == CLEAN_SCRIPT` (around line 334). Add:

```python
        self.assertTrue(result.draft_script)  # draft is carried through for eval edit ratio
```

Run: `.venv/bin/python -m pytest tests/unit/test_auto_generate.py -v`
Expected: PASS (all existing tests plus the new assertion)

- [ ] **Step 6: Extend `scripts/eval_genres.py`**

Replace the argument parsing and the inner run loop. The existing `--genres/--band/--configs/--out/--runs` behaviour must be unchanged when none of the new flags are given.

```python
# additions to imports at top of scripts/eval_genres.py
import random

from core.eval.report import compute_run_metrics, summarize
from core.eval.testset import load_testset
```

```python
# inside main(), after the existing parser.add_argument calls
    parser.add_argument("--testset", default="", help="path to a testset TOML (overrides --genres/--band/--runs)")
    parser.add_argument("--metrics", action="store_true", help="compute script/speech/mix metrics per run")
    parser.add_argument("--export-stems", action="store_true", help="export voice stem so speech metrics avoid the bed")
    parser.add_argument("--engine", default="f5", help="TTS engine for AutoConfig.tts_engine (f5, qwen3, ...)")
```

Build the job list so both modes share one loop:

```python
    if args.testset:
        jobs = [(case.id, case.genre, case.band, case.seed) for case in load_testset(Path(args.testset))]
    else:
        jobs = [(f"{slug}-{index}", slug, args.band, None) for slug in slugs for index in range(args.runs)]

    transcriber = None
    if args.metrics:
        from core.eval.speech_metrics import WhisperTranscriber
        transcriber = WhisperTranscriber()
```

Replace the `for slug in slugs: for index in range(args.runs):` block with:

```python
        for job_id, slug, band, seed in jobs:
            out_dir = out_root / name / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                from core.pipeline import MeditationPipeline
                pipeline = MeditationPipeline()
                extra = {}
                if seed is not None:
                    extra["seed"] = seed
                if args.export_stems:
                    extra["do_export_stems"] = True
                result = run(
                    "",
                    genre=slug,
                    config=AutoConfig.from_genre(load_pack(slug), band=band, tts_engine=args.engine),
                    pipeline=pipeline,
                    rng=random.Random(seed) if seed is not None else None,
                    **extra,
                )
                for src_path in (result.audio_path, result.script_path, result.meta_path):
                    if Path(src_path).is_file():
                        shutil.move(str(src_path), str(out_dir / Path(src_path).name))
                audio_out = out_dir / Path(result.audio_path).name
                row = {
                    "config": name, "genre": slug, "case": job_id, "band": band, "seed": seed,
                    "ok": True,
                    "seconds": round(time.monotonic() - started, 1),
                    "angle": result.angle,
                    "estimated_sec": round(result.estimated_sec, 1),
                    "originality": round(result.originality, 3),
                    "advisories": [v.code for v in result.violations],
                    "audio": str(audio_out),
                }
                if args.metrics:
                    stems = getattr(pipeline, "last_stems", None) or {}
                    voice_stem = stems.get("voice")
                    if voice_stem and Path(voice_stem).is_file():
                        moved = out_dir / "narration_stem.wav"
                        shutil.move(voice_stem, moved)
                        voice_stem = str(moved)
                    metrics = compute_run_metrics(
                        script=result.script,
                        draft=result.draft_script or None,
                        engine=args.engine,
                        content_type=load_pack(slug).content_type,
                        audio_path=str(audio_out),
                        voice_stem_path=voice_stem,
                        estimated_sec=result.estimated_sec,
                        transcriber=transcriber,
                    )
                    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    row["metrics"] = metrics
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "config": name, "genre": slug, "case": job_id, "band": band, "seed": seed,
                    "ok": False,
                    "seconds": round(time.monotonic() - started, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "metrics"}), flush=True)
```

After the config loop, before the existing `results.json` write, add:

```python
    if transcriber is not None:
        transcriber.unload()
    (out_root / "summary.md").write_text(summarize(rows), encoding="utf-8")
    print("\n" + summarize(rows))
```

Keep the existing `results.json` write and the final success-count print.

Whisper must not stay resident while the next job's LLM loads, because of the 32 GB ceiling. So call `transcriber.unload()` right after each `compute_run_metrics` (the next job reloads it lazily). Add that line inside the `if args.metrics:` block after `metrics = compute_run_metrics(...)`.

- [ ] **Step 7: Verify the runner still parses and the old mode is unchanged**

Run: `.venv/bin/python scripts/eval_genres.py --help`
Expected: help text lists `--testset`, `--metrics`, `--export-stems`, `--engine` along with the original flags.

Run: `.venv/bin/python -m pytest tests/unit/ -q -k "bench or eval or auto_generate"`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add core/eval/report.py core/eval/__init__.py core/auto_generate.py scripts/eval_genres.py tests/unit/test_eval_report.py tests/unit/test_auto_generate.py
git commit -m "feat(eval): wire metrics, test set and summary into eval_genres runner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Blind A/B export and scoring

**Files:**
- Create: `core/eval/ab.py`
- Create: `scripts/ab_export.py`
- Create: `scripts/ab_score.py`
- Modify: `core/eval/__init__.py` (exports)
- Test: `tests/unit/test_eval_ab.py`

**Interfaces:**
- Consumes: `loudness()` from Task 3 (for `integrated_lufs_dual_mono`).
- Produces:
  - `POSITIONS = (("start", 0.10), ("middle", 0.50), ("end", 0.85))`.
  - `find_case_audio(run_dir: Path) -> dict[str, Path]`: maps case id (subdirectory name) to the main mix wav (the `.wav` whose stem has a sibling `.meta.json`).
  - `cut_excerpts(audio: np.ndarray, sr: int, *, excerpt_sec: float = 75.0) -> list[tuple[str, np.ndarray, bool]]`, returning `(position, excerpt, truncated)`.
  - `level_match(audio: np.ndarray, sr: int, target_lufs: float = -20.0) -> np.ndarray`.
  - `build_pairs(baseline_dir: Path, challenger_dir: Path, out_dir: Path, *, seed: int = 0, excerpt_sec: float = 75.0) -> int` (number of pairs). It writes `pairs/NN_A.wav`, `pairs/NN_B.wav`, `key.json` and `sheet.csv`.
  - `sign_test_p(wins: int, n: int) -> float` (one-sided, P[X ≥ wins] for X ~ Bin(n, 0.5)).
  - `score(ab_dir: Path) -> dict` with keys `challenger_wins, baseline_wins, ties, rate, p, verdict, by_position`. The verdict is `SHIP`, `NO-SHIP` or `INCONCLUSIVE`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_eval_ab.py
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.eval.ab import build_pairs, cut_excerpts, find_case_audio, level_match, score, sign_test_p
from core.eval.mix_metrics import loudness

SR = 8000


def _noise(seconds: float, amp: float, seed: int) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(SR * seconds))).astype(np.float32)


def _make_run(root: Path, cases: dict[str, float]) -> Path:
    for case, amp in cases.items():
        d = root / case
        d.mkdir(parents=True)
        sf.write(d / "med_1.wav", _noise(300, amp, hash(case) % 1000), SR)
        (d / "med_1.meta.json").write_text("{}")
        sf.write(d / "narration_stem.wav", _noise(1, amp, 1), SR)  # must be ignored
    return root


class TestExcerpts(unittest.TestCase):
    def test_three_positions(self):
        parts = cut_excerpts(_noise(300, 0.1, 0), SR, excerpt_sec=75)
        self.assertEqual([p[0] for p in parts], ["start", "middle", "end"])
        self.assertTrue(all(len(p[1]) == 75 * SR and not p[2] for p in parts))

    def test_short_file_excerpt_truncated(self):
        parts = cut_excerpts(_noise(60, 0.1, 0), SR, excerpt_sec=75)
        self.assertTrue(all(p[2] for p in parts))
        self.assertTrue(all(0 < len(p[1]) <= 60 * SR for p in parts))

    def test_level_match(self):
        matched = level_match(_noise(20, 0.3, 0), SR, target_lufs=-20.0)
        self.assertAlmostEqual(loudness(matched, SR)["integrated_lufs_dual_mono"], -20.0, delta=0.1)


class TestPairs(unittest.TestCase):
    def test_find_case_audio_ignores_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run(Path(tmp) / "a", {"body_scan-short": 0.1})
            self.assertEqual(find_case_audio(run)["body_scan-short"].name, "med_1.wav")

    def test_missing_case_raises_with_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.1})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.1})
            with self.assertRaisesRegex(ValueError, "y-short"):
                build_pairs(a, b, Path(tmp) / "ab")

    def test_build_pairs_is_blind_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _make_run(Path(tmp) / "a", {"x-short": 0.1, "y-short": 0.2})
            b = _make_run(Path(tmp) / "b", {"x-short": 0.3, "y-short": 0.05})
            n = build_pairs(a, b, Path(tmp) / "ab1", seed=7, excerpt_sec=30)
            build_pairs(a, b, Path(tmp) / "ab2", seed=7, excerpt_sec=30)
            self.assertEqual(n, 6)
            key1 = json.loads((Path(tmp) / "ab1" / "key.json").read_text())
            key2 = json.loads((Path(tmp) / "ab2" / "key.json").read_text())
            self.assertEqual(key1, key2)
            self.assertEqual({k["A"] for k in key1["pairs"].values()} | {k["B"] for k in key1["pairs"].values()},
                             {"baseline", "challenger"})
            with open(Path(tmp) / "ab1" / "sheet.csv") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, ["pair", "choice", "reason"])  # no arm/case leakage


class TestScore(unittest.TestCase):
    def test_sign_test_thresholds(self):
        self.assertLess(sign_test_p(17, 24), 0.05)
        self.assertGreater(sign_test_p(16, 24), 0.05)

    def _scored(self, choices: list[str]) -> dict:
        tmp = Path(tempfile.mkdtemp())
        pairs = {f"{i:02d}": {"A": "challenger", "B": "baseline", "case": "c", "position": "start"}
                 for i in range(len(choices))}
        (tmp / "key.json").write_text(json.dumps({"pairs": pairs}))
        with open(tmp / "sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "choice", "reason"])
            for i, c in enumerate(choices):
                w.writerow([f"{i:02d}", c, ""])
        return score(tmp)

    def test_ship(self):
        result = self._scored(["A"] * 17 + ["B"] * 7)
        self.assertEqual(result["verdict"], "SHIP")
        self.assertEqual(result["challenger_wins"], 17)

    def test_ties_excluded_and_inconclusive(self):
        result = self._scored(["A"] * 10 + ["B"] * 8 + ["same"] * 6)
        self.assertEqual(result["ties"], 6)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")

    def test_no_ship_when_baseline_clearly_wins(self):
        self.assertEqual(self._scored(["B"] * 18 + ["A"] * 6)["verdict"], "NO-SHIP")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_ab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.eval.ab'`

- [ ] **Step 3: Write `core/eval/ab.py`**

```python
# core/eval/ab.py
"""Blind A/B listening: excerpt cutting, level matching, pairing, scoring.

Convention: run A is the baseline, run B the challenger. On disk the arms are
shuffled per pair into files labelled A/B; only key.json knows which is which.
"""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from core.eval.mix_metrics import loudness

POSITIONS = (("start", 0.10), ("middle", 0.50), ("end", 0.85))
SHIP_RATE = 0.60
SHIP_P = 0.05


def find_case_audio(run_dir: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for case_dir in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()):
        mixes = [w for w in case_dir.glob("*.wav") if w.with_suffix(".meta.json").is_file()]
        if mixes:
            cases[case_dir.name] = mixes[0]
    return cases


def cut_excerpts(audio: np.ndarray, sr: int, *, excerpt_sec: float = 75.0) -> list[tuple[str, np.ndarray, bool]]:
    n = len(audio)
    length = int(excerpt_sec * sr)
    truncated = n < length
    out = []
    for name, frac in POSITIONS:
        if truncated:
            out.append((name, audio.copy(), True))
            continue
        start = min(max(0, int(frac * n) - length // 2), n - length)
        out.append((name, audio[start:start + length].copy(), False))
    return out


def level_match(audio: np.ndarray, sr: int, target_lufs: float = -20.0) -> np.ndarray:
    measured = loudness(audio, sr)["integrated_lufs_dual_mono"]
    if measured is None:
        return audio
    gain = 10.0 ** ((target_lufs - measured) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def build_pairs(
    baseline_dir: Path,
    challenger_dir: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    excerpt_sec: float = 75.0,
) -> int:
    import soundfile as sf

    base, chal = find_case_audio(baseline_dir), find_case_audio(challenger_dir)
    missing = sorted(set(base) ^ set(chal))
    if missing:
        raise ValueError(f"Cases present in only one run: {', '.join(missing)}")

    rng = random.Random(seed)
    pairs_dir = Path(out_dir) / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict] = {}
    index = 0
    for case in sorted(base):
        a_audio, sr_a = sf.read(base[case], dtype="float32")
        b_audio, sr_b = sf.read(chal[case], dtype="float32")
        if sr_a != sr_b:
            raise ValueError(f"{case}: sample rates differ ({sr_a} vs {sr_b})")
        for (pos, a_ex, a_tr), (_pos, b_ex, b_tr) in zip(
            cut_excerpts(a_audio, sr_a, excerpt_sec=excerpt_sec),
            cut_excerpts(b_audio, sr_b, excerpt_sec=excerpt_sec),
        ):
            arms = [("baseline", a_ex), ("challenger", b_ex)]
            rng.shuffle(arms)
            pair_id = f"{index:02d}"
            for label, (_arm, excerpt) in zip(("A", "B"), arms):
                sf.write(pairs_dir / f"{pair_id}_{label}.wav", level_match(excerpt, sr_a), sr_a)
            key[pair_id] = {"A": arms[0][0], "B": arms[1][0], "case": case, "position": pos,
                            "truncated": bool(a_tr or b_tr)}
            index += 1

    (Path(out_dir) / "key.json").write_text(json.dumps({"seed": seed, "pairs": key}, indent=2), encoding="utf-8")
    with open(Path(out_dir) / "sheet.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "choice", "reason"])
        for pair_id in key:
            writer.writerow([pair_id, "", ""])
    return index


def sign_test_p(wins: int, n: int) -> float:
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def score(ab_dir: Path) -> dict:
    key = json.loads((Path(ab_dir) / "key.json").read_text(encoding="utf-8"))["pairs"]
    challenger = baseline = ties = 0
    by_position: dict[str, dict[str, int]] = {}
    with open(Path(ab_dir) / "sheet.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            choice = row["choice"].strip().upper()
            entry = key[row["pair"]]
            bucket = by_position.setdefault(entry["position"], {"challenger": 0, "baseline": 0, "ties": 0})
            if choice in ("A", "B"):
                winner = entry[choice]
                if winner == "challenger":
                    challenger += 1
                else:
                    baseline += 1
                bucket[winner] += 1
            elif choice:
                ties += 1
                bucket["ties"] += 1
    n = challenger + baseline
    rate = challenger / n if n else 0.0
    p = sign_test_p(challenger, n)
    if n and rate >= SHIP_RATE and p < SHIP_P:
        verdict = "SHIP"
    elif n and (1 - rate) >= SHIP_RATE and sign_test_p(baseline, n) < SHIP_P:
        verdict = "NO-SHIP"
    else:
        verdict = "INCONCLUSIVE"
    return {"challenger_wins": challenger, "baseline_wins": baseline, "ties": ties,
            "rate": round(rate, 3), "p": round(p, 4), "verdict": verdict, "by_position": by_position}
```

Add `build_pairs, cut_excerpts, find_case_audio, level_match, score, sign_test_p` to `core/eval/__init__.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_eval_ab.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Write the two CLI scripts**

```python
# scripts/ab_export.py
"""Export blind A/B listening pairs from a baseline and a challenger eval run.

Usage:
    python scripts/ab_export.py --a var/eval/base/baseline --b var/eval/new/baseline --out var/eval/ab-001
Then listen to pairs/NN_A.wav vs NN_B.wav and fill sheet.csv (choice = A, B or same).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.ab import build_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="baseline config dir (contains <case>/ subdirs)")
    parser.add_argument("--b", required=True, help="challenger config dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--excerpt-sec", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n = build_pairs(Path(args.a), Path(args.b), Path(args.out), seed=args.seed, excerpt_sec=args.excerpt_sec)
    print(f"Wrote {n} blind pairs to {Path(args.out) / 'pairs'}. Fill {Path(args.out) / 'sheet.csv'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# scripts/ab_score.py
"""Score a filled A/B sheet: wins, ties, sign test and SHIP verdict.

Usage:
    python scripts/ab_score.py --ab var/eval/ab-001
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval.ab import score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ab", required=True)
    args = parser.parse_args()
    result = score(Path(args.ab))
    print(json.dumps(result, indent=2))
    print(f"\nVerdict: {result['verdict']} (challenger {result['challenger_wins']}"
          f" / baseline {result['baseline_wins']} / ties {result['ties']}, p={result['p']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Check whether `scripts/eval_genres.py` already inserts the repo root into `sys.path`. If it does it differently, match its approach in both new scripts.

Run: `.venv/bin/python scripts/ab_export.py --help && .venv/bin/python scripts/ab_score.py --help`
Expected: both print usage without errors.

- [ ] **Step 6: Commit**

```bash
git add core/eval/ab.py core/eval/__init__.py scripts/ab_export.py scripts/ab_score.py tests/unit/test_eval_ab.py
git commit -m "feat(eval): add blind A/B pair export and sign-test scoring

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Documentation, full-suite check and baseline run

**Files:**
- Modify: `docs/auto_generation/README.md` (new section "Evaluating a change")
- Modify: `CLAUDE.md` (Folder Map: add `core/eval/` and `eval/testset.toml`; Build & Test: add the eval commands)
- Modify: `docs/COMPONENT_REGISTRY.md` and `docs/TASK_ROUTING.md` (one row each for `core/eval/`)

**Interfaces:**
- Consumes: everything above.
- Produces: docs, plus a baseline under `var/eval/baseline-2026-10-06/` (gitignored).

- [ ] **Step 1: Write the README section**

Append to `docs/auto_generation/README.md`, before "## Where the pieces live":

````markdown
## Evaluating a change

Every quality change is judged two ways: automatic metrics catch regressions,
and a blind A/B listening test makes the ship decision. Metrics can disagree
with ears, so metrics reject bad output and ears choose between good outputs.

1. **Baseline** (before the change):
   ```bash
   caffeinate -i .venv/bin/python scripts/eval_genres.py --testset eval/testset.toml \
       --metrics --export-stems --configs "baseline=ollama:qwen3.8:27b|ollama:gemma4:31b" \
       --out var/eval/base
   ```
2. **Challenger:** apply the change (or set its flag) and run the same command
   with `--out var/eval/new`.
3. **Compare metrics:** read `summary.md` in both dirs. Columns: slop
   density, judge edit ratio, Whisper CER, internal gaps, |1 − duration
   ratio|, and dual-mono LUFS.
4. **Export blind pairs:**
   `python scripts/ab_export.py --a var/eval/base/baseline --b var/eval/new/baseline --out var/eval/ab-001`
5. **Listen:** play each `pairs/NN_A.wav` against `NN_B.wav` and fill
   `sheet.csv` (`A`, `B` or `same`, plus a one-word reason).
6. **Score:** `python scripts/ab_score.py --ab var/eval/ab-001`. SHIP means
   the challenger won ≥ 60% of non-tie pairs with p < 0.05 (e.g. 17 of 24).

F5 renders are nondeterministic even with fixed seeds (up to ~4 LU
short-term drift), so always compare two runs made in the same session.
Never compare against a stored historical number. Speech metrics run on the
exported voice stem when `--export-stems` is set, otherwise on the full mix.
Whisper is unloaded after every job so it never sits beside an LLM under the
32 GB ceiling.
````

- [ ] **Step 2: Update `CLAUDE.md`, `COMPONENT_REGISTRY.md` and `TASK_ROUTING.md`**

In the `CLAUDE.md` Folder Map, under `core/`, add the line
`│   ├── eval/                         # eval harness: script/speech/mix metrics, test set, blind A/B`,
and at the root level add `├── eval/testset.toml               # fixed 24-job evaluation set`.

In Build & Test, add:

```bash
.venv/bin/python scripts/eval_genres.py --testset eval/testset.toml --metrics --export-stems --configs "baseline=ollama:qwen3.8:27b|ollama:gemma4:31b" --out var/eval/base
python scripts/ab_export.py --a <base>/<config> --b <new>/<config> --out var/eval/ab-001 && python scripts/ab_score.py --ab var/eval/ab-001
```

In `docs/COMPONENT_REGISTRY.md` and `docs/TASK_ROUTING.md`, add a row pointing "evaluate a quality change / A/B test" at `core/eval/` and `docs/auto_generation/README.md#evaluating-a-change`, following the existing table format in each file.

- [ ] **Step 3: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: all tests pass (660 before R0, plus the new eval tests). Fix any failure before continuing.

- [ ] **Step 4: Commit the docs**

```bash
git add docs/auto_generation/README.md CLAUDE.md docs/COMPONENT_REGISTRY.md docs/TASK_ROUTING.md
git commit -m "docs: document the R0 evaluation workflow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Smoke the runner on one case, then start the baseline**

First, a one-case smoke test so a long run doesn't fail late. Create `var/eval/smoke.toml`:

```toml
bands = ["short"]

[[cases]]
genre = "body_scan"
seed = 1900
```

Run: `caffeinate -i .venv/bin/python scripts/eval_genres.py --testset var/eval/smoke.toml --metrics --export-stems --configs "baseline=ollama:qwen3.8:27b|ollama:gemma4:31b" --out var/eval/smoke`

Expected: `1/1 runs succeeded`. `var/eval/smoke/baseline/body_scan-short/metrics.json` has numeric `cer`, `integrated_lufs_dual_mono` ≈ −13 (the mono-at-−16 bug, measured as dual-mono), and `speech_input: "voice_stem"`.

If that passes, start the full baseline in the background. It's 24 jobs at several minutes each, so expect hours:

Run (background): `caffeinate -i .venv/bin/python scripts/eval_genres.py --testset eval/testset.toml --metrics --export-stems --configs "baseline=ollama:qwen3.8:27b|ollama:gemma4:31b" --out var/eval/baseline-2026-10-06`

Report the smoke metrics to the user and say the baseline is running. Don't block on it.
