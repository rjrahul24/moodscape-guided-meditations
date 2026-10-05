# Genre-Driven Meditation Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce auto-generation input to two clicks — pick a genre, pick a duration band — while guaranteeing no two generated meditations are the same.

**Architecture:** A curated TOML pack per genre supplies technique, arc, imagery angles, safety caveats and the deterministic fields (`content_type`, `music_tags`, `pause_ratio`). A planner stage expands a pack into a prose creative brief, the existing generator/judge/repair loop turns that into a validated script, and a new deterministic originality check guards against repetition. Background music is tagged automatically by librosa feature extraction.

**Tech Stack:** Python 3.11 (`tomllib` is stdlib, read-only), librosa 0.10.2, httpx, Gradio, Ollama. **No new dependencies.**

**Spec:** [docs/superpowers/specs/2026-09-17-genre-driven-generation-design.md](../specs/2026-09-17-genre-driven-generation-design.md)

## Global Constraints

- **No new dependencies.** Everything needed is already in `requirements.txt`. TF-IDF is hand-rolled (no scikit-learn); TOML is read with stdlib `tomllib` and written by a small hand-rolled writer (no `tomli-w`).
- **Python 3.11.7.** `tomllib` is read-only — never `import tomllib` expecting a `dumps`.
- **`MeditationPipeline.generate()` is untouched.** No task in this plan modifies `core/pipeline.py`, `core/mixer.py`, `core/audio_processor.py`, or any TTS engine.
- **Backward compatibility is mandatory.** `auto_generate.run(prompt, ...)` keeps its current positional signature. Every existing test in `tests/unit/` and `tests/integration/` must still pass after every task.
- **Test style:** `unittest.TestCase` classes, matching `tests/unit/test_auto_generate.py`. Run with `.venv/bin/python -m pytest tests/unit/<file> -v`.
- **Model defaults:** planner+writer `ollama:qwen3.8:27b`, judge `ollama:gemma4:31b`.
- **Originality thresholds:** cosine > `0.80` FATAL, `0.65`–`0.80` ADVISORY.
  (Recalibrated after Task 6 from measurements on realistic-length scripts — see
  the constants' docstring in Task 8 for the measured distribution.)
- **Fatal vs advisory must not be flattened.** Safety hard-blocks and malformed markers are FATAL; duration drift and style are ADVISORY. See CLAUDE.md "Top Gotchas".
- **`parse_judge_response()` stripping `<problems>` is load-bearing** for originality repair (commit `c372b18`). Do not remove or reorder it.
- **Commit after every task** using Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Commit to the current branch. **Never push.**

---

## File Structure

**Phase 1 — Music tagging** (independent; no LLM, no genre packs)

| File | Responsibility |
|---|---|
| `core/background_tags.py` | Extract librosa features, map to tags, cache in `tags.toml`, lazy per-file tagging |
| `scripts/tag_backgrounds.py` | Bulk / forced re-tag CLI |
| `core/background_picker.py` (modify) | `prefer_tags=` filtering |
| `assets/backgrounds/tags.toml` | Generated; `declared` entries hand-edited |

**Phase 2 — Originality** (independent; usable by the existing prompt path)

| File | Responsibility |
|---|---|
| `core/originality.py` | Corpus store, tokenisation, IDF, cosine, rare-run overlap |
| `core/script_gen/linter.py` (modify) | `check_originality()` producing `Violation`s |
| `core/auto_generate.py` (modify) | Append to corpus on success |

**Phase 3 — Genre core**

| File | Responsibility |
|---|---|
| `core/genres.py` | Load + validate packs, list families, angle rotation with history |
| `docs/genre_packs/*.toml` | 46 packs + README recording tagger thresholds |
| `core/script_gen/planner.py` | Pack + angle + avoid-list → prose brief |
| `core/script_gen/rules.py` (modify) | `build_planner_system_prompt()` |
| `core/script_gen/engine.py` (modify) | `unload()` on the ABC |
| `core/script_gen/adapters/openai_compat.py` (modify) | Ollama `unload()`, `models_available()` |
| `core/auto_generate.py` (modify) | Planner stage, genre path, preflight, unload calls |
| `core/auto_tab.py` / `core/streaming_run.py` (modify) | Genre dropdown, duration radio, genre-aware validation |
| `scripts/eval_genres.py` | Model-config comparison harness |

---

# PHASE 1 — Music tagging

Independent of everything else. Delivers: drop a file into `assets/backgrounds/`, it gets tagged automatically.

---

### Task 1: Audio feature extraction

**Files:**
- Create: `core/background_tags.py`
- Test: `tests/unit/test_background_tags.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `extract_features(path: str) -> dict[str, float]` returning keys
  `centroid`, `flatness`, `flux`, `onset_rate`, `dynamics`, `percussive`.
  All floats. `flatness` is scaled x1000 so thresholds are readable.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_background_tags.py`:

```python
"""Tests for automatic background-music tagging.

Uses synthetic signals written to temporary WAVs -- no audio fixtures, no
network, and deterministic across machines.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.background_tags import extract_features

SR = 22050


def _write(path: Path, y: np.ndarray) -> str:
    sf.write(str(path), y.astype(np.float32), SR)
    return str(path)


def _pure_tone(seconds: float = 90.0, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * freq * t)


def _white_noise(seconds: float = 90.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return 0.3 * rng.standard_normal(int(SR * seconds))


class ExtractFeaturesTest(unittest.TestCase):
    def test_returns_all_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "tone.wav", _pure_tone())
            features = extract_features(path)
        self.assertEqual(
            set(features),
            {"centroid", "flatness", "flux", "onset_rate", "dynamics", "percussive"},
        )
        for key, value in features.items():
            self.assertIsInstance(value, float, msg=key)

    def test_pure_tone_is_tonal_and_noise_is_flat(self):
        """Spectral flatness must separate a sine from white noise."""
        with tempfile.TemporaryDirectory() as tmp:
            tone = extract_features(_write(Path(tmp) / "t.wav", _pure_tone()))
            noise = extract_features(_write(Path(tmp) / "n.wav", _white_noise()))
        self.assertLess(tone["flatness"], 1.0)
        self.assertGreater(noise["flatness"], tone["flatness"] * 10)

    def test_pure_tone_has_low_flux(self):
        """A sustained sine has almost no spectral change over time."""
        with tempfile.TemporaryDirectory() as tmp:
            tone = extract_features(_write(Path(tmp) / "t.wav", _pure_tone()))
        self.assertLess(tone["flux"], 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.background_tags'`

- [ ] **Step 3: Write minimal implementation**

Create `core/background_tags.py`:

```python
"""Automatic tagging of background instrumentals.

Genre packs name the kind of bed they want ("warm", "sparse", "drone"); this
module works out which tracks match, from the audio itself.

Two tiers of tag, with different trustworthiness:

  * ``measured`` -- written here from librosa features. Regenerating a track's
    entry overwrites these.
  * ``declared`` -- instrument/source identity a human supplies ("piano",
    "flute", "nature"). Feature extraction cannot identify instruments
    reliably, so these are never written or overwritten by code.

Tagging is lazy and incremental: a newly added track is analysed once on first
use and cached by (name, size, mtime). The user's whole workflow is "drop a
file into assets/backgrounds/".
"""

import logging

logger = logging.getLogger(__name__)

# Analysis window. 60s is long enough for stable statistics and short enough
# that a first-use analysis is ~2s; starting 25% in skips intros and fades,
# which are not representative of the bed.
ANALYSIS_SECONDS = 60.0
ANALYSIS_OFFSET_FRACTION = 0.25
ANALYSIS_SR = 22050


def extract_features(path: str) -> dict[str, float]:
    """Measure six spectral/temporal features of one track.

    Returns:
        centroid    -- Hz, brightness.
        flatness    -- spectral flatness x1000 (noise-like vs tonal).
        flux        -- mean onset strength; how much the spectrum moves.
        onset_rate  -- detected onsets per second.
        dynamics    -- RMS p95/p5 ratio; how much the level swings.
        percussive  -- fraction of energy that is percussive (HPSS).
    """
    import numpy as np
    import librosa

    duration = librosa.get_duration(path=path)
    offset = max(0.0, duration * ANALYSIS_OFFSET_FRACTION)
    y, sr = librosa.load(
        path,
        sr=ANALYSIS_SR,
        mono=True,
        offset=offset,
        duration=ANALYSIS_SECONDS,
    )

    spectrum = np.abs(librosa.stft(y))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    rms = librosa.feature.rms(S=spectrum)[0]
    _harmonic, percussive = librosa.effects.hpss(y)

    analysed_sec = len(y) / float(sr) if len(y) else 1.0

    return {
        "centroid": float(
            np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sr))
        ),
        "flatness": float(
            np.mean(librosa.feature.spectral_flatness(S=spectrum)) * 1000.0
        ),
        "flux": float(
            np.mean(
                librosa.onset.onset_strength(
                    S=librosa.power_to_db(spectrum**2), sr=sr
                )
            )
        ),
        "onset_rate": float(len(onsets) / analysed_sec),
        "dynamics": float(
            np.percentile(rms, 95) / (np.percentile(rms, 5) + 1e-9)
        ),
        "percussive": float(
            np.sum(percussive**2) / (np.sum(y**2) + 1e-9)
        ),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add core/background_tags.py tests/unit/test_background_tags.py
git commit -m "feat(background_tags): extract spectral features from a track"
```

---

### Task 2: Map features to tags

**Files:**
- Modify: `core/background_tags.py`
- Test: `tests/unit/test_background_tags.py`

**Interfaces:**
- Consumes: `extract_features()` from Task 1 (only its dict shape — this task's function takes a plain dict and needs no audio)
- Produces:
  - `MEASURED_VOCAB: frozenset[str]` — every tag this module can emit
  - `DECLARED_VOCAB: frozenset[str]` — instrument tags humans may write
  - `tags_from_features(features: dict[str, float]) -> list[str]` — sorted

**Thresholds** are calibrated against the measured distribution of the 20
existing tracks. They are recorded here and in `docs/genre_packs/README.md`
(Task 11) so they can be re-derived.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_background_tags.py` (before `if __name__`):

```python
from core.background_tags import (
    DECLARED_VOCAB,
    MEASURED_VOCAB,
    tags_from_features,
)


def _features(**overrides) -> dict:
    """A mid-range track that trips no threshold, plus overrides."""
    base = {
        "centroid": 850.0,
        "flatness": 0.5,
        "flux": 1.5,
        "onset_rate": 3.0,
        "dynamics": 2.5,
        "percussive": 0.02,
    }
    base.update(overrides)
    return base


class TagsFromFeaturesTest(unittest.TestCase):
    def test_mid_range_track_gets_only_a_brightness_tag(self):
        """Brightness always emits exactly one tag; nothing else should fire.

        Every track has a brightness, so 'dark'/'warm'/'bright' is a partition
        rather than a threshold pair. The other five features only tag
        extremes, so a mid-range track trips none of them.
        """
        self.assertEqual(tags_from_features(_features()), ["warm"])

    def test_brightness_bands(self):
        self.assertIn("dark", tags_from_features(_features(centroid=300.0)))
        self.assertIn("bright", tags_from_features(_features(centroid=1300.0)))
        self.assertIn("warm", tags_from_features(_features(centroid=800.0)))

    def test_low_flux_is_a_drone(self):
        self.assertIn("drone", tags_from_features(_features(flux=0.3)))

    def test_high_flux_is_evolving(self):
        self.assertIn("evolving", tags_from_features(_features(flux=2.2)))

    def test_near_silent_drone_is_never_labelled_busy(self):
        """The onset detector fires on noise floor in very quiet drones.

        One real track reports 5.47 onsets/s with flux 0.29 and zero detected
        beats. Without gating 'busy'/'sparse' on flux, the calmest beds in the
        library get labelled the busiest.
        """
        tags = tags_from_features(_features(flux=0.29, onset_rate=5.47))
        self.assertIn("drone", tags)
        self.assertNotIn("busy", tags)
        self.assertNotIn("sparse", tags)

    def test_onset_bands_apply_when_flux_is_high_enough(self):
        self.assertIn("sparse", tags_from_features(_features(flux=1.5, onset_rate=0.9)))
        self.assertIn("busy", tags_from_features(_features(flux=1.5, onset_rate=7.0)))

    def test_tonal_and_textured(self):
        self.assertIn("tonal", tags_from_features(_features(flatness=0.02)))
        self.assertIn("textured", tags_from_features(_features(flatness=4.6)))

    def test_struck_and_sustained(self):
        self.assertIn("struck", tags_from_features(_features(percussive=0.047)))
        self.assertIn("sustained", tags_from_features(_features(percussive=0.006)))

    def test_steady_and_dynamic(self):
        self.assertIn("steady", tags_from_features(_features(dynamics=1.7)))
        self.assertIn("dynamic", tags_from_features(_features(dynamics=4.4)))

    def test_every_emitted_tag_is_in_the_vocabulary(self):
        extremes = [
            _features(centroid=100.0, flux=0.2, flatness=0.0, percussive=0.001, dynamics=1.0),
            _features(centroid=3000.0, flux=5.0, flatness=9.0, percussive=0.5, dynamics=9.0, onset_rate=12.0),
        ]
        for features in extremes:
            for tag in tags_from_features(features):
                self.assertIn(tag, MEASURED_VOCAB)

    def test_vocabularies_do_not_overlap(self):
        self.assertEqual(MEASURED_VOCAB & DECLARED_VOCAB, frozenset())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: FAIL with `ImportError: cannot import name 'MEASURED_VOCAB'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/background_tags.py`:

```python
# --- Tag vocabulary and thresholds --------------------------------------
#
# Thresholds are calibrated against the measured distribution of the 20
# tracks in assets/backgrounds/ as of 2026-09-17, chosen so each band holds a
# meaningful share of the library rather than being empty or catching
# everything. Re-derive with `python scripts/tag_backgrounds.py --report`.

MEASURED_VOCAB = frozenset(
    {
        "dark", "warm", "bright",
        "drone", "evolving",
        "sparse", "busy",
        "tonal", "textured",
        "struck", "sustained",
        "steady", "dynamic",
    }
)

# Instrument/source identity. Feature extraction cannot determine these
# reliably, so they are only ever written by a human.
DECLARED_VOCAB = frozenset(
    {"piano", "flute", "strings", "nature", "voice", "bells"}
)

CENTROID_DARK_BELOW = 600.0
CENTROID_BRIGHT_ABOVE = 1050.0

FLUX_DRONE_BELOW = 0.6
FLUX_EVOLVING_ABOVE = 1.9

ONSET_SPARSE_BELOW = 1.5
ONSET_BUSY_ABOVE = 5.0

FLATNESS_TONAL_BELOW = 0.15
FLATNESS_TEXTURED_ABOVE = 1.5

PERCUSSIVE_STRUCK_ABOVE = 0.040
PERCUSSIVE_SUSTAINED_BELOW = 0.010

DYNAMICS_STEADY_BELOW = 2.0
DYNAMICS_DYNAMIC_ABOVE = 3.3


def tags_from_features(features: dict[str, float]) -> list[str]:
    """Map measured features onto the measured-tag vocabulary.

    Returns a sorted list so output is stable across runs and diffs of
    tags.toml stay readable.
    """
    tags: set[str] = set()

    centroid = features["centroid"]
    if centroid < CENTROID_DARK_BELOW:
        tags.add("dark")
    elif centroid > CENTROID_BRIGHT_ABOVE:
        tags.add("bright")
    else:
        tags.add("warm")

    flux = features["flux"]
    if flux < FLUX_DRONE_BELOW:
        tags.add("drone")
    elif flux > FLUX_EVOLVING_ABOVE:
        tags.add("evolving")

    # Onset detection is meaningless on near-silent drone material: the
    # detector fires on noise floor, so a bed with flux 0.29 can report 5.47
    # onsets/s and would otherwise be tagged the busiest track in the
    # library. Gate on the same threshold that defines a drone.
    if flux >= FLUX_DRONE_BELOW:
        onset_rate = features["onset_rate"]
        if onset_rate < ONSET_SPARSE_BELOW:
            tags.add("sparse")
        elif onset_rate > ONSET_BUSY_ABOVE:
            tags.add("busy")

    flatness = features["flatness"]
    if flatness < FLATNESS_TONAL_BELOW:
        tags.add("tonal")
    elif flatness > FLATNESS_TEXTURED_ABOVE:
        tags.add("textured")

    percussive = features["percussive"]
    if percussive > PERCUSSIVE_STRUCK_ABOVE:
        tags.add("struck")
    elif percussive < PERCUSSIVE_SUSTAINED_BELOW:
        tags.add("sustained")

    dynamics = features["dynamics"]
    if dynamics < DYNAMICS_STEADY_BELOW:
        tags.add("steady")
    elif dynamics > DYNAMICS_DYNAMIC_ABOVE:
        tags.add("dynamic")

    return sorted(tags)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add core/background_tags.py tests/unit/test_background_tags.py
git commit -m "feat(background_tags): map features to a two-tier tag vocabulary"
```

---

### Task 3: Lazy tagging with a persistent cache

**Files:**
- Modify: `core/background_tags.py`
- Create: `scripts/tag_backgrounds.py`
- Test: `tests/unit/test_background_tags.py`

**Interfaces:**
- Consumes: `extract_features()` (Task 1), `tags_from_features()` (Task 2)
- Produces:
  - `TAGS_PATH: Path` — `assets/backgrounds/tags.toml`
  - `load_tags(tags_path: Path | None = None) -> dict[str, dict]` — parsed file; `{}` when absent
  - `write_tags(entries: dict[str, dict], tags_path: Path | None = None) -> None`
  - `tags_for(paths: Sequence[str], tags_path: Path | None = None, extractor=None) -> dict[str, list[str]]`
    — path → combined `measured + declared`; analyses and caches anything missing or stale

The cache key is `(size, mtime)`. `declared` entries are keyed by filename and
are **never** overwritten by re-analysis.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_background_tags.py` (before `if __name__`):

```python
from core.background_tags import load_tags, tags_for, write_tags


class FakeExtractor:
    """Stands in for extract_features; counts how often each path is analysed."""

    def __init__(self, features: dict):
        self._features = features
        self.calls: list[str] = []

    def __call__(self, path: str) -> dict:
        self.calls.append(path)
        return dict(self._features)


class TagCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tags_path = self.root / "tags.toml"
        self.track = self.root / "track.mp3"
        self.track.write_bytes(b"not really audio")
        self.extractor = FakeExtractor(_features(centroid=300.0, flux=0.3))

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_loads_as_empty(self):
        self.assertEqual(load_tags(self.tags_path), {})

    def test_first_use_analyses_and_caches(self):
        result = tags_for(
            [str(self.track)], tags_path=self.tags_path, extractor=self.extractor
        )
        self.assertEqual(len(self.extractor.calls), 1)
        self.assertIn("dark", result[str(self.track)])
        self.assertTrue(self.tags_path.is_file())

    def test_second_use_reads_the_cache(self):
        for _ in range(2):
            tags_for(
                [str(self.track)], tags_path=self.tags_path, extractor=self.extractor
            )
        self.assertEqual(len(self.extractor.calls), 1)

    def test_changed_file_is_reanalysed(self):
        tags_for([str(self.track)], tags_path=self.tags_path, extractor=self.extractor)
        self.track.write_bytes(b"different content entirely")
        tags_for([str(self.track)], tags_path=self.tags_path, extractor=self.extractor)
        self.assertEqual(len(self.extractor.calls), 2)

    def test_declared_tags_survive_reanalysis(self):
        tags_for([str(self.track)], tags_path=self.tags_path, extractor=self.extractor)
        entries = load_tags(self.tags_path)
        entries["track.mp3"]["declared"] = ["piano"]
        write_tags(entries, self.tags_path)

        self.track.write_bytes(b"changed so it re-analyses")
        result = tags_for(
            [str(self.track)], tags_path=self.tags_path, extractor=self.extractor
        )

        self.assertIn("piano", result[str(self.track)])
        self.assertIn("piano", load_tags(self.tags_path)["track.mp3"]["declared"])

    def test_extraction_failure_yields_no_tags_and_does_not_raise(self):
        """One unreadable file must not break tagging for the rest."""

        def boom(path: str) -> dict:
            raise RuntimeError("corrupt file")

        result = tags_for(
            [str(self.track)], tags_path=self.tags_path, extractor=boom
        )
        self.assertEqual(result[str(self.track)], [])

    def test_roundtrip_through_toml(self):
        write_tags(
            {"a.mp3": {"measured": ["dark", "drone"], "declared": ["piano"],
                       "size": 12, "mtime": 34.5}},
            self.tags_path,
        )
        entries = load_tags(self.tags_path)
        self.assertEqual(entries["a.mp3"]["measured"], ["dark", "drone"])
        self.assertEqual(entries["a.mp3"]["declared"], ["piano"])
        self.assertEqual(entries["a.mp3"]["size"], 12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_tags'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/background_tags.py`:

```python
import tomllib
from collections.abc import Sequence
from pathlib import Path

TAGS_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets" / "backgrounds" / "tags.toml"
)


def _resolve(tags_path: Path | None) -> Path:
    return tags_path if tags_path is not None else TAGS_PATH


def load_tags(tags_path: Path | None = None) -> dict[str, dict]:
    """Read tags.toml. Returns {} when the file is absent or unreadable.

    A corrupt tags file must not break generation: the worst case is that
    every track is re-analysed, which is slow but correct.
    """
    path = _resolve(tags_path)
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read %s; re-analysing all tracks.", path)
        return {}
    return data.get("tracks", {})


def _toml_string(value: str) -> str:
    """Quote a string for TOML, escaping backslashes and double quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def write_tags(entries: dict[str, dict], tags_path: Path | None = None) -> None:
    """Write tags.toml.

    Hand-rolled rather than using a TOML writer: stdlib tomllib is read-only,
    and the structure here is a flat table of string lists and numbers, which
    does not justify a new dependency.
    """
    path = _resolve(tags_path)
    lines = [
        "# Generated by core/background_tags.py.",
        "#",
        "# 'measured' is written from audio analysis and is overwritten when a",
        "# track changes. 'declared' is yours: instrument and source identity",
        "# that analysis cannot determine. It is never overwritten.",
        "",
    ]
    for name in sorted(entries):
        entry = entries[name]
        lines.append(f"[tracks.{_toml_string(name)}]")
        lines.append(f"measured = {_toml_list(entry.get('measured', []))}")
        lines.append(f"declared = {_toml_list(entry.get('declared', []))}")
        lines.append(f"size = {int(entry.get('size', 0))}")
        lines.append(f"mtime = {float(entry.get('mtime', 0.0)):.3f}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tags_for(
    paths: Sequence[str],
    tags_path: Path | None = None,
    extractor=None,
) -> dict[str, list[str]]:
    """Return combined measured+declared tags for each path, tagging as needed.

    Analyses only tracks with no cache entry or whose (size, mtime) has
    changed, then writes the cache back once. A track that cannot be analysed
    gets an empty tag list rather than raising -- one corrupt file must not
    stop a generation run.
    """
    analyse = extractor if extractor is not None else extract_features
    entries = load_tags(tags_path)
    dirty = False
    result: dict[str, list[str]] = {}

    for raw_path in paths:
        path = Path(raw_path)
        name = path.name
        entry = entries.get(name)

        try:
            stat = path.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            logger.warning("Cannot stat background %s; skipping tags.", raw_path)
            result[raw_path] = []
            continue

        stale = (
            entry is None
            or int(entry.get("size", -1)) != size
            or abs(float(entry.get("mtime", -1.0)) - mtime) > 0.001
        )

        if stale:
            # Preserve the human's declared tags across re-analysis.
            declared = list(entry.get("declared", [])) if entry else []
            try:
                measured = tags_from_features(analyse(raw_path))
            except Exception:
                logger.warning(
                    "Could not analyse background %s; leaving it untagged.",
                    raw_path,
                    exc_info=True,
                )
                result[raw_path] = []
                continue
            entry = {
                "measured": measured,
                "declared": declared,
                "size": size,
                "mtime": mtime,
            }
            entries[name] = entry
            dirty = True

        result[raw_path] = sorted(
            set(entry.get("measured", [])) | set(entry.get("declared", []))
        )

    if dirty:
        write_tags(entries, tags_path)

    return result
```

Move the `import tomllib`, `from collections.abc import Sequence` and
`from pathlib import Path` lines to the top of the module with the existing
`import logging`, so the file has one import block.

Create `scripts/tag_backgrounds.py`:

```python
#!/usr/bin/env python
"""Bulk-tag the background library, or print the measured feature table.

Tagging happens automatically on first use (core/background_tags.tags_for),
so this script is never required. It exists for two cases: re-tagging
everything after changing a threshold, and printing the raw feature
distribution used to calibrate those thresholds.

    python scripts/tag_backgrounds.py            # tag anything new or changed
    python scripts/tag_backgrounds.py --force    # re-analyse every track
    python scripts/tag_backgrounds.py --report   # print the feature table
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.background_tags import (  # noqa: E402
    TAGS_PATH,
    extract_features,
    load_tags,
    tags_for,
    write_tags,
)
from core.upload_music import scan_backgrounds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-analyse every track"
    )
    parser.add_argument(
        "--report", action="store_true", help="print the raw feature table"
    )
    args = parser.parse_args()

    pool = scan_backgrounds()
    if not pool:
        print("No background tracks found.", file=sys.stderr)
        return 1
    paths = [path for _label, path in pool]

    if args.report:
        header = (
            f"{'track':44} {'centroid':>9} {'flatness':>9} {'flux':>7} "
            f"{'onset/s':>8} {'dynamics':>9} {'percussive':>11}"
        )
        print(header)
        for path in paths:
            f = extract_features(path)
            print(
                f"{Path(path).name[:44]:44} {f['centroid']:9.0f} "
                f"{f['flatness']:9.2f} {f['flux']:7.2f} {f['onset_rate']:8.2f} "
                f"{f['dynamics']:9.1f} {f['percussive']:11.3f}"
            )
        return 0

    if args.force:
        # Drop the cache keys so every track re-analyses, but keep declared
        # tags -- those are the human's and are not regenerable.
        entries = load_tags()
        for entry in entries.values():
            entry["size"] = -1
        write_tags(entries)

    tags = tags_for(paths)
    for path in paths:
        print(f"{Path(path).name[:50]:50} {', '.join(tags[path]) or '-'}")
    print(f"\nWrote {TAGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_background_tags.py -v`
Expected: PASS (20 tests)

- [ ] **Step 5: Tag the real library and inspect the result**

```bash
.venv/bin/python scripts/tag_backgrounds.py
```

Expected: every one of the 20 tracks prints with at least a brightness tag,
and `assets/backgrounds/tags.toml` is created. Sanity-check that
`mondamusic-meditation-512846.mp3` is tagged `drone` and **not** `busy` —
that is the noise-floor hazard from Task 2 verified against real audio.

- [ ] **Step 6: Commit**

```bash
git add core/background_tags.py scripts/tag_backgrounds.py \
        tests/unit/test_background_tags.py assets/backgrounds/tags.toml
git commit -m "feat(background_tags): lazy tagging with a persistent cache"
```

---

### Task 4: Genre-aware background selection

**Files:**
- Modify: `core/background_picker.py`
- Test: `tests/unit/test_background_picker.py`

**Interfaces:**
- Consumes: `tags_for()` (Task 3)
- Produces: `pick_background(*, scan=None, exclude=(), rng=None, prefer_tags=(), tag_lookup=None) -> tuple[str, str]`

`prefer_tags` narrows the pool to tracks carrying **all** requested tags.
When nothing matches, the full pool is used — variety is a preference, not a
reason to fail a job, matching the existing `exclude` behaviour.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_background_picker.py`:

```python
class PreferTagsTest(unittest.TestCase):
    POOL = [
        ("Calm Drone — 10:00", "/bg/drone.mp3"),
        ("Bright Piano — 08:00", "/bg/piano.mp3"),
        ("Dark Pad — 12:00", "/bg/pad.mp3"),
    ]
    TAGS = {
        "/bg/drone.mp3": ["drone", "dark", "sustained"],
        "/bg/piano.mp3": ["bright", "struck", "evolving"],
        "/bg/pad.mp3": ["dark", "sustained", "steady"],
    }

    def _scan(self):
        return list(self.POOL)

    def _lookup(self, paths):
        return {p: self.TAGS[p] for p in paths}

    def test_prefers_tracks_carrying_every_requested_tag(self):
        for _ in range(20):
            _label, path = pick_background(
                scan=self._scan,
                prefer_tags=["dark", "sustained"],
                tag_lookup=self._lookup,
            )
            self.assertIn(path, {"/bg/drone.mp3", "/bg/pad.mp3"})

    def test_single_tag_narrows_correctly(self):
        _label, path = pick_background(
            scan=self._scan, prefer_tags=["struck"], tag_lookup=self._lookup
        )
        self.assertEqual(path, "/bg/piano.mp3")

    def test_unmatchable_tags_fall_back_to_the_full_pool(self):
        _label, path = pick_background(
            scan=self._scan,
            prefer_tags=["rhythmic", "brass"],
            tag_lookup=self._lookup,
        )
        self.assertIn(path, {p for _, p in self.POOL})

    def test_exclude_still_applies_within_a_tag_filter(self):
        _label, path = pick_background(
            scan=self._scan,
            prefer_tags=["dark"],
            exclude=["/bg/pad.mp3"],
            tag_lookup=self._lookup,
        )
        self.assertEqual(path, "/bg/drone.mp3")

    def test_no_prefer_tags_never_calls_the_tag_lookup(self):
        """Tagging is lazy; an untagged library must stay fast when unused."""
        calls = []

        def spy(paths):
            calls.append(paths)
            return {}

        pick_background(scan=self._scan, tag_lookup=spy)
        self.assertEqual(calls, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_background_picker.py -v`
Expected: FAIL with `TypeError: pick_background() got an unexpected keyword argument 'prefer_tags'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `pick_background` in `core/background_picker.py`:

```python
def pick_background(
    *,
    scan=None,
    exclude: Sequence[str] = (),
    rng: random.Random | None = None,
    prefer_tags: Sequence[str] = (),
    tag_lookup=None,
) -> tuple[str, str]:
    """Choose one background instrumental at random.

    Args:
        scan: Zero-arg callable returning [(label, path), ...]. Defaults to
            scan_backgrounds. Injected in tests to avoid reading real audio.
        exclude: Paths of recently used tracks to avoid. If excluding them
            would leave nothing, the full pool is used instead — variety is a
            preference, not a reason to fail a job.
        rng: Inject a seeded Random for reproducible selection.
        prefer_tags: Only consider tracks carrying ALL of these tags. If that
            leaves nothing, the tag filter is dropped. Genre packs supply
            these so a running meditation does not land on a sleep drone.
        tag_lookup: Callable mapping [path, ...] -> {path: [tag, ...]}.
            Defaults to background_tags.tags_for. Only called when
            prefer_tags is non-empty, so an untagged library costs nothing.

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

    candidates = list(pool)

    if prefer_tags:
        lookup = tag_lookup
        if lookup is None:
            from core.background_tags import tags_for as lookup
        wanted = set(prefer_tags)
        tags = lookup([path for _label, path in candidates])
        tagged = [
            entry
            for entry in candidates
            if wanted.issubset(set(tags.get(entry[1], ())))
        ]
        if tagged:
            candidates = tagged
        else:
            logger.info(
                "No background matches tags %s; using the full library.",
                sorted(wanted),
            )

    excluded = set(exclude)
    candidates = [entry for entry in candidates if entry[1] not in excluded] or candidates

    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
```

Add `import logging` and `logger = logging.getLogger(__name__)` at the top of
`core/background_picker.py` if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_background_picker.py tests/unit/test_background_tags.py -v`
Expected: PASS — including every pre-existing `test_background_picker.py` test.

- [ ] **Step 5: Commit**

```bash
git add core/background_picker.py tests/unit/test_background_picker.py
git commit -m "feat(background_picker): filter the pool by genre music tags"
```

**Phase 1 is complete.** Dropping a file into `assets/backgrounds/` now tags
it automatically on first use, and callers can request a kind of bed.

---

# PHASE 2 — Originality

Independent of Phase 1 and of genre packs. Works on the existing prompt-driven
path immediately; Phase 3 adds the proactive layer on top.

---

### Task 5: The script corpus

**Files:**
- Create: `core/originality.py`
- Test: `tests/unit/test_originality.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CorpusEntry` — frozen dataclass with `script_id: str`, `genre: str`, `angle: str`, `created: str`, `text: str`
  - `add_to_corpus(script: str, *, genre: str, angle: str = "", corpus_dir: Path | None = None) -> str` (returns `script_id`)
  - `load_corpus(*, genre: str | None = None, limit: int | None = None, corpus_dir: Path | None = None) -> list[CorpusEntry]` (most recent first)
  - `recent_angles(genre: str, *, limit: int = 3, corpus_dir: Path | None = None) -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_originality.py`:

```python
"""Tests for the originality corpus and similarity math.

No network, no models, no audio. Every fixture is inline text.
"""

import tempfile
import unittest
from pathlib import Path

from core.originality import (
    add_to_corpus,
    load_corpus,
    recent_angles,
)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_corpus_loads_as_empty_list(self):
        self.assertEqual(load_corpus(corpus_dir=self.dir), [])

    def test_added_script_round_trips(self):
        add_to_corpus("the tide withdraws", genre="sleep", angle="tidal",
                      corpus_dir=self.dir)
        entries = load_corpus(corpus_dir=self.dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, "the tide withdraws")
        self.assertEqual(entries[0].genre, "sleep")
        self.assertEqual(entries[0].angle, "tidal")

    def test_ids_are_unique_even_within_one_second(self):
        ids = {
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
            for i in range(25)
        }
        self.assertEqual(len(ids), 25)

    def test_genre_filter_partitions_the_corpus(self):
        add_to_corpus("a", genre="sleep", corpus_dir=self.dir)
        add_to_corpus("b", genre="focus", corpus_dir=self.dir)
        add_to_corpus("c", genre="sleep", corpus_dir=self.dir)
        self.assertEqual(len(load_corpus(genre="sleep", corpus_dir=self.dir)), 2)
        self.assertEqual(len(load_corpus(genre="focus", corpus_dir=self.dir)), 1)

    def test_entries_come_back_most_recent_first(self):
        for i in range(5):
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
        texts = [e.text for e in load_corpus(genre="sleep", corpus_dir=self.dir)]
        self.assertEqual(texts[0], "script 4")

    def test_limit_takes_the_most_recent(self):
        for i in range(10):
            add_to_corpus(f"script {i}", genre="sleep", corpus_dir=self.dir)
        entries = load_corpus(genre="sleep", limit=3, corpus_dir=self.dir)
        self.assertEqual([e.text for e in entries],
                         ["script 9", "script 8", "script 7"])

    def test_recent_angles_are_most_recent_first_and_deduplicated(self):
        for angle in ["tidal", "staircase", "tidal", "meadow"]:
            add_to_corpus("x", genre="sleep", angle=angle, corpus_dir=self.dir)
        self.assertEqual(
            recent_angles("sleep", limit=3, corpus_dir=self.dir),
            ["meadow", "tidal", "staircase"],
        )

    def test_a_missing_script_file_is_skipped_not_fatal(self):
        script_id = add_to_corpus("gone", genre="sleep", corpus_dir=self.dir)
        (self.dir / "scripts" / f"{script_id}.txt").unlink()
        self.assertEqual(load_corpus(genre="sleep", corpus_dir=self.dir), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.originality'`

- [ ] **Step 3: Write minimal implementation**

Create `core/originality.py`:

```python
"""Guarantee that no two generated meditations are the same.

Two layers cooperate. This module supplies the machinery for both:

  * Proactive -- ``recent_angles`` and ``avoid_terms`` feed the planner an
    explicit "do not reuse this" list before a word is written.
  * Reactive  -- ``assess`` measures a finished script against the corpus so
    ``linter.check_originality`` can turn the result into a Violation that the
    existing repair loop already knows how to act on.

The hard part is not flagging generic meditation language. Every script
legitimately says "notice your breath". TF-IDF self-calibrates against exactly
that: terms appearing across the whole corpus get near-zero weight while
distinctive ones keep full weight, so no hand-maintained stoplist is needed.

That dictates an asymmetry worth stating plainly: IDF is computed over the
ENTIRE corpus (all genres), because that is what learns which phrases are
generic; similarity is measured WITHIN one genre, because that is where
collisions actually happen.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "var" / "originality"

_INDEX_NAME = "index.json"
_SCRIPTS_DIRNAME = "scripts"


@dataclass(frozen=True)
class CorpusEntry:
    """One previously generated script."""

    script_id: str
    genre: str
    angle: str
    created: str
    text: str


def _resolve(corpus_dir: Path | None) -> Path:
    return corpus_dir if corpus_dir is not None else CORPUS_DIR


def _read_index(root: Path) -> list[dict]:
    path = root / _INDEX_NAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Corpus index at %s is unreadable; treating as empty.", path)
        return []
    return data if isinstance(data, list) else []


def _write_index(root: Path, records: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / _INDEX_NAME).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def add_to_corpus(
    script: str,
    *,
    genre: str,
    angle: str = "",
    corpus_dir: Path | None = None,
) -> str:
    """Record a rendered script so later runs can be checked against it.

    Returns the new entry's script_id.
    """
    root = _resolve(corpus_dir)
    scripts_dir = root / _SCRIPTS_DIRNAME
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Microsecond precision plus a random suffix: a fast test loop can add
    # many entries inside one microsecond tick on some platforms, and a
    # collision would silently overwrite a prior script.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script_id = f"{stamp}-{os.urandom(3).hex()}"

    (scripts_dir / f"{script_id}.txt").write_text(script, encoding="utf-8")

    records = _read_index(root)
    records.append(
        {
            "script_id": script_id,
            "genre": genre,
            "angle": angle,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_index(root, records)
    return script_id


def load_corpus(
    *,
    genre: str | None = None,
    limit: int | None = None,
    corpus_dir: Path | None = None,
) -> list[CorpusEntry]:
    """Load corpus entries, most recent first.

    A record whose script file has been deleted is skipped rather than
    raising: the corpus is machine-local cache, and a missing file must not
    stop a generation run.
    """
    root = _resolve(corpus_dir)
    scripts_dir = root / _SCRIPTS_DIRNAME

    entries: list[CorpusEntry] = []
    for record in reversed(_read_index(root)):
        if genre is not None and record.get("genre") != genre:
            continue
        script_id = record.get("script_id", "")
        path = scripts_dir / f"{script_id}.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            logger.debug("Corpus script %s is missing; skipping.", path)
            continue
        entries.append(
            CorpusEntry(
                script_id=script_id,
                genre=record.get("genre", ""),
                angle=record.get("angle", ""),
                created=record.get("created", ""),
                text=text,
            )
        )
        if limit is not None and len(entries) >= limit:
            break
    return entries


def recent_angles(
    genre: str, *, limit: int = 3, corpus_dir: Path | None = None
) -> list[str]:
    """Angles most recently used for a genre, most recent first, deduplicated.

    Reads the index only -- no script bodies -- so it stays cheap as the
    corpus grows.
    """
    root = _resolve(corpus_dir)
    seen: list[str] = []
    for record in reversed(_read_index(root)):
        if record.get("genre") != genre:
            continue
        angle = record.get("angle", "")
        if angle and angle not in seen:
            seen.append(angle)
        if len(seen) >= limit:
            break
    return seen
```

Add `var/` to `.gitignore`:

```bash
printf '\n# Originality corpus: machine-local generated state.\nvar/\n' >> .gitignore
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add core/originality.py tests/unit/test_originality.py .gitignore
git commit -m "feat(originality): persist generated scripts as a comparison corpus"
```

---

### Task 6: TF-IDF similarity

**Files:**
- Modify: `core/originality.py`
- Test: `tests/unit/test_originality.py`

**Interfaces:**
- Consumes: nothing from Task 5 (pure functions over text)
- Produces:
  - `tokenize(text: str) -> list[str]`
  - `ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]`
  - `document_frequencies(token_lists: Sequence[Sequence[str]], max_n: int = 3) -> dict[tuple[str, ...], int]`
  - `build_idf(token_lists, max_n: int = 3) -> tuple[dict[tuple[str, ...], float], float]` — `(idf_map, default_idf_for_unseen_terms)`
  - `tfidf_vector(tokens, idf, default_idf, max_n: int = 3) -> dict[tuple[str, ...], float]` — L2-normalised
  - `cosine(a: dict, b: dict) -> float`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_originality.py` (before `if __name__`):

```python
from core.originality import (
    build_idf,
    cosine,
    document_frequencies,
    ngrams,
    tfidf_vector,
    tokenize,
)

# Two scripts that share only stock meditation language but say different
# things. The check MUST NOT flag this pair -- it is the whole point.
GENERIC_A = (
    "Settle in and let your shoulders drop. Notice your breath without "
    "changing it. When you are ready, let your eyes close. Feel the weight "
    "of your hands resting in your lap. There is nowhere else to be."
)
GENERIC_B = (
    "Settle in and let your shoulders drop. Notice your breath without "
    "changing it. Picture a narrow copper staircase descending into warm "
    "lamplight. Each step takes you further from the noise of the day."
)
# A near-duplicate of GENERIC_B: same storyline, lightly reworded.
NEAR_DUPLICATE_B = (
    "Settle in and let your shoulders relax. Notice your breathing without "
    "changing it. Imagine a narrow copper staircase descending into warm "
    "lamplight. Every step takes you further from the noise of the day."
)


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Breathe In, Slowly."), ["breathe", "in", "slowly"])

    def test_strips_bracket_markers(self):
        self.assertEqual(
            tokenize("Rest. [pause:5s] [breath] Now rise."),
            ["rest", "now", "rise"],
        )

    def test_empty_text_is_empty(self):
        self.assertEqual(tokenize("   "), [])


class NgramTest(unittest.TestCase):
    def test_produces_overlapping_tuples(self):
        self.assertEqual(
            ngrams(["a", "b", "c"], 2), [("a", "b"), ("b", "c")]
        )

    def test_too_few_tokens_yields_nothing(self):
        self.assertEqual(ngrams(["a"], 2), [])


class IdfTest(unittest.TestCase):
    def test_a_term_in_every_document_weighs_less_than_a_rare_one(self):
        docs = [tokenize(t) for t in ["breath calm", "breath storm", "breath river"]]
        idf, _default = build_idf(docs, max_n=1)
        self.assertLess(idf[("breath",)], idf[("calm",)])

    def test_unseen_terms_get_the_maximum_weight(self):
        docs = [tokenize("breath calm"), tokenize("breath storm")]
        idf, default = build_idf(docs, max_n=1)
        self.assertGreaterEqual(default, max(idf.values()))

    def test_document_frequency_counts_documents_not_occurrences(self):
        docs = [tokenize("breath breath breath"), tokenize("river")]
        df = document_frequencies(docs, max_n=1)
        self.assertEqual(df[("breath",)], 1)


class CosineTest(unittest.TestCase):
    def _vectors(self, texts, target_a, target_b):
        docs = [tokenize(t) for t in texts]
        idf, default = build_idf(docs)
        return (
            tfidf_vector(tokenize(target_a), idf, default),
            tfidf_vector(tokenize(target_b), idf, default),
        )

    def test_identical_text_scores_one(self):
        a, b = self._vectors([GENERIC_A, GENERIC_B], GENERIC_A, GENERIC_A)
        self.assertAlmostEqual(cosine(a, b), 1.0, places=6)

    def test_disjoint_text_scores_zero(self):
        a, b = self._vectors(["alpha beta", "gamma delta"], "alpha beta", "gamma delta")
        self.assertAlmostEqual(cosine(a, b), 0.0, places=6)

    def test_empty_vector_scores_zero_without_dividing_by_zero(self):
        self.assertEqual(cosine({}, {("a",): 1.0}), 0.0)

    def test_near_duplicates_score_higher_than_generic_overlap(self):
        """The central requirement, as a single ordering assertion.

        Two scripts sharing only stock meditation phrasing must be measurably
        less similar than a pair that shares an actual storyline.
        """
        corpus = [GENERIC_A, GENERIC_B, NEAR_DUPLICATE_B]
        generic_pair = self._vectors(corpus, GENERIC_A, GENERIC_B)
        duplicate_pair = self._vectors(corpus, GENERIC_B, NEAR_DUPLICATE_B)
        self.assertLess(cosine(*generic_pair), cosine(*duplicate_pair))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_idf'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/originality.py`:

```python
import math
from collections import Counter
from collections.abc import Sequence

# Bracket markers ([pause:5s], [breath]) are contract syntax, not prose, and
# appear in every script. Removing them keeps them out of the statistics.
_MARKER = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z]+")

MAX_NGRAM = 3


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with bracket markers removed."""
    return _WORD.findall(_MARKER.sub(" ", text.lower()))


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Overlapping n-grams. Empty when there are fewer than n tokens."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _all_terms(tokens: Sequence[str], max_n: int) -> list[tuple[str, ...]]:
    terms: list[tuple[str, ...]] = []
    for n in range(1, max_n + 1):
        terms.extend(ngrams(tokens, n))
    return terms


def document_frequencies(
    token_lists: Sequence[Sequence[str]], max_n: int = MAX_NGRAM
) -> dict[tuple[str, ...], int]:
    """How many documents each term appears in (not how many times)."""
    df: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        df.update(set(_all_terms(tokens, max_n)))
    return dict(df)


def build_idf(
    token_lists: Sequence[Sequence[str]], max_n: int = MAX_NGRAM
) -> tuple[dict[tuple[str, ...], float], float]:
    """Smoothed inverse document frequency over the whole corpus.

    Returns (idf_map, default_idf). The default applies to terms absent from
    the corpus, which are by definition maximally distinctive, so it is the
    largest weight the formula can produce.
    """
    n_docs = len(token_lists)
    df = document_frequencies(token_lists, max_n)
    idf = {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }
    default_idf = math.log(n_docs + 1) + 1.0
    return idf, default_idf


def tfidf_vector(
    tokens: Sequence[str],
    idf: dict[tuple[str, ...], float],
    default_idf: float,
    max_n: int = MAX_NGRAM,
) -> dict[tuple[str, ...], float]:
    """L2-normalised TF-IDF vector. Pre-normalising makes cosine a dot product."""
    counts = Counter(_all_terms(tokens, max_n))
    if not counts:
        return {}
    vector = {
        term: count * idf.get(term, default_idf) for term, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def cosine(a: dict[tuple[str, ...], float], b: dict[tuple[str, ...], float]) -> float:
    """Cosine similarity of two L2-normalised vectors, in [0, 1]."""
    if not a or not b:
        return 0.0
    # Iterate the smaller vector; the result is symmetric.
    if len(b) < len(a):
        a, b = b, a
    return sum(value * b.get(term, 0.0) for term, value in a.items())
```

Move `import math`, `from collections import Counter` and
`from collections.abc import Sequence` into the module's top import block.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add core/originality.py tests/unit/test_originality.py
git commit -m "feat(originality): hand-rolled TF-IDF cosine similarity"
```

---

### Task 7: Rare-passage overlap

**Files:**
- Modify: `core/originality.py`
- Test: `tests/unit/test_originality.py`

**Interfaces:**
- Consumes: `ngrams`, `document_frequencies` (Task 6)
- Produces: `longest_rare_run(tokens_a, tokens_b, df, *, n: int = 5, max_df: int = 2) -> tuple[int, str]`
  — `(shared_token_span_length, shared_text)`; `(0, "")` when nothing matches.

Cosine measures whole-document similarity and can miss one lifted paragraph
inside an otherwise different script. This catches that case.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_originality.py` (before `if __name__`):

```python
from core.originality import longest_rare_run

LIFTED = (
    "A different opening entirely, about morning light on a kitchen floor. "
    "Picture a narrow copper staircase descending into warm lamplight. "
    "Then something else again, about the sound of a kettle."
)


class LongestRareRunTest(unittest.TestCase):
    def test_no_shared_text_returns_zero(self):
        a, b = tokenize("alpha beta gamma delta epsilon zeta"), tokenize(
            "one two three four five six"
        )
        df = document_frequencies([a, b], max_n=5)
        span, text = longest_rare_run(a, b, df)
        self.assertEqual((span, text), (0, ""))

    def test_a_lifted_passage_is_found_inside_different_surroundings(self):
        a, b = tokenize(LIFTED), tokenize(GENERIC_B)
        df = document_frequencies([a, b], max_n=5)
        span, text = longest_rare_run(a, b, df)
        self.assertGreaterEqual(span, 9)
        self.assertIn("copper staircase descending", text)

    def test_common_phrases_are_excluded_by_the_df_filter(self):
        """A phrase appearing in many documents is not a lifted passage."""
        shared = tokenize("notice your breath without changing it at all today")
        corpus = [shared for _ in range(6)]
        df = document_frequencies(corpus, max_n=5)
        span, _text = longest_rare_run(shared, shared, df, max_df=2)
        self.assertEqual(span, 0)

    def test_identical_documents_return_their_full_length(self):
        tokens = tokenize(GENERIC_B)
        df = document_frequencies([tokens], max_n=5)
        span, _text = longest_rare_run(tokens, tokens, df)
        self.assertEqual(span, len(tokens))

    def test_too_short_inputs_do_not_raise(self):
        a = tokenize("only three words")
        df = document_frequencies([a], max_n=5)
        self.assertEqual(longest_rare_run(a, a, df), (0, ""))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: FAIL with `ImportError: cannot import name 'longest_rare_run'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/originality.py`:

```python
RUN_NGRAM = 5
RUN_MAX_DF = 2


def longest_rare_run(
    tokens_a: Sequence[str],
    tokens_b: Sequence[str],
    df: dict[tuple[str, ...], int],
    *,
    n: int = RUN_NGRAM,
    max_df: int = RUN_MAX_DF,
) -> tuple[int, str]:
    """Longest run of consecutive rare n-grams from A that also occur in B.

    Returns (shared_token_span, shared_text). A run of k consecutive
    overlapping n-grams spans k + n - 1 tokens.

    Only n-grams with document frequency <= max_df count, so stock meditation
    phrasing -- which appears across the whole corpus -- cannot trigger this.

    Approximation worth knowing: consecutive matching n-grams in A are not
    proven contiguous in B. In practice a run of overlapping rare n-grams that
    all appear in B is a lifted passage; the alternative (a full longest-common-
    substring over every corpus pair) is O(n*m) per pair and not worth the cost
    for the same answer.
    """
    rare_b = {
        gram for gram in ngrams(tokens_b, n) if df.get(gram, 0) <= max_df
    }
    if not rare_b:
        return 0, ""

    best_len = 0
    best_start = 0
    run = 0
    for index, gram in enumerate(ngrams(tokens_a, n)):
        if gram in rare_b and df.get(gram, 0) <= max_df:
            run += 1
            if run > best_len:
                best_len = run
                best_start = index - run + 1
        else:
            run = 0

    if best_len == 0:
        return 0, ""

    span = best_len + n - 1
    return span, " ".join(tokens_a[best_start : best_start + span])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py -v`
Expected: PASS (24 tests)

- [ ] **Step 5: Commit**

```bash
git add core/originality.py tests/unit/test_originality.py
git commit -m "feat(originality): detect lifted passages via rare n-gram runs"
```

---

### Task 8: Assess a script and turn the result into Violations

**Files:**
- Modify: `core/originality.py`
- Modify: `core/script_gen/linter.py`
- Test: `tests/unit/test_originality.py`, `tests/unit/test_script_linter.py`

**Interfaces:**
- Consumes: `CorpusEntry` (Task 5), `build_idf`/`tfidf_vector`/`cosine` (Task 6), `longest_rare_run` (Task 7)
- Produces:
  - `OriginalityReport` — frozen dataclass: `max_cosine: float`, `cosine_available: bool`, `nearest_id: str`, `shared_span: int`, `shared_text: str`, `corpus_size: int`
  - `assess(script: str, *, compare_against: Sequence[CorpusEntry], idf_texts: Sequence[str]) -> OriginalityReport`
  - `avoid_terms(entries: Sequence[CorpusEntry], idf_texts: Sequence[str], *, top_n: int = 12) -> list[str]`
  - `linter.check_originality(report: OriginalityReport, *, fatal_cosine: float = FATAL_COSINE, advisory_cosine: float = ADVISORY_COSINE) -> list[Violation]`
  - `linter.FATAL_COSINE = 0.72`, `linter.ADVISORY_COSINE = 0.55`

**Cold start:** below `SMALL_CORPUS_BELOW` (10) documents, IDF is meaningless
and everything looks similar, so cosine is not computed
(`cosine_available=False`). The rare-run check still works — it needs no
corpus statistics — but uses a higher token threshold, because with a tiny
corpus the df filter cannot distinguish a genuinely rare phrase from a stock
one, and stock meditation phrases can legitimately run 12+ tokens.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_originality.py` (before `if __name__`):

```python
from core.originality import CorpusEntry, assess, avoid_terms


def _entry(text: str, script_id: str = "x", angle: str = "") -> CorpusEntry:
    return CorpusEntry(
        script_id=script_id, genre="sleep", angle=angle, created="", text=text
    )


class AssessTest(unittest.TestCase):
    def _big_corpus(self, extra: list[str]) -> list[str]:
        """12 filler docs so the corpus clears the cold-start threshold."""
        filler = [
            f"Settle in and notice your breath. Today we rest with {word}."
            for word in "alpha bravo charlie delta echo foxtrot golf hotel "
                        "india juliet kilo lima".split()
        ]
        return filler + extra

    def test_empty_corpus_reports_nothing(self):
        report = assess("anything at all", compare_against=[], idf_texts=[])
        self.assertEqual(report.max_cosine, 0.0)
        self.assertEqual(report.shared_span, 0)
        self.assertFalse(report.cosine_available)

    def test_small_corpus_disables_cosine(self):
        report = assess(
            GENERIC_B,
            compare_against=[_entry(GENERIC_B)],
            idf_texts=[GENERIC_A, GENERIC_B],
        )
        self.assertFalse(report.cosine_available)
        self.assertEqual(report.max_cosine, 0.0)

    def test_large_corpus_enables_cosine(self):
        report = assess(
            GENERIC_B,
            compare_against=[_entry(GENERIC_B, script_id="dup")],
            idf_texts=self._big_corpus([GENERIC_A, GENERIC_B]),
        )
        self.assertTrue(report.cosine_available)
        self.assertGreater(report.max_cosine, 0.80)
        self.assertEqual(report.nearest_id, "dup")

    def test_generic_overlap_stays_below_the_advisory_band(self):
        """The requirement: shared stock phrasing must not read as a copy."""
        report = assess(
            GENERIC_A,
            compare_against=[_entry(GENERIC_B)],
            idf_texts=self._big_corpus([GENERIC_A, GENERIC_B]),
        )
        self.assertLess(report.max_cosine, 0.65)

    def test_lifted_passage_is_reported_even_when_cosine_is_low(self):
        report = assess(
            LIFTED,
            compare_against=[_entry(GENERIC_B)],
            idf_texts=self._big_corpus([GENERIC_B, LIFTED]),
        )
        self.assertGreaterEqual(report.shared_span, 9)
        self.assertIn("copper staircase", report.shared_text)


class AvoidTermsTest(unittest.TestCase):
    def test_returns_distinctive_multiword_terms_not_stock_phrases(self):
        corpus = [
            f"Settle in and notice your breath. Rest with {word}."
            for word in "alpha bravo charlie delta echo foxtrot".split()
        ]
        terms = avoid_terms(
            [_entry(GENERIC_B)], corpus + [GENERIC_B], top_n=6
        )
        joined = " | ".join(terms)
        self.assertIn("copper staircase", joined)
        self.assertNotIn("notice your breath", joined)

    def test_empty_input_returns_empty(self):
        self.assertEqual(avoid_terms([], []), [])
```

Append to `tests/unit/test_script_linter.py`:

```python
class CheckOriginalityTest(unittest.TestCase):
    def _report(self, **overrides):
        from core.originality import OriginalityReport

        base = {
            "max_cosine": 0.0,
            "cosine_available": True,
            "nearest_id": "prev-001",
            "shared_span": 0,
            "shared_text": "",
            "corpus_size": 50,
        }
        base.update(overrides)
        return OriginalityReport(**base)

    def test_an_original_script_produces_no_violations(self):
        from core.script_gen.linter import check_originality

        self.assertEqual(check_originality(self._report(max_cosine=0.2)), [])

    def test_high_similarity_is_fatal(self):
        from core.script_gen.linter import FATAL, check_originality

        violations = check_originality(self._report(max_cosine=0.92))
        self.assertEqual([v.code for v in violations], ["SCRIPT_TOO_SIMILAR"])
        self.assertEqual(violations[0].severity, FATAL)

    def test_middling_similarity_is_advisory(self):
        from core.script_gen.linter import ADVISORY, check_originality

        violations = check_originality(self._report(max_cosine=0.70))
        self.assertEqual([v.code for v in violations], ["SCRIPT_ECHOES_RECENT"])
        self.assertEqual(violations[0].severity, ADVISORY)

    def test_a_lifted_passage_is_fatal_and_quotes_the_text(self):
        from core.script_gen.linter import FATAL, check_originality

        violations = check_originality(
            self._report(shared_span=14, shared_text="a narrow copper staircase")
        )
        self.assertEqual([v.code for v in violations], ["PASSAGE_LIFTED"])
        self.assertEqual(violations[0].severity, FATAL)
        self.assertIn("narrow copper staircase", violations[0].message)

    def test_a_short_shared_run_is_ignored(self):
        from core.script_gen.linter import check_originality

        self.assertEqual(
            check_originality(self._report(shared_span=8, shared_text="and let it go")),
            [],
        )

    def test_a_small_corpus_uses_a_higher_run_threshold(self):
        """With few documents the df filter cannot tell rare from stock."""
        from core.script_gen.linter import check_originality

        small = self._report(
            corpus_size=3, cosine_available=False, shared_span=14,
            shared_text="notice your breath and let your shoulders drop now",
        )
        self.assertEqual(check_originality(small), [])

        large = self._report(shared_span=14, shared_text="a narrow copper staircase")
        self.assertEqual([v.code for v in check_originality(large)], ["PASSAGE_LIFTED"])

    def test_cosine_is_ignored_when_unavailable(self):
        from core.script_gen.linter import check_originality

        report = self._report(max_cosine=0.99, cosine_available=False, corpus_size=2)
        self.assertEqual(check_originality(report), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py tests/unit/test_script_linter.py -v`
Expected: FAIL with `ImportError: cannot import name 'assess'` and `cannot import name 'check_originality'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/originality.py`:

```python
# Below this many documents, IDF carries no signal: with three scripts every
# phrase looks rare, so cosine would flag unrelated meditations as copies.
SMALL_CORPUS_BELOW = 10


@dataclass(frozen=True)
class OriginalityReport:
    """How much a candidate script resembles what has already been made."""

    max_cosine: float
    cosine_available: bool
    nearest_id: str
    shared_span: int
    shared_text: str
    corpus_size: int


def assess(
    script: str,
    *,
    compare_against: Sequence[CorpusEntry],
    idf_texts: Sequence[str],
) -> OriginalityReport:
    """Measure a script against prior work.

    Args:
        script: The candidate script.
        compare_against: Prior scripts to compare with -- SAME GENRE ONLY.
            That is where collisions actually happen.
        idf_texts: Every script in the corpus, ALL GENRES. This is what
            teaches the weighting which phrases are generic meditation
            vocabulary, so it must not be narrowed to one genre.
    """
    if not compare_against:
        return OriginalityReport(
            max_cosine=0.0,
            cosine_available=False,
            nearest_id="",
            shared_span=0,
            shared_text="",
            corpus_size=len(idf_texts),
        )

    candidate_tokens = tokenize(script)
    other_tokens = {entry.script_id: tokenize(entry.text) for entry in compare_against}

    # df for the rare-run check is computed over every document available,
    # candidate included, so a phrase the candidate shares with many prior
    # scripts is correctly seen as common rather than lifted.
    run_df = document_frequencies(
        [candidate_tokens, *other_tokens.values()], max_n=RUN_NGRAM
    )

    best_span, best_text = 0, ""
    for tokens in other_tokens.values():
        span, text = longest_rare_run(candidate_tokens, tokens, run_df)
        if span > best_span:
            best_span, best_text = span, text

    cosine_available = len(idf_texts) >= SMALL_CORPUS_BELOW
    max_cosine, nearest_id = 0.0, ""

    if cosine_available:
        idf, default_idf = build_idf([tokenize(text) for text in idf_texts])
        candidate_vector = tfidf_vector(candidate_tokens, idf, default_idf)
        for script_id, tokens in other_tokens.items():
            score = cosine(
                candidate_vector, tfidf_vector(tokens, idf, default_idf)
            )
            if score > max_cosine:
                max_cosine, nearest_id = score, script_id

    return OriginalityReport(
        max_cosine=max_cosine,
        cosine_available=cosine_available,
        nearest_id=nearest_id,
        shared_span=best_span,
        shared_text=best_text,
        corpus_size=len(idf_texts),
    )


def avoid_terms(
    entries: Sequence[CorpusEntry],
    idf_texts: Sequence[str],
    *,
    top_n: int = 12,
) -> list[str]:
    """The most distinctive phrases in recent scripts, for the planner to avoid.

    Only multi-word terms are returned: single words ("staircase") over-
    constrain the writer, while phrases ("copper staircase descending") name
    the specific image that should not recur.
    """
    if not entries or not idf_texts:
        return []

    idf, default_idf = build_idf([tokenize(text) for text in idf_texts])
    weights: Counter[tuple[str, ...]] = Counter()
    for entry in entries:
        vector = tfidf_vector(tokenize(entry.text), idf, default_idf)
        for term, value in vector.items():
            if len(term) >= 2:
                weights[term] += value

    # TF-IDF weight alone ties constantly: every term unique to one entry
    # gets the same maximal weight, whether it is a content phrase ("copper
    # staircase") or a run of function words ("it picture a"). Break ties by
    # character length rather than insertion order -- a longer phrase is
    # disproportionately likely to be the specific, informative one, and
    # this needs no hand-maintained stoplist, just the term itself.
    ranked = sorted(
        weights.items(),
        key=lambda item: (item[1], sum(len(word) for word in item[0])),
        reverse=True,
    )
    return [" ".join(term) for term, _ in ranked[:top_n]]
```

Append to `core/script_gen/linter.py`:

```python
# --- Originality --------------------------------------------------------
#
# Calibrated 2026-09-18 against realistic-length (158-word) same-genre scripts
# with a 20-document corpus:
#
#     verbatim regeneration                  1.000
#     lightly edited repeat                  0.936
#     heavily reworded, SAME storyline       0.409
#     genuinely different story, same genre  0.467
#     unrelated content                      0.103
#
# Two things follow. First, cosine separates near-verbatim repeats
# (0.94-1.00) from everything else (<=0.47) with a wide empty gap, so the
# bands below sit in the middle of that gap rather than near either edge.
#
# Second, and more important: cosine CANNOT distinguish "reworded, same
# storyline" (0.409) from "genuinely different" (0.467) -- the ordering
# actually inverts, because rewording destroys n-gram overlap while two
# different meditations still share stock openings and closings. That is a
# lexical-vs-semantic limit, not a tuning problem, and no threshold fixes it.
#
# So this check catches near-verbatim regeneration, and the rare-run check
# below catches lifted passages. The defence against a repeated STORYLINE is
# the proactive layer -- angle rotation plus the avoid-list fed to the planner
# (core/originality.py::avoid_terms) -- which prevents the repeat being
# written at all. Closing the paraphrase gap reactively would need embedding
# similarity; see the spec's section 7 for why that is deferred.

FATAL_COSINE = 0.80
ADVISORY_COSINE = 0.65

# A shared run this long is a lifted passage rather than coincidence.
MIN_RUN_TOKENS = 12
# With a small corpus the document-frequency filter cannot distinguish a
# genuinely rare phrase from a stock one, and stock meditation phrasing can
# legitimately run past 12 tokens. Require most of a sentence before calling
# it a lift.
MIN_RUN_TOKENS_SMALL_CORPUS = 20


def check_originality(
    report,
    *,
    fatal_cosine: float = FATAL_COSINE,
    advisory_cosine: float = ADVISORY_COSINE,
) -> list[Violation]:
    """Turn an originality.OriginalityReport into Violations.

    Pure policy: all measurement lives in core/originality.py, so severity
    bands can be retuned here without touching the math.

    The messages quote the offending text, because the judge cannot remove a
    phrase it has not been shown. That relies on parse_judge_response()
    stripping the <problems> block before anything else (commit c372b18) --
    without it, quoting the overlap back to the judge re-injects it into the
    script and the repair loop poisons itself.
    """
    violations: list[Violation] = []

    run_threshold = (
        MIN_RUN_TOKENS
        if report.corpus_size >= SMALL_CORPUS_BELOW
        else MIN_RUN_TOKENS_SMALL_CORPUS
    )

    if report.shared_span >= run_threshold:
        violations.append(
            Violation(
                code="PASSAGE_LIFTED",
                severity=FATAL,
                message=(
                    f"A {report.shared_span}-word passage is reused almost "
                    f"verbatim from an earlier meditation: "
                    f"{report.shared_text!r}. Rewrite that passage with "
                    "different imagery and wording."
                ),
            )
        )

    if report.cosine_available:
        if report.max_cosine > fatal_cosine:
            violations.append(
                Violation(
                    code="SCRIPT_TOO_SIMILAR",
                    severity=FATAL,
                    message=(
                        f"This script is {report.max_cosine:.0%} similar to an "
                        "earlier meditation in the same genre. Change the "
                        "imagery, the structure and the specific language — a "
                        "reworded version of the same piece is not a new one."
                    ),
                )
            )
        elif report.max_cosine >= advisory_cosine:
            violations.append(
                Violation(
                    code="SCRIPT_ECHOES_RECENT",
                    severity=ADVISORY,
                    message=(
                        f"This script is {report.max_cosine:.0%} similar to an "
                        "earlier meditation in the same genre. Acceptable, but "
                        "its distinctive phrases will be added to the avoid-list "
                        "for future runs."
                    ),
                )
            )

    return violations
```

Also append the per-genre banned-phrase check to `core/script_gen/linter.py`:

```python
def check_banned_phrases(script: str, banned: Sequence[str]) -> list[Violation]:
    """Flag phrasings a genre pack forbids outright.

    FATAL rather than advisory: a pack's banned list is not a style
    preference but an explicit "a good writer in this genre never says this"
    ("in a better place" for grief, "push through the pain" for a workout).
    Each one is a targeted single-phrase edit, so it is cheap to repair and
    not the kind of violation that makes a weaker model unusable.

    Matched case-insensitively against tag-stripped, apostrophe-normalised
    prose, so a curly quote or a [pause:5s] marker mid-phrase cannot defeat it.
    """
    if not banned:
        return []

    prose = _normalize_apostrophes(_strip_tags(script)).lower()
    return [
        Violation(
            code="BANNED_PHRASE",
            severity=FATAL,
            message=(
                f"The phrase {phrase!r} is banned for this genre. Remove it "
                "and say what you mean without it."
            ),
        )
        for phrase in banned
        if _normalize_apostrophes(phrase).lower() in prose
    ]
```

Add `from collections.abc import Sequence` to `linter.py`'s imports.

Add these tests to `tests/unit/test_script_linter.py`:

```python
class CheckBannedPhrasesTest(unittest.TestCase):
    def test_no_banned_list_means_no_violations(self):
        from core.script_gen.linter import check_banned_phrases

        self.assertEqual(check_banned_phrases("anything at all", []), [])

    def test_a_banned_phrase_is_fatal(self):
        from core.script_gen.linter import FATAL, check_banned_phrases

        violations = check_banned_phrases(
            "They are in a better place now.", ["in a better place"]
        )
        self.assertEqual([v.code for v in violations], ["BANNED_PHRASE"])
        self.assertEqual(violations[0].severity, FATAL)

    def test_matching_ignores_case(self):
        from core.script_gen.linter import check_banned_phrases

        self.assertEqual(
            len(check_banned_phrases("Time Heals, they say.", ["time heals"])), 1
        )

    def test_a_curly_apostrophe_cannot_defeat_the_check(self):
        from core.script_gen.linter import check_banned_phrases

        script = "You\u2019ll move on soon."
        self.assertEqual(len(check_banned_phrases(script, ["you'll move on"])), 1)

    def test_an_unused_phrase_is_not_flagged(self):
        from core.script_gen.linter import check_banned_phrases

        self.assertEqual(check_banned_phrases("Rest here.", ["move on"]), [])
```

Add `from core.originality import SMALL_CORPUS_BELOW` to the imports at the
top of `linter.py`.

Export the new names from `core/script_gen/__init__.py`: add
`"ADVISORY_COSINE"`, `"FATAL_COSINE"`, `"check_banned_phrases"` and
`"check_originality"` to both the `from core.script_gen.linter import (...)`
block and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_originality.py tests/unit/test_script_linter.py -v`
Expected: PASS — including every pre-existing linter test.

- [ ] **Step 5: Commit**

```bash
git add core/originality.py core/script_gen/linter.py \
        core/script_gen/__init__.py tests/unit/test_originality.py \
        tests/unit/test_script_linter.py
git commit -m "feat(originality): assess scripts and report similarity violations"
```

---

### Task 9: Wire originality into the generation loop

**Files:**
- Modify: `core/auto_generate.py`
- Test: `tests/unit/test_auto_generate.py`

**Interfaces:**
- Consumes: `add_to_corpus`, `load_corpus`, `assess` (Tasks 5–8), `check_originality` (Task 8)
- Produces:
  - `AutoConfig` gains `genre: str = ""`, `angle: str = ""`, `originality: bool = True`, `corpus_dir: Path | None = None`
  - `AutoResult` gains `originality: float` (the max cosine measured, for calibration)
  - Run metadata gains `"genre"`, `"angle"`, `"originality_cosine"`, `"originality_shared_span"`

**Corpus partitioning:** the prompt-driven path passes `genre=""`, which is
its own partition. Prompt-driven runs are therefore compared only with other
prompt-driven runs, which is correct — they have no genre in common.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_auto_generate.py`:

```python
class OriginalityIntegrationTest(unittest.TestCase):
    """The core requirement: the same script twice must be caught."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, **overrides):
        values = {
            "corpus_dir": self.dir / "corpus",
            "background_scan": lambda: [("Bed — 10:00", "/bg/a.mp3")],
            "genre": "sleep",
        }
        values.update(overrides)
        return AutoConfig(**values)

    def _run_once(self, script: str, config):
        pipeline = StubPipeline(self.dir)
        return run(
            "a prompt",
            config=config,
            pipeline=pipeline,
            generator_engine=FakeScriptEngine([script]),
            judge_engine=FakeScriptEngine([judged(script)]),
        )

    def test_a_successful_run_is_added_to_the_corpus(self):
        from core.originality import load_corpus

        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)
        entries = load_corpus(genre="sleep", corpus_dir=config.corpus_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].text, CLEAN_SCRIPT)

    def test_regenerating_an_identical_script_is_fatal(self):
        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)

        with self.assertRaises(ScriptGenerationError) as ctx:
            self._run_once(CLEAN_SCRIPT, config)

        self.assertIn("PASSAGE_LIFTED", str(ctx.exception))

    def test_originality_can_be_switched_off(self):
        config = self._config(originality=False)
        self._run_once(CLEAN_SCRIPT, config)
        result = self._run_once(CLEAN_SCRIPT, config)
        self.assertTrue(result.audio_path)

    def test_metadata_records_the_similarity_score(self):
        config = self._config()
        result = self._run_once(CLEAN_SCRIPT, config)
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertIn("originality_cosine", meta)
        self.assertIn("originality_shared_span", meta)
        self.assertEqual(meta["genre"], "sleep")

    def test_a_different_script_in_the_same_genre_passes(self):
        config = self._config()
        self._run_once(CLEAN_SCRIPT, config)
        other = (
            "Let the day set itself down for a moment.\n\n"
            "[pause:5s]\n\n"
            "Copper light moves slowly along the far wall.\n\n"
            "[pause:5s]\n\n"
            "Nothing here needs deciding tonight."
        )
        result = self._run_once(other, config)
        self.assertTrue(result.audio_path)
```

Ensure `tempfile` and `Path` are imported at the top of the test file — they
already are.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_auto_generate.py -v`
Expected: FAIL with `TypeError: AutoConfig.__init__() got an unexpected keyword argument 'corpus_dir'`

- [ ] **Step 3: Write minimal implementation**

In `core/auto_generate.py`:

1. Add to the imports:

```python
from core.originality import add_to_corpus, assess, load_corpus
from core.script_gen.linter import check_originality
```

2. Add these fields to `AutoConfig` (after `content_type`):

```python
    # Genre partitions the originality corpus. The prompt-driven path leaves
    # this empty, which is its own partition -- prompt runs share no genre
    # with each other and should only be compared among themselves.
    genre: str = ""
    angle: str = ""
    originality: bool = True
    # None uses core.originality.CORPUS_DIR. Injected in tests.
    corpus_dir: Path | None = None
```

3. Add `originality: float = 0.0` to `AutoResult`, and
   `originality_cosine: float = 0.0` plus `originality_shared_span: int = 0`
   to `ScriptOutcome`.

4. In `generate_script`, replace the `violations = check(...)` line with:

```python
        violations = check(
            script,
            estimated_sec=estimated_sec,
            target_min_sec=config.target_min_sec,
            target_max_sec=config.target_max_sec,
        )

        report = None
        if config.originality:
            # IDF over EVERY genre (that is what learns generic meditation
            # vocabulary); comparison within this genre only (that is where
            # collisions happen). See core/originality.py.
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
```

5. In the `if not fatal:` branch, pass the new fields through:

```python
            return ScriptOutcome(
                script=script,
                draft_script=draft_script,
                changelog=changelog,
                violations=violations,
                estimated_sec=estimated_sec,
                repairs_used=repairs_used,
                originality_cosine=report.max_cosine if report else 0.0,
                originality_shared_span=report.shared_span if report else 0,
            )
```

6. In `run()`, immediately after the `pipeline.generate(...)` call returns and
   before the `config.recent_backgrounds.append(...)` line:

```python
    if config.originality:
        # Recorded only after a successful render: a script that never became
        # audio should not constrain future runs.
        add_to_corpus(
            outcome.script,
            genre=config.genre,
            angle=config.angle,
            corpus_dir=config.corpus_dir,
        )
```

7. Add to the metadata dict written in `run()`:

```python
                "genre": config.genre,
                "angle": config.angle,
                "originality_cosine": outcome.originality_cosine,
                "originality_shared_span": outcome.originality_shared_span,
```

8. Add `originality=outcome.originality_cosine,` to the returned `AutoResult`.

9. Wire the originality environment variables into `AutoConfig.from_env()` so
   the spec's configuration surface actually exists. Add a bool parser beside
   the existing `_parse_env_float` / `_parse_env_int`:

```python
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
```

   and extend the `values` dict inside `from_env`:

```python
            "originality": _parse_env_bool("MOODSCAPE_ORIGINALITY", True),
```

   Add these tests to `tests/unit/test_auto_generate.py`:

```python
class OriginalityEnvTest(unittest.TestCase):
    def test_default_is_on(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOODSCAPE_ORIGINALITY", None)
            self.assertTrue(AutoConfig.from_env().originality)

    def test_zero_disables_it(self):
        with patch.dict(os.environ, {"MOODSCAPE_ORIGINALITY": "0"}):
            self.assertFalse(AutoConfig.from_env().originality)

    def test_a_typo_is_reported_not_silently_defaulted(self):
        with patch.dict(os.environ, {"MOODSCAPE_ORIGINALITY": "maybe"}):
            with self.assertRaises(ScriptGenerationError):
                AutoConfig.from_env()
```

   Ensure `os` is imported in the test file.

10. Make the similarity thresholds environment-tunable, since the spec
    documents them as provisional pending calibration. In
    `core/script_gen/linter.py`, replace the two module constants with:

```python
FATAL_COSINE = float(os.environ.get("MOODSCAPE_ORIGINALITY_FATAL", "0.80"))
ADVISORY_COSINE = float(os.environ.get("MOODSCAPE_ORIGINALITY_ADVISORY", "0.65"))
```

    and add `import os` to `linter.py`. These are read at import time
    deliberately: they are a calibration knob set before a batch, not a
    per-run setting, and `check_originality()` already takes explicit
    overrides for tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS — the whole unit suite, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
git add core/auto_generate.py tests/unit/test_auto_generate.py
git commit -m "feat(auto_generate): check originality and record every rendered script"
```

**Phase 2 is complete.** Generating the same meditation twice now fails with a
named, actionable violation that the existing repair loop can act on.

---

# PHASE 3 — Genre core

---

### Task 10: The genre pack registry

**Files:**
- Create: `core/genres.py`
- Test: `tests/unit/test_genres.py`

**Interfaces:**
- Consumes: `MEASURED_VOCAB`, `DECLARED_VOCAB` (Task 2); `CONTENT_PROFILES` (existing)
- Produces:
  - `GenrePackError(Exception)`
  - `Angle` — frozen dataclass: `name: str`, `imagery: tuple[str, ...]`
  - `GenrePack` — frozen dataclass: `slug: str`, `label: str`, `family: str`, `content_type: str`, `music_tags: tuple[str, ...]`, `pause_ratio: float`, `technique: str`, `arc: tuple[str, ...]`, `safety: str`, `banned: tuple[str, ...]`, `angles: tuple[Angle, ...]`
  - `GENRE_PACKS_DIR: Path`
  - `load_pack(slug: str, packs_dir: Path | None = None) -> GenrePack`
  - `load_all_packs(packs_dir: Path | None = None) -> dict[str, GenrePack]`
  - `genre_choices(packs_dir: Path | None = None) -> list[tuple[str, list[tuple[str, str]]]]` — `[(family, [(label, slug), ...]), ...]`
  - `pick_angle(pack: GenrePack, recent: Sequence[str] = (), rng=None) -> Angle`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_genres.py`:

```python
"""Tests for the genre pack registry."""

import random
import tempfile
import unittest
from pathlib import Path

from core.genres import (
    GENRE_PACKS_DIR,
    Angle,
    GenrePack,
    GenrePackError,
    genre_choices,
    load_all_packs,
    load_pack,
    pick_angle,
)

VALID = """
label        = "Grief & Loss"
family       = "Emotional"
content_type = "meditation"
music_tags   = ["warm", "sparse"]
pause_ratio  = 0.34

technique = "RAIN, held loosely."
arc = ["arrival", "body", "memory", "kindness", "return"]
safety = "No stage models. Do not imply closure."
banned = ["time heals", "move on"]

[[angles]]
name    = "the empty chair"
imagery = ["a chair by a window", "afternoon light"]

[[angles]]
name    = "tidal"
imagery = ["a shoreline at dusk", "wet sand"]
"""


class LoadPackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, body: str) -> None:
        (self.dir / f"{name}.toml").write_text(body, encoding="utf-8")

    def test_a_valid_pack_loads(self):
        self._write("grief_and_loss", VALID)
        pack = load_pack("grief_and_loss", packs_dir=self.dir)
        self.assertIsInstance(pack, GenrePack)
        self.assertEqual(pack.slug, "grief_and_loss")
        self.assertEqual(pack.label, "Grief & Loss")
        self.assertEqual(pack.content_type, "meditation")
        self.assertEqual(pack.music_tags, ("warm", "sparse"))
        self.assertEqual(len(pack.angles), 2)
        self.assertIsInstance(pack.angles[0], Angle)

    def test_a_missing_pack_names_the_path(self):
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("nope", packs_dir=self.dir)
        self.assertIn("nope", str(ctx.exception))

    def test_a_missing_required_field_is_rejected(self):
        self._write("bad", VALID.replace('family       = "Emotional"', ""))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("family", str(ctx.exception))

    def test_an_unknown_content_type_is_rejected(self):
        self._write("bad", VALID.replace('"meditation"', '"podcast"'))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("podcast", str(ctx.exception))

    def test_an_unknown_music_tag_is_rejected(self):
        self._write("bad", VALID.replace('["warm", "sparse"]', '["funky"]'))
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("funky", str(ctx.exception))

    def test_an_out_of_range_pause_ratio_is_rejected(self):
        self._write("bad", VALID.replace("0.34", "0.95"))
        with self.assertRaises(GenrePackError):
            load_pack("bad", packs_dir=self.dir)

    def test_fewer_than_two_angles_is_rejected(self):
        body = VALID.split("[[angles]]")[0] + """
[[angles]]
name    = "only one"
imagery = ["a thing"]
"""
        self._write("bad", body)
        with self.assertRaises(GenrePackError) as ctx:
            load_pack("bad", packs_dir=self.dir)
        self.assertIn("angles", str(ctx.exception))

    def test_duplicate_angle_names_are_rejected(self):
        self._write("bad", VALID.replace('"tidal"', '"the empty chair"'))
        with self.assertRaises(GenrePackError):
            load_pack("bad", packs_dir=self.dir)


class ShippedPacksTest(unittest.TestCase):
    """Every pack in the repository must load. A malformed pack cannot ship."""

    def test_all_shipped_packs_load_and_validate(self):
        packs = load_all_packs()
        self.assertEqual(len(packs), 46)
        for slug, pack in packs.items():
            self.assertEqual(pack.slug, slug)
            self.assertTrue(pack.technique.strip(), msg=slug)
            self.assertTrue(pack.safety.strip(), msg=slug)
            self.assertGreaterEqual(len(pack.arc), 3, msg=slug)
            self.assertGreaterEqual(len(pack.angles), 2, msg=slug)

    def test_labels_are_unique(self):
        labels = [pack.label for pack in load_all_packs().values()]
        self.assertEqual(len(labels), len(set(labels)))

    def test_genre_choices_group_by_family(self):
        choices = genre_choices()
        families = [family for family, _entries in choices]
        self.assertEqual(len(families), len(set(families)))
        total = sum(len(entries) for _family, entries in choices)
        self.assertEqual(total, 46)

    def test_packs_dir_exists(self):
        self.assertTrue(GENRE_PACKS_DIR.is_dir())


class PickAngleTest(unittest.TestCase):
    def _pack(self, names: list[str]) -> GenrePack:
        return GenrePack(
            slug="s", label="L", family="F", content_type="meditation",
            music_tags=("warm",), pause_ratio=0.3, technique="t",
            arc=("a", "b", "c"), safety="s", banned=(),
            angles=tuple(Angle(name=n, imagery=("x",)) for n in names),
        )

    def test_excludes_recent_angles(self):
        pack = self._pack(["one", "two", "three"])
        for _ in range(30):
            self.assertEqual(
                pick_angle(pack, recent=["one", "two"]).name, "three"
            )

    def test_falls_back_to_all_angles_when_everything_is_recent(self):
        pack = self._pack(["one", "two"])
        angle = pick_angle(pack, recent=["one", "two"])
        self.assertIn(angle.name, {"one", "two"})

    def test_is_reproducible_with_a_seeded_rng(self):
        pack = self._pack(["one", "two", "three", "four"])
        first = pick_angle(pack, rng=random.Random(7)).name
        second = pick_angle(pack, rng=random.Random(7)).name
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_genres.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.genres'`

- [ ] **Step 3: Write minimal implementation**

Create `core/genres.py`:

```python
"""The genre registry: what a genre is, and what it decides.

A genre pack is a TOML file holding two kinds of thing:

  * Curated creative material -- technique, session arc, imagery angles,
    banned phrasings, safety caveats -- that the planner adapts per run. This
    is what lets a mid-sized local model write a credible grief meditation:
    it adapts known-good material rather than inventing doctrine.
  * Deterministic fields the pipeline reads directly -- content_type,
    music_tags, pause_ratio. Because these are decided here rather than by a
    model, the planner's output can stay plain prose with nothing to parse.

Packs are read at call time, so editing one takes effect on the next run with
no restart -- the same contract rules.py uses for the prompting guides.
"""

import os
import random
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.background_tags import DECLARED_VOCAB, MEASURED_VOCAB
from core.content_profiles import CONTENT_PROFILES

# MOODSCAPE_GENRE_PACKS_DIR lets a harness or an experiment point at an
# alternative pack set without copying the repository.
GENRE_PACKS_DIR = Path(
    os.environ.get(
        "MOODSCAPE_GENRE_PACKS_DIR",
        Path(__file__).resolve().parent.parent / "docs" / "genre_packs",
    )
)

MAX_PAUSE_RATIO = 0.6
MIN_ANGLES = 2
MIN_ARC_STEPS = 3

_KNOWN_TAGS = MEASURED_VOCAB | DECLARED_VOCAB

_REQUIRED = (
    "label", "family", "content_type", "music_tags", "pause_ratio",
    "technique", "arc", "safety",
)


class GenrePackError(Exception):
    """A pack is missing, malformed, or references something unknown."""


@dataclass(frozen=True)
class Angle:
    """One distinct treatment of a genre: an arc and imagery set."""

    name: str
    imagery: tuple[str, ...]


@dataclass(frozen=True)
class GenrePack:
    """Everything a genre decides."""

    slug: str
    label: str
    family: str
    content_type: str
    music_tags: tuple[str, ...]
    pause_ratio: float
    technique: str
    arc: tuple[str, ...]
    safety: str
    banned: tuple[str, ...]
    angles: tuple[Angle, ...]


def _resolve(packs_dir: Path | None) -> Path:
    return packs_dir if packs_dir is not None else GENRE_PACKS_DIR


def load_pack(slug: str, packs_dir: Path | None = None) -> GenrePack:
    """Load and validate one pack.

    Validation is strict and happens at load, not mid-run: a typo'd tag
    should fail before a model is loaded, not after five minutes of
    generation.
    """
    path = _resolve(packs_dir) / f"{slug}.toml"
    if not path.is_file():
        raise GenrePackError(
            f"No genre pack for {slug!r}. Expected {path}."
        )

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise GenrePackError(f"{path} is not valid TOML: {exc}") from exc

    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise GenrePackError(
            f"{path} is missing required field(s): {', '.join(missing)}."
        )

    content_type = data["content_type"]
    if content_type not in CONTENT_PROFILES:
        raise GenrePackError(
            f"{path} names content_type {content_type!r}, which is not one of "
            f"{sorted(CONTENT_PROFILES)}."
        )

    music_tags = tuple(data["music_tags"])
    unknown = sorted(set(music_tags) - _KNOWN_TAGS)
    if unknown:
        raise GenrePackError(
            f"{path} uses unknown music tag(s): {', '.join(unknown)}. "
            f"Known tags: {', '.join(sorted(_KNOWN_TAGS))}."
        )

    pause_ratio = float(data["pause_ratio"])
    if not 0.0 <= pause_ratio <= MAX_PAUSE_RATIO:
        raise GenrePackError(
            f"{path} has pause_ratio {pause_ratio}, outside 0-{MAX_PAUSE_RATIO}."
        )

    arc = tuple(data["arc"])
    if len(arc) < MIN_ARC_STEPS:
        raise GenrePackError(
            f"{path} has {len(arc)} arc step(s); at least {MIN_ARC_STEPS} are "
            "needed to shape a session."
        )

    raw_angles = data.get("angles", [])
    if len(raw_angles) < MIN_ANGLES:
        raise GenrePackError(
            f"{path} has {len(raw_angles)} angles; at least {MIN_ANGLES} are "
            "needed so repeated runs of this genre can differ."
        )

    angles = tuple(
        Angle(name=entry["name"], imagery=tuple(entry.get("imagery", [])))
        for entry in raw_angles
    )
    names = [angle.name for angle in angles]
    if len(names) != len(set(names)):
        raise GenrePackError(f"{path} has duplicate angle names: {names}.")

    return GenrePack(
        slug=slug,
        label=data["label"],
        family=data["family"],
        content_type=content_type,
        music_tags=music_tags,
        pause_ratio=pause_ratio,
        technique=data["technique"].strip(),
        arc=arc,
        safety=data["safety"].strip(),
        banned=tuple(data.get("banned", [])),
        angles=angles,
    )


def load_all_packs(packs_dir: Path | None = None) -> dict[str, GenrePack]:
    """Load every pack in the directory, keyed by slug."""
    root = _resolve(packs_dir)
    if not root.is_dir():
        raise GenrePackError(f"Genre packs directory not found: {root}.")
    return {
        path.stem: load_pack(path.stem, packs_dir=root)
        for path in sorted(root.glob("*.toml"))
    }


def genre_choices(
    packs_dir: Path | None = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Packs grouped by family for the UI dropdown.

    Returns [(family, [(label, slug), ...]), ...], families and labels both
    sorted, so 46 entries stay scannable.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for slug, pack in load_all_packs(packs_dir).items():
        grouped.setdefault(pack.family, []).append((pack.label, slug))
    return [
        (family, sorted(grouped[family])) for family in sorted(grouped)
    ]


def pick_angle(
    pack: GenrePack,
    recent: Sequence[str] = (),
    rng: random.Random | None = None,
) -> Angle:
    """Choose an angle, avoiding recently used ones.

    Mirrors background_picker's exclude-recent rule: if avoiding everything
    would leave nothing, the full set is used. Variety is a preference, not a
    reason to fail a job.
    """
    excluded = set(recent)
    candidates = [a for a in pack.angles if a.name not in excluded] or list(pack.angles)
    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
```

**Note added during execution:** the code above validates presence, range and
membership but not **type**, which lets `arc = "a string"` through silently —
`tuple("...")` iterates characters into a garbage tuple that passes the length
check. The shipped implementation adds `_require_str`, `_require_str_list` and
`_require_float` helpers that raise `GenrePackError` naming the field and what
was found, and routes every field through them. See commit `f390acb`.

- [ ] **Step 4: Run test to verify the non-pack tests pass**

Run: `.venv/bin/python -m pytest tests/unit/test_genres.py -v -k "not Shipped"`
Expected: PASS. The `ShippedPacksTest` class still fails — Task 11 writes the packs.

- [ ] **Step 5: Commit**

```bash
git add core/genres.py tests/unit/test_genres.py
git commit -m "feat(genres): TOML genre pack registry with strict validation"
```

---

### Task 11: Author the 46 genre packs

**Files:**
- Create: `docs/genre_packs/<slug>.toml` × 46
- Create: `docs/genre_packs/README.md`
- Test: `tests/unit/test_genres.py::ShippedPacksTest` (already written in Task 10)

**Interfaces:**
- Consumes: the `GenrePack` field contract from Task 10
- Produces: 46 validated packs on disk

**The deterministic fields are specified exactly below.** Use these values
verbatim — they encode decisions about audio profile, music matching and
silence budget that are not the pack author's to re-make.

| slug | label | family | content_type | music_tags | pause_ratio |
|---|---|---|---|---|---|
| `fall_asleep` | Fall Asleep | Sleep & Rest | sleep_story | dark, sustained | 0.15 |
| `deep_sleep` | Deep Sleep | Sleep & Rest | sleep_story | drone, sustained | 0.14 |
| `racing_mind_at_night` | Racing Mind at Night | Sleep & Rest | meditation | warm, steady | 0.30 |
| `yoga_nidra` | Yoga Nidra (NSDR) | Sleep & Rest | meditation | drone, tonal | 0.38 |
| `power_nap` | Power Nap | Sleep & Rest | sleep_story | warm, sustained | 0.18 |
| `back_to_sleep` | Back to Sleep | Sleep & Rest | sleep_story | dark, drone | 0.16 |
| `stress_relief` | Stress Relief | Stress & Anxiety | meditation | warm, steady | 0.32 |
| `anxiety_relief` | Anxiety Relief | Stress & Anxiety | meditation | warm, sustained | 0.34 |
| `panic_sos` | Panic SOS | Stress & Anxiety | meditation | steady, tonal | 0.22 |
| `overwhelm` | Overwhelm | Stress & Anxiety | meditation | warm, sparse | 0.33 |
| `burnout_recovery` | Burnout Recovery | Stress & Anxiety | meditation | warm, sustained | 0.35 |
| `worry_and_rumination` | Worry & Rumination | Stress & Anxiety | meditation | steady, tonal | 0.32 |
| `deep_work` | Deep Work | Focus & Work | meditation | steady, sustained | 0.26 |
| `study` | Study | Focus & Work | meditation | steady, tonal | 0.26 |
| `pre_meeting_calm` | Pre-Meeting Calm | Focus & Work | meditation | warm, steady | 0.28 |
| `work_break_reset` | Work Break Reset | Focus & Work | meditation | warm, sparse | 0.30 |
| `creative_flow` | Creative Flow | Focus & Work | meditation | warm, evolving | 0.28 |
| `decision_clarity` | Decision Clarity | Focus & Work | meditation | steady, tonal | 0.30 |
| `workout_warm_up` | Workout Warm-Up | Body & Movement | meditation | bright, evolving | 0.18 |
| `workout_cool_down` | Workout Cool-Down | Body & Movement | meditation | warm, sustained | 0.28 |
| `running` | Running | Body & Movement | meditation | steady, evolving | 0.16 |
| `walking` | Walking | Body & Movement | meditation | warm, steady | 0.22 |
| `stretching_and_yoga` | Stretching & Yoga | Body & Movement | meditation | warm, sustained | 0.30 |
| `body_scan` | Body Scan | Body & Movement | meditation | drone, sustained | 0.40 |
| `pain_and_discomfort` | Pain & Discomfort | Body & Movement | meditation | warm, sparse | 0.36 |
| `box_breathing` | Box Breathing | Breathwork | meditation | steady, tonal | 0.46 |
| `calming_breath_4_7_8` | 4-7-8 Calming Breath | Breathwork | meditation | warm, steady | 0.48 |
| `coherent_breathing` | Coherent Breathing | Breathwork | meditation | drone, steady | 0.45 |
| `energising_breath` | Energising Breath | Breathwork | meditation | bright, steady | 0.38 |
| `breath_awareness` | Breath Awareness | Breathwork | meditation | sparse, sustained | 0.42 |
| `self_compassion` | Self-Compassion | Emotional | meditation | warm, sustained | 0.34 |
| `grief_and_loss` | Grief & Loss | Emotional | meditation | warm, sparse | 0.34 |
| `loneliness` | Loneliness | Emotional | meditation | warm, steady | 0.32 |
| `anger_and_frustration` | Anger & Frustration | Emotional | meditation | dark, steady | 0.30 |
| `confidence` | Confidence | Emotional | meditation | bright, steady | 0.28 |
| `letting_go` | Letting Go | Emotional | meditation | sparse, sustained | 0.36 |
| `forgiveness` | Forgiveness | Emotional | meditation | warm, sustained | 0.34 |
| `morning_start` | Morning Start | Morning & Energy | meditation | bright, evolving | 0.26 |
| `intention_setting` | Intention Setting | Morning & Energy | meditation | warm, steady | 0.30 |
| `energy_boost` | Energy Boost | Morning & Energy | meditation | bright, busy | 0.20 |
| `commute` | Commute | Morning & Energy | meditation | steady, evolving | 0.24 |
| `mindfulness_basics` | Mindfulness Basics | Presence | meditation | warm, steady | 0.34 |
| `loving_kindness` | Loving-Kindness | Presence | meditation | warm, sustained | 0.33 |
| `gratitude` | Gratitude | Presence | meditation | warm, evolving | 0.30 |
| `open_awareness` | Open Awareness | Presence | meditation | drone, sparse | 0.40 |
| `evening_wind_down` | Evening Wind-Down | Presence | meditation | dark, sustained | 0.32 |

**Note on `music_tags`:** exactly two *measured* tags per pack. The filter is
conjunctive (a track must carry every tag), and with a 20-track library three
tags would match nothing and silently fall back to the full pool. The spec's
example shows three including a `declared` tag — that illustrates the field
format; add `declared` tags to packs only after hand-tagging the library.

**Rules for the prose fields**, which are the author's craft:

- `technique` — 1–3 sentences naming the actual practice (RAIN, body scan,
  noting, metta phrasing, box breathing counts). Name a real method, not a mood.
- `arc` — 4–6 short phrases, the session's shape start to finish.
- `safety` — 2–4 sentences of what this genre must never do. Be specific to
  the genre; generic safety already lives in `content_safety_rules.md`.
- `banned` — 3–8 exact phrases a good writer in this genre would never use.
- `angles` — **3 per pack** (minimum 2 enforced; 3 gives real rotation), each
  with a distinct `name` and 3–4 concrete `imagery` items. Two angles of one
  genre must not be able to produce the same meditation.

**Two complete exemplars.** Follow these exactly in shape and length.

`docs/genre_packs/grief_and_loss.toml`:

```toml
label        = "Grief & Loss"
family       = "Emotional"
content_type = "meditation"
music_tags   = ["warm", "sparse"]
pause_ratio  = 0.34

technique = """
RAIN, held loosely: Recognise what is present, Allow it without fixing,
Investigate with kindness, Nurture. Never resolve the grief — the practice is
company, not repair.
"""

arc = [
    "arrival & permission",
    "the body's weather",
    "one memory, held lightly",
    "self-kindness",
    "return",
]

safety = """
No stage models of grief and no timelines. Never instruct the listener to let
go of the person, or to feel better. Offer an exit at every turn — it is
always fine to stop. Do not assume the relationship, the cause, or that the
death was recent.
"""

banned = [
    "everything happens for a reason",
    "in a better place",
    "time heals",
    "move on",
    "closure",
    "at least",
]

[[angles]]
name    = "the empty chair"
imagery = ["a chair by a window", "afternoon light on the floor", "a cup gone cold"]

[[angles]]
name    = "tidal"
imagery = ["a shoreline at dusk", "waves that arrive and withdraw", "wet sand holding a shape"]

[[angles]]
name    = "carrying"
imagery = ["a stone warmed in a pocket", "a coat that still smells of someone", "a path walked many times"]
```

`docs/genre_packs/box_breathing.toml`:

```toml
label        = "Box Breathing"
family       = "Breathwork"
content_type = "meditation"
music_tags   = ["steady", "tonal"]
pause_ratio  = 0.46

technique = """
Equal four-count box breathing: inhale four, hold four, exhale four, hold
four. Count aloud for the first rounds, then let the listener keep the shape
alone. Silence carries the counts, so most of this runtime is [pause] markers.
"""

arc = [
    "posture & a first ordinary breath",
    "learning the four counts",
    "guided rounds",
    "unguided rounds",
    "release the count",
]

safety = """
Holds never exceed four seconds, and every hold is optional — say so. Anyone
lightheaded should return to ordinary breathing immediately, and the script
must say that before the first hold. Do not present this as a treatment for
panic disorder or asthma.
"""

banned = [
    "force the breath",
    "push through",
    "as long as you can",
    "empty your lungs completely",
]

[[angles]]
name    = "the drawn square"
imagery = ["a fingertip tracing a square", "four equal sides", "returning to the first corner"]

[[angles]]
name    = "the lantern room"
imagery = ["a room with four windows", "light arriving at each in turn", "an even glow"]

[[angles]]
name    = "metronome"
imagery = ["a slow pendulum", "an unhurried clock in another room", "a tide that keeps its own time"]
```

Also create `docs/genre_packs/README.md` documenting: the field contract, the
"two measured tags" rule, the prose rules above, and the **tagger thresholds**
from Task 2 with the measured distribution they came from, so they can be
re-derived with `python scripts/tag_backgrounds.py --report`.

- [ ] **Step 1: Write the two exemplar packs**

Create `grief_and_loss.toml` and `box_breathing.toml` exactly as above.

- [ ] **Step 2: Verify the loader accepts them**

Run: `.venv/bin/python -c "from core.genres import load_pack; print(load_pack('grief_and_loss').label, load_pack('box_breathing').label)"`
Expected: `Grief & Loss Box Breathing`

- [ ] **Step 3: Write the remaining 44 packs**

One file per remaining row of the table, following the exemplars. Work
family by family and commit per family, so a reviewer can reject one family
without rejecting all 46.

- [ ] **Step 4: Run the full validation**

Run: `.venv/bin/python -m pytest tests/unit/test_genres.py -v`
Expected: PASS (all 15 tests, including `ShippedPacksTest`)

- [ ] **Step 5: Commit**

```bash
git add docs/genre_packs/
git commit -m "feat(genres): author the 46 genre packs"
```

---

### Task 12: Model unloading and preflight

**Files:**
- Modify: `core/script_gen/engine.py`
- Modify: `core/script_gen/adapters/openai_compat.py`
- Test: `tests/unit/test_script_engine.py`, `tests/unit/test_openai_compat_adapter.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ScriptEngine.unload() -> None` — default no-op
  - `ScriptEngine.preflight() -> None` — default no-op; raises `RuntimeError` naming the fix when the model is unavailable
  - `FakeScriptEngine` records `unload_calls: int` and `preflight_calls: int`

**Why this exists:** `keep_alive` is silently ignored on Ollama's
`/v1/chat/completions` and honoured only on the native `/api/*` endpoints.
Without an explicit unload, an 18 GB model stays resident while F5-TTS and
Demucs load, and a 32 GB machine swaps — which is where this project's MPS
deallocation bus errors live.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_script_engine.py`:

```python
class UnloadAndPreflightTest(unittest.TestCase):
    def test_fake_engine_counts_unloads(self):
        engine = FakeScriptEngine(["x"])
        self.assertEqual(engine.unload_calls, 0)
        engine.unload()
        engine.unload()
        self.assertEqual(engine.unload_calls, 2)

    def test_fake_engine_counts_preflights(self):
        engine = FakeScriptEngine(["x"])
        engine.preflight()
        self.assertEqual(engine.preflight_calls, 1)

    def test_anthropic_engine_unload_is_a_harmless_no_op(self):
        """Hosted providers have nothing to unload; the call must not fail."""
        from core.script_gen.adapters.anthropic_api import AnthropicEngine

        engine = AnthropicEngine("claude-opus-5", api_key_env="ANTHROPIC_API_KEY")
        engine.unload()
        engine.preflight()
```

Append to `tests/unit/test_openai_compat_adapter.py`:

```python
class OllamaUnloadTest(unittest.TestCase):
    def _engine(self, handler, provider="ollama"):
        return OpenAICompatEngine(
            provider=provider,
            model="qwen3.8:27b",
            base_url="http://localhost:11434/v1",
            api_key_env=None,
            transport=httpx.MockTransport(handler),
        )

    def test_unload_posts_keep_alive_zero_to_the_native_endpoint(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok"})

        self._engine(handler).unload()
        self.assertEqual(seen["url"], "http://localhost:11434/api/generate")
        self.assertEqual(seen["body"]["keep_alive"], 0)
        self.assertEqual(seen["body"]["model"], "qwen3.8:27b")

    def test_unload_is_a_no_op_for_hosted_providers(self):
        def handler(request):
            raise AssertionError("hosted providers must not be called on unload")

        self._engine(handler, provider="groq").unload()

    def test_a_failing_unload_never_raises(self):
        """A model that will not unload is a memory problem, not a job failure."""

        def handler(request):
            return httpx.Response(500, text="boom")

        self._engine(handler).unload()

    def test_preflight_passes_when_the_model_is_present(self):
        def handler(request):
            return httpx.Response(
                200, json={"models": [{"name": "qwen3.8:27b"}, {"name": "gemma4:31b"}]}
            )

        self._engine(handler).preflight()

    def test_preflight_names_the_pull_command_when_the_model_is_missing(self):
        def handler(request):
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

        with self.assertRaises(RuntimeError) as ctx:
            self._engine(handler).preflight()
        self.assertIn("ollama pull qwen3.8:27b", str(ctx.exception))

    def test_preflight_is_silent_when_ollama_is_unreachable(self):
        """Preflight is an early warning, not a second connectivity check.

        complete() already reports an unreachable Ollama with a good message;
        failing here too would just replace it with a worse one.
        """

        def handler(request):
            raise httpx.ConnectError("refused")

        self._engine(handler).preflight()
```

Ensure `json` and `httpx` are imported in `test_openai_compat_adapter.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_script_engine.py tests/unit/test_openai_compat_adapter.py -v`
Expected: FAIL with `AttributeError: 'FakeScriptEngine' object has no attribute 'unload_calls'`

- [ ] **Step 3: Write minimal implementation**

In `core/script_gen/engine.py`, add to the `ScriptEngine` ABC (after
`complete`, before `name`):

```python
    def unload(self) -> None:
        """Release backend resources held by this model.

        Default: no-op. Hosted providers hold nothing locally, so only the
        local Ollama adapter overrides this.

        Called between pipeline stages. On a 32 GB machine an 18 GB model
        left resident while F5-TTS and Demucs load means swap, and swap under
        Metal is where this project's deallocation bus errors live.
        """

    def preflight(self) -> None:
        """Fail fast if this engine cannot serve its model.

        Default: no-op.

        Raises:
            RuntimeError: Naming the model and the exact command to fix it.
        """
```

In `FakeScriptEngine.__init__` add `self.unload_calls = 0` and
`self.preflight_calls = 0`, and add:

```python
    def unload(self) -> None:
        self.unload_calls += 1

    def preflight(self) -> None:
        self.preflight_calls += 1
```

In `core/script_gen/adapters/openai_compat.py`, add to `OpenAICompatEngine`:

```python
    def _native_base(self) -> str:
        """Ollama's native API root.

        base_url points at the OpenAI-compatible surface
        (http://host:11434/v1), but keep_alive and the model list are only
        available on the native API one level up.
        """
        return self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url

    def unload(self) -> None:
        """Ask Ollama to evict this model immediately.

        No-op for every other provider. Failures are logged and swallowed: a
        model that will not unload is a memory-pressure problem for the next
        stage, not a reason to fail a job that has already produced a script.
        """
        if self._provider != "ollama":
            return
        url = f"{self._native_base()}/api/generate"
        try:
            with httpx.Client(timeout=30.0, transport=self._transport) as client:
                client.post(url, json={"model": self._model, "keep_alive": 0})
        except Exception:
            logger.warning(
                "Could not unload %s from Ollama; the next stage may be "
                "memory-constrained.",
                self._model,
                exc_info=True,
            )

    def preflight(self) -> None:
        """Check the model is pulled before a long run begins.

        No-op for every provider but Ollama. An unreachable server is NOT
        reported here: complete() already produces a good message for that,
        and duplicating it would only make the error worse.

        Raises:
            RuntimeError: If Ollama is reachable and the model is absent.
        """
        if self._provider != "ollama":
            return
        url = f"{self._native_base()}/api/tags"
        try:
            with httpx.Client(timeout=10.0, transport=self._transport) as client:
                response = client.get(url)
            names = {
                entry.get("name", "")
                for entry in response.json().get("models", [])
            }
        except Exception:
            logger.debug("Preflight could not reach %s; skipping.", url, exc_info=True)
            return

        if self._model not in names:
            raise RuntimeError(
                f"Ollama does not have model {self._model!r}. Pull it first:\n"
                f"    ollama pull {self._model}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_script_engine.py tests/unit/test_openai_compat_adapter.py -v`
Expected: PASS, including every pre-existing adapter test.

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/engine.py core/script_gen/adapters/openai_compat.py \
        tests/unit/test_script_engine.py tests/unit/test_openai_compat_adapter.py
git commit -m "feat(script_gen): explicit Ollama unload and model preflight"
```

---

### Task 13: The planner

**Files:**
- Create: `core/script_gen/planner.py`
- Modify: `core/script_gen/rules.py`
- Modify: `core/script_gen/__init__.py`
- Test: `tests/unit/test_planner.py`, `tests/unit/test_script_rules.py`

**Interfaces:**
- Consumes: `GenrePack`, `Angle` (Task 10); `ScriptEngine` (existing)
- Produces:
  - `rules.build_planner_system_prompt(content_type: str, target_min_sec: float, target_max_sec: float, guides_dir: Path | None = None) -> str`
  - `planner.plan(engine, pack, angle, *, system: str, target_min_sec: float, target_max_sec: float, avoid: Sequence[str] = (), steer: str = "", max_tokens: int = 1024) -> str`

**The planner system prompt deliberately omits the formatting guide.** The
planner writes a brief, not a script, so it does not need the 3,000-word
engine contract — only the safety rules and the duration budget. That cuts
this stage's prefill from ~4,700 tokens to ~800.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_planner.py`:

```python
"""Tests for the planner: genre pack + angle -> prose creative brief."""

import unittest

from core.genres import Angle, GenrePack
from core.script_gen.engine import FakeScriptEngine
from core.script_gen.planner import plan

PACK = GenrePack(
    slug="grief_and_loss",
    label="Grief & Loss",
    family="Emotional",
    content_type="meditation",
    music_tags=("warm", "sparse"),
    pause_ratio=0.34,
    technique="RAIN, held loosely. Never resolve the grief.",
    arc=("arrival", "the body's weather", "one memory", "kindness", "return"),
    safety="No stage models. Do not imply closure.",
    banned=("time heals", "move on"),
    angles=(
        Angle(name="the empty chair", imagery=("a chair by a window", "cold tea")),
        Angle(name="tidal", imagery=("a shoreline at dusk", "wet sand")),
    ),
)

BRIEF = "Write a grief meditation built around a chair by a window."


class PlanTest(unittest.TestCase):
    def _plan(self, **kwargs):
        engine = FakeScriptEngine([BRIEF])
        result = plan(
            engine,
            PACK,
            PACK.angles[0],
            system="SYSTEM",
            target_min_sec=360.0,
            target_max_sec=600.0,
            **kwargs,
        )
        return engine, result

    def test_returns_the_models_brief_verbatim(self):
        _engine, brief = self._plan()
        self.assertEqual(brief, BRIEF)

    def test_strips_a_markdown_fence(self):
        engine = FakeScriptEngine([f"```\n{BRIEF}\n```"])
        brief = plan(
            engine, PACK, PACK.angles[0], system="S",
            target_min_sec=360.0, target_max_sec=600.0,
        )
        self.assertEqual(brief, BRIEF)

    def test_the_prompt_carries_the_pack_and_the_chosen_angle(self):
        engine, _brief = self._plan()
        user = engine.calls[0]["user"]
        self.assertIn("Grief & Loss", user)
        self.assertIn("RAIN, held loosely", user)
        self.assertIn("the empty chair", user)
        self.assertIn("a chair by a window", user)
        self.assertIn("the body's weather", user)

    def test_the_prompt_carries_the_banned_phrases(self):
        engine, _brief = self._plan()
        self.assertIn("time heals", engine.calls[0]["user"])

    def test_the_prompt_does_not_leak_the_other_angle(self):
        """One angle per run, or the writer blends them into mush."""
        engine, _brief = self._plan()
        self.assertNotIn("tidal", engine.calls[0]["user"])
        self.assertNotIn("wet sand", engine.calls[0]["user"])

    def test_the_avoid_list_appears_when_given(self):
        engine, _brief = self._plan(avoid=["copper staircase", "a distant train"])
        self.assertIn("copper staircase", engine.calls[0]["user"])

    def test_no_avoid_section_when_the_list_is_empty(self):
        engine, _brief = self._plan()
        self.assertNotIn("Do not reuse", engine.calls[0]["user"])

    def test_steer_text_is_included(self):
        engine, _brief = self._plan(steer="for a night shift worker")
        self.assertIn("night shift worker", engine.calls[0]["user"])

    def test_the_pause_budget_is_expressed_as_a_percentage(self):
        engine, _brief = self._plan()
        user = engine.calls[0]["user"]
        # 34% of the 360-600s window is roughly 122-204 seconds of silence.
        self.assertIn("34%", user)

    def test_the_system_prompt_is_passed_through(self):
        engine, _brief = self._plan()
        self.assertEqual(engine.calls[0]["system"], "SYSTEM")


if __name__ == "__main__":
    unittest.main()
```

Append to `tests/unit/test_script_rules.py`:

```python
class PlannerSystemPromptTest(unittest.TestCase):
    def test_includes_the_safety_rules_and_the_duration_window(self):
        from core.script_gen.rules import build_planner_system_prompt

        prompt = build_planner_system_prompt("meditation", 360.0, 600.0)
        self.assertIn("6", prompt)
        self.assertIn("10", prompt)
        self.assertIn("safety", prompt.lower())

    def test_omits_the_engine_formatting_guide(self):
        """The planner writes a brief, not a script -- the 3,000-word TTS
        formatting contract would be pure wasted prefill on every run."""
        from core.script_gen.rules import (
            build_planner_system_prompt,
            load_guide,
        )

        prompt = build_planner_system_prompt("meditation", 360.0, 600.0)
        guide = load_guide("f5", "meditation")
        self.assertNotIn(guide[:200], prompt)

    def test_says_the_output_is_a_brief_not_a_script(self):
        from core.script_gen.rules import build_planner_system_prompt

        self.assertIn("brief", build_planner_system_prompt("meditation", 360.0, 600.0).lower())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_script_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.script_gen.planner'`

- [ ] **Step 3: Write minimal implementation**

Append to `core/script_gen/rules.py`:

```python
def build_planner_system_prompt(
    content_type: str,
    target_min_sec: float,
    target_max_sec: float,
    guides_dir: Path | None = None,
) -> str:
    """System prompt for pass 0 — turning a genre pack into a creative brief.

    Deliberately omits the engine formatting guide. The planner produces a
    brief for the writer, not a script for the TTS engine, so the ~3,000-word
    marker contract would be wasted prefill on every run. The writer's own
    system prompt still carries it.
    """
    noun = "sleep story" if content_type == "sleep_story" else "guided meditation"
    return "\n\n".join(
        [
            f"You are a creative director for {noun} audio. You are given a "
            "genre's established technique, session arc and imagery, and you "
            "write a short creative brief that a writer will turn into a "
            "finished script. You do not write the script yourself.",
            "The brief must be specific: name the images, the order of the "
            "sections, and the emotional movement. A vague brief produces a "
            "generic meditation. Work within the material you are given "
            "rather than inventing a different practice.",
            _duration_clause(target_min_sec, target_max_sec),
            "# Content safety rules\n\n" + load_safety_rules(guides_dir),
            "Output the brief as plain prose, 150-250 words. No headings, no "
            "bullet lists, no preamble, and no script text.",
        ]
    )
```

Create `core/script_gen/planner.py`:

```python
"""Pass 0: turn a genre pack and one angle into a prose creative brief.

The brief is plain prose with nothing to parse. Everything the pipeline needs
to act on deterministically -- content_type, music_tags, pause_ratio -- is
already decided by the pack, so the planner is free to be purely creative and
there is no structured-output failure mode to handle.
"""

from collections.abc import Sequence

from core.genres import Angle, GenrePack
from core.script_gen.engine import ScriptEngine
from core.script_gen.generator import strip_wrapper


def _bullet(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def plan(
    engine: ScriptEngine,
    pack: GenrePack,
    angle: Angle,
    *,
    system: str,
    target_min_sec: float,
    target_max_sec: float,
    avoid: Sequence[str] = (),
    steer: str = "",
    max_tokens: int = 1024,
) -> str:
    """Produce a creative brief for one run of one genre.

    Only the chosen angle is shown. Handing the model every angle at once
    produces a blend of all of them, which is both worse and less varied than
    committing to one.
    """
    sections = [
        f"Genre: {pack.label} ({pack.family})",
        f"Technique:\n{pack.technique}",
        f"Session arc:\n{_bullet(pack.arc)}",
        f"Angle for this session: {angle.name}",
        f"Imagery to build on:\n{_bullet(angle.imagery)}",
        (
            f"Silence budget: about {pack.pause_ratio:.0%} of the runtime "
            "should be silence held by [pause:Xs] markers. Say in the brief "
            "where the long pauses belong."
        ),
        f"Genre safety notes:\n{pack.safety}",
    ]

    if pack.banned:
        sections.append(
            "Never use these phrasings:\n" + _bullet(pack.banned)
        )

    if avoid:
        sections.append(
            "Do not reuse these images or phrases — recent meditations in "
            "this genre already used them:\n" + _bullet(avoid)
        )

    if steer.strip():
        sections.append(f"Additional request from the listener:\n{steer.strip()}")

    user = (
        "Write the creative brief for one session.\n\n"
        + "\n\n".join(sections)
        + "\n\nOutput only the brief."
    )

    return strip_wrapper(engine.complete(system, user, max_tokens=max_tokens))
```

Export from `core/script_gen/__init__.py`: add
`from core.script_gen.planner import plan` and
`from core.script_gen.rules import build_planner_system_prompt`, plus
`"plan"` and `"build_planner_system_prompt"` in `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_planner.py tests/unit/test_script_rules.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/planner.py core/script_gen/rules.py \
        core/script_gen/__init__.py tests/unit/test_planner.py \
        tests/unit/test_script_rules.py
git commit -m "feat(script_gen): planner turns a genre pack into a creative brief"
```

---

### Task 14: The genre path through the orchestrator

**Files:**
- Modify: `core/auto_generate.py`
- Test: `tests/unit/test_auto_generate.py`

**Interfaces:**
- Consumes: `load_pack`, `pick_angle` (Task 10); `plan`, `build_planner_system_prompt` (Task 13); `unload`, `preflight` (Task 12); `recent_angles`, `avoid_terms` (Tasks 5, 8)
- Produces:
  - `DEFAULT_PLANNER = "ollama:qwen3.8:27b"`, `DEFAULT_GENERATOR = "ollama:qwen3.8:27b"`, `DEFAULT_JUDGE = "ollama:gemma4:31b"`
  - `DURATION_BANDS: dict[str, tuple[float, float]]` — `{"short": (180.0, 360.0), "medium": (360.0, 600.0), "long": (600.0, 900.0)}`
  - `AutoConfig.from_genre(pack, *, band: str = "medium", **overrides) -> AutoConfig`
  - `run(prompt="", *, genre: str | None = None, steer: str = "", planner_engine=None, ...)`
  - `ScriptOutcome` gains `brief: str`; `AutoResult` gains `brief: str`, `genre: str`, `angle: str`

**Content type ownership, stated once:** `run()` never overrides
`config.content_type`. The UI pre-fills it from the pack via a change
handler, and non-UI callers use `AutoConfig.from_genre()`, which copies the
pack's value. Nothing infers it silently at run time.

**Planner and writer share one model by default**, so the planner stage costs
no model swap: `unload()` is called after the *writer*, not between planning
and writing. When the two specs differ, the planner is unloaded first.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_auto_generate.py`:

```python
class GenrePathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *, genre="grief_and_loss", band="medium", steer="", **kw):
        from core.auto_generate import DURATION_BANDS
        from core.genres import load_pack

        pack = load_pack(genre)
        config = AutoConfig.from_genre(
            pack,
            band=band,
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        self.planner = FakeScriptEngine(["A brief about a chair by a window."])
        self.generator = FakeScriptEngine([CLEAN_SCRIPT])
        self.judge = FakeScriptEngine([judged(CLEAN_SCRIPT)])
        self.pipeline = StubPipeline(self.dir)
        return run(
            "",
            genre=genre,
            steer=steer,
            config=config,
            pipeline=self.pipeline,
            planner_engine=self.planner,
            generator_engine=self.generator,
            judge_engine=self.judge,
            **kw,
        ), config

    def test_duration_bands_cover_the_three_ui_options(self):
        from core.auto_generate import DURATION_BANDS

        self.assertEqual(DURATION_BANDS["short"], (180.0, 360.0))
        self.assertEqual(DURATION_BANDS["medium"], (360.0, 600.0))
        self.assertEqual(DURATION_BANDS["long"], (600.0, 900.0))

    def test_from_genre_copies_the_packs_deterministic_fields(self):
        from core.genres import load_pack

        config = AutoConfig.from_genre(load_pack("fall_asleep"), band="short")
        self.assertEqual(config.content_type, "sleep_story")
        self.assertEqual(config.genre, "fall_asleep")
        self.assertEqual(config.target_min_sec, 180.0)
        self.assertEqual(config.target_max_sec, 360.0)

    def test_the_planners_brief_becomes_the_writers_prompt(self):
        _result, _config = self._run()
        self.assertIn(
            "A brief about a chair by a window.", self.generator.calls[0]["user"]
        )

    def test_the_chosen_angle_is_recorded(self):
        from core.genres import load_pack

        result, _config = self._run()
        names = {a.name for a in load_pack("grief_and_loss").angles}
        self.assertIn(result.angle, names)

    def test_steer_text_reaches_the_planner(self):
        _result, _config = self._run(steer="after a long hospital week")
        self.assertIn("hospital week", self.planner.calls[0]["user"])

    def test_the_packs_music_tags_reach_the_background_picker(self):
        seen = {}

        def fake_pick(**kwargs):
            seen.update(kwargs)
            return ("Bed — 10:00", "/bg/a.mp3")

        with patch("core.auto_generate.pick_background", fake_pick):
            self._run()
        self.assertEqual(tuple(seen["prefer_tags"]), ("warm", "sparse"))

    def test_every_engine_is_preflighted_and_unloaded(self):
        self._run()
        self.assertEqual(self.judge.preflight_calls, 1)
        self.assertGreaterEqual(self.generator.unload_calls, 1)
        self.assertGreaterEqual(self.judge.unload_calls, 1)

    def test_a_shared_planner_and_writer_engine_is_not_unloaded_between_them(self):
        """The default config uses one model for both stages; unloading
        between them would pay an 18 GB reload for nothing."""
        shared = FakeScriptEngine(["A brief.", CLEAN_SCRIPT])
        from core.genres import load_pack

        config = AutoConfig.from_genre(
            load_pack("grief_and_loss"),
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )
        run(
            "", genre="grief_and_loss", config=config,
            pipeline=StubPipeline(self.dir),
            planner_engine=shared, generator_engine=shared,
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )
        self.assertEqual(shared.unload_calls, 1)

    def test_the_brief_and_genre_land_in_the_metadata(self):
        result, _config = self._run()
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["genre"], "grief_and_loss")
        self.assertIn("brief", meta)
        self.assertTrue(meta["angle"])

    def test_recently_used_angles_are_avoided(self):
        from core.genres import load_pack
        from core.originality import add_to_corpus

        pack = load_pack("grief_and_loss")
        corpus = self.dir / "corpus"
        for angle in list(pack.angles)[:-1]:
            add_to_corpus("x", genre="grief_and_loss", angle=angle.name,
                          corpus_dir=corpus)
        result, _config = self._run()
        self.assertEqual(result.angle, pack.angles[-1].name)

    def test_the_prompt_path_still_works_unchanged(self):
        """Backward compatibility: no genre, positional prompt, as before."""
        result = run(
            "I feel anxious.",
            config=AutoConfig(
                corpus_dir=self.dir / "corpus",
                background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
            ),
            pipeline=StubPipeline(self.dir),
            generator_engine=FakeScriptEngine([CLEAN_SCRIPT]),
            judge_engine=FakeScriptEngine([judged(CLEAN_SCRIPT)]),
        )
        self.assertTrue(result.audio_path)
        self.assertEqual(result.genre, "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_auto_generate.py -v`
Expected: FAIL with `ImportError: cannot import name 'DURATION_BANDS'`

- [ ] **Step 3: Write minimal implementation**

In `core/auto_generate.py`:

1. Replace the default model constants:

```python
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
```

2. Add imports, and extend the linter import Task 9 added — `check_banned_phrases`
   is called for the first time in this task:

```python
from core.script_gen.linter import check_banned_phrases, check_originality
from core.genres import load_pack, pick_angle
from core.originality import avoid_terms, recent_angles
from core.script_gen.planner import plan
from core.script_gen.rules import build_planner_system_prompt
```

3. Add `music_tags: tuple[str, ...] = ()` to `AutoConfig`, and this classmethod:

```python
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
```

4. Add `brief: str = ""` to `ScriptOutcome`, and `brief: str = ""`,
   `genre: str = ""`, `angle: str = ""` to `AutoResult`.

5. Add a planner stage to `generate_script`. Change its signature to accept
   `planner_engine=None` and `pack=None`, `angle=None`, `steer=""`, and
   insert before the existing "Writing draft script" step:

```python
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
```

Then, after the `review(...)` call and after the repair loop exits (i.e. in
both the success return path and the failure path), call
`generator_engine.unload()` once the draft exists and `judge_engine.unload()`
once no more repairs will run. The simplest correct placement is:
`generator_engine.unload()` immediately after `draft(...)` returns, and
`judge_engine.unload()` in a `finally:` around the repair loop.

6. Set `brief=brief` on the returned `ScriptOutcome`.

7. Change `run()`'s signature to
   `def run(prompt: str = "", *, genre: str | None = None, steer: str = "", config=None, pipeline=None, planner_engine=None, generator_engine=None, judge_engine=None, rng=None, progress_cb=None, **pipeline_kwargs)`
   and add, before `generate_script` is called:

```python
    pack, angle = None, None
    if genre:
        pack = load_pack(genre)
        angle = pick_angle(
            pack,
            recent=recent_angles(genre, limit=3, corpus_dir=config.corpus_dir),
            rng=rng,
        )
        config.angle = angle.name
        if planner_engine is None:
            planner_engine = build_engine(
                os.environ.get("MOODSCAPE_SCRIPT_PLANNER", DEFAULT_PLANNER)
            )

    # Fail in seconds rather than five minutes in.
    for engine in (planner_engine, generator_engine, judge_engine):
        if engine is not None:
            engine.preflight()
```

8. Add the per-genre banned-phrase check inside `generate_script`'s lint loop.
   `pack` only exists in this task, which is why the call belongs here rather
   than with the rest of the originality wiring. Insert it immediately after
   the existing `violations = check(...)` call and before `report = None`:

```python
        if pack is not None:
            violations = violations + check_banned_phrases(script, pack.banned)
```

9. Update the `generate_script(...)` call inside `run()` to forward the new
   arguments — without this the planner stage is dead code:

```python
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
```

10. Pass `prefer_tags=config.music_tags` to the `pick_background(...)` call.

11. Add `"brief": outcome.brief` to the metadata dict, and
    `brief=outcome.brief, genre=config.genre, angle=config.angle` to the
    returned `AutoResult`.

- [ ] **Step 4: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS — all tests, old and new.

- [ ] **Step 5: Commit**

```bash
git add core/auto_generate.py tests/unit/test_auto_generate.py
git commit -m "feat(auto_generate): genre path with planner, preflight and unload"
```

---

### Task 15: The two-click UI

**Files:**
- Modify: `core/streaming_run.py`
- Modify: `core/auto_tab.py`
- Test: `tests/unit/test_streaming_run.py`, `tests/unit/test_auto_tab.py`

**Interfaces:**
- Consumes: `genre_choices`, `load_pack` (Task 10); `DURATION_BANDS`, `AutoConfig.from_genre` (Task 14)
- Produces:
  - `StreamingRun(prompt="", *, genre=None, steer="", config=None, runner=None, **kwargs)`
  - `auto_tab.build_auto_tab()` returns additional keys `"genre"`, `"band"`, `"steer"`
  - `auto_tab.genre_dropdown_choices() -> list[tuple[str, str]]` — `[("Family — Label", slug), ...]`

Gradio dropdowns have no native option groups, so the family is folded into
the visible label. 46 entries stay scannable because they sort by family.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_streaming_run.py`:

```python
class GenreModeTest(unittest.TestCase):
    def test_an_empty_prompt_is_fine_when_a_genre_is_given(self):
        seen = {}

        def runner(prompt, **kwargs):
            seen.update({"prompt": prompt, **kwargs})
            return "RESULT"

        stream = StreamingRun("", genre="fall_asleep", runner=runner)
        list(stream)
        self.assertEqual(stream.result, "RESULT")
        self.assertFalse(stream.invalid_input)
        self.assertEqual(seen["genre"], "fall_asleep")

    def test_no_prompt_and_no_genre_is_still_rejected(self):
        stream = StreamingRun("", runner=lambda *a, **k: None)
        list(stream)
        self.assertTrue(stream.invalid_input)
        self.assertIn("genre", stream.error.lower())

    def test_steer_text_is_forwarded(self):
        seen = {}

        def runner(prompt, **kwargs):
            seen.update(kwargs)
            return "RESULT"

        list(StreamingRun("", genre="fall_asleep", steer="by the sea", runner=runner))
        self.assertEqual(seen["steer"], "by the sea")
```

Append to `tests/unit/test_auto_tab.py`:

```python
class GenreControlsTest(unittest.TestCase):
    def test_dropdown_choices_cover_every_pack_and_name_the_family(self):
        from core.auto_tab import genre_dropdown_choices

        choices = genre_dropdown_choices()
        self.assertEqual(len(choices), 46)
        labels = [label for label, _slug in choices]
        self.assertTrue(any("Sleep & Rest — Fall Asleep" == l for l in labels))
        self.assertEqual(len(set(labels)), len(labels))

    def test_every_choice_value_is_a_loadable_slug(self):
        from core.auto_tab import genre_dropdown_choices
        from core.genres import load_pack

        for _label, slug in genre_dropdown_choices():
            load_pack(slug)

    def test_the_tab_exposes_the_genre_band_and_steer_controls(self):
        import gradio as gr

        from core.auto_tab import build_auto_tab

        with gr.Blocks():
            components = build_auto_tab()
        for key in ("genre", "band", "steer"):
            self.assertIn(key, components)

    def test_changing_genre_prefills_the_content_type(self):
        from core.auto_tab import content_type_for_genre

        self.assertEqual(content_type_for_genre("fall_asleep"), "sleep_story")
        self.assertEqual(content_type_for_genre("grief_and_loss"), "meditation")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_streaming_run.py tests/unit/test_auto_tab.py -v`
Expected: FAIL with `TypeError: StreamingRun.__init__() got an unexpected keyword argument 'genre'`

- [ ] **Step 3: Write minimal implementation**

In `core/streaming_run.py`, change `__init__` and the validation:

```python
    def __init__(
        self,
        prompt: str = "",
        *,
        genre: str | None = None,
        steer: str = "",
        config=None,
        runner=None,
        **kwargs,
    ):
        self._prompt = prompt
        self._genre = genre
        self._steer = steer
        self._config = config
        self._runner = runner if runner is not None else default_run
        self._kwargs = kwargs
        self.result: AutoResult | None = None
        self.error: str | None = None
        self.invalid_input: bool = False
```

In `__iter__`, replace the prompt guard:

```python
        if not self._genre and not (self._prompt or "").strip():
            self.error = "Pick a genre, or enter a prompt first."
            self.invalid_input = True
            return
```

and extend the runner call:

```python
                self.result = self._runner(
                    self._prompt,
                    genre=self._genre,
                    steer=self._steer,
                    config=self._config,
                    progress_cb=progress_cb,
                    **self._kwargs,
                )
```

In `core/auto_tab.py`, add:

```python
from core.auto_generate import (
    DEFAULT_GENERATOR,
    DEFAULT_JUDGE,
    DEFAULT_PLANNER,
    DURATION_BANDS,
    AutoConfig,
)
from core.genres import genre_choices, load_pack

BAND_CHOICES = [
    ("3–6 min", "short"),
    ("6–10 min", "medium"),
    ("10–15 min", "long"),
]


def genre_dropdown_choices() -> list[tuple[str, str]]:
    """Flatten the family grouping into Gradio's (label, value) pairs.

    Gradio dropdowns have no option groups, so the family is folded into the
    visible label. Sorting by family keeps 46 entries scannable.
    """
    return [
        (f"{family} — {label}", slug)
        for family, entries in genre_choices()
        for label, slug in entries
    ]


def content_type_for_genre(slug: str) -> str:
    """The audio profile a genre renders as — used to pre-fill the dropdown."""
    return load_pack(slug).content_type
```

Replace `auto_generate_handler` with:

```python
def auto_generate_handler(
    genre,
    band,
    steer,
    content_type,
    tts_engine,
    planner_spec,
    generator_spec,
    judge_spec,
):
    """Genre + length -> finished meditation, streaming progress to the UI."""
    os.environ["MOODSCAPE_SCRIPT_PLANNER"] = planner_spec
    os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = generator_spec
    os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge_spec

    if not genre:
        yield None, "", "", "Pick a genre first."
        return

    # content_type comes from the dropdown, which the genre change handler
    # pre-filled from the pack. The user's override, if any, wins -- run()
    # never re-derives it.
    config = AutoConfig.from_genre(
        load_pack(genre), band=band, content_type=content_type,
        tts_engine=tts_engine,
    )

    run = StreamingRun("", genre=genre, steer=steer, config=config)
    for update in run:
        yield None, "", "", update.message

    if run.result is None:
        message = run.error if run.invalid_input else f"Failed: {run.error}"
        yield None, "", "", message
        return

    result = run.result
    status = (
        f"Done. {result.genre} / {result.angle}. "
        f"Background: {result.background}. "
        f"Estimated {result.estimated_sec / 60:.1f} min."
    )
    advisories = "\n".join(f"- [{v.code}] {v.message}" for v in result.violations)
    if advisories:
        status = f"{status}\n\nAdvisories:\n{advisories}"

    yield result.audio_path, result.script, result.changelog, status
```

In `build_auto_tab()`, replace the `prompt` textbox with:

```python
                genre = gr.Dropdown(
                    choices=genre_dropdown_choices(),
                    value="stress_relief",
                    label="Genre",
                    elem_classes="dropdown-container",
                )
                band = gr.Radio(
                    choices=BAND_CHOICES,
                    value="medium",
                    label="Length",
                )
                with gr.Accordion(
                    "Steer this one", open=False, elem_classes="accordion-section"
                ):
                    steer = gr.Textbox(
                        label="Anything else? (optional)",
                        placeholder="by the ocean · for a night shift",
                        lines=2,
                    )
```

Add a planner model textbox next to the generator and judge ones, add the
genre change handler, and update the click wiring:

```python
        genre.change(
            fn=content_type_for_genre, inputs=[genre], outputs=[content_type]
        )
        button.click(
            fn=auto_generate_handler,
            inputs=[genre, band, steer, content_type, tts_engine,
                    planner, generator, judge],
            outputs=[audio, script, changelog, status],
        )
```

Add `"genre": genre, "band": band, "steer": steer, "planner": planner` to the
returned dict and remove the `"prompt"` key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/ -v`
Expected: PASS

- [ ] **Step 5: Verify the tab renders**

```bash
.venv/bin/python -c "
import gradio as gr
from core.auto_tab import build_auto_tab
with gr.Blocks() as demo:
    build_auto_tab()
print('Auto-Generate tab builds cleanly')
"
```

- [ ] **Step 6: Commit**

```bash
git add core/auto_tab.py core/streaming_run.py \
        tests/unit/test_auto_tab.py tests/unit/test_streaming_run.py
git commit -m "feat(auto_tab): genre dropdown and duration band replace the prompt box"
```

---

### Task 16: End-to-end test, eval harness, and docs

**Files:**
- Create: `tests/integration/test_genre_e2e.py`
- Create: `scripts/eval_genres.py`
- Modify: `CLAUDE.md`, `docs/auto_generation/README.md`

**Why a new file rather than extending `test_auto_generate_e2e.py`:** that
file is gated behind `MOODSCAPE_E2E=1` and drives the *real* pipeline, so
everything in it renders audio and is skipped by default. These tests use a
stub pipeline and must always run, so they get their own file with its own
stub. (The spec said to reuse that file's "stub-pipeline pattern" — the stub
pattern actually lives in `tests/unit/test_auto_generate.py`; the spec was
wrong about where.)

**Interfaces:**
- Consumes: everything above
- Produces: `scripts/eval_genres.py` CLI

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_genre_e2e.py`:

```python
"""Genre -> finished meditation, end to end with a stub pipeline.

No model, no network, no audio rendering, so this always runs -- unlike
test_auto_generate_e2e.py, which drives the real pipeline behind
MOODSCAPE_E2E=1.
"""

import json
import tempfile
import unittest
from pathlib import Path


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


REALISTIC_SCRIPT = (
    "Let the day set itself down for a moment.\n\n"
    "[pause:6s]\n\n"
    "Copper light moves slowly along the far wall, and nothing here needs "
    "deciding tonight.\n\n"
    "[pause:8s]\n\n"
    "Feel the weight of your hands where they rest.\n\n"
    "[pause:6s]\n\n"
    "When you are ready, let the room come back.\n"
)


def judged(script: str) -> str:
    return f"<script>\n{script}\n</script>\n<changelog>\n- none\n</changelog>"


class GenreEndToEndTest(unittest.TestCase):
    """Genre -> audio with a stub pipeline. No model, no network, no render."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, genre="stress_relief"):
        from core.auto_generate import AutoConfig
        from core.genres import load_pack

        return AutoConfig.from_genre(
            load_pack(genre),
            band="medium",
            corpus_dir=self.dir / "corpus",
            background_scan=lambda: [("Bed — 10:00", "/bg/a.mp3")],
        )

    def test_a_genre_run_produces_audio_script_and_metadata(self):
        from core.auto_generate import run
        from core.script_gen.engine import FakeScriptEngine

        config = self._config()
        result = run(
            "",
            genre="stress_relief",
            config=config,
            pipeline=StubPipeline(self.dir),
            planner_engine=FakeScriptEngine(["A brief."]),
            generator_engine=FakeScriptEngine([REALISTIC_SCRIPT]),
            judge_engine=FakeScriptEngine([
                f"<script>\n{REALISTIC_SCRIPT}\n</script>\n<changelog>\n- none\n</changelog>"
            ]),
        )
        self.assertTrue(Path(result.audio_path).is_file())
        self.assertTrue(Path(result.script_path).is_file())
        meta = json.loads(Path(result.meta_path).read_text())
        self.assertEqual(meta["genre"], "stress_relief")
        self.assertTrue(meta["angle"])

    def test_the_same_genre_twice_with_the_same_text_is_rejected(self):
        """The requirement, end to end: no two meditations may be the same."""
        from core.auto_generate import ScriptGenerationError, run
        from core.script_gen.engine import FakeScriptEngine

        config = self._config()

        def once():
            return run(
                "",
                genre="stress_relief",
                config=config,
                pipeline=StubPipeline(self.dir),
                planner_engine=FakeScriptEngine(["A brief."]),
                generator_engine=FakeScriptEngine([REALISTIC_SCRIPT]),
                judge_engine=FakeScriptEngine([
                    f"<script>\n{REALISTIC_SCRIPT}\n</script>\n<changelog>\n- none\n</changelog>"
                ]),
            )

        once()
        with self.assertRaises(ScriptGenerationError) as ctx:
            once()
        self.assertIn("PASSAGE_LIFTED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails, then passes**

Run: `.venv/bin/python -m pytest tests/integration/test_genre_e2e.py -v`

If the first run already passes, the work of Tasks 10–15 is correct and this
test is a regression guard. If it fails, fix the implementation — not the
test.

- [ ] **Step 3: Write the eval harness**

Create `scripts/eval_genres.py`:

```python
#!/usr/bin/env python
"""Render a matrix of genres x model configurations for listening tests.

Benchmarks rank models on generic creative writing. They cannot tell you
which model writes the better MEDITATION, which is what this decides. Writes
every script, audio file and metric into one directory so configurations can
be compared by ear.

    python scripts/eval_genres.py --genres stress_relief,grief_and_loss \\
        --configs "qwen=ollama:qwen3.8:27b|ollama:gemma4:31b" \\
                  "muse=ollama:muse-glimmer:30b|ollama:gemma4:31b" \\
        --out /tmp/eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.auto_generate import AutoConfig, run  # noqa: E402
from core.genres import load_all_packs, load_pack  # noqa: E402


def _parse_config(spec: str) -> tuple[str, str, str]:
    """'name=writer_spec|judge_spec' -> (name, writer_spec, judge_spec)."""
    name, _, models = spec.partition("=")
    writer, _, judge = models.partition("|")
    if not (name and writer and judge):
        raise SystemExit(f"Malformed --configs entry: {spec!r}")
    return name, writer, judge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genres", default="", help="comma-separated slugs; default all")
    parser.add_argument("--band", default="medium", choices=["short", "medium", "long"])
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=1, help="runs per genre per config")
    args = parser.parse_args()

    slugs = (
        [s.strip() for s in args.genres.split(",") if s.strip()]
        or sorted(load_all_packs())
    )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in args.configs:
        name, writer, judge = _parse_config(spec)
        os.environ["MOODSCAPE_SCRIPT_PLANNER"] = writer
        os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = writer
        os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge

        for slug in slugs:
            for index in range(args.runs):
                out_dir = out_root / name / f"{slug}-{index}"
                out_dir.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                try:
                    result = run(
                        "",
                        genre=slug,
                        config=AutoConfig.from_genre(load_pack(slug), band=args.band),
                        output_dir=str(out_dir),
                    )
                    rows.append(
                        {
                            "config": name, "genre": slug, "run": index,
                            "ok": True,
                            "seconds": round(time.monotonic() - started, 1),
                            "angle": result.angle,
                            "estimated_sec": round(result.estimated_sec, 1),
                            "originality": round(result.originality, 3),
                            "advisories": [v.code for v in result.violations],
                            "audio": result.audio_path,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "config": name, "genre": slug, "run": index,
                            "ok": False,
                            "seconds": round(time.monotonic() - started, 1),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                print(json.dumps(rows[-1]), flush=True)

    (out_root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = sum(1 for row in rows if row["ok"])
    print(f"\n{ok}/{len(rows)} runs succeeded. Results: {out_root / 'results.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

**If `MeditationPipeline.generate()` does not accept an `output_dir` kwarg,
drop that argument** rather than modifying the pipeline — the pipeline is out
of scope for this plan. Move the produced files into `out_dir` afterwards
instead.

- [ ] **Step 4: Update the documentation**

In `CLAUDE.md`:
- Under **Auto-Generation Flow**, insert the planner as step 0 and note that
  genre + duration band replaces the free-text prompt.
- Add the new env vars to the relevant list.
- Add two entries to **Top Gotchas**:
  - *Ollama ignores `keep_alive` on `/v1`* — it is honoured only on `/api/*`,
    so `ScriptEngine.unload()` exists and must be called between stages, or an
    18 GB model stays resident while F5 loads and the machine swaps.
  - *Originality repair depends on `<problems>` stripping* — quoting an
    overlap back to the judge re-injects it unless `parse_judge_response()`
    strips the block first (commit `c372b18`).
- Add `docs/genre_packs/` to the **Folder Map** and the **Where to Look** table.

In `docs/auto_generation/README.md`: document the genre path, the pack format,
the two-tier music tags, the originality thresholds and how to calibrate them,
and `scripts/eval_genres.py`.

- [ ] **Step 5: Run everything**

```bash
.venv/bin/python -m pytest tests/unit/ -v
.venv/bin/python -m pytest tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_genre_e2e.py scripts/eval_genres.py \
        CLAUDE.md docs/auto_generation/README.md
git commit -m "test: genre end-to-end coverage, eval harness, and docs"
```

---

## First real run

Not a task — the checklist for the first genuine generation after the plan
is complete.

```bash
ollama pull qwen3.8:27b     # ~18 GB
ollama pull gemma4:31b      # ~19 GB
df -h /                     # needs ~37 GB free; the volume was at 91%
.venv/bin/python app.py     # Auto-Generate tab -> pick a genre -> Generate
```

Expect roughly 5.3 minutes of LLM time before the audio render begins. If
that is too slow for iterating on packs, every stage takes a hosted
open-weight provider with no code change:

```bash
export MOODSCAPE_SCRIPT_PLANNER=groq:qwen3.8-27b
export MOODSCAPE_SCRIPT_GENERATOR=groq:qwen3.8-27b
```
