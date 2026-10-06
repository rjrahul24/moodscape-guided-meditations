# Track Q: Qwen3-TTS Sandbox Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Qwen3-TTS (Apache-2.0) as an experimental, meditation-tailored TTS engine, selectable only in the Sandbox tab, so it can be compared with F5 by ear and on the R0 harness.

**Architecture:**
- A new `core/qwen3_tts/` package mirrors `core/chatterbox_tts/`: presets, preprocessor, engine and postprocessor.
- The engine wraps `mlx_audio`'s Qwen3-TTS in three modes (`clone`, `custom`, `design`) behind the existing `SpeechEngine` contract (mono float32 @ 24 kHz plus a bool activity mask).
- Pipeline routing follows the Chatterbox branches. Mastering and the voice chain reuse Chatterbox's dry chain.
- Qwen3 appears in engine lists only when callers ask for experimental engines (the sandbox does; the Manual and Auto tabs do not).

**Tech Stack:** Python 3.11, `mlx-audio` 0.5.4 (installed; ships `mlx_audio/tts/models/qwen3_tts`), MLX, numpy, soxr (via the existing `core.audio_processor`), librosa (time-stretch arm), pedalboard, Gradio, unittest-style tests run by pytest.

**Spec:** `docs/superpowers/specs/2026-10-06-track-q-qwen3-tts-sandbox-design.md`

## Global Constraints

- Work only in the git worktree `../moodscape-qwen3-tts` on branch `qwen3-tts`, created from `dev`. Do not merge, and never push.
- Qwen3 must **not** appear in the Manual or Auto-Generate tabs. The F5 + upload golden path must not change behaviour.
- `mlx-audio` facts:
  - `load_model(repo)` from `mlx_audio.tts.utils`.
  - Base models clone via `model.generate(text=, ref_audio=, ref_text=, temperature=, lang_code=)`, with **no `instruct`**.
  - CustomVoice via `model.generate_custom_voice(text=, speaker=, language="English", instruct=, temperature=)`.
  - VoiceDesign via `model.generate_voice_design(text=, language="English", instruct=, temperature=)`.
  - Each returns an iterable of results with `.audio` (`mx.array`) and `.sample_rate`.
  - `speed` is **not implemented** upstream. Never rely on it.
- Engine output contract: `synthesize(...) -> (np.float32 mono @ 24000 Hz, np.bool_ mask of equal length)`.
- Defaults:
  - `MOODSCAPE_QWEN3_MODE=clone`
  - `MOODSCAPE_QWEN3_TEMPERATURE=0.7`
  - `MOODSCAPE_QWEN3_CHUNK_CHARS=200`
  - `MOODSCAPE_QWEN3_STRETCH=1.0` (off)
  - `MOODSCAPE_QWEN3_DF_WET=0.5`
  - Registry presets: reverb 0.10, duck −16.
- Memory: load on synthesise, unload after (`mx.clear_cache()` + `gc.collect()`). Never resident beside an LLM or F5.
- No pedalboard `Limiter` anywhere.
- `compute_cache_key` must produce **identical hashes** for all existing callers. New Qwen-specific fields go in an optional `extra` dict that is omitted from the key payload when `None`.
- Tests are unittest classes under `tests/unit/test_qwen3_*.py`. The model is mocked in unit tests. The live smoke test is gated by `MOODSCAPE_QWEN3_LIVE=1`.
- Commits use conventional-commit messages ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Mode/argument mismatch**: an `instruct` set by env or the UI while the mode is `clone` must be dropped (with one warning), never passed to `model.generate()`, which would raise. *Pinned in Task 3* (`test_clone_mode_never_passes_instruct`).
2. **Model returns a non-24 kHz rate or several result segments**: these must be concatenated and resampled to exactly 24 kHz. *Pinned in Task 3* (`test_multi_result_and_resample`).
3. **Empty or silent output for a chunk**: retry once with seed + 1000, then raise naming the chunk text. Never return silence that would pass as speech. *Pinned in Task 3* (`test_empty_output_retries_then_raises`).
4. **Existing cache keys change** after the `extra` parameter is added: this would silently invalidate every cached F5/Kokoro voice. *Pinned in Task 4* (`test_existing_cache_key_unchanged`).
5. **Qwen3 leaking into Manual/Auto engine dropdowns**. *Pinned in Task 4* (`test_qwen3_hidden_from_default_engine_list`).

---

### Task 1: Worktree and presets module

**Files:**
- Create: `core/qwen3_tts/__init__.py`
- Create: `core/qwen3_tts/presets.py`
- Test: `tests/unit/test_qwen3_presets.py`

**Interfaces:**
- Produces:
  - `MODES = ("clone", "custom", "design")`
  - `MODEL_REPOS: dict[str, str]`
  - `DEFAULT_SPEAKER: str`
  - `INSTRUCT_PRESETS: dict[str, str]` (keys `meditation`, `sleep_story`)
  - `DESIGN_PRESETS: dict[str, str]` (same keys)
  - `mode_from_env() -> str`
  - `temperature_from_env() -> float`
  - `chunk_chars_from_env() -> int`
  - `stretch_from_env() -> float`
  - `instruct_for(mode: str, content_type: str) -> str | None` (env `MOODSCAPE_QWEN3_INSTRUCT` overrides; returns `None` in clone mode)
  - `speaker_from_env() -> str`

- [ ] **Step 1: Create the worktree**

Run from the main repo (`/Users/rahul/Downloads/moodscape-guided-meditations`):

```bash
git worktree add ../moodscape-qwen3-tts -b qwen3-tts dev
cd ../moodscape-qwen3-tts
ln -s ../moodscape-guided-meditations/.venv .venv
ln -s ../moodscape-guided-meditations/.env .env
.venv/bin/python -m pytest tests/unit/ -q
```

Expected: the worktree is created on the new branch and the unit suite passes (baseline green). All later steps run inside `../moodscape-qwen3-tts`. Do not add the `.venv` or `.env` symlinks to git: check `git status` before each commit, and add them to `.git/info/exclude` if they show up.

- [ ] **Step 2: Verify the MLX model repo ids and preset speakers**

Run:

```bash
.venv/bin/python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author="mlx-community", search="Qwen3-TTS", limit=50):
    print(m.id)
EOF
```

Expected: a list that includes `-Base-`, `-CustomVoice-` and `-VoiceDesign-` variants. Pick, for each mode, the **1.7B bf16** repo if listed (else the 1.7B 8bit, else 0.6B for Base). Record the exact ids in Step 4's `MODEL_REPOS`. If the network is unavailable, use the README's documented ids: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`, `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`, `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`. Note which were used in the findings note (Task 6).

Then find the CustomVoice preset speaker names:

```bash
grep -n -i "speaker\|vivian\|serena" .venv/lib/python3.11/site-packages/mlx_audio/tts/models/qwen3_tts/config.py | head -20
```

Pick a warm female preset (`serena` or `vivian` are documented in the README examples). Use it as `DEFAULT_SPEAKER`.

- [ ] **Step 3: Write the failing test**

```python
# tests/unit/test_qwen3_presets.py
import os
import unittest
from unittest.mock import patch

from core.qwen3_tts import presets


class TestQwen3Presets(unittest.TestCase):
    def test_every_mode_has_a_repo(self):
        self.assertEqual(set(presets.MODEL_REPOS), set(presets.MODES))

    def test_env_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in ("MOODSCAPE_QWEN3_MODE", "MOODSCAPE_QWEN3_TEMPERATURE",
                        "MOODSCAPE_QWEN3_CHUNK_CHARS", "MOODSCAPE_QWEN3_STRETCH"):
                os.environ.pop(key, None)
            self.assertEqual(presets.mode_from_env(), "clone")
            self.assertEqual(presets.temperature_from_env(), 0.7)
            self.assertEqual(presets.chunk_chars_from_env(), 200)
            self.assertEqual(presets.stretch_from_env(), 1.0)

    def test_unknown_mode_rejected(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": "turbo"}):
            with self.assertRaisesRegex(ValueError, "turbo"):
                presets.mode_from_env()

    def test_instruct_none_in_clone_mode(self):
        self.assertIsNone(presets.instruct_for("clone", "meditation"))

    def test_instruct_by_content_type(self):
        os.environ.pop("MOODSCAPE_QWEN3_INSTRUCT", None)
        self.assertIn("slowly", presets.instruct_for("custom", "meditation").lower())
        self.assertIn("bedtime", presets.instruct_for("custom", "sleep_story").lower())
        self.assertIn("narrator", presets.instruct_for("design", "meditation").lower())

    def test_instruct_env_override(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_INSTRUCT": "Whisper it."}):
            self.assertEqual(presets.instruct_for("custom", "meditation"), "Whisper it.")

    def test_unknown_content_type_falls_back_to_meditation(self):
        os.environ.pop("MOODSCAPE_QWEN3_INSTRUCT", None)
        self.assertEqual(presets.instruct_for("custom", "guided_meditation"),
                         presets.INSTRUCT_PRESETS["meditation"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the test to verify it fails, then implement**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_presets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.qwen3_tts'`

```python
# core/qwen3_tts/presets.py
"""Model ids, meditation delivery presets and env-driven settings for Qwen3-TTS."""

import os

MODES = ("clone", "custom", "design")

# Resolved in Task 1 Step 2 from the mlx-community listing; prefer 1.7B bf16.
MODEL_REPOS: dict[str, str] = {
    "clone": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "custom": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

DEFAULT_SPEAKER = "serena"

INSTRUCT_PRESETS: dict[str, str] = {
    "meditation": (
        "Speak slowly and softly in a warm, intimate, unhurried voice, with a "
        "gentle falling cadence at the end of each sentence, as a calm meditation guide."
    ),
    "sleep_story": (
        "Speak very slowly and quietly, soft and low, drowsy and soothing, like "
        "reading a bedtime story to someone who is falling asleep."
    ),
}

DESIGN_PRESETS: dict[str, str] = {
    "meditation": (
        "A warm adult female narrator in her thirties with a low, soft, fully voiced "
        "tone. Calm, unhurried and intimate, like a meditation teacher speaking close "
        "to the microphone."
    ),
    "sleep_story": (
        "A gentle adult female narrator with a low, hushed but fully voiced tone. "
        "Drowsy, soothing and very slow, like a bedtime storyteller."
    ),
}


def mode_from_env() -> str:
    mode = os.environ.get("MOODSCAPE_QWEN3_MODE", "clone").strip().lower()
    if mode not in MODES:
        raise ValueError(f"MOODSCAPE_QWEN3_MODE={mode!r} is not one of {MODES}.")
    return mode


def temperature_from_env() -> float:
    return float(os.environ.get("MOODSCAPE_QWEN3_TEMPERATURE", "0.7"))


def chunk_chars_from_env() -> int:
    return int(os.environ.get("MOODSCAPE_QWEN3_CHUNK_CHARS", "200"))


def stretch_from_env() -> float:
    return float(os.environ.get("MOODSCAPE_QWEN3_STRETCH", "1.0"))


def speaker_from_env() -> str:
    return os.environ.get("MOODSCAPE_QWEN3_SPEAKER", DEFAULT_SPEAKER)


def instruct_for(mode: str, content_type: str) -> str | None:
    """Delivery instruction for custom mode, voice description for design mode."""
    if mode == "clone":
        return None
    override = os.environ.get("MOODSCAPE_QWEN3_INSTRUCT", "").strip()
    if override:
        return override
    key = content_type if content_type in INSTRUCT_PRESETS else "meditation"
    return (DESIGN_PRESETS if mode == "design" else INSTRUCT_PRESETS)[key]
```

```python
# core/qwen3_tts/__init__.py
"""Qwen3-TTS engine (experimental, sandbox only) — Apache-2.0 alternative to F5."""

from core.qwen3_tts import presets

__all__ = ["presets"]
```

Update `MODEL_REPOS` and `DEFAULT_SPEAKER` with the values found in Step 2.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_presets.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add core/qwen3_tts/__init__.py core/qwen3_tts/presets.py tests/unit/test_qwen3_presets.py
git commit -m "feat(qwen3_tts): add model ids and meditation delivery presets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Preprocessor

**Files:**
- Create: `core/qwen3_tts/preprocessor.py`
- Modify: `core/qwen3_tts/__init__.py` (export `prepare_segments`)
- Test: `tests/unit/test_qwen3_preprocessor.py`

**Interfaces:**
- Consumes:
  - `core.f5_tts.preprocessor.parse_script(script, paragraph_pause_sec=...) -> list[dict]` and `normalize_for_f5(text) -> str` (both exist).
  - `core.content_profiles.get_profile(content_type)["f5_paragraph_pause_sec"]`.
  - `presets.chunk_chars_from_env()`.
- Produces:
  - `normalize_for_qwen3(text: str) -> str`.
  - `split_into_chunks(text: str, max_chars: int) -> list[str]`.
  - `prepare_segments(script: str, content_type: str = "meditation") -> list[dict]`. Segment dicts are the same shape as F5's: `{"type":"speech","text":str,"voice":str|None}`, `{"type":"pause","duration_sec":float}`, `{"type":"breath","subtype":str}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qwen3_preprocessor.py
import os
import unittest
from unittest.mock import patch

from core.qwen3_tts.preprocessor import normalize_for_qwen3, prepare_segments, split_into_chunks

LONG = " ".join(f"This is calm sentence number {i} for the listener." for i in range(12))


class TestQwen3Preprocessor(unittest.TestCase):
    def test_pauses_and_breaths_preserved(self):
        segs = prepare_segments("Rest here. [pause:4s] [breath] Feel your hands.")
        types = [s["type"] for s in segs]
        self.assertIn("pause", types)
        self.assertIn("breath", types)
        self.assertEqual(next(s for s in segs if s["type"] == "pause")["duration_sec"], 4.0)

    def test_paragraph_pause_follows_content_type(self):
        med = prepare_segments("One line.\n\nTwo line.", content_type="meditation")
        sleep = prepare_segments("One line.\n\nTwo line.", content_type="sleep_story")
        med_p = next(s for s in med if s["type"] == "pause")["duration_sec"]
        sleep_p = next(s for s in sleep if s["type"] == "pause")["duration_sec"]
        self.assertGreater(med_p, sleep_p)

    def test_chunks_respect_limit_at_sentence_boundaries(self):
        chunks = split_into_chunks(normalize_for_qwen3(LONG), max_chars=200)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertTrue(all(c.endswith(".") for c in chunks))

    def test_single_overlong_sentence_is_not_dropped(self):
        sentence = "word " * 80 + "end."
        chunks = split_into_chunks(sentence.strip(), max_chars=200)
        self.assertEqual(" ".join(chunks).split(), sentence.split())

    def test_env_chunk_size_used(self):
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_CHUNK_CHARS": "80"}):
            speech = [s for s in prepare_segments(LONG) if s["type"] == "speech"]
        self.assertTrue(all(len(s["text"]) <= 80 for s in speech))

    def test_normalization_matches_f5_rules(self):
        self.assertNotIn("...", normalize_for_qwen3("Breathe... and rest."))
        self.assertNotIn("RELAX", normalize_for_qwen3("Now RELAX fully."))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_preprocessor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/qwen3_tts/preprocessor.py
"""Script -> segments for Qwen3-TTS.

Uses the shared F5 script grammar ([pause:Xs], [breath], paragraph pauses by
content type) and F5's text normalisation, wrapped so Qwen can diverge later.
Qwen3-TTS is autoregressive with long context, so chunks can be longer than
F5's; the size is env-tunable (MOODSCAPE_QWEN3_CHUNK_CHARS, default 200).
"""

import re

from core.f5_tts.preprocessor import normalize_for_f5, parse_script
from core.qwen3_tts.presets import chunk_chars_from_env

_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def normalize_for_qwen3(text: str) -> str:
    return normalize_for_f5(text)


def _split_long(sentence: str, max_chars: int) -> list[str]:
    parts, current = [], ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for sentence in (s.strip() for s in _SENTENCES.split(text) if s.strip()):
        pieces = [sentence] if len(sentence) <= max_chars else _split_long(sentence, max_chars)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def prepare_segments(script: str, content_type: str = "meditation") -> list[dict]:
    from core.content_profiles import get_profile

    pause = get_profile(content_type)["f5_paragraph_pause_sec"]
    max_chars = chunk_chars_from_env()
    out: list[dict] = []
    voice = None
    for seg in parse_script(script, paragraph_pause_sec=pause):
        if seg["type"] == "voice":
            voice = seg["voice"]
        elif seg["type"] == "speech":
            for chunk in split_into_chunks(normalize_for_qwen3(seg["text"]), max_chars):
                out.append({"type": "speech", "text": chunk, "voice": voice})
        else:
            out.append(seg)
    return out
```

Add `prepare_segments` to `core/qwen3_tts/__init__.py` exports.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_preprocessor.py -v`
Expected: PASS (6 tests). If `test_chunks_respect_limit_at_sentence_boundaries` fails because `normalize_for_f5` changes trailing periods, inspect its output on `LONG` and assert on the real terminal punctuation instead of weakening the length check.

- [ ] **Step 5: Commit**

```bash
git add core/qwen3_tts/preprocessor.py core/qwen3_tts/__init__.py tests/unit/test_qwen3_preprocessor.py
git commit -m "feat(qwen3_tts): add preprocessor with sentence-boundary chunking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Engine and output conditioning

**Files:**
- Create: `core/qwen3_tts/postprocessor.py`
- Create: `core/qwen3_tts/engine.py`
- Modify: `core/qwen3_tts/__init__.py` (exports)
- Test: `tests/unit/test_qwen3_engine.py`

**Interfaces:**
- Consumes:
  - `presets.*` (Task 1).
  - `core.f5_tts.voice_registry.get_voice(slug)` and `scan()`. These map slug → phase → `{"audio": Path, "transcript": Path | str}`.
  - `core.breath_sounds.load_breath(subtype, target_sr=24000) -> np.ndarray`.
  - `core.speech_engine.SpeechEngine`, `SAMPLE_RATE` (24000).
  - `core.audio_processor.resample_highly_accurate(audio, from_sr, to_sr)` (soxr VHQ; already used by `pipeline.py`).
- Produces:
  - `postprocessor.condition_chunk(audio: np.ndarray, sr: int) -> np.ndarray` (resample to 24 k, trim leading/trailing silence below −45 dBFS, 5 ms edge fades).
  - `postprocessor.time_stretch(audio: np.ndarray, rate: float) -> np.ndarray` (identity when `rate == 1.0`).
  - `postprocessor.ChatterboxMasteringEngine` and `build_chatterbox_voice_chain` re-exported as `Qwen3MasteringEngine` and `build_qwen3_voice_chain`.
  - `INTER_CHUNK_GAP_SEC = 0.4`.
  - `class Qwen3Engine(SpeechEngine)`:
    - `__init__(voice_slug: str | None = None, *, mode: str | None = None, loader=None)`
    - `load_model()`, `unload_model()`
    - `synthesize(segments, voice=None, speed=1.0, progress_cb=None, seed=None, content_type="meditation") -> tuple[np.ndarray, np.ndarray]`
    - `get_available_voices()`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qwen3_engine.py
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from core.qwen3_tts.engine import Qwen3Engine
from core.qwen3_tts.postprocessor import condition_chunk, time_stretch


def _result(seconds=1.0, sr=24000, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return SimpleNamespace(audio=(amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sample_rate=sr)


def _engine(mode, model):
    with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": mode}):
        eng = Qwen3Engine(voice_slug="Brittney", loader=lambda repo: model)
    eng.load_model()
    return eng


SEGS = [{"type": "speech", "text": "Rest here.", "voice": None},
        {"type": "pause", "duration_sec": 1.0},
        {"type": "speech", "text": "Feel your hands.", "voice": None}]


class TestPostprocessor(unittest.TestCase):
    def test_trims_silence_and_resamples(self):
        sr = 48000
        tone = _result(1.0, sr).audio
        padded = np.concatenate([np.zeros(sr // 2, np.float32), tone, np.zeros(sr // 2, np.float32)])
        out = condition_chunk(padded, sr)
        self.assertAlmostEqual(len(out) / 24000, 1.0, delta=0.05)

    def test_stretch_identity(self):
        x = _result().audio
        self.assertIs(time_stretch(x, 1.0), x)


class TestQwen3Engine(unittest.TestCase):
    def test_clone_mode_never_passes_instruct(self):
        model = MagicMock()
        model.generate.return_value = [_result()]
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_INSTRUCT": "Whisper."}):
            _engine("clone", model).synthesize(SEGS)
        kwargs = model.generate.call_args.kwargs
        self.assertNotIn("instruct", kwargs)
        self.assertIn("ref_audio", kwargs)
        self.assertTrue(kwargs["ref_text"])

    def test_custom_mode_uses_speaker_and_instruct(self):
        model = MagicMock()
        model.generate_custom_voice.return_value = [_result()]
        _engine("custom", model).synthesize(SEGS, content_type="sleep_story")
        kwargs = model.generate_custom_voice.call_args.kwargs
        self.assertIn("bedtime", kwargs["instruct"].lower())
        self.assertEqual(kwargs["language"], "English")
        self.assertIn("speaker", kwargs)

    def test_design_mode_uses_description(self):
        model = MagicMock()
        model.generate_voice_design.return_value = [_result()]
        _engine("design", model).synthesize(SEGS)
        self.assertIn("narrator", model.generate_voice_design.call_args.kwargs["instruct"].lower())

    def test_contract_and_pause_length(self):
        model = MagicMock()
        model.generate.return_value = [_result(1.0)]
        audio, activity = _engine("clone", model).synthesize(SEGS)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(activity.dtype, np.bool_)
        self.assertEqual(len(audio), len(activity))
        self.assertGreaterEqual(len(audio) / 24000, 2.0 + 1.0)  # two ~1 s chunks + 1 s pause

    def test_multi_result_and_resample(self):
        model = MagicMock()
        model.generate.return_value = [_result(0.5, 48000), _result(0.5, 48000)]
        audio, _ = _engine("clone", model).synthesize(SEGS[:1])
        self.assertAlmostEqual(len(audio) / 24000, 1.0, delta=0.1)

    def test_seed_progression(self):
        model = MagicMock()
        model.generate.return_value = [_result()]
        with patch("core.qwen3_tts.engine._seed") as seed:
            _engine("clone", model).synthesize(SEGS, seed=10)
        self.assertEqual([c.args[0] for c in seed.call_args_list], [10, 11])

    def test_empty_output_retries_then_raises(self):
        model = MagicMock()
        model.generate.return_value = [SimpleNamespace(audio=np.zeros(10, np.float32), sample_rate=24000)]
        with self.assertRaisesRegex(RuntimeError, "Rest here"):
            _engine("clone", model).synthesize(SEGS[:1], seed=5)
        self.assertEqual(model.generate.call_count, 2)

    def test_missing_weights_names_repo(self):
        def broken(repo):
            raise OSError("not found")
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": "clone"}):
            eng = Qwen3Engine(voice_slug="Brittney", loader=broken)
        with self.assertRaisesRegex(RuntimeError, "Qwen3-TTS"):
            eng.load_model()

    def test_unload_clears_model(self):
        eng = _engine("clone", MagicMock())
        eng.unload_model()
        self.assertIsNone(eng._model)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `postprocessor.py`**

```python
# core/qwen3_tts/postprocessor.py
"""Output conditioning for Qwen3-TTS chunks, plus the (shared) voice chain.

Qwen's codec-decoded output is close to Chatterbox's in character, so the
mastering engine and voice FX chain reuse Chatterbox's dry chain.
"""

import numpy as np

from core.chatterbox_tts.postprocessor import (
    ChatterboxMasteringEngine as Qwen3MasteringEngine,
    build_chatterbox_voice_chain as build_qwen3_voice_chain,
)

TARGET_SR = 24000
_TRIM_DBFS = -45.0
_FADE_MS = 5.0


def condition_chunk(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 24 kHz, trim edge silence, apply short edge fades."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != TARGET_SR:
        from core.audio_processor import resample_highly_accurate
        x = resample_highly_accurate(x, sr, TARGET_SR).astype(np.float32)
    if x.size == 0:
        return x
    loud = np.flatnonzero(np.abs(x) > 10 ** (_TRIM_DBFS / 20.0))
    if loud.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = x[loud[0]: loud[-1] + 1].copy()
    fade = min(len(x) // 2, int(TARGET_SR * _FADE_MS / 1000.0))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Optional pacing arm (rate < 1 slows). Identity at 1.0."""
    if rate == 1.0:
        return audio
    import librosa
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate).astype(np.float32)


__all__ = ["Qwen3MasteringEngine", "build_qwen3_voice_chain", "condition_chunk", "time_stretch"]
```

- [ ] **Step 4: Write `engine.py`**

```python
# core/qwen3_tts/engine.py
"""Qwen3-TTS SpeechEngine on mlx-audio (experimental, sandbox only).

Three mutually exclusive modes, because the upstream model variants are:
  clone  -> Base model, our reference clip + transcript, no instruct
  custom -> CustomVoice model, preset speaker + delivery instruct
  design -> VoiceDesign model, instruct is a narrator description
Upstream `speed` is not implemented; pacing comes from pauses (and the
optional MOODSCAPE_QWEN3_STRETCH arm).
"""

import gc
import logging
import os
from pathlib import Path

import numpy as np

from core.qwen3_tts import presets
from core.qwen3_tts.postprocessor import TARGET_SR, condition_chunk, time_stretch
from core.speech_engine import SAMPLE_RATE, SpeechEngine

logger = logging.getLogger("moodscape.qwen3_tts")

INTER_CHUNK_GAP_SEC = 0.4
_RETRY_SEED_OFFSET = 1000


def _default_loader(repo: str):
    from mlx_audio.tts.utils import load_model
    return load_model(repo)


def _seed(value: int) -> None:
    import mlx.core as mx
    mx.random.seed(int(value))


def _to_numpy(audio) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class Qwen3Engine(SpeechEngine):
    def __init__(self, voice_slug: str | None = None, *, mode: str | None = None, loader=None) -> None:
        self.voice_slug = voice_slug
        self.mode = mode or presets.mode_from_env()
        self._loader = loader or _default_loader
        self._model = None
        self._warned_instruct = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def load_model(self) -> None:
        repo = presets.MODEL_REPOS[self.mode]
        try:
            self._model = self._loader(repo)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Qwen3-TTS ({self.mode}) from {repo}: {exc}. "
                f"Download it with: huggingface-cli download {repo}"
            ) from exc

    def unload_model(self) -> None:
        self._model = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def get_available_voices(self) -> list[dict]:
        from core.f5_tts import voice_registry
        return [{"id": slug, "name": slug, "description": "Reference clone (clone mode)"}
                for slug in sorted(voice_registry.scan())]

    # ── synthesis ────────────────────────────────────────────────────────
    def _reference(self, voice: str | None) -> tuple[str, str]:
        from core.f5_tts import voice_registry
        registry = voice_registry.scan()
        slug = voice or self.voice_slug or sorted(registry)[0]
        phase = voice_registry.get_voice(slug)["default"]
        transcript = phase["transcript"]
        text = Path(transcript).read_text(encoding="utf-8").strip() if isinstance(transcript, Path) else str(transcript)
        return str(phase["audio"]), text

    def _call(self, text: str, *, content_type: str, temperature: float, ref: tuple[str, str] | None):
        instruct = presets.instruct_for(self.mode, content_type)
        if self.mode == "clone":
            if os.environ.get("MOODSCAPE_QWEN3_INSTRUCT") and not self._warned_instruct:
                logger.warning("Qwen3 clone mode ignores MOODSCAPE_QWEN3_INSTRUCT (Base model has no instruct).")
                self._warned_instruct = True
            return self._model.generate(text=text, ref_audio=ref[0], ref_text=ref[1],
                                        temperature=temperature, lang_code="english")
        if self.mode == "custom":
            return self._model.generate_custom_voice(text=text, speaker=presets.speaker_from_env(),
                                                     language="English", instruct=instruct,
                                                     temperature=temperature)
        return self._model.generate_voice_design(text=text, language="English", instruct=instruct,
                                                 temperature=temperature)

    def _render(self, text: str, seed: int, **kwargs) -> np.ndarray:
        for attempt_seed in (seed, seed + _RETRY_SEED_OFFSET):
            _seed(attempt_seed)
            results = list(self._call(text, **kwargs))
            parts = [condition_chunk(_to_numpy(r.audio), int(r.sample_rate)) for r in results]
            audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            if audio.size and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4:
                return audio
            logger.warning("Qwen3 produced empty audio (seed %d) for %r; retrying", attempt_seed, text[:60])
        raise RuntimeError(f"Qwen3-TTS produced no audio for chunk: {text!r}")

    def synthesize(self, segments, voice=None, speed=1.0, progress_cb=None, seed=None,
                   content_type: str = "meditation"):
        if self._model is None:
            self.load_model()
        import random
        base_seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        temperature = presets.temperature_from_env()
        stretch = presets.stretch_from_env()
        ref = self._reference(voice) if self.mode == "clone" else None
        gap = np.zeros(int(INTER_CHUNK_GAP_SEC * TARGET_SR), dtype=np.float32)

        audio_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        speech_idx = 0
        total = len(segments)
        for index, seg in enumerate(segments):
            if seg["type"] == "speech":
                chunk = self._render(seg["text"], base_seed + speech_idx, content_type=content_type,
                                     temperature=temperature, ref=ref)
                chunk = time_stretch(chunk, stretch)
                speech_idx += 1
                threshold = float(np.abs(chunk).mean()) * 0.15
                if audio_parts and mask_parts and mask_parts[-1].any():
                    audio_parts.append(gap)
                    mask_parts.append(np.zeros(len(gap), dtype=bool))
                audio_parts.append(chunk)
                mask_parts.append(np.abs(chunk) > threshold)
            elif seg["type"] == "pause":
                silence = np.zeros(int(float(seg["duration_sec"]) * TARGET_SR), dtype=np.float32)
                audio_parts.append(silence)
                mask_parts.append(np.zeros(len(silence), dtype=bool))
            elif seg["type"] == "breath":
                from core.breath_sounds import load_breath
                breath = load_breath(seg["subtype"], target_sr=TARGET_SR).astype(np.float32)
                audio_parts.append(breath)
                mask_parts.append(np.zeros(len(breath), dtype=bool))
            if progress_cb:
                progress_cb(index + 1, total)

        if not audio_parts:
            return np.zeros(0, np.float32), np.zeros(0, bool)
        assert TARGET_SR == SAMPLE_RATE
        return np.concatenate(audio_parts).astype(np.float32), np.concatenate(mask_parts).astype(bool)
```

Update `core/qwen3_tts/__init__.py` to export `Qwen3Engine`, `prepare_segments`, `Qwen3MasteringEngine` and `build_qwen3_voice_chain`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_engine.py -v`
Expected: PASS (11 tests). If `condition_chunk`'s trim threshold removes a 0.3-amplitude tone's zero-crossing ends, the duration assertions still hold within their deltas. If `resample_highly_accurate` has a different name or signature, check `core/audio_processor.py` and use the soxr-VHQ helper `pipeline.py` uses for 24 → 44.1 kHz.

- [ ] **Step 6: Commit**

```bash
git add core/qwen3_tts/ tests/unit/test_qwen3_engine.py
git commit -m "feat(qwen3_tts): add mlx-audio engine with clone/custom/design modes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Registry, pipeline routing and cache-key separation

**Files:**
- Modify: `core/engine_registry.py` (`BUILTIN_ENGINES["qwen3"]`, `list_all_engines(include_experimental=False)`, `get_engine` branch)
- Modify: `core/tts_cache.py` (`compute_cache_key(..., extra: dict | None = None)`)
- Modify: `core/pipeline.py` (qwen3 branches next to every chatterbox branch)
- Modify: `core/sandbox_tab.py` (the two `list_all_engines()` calls and the dropdown pass `include_experimental=True`)
- Test: `tests/unit/test_qwen3_integration.py`

**Interfaces:**
- Consumes: `Qwen3Engine`, `prepare_segments`, `Qwen3MasteringEngine` and `build_qwen3_voice_chain` (Tasks 2–3).
- Produces:
  - `get_engine("qwen3", voice_slug=...) -> Qwen3Engine`.
  - `list_all_engines(include_experimental: bool = False)`.
  - `compute_cache_key(..., extra=None)`.
  - The pipeline routes `base_engine == "qwen3"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_qwen3_integration.py
import unittest

from core.engine_registry import BUILTIN_ENGINES, get_engine, list_all_engines
from core.tts_cache import compute_cache_key


class TestQwen3Registry(unittest.TestCase):
    def test_builtin_entry(self):
        info = BUILTIN_ENGINES["qwen3"]
        self.assertEqual(info["base_engine"], "qwen3")
        self.assertTrue(info["experimental"])
        self.assertEqual(info["presets"]["reverb_amount"], 0.10)

    def test_qwen3_hidden_from_default_engine_list(self):
        self.assertNotIn("qwen3", [eid for _l, eid in list_all_engines()])
        self.assertIn("qwen3", [eid for _l, eid in list_all_engines(include_experimental=True)])

    def test_get_engine_returns_qwen3(self):
        from core.qwen3_tts.engine import Qwen3Engine
        self.assertIsInstance(get_engine("qwen3", voice_slug="Brittney"), Qwen3Engine)


class TestCacheKey(unittest.TestCase):
    def test_existing_cache_key_unchanged(self):
        # Golden hash captured from the pre-change implementation in Step 2.
        key = compute_cache_key(script="Rest.", content_type="meditation", tts_engine="f5",
                                voice="Brittney", speed=0.8, f5_cfg_strength=2.0)
        self.assertEqual(key, GOLDEN_KEY)

    def test_extra_changes_key(self):
        base = dict(script="Rest.", tts_engine="qwen3", voice="Brittney", speed=0.8)
        a = compute_cache_key(**base, extra={"mode": "clone", "temperature": 0.7})
        b = compute_cache_key(**base, extra={"mode": "custom", "temperature": 0.7})
        self.assertNotEqual(a, b)


class TestPipelineRouting(unittest.TestCase):
    def test_pipeline_source_routes_qwen3(self):
        import inspect
        import core.pipeline as p
        src = inspect.getsource(p)
        self.assertIn("core.qwen3_tts.preprocessor", src)
        self.assertIn("Qwen3MasteringEngine", src)
        self.assertIn("MOODSCAPE_QWEN3_DF_WET", src)


GOLDEN_KEY = "REPLACE_IN_STEP_2"

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Capture the golden cache key before changing `tts_cache.py`**

Run:

```bash
.venv/bin/python -c "from core.tts_cache import compute_cache_key; print(compute_cache_key(script='Rest.', content_type='meditation', tts_engine='f5', voice='Brittney', speed=0.8, f5_cfg_strength=2.0))"
```

Paste the printed 64-hex string into `GOLDEN_KEY` in the test file.

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_integration.py -v`
Expected: FAIL. `test_existing_cache_key_unchanged` passes; the qwen3 tests fail with `KeyError: 'qwen3'` and `TypeError` on `include_experimental` / `extra`.

- [ ] **Step 3: Implement the registry changes**

In `core/engine_registry.py`, add to `BUILTIN_ENGINES` after `chatterbox`:

```python
    "qwen3": {
        "id": "qwen3",
        "name": "Qwen3-TTS (experimental)",
        "description": "Alibaba Qwen3-TTS on MLX (Apache-2.0): clone / custom-voice / voice-design modes.",
        "base_engine": "qwen3",
        "default_voice": "Brittney",
        "experimental": True,
        "presets": {
            "speed": 1.0,
            "reverb_amount": 0.10,
            "duck_amount_db": -16.0,
            "df_wet": 0.5,
            "temperature": 0.7,
            "mode": "clone",
        },
    },
```

Change `list_all_engines` to:

```python
def list_all_engines(include_experimental: bool = False) -> list[tuple[str, str]]:
    """Return all available engines as (label, id) pairs for dropdowns.

    Built-in engines come first, followed by any promoted models. Experimental
    built-ins (e.g. Qwen3-TTS) appear only when include_experimental=True,
    which the Sandbox tab passes and the Manual/Auto tabs do not.
    """
    choices: list[tuple[str, str]] = [
        ("F5-TTS", "f5"),
        ("Kokoro", "kokoro"),
        ("Chatterbox TTS", "chatterbox"),
    ]
    if include_experimental:
        choices.extend(
            (info["name"], eid) for eid, info in BUILTIN_ENGINES.items() if info.get("experimental")
        )
    # (existing promoted-model loop unchanged)
```

In `get_engine`, after the chatterbox branch:

```python
    if norm_id == "qwen3":
        from core.qwen3_tts.engine import Qwen3Engine
        return Qwen3Engine(voice_slug=voice_slug)
```

In `core/sandbox_tab.py`, change all three `list_all_engines()` calls (lines ~436, ~453, ~533) to `list_all_engines(include_experimental=True)`.

- [ ] **Step 4: Implement the cache-key `extra` parameter**

In `core/tts_cache.py`, add `extra: dict | None = None,` as the last parameter of `compute_cache_key`. After building `key_dict`, add:

```python
    if extra:
        key_dict["extra"] = {str(k): extra[k] for k in sorted(extra)}
```

Because `key_dict` is unchanged when `extra` is falsy, every existing hash is identical.

- [ ] **Step 5: Implement the pipeline routing**

In `core/pipeline.py`, make each change next to its Chatterbox counterpart:

1. Engine normalisation (around line 167):

```python
        if base_engine not in ("f5", "kokoro", "chatterbox", "qwen3"):
            lowered = str(tts_engine).lower()
            base_engine = (
                "f5" if "f5" in lowered
                else "chatterbox" if "chatterbox" in lowered
                else "qwen3" if "qwen3" in lowered
                else "kokoro"
            )
```

2. Step 1 preprocessor selection: add before the `elif base_engine == "f5":` line

```python
                elif base_engine == "qwen3":
                    from core.qwen3_tts.preprocessor import prepare_segments as _prepare
```

   Keep `chatterbox` as the first `if`.

3. Cache key block: after the `cfg_val` assignment, add the Qwen extras, extend `cache_voice_id`, and pass `extra`:

```python
                    qwen3_extra = None
                    if base_engine == "qwen3":
                        from core.qwen3_tts import presets as _q3
                        qwen3_extra = {
                            "mode": _q3.mode_from_env(),
                            "temperature": _q3.temperature_from_env(),
                            "instruct": os.environ.get("MOODSCAPE_QWEN3_INSTRUCT", ""),
                            "speaker": _q3.speaker_from_env(),
                            "stretch": _q3.stretch_from_env(),
                            "chunk_chars": _q3.chunk_chars_from_env(),
                        }
                    cache_voice_id = f5_voice_slug if base_engine in ("f5", "chatterbox", "qwen3") else voice
```

   Then add `extra=qwen3_extra,` to the `compute_cache_key(...)` call.

4. Synthesis: add before the final `else:` of the synthesize dispatch

```python
                    elif base_engine == "qwen3":
                        voice_audio, voice_activity = tts.synthesize(
                            segments, voice=f5_voice_slug, speed=speed,
                            progress_cb=tts_progress, seed=seed, content_type=content_type,
                        )
```

5. Cache metadata `"voice"`: change the tuple to `("f5", "chatterbox", "qwen3")`.

6. Mastering engine init: add after the chatterbox `elif`

```python
                elif base_engine == "qwen3":
                    _progress(progress_cb, 0.38, "Preparing Qwen3-TTS mastering chain...")
                    from core.qwen3_tts.postprocessor import Qwen3MasteringEngine
                    mastering_engine = Qwen3MasteringEngine(sample_rate=SAMPLE_RATE)
```

7. DeepFilterNet wet: add after the chatterbox `elif`

```python
                    elif base_engine == "qwen3":
                        df_wet = float(os.environ.get("MOODSCAPE_QWEN3_DF_WET", "0.5"))
```

8. Step 7 voice FX: add after the chatterbox `elif` block

```python
                elif base_engine == "qwen3":
                    from core.qwen3_tts.postprocessor import build_qwen3_voice_chain
                    from core.kokoro_tts.postprocessor import apply_fx
                    voice_chain = build_qwen3_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
```

After the edits, grep to confirm there's one qwen3 branch per chatterbox branch:

Run: `grep -n "chatterbox\|qwen3" core/pipeline.py`
Expected: every chatterbox decision point has a qwen3 sibling. `apply_fx(..., engine=base_engine)` receives `"qwen3"`; read `core/kokoro_tts/postprocessor.py::apply_fx` and confirm an unknown engine name takes the generic path. If it special-cases `"chatterbox"`, mirror that case for `"qwen3"`.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_qwen3_integration.py tests/unit/test_engine_registry.py tests/unit/test_tts_cache.py tests/unit/test_pipeline_cache.py tests/unit/test_sandbox_tab.py tests/unit/test_auto_tab.py -v`
Expected: PASS. If an existing registry test asserts the exact `list_all_engines()` output, it should still pass, because the default list is unchanged.

- [ ] **Step 7: Commit**

```bash
git add core/engine_registry.py core/tts_cache.py core/pipeline.py core/sandbox_tab.py tests/unit/test_qwen3_integration.py
git commit -m "feat(qwen3_tts): register experimental engine and route it through the pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Sandbox UI controls

**Files:**
- Modify: `core/sandbox_tab.py`
- Test: `tests/unit/test_sandbox_tab.py` (add tests)

**Interfaces:**
- Consumes: `presets.MODES`, `presets.DEFAULT_SPEAKER`, `presets.instruct_for` and `BUILTIN_ENGINES["qwen3"]["presets"]`.
- Produces:
  - New components in the dict returned by `build_sandbox_tab()`: `qwen3_settings` (Group), `qwen3_mode`, `qwen3_speaker`, `qwen3_instruct`, `qwen3_temperature`, `qwen3_stretch`.
  - `sandbox_generate_handler` gains trailing parameters `qwen3_mode, qwen3_speaker, qwen3_instruct, qwen3_temperature, qwen3_stretch` (defaults `"clone", presets.DEFAULT_SPEAKER, "", 0.7, 1.0`) and sets the matching `MOODSCAPE_QWEN3_*` env vars, including `MOODSCAPE_QWEN3_DF_WET` from `df_wet_val`.
  - `apply_qwen3_env(mode, speaker, instruct, temperature, stretch, df_wet) -> None` as a module-level, testable helper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_sandbox_tab.py`:

```python
class TestQwen3SandboxControls(unittest.TestCase):
    def test_components_exposed(self):
        with gr.Blocks():
            components = build_sandbox_tab()
        for key in ("qwen3_settings", "qwen3_mode", "qwen3_speaker", "qwen3_instruct",
                    "qwen3_temperature", "qwen3_stretch"):
            self.assertIn(key, components)

    def test_engine_dropdown_offers_qwen3(self):
        with gr.Blocks():
            components = build_sandbox_tab()
        values = [c[1] if isinstance(c, (tuple, list)) else c for c in components["engine_dropdown"].choices]
        self.assertIn("qwen3", values)

    def test_apply_qwen3_env(self):
        import os
        from core.sandbox_tab import apply_qwen3_env
        apply_qwen3_env("custom", "serena", "Soft.", 0.6, 0.95, 0.4)
        self.assertEqual(os.environ["MOODSCAPE_QWEN3_MODE"], "custom")
        self.assertEqual(os.environ["MOODSCAPE_QWEN3_SPEAKER"], "serena")
        self.assertEqual(os.environ["MOODSCAPE_QWEN3_INSTRUCT"], "Soft.")
        self.assertEqual(float(os.environ["MOODSCAPE_QWEN3_TEMPERATURE"]), 0.6)
        self.assertEqual(float(os.environ["MOODSCAPE_QWEN3_STRETCH"]), 0.95)
        self.assertEqual(float(os.environ["MOODSCAPE_QWEN3_DF_WET"]), 0.4)
```

If `build_sandbox_tab()`'s returned dict uses a different key for the engine dropdown, read the existing `required_keys` test and use that key.

Run: `.venv/bin/python -m pytest tests/unit/test_sandbox_tab.py -v -k Qwen3`
Expected: FAIL (`KeyError` / `ImportError`).

- [ ] **Step 2: Implement the helper and the handler parameters**

Add near the top of `core/sandbox_tab.py`:

```python
from core.qwen3_tts import presets as qwen3_presets


def apply_qwen3_env(mode: str, speaker: str, instruct: str, temperature: float,
                    stretch: float, df_wet: float) -> None:
    """Pass sandbox Qwen3 controls to the engine via its MOODSCAPE_QWEN3_* env vars."""
    os.environ["MOODSCAPE_QWEN3_MODE"] = str(mode)
    os.environ["MOODSCAPE_QWEN3_SPEAKER"] = str(speaker)
    os.environ["MOODSCAPE_QWEN3_INSTRUCT"] = str(instruct or "")
    os.environ["MOODSCAPE_QWEN3_TEMPERATURE"] = str(float(temperature))
    os.environ["MOODSCAPE_QWEN3_STRETCH"] = str(float(stretch))
    os.environ["MOODSCAPE_QWEN3_DF_WET"] = str(float(df_wet))
```

Append to `sandbox_generate_handler`'s parameter list (after `uploaded_music_file`):

```python
    qwen3_mode: str = "clone",
    qwen3_speaker: str = qwen3_presets.DEFAULT_SPEAKER,
    qwen3_instruct: str = "",
    qwen3_temperature: float = 0.7,
    qwen3_stretch: float = 1.0,
```

After the existing env-var block in the handler:

```python
    if base_engine == "qwen3":
        apply_qwen3_env(qwen3_mode, qwen3_speaker, qwen3_instruct, qwen3_temperature,
                        qwen3_stretch, df_wet_val)
```

- [ ] **Step 3: Add the controls and wiring**

Inside the "Model & Voice" accordion, after `voice_dropdown`:

```python
                    with gr.Group(visible=False) as qwen3_settings:
                        qwen3_mode = gr.Dropdown(
                            choices=list(qwen3_presets.MODES), value="clone",
                            label="Qwen3 Mode",
                            info="clone = our reference voice; custom = preset speaker + style; design = described voice",
                        )
                        qwen3_speaker = gr.Textbox(value=qwen3_presets.DEFAULT_SPEAKER, label="Preset Speaker (custom mode)")
                        qwen3_instruct = gr.Textbox(
                            value=qwen3_presets.INSTRUCT_PRESETS["meditation"], lines=3,
                            label="Delivery Instruction / Voice Description",
                            info="Ignored in clone mode.",
                        )
                        with gr.Row():
                            qwen3_temperature = gr.Slider(0.3, 1.0, 0.7, step=0.05, label="Temperature")
                            qwen3_stretch = gr.Slider(0.80, 1.0, 1.0, step=0.01, label="Time-Stretch (1.0 = off)")
```

Extend `on_engine_change` to return one extra update, `gr.update(visible=(base == "qwen3"))`. Add `qwen3_settings` to the end of its `outputs=[...]` list.

Extend `on_benchmark_change` to also return `gr.update(value=qwen3_presets.INSTRUCT_PRESETS["sleep_story" if is_sleep else "meditation"])`. Add `qwen3_instruct` to its `outputs` (append at the end, keeping the existing order).

Append `qwen3_mode, qwen3_speaker, qwen3_instruct, qwen3_temperature, qwen3_stretch` to the end of `generate_btn.click(... inputs=[...])`.

Add the six new components to the dict `build_sandbox_tab()` returns.

- [ ] **Step 4: Run the sandbox tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sandbox_tab.py -v`
Expected: PASS (existing plus 3 new)

- [ ] **Step 5: Commit**

```bash
git add core/sandbox_tab.py tests/unit/test_sandbox_tab.py
git commit -m "feat(sandbox): add Qwen3-TTS mode, instruction and sampling controls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Live smoke tests, docs and findings note

**Files:**
- Create: `tests/integration/test_qwen3_live.py`
- Create: `docs/model_implementation_guides/qwen3_tts.md`
- Create: `docs/superpowers/notes/2026-10-qwen3-tts-findings.md`
- Modify: `CLAUDE.md` (model list in Setup & Run: add the Qwen3 repos; Folder Map: add `core/qwen3_tts/`)

**Interfaces:**
- Consumes: everything above.
- Produces: documentation plus a findings note with a recommended mode/settings for the R0 A/B.

- [ ] **Step 1: Write the gated live smoke test**

```python
# tests/integration/test_qwen3_live.py
import os
import unittest

import numpy as np

SCRIPT = (
    "Find a comfortable position, and let your body settle. [pause:3s] "
    "Notice the gentle weight of your hands. [pause:4s] "
    "There is nowhere else you need to be right now."
)


@unittest.skipUnless(os.environ.get("MOODSCAPE_QWEN3_LIVE") == "1", "set MOODSCAPE_QWEN3_LIVE=1 to run")
class TestQwen3Live(unittest.TestCase):
    def _render(self, mode: str):
        from unittest.mock import patch
        from core.qwen3_tts.engine import Qwen3Engine
        from core.qwen3_tts.preprocessor import prepare_segments
        with patch.dict(os.environ, {"MOODSCAPE_QWEN3_MODE": mode}):
            engine = Qwen3Engine(voice_slug="Brittney")
            engine.load_model()
            try:
                return engine.synthesize(prepare_segments(SCRIPT), seed=42)
            finally:
                engine.unload_model()

    def _check(self, audio, activity):
        self.assertFalse(np.isnan(audio).any())
        self.assertGreater(np.abs(audio).max(), 0.01)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        self.assertLess(zcr, 0.4)  # same noise heuristic as the pipeline's vocal sanity check
        seconds = len(audio) / 24000
        self.assertGreater(seconds, 7.0 + 4.0)    # pauses alone are 7 s
        self.assertLess(seconds, 7.0 + 30.0)
        self.assertTrue(activity.any())

    def test_clone(self):
        self._check(*self._render("clone"))

    def test_custom(self):
        self._check(*self._render("custom"))

    def test_design(self):
        self._check(*self._render("design"))
```

Run: `MOODSCAPE_QWEN3_LIVE=1 .venv/bin/python -m pytest tests/integration/test_qwen3_live.py -v -s`
Expected: PASS for all three modes. The first run downloads the three model repos, several GB. If a repo id 404s, fix `MODEL_REPOS` (Task 1 Step 2) and rerun. If a mode's `generate*` method rejects a keyword (e.g. `temperature` on VoiceDesign), read its signature in `mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`, drop only that keyword in `_call`, and add a unit test pinning the call shape.

- [ ] **Step 2: Full pipeline renders for listening**

Render the sandbox's 5-minute meditation and sleep-story scripts through the full pipeline for each mode, plus F5 for reference. This is the same code path the Sandbox tab uses:

```bash
for MODE in clone custom design; do
  MOODSCAPE_QWEN3_MODE=$MODE .venv/bin/python - <<EOF
from core.pipeline import MeditationPipeline
from core.sandbox_scripts import MEDITATION_5MIN_SCRIPT
path, status = MeditationPipeline().generate(
    script=MEDITATION_5MIN_SCRIPT, music_prompt="", tts_engine="qwen3",
    f5_voice_slug="Brittney", speed=1.0, reverb_amount=0.10, duck_amount_db=-16.0,
    music_model="upload", generation_mode="Vocals Only", content_type="meditation", seed=42,
    use_tts_cache=False,
)
print("$MODE", path, status)
EOF
done
```

Expected: three WAV paths printed with no exception. Record wall-clock time per mode, and RAM from Activity Monitor or `/usr/bin/time -l`. Render the F5 reference the same way with `tts_engine="f5"` and `speed=0.80`.

- [ ] **Step 3: Write the implementation guide**

Write `docs/model_implementation_guides/qwen3_tts.md` covering:
- **What it is:** licence Apache-2.0 and the three modes with their exact repo ids.
- **Env flags:** `MOODSCAPE_QWEN3_MODE`, `_TEMPERATURE`, `_CHUNK_CHARS`, `_STRETCH`, `_INSTRUCT`, `_SPEAKER`, `_DF_WET`, with defaults.
- **Instruct presets** (verbatim).
- **Known limits:** upstream `speed` is unimplemented; clone mode has no instruct; it's sandbox-only until promoted.
- **Memory:** load/unload behaviour and measured RAM.
- **How to promote** via the existing sandbox promote flow.

Follow the heading style of `docs/model_implementation_guides/` siblings.

- [ ] **Step 4: Write the findings note**

Write `docs/superpowers/notes/2026-10-qwen3-tts-findings.md` with:
- Model repos used.
- Per-mode wall-clock time and real-time factor, plus peak RAM.
- Listening impressions per mode on the meditation and sleep scripts vs the F5 render: warmth, pacing, artifacts, monotony across the 5 minutes, and pronunciation.
- Any chunk failures or retries seen in logs.
- **Recommendation:** the mode plus temperature/chunk/stretch settings to carry into the R0 A/B against F5, and whether the chunk size of 200 should change.

State plainly anything that wasn't measured.

- [ ] **Step 5: Update `CLAUDE.md`**

In Setup & Run's "Models actually used" sentence, add the Qwen3 repo ids, marked "(sandbox-only, experimental)". In the Folder Map, add `│   ├── qwen3_tts/                     # Qwen3-TTS on MLX (experimental, sandbox only)` next to the other engine packages.

- [ ] **Step 6: Full suite, then commit**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: all pass.

```bash
git add tests/integration/test_qwen3_live.py docs/model_implementation_guides/qwen3_tts.md docs/superpowers/notes/2026-10-qwen3-tts-findings.md CLAUDE.md
git commit -m "docs(qwen3_tts): add implementation guide, live smoke tests and findings note

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Report to the user: the branch name, the commits, the findings note path and the three rendered WAV paths for listening. Do not merge.
