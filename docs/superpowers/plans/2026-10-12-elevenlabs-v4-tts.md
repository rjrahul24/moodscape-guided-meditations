# ElevenLabs v4 TTS Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ElevenLabs `eleven_v4` as a cloud TTS provider behind an
"ElevenLabs | Local Models" toggle, remove all breath/vocal sounds from every
engine, and add a post-generation Remix (voice/music dB) that re-runs only mix →
master → export.

**Architecture:** A new `core/elevenlabs_tts/` package implements the existing
`SpeechEngine` contract at a native 48 kHz (new optional `sample_rate` class
attribute), reusing the shared script grammar so script generation is untouched.
The pipeline gains an `elevenlabs` branch (no upsample, no DeepFilterNet, light
clean chain) and saves a mix session before mixing; `core/remix.py` replays the
mix from that session. Testable UI logic lives in `core/provider_ui.py`;
`app.py` only wires it.

**Tech Stack:** Python 3.11, numpy, pedalboard, pyloudnorm, `elevenlabs`
SDK 2.71.0, Gradio 6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-elevenlabs-v4-tts-design.md`

## Global Constraints

- **Branch / git:** work directly on `dev-test-e11labs` (the user created it
  for this feature; it is the isolation — no extra worktree). Conventional
  commits; every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. **Never push.**
- **Tests:** `.venv/bin/python -m pytest tests/unit/ -q` stays green
  (**1050 passed** at start). Unit tests never touch the network or load real
  models: the ElevenLabs SDK is always faked. New test functions/classes go
  above any `if __name__ == "__main__":` guard.
- **app.py cannot be imported in tests** (it registers `atexit os._exit`).
  Put testable UI logic in `core/provider_ui.py`; `app.py` changes are
  verified with `.venv/bin/python -m py_compile app.py` plus the controller's
  browser check.
- **Local Models must be unchanged** except breath markers now render as
  silence. F5 + uploaded background is the golden path: its local branch in
  `core/pipeline.py` must execute the exact same operations as before.
- **No vocal sounds anywhere:** no breath/inhale/exhale samples, no sighs, no
  reaction tags. Breath markers render as silence of
  `{"breath": 1.2, "inhale": 1.5, "exhale": 1.8}` seconds.
- **ElevenLabs model:** `eleven_v4`, output `pcm_48000`, voice settings only
  `stability` + `similarity_boost` (+ `use_speaker_boost=True`). Never send
  `speed` or `style`.
- **Defaults:** stability 0.5, similarity 0.75, direction
  `[soft, slow, calm voice]`, seed 42, chunk max 1500 chars, hard ceiling 9000,
  crossfade 30 ms, chunk LUFS −21, stitching on (≤3 `previous_request_ids`),
  retries 3, mix sessions kept 5.
- **Env vars:** every TTS-affecting var starts with `MOODSCAPE_ELEVENLABS_`
  (cache fingerprint). `ELEVENLABS_API_KEY` lives in `.env`, never committed,
  never logged.
- **Invariants not to touch:** FATAL/ADVISORY linter split, writer/judge
  independence, `<problems>` stripping, `unload()` after each LLM stage, no
  pedalboard `Limiter` added anywhere new, `atexit` exit hook in `app.py`.
- **Real models / real API:** only the controller runs them (Tasks 0, 18 and 19).
- **Shared test fakes:** `tests/unit/pipeline_fakes.py` (created in Task 9) holds
  `MockTTS24`, `FakeMusic`, `FakeEL`; later tests import them with
  `from tests.unit.pipeline_fakes import ...` (works because tests run via
  `python -m pytest` from the repo root).

## Review Focus

1. **Manual-tab script with no paragraph breaks and >9000 chars** → must be
   split into several API requests on sentence boundaries, never one oversize
   request. Test in Task 3 (`test_oversize_paragraph_splits_under_hard_ceiling`).
2. **Script containing user-typed bracket text such as `[soft]` or a
   sentence ending inside brackets** (Manual tab bypasses the linter) → chunker
   never splits inside `[...]`. Test in Task 3
   (`test_never_splits_inside_brackets`).
3. **ElevenLabs fails on chunk 5 of 9 (402 out of credits)** → error names
   "4 of 9 chunks rendered", and a re-run bills only chunks 5–9. Test in
   Task 5 (`test_midrun_failure_reports_progress_and_resume_is_cheap`).
4. **Remix after the session was pruned or the app restarted** → clear
   "Mix session not found — generate again", never a traceback. Test in
   Task 12 (`test_missing_session_raises_clear_error`).
5. **Toggling back to Local Models after an ElevenLabs run** → local engine
   cache keys and pipeline path identical to before (ElevenLabs params must
   not leak into the local cache key). Test in Task 9
   (`test_local_engine_cache_key_ignores_elevenlabs_params`).

---

## Phase 0 — Setup and baselines (controller)

### Task 0 (controller): SDK install, SDK-surface check, baselines

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install and pin the SDK**

```bash
.venv/bin/pip install "elevenlabs==2.71.0"
```

Add to `requirements.txt` (after `pyloudnorm`):

```
elevenlabs==2.71.0   # ElevenLabs v4 cloud TTS (core/elevenlabs_tts)
```

- [ ] **Step 2: Verify the SDK surface the plan relies on**

```bash
.venv/bin/python - <<'EOF'
import inspect
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings
c = ElevenLabs(api_key="x")
sig = inspect.signature(c.text_to_speech.with_raw_response.convert)
for p in ("voice_id", "text", "model_id", "output_format", "voice_settings",
          "seed", "previous_request_ids", "apply_text_normalization"):
    assert p in sig.parameters, p
assert hasattr(c.voices, "get")
VoiceSettings(stability=0.5, similarity_boost=0.75, use_speaker_boost=True)
print("SDK surface OK")
EOF
```

Expected: `SDK surface OK`. If a name differs, record a ledger `Ruling:` and
adjust **only** `core/elevenlabs_tts/client.py` in Task 4 (it is the single
file that touches the SDK).

- [ ] **Step 3: Unit-test baseline**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: `1050 passed`.

- [ ] **Step 4: Golden-path audio baseline (F5 + upload, real model)**

Run under the heavy-model lock (no other model process running):

```bash
mkdir -p var/eval/e11-baseline
cat > var/eval/e11-baseline/script.txt <<'EOF'
Find a comfortable position and let your eyes close.

Notice the weight of your body resting where you are. [pause:4s]

Let your shoulders soften, and let the day begin to fall away.
EOF
.venv/bin/python - <<'EOF'
import shutil
from core.pipeline import MeditationPipeline
bg = "assets/backgrounds/freemusicforvideo-meditation-background-409198.mp3"
out, status = MeditationPipeline().generate(
    script=open("var/eval/e11-baseline/script.txt").read(), music_prompt="",
    tts_engine="f5", f5_voice_slug=None, music_model="upload",
    uploaded_music_path=bg, seed=1234, use_tts_cache=True)
shutil.copy(out, "var/eval/e11-baseline/before.wav"); print(status)
EOF
```

Expected: `var/eval/e11-baseline/before.wav` exists. (The script has no breath
markers, so the TTS cache entry stays valid after Task 1.)

- [ ] **Step 5: API key check (do not print it)**

```bash
grep -q '^ELEVENLABS_API_KEY=.\+' .env && echo "key present" || echo "KEY MISSING"
```

If missing, record it in the ledger; the user adds it before Task 19. Nothing
before Task 19 needs a real key.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin elevenlabs SDK 2.71.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase 1 — Repo-wide: no vocal sounds

### Task 1: Breath markers render as silence in every local engine

**Files:**
- Modify: `core/speech_engine.py`
- Modify: `core/script_gen/duration.py:49-55`
- Modify: `core/f5_tts/engine.py:746-751`
- Modify: `core/kokoro_tts/engine.py:321-326`
- Modify: `core/chatterbox_tts/engine.py:417-425` (add a `breath` branch)
- Delete: `core/breath_sounds.py`, `assets/breath_sounds/`, `scripts/generate_breath_samples.py`
- Test: `tests/unit/test_no_breath_sounds.py`

**Interfaces:**
- Produces: `core.speech_engine.BREATH_SILENCE_SEC: dict[str, float]`,
  `core.speech_engine.breath_silence_sec(subtype: str) -> float`
  (unknown subtype → 1.2). `core.script_gen.duration.BREATH_SEC` stays as an
  alias of `BREATH_SILENCE_SEC`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_no_breath_sounds.py`:

```python
"""Breath markers must render as silence — never a breath recording."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from core.speech_engine import BREATH_SILENCE_SEC, breath_silence_sec

ROOT = Path(__file__).resolve().parents[2]


def test_breath_silence_table():
    assert BREATH_SILENCE_SEC == {"breath": 1.2, "inhale": 1.5, "exhale": 1.8}
    assert breath_silence_sec("inhale") == 1.5
    assert breath_silence_sec("unknown") == 1.2


def test_duration_alias_matches():
    from core.script_gen.duration import BREATH_SEC
    assert BREATH_SEC == BREATH_SILENCE_SEC


def test_breath_sample_module_and_assets_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("core.breath_sounds")
    assert not (ROOT / "assets" / "breath_sounds").exists()


def test_no_engine_references_breath_samples():
    for rel in ("core/f5_tts/engine.py", "core/kokoro_tts/engine.py",
                "core/chatterbox_tts/engine.py"):
        src = (ROOT / rel).read_text()
        assert "load_breath" not in src, rel
        assert "breath_sounds" not in src, rel


def test_chatterbox_breath_segment_is_silence(monkeypatch):
    from core.chatterbox_tts import engine as cb

    eng = cb.ChatterboxEngine.__new__(cb.ChatterboxEngine)
    out = eng._breath_piece({"type": "breath", "subtype": "exhale"})
    audio, act = out
    assert len(audio) == int(round(1.8 * cb.SAMPLE_RATE))
    assert not audio.any() and not act.any()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_no_breath_sounds.py -v`
Expected: FAIL (`ImportError: cannot import name 'BREATH_SILENCE_SEC'`).

- [ ] **Step 3: Implement**

`core/speech_engine.py` — add below `SAMPLE_RATE = 24000`:

```python
# Breath markers ([breath]/[inhale]/[exhale]) render as SILENCE of these
# lengths in every engine. No model produces natural breath or sigh sounds,
# so no engine may ever emit one (user rule, 2026-10-12).
BREATH_SILENCE_SEC: dict[str, float] = {"breath": 1.2, "inhale": 1.5, "exhale": 1.8}


def breath_silence_sec(subtype: str) -> float:
    """Silent-beat length for a breath marker; unknown subtypes use 'breath'."""
    return BREATH_SILENCE_SEC.get(subtype, BREATH_SILENCE_SEC["breath"])
```

and inside `class SpeechEngine`, above `load_model`:

```python
    #: Native output rate. Local engines produce 24 kHz; an engine may override
    #: (ElevenLabs returns 48 kHz). The pipeline honours this attribute.
    sample_rate: int = SAMPLE_RATE
```

(Update the class docstring sentence "Every engine produces mono float32 audio
at 24 000 Hz" to "at `sample_rate` Hz (24 000 unless the engine overrides it)".)

`core/script_gen/duration.py` — replace the `BREATH_SEC` dict and its comment:

```python
# Silent-beat lengths for [breath]/[inhale]/[exhale]; every engine renders
# these markers as silence of exactly this length (no breath samples).
from core.speech_engine import BREATH_SILENCE_SEC as BREATH_SEC  # noqa: E402
```

`core/f5_tts/engine.py` — replace the `elif seg["type"] == "breath":` branch:

```python
            elif seg["type"] == "breath":
                # Breath markers are a silent beat (no breath samples), rendered
                # exactly like a pause of the marker's length.
                from core.kokoro_tts.postprocessor import generate_room_tone
                from core.speech_engine import breath_silence_sec
                beat = generate_room_tone(breath_silence_sec(seg["subtype"]), sr=SAMPLE_RATE)
                chunks.append((beat, np.zeros(len(beat), dtype=bool), "breath", None, None))
```

`core/kokoro_tts/engine.py` — replace the `elif segment["type"] == "breath":` branch:

```python
            elif segment["type"] == "breath":
                # Silent beat, rendered like a pause (no breath samples).
                from core.speech_engine import breath_silence_sec
                beat = generate_room_tone(breath_silence_sec(segment["subtype"]), sr=SAMPLE_RATE)
                audio_chunks.append(beat)
                activity_chunks.append(np.zeros(len(beat), dtype=bool))
```

`core/chatterbox_tts/engine.py` — add a method on `ChatterboxEngine`:

```python
    @staticmethod
    def _breath_piece(seg: dict) -> tuple[np.ndarray, np.ndarray]:
        """Silent beat for a breath marker (no breath samples)."""
        from core.speech_engine import breath_silence_sec
        n = int(round(breath_silence_sec(seg.get("subtype", "breath")) * SAMPLE_RATE))
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)
```

and in `synthesize`, directly after the `if seg_type == "pause":` block:

```python
            elif seg_type == "breath":
                beat, beat_act = self._breath_piece(seg)
                audio_pieces.append(beat)
                activity_pieces.append(beat_act)
```

Delete the sample machinery:

```bash
git rm -r core/breath_sounds.py assets/breath_sounds scripts/generate_breath_samples.py
grep -rn "breath_sounds\|load_breath\|generate_breath_samples" core scripts tests app.py
```

Expected grep output: nothing. (Doc references are cleaned in Task 17.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_no_breath_sounds.py tests/unit/test_duration_estimate.py tests/unit/test_chatterbox_engine.py tests/unit/test_f5_engine_chunking.py tests/unit/test_tts_engines.py -q`
Expected: all PASS. Then the full suite: `.venv/bin/python -m pytest tests/unit/ -q` → green.

- [ ] **Step 5: Commit**

```bash
git add -A core/speech_engine.py core/script_gen/duration.py core/f5_tts/engine.py core/kokoro_tts/engine.py core/chatterbox_tts/engine.py tests/unit/test_no_breath_sounds.py
git commit -m "feat!: render breath markers as silence in every engine; remove breath samples

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 1 gate (controller)

- [ ] Full suite green. Dispatch a phase reviewer on `git diff <Task0 commit>..HEAD`
  with focus: no engine can still emit a breath sample; local pause/room-tone
  behaviour unchanged; `SpeechEngine.sample_rate` default is 24000.

---

## Phase 2 — ElevenLabs core package (no pipeline wiring)

### Task 2: Voice registry

**Files:**
- Create: `assets/elevenlabs/voices.toml`
- Create: `core/elevenlabs_tts/__init__.py`, `core/elevenlabs_tts/voices.py`
- Test: `tests/unit/test_elevenlabs_voices.py`

**Interfaces:**
- Produces: `ElevenLabsVoice(id: str, name: str, notes: str = "")` with
  `.label -> str`; `load_registry(path: Path | None = None) -> VoiceRegistry`
  (`voices: tuple[ElevenLabsVoice, ...]`, `default_voice: str`,
  `family_map: dict[str, str]`); `list_voices(path=None) -> list[ElevenLabsVoice]`;
  `default_voice_id(path=None) -> str`; `voice_for_family(family: str, path=None) -> str`.

- [ ] **Step 1: Write the failing test**

```python
"""ElevenLabs voice registry (assets/elevenlabs/voices.toml)."""

import pytest

from core.elevenlabs_tts.voices import (
    default_voice_id, list_voices, load_registry, voice_for_family,
)


def _write(tmp_path, body):
    p = tmp_path / "voices.toml"
    p.write_text(body)
    return p


def test_shipped_registry_has_old_repo_voices_and_brittney_default():
    ids = {v.id for v in list_voices()}
    assert "pjcYQlDFKMbcOUp6F5GD" in ids and "pMsXgVXv3BLzUgSXRplE" in ids
    assert len(ids) == 9
    assert default_voice_id() == "pjcYQlDFKMbcOUp6F5GD"


def test_family_map_falls_back_to_default(tmp_path):
    p = _write(tmp_path, 'default_voice = "a"\n[[voice]]\nid = "a"\nname = "A"\n'
                         '[[voice]]\nid = "b"\nname = "B"\n[family_map]\nsleep = "b"\n')
    assert voice_for_family("sleep", path=p) == "b"
    assert voice_for_family("focus", path=p) == "a"


def test_unknown_family_voice_raises_with_known_list(tmp_path):
    p = _write(tmp_path, 'default_voice = "a"\n[[voice]]\nid = "a"\nname = "A"\n'
                         '[family_map]\nsleep = "zzz"\n')
    with pytest.raises(ValueError, match="zzz.*known"):
        load_registry(p)


def test_default_must_be_a_listed_voice(tmp_path):
    p = _write(tmp_path, 'default_voice = "nope"\n[[voice]]\nid = "a"\nname = "A"\n')
    with pytest.raises(ValueError, match="default_voice"):
        load_registry(p)


def test_label_includes_notes():
    v = next(v for v in list_voices() if v.name == "Serena")
    assert v.label.startswith("Serena — ")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_elevenlabs_voices.py -v`
Expected: FAIL (`ModuleNotFoundError: core.elevenlabs_tts`).

- [ ] **Step 3: Implement**

`assets/elevenlabs/voices.toml`:

```toml
# ElevenLabs voices offered in the UI. Seeded from moodscape-mix-lib's curated
# meditation list; the user will add reviewed Eleven v4 voices later.
# Community voices must be added to the account ("Add to my voices") first.
default_voice = "pjcYQlDFKMbcOUp6F5GD"   # Brittney (matches the local narrator)

[[voice]]
id = "pjcYQlDFKMbcOUp6F5GD"
name = "Brittney"
notes = "female · calm narrator"

[[voice]]
id = "pMsXgVXv3BLzUgSXRplE"
name = "Serena"
notes = "female · warm, meditative · stock"

[[voice]]
id = "Qggl4b0xRMiqOwhPtVWT"
name = "Clara"
notes = "female"

[[voice]]
id = "VhxAIIZM8IRmnl5fyeyk"
name = "Valory"
notes = "female"

[[voice]]
id = "BpjGufoPiobT79j2vtj4"
name = "Priyanka"
notes = "female"

[[voice]]
id = "WuBPEavIaQB56EnsGvFh"
name = "Eryn"
notes = "female · calm therapist"

[[voice]]
id = "zA6D7RyKdc2EClouEMkP"
name = "AImee"
notes = "female · soft, ASMR"

[[voice]]
id = "wgHvco1wiREKN0BdyVx5"
name = "Drew"
notes = "male · deep, soothing"

[[voice]]
id = "xGDJhCwcqw94ypljc95Z"
name = "Archer"
notes = "male · narration"

# Genre family -> voice id. Empty until the v4 voice review; unmapped
# families use default_voice.
[family_map]
```

`core/elevenlabs_tts/voices.py`:

```python
"""ElevenLabs voice registry loaded from assets/elevenlabs/voices.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VOICES_PATH = Path(__file__).resolve().parents[2] / "assets" / "elevenlabs" / "voices.toml"


@dataclass(frozen=True)
class ElevenLabsVoice:
    id: str
    name: str
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} — {self.notes}" if self.notes else self.name


@dataclass(frozen=True)
class VoiceRegistry:
    voices: tuple[ElevenLabsVoice, ...]
    default_voice: str
    family_map: dict[str, str] = field(default_factory=dict)


def load_registry(path: Path | None = None) -> VoiceRegistry:
    """Parse and validate the registry. Raises ValueError on any bad reference."""
    data = tomllib.loads(Path(path or VOICES_PATH).read_text(encoding="utf-8"))
    voices = tuple(
        ElevenLabsVoice(id=v["id"], name=v["name"], notes=v.get("notes", ""))
        for v in data.get("voice", [])
    )
    if not voices:
        raise ValueError("voices.toml lists no [[voice]] entries.")
    known = sorted(v.id for v in voices)
    default = data.get("default_voice", "")
    if default not in known:
        raise ValueError(f"default_voice {default!r} is not a listed voice; known: {known}")
    family_map = dict(data.get("family_map", {}))
    for family, voice_id in family_map.items():
        if voice_id not in known:
            raise ValueError(
                f"family_map maps {family!r} to unknown voice {voice_id!r}; known: {known}"
            )
    return VoiceRegistry(voices=voices, default_voice=default, family_map=family_map)


def list_voices(path: Path | None = None) -> list[ElevenLabsVoice]:
    return list(load_registry(path).voices)


def default_voice_id(path: Path | None = None) -> str:
    return load_registry(path).default_voice


def voice_for_family(family: str, path: Path | None = None) -> str:
    """Mapped voice for a genre family, else the default voice."""
    registry = load_registry(path)
    return registry.family_map.get(family, registry.default_voice)
```

`core/elevenlabs_tts/__init__.py` (exports grow in later tasks):

```python
"""ElevenLabs v4 cloud TTS engine."""

from core.elevenlabs_tts.voices import (
    ElevenLabsVoice,
    default_voice_id,
    list_voices,
    voice_for_family,
)

__all__ = ["ElevenLabsVoice", "default_voice_id", "list_voices", "voice_for_family"]
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/unit/test_elevenlabs_voices.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add assets/elevenlabs/voices.toml core/elevenlabs_tts/__init__.py core/elevenlabs_tts/voices.py tests/unit/test_elevenlabs_voices.py
git commit -m "feat(elevenlabs): voice registry seeded from moodscape-mix-lib

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Delivery direction + preprocessor

**Files:**
- Create: `core/elevenlabs_tts/direction.py`, `core/elevenlabs_tts/preprocessor.py`
- Modify: `core/content_profiles.py` (add `elevenlabs_paragraph_pause_sec`)
- Test: `tests/unit/test_elevenlabs_direction.py`, `tests/unit/test_elevenlabs_preprocessor.py`

**Interfaces:**
- Consumes: `core.f5_tts.preprocessor.parse_script(script, paragraph_pause_sec) -> list[dict]`,
  `core.text_utils.expand_text(text) -> str`, `core.speech_engine.breath_silence_sec`.
- Produces: `DEFAULT_DIRECTION = "[soft, slow, calm voice]"`;
  `sanitize_direction(text: str | None) -> str` (raises `ValueError`);
  `default_direction() -> str` (env `MOODSCAPE_ELEVENLABS_DIRECTION` or default);
  `HARD_MAX_CHARS = 9000`; `chunk_max_chars() -> int`;
  `prepare_segments(script: str, content_type: str = "meditation", direction: str | None = None) -> list[dict]`
  returning only `{"type": "speech", "text": str}` / `{"type": "pause", "duration_sec": float}`;
  `estimate_billed_chars(segments: list[dict]) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_elevenlabs_direction.py`:

```python
import pytest

from core.elevenlabs_tts.direction import DEFAULT_DIRECTION, default_direction, sanitize_direction


@pytest.mark.parametrize("raw,expected", [
    ("[soft, slow, calm voice]", "[soft, slow, calm voice]"),
    ("soft, slow, calm voice", "[soft, slow, calm voice]"),
    ("  [warm]  [unhurried] ", "[warm unhurried]"),
    ("", DEFAULT_DIRECTION),
    (None, DEFAULT_DIRECTION),
])
def test_normalizes(raw, expected):
    assert sanitize_direction(raw) == expected


@pytest.mark.parametrize("bad", ["[sighs]", "soft with a breath", "[exhales slowly]",
                                 "laughing softly", "[yawn]", "gentle humming"])
def test_rejects_vocal_sound_words(bad):
    with pytest.raises(ValueError, match="voice quality"):
        sanitize_direction(bad)


def test_allows_words_that_merely_contain_sound_substrings():
    assert sanitize_direction("[human, humble tone]") == "[human, humble tone]"


def test_too_long_rejected():
    with pytest.raises(ValueError, match="80"):
        sanitize_direction("calm " * 30)


def test_env_default(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "warm, low voice")
    assert default_direction() == "[warm, low voice]"
```

`tests/unit/test_elevenlabs_preprocessor.py`:

```python
import re

from core.elevenlabs_tts.direction import DEFAULT_DIRECTION
from core.elevenlabs_tts.preprocessor import (
    HARD_MAX_CHARS, estimate_billed_chars, prepare_segments,
)

SCRIPT = (
    "Find a comfortable position. Let your eyes close.\n\n"
    "Notice your breath. [pause:4s]\n\n"
    "[inhale] Breathe in slowly. [exhale] And let it go."
)


def _speech(segs):
    return [s["text"] for s in segs if s["type"] == "speech"]


def test_only_speech_and_pause_types():
    assert {s["type"] for s in prepare_segments(SCRIPT)} == {"speech", "pause"}


def test_direction_prefixes_every_chunk():
    for text in _speech(prepare_segments(SCRIPT)):
        assert text.startswith(DEFAULT_DIRECTION + " ")


def test_breath_markers_become_pauses_of_breath_length():
    segs = prepare_segments(SCRIPT)
    pauses = [s["duration_sec"] for s in segs if s["type"] == "pause"]
    assert 1.5 in pauses and 1.8 in pauses
    assert 4.0 in pauses
    assert 3.0 in pauses  # meditation paragraph pause


def test_sleep_story_paragraph_pause():
    segs = prepare_segments("One.\n\nTwo.", content_type="sleep_story")
    assert [s["duration_sec"] for s in segs if s["type"] == "pause"] == [2.0]


def test_no_sound_or_reaction_tags_reach_api_text():
    for text in _speech(prepare_segments(SCRIPT)):
        body = text[len(DEFAULT_DIRECTION):]
        assert "[" not in body
        assert not re.search(r"\b(breath|inhale|exhale|sigh)", text.split("]", 1)[0])


def test_oversize_paragraph_splits_under_hard_ceiling(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "1500")
    para = " ".join(f"Sentence number {i} drifts by softly." for i in range(400))
    texts = _speech(prepare_segments(para))
    assert len(texts) > 1
    assert all(len(t) <= 1500 for t in texts)
    assert all(len(t) <= HARD_MAX_CHARS for t in texts)


def test_never_splits_inside_brackets(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", "200")
    para = ("Rest here. " * 8) + "[keep this. Together] " + ("Rest here. " * 8)
    for text in _speech(prepare_segments(para)):
        body = text[len(DEFAULT_DIRECTION):]
        assert body.count("[") == body.count("]")


def test_numbers_expanded():
    assert "four" in _speech(prepare_segments("Count to 4."))[0].lower()


def test_estimate_billed_chars():
    segs = prepare_segments(SCRIPT)
    assert estimate_billed_chars(segs) == sum(len(t) for t in _speech(segs))


def test_custom_direction_applied():
    texts = _speech(prepare_segments("Hello there.", direction="warm, low voice"))
    assert texts == ["[warm, low voice] Hello there."]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/unit/test_elevenlabs_direction.py tests/unit/test_elevenlabs_preprocessor.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`core/content_profiles.py`: in `"meditation"` add
`"elevenlabs_paragraph_pause_sec": 3.0,` and in `"sleep_story"` add
`"elevenlabs_paragraph_pause_sec": 2.0,` (next to the `f5_` keys). Update the
module docstring line naming `kokoro_paragraph_pause_sec` /
`f5_paragraph_pause_sec` to include `elevenlabs_paragraph_pause_sec`.

`core/elevenlabs_tts/direction.py`:

```python
"""Delivery direction: the voice-quality instruction prefixed to every chunk.

Eleven v4 has no speed/style settings, so pacing and tone come from an inline
bracketed direction. It must describe VOICE QUALITY only — never a sound
(breaths, sighs, laughs): no model renders those naturally (user rule).
"""

from __future__ import annotations

import os
import re

DEFAULT_DIRECTION = "[soft, slow, calm voice]"
MAX_DIRECTION_CHARS = 80

_SOUND_WORDS = (
    "breath", "breaths", "breathe", "breathes", "breathing", "breathy",
    "inhale", "inhales", "inhaling", "exhale", "exhales", "exhaling",
    "sigh", "sighs", "sighing", "gasp", "gasps", "gasping",
    "laugh", "laughs", "laughing", "chuckle", "chuckles", "giggle", "giggles",
    "yawn", "yawns", "yawning", "hum", "hums", "humming",
    "cough", "coughs", "sniff", "sniffs", "gulp", "gulps", "swallow", "swallows",
    "whimper", "whimpers", "sob", "sobs", "sobbing", "cry", "cries", "crying",
)
_SOUND_RE = re.compile(r"\b(" + "|".join(_SOUND_WORDS) + r")\b", re.IGNORECASE)


def sanitize_direction(text: str | None) -> str:
    """Normalize to a single ``[...]`` group; reject sound words and overlong text."""
    inner = re.sub(r"[\[\]]", " ", text or "")
    inner = re.sub(r"\s+", " ", inner).strip(" ,")
    if not inner:
        return DEFAULT_DIRECTION
    match = _SOUND_RE.search(inner)
    if match:
        raise ValueError(
            "Delivery direction must describe voice quality, not sounds "
            f"(found {match.group(0)!r}). Example: {DEFAULT_DIRECTION}"
        )
    result = f"[{inner}]"
    if len(result) > MAX_DIRECTION_CHARS:
        raise ValueError(f"Delivery direction is {len(result)} chars; max is {MAX_DIRECTION_CHARS}.")
    return result


def default_direction() -> str:
    return sanitize_direction(os.environ.get("MOODSCAPE_ELEVENLABS_DIRECTION") or DEFAULT_DIRECTION)
```

`core/elevenlabs_tts/preprocessor.py`:

```python
"""ElevenLabs preprocessing: shared script grammar -> API-ready chunks + pauses.

Scripts stay engine-neutral ([pause:Xs], [breath]/[inhale]/[exhale], blank-line
paragraphs). Pauses and breath markers become programmatic silence; speech
between them becomes chunks, each prefixed with the delivery direction. No
other tag is ever added.
"""

from __future__ import annotations

import os
import re

from core.elevenlabs_tts.direction import default_direction, sanitize_direction
from core.f5_tts.preprocessor import parse_script
from core.speech_engine import breath_silence_sec
from core.text_utils import expand_text

HARD_MAX_CHARS = 9000  # < 10,000-char eleven_v4 request limit, incl. direction
DEFAULT_CHUNK_MAX_CHARS = 1500
_SENTENCE_END = re.compile(r"[.!?…]+[\"')]?\s+")


def chunk_max_chars() -> int:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS", str(DEFAULT_CHUNK_MAX_CHARS))
    return max(200, min(HARD_MAX_CHARS, int(raw)))


def _split_sentences(text: str) -> list[str]:
    """Split after sentence punctuation, never inside [...]."""
    parts: list[str] = []
    depth, start, i = 0, 0, 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        elif depth == 0:
            m = _SENTENCE_END.match(text, i)
            if m:
                parts.append(text[start:m.end()].strip())
                start = i = m.end()
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def _split_words(sentence: str, limit: int) -> list[str]:
    out, buf = [], ""
    for word in sentence.split():
        if buf and len(buf) + 1 + len(word) > limit:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out


def _pack(sentences: list[str], limit: int) -> list[str]:
    chunks, cur = [], ""
    for sentence in sentences:
        pieces = _split_words(sentence, limit) if len(sentence) > limit else [sentence]
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def prepare_segments(script: str, content_type: str = "meditation",
                     direction: str | None = None) -> list[dict]:
    from core.content_profiles import get_profile

    tag = sanitize_direction(direction) if direction is not None else default_direction()
    limit = chunk_max_chars() - len(tag) - 1
    raw = parse_script(
        script, paragraph_pause_sec=get_profile(content_type)["elevenlabs_paragraph_pause_sec"]
    )

    out: list[dict] = []
    speech_buf: list[str] = []

    def flush() -> None:
        if not speech_buf:
            return
        text = " ".join(speech_buf)
        speech_buf.clear()
        for piece in _pack(_split_sentences(text), limit):
            out.append({"type": "speech", "text": f"{tag} {piece}"})

    for seg in raw:
        kind = seg["type"]
        if kind == "speech":
            speech_buf.append(expand_text(seg["text"]).strip())
        elif kind == "pause":
            flush()
            out.append({"type": "pause", "duration_sec": float(seg["duration_sec"])})
        elif kind == "breath":
            flush()
            out.append({"type": "pause", "duration_sec": breath_silence_sec(seg["subtype"])})
        # "voice" segments: one ElevenLabs voice per render — ignored.
    flush()
    return out


def estimate_billed_chars(segments: list[dict]) -> int:
    """Characters sent to the API (what ElevenLabs bills) for these segments."""
    return sum(len(s["text"]) for s in segments if s["type"] == "speech")
```

Add to `core/elevenlabs_tts/__init__.py` exports: `DEFAULT_DIRECTION`,
`sanitize_direction`, `default_direction`, `prepare_segments`,
`estimate_billed_chars`.

- [ ] **Step 4: Run tests** — both new files PASS; `tests/unit/test_content_profiles.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add core/elevenlabs_tts core/content_profiles.py tests/unit/test_elevenlabs_direction.py tests/unit/test_elevenlabs_preprocessor.py
git commit -m "feat(elevenlabs): delivery-direction sanitizer and preprocessor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: API client and per-chunk cache

**Files:**
- Create: `core/elevenlabs_tts/client.py`, `core/elevenlabs_tts/chunk_cache.py`
- Test: `tests/unit/test_elevenlabs_client.py`, `tests/unit/test_elevenlabs_chunk_cache.py`

**Interfaces:**
- Produces (client): `DEFAULT_MODEL = "eleven_v4"`, `OUTPUT_FORMAT = "pcm_48000"`,
  `get_api_key() -> str`, `has_api_key() -> bool`, `model_id() -> str`,
  `max_retries() -> int`, `class ElevenLabsConfigError(RuntimeError)`,
  `class ElevenLabsAPIError(RuntimeError)` with `.status: int | None`,
  `with_retries(fn, *, attempts: int, sleep=time.sleep)`,
  `class ElevenLabsClient(api_key: str | None = None, sdk=None, sleep=time.sleep)` with
  `.convert(text, voice_id, *, model_id, stability, similarity, seed, previous_request_ids=None) -> tuple[bytes, str | None]`
  and `.voice_exists(voice_id) -> bool`.
- Produces (cache): `cache_dir() -> Path`,
  `chunk_payload(text, voice_id, model_id, stability, similarity, seed) -> dict`,
  `load(payload) -> tuple[bytes, str | None] | None`,
  `store(payload, pcm: bytes, request_id: str | None) -> None`,
  `is_cached(payload) -> bool`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_elevenlabs_chunk_cache.py`:

```python
import pytest

from core.elevenlabs_tts import chunk_cache


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))


def _p(**kw):
    base = dict(text="[soft] Hi.", voice_id="v", model_id="eleven_v4",
                stability=0.5, similarity=0.75, seed=42)
    base.update(kw)
    return chunk_cache.chunk_payload(**base)


def test_round_trip_with_request_id():
    chunk_cache.store(_p(), b"\x01\x02", "req-1")
    assert chunk_cache.load(_p()) == (b"\x01\x02", "req-1")
    assert chunk_cache.is_cached(_p())


@pytest.mark.parametrize("change", [dict(text="x"), dict(voice_id="w"),
                                    dict(stability=0.6), dict(similarity=0.7),
                                    dict(seed=1), dict(model_id="eleven_v3")])
def test_any_setting_change_misses(change):
    chunk_cache.store(_p(), b"\x01\x02", "req-1")
    assert chunk_cache.load(_p(**change)) is None


def test_payload_has_no_request_ids():
    assert "previous_request_ids" not in _p()
    assert _p()["output_format"] == "pcm_48000"
```

`tests/unit/test_elevenlabs_client.py`:

```python
import pytest

from core.elevenlabs_tts.client import (
    ElevenLabsAPIError, ElevenLabsClient, ElevenLabsConfigError, with_retries,
)


class _Err(Exception):
    def __init__(self, status):
        super().__init__(f"status {status}")
        self.status_code = status


class _Resp:
    def __init__(self, data, headers):
        self.data, self.headers = data, headers
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class _FakeSDK:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        sdk = self
        class _Raw:
            def convert(self, **kw):
                sdk.calls.append(kw)
                out = sdk.outcomes.pop(0)
                if isinstance(out, Exception):
                    raise out
                return _Resp([out], {"request-id": f"rid{len(sdk.calls)}"})
        class _TTS:
            with_raw_response = _Raw()
        class _Voices:
            def get(self, voice_id):
                if voice_id == "missing":
                    raise _Err(404)
                return object()
        self.text_to_speech = _TTS()
        self.voices = _Voices()


def _client(outcomes):
    return ElevenLabsClient(api_key="k", sdk=_FakeSDK(outcomes), sleep=lambda s: None)


def _convert(c, **kw):
    return c.convert("[soft] Hi.", "v", model_id="eleven_v4", stability=0.5,
                     similarity=0.75, seed=42, **kw)


def test_missing_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ElevenLabsConfigError, match="ELEVENLABS_API_KEY"):
        ElevenLabsClient(sdk=object())


def test_returns_pcm_and_request_id_and_sends_v4_params():
    c = _client([b"\x00\x01"])
    assert _convert(c, previous_request_ids=["a", "b", "c", "d"]) == (b"\x00\x01", "rid1")
    kw = c._sdk.calls[0]
    assert kw["output_format"] == "pcm_48000" and kw["model_id"] == "eleven_v4"
    assert kw["previous_request_ids"] == ["b", "c", "d"]
    vs = kw["voice_settings"]
    assert vs.stability == 0.5 and vs.similarity_boost == 0.75
    sent = {k for k, v in vars(vs).items() if v is not None}
    assert "speed" not in sent and "style" not in sent


@pytest.mark.parametrize("status", [None, 429, 500, 503])
def test_transient_errors_retry(status):
    c = _client([_Err(status), b"\x00\x01"])
    assert _convert(c)[0] == b"\x00\x01"
    assert len(c._sdk.calls) == 2


@pytest.mark.parametrize("status,needle", [(401, "API key"), (402, "credits"), (422, "422")])
def test_client_errors_fail_fast(status, needle):
    c = _client([_Err(status), b"never"])
    with pytest.raises(ElevenLabsAPIError, match=needle) as ei:
        _convert(c)
    assert ei.value.status == status
    assert len(c._sdk.calls) == 1


def test_retry_budget_exhausts():
    calls = []
    def boom():
        calls.append(1)
        raise _Err(503)
    with pytest.raises(ElevenLabsAPIError):
        with_retries(boom, attempts=3, sleep=lambda s: None)
    assert len(calls) == 3


def test_voice_exists():
    c = _client([])
    assert c.voice_exists("ok") is True
    assert c.voice_exists("missing") is False
```

- [ ] **Step 2: Run to verify failure** — both files FAIL with module not found.

- [ ] **Step 3: Implement**

`core/elevenlabs_tts/chunk_cache.py`:

```python
"""Content-addressed PCM cache: never pay twice for the same chunk.

Key = SHA-256 of the payload (text, voice, model, settings, seed, format).
Request ids are stored beside the PCM but are NOT part of the key — they are
render-time continuity hints, not inputs that change the audio contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from core.elevenlabs_tts.client import OUTPUT_FORMAT

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "var" / "elevenlabs_cache"


def cache_dir() -> Path:
    d = Path(os.environ.get("MOODSCAPE_ELEVENLABS_CACHE_DIR", DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunk_payload(text: str, voice_id: str, model_id: str, stability: float,
                  similarity: float, seed: int) -> dict:
    return {
        "text": text, "voice_id": voice_id, "model_id": model_id,
        "stability": round(float(stability), 3), "similarity": round(float(similarity), 3),
        "seed": int(seed), "output_format": OUTPUT_FORMAT,
    }


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def is_cached(payload: dict) -> bool:
    return (cache_dir() / f"{_key(payload)}.pcm").is_file()


def load(payload: dict) -> tuple[bytes, str | None] | None:
    base = cache_dir() / _key(payload)
    pcm_path = base.with_suffix(".pcm")
    if not pcm_path.is_file():
        return None
    request_id = None
    meta_path = base.with_suffix(".json")
    if meta_path.is_file():
        try:
            request_id = json.loads(meta_path.read_text(encoding="utf-8")).get("request_id")
        except (OSError, ValueError):
            request_id = None
    return pcm_path.read_bytes(), request_id


def store(payload: dict, pcm: bytes, request_id: str | None) -> None:
    base = cache_dir() / _key(payload)
    tmp = base.with_suffix(".pcm.part")
    tmp.write_bytes(pcm)
    tmp.rename(base.with_suffix(".pcm"))
    meta = {**payload, "request_id": request_id, "created_at": time.time()}
    base.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
```

`core/elevenlabs_tts/client.py`:

```python
"""Thin wrapper over the elevenlabs SDK — the ONLY module that imports it.

Retries transient failures (network/timeout, 429, 5xx) with exponential
backoff; other 4xx fail immediately with a readable message.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("moodscape.elevenlabs")

DEFAULT_MODEL = "eleven_v4"
OUTPUT_FORMAT = "pcm_48000"
_STATUS_HINTS = {
    401: "invalid ElevenLabs API key (check ELEVENLABS_API_KEY)",
    402: "ElevenLabs account is out of credits",
    404: "voice or model not found in this ElevenLabs account",
}


class ElevenLabsConfigError(RuntimeError):
    """Missing or invalid local configuration (no network call was made)."""


class ElevenLabsAPIError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def get_api_key() -> str:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()


def has_api_key() -> bool:
    return bool(get_api_key())


def model_id() -> str:
    return os.environ.get("MOODSCAPE_ELEVENLABS_MODEL", DEFAULT_MODEL)


def max_retries() -> int:
    return max(1, int(os.environ.get("MOODSCAPE_ELEVENLABS_MAX_RETRIES", "3")))


def _status_of(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _is_transient(status: int | None) -> bool:
    return status is None or status == 429 or status >= 500


def _describe(status: int | None, exc: BaseException) -> str:
    hint = _STATUS_HINTS.get(status)
    prefix = f"ElevenLabs error {status}" if status else "ElevenLabs request failed"
    return f"{prefix}: {hint + ' — ' if hint else ''}{exc}"


def with_retries(fn, *, attempts: int, sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ElevenLabsAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 — classified below
            status = _status_of(exc)
            if not _is_transient(status) or attempt >= attempts:
                raise ElevenLabsAPIError(status, _describe(status, exc)) from exc
            delay = 2 ** (attempt - 1)
            logger.warning("ElevenLabs transient failure (status=%s, attempt %d/%d); retrying in %ss",
                           status, attempt, attempts, delay)
            sleep(delay)


def _request_id(headers) -> str | None:
    for key in ("request-id", "x-request-id", "x-trace-id"):
        value = headers.get(key)
        if value:
            return value
    return None


class ElevenLabsClient:
    def __init__(self, api_key: str | None = None, sdk=None, sleep=time.sleep):
        key = api_key or get_api_key()
        if not key:
            raise ElevenLabsConfigError(
                "ELEVENLABS_API_KEY is not set. Add it to .env to use the ElevenLabs provider."
            )
        if sdk is None:
            from elevenlabs.client import ElevenLabs
            sdk = ElevenLabs(api_key=key)
        self._sdk = sdk
        self._sleep = sleep
        self._attempts = max_retries()

    def convert(self, text: str, voice_id: str, *, model_id: str, stability: float,
                similarity: float, seed: int,
                previous_request_ids: list[str] | None = None) -> tuple[bytes, str | None]:
        from elevenlabs.types import VoiceSettings

        kwargs = dict(
            text=text, voice_id=voice_id, model_id=model_id, output_format=OUTPUT_FORMAT,
            voice_settings=VoiceSettings(stability=stability, similarity_boost=similarity,
                                         use_speaker_boost=True),
            seed=int(seed), apply_text_normalization="auto",
        )
        if previous_request_ids:
            kwargs["previous_request_ids"] = list(previous_request_ids)[-3:]

        def call() -> tuple[bytes, str | None]:
            with self._sdk.text_to_speech.with_raw_response.convert(**kwargs) as resp:
                rid = _request_id(resp.headers)
                data = b"".join(chunk for chunk in resp.data if chunk)
            if not data:
                raise RuntimeError("ElevenLabs returned no audio data.")
            return data, rid

        return with_retries(call, attempts=self._attempts, sleep=self._sleep)

    def voice_exists(self, voice_id: str) -> bool:
        try:
            self._sdk.voices.get(voice_id)
            return True
        except Exception as exc:  # noqa: BLE001
            if _status_of(exc) in (400, 404):
                return False
            raise ElevenLabsAPIError(_status_of(exc), _describe(_status_of(exc), exc)) from exc
```

(Note: `chunk_cache` imports `OUTPUT_FORMAT` from `client`; `client` imports
nothing from the package, so there is no cycle.)

Add `ElevenLabsClient`, `ElevenLabsAPIError`, `ElevenLabsConfigError`,
`has_api_key` to `__init__.py` exports.

- [ ] **Step 4: Run tests** — both files PASS.

- [ ] **Step 5: Commit**

```bash
git add core/elevenlabs_tts tests/unit/test_elevenlabs_client.py tests/unit/test_elevenlabs_chunk_cache.py
git commit -m "feat(elevenlabs): SDK client with retries and per-chunk PCM cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Assembly DSP + `ElevenLabsEngine`

**Files:**
- Create: `core/elevenlabs_tts/assembly.py`, `core/elevenlabs_tts/engine.py`
- Test: `tests/unit/test_elevenlabs_assembly.py`, `tests/unit/test_elevenlabs_engine.py`

**Interfaces:**
- Consumes: Task 4 `ElevenLabsClient.convert`, `chunk_cache.*`, `client.model_id`;
  Task 2 `default_voice_id`, `list_voices`.
- Produces (assembly): `pcm16_to_float(pcm: bytes) -> np.ndarray`,
  `chunk_lufs_target() -> float | None`,
  `normalize_chunk_lufs(y, sr, target_lufs, gate_lufs=-50.0) -> np.ndarray`,
  `activity_mask(y, sr, frame_ms=20.0, rel_db=-35.0, hangover_ms=150.0) -> np.ndarray[bool]`,
  `crossfade_concat(pieces, masks, sr, xfade_ms) -> tuple[np.ndarray, np.ndarray]`.
- Produces (engine): `ElevenLabsParams(voice_id, stability, similarity, model_id, seed)` dataclass with
  `.cache_extra(direction: str) -> dict`;
  `resolve_params(voice_id=None, stability=None, similarity=None, seed=None) -> ElevenLabsParams`;
  `class ElevenLabsEngine(SpeechEngine)`: `sample_rate = 48000`,
  `__init__(params: ElevenLabsParams | None = None, client=None)`,
  `.params`, `.last_manifest: dict`, `.last_qa = None`,
  `synthesize(segments, voice=None, speed=None, progress_cb=None, **kwargs) -> (audio_48k float32, mask bool)`.
  `last_manifest` keys: `model_id, voice_id, stability, similarity, seed, chars_billed, chars_cached, cache_hits, cache_misses, request_ids`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_elevenlabs_assembly.py`:

```python
import numpy as np

from core.elevenlabs_tts.assembly import (
    activity_mask, chunk_lufs_target, crossfade_concat, normalize_chunk_lufs, pcm16_to_float,
)

SR = 48000


def _tone(sec, amp=0.3, f=220.0):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_pcm16_to_float():
    pcm = np.array([0, 16384, -32768], dtype=np.int16).tobytes()
    assert np.allclose(pcm16_to_float(pcm), [0.0, 0.5, -1.0])


def test_crossfade_lengths_and_mask_alignment():
    a, b = _tone(1.0), np.zeros(SR, dtype=np.float32)
    ma, mb = np.ones(SR, bool), np.zeros(SR, bool)
    out, m = crossfade_concat([a, b, a], [ma, mb, ma], SR, 30.0)
    xn = int(SR * 0.03)
    assert len(out) == 3 * SR - 2 * xn
    assert len(m) == len(out) and out.dtype == np.float32


def test_mask_marks_inner_gap_inactive():
    y = np.concatenate([_tone(1.0), np.zeros(SR, np.float32), _tone(1.0)])
    m = activity_mask(y, SR)
    assert m[: int(0.9 * SR)].mean() > 0.95
    assert m[int(1.3 * SR): int(1.9 * SR)].mean() < 0.05


def test_mask_all_false_for_silent_chunk():
    assert not activity_mask(np.zeros(SR, np.float32), SR).any()


def test_chunk_lufs_normalizes_and_passes_short_or_silent():
    y = _tone(2.0, amp=0.05)
    out = normalize_chunk_lufs(y, SR, -21.0)
    assert np.max(np.abs(out)) > np.max(np.abs(y))
    short = _tone(0.2)
    assert normalize_chunk_lufs(short, SR, -21.0) is short
    quiet = _tone(2.0, amp=1e-5)
    assert normalize_chunk_lufs(quiet, SR, -21.0) is quiet


def test_chunk_lufs_target_env(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "off")
    assert chunk_lufs_target() is None
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "-20")
    assert chunk_lufs_target() == -20.0
```

`tests/unit/test_elevenlabs_engine.py`:

```python
import numpy as np
import pytest

from core.elevenlabs_tts.client import ElevenLabsAPIError
from core.elevenlabs_tts.engine import ElevenLabsEngine, resolve_params

SR = 48000


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))


def _pcm(sec=0.5):
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()


class FakeClient:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at
    def convert(self, text, voice_id, **kw):
        self.calls.append(dict(text=text, voice_id=voice_id, **kw))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise ElevenLabsAPIError(402, "ElevenLabs error 402: out of credits")
        return _pcm(), f"rid{len(self.calls)}"


def _segs(n):
    segs = []
    for i in range(n):
        segs.append({"type": "speech", "text": f"[soft, slow, calm voice] Line {i}."})
        segs.append({"type": "pause", "duration_sec": 1.0})
    return segs


def _engine(client):
    return ElevenLabsEngine(params=resolve_params(voice_id="v"), client=client)


def test_output_contract():
    audio, mask = _engine(FakeClient()).synthesize(_segs(2))
    assert audio.dtype == np.float32 and mask.dtype == bool
    assert len(audio) == len(mask)
    assert ElevenLabsEngine.sample_rate == 48000


def test_pauses_inactive_speech_active():
    audio, mask = _engine(FakeClient()).synthesize(_segs(1))
    assert mask[: int(0.4 * SR)].mean() > 0.9
    assert mask[int(0.75 * SR): int(1.3 * SR)].mean() < 0.05


def test_stitching_passes_at_most_three_ids_including_cache_hits():
    _engine(FakeClient()).synthesize(_segs(2))          # warm cache for lines 0-1
    client = FakeClient()
    _engine(client).synthesize(_segs(5))                 # lines 0-1 hit, 2-4 miss
    assert len(client.calls) == 3
    assert client.calls[0]["previous_request_ids"] == ["rid1", "rid2"]
    assert len(client.calls[-1]["previous_request_ids"]) == 3


def test_stitch_off(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_STITCH", "0")
    client = FakeClient()
    _engine(client).synthesize(_segs(2))
    assert all(c["previous_request_ids"] is None for c in client.calls)


def test_manifest_counts_billing():
    eng = _engine(FakeClient())
    eng.synthesize(_segs(2))
    m = eng.last_manifest
    assert m["cache_misses"] == 2 and m["cache_hits"] == 0
    assert m["chars_billed"] == sum(len(f"[soft, slow, calm voice] Line {i}.") for i in range(2))
    assert m["model_id"] == "eleven_v4" and m["voice_id"] == "v"
    eng2 = _engine(FakeClient())
    eng2.synthesize(_segs(2))
    assert eng2.last_manifest["chars_billed"] == 0 and eng2.last_manifest["cache_hits"] == 2


def test_midrun_failure_reports_progress_and_resume_is_cheap():
    with pytest.raises(ElevenLabsAPIError, match="4 of 9 chunks rendered") as ei:
        _engine(FakeClient(fail_at=5)).synthesize(_segs(9))
    assert ei.value.status == 402
    client = FakeClient()
    _engine(client).synthesize(_segs(9))
    assert len(client.calls) == 5


def test_speed_ignored_and_v4_settings_sent():
    client = FakeClient()
    _engine(client).synthesize(_segs(1), speed=0.7)
    call = client.calls[0]
    assert call["model_id"] == "eleven_v4" and call["seed"] == 42
    assert call["stability"] == 0.5 and call["similarity"] == 0.75
    assert "speed" not in call


def test_cache_extra_contains_all_render_params():
    p = resolve_params(voice_id="v", stability=0.4)
    extra = p.cache_extra("[warm]")
    assert extra == {"voice_id": "v", "stability": 0.4, "similarity": 0.75,
                     "model_id": "eleven_v4", "seed": 42, "direction": "[warm]"}
```

- [ ] **Step 2: Run to verify failure** — FAIL (module not found).

- [ ] **Step 3: Implement**

`core/elevenlabs_tts/assembly.py`:

```python
"""Pure DSP for assembling ElevenLabs chunks into one 48 kHz voice track."""

from __future__ import annotations

import os
import warnings

import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def chunk_lufs_target() -> float | None:
    raw = os.environ.get("MOODSCAPE_ELEVENLABS_CHUNK_LUFS", "-21").strip().lower()
    return None if raw == "off" else float(raw)


def normalize_chunk_lufs(y: np.ndarray, sr: int, target_lufs: float,
                         gate_lufs: float = -50.0) -> np.ndarray:
    """Match each chunk's integrated loudness; passthrough (same object) when
    the chunk is < 400 ms or effectively silent, so near-silence is never boosted."""
    if y.size < int(sr * 0.4):
        return y
    import pyloudnorm as pyln

    measured = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float32)))
    if not np.isfinite(measured) or measured < gate_lufs:
        return y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = pyln.normalize.loudness(y.astype(np.float32), measured, target_lufs)
    return out.astype(np.float32, copy=False)


def activity_mask(y: np.ndarray, sr: int, frame_ms: float = 20.0,
                  rel_db: float = -35.0, hangover_ms: float = 150.0) -> np.ndarray:
    """True where speech is active: frame RMS within rel_db of the chunk's loudest
    frame, extended forward by hangover_ms. Gaps inside a chunk read inactive so
    the music bed can rise there."""
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=bool)
    hop = max(1, int(sr * frame_ms / 1000.0))
    frames = -(-n // hop)
    padded = np.zeros(frames * hop, dtype=np.float32)
    padded[:n] = y
    rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    peak = float(db.max())
    if peak < -60.0:
        return np.zeros(n, dtype=bool)
    active = db > peak + rel_db
    hang = int(np.ceil(hangover_ms / frame_ms))
    if hang > 0:
        active = np.convolve(active.astype(np.int32), np.ones(hang + 1, np.int32))[:frames] > 0
    return np.repeat(active, hop)[:n]


def crossfade_concat(pieces: list[np.ndarray], masks: list[np.ndarray], sr: int,
                     xfade_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Join pieces with equal-power crossfades; masks overlap with OR so
    len(mask) == len(audio) always."""
    xn = max(1, int(sr * xfade_ms / 1000.0))
    out = pieces[0].astype(np.float32, copy=True)
    mask = masks[0].astype(bool, copy=True)
    for y, m in zip(pieces[1:], masks[1:]):
        y = y.astype(np.float32, copy=False)
        n = min(xn, len(out), len(y))
        if n <= 1:
            out = np.concatenate([out, y])
            mask = np.concatenate([mask, m])
            continue
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        cross = out[-n:] * np.cos(t * np.pi / 2) + y[:n] * np.sin(t * np.pi / 2)
        out = np.concatenate([out[:-n], cross, y[n:]])
        mask = np.concatenate([mask[:-n], mask[-n:] | m[:n], m[n:]])
    return out.astype(np.float32, copy=False), mask
```

`core/elevenlabs_tts/engine.py`:

```python
"""ElevenLabsEngine — Eleven v4 cloud TTS behind the SpeechEngine contract.

Returns 48 kHz audio natively (sample_rate = 48000); the pipeline skips its
upsample for it. No model weights: load/unload only manage the API client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from core.elevenlabs_tts import chunk_cache
from core.elevenlabs_tts.assembly import (
    activity_mask, chunk_lufs_target, crossfade_concat, normalize_chunk_lufs, pcm16_to_float,
)
from core.elevenlabs_tts.client import ElevenLabsAPIError, ElevenLabsClient, model_id
from core.elevenlabs_tts.voices import default_voice_id, list_voices
from core.speech_engine import SpeechEngine

logger = logging.getLogger("moodscape.elevenlabs")
SR = 48000


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class ElevenLabsParams:
    voice_id: str
    stability: float
    similarity: float
    model_id: str
    seed: int

    def cache_extra(self, direction: str) -> dict:
        """Everything that changes ElevenLabs audio, for the whole-render TTS cache key."""
        return {"voice_id": self.voice_id, "stability": self.stability,
                "similarity": self.similarity, "model_id": self.model_id,
                "seed": self.seed, "direction": direction}


def resolve_params(voice_id: str | None = None, stability: float | None = None,
                   similarity: float | None = None, seed: int | None = None) -> ElevenLabsParams:
    return ElevenLabsParams(
        voice_id=voice_id or default_voice_id(),
        stability=round(float(stability if stability is not None
                              else _env_float("MOODSCAPE_ELEVENLABS_STABILITY", 0.5)), 3),
        similarity=round(float(similarity if similarity is not None
                               else _env_float("MOODSCAPE_ELEVENLABS_SIMILARITY", 0.75)), 3),
        model_id=model_id(),
        seed=int(seed if seed is not None else os.environ.get("MOODSCAPE_ELEVENLABS_SEED", "42")),
    )


class ElevenLabsEngine(SpeechEngine):
    sample_rate = SR

    def __init__(self, params: ElevenLabsParams | None = None, client=None):
        self.params = params or resolve_params()
        self._client = client
        self._injected = client is not None
        self.last_manifest: dict = {}
        self.last_qa = None

    def load_model(self) -> None:
        if self._client is None:
            self._client = ElevenLabsClient()

    def unload_model(self) -> None:
        if not self._injected:
            self._client = None

    def get_available_voices(self) -> list[dict]:
        return [{"id": v.id, "name": v.name, "description": v.notes} for v in list_voices()]

    def synthesize(self, segments, voice=None, speed=None, progress_cb=None, **kwargs):
        """Render segments; `speed` is ignored (Eleven v4 has no speed setting)."""
        if self._client is None:
            self.load_model()
        p = self.params
        voice_id = voice or p.voice_id
        stitch = os.environ.get("MOODSCAPE_ELEVENLABS_STITCH", "1") != "0"
        target = chunk_lufs_target()
        xfade_ms = _env_float("MOODSCAPE_ELEVENLABS_XFADE_MS", 30.0)

        total_speech = sum(1 for s in segments if s["type"] == "speech")
        pieces: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        recent: list[str] = []
        manifest = {"model_id": p.model_id, "voice_id": voice_id, "stability": p.stability,
                    "similarity": p.similarity, "seed": p.seed, "chars_billed": 0,
                    "chars_cached": 0, "cache_hits": 0, "cache_misses": 0, "request_ids": []}
        done = 0
        for idx, seg in enumerate(segments):
            if seg["type"] == "pause":
                n = int(round(float(seg["duration_sec"]) * SR))
                pieces.append(np.zeros(n, dtype=np.float32))
                masks.append(np.zeros(n, dtype=bool))
            elif seg["type"] == "speech":
                text = seg["text"]
                payload = chunk_cache.chunk_payload(text, voice_id, p.model_id,
                                                    p.stability, p.similarity, p.seed)
                hit = chunk_cache.load(payload)
                if hit is not None:
                    pcm, rid = hit
                    manifest["cache_hits"] += 1
                    manifest["chars_cached"] += len(text)
                else:
                    try:
                        pcm, rid = self._client.convert(
                            text, voice_id, model_id=p.model_id, stability=p.stability,
                            similarity=p.similarity, seed=p.seed,
                            previous_request_ids=list(recent) if (stitch and recent) else None,
                        )
                    except Exception as exc:
                        status = getattr(exc, "status", None)
                        raise ElevenLabsAPIError(
                            status,
                            f"{exc} — {done} of {total_speech} chunks rendered before failure "
                            "(cached; re-run resumes).",
                        ) from exc
                    chunk_cache.store(payload, pcm, rid)
                    manifest["cache_misses"] += 1
                    manifest["chars_billed"] += len(text)
                if rid:
                    recent = (recent + [rid])[-3:]
                    manifest["request_ids"].append(rid)
                y = pcm16_to_float(pcm)
                if target is not None:
                    y = normalize_chunk_lufs(y, SR, target)
                pieces.append(y)
                masks.append(activity_mask(y, SR))
                done += 1
            if progress_cb is not None:
                progress_cb(idx + 1, len(segments))

        self.last_manifest = manifest
        if not pieces:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
        logger.info("ElevenLabs: %d chunks (%d cached), %d chars billed",
                    done, manifest["cache_hits"], manifest["chars_billed"])
        return crossfade_concat(pieces, masks, SR, xfade_ms)
```

Note for `test_stitching_passes_at_most_three_ids_including_cache_hits`: the
first warm run stores rid1/rid2 on lines 0–1; the second run's first miss
(line 2) receives `["rid1", "rid2"]` from the cache hits. Note the fake
client's `convert(**kw)` records `previous_request_ids=None` when unstitched.

Add `ElevenLabsEngine`, `ElevenLabsParams`, `resolve_params` to `__init__.py`.

- [ ] **Step 4: Run tests** — both files PASS.

- [ ] **Step 5: Commit**

```bash
git add core/elevenlabs_tts tests/unit/test_elevenlabs_assembly.py tests/unit/test_elevenlabs_engine.py
git commit -m "feat(elevenlabs): 48 kHz engine with stitching, chunk LUFS, crossfade and activity mask

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: ElevenLabs voice post-processing

**Files:**
- Create: `core/elevenlabs_tts/postprocessor.py`
- Modify: `core/f5_tts/postprocessor.py:363-396` (extract `parallel_reverb_mix`, behaviour identical)
- Test: `tests/unit/test_elevenlabs_postprocessor.py`

**Interfaces:**
- Produces: `core.f5_tts.postprocessor.parallel_reverb_mix(reverb_amount: float, ir_name: str) -> pedalboard.Mix`;
  `deess_elevenlabs(audio: np.ndarray, sr: int) -> np.ndarray`;
  `build_elevenlabs_voice_chain(reverb_amount: float = 0.0, ir_name: str = "warm_studio") -> Pedalboard`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from pedalboard import Convolution, Gain, HighpassFilter, Limiter, LowpassFilter, Mix, Pedalboard

from core.elevenlabs_tts.postprocessor import build_elevenlabs_voice_chain, deess_elevenlabs
from core.f5_tts.postprocessor import build_f5_voice_chain

SR = 48000


def _noise(sec=1.0):
    return (np.random.default_rng(0).standard_normal(int(SR * sec)) * 0.1).astype(np.float32)


def _old_f5_chain(reverb_amount=0.15, ir_name="warm_studio"):
    from core.audio_processor import DEFAULT_IR, IR_CATALOG
    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]
    dry_db = 20 * np.log10(max(1.0 - reverb_amount, 1e-5))
    wet_db = 20 * np.log10(max(reverb_amount, 1e-5))
    return Pedalboard([
        Mix([Gain(gain_db=dry_db), Pedalboard([
            Convolution(impulse_response_filename=ir_path, mix=1.0),
            HighpassFilter(cutoff_frequency_hz=300), LowpassFilter(cutoff_frequency_hz=6000),
            Gain(gain_db=wet_db)])]),
        Limiter(threshold_db=-1.0),
    ])


def test_f5_chain_refactor_is_bit_identical():
    x = _noise().reshape(1, -1)
    assert np.array_equal(build_f5_voice_chain()(x, SR), _old_f5_chain()(x, SR))


def test_dry_chain_is_hpf_only():
    chain = build_elevenlabs_voice_chain(0.0)
    assert len(chain) == 1 and isinstance(chain[0], HighpassFilter)


def test_reverb_adds_parallel_mix_and_no_limiter():
    chain = build_elevenlabs_voice_chain(0.2)
    assert any(isinstance(p, Mix) for p in chain)
    assert not any(isinstance(p, Limiter) for p in chain)


def test_deess_toggle(monkeypatch):
    x = _noise()
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DEESS", "0")
    assert deess_elevenlabs(x, SR) is x
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DEESS", "1")
    y = deess_elevenlabs(x, SR)
    assert y.shape == x.shape and not np.array_equal(x, y)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module not found).

- [ ] **Step 3: Implement**

In `core/f5_tts/postprocessor.py`, add above `build_f5_voice_chain` and make
`build_f5_voice_chain` use it (its body becomes the two lines shown):

```python
def parallel_reverb_mix(reverb_amount: float, ir_name: str = "warm_studio") -> Mix:
    """Dry + (IR → HPF300 → LPF6k) wet in parallel — the "Abbey Road trick"."""
    from core.audio_processor import IR_CATALOG, DEFAULT_IR
    from pedalboard import Mix, Gain

    reverb_amount = float(np.clip(reverb_amount, 0.0, 0.5))
    ir_path = IR_CATALOG.get(ir_name, IR_CATALOG[DEFAULT_IR])["path"]
    dry_db = 20 * np.log10(max(1.0 - reverb_amount, 1e-5))
    wet_db = 20 * np.log10(max(reverb_amount, 1e-5))
    return Mix([
        Gain(gain_db=dry_db),
        Pedalboard([
            Convolution(impulse_response_filename=ir_path, mix=1.0),
            HighpassFilter(cutoff_frequency_hz=300),
            LowpassFilter(cutoff_frequency_hz=6000),
            Gain(gain_db=wet_db),
        ]),
    ])


def build_f5_voice_chain(reverb_amount: float = 0.15, ir_name: str = "warm_studio") -> Pedalboard:
    """(docstring unchanged)"""
    return Pedalboard([parallel_reverb_mix(reverb_amount, ir_name), Limiter(threshold_db=-1.0)])
```

(Ensure `Mix` is importable at module scope for the annotation, or annotate as
`"Mix"`; keep the existing `Limiter` — the F5 chain is golden path and must not
change.)

`core/elevenlabs_tts/postprocessor.py`:

```python
"""Light, clean post-processing for ElevenLabs voice.

ElevenLabs output is studio-clean: no DeepFilterNet, no mastering engine, no
saturation. Just a gentle de-ess, an HPF, and optional room reverb.
"""

from __future__ import annotations

import os

import numpy as np
from pedalboard import HighpassFilter, Pedalboard


def deess_elevenlabs(audio: np.ndarray, sr: int) -> np.ndarray:
    if os.environ.get("MOODSCAPE_ELEVENLABS_DEESS", "1") == "0":
        return audio
    from core.f5_tts.postprocessor import split_band_deess
    return split_band_deess(audio, sr)


def build_elevenlabs_voice_chain(reverb_amount: float = 0.0,
                                 ir_name: str = "warm_studio") -> Pedalboard:
    plugins = [HighpassFilter(cutoff_frequency_hz=70)]
    if reverb_amount > 0:
        from core.f5_tts.postprocessor import parallel_reverb_mix
        plugins.append(parallel_reverb_mix(reverb_amount, ir_name))
    return Pedalboard(plugins)
```

- [ ] **Step 4: Run tests** — new file PASS; `tests/unit/test_meditation_mastering.py tests/unit/test_f5_phases.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add core/elevenlabs_tts/postprocessor.py core/f5_tts/postprocessor.py tests/unit/test_elevenlabs_postprocessor.py
git commit -m "feat(elevenlabs): clean voice chain; extract shared parallel reverb (F5 bit-identical)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 2 gate (controller)

- [ ] Full suite green. Phase reviewer on the Phase 2 diff, focus: SDK touched
  only in `client.py`; no sound/reaction tags can reach `convert()`; mask and
  audio lengths always equal; `build_f5_voice_chain` unchanged in behaviour;
  cache key excludes request ids.

---

## Phase 3 — Pipeline and caches integration

### Task 7: Engine registry

**Files:**
- Modify: `core/engine_registry.py`
- Test: `tests/unit/test_engine_registry.py` (append)

**Interfaces:**
- Produces: `BUILTIN_ENGINES["elevenlabs"]` with `provider="cloud"`; the three
  local entries gain `provider="local"`; `list_engines(provider: str | None = None) -> list[tuple[str, str]]`;
  `get_engine("elevenlabs", elevenlabs_params=ElevenLabsParams | None)` returns `ElevenLabsEngine`.
  `list_all_engines()` unchanged (local only — it feeds the Local Models radio).

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_engine_registry.py`, above any `__main__` guard):

```python
def test_elevenlabs_registered_as_cloud():
    from core.engine_registry import BUILTIN_ENGINES, get_engine_info
    info = get_engine_info("elevenlabs")
    assert info["provider"] == "cloud" and info["base_engine"] == "elevenlabs"
    assert {BUILTIN_ENGINES[k]["provider"] for k in ("f5", "kokoro", "chatterbox")} == {"local"}


def test_list_all_engines_stays_local_only():
    from core.engine_registry import list_all_engines, list_engines
    assert "elevenlabs" not in [eid for _l, eid in list_all_engines()]
    assert list_engines("cloud") == [("ElevenLabs", "elevenlabs")]


def test_get_engine_elevenlabs():
    from core.elevenlabs_tts.engine import ElevenLabsEngine, resolve_params
    from core.engine_registry import get_engine
    p = resolve_params(voice_id="v")
    eng = get_engine("elevenlabs", elevenlabs_params=p)
    assert isinstance(eng, ElevenLabsEngine) and eng.params == p
```

- [ ] **Step 2: Run to verify failure** — FAIL (`KeyError`/`None` for elevenlabs).

- [ ] **Step 3: Implement**

In `BUILTIN_ENGINES` add `"provider": "local",` to kokoro, f5, chatterbox, and:

```python
    "elevenlabs": {
        "id": "elevenlabs",
        "name": "ElevenLabs",
        "description": "ElevenLabs Eleven v4 cloud TTS (48 kHz, API).",
        "base_engine": "elevenlabs",
        "provider": "cloud",
        "default_voice": None,  # from assets/elevenlabs/voices.toml
        "presets": {"reverb_amount": 0.0},
    },
```

Add after `list_all_engines`:

```python
def list_engines(provider: str | None = None) -> list[tuple[str, str]]:
    """Built-in engines as (label, id), optionally filtered by provider."""
    return [
        (info["name"], eid) for eid, info in BUILTIN_ENGINES.items()
        if provider is None or info.get("provider") == provider
    ]
```

In `get_engine`, before the promoted-models lookup:

```python
    if norm_id == "elevenlabs":
        from core.elevenlabs_tts.engine import ElevenLabsEngine
        return ElevenLabsEngine(params=kwargs.get("elevenlabs_params"))
```

- [ ] **Step 4: Run tests** — `tests/unit/test_engine_registry.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add core/engine_registry.py tests/unit/test_engine_registry.py
git commit -m "feat(registry): register ElevenLabs as a cloud provider engine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: TTS cache key and env fingerprint

**Files:**
- Modify: `core/tts_cache.py`
- Test: `tests/unit/test_tts_cache.py` (append)

**Interfaces:**
- Produces: `compute_cache_key(..., extra: dict | None = None)`; env prefixes
  include `MOODSCAPE_ELEVENLABS_`; `MOODSCAPE_ELEVENLABS_CACHE_DIR` and
  `MOODSCAPE_ELEVENLABS_MAX_RETRIES` are excluded (they never change audio).

- [ ] **Step 1: Write the failing tests** (append):

```python
class TestElevenLabsCacheKey:
    def test_extra_changes_key_and_none_is_backward_compatible(self):
        base = dict(script="Hi.", tts_engine="elevenlabs", voice="v")
        assert compute_cache_key(**base) == compute_cache_key(**base, extra=None)
        assert compute_cache_key(**base, extra={"stability": 0.5}) != \
               compute_cache_key(**base, extra={"stability": 0.6})

    def test_elevenlabs_env_in_fingerprint_except_non_audio_vars(self):
        fp = tts_env_fingerprint({"MOODSCAPE_ELEVENLABS_SEED": "7",
                                  "MOODSCAPE_ELEVENLABS_CACHE_DIR": "/x",
                                  "MOODSCAPE_ELEVENLABS_MAX_RETRIES": "5"})
        assert fp == "MOODSCAPE_ELEVENLABS_SEED=7"
```

- [ ] **Step 2: Run to verify failure** — FAIL (`unexpected keyword 'extra'`).

- [ ] **Step 3: Implement**

```python
_TTS_ENV_PREFIXES = ("MOODSCAPE_F5_", "MOODSCAPE_CHATTERBOX_", "MOODSCAPE_ELEVENLABS_")
# Prefixed vars that never change rendered audio (paths, retry budgets).
_TTS_ENV_EXCLUDE = {"MOODSCAPE_ELEVENLABS_CACHE_DIR", "MOODSCAPE_ELEVENLABS_MAX_RETRIES"}
```

In `tts_env_fingerprint`: filter `k.startswith(_TTS_ENV_PREFIXES) and k not in _TTS_ENV_EXCLUDE`.
In `compute_cache_key`: add parameter `extra: dict | None = None` (last) and,
after the `env_fingerprint` block, `if extra: key_dict["extra"] = extra`.
Document `extra` in the docstring ("engine-specific render params, e.g.
ElevenLabs voice/stability/similarity/direction/model/seed").

- [ ] **Step 4: Run tests** — `tests/unit/test_tts_cache.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add core/tts_cache.py tests/unit/test_tts_cache.py
git commit -m "feat(tts_cache): engine-specific extra key params and ELEVENLABS env fingerprint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: Pipeline ElevenLabs branch

**Files:**
- Modify: `core/pipeline.py` (signature ~L46-80; engine resolution ~L169-175;
  preprocessing ~L183-191; cache ~L207-240; engine creation ~L246-262;
  synthesize ~L271-286; cache save ~L293-310; mastering ~L355-368;
  upsample ~L375-395; DeepFilterNet ~L402-413; voice FX ~L524-544)
- Create: `tests/unit/pipeline_fakes.py` (shared fakes for Tasks 9, 11, 12)
- Test: `tests/unit/test_pipeline_elevenlabs.py`

**Interfaces:**
- Consumes: Task 3 `prepare_segments`, `sanitize_direction`, `default_direction`;
  Task 5 `resolve_params`, `ElevenLabsEngine`, `ElevenLabsParams.cache_extra`;
  Task 6 `deess_elevenlabs`, `build_elevenlabs_voice_chain`; Task 8 `extra`.
- Produces: `MeditationPipeline.generate(..., elevenlabs_voice_id: str | None = None,
  elevenlabs_stability: float | None = None, elevenlabs_similarity: float | None = None,
  elevenlabs_direction: str | None = None)`; attribute
  `MeditationPipeline.last_tts_manifest: dict | None` (ElevenLabs manifest,
  also restored from the TTS cache metadata key `"elevenlabs"` on a cache hit).

- [ ] **Step 1: Write the failing tests**

`tests/unit/pipeline_fakes.py` (not a test module — no `test_` prefix):

```python
"""Deterministic fakes for running MeditationPipeline without models or network."""

import numpy as np

from core.speech_engine import SpeechEngine


class MockTTS24(SpeechEngine):
    """Local-style engine: 3 s of constant 24 kHz audio."""

    def __init__(self):
        self.synth = 0
        self.last_qa = None

    def load_model(self): pass
    def unload_model(self): pass
    def get_available_voices(self): return []

    def synthesize(self, segments, voice=None, speed=1.0, progress_cb=None, seed=None, **kw):
        self.synth += 1
        t = np.arange(24000 * 3) / 24000
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), np.ones(24000 * 3, bool)


class FakeEL(SpeechEngine):
    sample_rate = 48000

    def __init__(self):
        self.synth = 0
        self.last_manifest = {"chars_billed": 42, "cache_hits": 0}
        self.last_qa = None

    def load_model(self): pass
    def unload_model(self): pass
    def get_available_voices(self): return []

    def synthesize(self, segments, voice=None, speed=None, progress_cb=None, **kw):
        self.synth += 1
        assert all(s["type"] in ("speech", "pause") for s in segments)
        t = np.arange(48000 * 3) / 48000
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), np.ones(48000 * 3, bool)


class FakeMusic:
    def __init__(self, *a, **k): pass
    def load_model(self): pass
    def unload_model(self): pass
    def generate(self, prompt, duration_sec, progress_cb=None, **kw):
        return np.full(int(duration_sec * 48000), 0.05, dtype=np.float32)
```

`tests/unit/test_pipeline_elevenlabs.py`:

```python
"""Pipeline ElevenLabs branch: native 48 kHz, no upsample/DF/mastering, clean chain."""

import pytest

from core.pipeline import MeditationPipeline
from tests.unit.pipeline_fakes import FakeEL, FakeMusic, MockTTS24


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    calls = {"deess": 0}
    def no(*a, **k):
        raise AssertionError("must not run for ElevenLabs")
    monkeypatch.setattr("core.deepfilter_enhancer.enhance_voice_deepfilter", no)
    monkeypatch.setattr("core.audio_processor.upsample_audio", no)
    def spy(audio, sr):
        calls["deess"] += 1
        return audio
    monkeypatch.setattr("core.elevenlabs_tts.postprocessor.deess_elevenlabs", spy)
    return calls


def _run(pipe, eng, **kw):
    return pipe.generate(script="Close your eyes. [pause:2s]\n\nRest.", music_prompt="",
                         tts_engine="elevenlabs", custom_tts_engine=eng, music_model="upload",
                         uploaded_music_path="dummy.wav", elevenlabs_voice_id="v", **kw)


def test_elevenlabs_branch_skips_upsample_and_denoise(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    out, status = _run(pipe, eng)
    assert out and eng.synth == 1 and env["deess"] == 1
    assert pipe.last_tts_manifest["chars_billed"] == 42


def test_cache_hit_restores_48k_and_manifest(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng)
    _run(pipe, eng)
    assert eng.synth == 1 and pipe.last_tts_cache_hit is True
    assert pipe.last_tts_manifest["chars_billed"] == 42


def test_changed_elevenlabs_param_misses_cache(env):
    pipe, eng = MeditationPipeline(), FakeEL()
    _run(pipe, eng, elevenlabs_stability=0.5)
    _run(pipe, eng, elevenlabs_stability=0.6)
    _run(pipe, eng, elevenlabs_direction="[warm, low voice]")
    assert eng.synth == 3


def test_invalid_direction_fails_before_tts(env):
    eng = FakeEL()
    with pytest.raises(ValueError, match="voice quality"):
        _run(MeditationPipeline(), eng, elevenlabs_direction="[sighs]")
    assert eng.synth == 0


def test_local_engine_cache_key_ignores_elevenlabs_params(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    from core import tts_cache
    seen = []
    real = tts_cache.compute_cache_key
    monkeypatch.setattr(tts_cache, "compute_cache_key",
                        lambda **kw: seen.append(kw) or real(**kw))
    MeditationPipeline().generate(script="Rest.", music_prompt="", tts_engine="mock",
                                  custom_tts_engine=MockTTS24(), music_model="upload",
                                  uploaded_music_path="dummy.wav",
                                  elevenlabs_stability=0.9, elevenlabs_voice_id="zzz")
    assert seen and seen[0].get("extra") is None
```

- [ ] **Step 2: Run to verify failure** — FAIL (`unexpected keyword 'elevenlabs_voice_id'`).

- [ ] **Step 3: Implement** — edits to `core/pipeline.py`, in order:

1. `__init__`: add `self.last_tts_manifest = None`. `generate()` start: reset
   `self.last_tts_manifest = None`.
2. Signature: append the four `elevenlabs_*` keyword args (defaults `None`)
   after `reuse_last_voice`, and document them in the Args docstring.
3. Engine resolution: change the allow-list to
   `("f5", "kokoro", "chatterbox", "elevenlabs")`.
4. Right after `base_engine` is final, resolve ElevenLabs params once
   (raises `ValueError` on a bad direction before any TTS work):

```python
        el_params = el_direction = None
        if base_engine == "elevenlabs":
            from core.elevenlabs_tts.direction import default_direction, sanitize_direction
            from core.elevenlabs_tts.engine import resolve_params
            el_direction = (sanitize_direction(elevenlabs_direction)
                            if elevenlabs_direction is not None else default_direction())
            el_params = resolve_params(
                voice_id=elevenlabs_voice_id, stability=elevenlabs_stability,
                similarity=elevenlabs_similarity,
                seed=seed if seed_is_explicit else None,
            )
```

5. Preprocessing: add a first branch

```python
                if base_engine == "elevenlabs":
                    from core.elevenlabs_tts.preprocessor import prepare_segments as _el_prepare
                    segments = _el_prepare(script, content_type=content_type, direction=el_direction)
                else:
                    (existing chatterbox/f5/kokoro if-chain, then)
                    segments = _prepare(script, content_type=content_type)
```

6. Cache key: `cache_voice_id = el_params.voice_id if base_engine == "elevenlabs" else (existing expression)`;
   pass `extra=el_params.cache_extra(el_direction) if el_params else None` to
   `compute_cache_key`.
7. Cache hit: after `voice_activity = cached_voice["voice_activity"]` add

```python
                    tts_sr = int(cached_voice.get("sample_rate", SAMPLE_RATE))
                    cached_el = (cached_voice.get("metadata") or {}).get("elevenlabs")
                    self.last_tts_manifest = cached_el if isinstance(cached_el, dict) else None
```

   and in the `reuse_last_voice` branch the same `tts_sr` line (put `tts_sr`
   assignment inside the shared `if cached_voice is not None:` block so both
   paths get it).
8. Engine creation: before `elif tts_engine == "f5":` add

```python
                    elif base_engine == "elevenlabs":
                        from core.elevenlabs_tts.engine import ElevenLabsEngine
                        tts = ElevenLabsEngine(params=el_params)
                        _progress(progress_cb, 0.05, "Connecting to ElevenLabs...")
```

   and when `custom_tts_engine` is an ElevenLabs-style engine nothing changes
   (it is used as-is).
9. Synthesis: add before the final `else:`

```python
                    elif base_engine == "elevenlabs":
                        voice_audio, voice_activity = tts.synthesize(
                            segments, voice=el_params.voice_id, progress_cb=tts_progress,
                        )
```

   After synthesis: `tts_sr = int(getattr(tts, "sample_rate", SAMPLE_RATE))`;
   change the `logger.info("TTS complete …", len(voice_audio) / SAMPLE_RATE)` to
   divide by `tts_sr`; capture
   `manifest = getattr(tts, "last_manifest", None); self.last_tts_manifest = manifest if isinstance(manifest, dict) and manifest else None`.
10. Cache save: `sample_rate=tts_sr`; `"voice": cache_voice_id`; and
    `if self.last_tts_manifest: meta["elevenlabs"] = self.last_tts_manifest`.
11. Mastering engine: no change needed (stays `None` for elevenlabs) — add a
    comment `# ElevenLabs: no mastering engine; de-essed after the upsample step.`
12. Upsample: wrap the existing upsample block (from `if mix_sr == 48000:` through
    the pad/trim lines) as

```python
                if tts_sr == mix_sr:
                    pass  # ElevenLabs returns pcm_48000 — already at the mix rate
                else:
                    if tts_sr != SAMPLE_RATE:
                        raise ValueError(f"Unsupported TTS sample rate {tts_sr} for mix rate {mix_sr}.")
                    (existing upsample + pad/trim code, unchanged, indented)
```

13. DeepFilterNet: `if mix_sr == 48000 and base_engine != "elevenlabs":`; then
    directly after the DeepFilterNet block:

```python
                if base_engine == "elevenlabs":
                    from core.elevenlabs_tts import postprocessor as el_post
                    voice_audio = el_post.deess_elevenlabs(voice_audio, mix_sr)
```

    (Call through the module attribute so the test's monkeypatch is honoured.)
14. Voice FX: add before the final `else:`

```python
                elif base_engine == "elevenlabs":
                    from core.elevenlabs_tts.postprocessor import build_elevenlabs_voice_chain
                    from core.kokoro_tts.postprocessor import apply_fx
                    voice_chain = build_elevenlabs_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
```

    and change the apply line to
    `voice_audio = apply_fx(voice_audio, voice_chain, mix_sr, tape_saturation=(base_engine != "elevenlabs"), engine=base_engine)`.

Every local-engine line executes exactly as before (default
`tape_saturation=True`, same upsample code, same DeepFilterNet call).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_pipeline_elevenlabs.py tests/unit/test_pipeline_cache.py -v`
Expected: PASS. Then the full suite → green.

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/unit/pipeline_fakes.py tests/unit/test_pipeline_elevenlabs.py
git commit -m "feat(pipeline): ElevenLabs branch — native 48 kHz, no upsample/denoise, clean chain

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 10: Script-gen plumbing (WPM, estimator, guides)

**Files:**
- Modify: `core/script_gen/duration.py:44-47, 69-79`
- Modify: `core/script_gen/rules.py:29-39`
- Test: `tests/unit/test_duration_estimate.py`, `tests/unit/test_script_rules.py` (append)

**Interfaces:**
- Produces: `DEFAULT_WPM["elevenlabs"] = 85.0`; `estimate_duration_sec(..., engine="elevenlabs")`
  uses the ElevenLabs preprocessor; `rules.GUIDE_ENGINE_ALIASES = {"elevenlabs": "f5"}`;
  `load_guide("elevenlabs", ct)` returns the F5 guide rendered with
  `guide_context("elevenlabs", ct)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_duration_estimate.py`:

```python
class TestElevenLabsEstimate(unittest.TestCase):
    def test_close_to_f5_and_breaths_counted(self):
        script = "Rest here now. [pause:5s]\n\nLet go. [exhale] Soften."
        el = estimate_duration_sec(script, engine="elevenlabs")
        f5 = estimate_duration_sec(script, engine="f5")
        self.assertGreater(el, 5.0 + 1.8)
        self.assertAlmostEqual(el, f5, delta=1.0)
```

(Use the module's existing import style; if it uses plain pytest functions,
write a plain function instead.)

Append to `tests/unit/test_script_rules.py`:

```python
def test_elevenlabs_uses_f5_guides():
    from core.script_gen.rules import load_guide
    for ct in ("meditation", "sleep_story"):
        el = load_guide("elevenlabs", ct)
        assert el == load_guide("f5", ct)
```

- [ ] **Step 2: Run to verify failure** — FAIL (`KeyError 'elevenlabs'` / `ValueError Unknown engine`).

- [ ] **Step 3: Implement**

`duration.py`: add `"elevenlabs": 85.0,` to `DEFAULT_WPM` with comment
`# seeded from F5; recalibrate from log_estimate_accuracy after real renders`.
In `_load_prepare_segments` add:

```python
    if engine == "elevenlabs":
        from core.elevenlabs_tts.preprocessor import prepare_segments as _el

        def prepare_segments(script, content_type="meditation"):
            # The direction prefix is not spoken text; strip it so WPM counts words only.
            segs = _el(script, content_type=content_type)
            return [
                {**s, "text": s["text"].split("] ", 1)[1]} if s["type"] == "speech" else s
                for s in segs
            ]
        return prepare_segments
```

and update the error message to list `'f5', 'kokoro' or 'elevenlabs'`.

`rules.py`:

```python
# Engines that share another engine's prompting guide (the script format is
# engine-neutral; ElevenLabs renders F5-style scripts).
GUIDE_ENGINE_ALIASES = {"elevenlabs": "f5"}
```

and in `load_guide`: `guide_engine = GUIDE_ENGINE_ALIASES.get(engine, engine)`;
format the filename with `engine=guide_engine`; keep
`guide_context(engine, content_type)` (it now resolves `elevenlabs` values —
`elevenlabs_paragraph_pause_sec` and `DEFAULT_WPM["elevenlabs"]` exist, and the
non-kokoro branch supplies `CHUNK_CHARS`).

- [ ] **Step 4: Run tests** — both files PASS; `tests/unit/test_guide_rendering.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/duration.py core/script_gen/rules.py tests/unit/test_duration_estimate.py tests/unit/test_script_rules.py
git commit -m "feat(script_gen): ElevenLabs WPM, duration estimate and F5 guide alias

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 3 gate (controller)

- [ ] Full suite green. Phase reviewer on the Phase 3 diff, focus: every
  local-engine statement in `pipeline.generate` is unchanged (diff the local
  branches line by line); ElevenLabs params never reach the local cache key;
  `last_tts_manifest` survives cache hits; guides/estimator for f5/kokoro
  unchanged.

---

## Phase 4 — Remix (both providers)

### Task 11: Mix sessions — save in pipeline, prune

**Files:**
- Create: `core/mix_session.py`
- Modify: `core/pipeline.py` (Step 9 mix ~L652-667 and after export ~L700)
- Test: `tests/unit/test_mix_session.py`

**Interfaces:**
- Produces: `class MixSessionNotFound(FileNotFoundError)`; `sessions_dir() -> Path`
  (env `MOODSCAPE_MIX_SESSIONS_DIR`, default `var/mix_sessions`);
  `new_run_id() -> str`; `save_session(run_id, voice_audio, voice_activity, music_audio, params: dict) -> Path`;
  `update_params(run_id, updates: dict) -> None`;
  `load_session(run_id) -> MixSession` (`voice_audio`, `voice_activity`, `music_audio`, `params`);
  `prune_sessions(keep: int | None = None) -> int` (env `MOODSCAPE_MIX_SESSIONS_KEEP`, default 5);
  `MeditationPipeline.last_run_id: str | None`.
- `params` keys written by the pipeline: `mix_sr, duck_amount_db, music_volume_db,
  fade_in_sec, fade_out_sec, stereo_output, phrases, bed_overrides, target_lufs,
  mono_offset, output_format, export_sr, use_upload, content_type, env, output_path`.

- [ ] **Step 1: Write the failing tests**

```python
import os
import time

import numpy as np
import pytest

from core import mix_session as ms


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path))


def _save(run_id):
    v = np.ones(10, np.float32); m = np.ones(10, bool); mu = np.zeros(12, np.float32)
    return ms.save_session(run_id, v, m, mu, {"mix_sr": 48000, "phrases": [(0.0, 1.0)]})


def test_round_trip_and_update():
    _save("r1")
    ms.update_params("r1", {"output_path": "/tmp/x.wav"})
    s = ms.load_session("r1")
    assert s.voice_audio.dtype == np.float32 and s.voice_activity.dtype == bool
    assert len(s.music_audio) == 12
    assert s.params["output_path"] == "/tmp/x.wav" and s.params["phrases"] == [[0.0, 1.0]]


def test_missing_raises_clear_error():
    with pytest.raises(ms.MixSessionNotFound, match="generate again"):
        ms.load_session("nope")


def test_prune_keeps_newest(monkeypatch):
    for i in range(7):
        p = _save(f"r{i}")
        os.utime(p, (time.time() + i, time.time() + i))
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_KEEP", "5")
    assert ms.prune_sessions() == 2
    with pytest.raises(ms.MixSessionNotFound):
        ms.load_session("r0")
    ms.load_session("r6")


def test_run_ids_unique():
    assert ms.new_run_id() != ms.new_run_id()


def test_pipeline_saves_session(tmp_path, monkeypatch):
    from core.pipeline import MeditationPipeline
    from tests.unit.pipeline_fakes import FakeMusic, MockTTS24
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    pipe = MeditationPipeline()
    out, _ = pipe.generate(script="Rest.", music_prompt="", tts_engine="mock",
                           custom_tts_engine=MockTTS24(), music_model="upload",
                           uploaded_music_path="dummy.wav")
    s = ms.load_session(pipe.last_run_id)
    assert s.params["output_path"] == out and s.params["mix_sr"] == 48000
    assert len(s.voice_audio) == len(s.voice_activity)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module not found).

- [ ] **Step 3: Implement**

`core/mix_session.py`:

```python
"""Saved mix inputs so Remix can re-run only mix -> master -> export.

A session is the processed voice (+ activity mask) and processed music exactly
as passed to mixer.mix(), plus the parameters used. Sessions are large
(~350 MB for 15 min of 48 kHz float32), so only the newest few are kept.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "var" / "mix_sessions"


class MixSessionNotFound(FileNotFoundError):
    pass


@dataclass
class MixSession:
    voice_audio: np.ndarray
    voice_activity: np.ndarray
    music_audio: np.ndarray
    params: dict


def sessions_dir() -> Path:
    d = Path(os.environ.get("MOODSCAPE_MIX_SESSIONS_DIR", DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_session(run_id: str, voice_audio, voice_activity, music_audio, params: dict) -> Path:
    d = sessions_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "voice.npy", np.asarray(voice_audio, dtype=np.float32))
    np.save(d / "voice_activity.npy", np.asarray(voice_activity, dtype=bool))
    np.save(d / "music.npy", np.asarray(music_audio, dtype=np.float32))
    (d / "mix_params.json").write_text(json.dumps(_json_safe(params), indent=2), encoding="utf-8")
    return d


def _dir_for(run_id: str) -> Path:
    d = sessions_dir() / run_id
    if not run_id or not (d / "mix_params.json").is_file():
        raise MixSessionNotFound(f"Mix session not found ({run_id!r}) — generate again to remix.")
    return d


def update_params(run_id: str, updates: dict) -> None:
    path = _dir_for(run_id) / "mix_params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    params.update(_json_safe(updates))
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_session(run_id: str) -> MixSession:
    d = _dir_for(run_id)
    return MixSession(
        voice_audio=np.load(d / "voice.npy"),
        voice_activity=np.load(d / "voice_activity.npy"),
        music_audio=np.load(d / "music.npy"),
        params=json.loads((d / "mix_params.json").read_text(encoding="utf-8")),
    )


def prune_sessions(keep: int | None = None) -> int:
    keep = keep if keep is not None else int(os.environ.get("MOODSCAPE_MIX_SESSIONS_KEEP", "5"))
    dirs = sorted((p for p in sessions_dir().iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    stale = dirs[max(0, keep):]
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)
    return len(stale)
```

`core/pipeline.py`:
- `__init__` and start of `generate()`: `self.last_run_id = None`.
- In the Step 9 `else:` branch (voice + music), immediately **before**
  `mixed = mix(`:

```python
                run_id = None
                try:
                    from core import mix_session
                    run_id = mix_session.new_run_id()
                    mix_session.save_session(run_id, voice_audio, voice_activity, music_audio, {
                        "mix_sr": mix_sr, "duck_amount_db": duck_amount_db,
                        "music_volume_db": effective_music_volume_db,
                        "fade_in_sec": fade_in_sec, "fade_out_sec": fade_out_sec,
                        "stereo_output": bool(stereo_output), "phrases": mix_phrases,
                        "bed_overrides": bed_profile["duck_kwargs"] if bed_profile else None,
                        "target_lufs": target_lufs, "mono_offset": mono_offset,
                        "output_format": output_format,
                        "export_sr": 48000 if (use_lyria or use_upload or upsample_48k) else TARGET_SR,
                        "use_upload": bool(use_upload), "content_type": content_type,
                        "env": {k: v for k, v in os.environ.items() if k.startswith("MOODSCAPE_")},
                    })
                    mix_session.prune_sessions()
                except Exception as e:  # Remix is a convenience; never fail a render for it
                    logger.warning("Could not save mix session: %s", e)
                    run_id = None
                self.last_run_id = run_id
```

- After `output_path = export_audio(...)`:

```python
            if self.last_run_id:
                try:
                    from core import mix_session
                    mix_session.update_params(self.last_run_id, {"output_path": output_path})
                except Exception as e:
                    logger.warning("Could not record output path for mix session: %s", e)
```

- [ ] **Step 4: Run tests** — `tests/unit/test_mix_session.py` PASS; full suite green
  (existing pipeline tests now also write sessions — confirm they set
  `MOODSCAPE_MIX_SESSIONS_DIR`; if a test does not, sessions go to
  `var/mix_sessions`, which is gitignored and pruned — acceptable, but add the
  env to `tests/unit/test_pipeline_cache.py`'s `temp_tts_cache` fixture so unit
  runs never touch `var/`).

- [ ] **Step 5: Commit**

```bash
git add core/mix_session.py core/pipeline.py tests/unit/test_mix_session.py tests/unit/test_pipeline_cache.py
git commit -m "feat(pipeline): save mix sessions for post-generation remix

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 12: `core/remix.py`

**Files:**
- Create: `core/remix.py`
- Test: `tests/unit/test_remix.py`

**Interfaces:**
- Consumes: Task 11 `load_session`, `MixSessionNotFound`.
- Produces: `remix(run_id: str, voice_db: float = 0.0, music_db: float = 0.0, progress_cb=None) -> str`
  (path of the new file); `remix_output_path(original: str, voice_db: float, music_db: float) -> Path`;
  `TRIM_RANGE_DB = (-12.0, 12.0)`.

- [ ] **Step 1: Write the failing tests**

```python
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import mix_session as ms
from core.remix import remix, remix_output_path

SR = 48000


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_MIX_SESSIONS_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("MOODSCAPE_TTS_CACHE_DIR", str(tmp_path / "tts"))


def _pipeline_run(tmp_path, monkeypatch):
    from core.pipeline import MeditationPipeline
    from tests.unit.pipeline_fakes import FakeMusic, MockTTS24
    monkeypatch.setattr("core.upload_music.UploadMusicEngine", FakeMusic)
    pipe = MeditationPipeline()
    out, _ = pipe.generate(script="Rest here. [pause:2s]\n\nLet go.", music_prompt="",
                           tts_engine="mock", custom_tts_engine=MockTTS24(), music_model="upload",
                           uploaded_music_path="dummy.wav")
    return pipe.last_run_id, out


def test_zero_trims_reproduce_original_exactly(tmp_path, monkeypatch):
    run_id, original = _pipeline_run(tmp_path, monkeypatch)
    new = remix(run_id, 0.0, 0.0)
    assert new != original and Path(original).exists()
    a, _ = sf.read(original); b, _ = sf.read(new)
    assert np.array_equal(a, b)


def _band_db(x, f):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    band = (freqs > f * 0.9) & (freqs < f * 1.1)
    return 20 * np.log10(spec[band].sum() + 1e-12)


def test_voice_trim_shifts_voice_to_music_ratio(tmp_path):
    t = np.arange(SR * 20) / SR
    voice = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    act = np.zeros(len(t), bool); act[: SR * 10] = True
    music = (0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    out0 = str(tmp_path / "orig.wav")
    sf.write(out0, np.zeros(10), SR)
    ms.save_session("syn", voice, act, music, {
        "mix_sr": SR, "duck_amount_db": -16.0, "music_volume_db": -16.0,
        "fade_in_sec": 1.5, "fade_out_sec": 5.0, "stereo_output": False,
        "phrases": None, "bed_overrides": None, "target_lufs": -16.0, "mono_offset": -3.01,
        "output_format": "wav", "export_sr": SR, "use_upload": True,
        "content_type": "meditation", "env": {}, "output_path": out0})
    base, _ = sf.read(remix("syn", 0.0, 0.0))
    up, _ = sf.read(remix("syn", 6.0, 0.0))
    ratio0 = _band_db(base, 1000) - _band_db(base, 150)
    ratio6 = _band_db(up, 1000) - _band_db(up, 150)
    assert ratio6 - ratio0 == pytest.approx(6.0, abs=1.5)


def test_output_name_and_no_overwrite(tmp_path):
    orig = tmp_path / "a.wav"
    orig.write_bytes(b"x")
    p = remix_output_path(str(orig), 2.0, -3.0)
    assert p.name == "a_remix-v+2-m-3.wav"
    p.write_bytes(b"y")
    assert remix_output_path(str(orig), 2.0, -3.0).name == "a_remix-v+2-m-3-2.wav"


def test_missing_session_raises_clear_error():
    with pytest.raises(ms.MixSessionNotFound, match="generate again"):
        remix("gone", 1.0, 0.0)


def test_trims_out_of_range_rejected(tmp_path, monkeypatch):
    run_id, _ = _pipeline_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="-12"):
        remix(run_id, 13.0, 0.0)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module not found).

- [ ] **Step 3: Implement** `core/remix.py`:

```python
"""Remix: re-run only mix -> master -> export from a saved mix session.

Voice/music trims change the voice-to-music BALANCE; export re-normalizes the
final loudness, so equal trims on both are inaudible by design.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from core.audio_processor import make_master_chain
from core.mix_session import load_session
from core.mixer import export_audio, mix
from core.qa_monitor import run_qa_checks

logger = logging.getLogger("moodscape.remix")
TRIM_RANGE_DB = (-12.0, 12.0)


@contextlib.contextmanager
def _session_env(saved: dict):
    """Apply the MOODSCAPE_* env the original mix ran with, then restore."""
    current = {k: v for k, v in os.environ.items() if k.startswith("MOODSCAPE_")}
    try:
        for k in current:
            if k not in saved:
                os.environ.pop(k, None)
        os.environ.update(saved)
        yield
    finally:
        for k in list(os.environ):
            if k.startswith("MOODSCAPE_") and k not in current:
                os.environ.pop(k, None)
        os.environ.update(current)


def remix_output_path(original: str, voice_db: float, music_db: float) -> Path:
    src = Path(original)
    stem = f"{src.stem}_remix-v{voice_db:+g}-m{music_db:+g}"
    candidate = src.with_name(f"{stem}{src.suffix}")
    n = 2
    while candidate.exists():
        candidate = src.with_name(f"{stem}-{n}{src.suffix}")
        n += 1
    return candidate


def _check(name: str, value: float) -> float:
    lo, hi = TRIM_RANGE_DB
    if not lo <= value <= hi:
        raise ValueError(f"{name} trim {value} dB is outside {lo:g}..{hi:g} dB.")
    return float(value)


def remix(run_id: str, voice_db: float = 0.0, music_db: float = 0.0, progress_cb=None) -> str:
    voice_db = _check("Voice", voice_db)
    music_db = _check("Music", music_db)
    session = load_session(run_id)
    p = session.params
    voice = session.voice_audio
    if voice_db != 0.0:
        voice = (voice * np.float32(10.0 ** (voice_db / 20.0))).astype(np.float32)
    phrases = [tuple(x) for x in p["phrases"]] if p.get("phrases") else None
    with _session_env(p.get("env", {})):
        mixed = mix(
            voice, session.voice_activity, session.music_audio,
            sample_rate=p["mix_sr"], duck_amount_db=p["duck_amount_db"],
            music_volume_db=float(np.clip(p["music_volume_db"] + music_db, -40.0, 0.0)),
            fade_in_sec=p["fade_in_sec"], fade_out_sec=p["fade_out_sec"],
            stereo_output=p["stereo_output"], phrases=phrases,
            bed_overrides=p.get("bed_overrides"),
        )
        qa = run_qa_checks(mixed, p["mix_sr"], is_vocals_only=False,
                           is_uploaded_music=p.get("use_upload", False))
        if not qa["clipping"]["passed"]:
            logger.warning("Remix %s: pre-master clipping detected", run_id)
        tmp = export_audio(
            mixed, sample_rate=p["mix_sr"], output_format=p["output_format"],
            target_sample_rate=p["export_sr"], master_chain=make_master_chain(),
            target_lufs=p["target_lufs"], mono_playback_offset_db=p["mono_offset"],
        )
    dest = remix_output_path(p["output_path"], voice_db, music_db)
    shutil.move(tmp, dest)
    dest.with_suffix(".meta.json").write_text(json.dumps(
        {"remix_of": p["output_path"], "run_id": run_id,
         "voice_db": voice_db, "music_db": music_db}, indent=2), encoding="utf-8")
    return str(dest)
```

- [ ] **Step 4: Run tests** — `tests/unit/test_remix.py` PASS. If
  `test_zero_trims_reproduce_original_exactly` fails on a tiny float difference,
  find the non-determinism (do NOT loosen the test): the mix must be a pure
  function of the saved inputs + env.

- [ ] **Step 5: Commit**

```bash
git add core/remix.py tests/unit/test_remix.py
git commit -m "feat(remix): re-run mix/master/export from a saved session with voice/music trims

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 4 gate (controller)

- [ ] Full suite green. Phase reviewer on the Phase 4 diff, focus: session save
  can never fail a render; `remix(…, 0, 0)` exactness; env restore is complete
  (no MOODSCAPE_* leakage between remix and later generations); original file
  never overwritten; disk retention honours the keep count.

---

## Phase 5 — Auto-generate and eval

### Task 13: Auto-generate with ElevenLabs

**Files:**
- Modify: `core/auto_generate.py` (`AutoConfig` ~L162-167; `from_genre` ~L241-250;
  `render_script` ~L841-912; `run` ~L1009-1011)
- Modify: `scripts/eval_genres.py:181` (help text only)
- Test: `tests/unit/test_auto_generate_elevenlabs.py`

**Interfaces:**
- Consumes: Task 2 `voice_for_family`; Task 3 `prepare_segments`,
  `estimate_billed_chars`, `default_direction`; Task 4 `ElevenLabsClient`,
  `ElevenLabsConfigError`; Task 9 pipeline kwargs + `last_tts_manifest`.
- Produces: `AutoConfig.elevenlabs_voice_id: str | None = None`;
  `tts_preflight(config: AutoConfig, client_factory=None) -> None`
  (raises `ScriptGenerationError`); meta key `"elevenlabs"` (manifest +
  `"estimated_chars"`); `RenderResult.voice` = ElevenLabs voice id for that engine.

- [ ] **Step 1: Write the failing tests**

```python
"""Auto-generate on the ElevenLabs provider (no network: fakes only)."""

import json

import pytest

from core.auto_generate import AutoConfig, ScriptGenerationError, render_script, tts_preflight
from core.genres import load_pack


class _Client:
    def __init__(self, exists=True):
        self.exists = exists
    def voice_exists(self, voice_id):
        return self.exists


class _Pipe:
    def __init__(self, path):
        self.path, self.kwargs = path, None
        self.last_tts_manifest = {"chars_billed": 10, "cache_hits": 1}
    def generate(self, **kw):
        self.kwargs = kw
        self.path.write_bytes(b"RIFF")
        return str(self.path), "ok"


def test_from_genre_picks_elevenlabs_default_voice():
    cfg = AutoConfig.from_genre(load_pack("stress_relief"), tts_engine="elevenlabs")
    assert cfg.elevenlabs_voice_id == "pjcYQlDFKMbcOUp6F5GD"
    assert cfg.voice_slug is None


def test_preflight_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ScriptGenerationError, match="ELEVENLABS_API_KEY"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"))


def test_preflight_unknown_voice(monkeypatch):
    with pytest.raises(ScriptGenerationError, match="Voice Library"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"),
                      client_factory=lambda: _Client(exists=False))


def test_preflight_bad_env_direction(monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_DIRECTION", "[sighs]")
    with pytest.raises(ScriptGenerationError, match="voice quality"):
        tts_preflight(AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v"),
                      client_factory=lambda: _Client())


def test_preflight_noop_for_local():
    tts_preflight(AutoConfig(tts_engine="f5"))


def test_render_script_passes_voice_and_records_meta(tmp_path, monkeypatch):
    monkeypatch.setattr("core.auto_generate._measure_actual_duration_sec", lambda p: 60.0)
    pipe = _Pipe(tmp_path / "out.wav")
    cfg = AutoConfig(tts_engine="elevenlabs", elevenlabs_voice_id="v")
    res = render_script("Rest here. [pause:3s]\n\nLet go.", config=cfg,
                        background_label="bg", background_path="bg.mp3", pipeline=pipe)
    assert pipe.kwargs["elevenlabs_voice_id"] == "v"
    assert "f5_voice_slug" not in pipe.kwargs
    meta = json.loads(open(res.meta_path).read())
    assert meta["voice"] == "v" and meta["elevenlabs"]["chars_billed"] == 10
    assert meta["elevenlabs"]["estimated_chars"] > 0
    assert res.voice == "v"
```

- [ ] **Step 2: Run to verify failure** — FAIL (`ImportError: tts_preflight`).

- [ ] **Step 3: Implement**

`AutoConfig`: after `voice_slug` add

```python
    # ElevenLabs voice id (provider = ElevenLabs); set by from_genre.
    elevenlabs_voice_id: str | None = None
```

`from_genre`, after the `_VOICE_SLUG_ENGINES` block:

```python
        if engine == "elevenlabs" and "elevenlabs_voice_id" not in overrides:
            from core.elevenlabs_tts.voices import voice_for_family as el_voice_for_family
            pack_values["elevenlabs_voice_id"] = el_voice_for_family(pack.family)
```

New function (module level, near `render_script`):

```python
def tts_preflight(config: AutoConfig, client_factory=None) -> None:
    """Fail in seconds — before any LLM stage — if the TTS provider cannot render."""
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
```

`render_script`:
- after the `voice_slug` block:

```python
    el_voice = None
    if config.tts_engine == "elevenlabs":
        from core.elevenlabs_tts.preprocessor import estimate_billed_chars, prepare_segments
        from core.elevenlabs_tts.voices import default_voice_id
        el_voice = config.elevenlabs_voice_id or default_voice_id()
        generate_kwargs["elevenlabs_voice_id"] = el_voice
        est_chars = estimate_billed_chars(prepare_segments(script, content_type=config.content_type))
        logger.info("ElevenLabs: ~%d characters for this render (before cache)", est_chars)
        if progress_cb is not None:
            progress_cb(0.0, f"ElevenLabs: ~{est_chars:,} characters (cached chunks are free)")
```

- in `meta.update`, change `"voice": voice_slug` to `"voice": el_voice or voice_slug`;
  after the `tts_cache_hit` block:

```python
    if el_voice is not None:
        manifest = getattr(pipeline, "last_tts_manifest", None)
        meta["elevenlabs"] = {**(manifest if isinstance(manifest, dict) else {}),
                              "estimated_chars": est_chars}
```

- `RenderResult(... voice=el_voice or voice_slug or "")`.

`run()`: immediately **before** the existing
`for engine in (planner_engine, generator_engine, judge_engine):` preflight loop
(and before any `build_engine` work would matter), call `tts_preflight(config)`.
Move the call to the very top of `run()` after `config = config or AutoConfig.from_env()`
so it fails before engines are built.

`scripts/eval_genres.py`: help text for `--engine` becomes
`"TTS engine for AutoConfig.tts_engine (f5, elevenlabs, ...); elevenlabs bills per character — see meta.json 'elevenlabs'"`.

- [ ] **Step 4: Run tests** — new file PASS; `tests/unit/test_auto_generate.py` PASS
  (existing runs use `tts_engine="f5"` → preflight no-op).

- [ ] **Step 5: Commit**

```bash
git add core/auto_generate.py scripts/eval_genres.py tests/unit/test_auto_generate_elevenlabs.py
git commit -m "feat(auto): ElevenLabs voice, fail-fast TTS preflight and billing meta

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 5 gate (controller)

- [ ] Full suite green. Phase reviewer, focus: preflight runs before any LLM
  build/load; local auto-gen path byte-identical (`voice_for_family` still used
  for F5); meta never contains the API key.

---

## Phase 6 — UI

### Task 14: `core/provider_ui.py` (testable UI logic)

**Files:**
- Create: `core/provider_ui.py`
- Test: `tests/unit/test_provider_ui.py`

**Interfaces:**
- Consumes: Tasks 2–5, 11–12.
- Produces:
  `PROVIDERS = ("ElevenLabs", "Local Models")`, `DEFAULT_PROVIDER = "Local Models"`,
  `is_cloud(provider: str) -> bool`,
  `resolve_tts_engine(provider: str, local_choice: str) -> str`,
  `voice_choices(validation: dict[str, bool] | None = None) -> list[tuple[str, str]]`,
  `validate_account_voices(client_factory=None) -> dict[str, bool] | None`,
  `api_key_banner() -> str`,
  `cost_line(script, content_type, direction, voice_id, stability, similarity) -> str`,
  `provider_visibility(provider: str, mode: str) -> dict[str, bool]`
  (keys: `tts_engine_radio, kokoro_settings, f5_settings, speed_slider,
  f5_cfg_slider, microprosody_checkbox, elevenlabs_settings`),
  `remix_ui(run_id: str | None, voice_db: float, music_db: float) -> tuple[str | None, str]`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from core import provider_ui as ui


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MOODSCAPE_ELEVENLABS_CACHE_DIR", str(tmp_path))


def test_resolve_engine():
    assert ui.resolve_tts_engine("ElevenLabs", "F5-TTS") == "elevenlabs"
    assert ui.resolve_tts_engine("Local Models", "F5-TTS") == "f5"
    assert ui.resolve_tts_engine("Local Models", "Kokoro") == "kokoro"
    assert ui.resolve_tts_engine("Local Models", "Chatterbox TTS") == "chatterbox"


def test_visibility_cloud_vs_local():
    cloud = ui.provider_visibility("ElevenLabs", "Instrumental + Vocal")
    assert cloud["elevenlabs_settings"] and not cloud["tts_engine_radio"]
    assert not cloud["speed_slider"] and not cloud["f5_settings"]
    local = ui.provider_visibility("Local Models", "Instrumental + Vocal")
    assert local["tts_engine_radio"] and not local["elevenlabs_settings"]
    assert local["speed_slider"]
    inst = ui.provider_visibility("ElevenLabs", "Instrumental Only")
    assert not inst["elevenlabs_settings"]


def test_voice_choices_mark_missing():
    choices = dict((v, label) for label, v in ui.voice_choices({"pMsXgVXv3BLzUgSXRplE": False}))
    assert choices["pMsXgVXv3BLzUgSXRplE"].endswith("(not in account)")
    assert not choices["pjcYQlDFKMbcOUp6F5GD"].endswith("(not in account)")


def test_validate_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ui.validate_account_voices() is None
    assert "ELEVENLABS_API_KEY" in ui.api_key_banner()


def test_cost_line_counts_and_cache():
    from core.elevenlabs_tts import chunk_cache
    from core.elevenlabs_tts.engine import resolve_params
    from core.elevenlabs_tts.preprocessor import prepare_segments
    line = ui.cost_line("Rest here.\n\nLet go.", "meditation", None, "v", 0.5, 0.75)
    assert "2 chunks" in line and "0/2 cached" in line
    p = resolve_params(voice_id="v", stability=0.5, similarity=0.75)
    first = prepare_segments("Rest here.\n\nLet go.")[0]["text"]
    chunk_cache.store(chunk_cache.chunk_payload(first, "v", p.model_id, 0.5, 0.75, p.seed), b"\0\0", None)
    assert "1/2 cached" in ui.cost_line("Rest here.\n\nLet go.", "meditation", None, "v", 0.5, 0.75)


def test_cost_line_reports_bad_direction():
    assert "voice quality" in ui.cost_line("Hi.", "meditation", "[sighs]", "v", 0.5, 0.75)


def test_remix_ui_without_run():
    path, msg = ui.remix_ui(None, 1.0, 0.0)
    assert path is None and "Generate" in msg
```

- [ ] **Step 2: Run to verify failure** — FAIL (module not found).

- [ ] **Step 3: Implement** `core/provider_ui.py`:

```python
"""Testable logic behind the Voice Provider toggle and Remix panel.

app.py cannot be imported in tests, so every decision the UI makes lives here
and app.py only wires components to these functions.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("moodscape.provider_ui")

PROVIDERS = ("ElevenLabs", "Local Models")
DEFAULT_PROVIDER = "Local Models"
_LOCAL_LABELS = {"F5-TTS": "f5", "Kokoro": "kokoro", "Chatterbox TTS": "chatterbox"}


def is_cloud(provider: str) -> bool:
    return provider == "ElevenLabs"


def resolve_tts_engine(provider: str, local_choice: str) -> str:
    if is_cloud(provider):
        return "elevenlabs"
    if local_choice in _LOCAL_LABELS:
        return _LOCAL_LABELS[local_choice]
    from core.engine_registry import list_all_engines
    return {label: eid for label, eid in list_all_engines()}.get(local_choice, local_choice.lower())


def provider_visibility(provider: str, mode: str) -> dict[str, bool]:
    vocal = mode != "Instrumental Only"
    cloud = is_cloud(provider)
    return {
        "tts_engine_radio": vocal and not cloud,
        "kokoro_settings": False,   # recomputed by the local engine radio handler
        "f5_settings": vocal and not cloud,
        "speed_slider": vocal and not cloud,
        "f5_cfg_slider": not cloud,
        "microprosody_checkbox": not cloud,
        "elevenlabs_settings": vocal and cloud,
    }


def api_key_banner() -> str:
    from core.elevenlabs_tts.client import has_api_key
    if has_api_key():
        return ""
    return "⚠️ ELEVENLABS_API_KEY is not set in .env — add it to generate with ElevenLabs."


_VALIDATION_CACHE: dict[str, bool] | None = None


def validate_account_voices(client_factory=None) -> dict[str, bool] | None:
    """Which registry voices exist in the account. Memoized per process for the
    real client (one round of GET /v1/voices/{id}, not one per toggle)."""
    global _VALIDATION_CACHE
    from core.elevenlabs_tts.client import ElevenLabsClient, has_api_key
    from core.elevenlabs_tts.voices import list_voices
    if client_factory is None:
        if not has_api_key():
            return None
        if _VALIDATION_CACHE is not None:
            return _VALIDATION_CACHE
    try:
        client = (client_factory or ElevenLabsClient)()
        result = {v.id: client.voice_exists(v.id) for v in list_voices()}
    except Exception as exc:  # validation is advisory; never block the UI
        logger.warning("ElevenLabs voice validation failed: %s", exc)
        return None
    if client_factory is None:
        _VALIDATION_CACHE = result
    return result


def voice_choices(validation: dict[str, bool] | None = None) -> list[tuple[str, str]]:
    from core.elevenlabs_tts.voices import list_voices
    out = []
    for v in list_voices():
        label = v.label
        if validation is not None and validation.get(v.id) is False:
            label = f"{label} (not in account)"
        out.append((label, v.id))
    return out


def cost_line(script: str, content_type: str, direction: str | None, voice_id: str,
              stability: float, similarity: float) -> str:
    from core.elevenlabs_tts import chunk_cache
    from core.elevenlabs_tts.engine import resolve_params
    from core.elevenlabs_tts.preprocessor import estimate_billed_chars, prepare_segments
    if not (script or "").strip():
        return "≈ 0 characters"
    try:
        segs = prepare_segments(script, content_type=content_type, direction=direction or None)
    except ValueError as exc:
        return f"⚠️ {exc}"
    p = resolve_params(voice_id=voice_id, stability=stability, similarity=similarity)
    speech = [s["text"] for s in segs if s["type"] == "speech"]
    cached = sum(chunk_cache.is_cached(chunk_cache.chunk_payload(
        t, p.voice_id, p.model_id, p.stability, p.similarity, p.seed)) for t in speech)
    return (f"≈ {estimate_billed_chars(segs):,} characters · {len(speech)} chunks · "
            f"{cached}/{len(speech)} cached")


def remix_ui(run_id: str | None, voice_db: float, music_db: float) -> tuple[str | None, str]:
    from core.mix_session import MixSessionNotFound
    from core.remix import remix
    if not run_id:
        return None, "Generate a meditation first, then remix it."
    try:
        path = remix(run_id, float(voice_db), float(music_db))
    except (MixSessionNotFound, ValueError) as exc:
        return None, str(exc)
    return path, f"Remixed (voice {voice_db:+g} dB, music {music_db:+g} dB) → {path}"
```

- [ ] **Step 4: Run tests** — PASS.

- [ ] **Step 5: Commit**

```bash
git add core/provider_ui.py tests/unit/test_provider_ui.py
git commit -m "feat(ui): testable provider/cost/remix logic for the Voice Provider toggle

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 15: Manual tab wiring in `app.py`

**Files:**
- Modify: `app.py` (header ~L921-933; `generate_meditation` ~L191-371; output
  column ~L987-1001; Voice Engine area ~L1026-1052; callbacks ~L1148-1285)

**Interfaces:**
- Consumes: Task 14 everything; Task 9 pipeline kwargs, `pipeline.last_run_id`,
  `pipeline.last_tts_manifest`.
- Produces: `provider_radio` (module-level component, passed to
  `build_auto_tab(provider=provider_radio)` in Task 16).

- [ ] **Step 1: Add the provider toggle** directly after the header
  `gr.HTML(...)` and before `with gr.Tabs():`:

```python
    from core.provider_ui import DEFAULT_PROVIDER, PROVIDERS
    provider_radio = gr.Radio(
        choices=list(PROVIDERS), value=DEFAULT_PROVIDER, label="Voice Provider",
        info="ElevenLabs = Eleven v4 cloud voices (billed per character). "
             "Local Models = F5 / Kokoro / Chatterbox on this Mac.",
        elem_classes="pill-radio",
    )
```

- [ ] **Step 2: Add the ElevenLabs settings group** immediately after the
  `with gr.Group(visible=True, elem_id="f5-group") as f5_settings:` block:

```python
                        from core import provider_ui
                        from core.elevenlabs_tts.direction import DEFAULT_DIRECTION
                        from core.elevenlabs_tts.voices import default_voice_id
                        with gr.Group(visible=False, elem_id="elevenlabs-group") as elevenlabs_settings:
                            el_banner = gr.Markdown(provider_ui.api_key_banner())
                            el_voice_dropdown = gr.Dropdown(
                                choices=provider_ui.voice_choices(), value=default_voice_id(),
                                label="ElevenLabs Voice", elem_classes="dropdown-container",
                            )
                            with gr.Row():
                                el_stability = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Stability",
                                                         info="Lower = more expressive; higher = steadier.")
                                el_similarity = gr.Slider(0.0, 1.0, 0.75, step=0.05, label="Similarity",
                                                          info="How closely output matches the voice.")
                            el_direction = gr.Textbox(
                                value=DEFAULT_DIRECTION, label="Delivery direction",
                                info="Voice quality only (no breaths/sighs). Eleven v4 has no speed control — "
                                     "pacing comes from this and the script's pauses.",
                            )
                            el_cost = gr.Markdown("")
```

- [ ] **Step 3: Add the Remix panel** after `status_display = gr.HTML(...)`:

```python
                    run_id_state = gr.State(None)
                    with gr.Group(visible=False) as remix_group:
                        gr.Markdown("#### Remix levels")
                        with gr.Row():
                            remix_voice_db = gr.Slider(-12, 12, 0, step=0.5, label="Voice level (dB)",
                                                       info="Changes the voice-to-music balance; final loudness is re-normalized.")
                            remix_music_db = gr.Slider(-12, 12, 0, step=0.5, label="Music level (dB)")
                        remix_btn = gr.Button("Remix", variant="secondary")
```

- [ ] **Step 4: Extend `generate_meditation`.** Append parameters
  `provider_choice, el_voice, el_stab, el_sim, el_dir` (after
  `use_tts_cache_flag`). Replace the `# Map TTS engine label to key` block with
  `tts_engine = provider_ui.resolve_tts_engine(provider_choice, tts_engine_choice)`
  (add `from core import provider_ui` at the top of the function). Before the
  queue setup, for ElevenLabs validate early:

```python
    if tts_engine == "elevenlabs":
        from core.elevenlabs_tts.client import has_api_key
        from core.elevenlabs_tts.direction import sanitize_direction
        if not has_api_key():
            yield None, _render_status("Error: ELEVENLABS_API_KEY missing", 0.0, "Add it to your .env file")
            return
        try:
            sanitize_direction(el_dir)
        except ValueError as exc:
            yield None, _render_status("Error: invalid delivery direction", 0.0, str(exc))
            return
```

  In the `pipeline.generate(...)` call add
  `elevenlabs_voice_id=el_voice if tts_engine == "elevenlabs" else None,`
  `elevenlabs_stability=float(el_stab) if tts_engine == "elevenlabs" else None,`
  `elevenlabs_similarity=float(el_sim) if tts_engine == "elevenlabs" else None,`
  `elevenlabs_direction=el_dir if tts_engine == "elevenlabs" else None,`.
  Final status: replace the `tts_label` line with

```python
    tts_label = {"f5": "F5-TTS", "kokoro": "Kokoro", "chatterbox": "Chatterbox",
                 "elevenlabs": "ElevenLabs v4"}.get(tts_engine, tts_engine)
    manifest = pipeline.last_tts_manifest or {}
    if tts_engine == "elevenlabs" and manifest:
        tts_label += (f" ({manifest.get('chars_billed', 0):,} chars billed, "
                      f"{manifest.get('cache_hits', 0)} chunks cached)")
```

- [ ] **Step 5: Wire events** (inside the Manual tab, after the existing
  callbacks):

```python
            def _apply_provider(provider, mode, local_engine):
                vis = provider_ui.provider_visibility(provider, mode)
                show_kokoro = (not provider_ui.is_cloud(provider)) and local_engine == "Kokoro" \
                    and mode != "Instrumental Only"
                show_f5 = vis["f5_settings"] and local_engine != "Kokoro"
                updates = [
                    gr.update(visible=vis["tts_engine_radio"]),
                    gr.update(visible=show_kokoro),
                    gr.update(visible=show_f5),
                    gr.update(visible=vis["speed_slider"]),
                    gr.update(visible=vis["f5_cfg_slider"]),
                    gr.update(visible=vis["microprosody_checkbox"]),
                    gr.update(visible=vis["elevenlabs_settings"]),
                ]
                if provider_ui.is_cloud(provider):
                    from core.elevenlabs_tts.client import has_api_key
                    validation = provider_ui.validate_account_voices()
                    updates += [gr.update(value=0.0),
                                gr.update(choices=provider_ui.voice_choices(validation)),
                                gr.update(value=provider_ui.api_key_banner()),
                                gr.update(interactive=has_api_key())]
                else:
                    updates += [gr.update(), gr.update(), gr.update(), gr.update(interactive=True)]
                return updates

            provider_radio.change(
                fn=_apply_provider,
                inputs=[provider_radio, generation_mode, tts_engine_radio],
                outputs=[tts_engine_radio, kokoro_settings, f5_settings, speed_slider,
                         f5_cfg_slider, microprosody_checkbox, elevenlabs_settings,
                         reverb_slider, el_voice_dropdown, el_banner, generate_btn],
            )

            _cost_inputs = [script_input, content_type_dropdown, el_direction, el_voice_dropdown,
                            el_stability, el_similarity]

            def _cost(script, ct_label, direction, voice_id, stab, sim):
                from core.content_profiles import normalize_content_type
                return provider_ui.cost_line(script, normalize_content_type(ct_label), direction,
                                             voice_id, stab, sim)

            for comp in _cost_inputs:
                comp.change(fn=_cost, inputs=_cost_inputs, outputs=[el_cost])

            def _after_generate():
                run_id = pipeline.last_run_id
                return run_id, gr.update(visible=bool(run_id))

            def _remix(run_id, voice_db, music_db):
                path, message = provider_ui.remix_ui(run_id, voice_db, music_db)
                if path is None:
                    return gr.update(), _render_status("Remix failed", 0.0, message)
                return path, _render_status("Remix complete", 1.0, message)

            remix_btn.click(
                fn=_remix,
                inputs=[run_id_state, remix_voice_db, remix_music_db],
                outputs=[audio_output, status_display],
            )

            provider_radio.change(
                None, inputs=[provider_radio], outputs=None,
                js="(p) => { try { localStorage.setItem('moodscape_provider', p); } catch (e) {} return p; }",
            )
```

  Change the existing `generate_btn.click(...)` inputs to append
  `provider_radio, el_voice_dropdown, el_stability, el_similarity, el_direction`
  (in that order, matching the new parameters), and chain
  `.then(fn=_after_generate, outputs=[run_id_state, remix_group])`.
  Also make `toggle_mode_settings` and `toggle_tts_engine_ui` respect the
  provider: add `provider_radio` as their last input and, when
  `provider_ui.is_cloud(provider)`, return `visible=False` for
  `kokoro_settings`, `speed_slider`, `f5_settings` (keep all other outputs as
  today).

  After the `with gr.Tabs():` block closes (still inside `gr.Blocks`), restore
  the saved provider on load:

```python
    demo.load(
        None, inputs=None, outputs=[provider_radio],
        js="() => { try { return localStorage.getItem('moodscape_provider') || 'Local Models'; } "
           "catch (e) { return 'Local Models'; } }",
    )
```

- [ ] **Step 6: Verify compile and suite**

Run: `.venv/bin/python -m py_compile app.py && .venv/bin/python -m pytest tests/unit/ -q`
Expected: no output from py_compile; suite green. (Browser verification is
the controller's Task 19.)

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat(ui): Voice Provider toggle, ElevenLabs controls, cost line and Remix panel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 16: Auto-Generate tab provider support

**Files:**
- Modify: `core/auto_tab.py`
- Modify: `app.py` (`build_auto_tab()` call → `build_auto_tab(provider=provider_radio)`)
- Test: `tests/unit/test_auto_tab.py` (append)

**Interfaces:**
- Consumes: Task 14 `is_cloud`; Task 13 meta `"elevenlabs"`.
- Produces: `build_auto_tab(provider=None) -> dict`;
  `auto_generate_handler(genre, band, steer, content_type, tts_engine, planner_spec, generator_spec, judge_spec, provider="Local Models")`;
  `billing_note(meta_path: str) -> str`.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_handler_uses_elevenlabs_when_provider_is_cloud(monkeypatch):
    from core import auto_tab
    seen = {}

    class _Run:
        result, error, invalid_input = None, "stop", True
        def __init__(self, *a, config=None, **k):
            seen["engine"] = config.tts_engine
        def __iter__(self):
            return iter(())

    monkeypatch.setattr(auto_tab, "StreamingRun", _Run)
    list(auto_tab.auto_generate_handler("stress_relief", "short", "", "meditation", "f5",
                                        "", "", "", provider="ElevenLabs"))
    assert seen["engine"] == "elevenlabs"
    list(auto_tab.auto_generate_handler("stress_relief", "short", "", "meditation", "f5",
                                        "", "", "", provider="Local Models"))
    assert seen["engine"] == "f5"


def test_billing_note(tmp_path):
    import json
    from core.auto_tab import billing_note
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"elevenlabs": {"chars_billed": 1234, "cache_hits": 2}}))
    assert billing_note(str(p)) == "ElevenLabs: 1,234 characters billed (2 chunks cached)."
    p.write_text("{}")
    assert billing_note(str(p)) == ""
```

- [ ] **Step 2: Run to verify failure** — FAIL (`unexpected keyword 'provider'`).

- [ ] **Step 3: Implement**

`auto_generate_handler`: add `provider="Local Models"` as the last parameter;
before building the config:

```python
    from core.provider_ui import is_cloud
    if is_cloud(provider):
        tts_engine = "elevenlabs"
```

Add:

```python
def billing_note(meta_path: str) -> str:
    import json
    try:
        el = json.loads(open(meta_path, encoding="utf-8").read()).get("elevenlabs")
    except (OSError, ValueError):
        return ""
    if not el:
        return ""
    return (f"ElevenLabs: {el.get('chars_billed', 0):,} characters billed "
            f"({el.get('cache_hits', 0)} chunks cached).")
```

and in the success status append `billing_note(result.meta_path)` on a new line
when non-empty (`AutoResult` already carries `meta_path`).

`build_auto_tab(provider=None)`: when `provider` is given, include it as the
last input of `button.click`, and add

```python
        if provider is not None:
            from core.provider_ui import is_cloud
            provider.change(fn=lambda p: gr.update(visible=not is_cloud(p)),
                            inputs=[provider], outputs=[tts_engine])
```

In `app.py` change `build_auto_tab()` to `build_auto_tab(provider=provider_radio)`.

- [ ] **Step 4: Run tests** — `tests/unit/test_auto_tab.py` PASS; `py_compile app.py` OK; suite green.

- [ ] **Step 5: Commit**

```bash
git add core/auto_tab.py app.py tests/unit/test_auto_tab.py
git commit -m "feat(auto_tab): follow the Voice Provider toggle; show ElevenLabs billing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Phase 6 gate (controller)

- [ ] Full suite green; `py_compile app.py`. Phase reviewer on Phase 6 diff,
  focus: every `generate_btn` input matches the handler's parameter order; Local
  Models UI unchanged by default; no API call on page load without a key;
  localStorage access wrapped in try/catch.

---

## Phase 7 — Docs

### Task 17: Documentation

**Files:**
- Modify: `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/GOTCHAS.md`,
  `docs/COMPONENT_REGISTRY.md`, `docs/TASK_ROUTING.md`,
  `docs/setup_and_execution/checkpoint_locations.md`,
  `docs/optimization_and_processing/post-processing-pipeline.md`,
  `docs/optimization_and_processing/audio_processing.md`
- Create: `docs/model_implementation_guides/elevenlabs_v4.md`

- [ ] **Step 1: Write `docs/model_implementation_guides/elevenlabs_v4.md`** —
  sections: *What v4 is* (spec §2 bullets, with source links), *How MoodScape
  uses it* (chunking, direction, stitching, caches, 48 kHz, clean chain),
  *Tuning knobs* (the env table from spec §8), *Known limits* (no speed; tags
  best-effort; clones may need v4 retraining; per-character billing).
- [ ] **Step 2: Update `CLAUDE.md`:**
  - Intro paragraph: "Two TTS engines" → "local TTS engines (Kokoro, F5-TTS,
    Chatterbox) plus a cloud provider (ElevenLabs v4) selected by a top-of-app
    Voice Provider toggle".
  - Setup: `.env` must define `HF_TOKEN`, `GOOGLE_API_KEY`, and (for the
    ElevenLabs provider) `ELEVENLABS_API_KEY`.
  - Folder map: add `core/elevenlabs_tts/`, `core/mix_session.py`,
    `core/remix.py`, `core/provider_ui.py`, `assets/elevenlabs/voices.toml`;
    remove `breath_sounds.py`, `assets/breath_sounds/`, `generate_breath_samples.py`.
  - Pipeline flow step 2/6: "TTS synth → 24 kHz mono (local) or 48 kHz
    (ElevenLabs, no upsample/DeepFilterNet)"; step 9: "saves a mix session
    (`var/mix_sessions/`) for Remix".
  - Top Gotchas: add **No vocal sounds** (breath markers render as silence in
    every engine; ElevenLabs direction rejects sound words) and **Eleven v4 has
    no speed/style** (pacing = direction + punctuation + programmatic silence;
    UI hides Speed for ElevenLabs).
  - TTS cache gotcha: prefixes now include `MOODSCAPE_ELEVENLABS_*`.
  - Flag table: add rows for every `MOODSCAPE_ELEVENLABS_*` var and
    `MOODSCAPE_MIX_SESSIONS_KEEP` (defaults from Global Constraints).
- [ ] **Step 3: Update the other docs** — ARCHITECTURE (engine list, 48 kHz
  path, Remix), GOTCHAS (same two gotchas), COMPONENT_REGISTRY (new modules),
  TASK_ROUTING ("Change ElevenLabs voices → assets/elevenlabs/voices.toml";
  "Remix → core/remix.py"), and remove breath-sample references from the
  checkpoint/post-processing docs.
- [ ] **Step 4: Verify no stale references**

Run: `grep -rn "breath_sounds\|generate_breath_samples\|load_breath" --include='*.md' . | grep -v docs/superpowers`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs
git commit -m "docs: ElevenLabs v4 provider, remix, and no-vocal-sounds rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase 8 — Verification and final review (controller)

### Task 18 (controller): Integration test file (key-gated)

**Files:**
- Create: `tests/integration/test_elevenlabs_live.py`

- [ ] **Step 1: Write it**

```python
"""Live ElevenLabs render (bills ~150 characters). Skipped without a key."""

import os

import pytest
import soundfile as sf

pytestmark = pytest.mark.skipif(not os.environ.get("ELEVENLABS_API_KEY"),
                                reason="ELEVENLABS_API_KEY not set")


def test_one_paragraph_full_pipeline():
    from core.pipeline import MeditationPipeline
    pipe = MeditationPipeline()
    bg = "assets/backgrounds/freemusicforvideo-meditation-background-409198.mp3"
    out, _ = pipe.generate(script="Let your shoulders soften. [pause:2s]\n\nRest here.",
                           music_prompt="", tts_engine="elevenlabs", music_model="upload",
                           uploaded_music_path=bg)
    audio, sr = sf.read(out)
    assert sr == 48000 and len(audio) > sr * 5
    m = pipe.last_tts_manifest
    assert m["model_id"] == "eleven_v4" and m["chars_billed"] + m["chars_cached"] > 0
```

- [ ] **Step 2: Run** (needs the user's key in `.env`; `app.py`-style dotenv is
  not loaded by pytest, so export it for the run):

```bash
set -a; source .env; set +a; .venv/bin/python -m pytest tests/integration/test_elevenlabs_live.py -v
```

Expected: PASS (or SKIP with the reason recorded in the ledger if no key yet).

- [ ] **Step 3: Commit** — `git add tests/integration/test_elevenlabs_live.py && git commit -m "test: key-gated live ElevenLabs integration render" ...` (with the Co-Authored-By trailer).

### Task 19 (controller): Smoke, golden-path regression, browser check

- [ ] **Step 1: Golden path regression** — re-run the Task 0 Step 4 command
  writing `after.wav` (TTS cache hit expected), then:

```bash
.venv/bin/python - <<'EOF'
import numpy as np, pyloudnorm as pyln, soundfile as sf
a, sr = sf.read("var/eval/e11-baseline/before.wav"); b, _ = sf.read("var/eval/e11-baseline/after.wav")
m = pyln.Meter(sr)
print("LUFS before/after:", m.integrated_loudness(a), m.integrated_loudness(b))
n = min(len(a), len(b)); print("max abs diff:", float(np.max(np.abs(a[:n] - b[:n]))), "len", len(a), len(b))
EOF
```

Expected: identical lengths, |ΔLUFS| ≤ 0.1, max abs diff ≈ 0 (the local path is
unchanged and the script has no breath markers). Anything larger is a
regression — fix before proceeding.

- [ ] **Step 2: ElevenLabs smoke** (key required): one ~3-minute meditation via
  `MeditationPipeline.generate(tts_engine="elevenlabs", …)` with Brittney and
  one with Serena; record chars billed and output paths in the ledger; then
  `remix(run_id, +4, 0)` and `remix(run_id, 0, -4)` on one of them.
- [ ] **Step 3: Auto-Generate smoke** — one `stress_relief` short run with
  `AutoConfig.from_genre(..., tts_engine="elevenlabs")`; confirm meta
  `"elevenlabs"` block and `estimate_ratio`.
- [ ] **Step 4: Browser check** — start the app (`.claude/launch.json` entry
  `python app.py` on port 7860 via `preview_start`), then verify: default
  provider is Local Models with today's controls; switching to ElevenLabs hides
  Voice Engine/Speed/F5 controls and shows the ElevenLabs group with a cost
  line; a bad direction (`[sighs]`) shows the inline error; generate a short
  ElevenLabs piece; Remix panel appears and a +3/−3 remix plays; reload keeps
  the provider; the Auto-Generate tab hides its engine dropdown under
  ElevenLabs. Take screenshots for the user.
- [ ] **Step 5: Calibrate WPM** — from the smoke runs'
  `log_estimate_accuracy` lines, if `DEFAULT_WPM["elevenlabs"]` is off by more
  than 10 %, update it in `core/script_gen/duration.py` with the measured
  value and a dated comment; commit `fix(duration): calibrate ElevenLabs WPM from smoke renders`.

### Task 20 (controller): Final whole-branch review → one fix wave → report

- [ ] Dispatch the final code reviewer (most capable model) on
  `git diff a270908..HEAD` with the spec, this plan's Global Constraints and
  Review Focus. One fix dispatch, one scoped re-review, adjudicate residuals
  in the ledger.
- [ ] Hand the user: the listening files from Task 19 (Brittney, Serena, the
  two remixes, the auto-gen output), measured chars billed, the golden-path
  comparison numbers, and screenshots. Listening judgement is theirs.
- [ ] Then use superpowers:finishing-a-development-branch (never push without
  the user's explicit request).
