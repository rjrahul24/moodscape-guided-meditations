# R1 Script-Generation Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the writer and judge consistent instructions, model-card sampling and stronger deterministic checks, and give each genre family a suitable voice.

**Architecture:**
- Small new modules in `core/script_gen/` (`sampling`, `slop`, `guide_context`) plus `core/voice_selection.py`. Edits to the existing linter, rules, adapters and orchestrator.
- Prompting guides become templates rendered from code constants at load time.
- Genre packs gain two optional fields enforced by a pack-scoped linter entry point.
- The audio path is untouched.

**Tech Stack:** Python 3.11, httpx (OpenAI-compatible adapter), anthropic SDK, librosa (pyin, spectral_centroid), soundfile, TOML packs, unittest-style tests run by pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-r1-script-gen-fixes-design.md`

## Global Constraints

- No change to the audio path: `core/pipeline.py`, `core/mixer.py`, FX chains and TTS engines. R1 only passes an existing `f5_voice_slug` argument through.
- Keep the FATAL/ADVISORY split, writer/judge independence, `<problems>` stripping in `parse_judge_response`, and `unload()` after each LLM stage, all unchanged.
- ADVISORY violations never trigger a repair call.
- Guides stay `.md` files read at call time. `{{NAME}}` placeholders are rendered from `guide_context(engine, content_type)`. An unknown placeholder raises `KeyError` naming it, and a leftover `{{` after rendering raises `ValueError`.
- Sampling table:
  - qwen3* planner/writer: `temperature=0.7, top_p=0.80, presence_penalty=1.5`.
  - qwen3* judge/repair: `0.7, 0.80, 0.0`.
  - gemma4* any stage: `1.0, 0.95, None`.
  - Other models: `1.0, None, None`.
  - Env overrides are `MOODSCAPE_SAMPLING_<STAGE>_<TEMPERATURE|TOP_P|PRESENCE_PENALTY>`.
- `SLOP_DENSITY_MAX = 2.0` hits per 150 words.
- New FATAL codes: `RAPID_BREATHING`, `PAIN_ELIMINATION`, `ENERGY_HEALING_CLAIM`, `MISSING_DISCLAIMER`.
- New ADVISORY codes: `SLOP_DENSITY`, `SLOP_HARD`.
- Pack fields:
  - `max_breath_hold_sec`: int, 1–7, default 7. Set to 4 in panic_sos, anxiety_relief and overwhelm.
  - `requires_disclaimer`: bool, default false. Set true on box_breathing, breath_awareness, calming_breath_4_7_8, coherent_breathing and energising_breath.
  - `banned`: limit raised to 3–10, enforced by `load_pack`.
- Sleep-story `pause_ratio` = 0.20 for deep_sleep, fall_asleep, back_to_sleep and power_nap.
- Voice profiles:
  - `calm`: Sleep & Rest, Stress & Anxiety.
  - `energy`: Morning & Energy, Body & Movement.
  - `expressive`: Emotional, Presence.
  - `steady`: Focus & Work, Breathwork.
  - Unknown family → `steady`.
  - Override file `assets/speakers/voice_map.toml`; cache `var/voice_traits.json`.
- Tests are `unittest.TestCase` classes, run with `.venv/bin/python -m pytest`. No real LLM, TTS or Whisper calls in unit tests.
- Commits are conventional and end with a `Co-Authored-By:` trailer naming the model that wrote them. Commit to `dev`. Never push.
- Never load real models in this plan. If any step seems to need one, stop and report.

## Review Focus

1. **A curly apostrophe or a `[pause:Xs]` tag inside a new safety phrase** ("don’t breathe as fast as you can", "breathe into the pain [pause:3s] until it disappears") must still be caught. *Pinned in Task 4* (`test_tags_and_curly_quotes_do_not_defeat_new_rules`).
2. **A legitimate Kapalabhati script for `energising_breath`** ("a short, sharp exhale through the nose … twenty gentle pumps") must NOT trip `RAPID_BREATHING`. *Pinned in Task 4* (`test_kapalabhati_phrasing_allowed`).
3. **`voice_map.toml` naming a voice that doesn't exist** must fail loudly at config build, not silently render the wrong voice. *Pinned in Task 8* (`test_override_unknown_slug_raises`).
4. **A model spec with a path-style name** (`openrouter:qwen/qwen3-32b`) or an unknown provider must still resolve sampling by family, or fall back. *Pinned in Task 1* (`test_family_from_path_style_name`).
5. **A guide edited later with a typo'd placeholder** (`{{CHUNK_CHAR}}`) must fail at prompt build, before any LLM call. *Pinned in Task 6* (`test_unknown_placeholder_raises_before_llm`).

---

### Task 1: Sampling parameters and adapter plumbing

**Files:**
- Create: `core/script_gen/sampling.py`
- Modify: `core/script_gen/engine.py` (`ScriptEngine.complete`, `FakeScriptEngine.complete`)
- Modify: `core/script_gen/adapters/openai_compat.py` (`complete`)
- Modify: `core/script_gen/adapters/anthropic_api.py` (`complete`)
- Modify: `core/script_gen/__init__.py` (export `SamplingParams`, `sampling_for`)
- Test: `tests/unit/test_script_sampling.py`

**Interfaces:**
- Produces:
  - `STAGES = ("planner", "writer", "judge", "repair")`.
  - `@dataclass(frozen=True) class SamplingParams: temperature: float; top_p: float | None = None; presence_penalty: float | None = None`.
  - `SamplingParams.as_dict() -> dict` (all three keys).
  - `sampling_for(stage: str, engine_name: str) -> SamplingParams`. `engine_name` is a `provider:model` string as returned by `ScriptEngine.name`. It raises `ValueError` naming the env var on a malformed override, and `ValueError` on an unknown stage.
  - `ScriptEngine.complete(system, user, *, max_tokens=4096, temperature=1.0, sampling: SamplingParams | None = None) -> str`.
  - `FakeScriptEngine.calls[i]["sampling"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_script_sampling.py
import json
import os
import unittest
from unittest.mock import patch

import httpx

from core.script_gen.adapters.openai_compat import OpenAICompatEngine
from core.script_gen.engine import FakeScriptEngine
from core.script_gen.sampling import SamplingParams, sampling_for


def _clear_env():
    return patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                   if not k.startswith("MOODSCAPE_SAMPLING_")}, clear=True)


class TestSamplingFor(unittest.TestCase):
    def test_qwen_writer_uses_card_non_thinking_values(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 1.5))
            self.assertEqual(sampling_for("planner", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 1.5))

    def test_qwen_as_judge_has_no_presence_penalty(self):
        with _clear_env():
            self.assertEqual(sampling_for("judge", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 0.0))
            self.assertEqual(sampling_for("repair", "ollama:qwen3.8:27b"), SamplingParams(0.7, 0.80, 0.0))

    def test_gemma_any_stage(self):
        with _clear_env():
            for stage in ("planner", "writer", "judge", "repair"):
                self.assertEqual(sampling_for(stage, "ollama:gemma4:31b"), SamplingParams(1.0, 0.95, None))

    def test_unknown_model_keeps_todays_behaviour(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "ollama:llama3.2:3b"), SamplingParams(1.0, None, None))

    def test_family_from_path_style_name(self):
        with _clear_env():
            self.assertEqual(sampling_for("writer", "openrouter:qwen/qwen3-32b"), SamplingParams(0.7, 0.80, 1.5))
            self.assertEqual(sampling_for("writer", "weird"), SamplingParams(1.0, None, None))

    def test_env_override_and_none(self):
        env = {"MOODSCAPE_SAMPLING_WRITER_PRESENCE_PENALTY": "0.5",
               "MOODSCAPE_SAMPLING_WRITER_TOP_P": "none"}
        with _clear_env(), patch.dict(os.environ, env):
            self.assertEqual(sampling_for("writer", "ollama:qwen3.8:27b"), SamplingParams(0.7, None, 0.5))

    def test_bad_env_names_variable(self):
        with _clear_env(), patch.dict(os.environ, {"MOODSCAPE_SAMPLING_JUDGE_TEMPERATURE": "hot"}):
            with self.assertRaisesRegex(ValueError, "MOODSCAPE_SAMPLING_JUDGE_TEMPERATURE"):
                sampling_for("judge", "ollama:gemma4:31b")

    def test_unknown_stage(self):
        with self.assertRaisesRegex(ValueError, "stage"):
            sampling_for("polish", "ollama:gemma4:31b")

    def test_as_dict(self):
        self.assertEqual(SamplingParams(0.7, 0.8, None).as_dict(),
                         {"temperature": 0.7, "top_p": 0.8, "presence_penalty": None})


class TestAdapters(unittest.TestCase):
    def _engine(self, captured):
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return OpenAICompatEngine("ollama", "qwen3.8:27b", "http://x/v1", transport=httpx.MockTransport(handler))

    def test_openai_compat_sends_sampling_fields(self):
        captured = []
        self._engine(captured).complete("s", "u", sampling=SamplingParams(0.7, 0.8, 1.5))
        body = captured[0]
        self.assertEqual((body["temperature"], body["top_p"], body["presence_penalty"]), (0.7, 0.8, 1.5))

    def test_openai_compat_omits_none_fields(self):
        captured = []
        self._engine(captured).complete("s", "u", sampling=SamplingParams(1.0))
        self.assertNotIn("top_p", captured[0])
        self.assertNotIn("presence_penalty", captured[0])

    def test_openai_compat_without_sampling_unchanged(self):
        captured = []
        self._engine(captured).complete("s", "u", temperature=1.0)
        self.assertEqual(captured[0]["temperature"], 1.0)
        self.assertNotIn("top_p", captured[0])

    def test_fake_engine_records_sampling(self):
        fake = FakeScriptEngine(["x"])
        fake.complete("s", "u", sampling=SamplingParams(0.7))
        self.assertEqual(fake.calls[0]["sampling"], SamplingParams(0.7))


if __name__ == "__main__":
    unittest.main()
```

Also add to `tests/unit/test_anthropic_adapter.py` a test asserting that `sampling` is accepted and **not** forwarded. Follow that file's existing client-mocking pattern: the `messages.stream` call's kwargs must contain no `temperature`, `top_p` or `presence_penalty` when `sampling=SamplingParams(0.7, 0.8, 1.5)` is passed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_script_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.script_gen.sampling'`

- [ ] **Step 3: Implement `sampling.py`**

```python
# core/script_gen/sampling.py
"""Per-stage sampling parameters for script-generation model calls.

Values come from the model cards (Qwen3.x non-thinking: 0.7 / 0.80 / presence
1.5; Gemma 4: 1.0 / 0.95). top_k and min_p are deliberately absent: the
installed Ollama Modelfiles already set them (qwen top_k 20, gemma top_k 64),
and the OpenAI-compatible /v1 endpoint cannot override them anyway.
"""

import os
from dataclasses import asdict, dataclass

STAGES = ("planner", "writer", "judge", "repair")


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float | None = None
    presence_penalty: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = SamplingParams(1.0)

# (family prefix, stages, params) — first match wins.
_TABLE: tuple[tuple[str, frozenset[str], SamplingParams], ...] = (
    ("qwen3", frozenset({"planner", "writer"}), SamplingParams(0.7, 0.80, 1.5)),
    ("qwen3", frozenset({"judge", "repair"}), SamplingParams(0.7, 0.80, 0.0)),
    ("gemma4", frozenset(STAGES), SamplingParams(1.0, 0.95, None)),
)

_FIELDS = ("temperature", "top_p", "presence_penalty")


def _bare_model(engine_name: str) -> str:
    model = engine_name.split(":", 1)[1] if ":" in engine_name else engine_name
    return model.rsplit("/", 1)[-1].lower()


def _env_value(var: str, current):
    raw = os.environ.get(var)
    if raw is None:
        return current
    text = raw.strip().lower()
    if text in ("", "none", "null"):
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Environment variable {var}={raw!r} is not a number (or 'none').") from exc


def sampling_for(stage: str, engine_name: str) -> SamplingParams:
    """Sampling for one pipeline stage of one model, with env overrides."""
    if stage not in STAGES:
        raise ValueError(f"Unknown sampling stage {stage!r}; expected one of {STAGES}.")
    model = _bare_model(engine_name)
    params = DEFAULT
    for prefix, stages, candidate in _TABLE:
        if model.startswith(prefix) and stage in stages:
            params = candidate
            break
    values = params.as_dict()
    for field in _FIELDS:
        values[field] = _env_value(f"MOODSCAPE_SAMPLING_{stage.upper()}_{field.upper()}", values[field])
    if values["temperature"] is None:
        values["temperature"] = DEFAULT.temperature
    return SamplingParams(**values)
```

- [ ] **Step 4: Plumb `sampling` through the engines**

In `core/script_gen/engine.py`:
- Add `from core.script_gen.sampling import SamplingParams` at the top. `sampling.py` imports nothing from `engine`, so there is no cycle.
- Add `sampling: SamplingParams | None = None,` after `temperature` in both `ScriptEngine.complete` (abstract, and docstring: "When given, overrides temperature and adds top_p/presence_penalty where the provider supports them.") and `FakeScriptEngine.complete`.
- In `FakeScriptEngine.complete`, add `"sampling": sampling,` to the recorded dict.

In `core/script_gen/adapters/openai_compat.py` `complete`:
- Add the same parameter.
- After `payload` is built:

```python
        if sampling is not None:
            payload["temperature"] = sampling.temperature
            if sampling.top_p is not None:
                payload["top_p"] = sampling.top_p
            if sampling.presence_penalty is not None:
                payload["presence_penalty"] = sampling.presence_penalty
```

In `core/script_gen/adapters/anthropic_api.py` `complete`:
- Add the same parameter.
- Extend the existing comment: `# sampling (temperature/top_p/presence_penalty) is deliberately not forwarded: thinking-enabled models reject sampling parameters.` Don't forward it.

Export `SamplingParams` and `sampling_for` from `core/script_gen/__init__.py` (both the import and `__all__`).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_script_sampling.py tests/unit/test_anthropic_adapter.py tests/unit/test_openai_compat_adapter.py tests/unit/test_script_engine.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add core/script_gen/sampling.py core/script_gen/engine.py core/script_gen/adapters/ core/script_gen/__init__.py tests/unit/test_script_sampling.py tests/unit/test_anthropic_adapter.py
git commit -m "feat(script_gen): add per-stage model-card sampling and adapter plumbing"
```

---

### Task 2: Per-stage sampling in the generation loop

**Files:**
- Modify: `core/script_gen/planner.py` (`plan`), `core/script_gen/generator.py` (`draft`), `core/script_gen/judge.py` (`review`, `repair`)
- Modify: `core/auto_generate.py` (`generate_script`, `ScriptOutcome`, `run()` meta)
- Test: `tests/unit/test_auto_generate.py` and `tests/unit/test_generator_judge.py` (add tests)

**Interfaces:**
- Consumes: `sampling_for(stage, engine_name)` and `SamplingParams` (Task 1); `FakeScriptEngine.calls[i]["sampling"]`.
- Produces:
  - `plan(..., sampling: SamplingParams | None = None)`.
  - `draft(engine, prompt, system, *, max_tokens=4096, sampling=None)`.
  - `review(engine, draft_script, system, *, max_tokens=4096, sampling=None)`.
  - `repair(engine, script, violations, system, *, max_tokens=4096, sampling=None)`.
  - `ScriptOutcome.sampling: dict` (stage → `as_dict()`).
  - `meta.json["sampling"]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_generator_judge.py`:

```python
class TestSamplingForwarding(unittest.TestCase):
    def test_draft_review_repair_forward_sampling(self):
        from core.script_gen.engine import FakeScriptEngine
        from core.script_gen.generator import draft
        from core.script_gen.judge import repair, review
        from core.script_gen.linter import Violation
        from core.script_gen.sampling import SamplingParams

        s = SamplingParams(0.7, 0.8, 1.5)
        eng = FakeScriptEngine(["Rest here.", "<script>Rest.</script><changelog>-</changelog>"])
        draft(eng, "p", "sys", sampling=s)
        review(eng, "Rest here.", "sys", sampling=s)
        repair(eng, "Rest.", [Violation("X", "fatal", "m")], "sys", sampling=s)
        self.assertEqual([c["sampling"] for c in eng.calls], [s, s, s])
```

Add to `tests/unit/test_auto_generate.py`. Reuse that file's existing fake-engine and fake-pipeline helpers and its genre-path test setup, and follow the existing successful genre-run test:

```python
    def test_generate_script_uses_per_stage_sampling(self):
        # Build planner/generator/judge as FakeScriptEngine instances whose .name
        # is patched to "ollama:qwen3.8:27b" (planner+generator) and "ollama:gemma4:31b" (judge),
        # e.g. via unittest.mock.patch.object(type(engine), "name", new_callable=PropertyMock).
        # Run generate_script on the genre path (pack + angle given) with a clean script.
        # Assert:
        #   planner call sampling == SamplingParams(0.7, 0.80, 1.5)
        #   writer  call sampling == SamplingParams(0.7, 0.80, 1.5)
        #   judge   call sampling == SamplingParams(1.0, 0.95, None)
        #   outcome.sampling == {"planner": {...}, "writer": {...}, "judge": {...}, "repair": {...}}
```

Write this test fully, using the concrete fixtures already in that file (their names are visible there). It must assert the three exact `SamplingParams` values above, and that `outcome.sampling` has the four stage keys.

Add one test that a malformed `MOODSCAPE_SAMPLING_WRITER_TEMPERATURE` makes `generate_script` raise `ScriptGenerationError` naming the variable **before any engine call** (all fakes' `calls == []`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_generator_judge.py tests/unit/test_auto_generate.py -v -k "sampling or Sampling"`
Expected: FAIL (`TypeError: unexpected keyword argument 'sampling'`)

- [ ] **Step 3: Implement**

`planner.plan`, `generator.draft`, `judge.review` and `judge.repair` each gain a `sampling: SamplingParams | None = None` keyword (import it from `core.script_gen.sampling`) and pass `sampling=sampling` to `engine.complete(...)`.

In `core/auto_generate.py`:
- Import `sampling_for`.
- Add `sampling: dict = field(default_factory=dict)` to `ScriptOutcome` (import `field` from dataclasses; it's already imported).
- At the top of `generate_script`, before the `try`:

```python
    try:
        stage_sampling = {
            "planner": sampling_for("planner", planner_engine.name) if planner_engine is not None else None,
            "writer": sampling_for("writer", generator_engine.name),
            "judge": sampling_for("judge", judge_engine.name),
            "repair": sampling_for("repair", judge_engine.name),
        }
    except ValueError as exc:
        raise ScriptGenerationError(str(exc)) from exc
```

- Pass `sampling=stage_sampling["planner"]` to `plan(...)`, `sampling=stage_sampling["writer"]` to `draft(...)`, `sampling=stage_sampling["judge"]` to `review(...)`, and `sampling=stage_sampling["repair"]` to `repair(...)`.
- In the successful `ScriptOutcome(...)`, add `sampling={k: v.as_dict() for k, v in stage_sampling.items() if v is not None}`.
- In `run()`'s `meta.json` dict, add `"sampling": outcome.sampling,`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_generator_judge.py tests/unit/test_auto_generate.py tests/unit/test_planner.py -v`
Expected: PASS (existing plus new)

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/planner.py core/script_gen/generator.py core/script_gen/judge.py core/auto_generate.py tests/unit/test_generator_judge.py tests/unit/test_auto_generate.py
git commit -m "feat(auto_generate): apply per-stage sampling and record it in meta.json"
```

---

### Task 3: Slop lexicon in script_gen and the ADVISORY slop check

**Files:**
- Create: `core/script_gen/slop.py` (moved from `core/eval/slop.py`)
- Modify: `core/eval/slop.py` (becomes a re-export)
- Modify: `core/script_gen/linter.py` (`check_slop`, called from `check`; `format_for_repair` style section)
- Modify: `core/auto_generate.py` (repair includes slop advisories as low-priority style items)
- Test: `tests/unit/test_script_linter.py` (add), `tests/unit/test_auto_generate.py` (add)

**Interfaces:**
- Consumes: the existing `core/eval/slop.py` contents (`SLOP_HARD`, `SLOP_RATE_LIMITED`, `SLOP_PATTERNS`, `SlopHit`, `slop_hits`, `slop_density`).
- Produces:
  - `core.script_gen.slop`: the same public names, plus `slop_guidance(top_n: int = 20) -> str`.
  - `linter.SLOP_DENSITY_MAX = 2.0`.
  - `linter.check_slop(script) -> list[Violation]`, with codes `SLOP_DENSITY` and `SLOP_HARD` (both ADVISORY).
  - `format_for_repair(violations)` renders FATAL items numbered first, then (if any ADVISORY slop items) a line `Style (lower priority, fix only while making the changes above):` followed by those items.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_script_linter.py`:

```python
class TestSlopChecks(unittest.TestCase):
    def test_dense_slop_is_advisory(self):
        from core.script_gen.linter import ADVISORY, check_slop
        text = "Gently, softly, slowly let go and drift. " * 6
        v = [x for x in check_slop(text) if x.code == "SLOP_DENSITY"]
        self.assertEqual(len(v), 1)
        self.assertEqual(v[0].severity, ADVISORY)

    def test_hard_tier_phrase_reported_once_per_phrase(self):
        from core.script_gen.linter import check_slop
        text = "Let it wash over you. Again let it wash over you. A tapestry of light."
        codes = [(x.code, x.message) for x in check_slop(text) if x.code == "SLOP_HARD"]
        self.assertEqual(len(codes), 2)

    def test_clean_script_has_no_slop(self):
        from core.script_gen.linter import check_slop
        self.assertEqual(check_slop("Notice the weight of your hands on your knees. " * 5), [])

    def test_check_includes_slop(self):
        from core.script_gen.linter import check
        self.assertIn("SLOP_HARD", [v.code for v in check("A tapestry of calm.")])

    def test_repair_text_orders_fatal_before_style(self):
        from core.script_gen.linter import ADVISORY, FATAL, Violation, format_for_repair
        text = format_for_repair([Violation("SLOP_HARD", ADVISORY, "style msg"),
                                  Violation("BREATH_HOLD", FATAL, "fatal msg")])
        self.assertLess(text.index("fatal msg"), text.index("Style (lower priority"))
        self.assertLess(text.index("Style (lower priority"), text.index("style msg"))


class TestSlopModuleLocation(unittest.TestCase):
    def test_eval_reexports_script_gen_slop(self):
        import core.eval.slop as e
        import core.script_gen.slop as s
        self.assertIs(e.slop_hits, s.slop_hits)

    def test_guidance_lists_items(self):
        from core.script_gen.slop import slop_guidance
        g = slop_guidance(5)
        self.assertIn("Avoid:", g)
        self.assertEqual(len(g.split("Avoid:")[1].split(",")), 5)
```

Add to `tests/unit/test_auto_generate.py` (using that file's fixtures): a run whose judge output has one FATAL (e.g. an unknown tag) **and** dense slop triggers a repair whose prompt (`judge_engine.calls[-1]["user"]`) contains `Style (lower priority`. A clean-but-sloppy script (ADVISORY only) triggers **no** repair call, i.e. the judge call count stays at 1.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v -k "Slop"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement**
- `git mv core/eval/slop.py core/script_gen/slop.py`. Fix its module docstring to say it lives in script_gen and is shared by the linter and the eval harness.
- Add to it:

```python
def slop_guidance(top_n: int = 20) -> str:
    """One-line writer guidance listing the worst offenders (hard tier first)."""
    items = list(SLOP_HARD) + [p for p in SLOP_RATE_LIMITED if p not in SLOP_HARD]
    return (
        "Prefer concrete, specific nouns and verbs. Use gently, softly or slowly "
        "at most once per 150 words. Avoid: " + ", ".join(items[:top_n])
    )
```

- Recreate `core/eval/slop.py` as:

```python
"""Re-export: the slop lexicon lives in core.script_gen.slop (shared with the linter)."""

from core.script_gen.slop import (  # noqa: F401
    SLOP_HARD, SLOP_PATTERNS, SLOP_RATE_LIMITED, SlopHit, slop_density, slop_hits,
)
```

- In `linter.py`, add near the other constants:

```python
# Slop density above this (hits per 150 words) is reported as ADVISORY.
SLOP_DENSITY_MAX = 2.0
```

and the function:

```python
def check_slop(script: str) -> list[Violation]:
    """Stock meditation phrasing. ADVISORY only: never triggers a repair."""
    from core.script_gen.slop import slop_density, slop_hits

    hits = slop_hits(script)
    violations: list[Violation] = []
    density = slop_density(script)
    if density > SLOP_DENSITY_MAX:
        top = sorted({h.term for h in hits}, key=lambda t: -sum(1 for h in hits if h.term == t))[:5]
        violations.append(Violation(
            code="SLOP_DENSITY", severity=ADVISORY,
            message=(f"Stock phrasing density {density:.1f} per 150 words (limit "
                     f"{SLOP_DENSITY_MAX}). Replace with concrete, specific language: {', '.join(top)}."),
        ))
    for term in sorted({h.term for h in hits if h.tier == "hard"}):
        violations.append(Violation(
            code="SLOP_HARD", severity=ADVISORY,
            message=f"Cliche {term!r}. Say something specific instead.",
        ))
    return violations
```

  Call it at the end of `check()`: `violations += check_slop(script)`, before the duration check, so the order stays deterministic.
- `format_for_repair`: split into `fatal = [v for v in violations if v.severity == FATAL]` and `style = [v for v in violations if v.code in ("SLOP_DENSITY", "SLOP_HARD")]`. Number the fatal lines exactly as today. If `style` is non-empty, append `"\nStyle (lower priority, fix only while making the changes above):"` and `- [CODE] message` lines. Any other violation passed in renders as today (numbered), so existing callers and tests keep working.
- In `core/auto_generate.py`'s repair call, pass `fatal + [v for v in violations if v.code in ("SLOP_DENSITY", "SLOP_HARD")]` instead of `fatal`. The decision to repair is still `if not fatal: return`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py tests/unit/test_auto_generate.py tests/unit/test_eval_slop.py tests/unit/test_eval_script_metrics.py tests/unit/test_generator_judge.py -v`
Expected: PASS. The R0 eval tests must still pass through the re-export.

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/slop.py core/eval/slop.py core/script_gen/linter.py core/auto_generate.py tests/unit/test_script_linter.py tests/unit/test_auto_generate.py
git commit -m "feat(linter): add ADVISORY slop checks and low-priority style items in repairs"
```

---

### Task 4: New global safety hard-blocks

**Files:**
- Modify: `core/script_gen/linter.py` (`_SAFETY_RULES`)
- Modify: `docs/prompting_guides/content_safety_rules.md`
- Test: `tests/unit/test_script_linter.py` (add)

**Interfaces:**
- Produces: FATAL codes `RAPID_BREATHING`, `PAIN_ELIMINATION`, `ENERGY_HEALING_CLAIM`, emitted by `check_safety()` (so by `check()` too).

- [ ] **Step 1: Write the failing tests**

```python
class TestNewSafetyRules(unittest.TestCase):
    def codes(self, text):
        from core.script_gen.linter import check_safety
        return [v.code for v in check_safety(text)]

    def test_rapid_breathing_blocked(self):
        for text in ("Now hyperventilate for a moment.",
                     "Try the breath of fire.",
                     "Breathe in and out as fast as you can.",
                     "Take thirty fast breaths.",
                     "Do 40 rapid exhales."):
            self.assertIn("RAPID_BREATHING", self.codes(text), text)

    def test_kapalabhati_phrasing_allowed(self):
        text = ("Take a short, sharp exhale through the nose, and let the inhale happen "
                "on its own. Twenty gentle pumps, at your own pace.")
        self.assertNotIn("RAPID_BREATHING", self.codes(text))

    def test_pain_elimination_blocked(self):
        for text in ("Breathe into the pain until it disappears.",
                     "We will make the pain go away.",
                     "Your pain will dissolve."):
            self.assertIn("PAIN_ELIMINATION", self.codes(text), text)

    def test_noticing_pain_allowed(self):
        self.assertNotIn("PAIN_ELIMINATION", self.codes("Notice the pain, and let your breath move around it."))

    def test_energy_healing_blocked(self):
        for text in ("Your chakras will heal you.", "This energy cleanses your body.",
                     "Healing vibrations repair the body.", "Feel how vibrations heal."):
            self.assertIn("ENERGY_HEALING_CLAIM", self.codes(text), text)

    def test_energy_word_alone_allowed(self):
        self.assertNotIn("ENERGY_HEALING_CLAIM", self.codes("Notice your energy as you stand."))

    def test_tags_and_curly_quotes_do_not_defeat_new_rules(self):
        self.assertIn("RAPID_BREATHING", self.codes("Don’t stop, breathe as fast as [pause:2s] you can."))
        self.assertIn("PAIN_ELIMINATION", self.codes("Breathe into the pain [pause:3s] until it disappears."))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py -v -k NewSafety`
Expected: FAIL

- [ ] **Step 3: Implement**

Append to `_SAFETY_RULES` (keep the existing entries and their order):

```python
    (
        "RAPID_BREATHING",
        re.compile(
            r"\bhyperventilat\w*"
            r"|\bbreath\s+of\s+fire\b"
            r"|\b(?:breath\w*|inhal\w*|exhal\w*)\b(?:\W+\w+){0,6}?\W+as\s+fast\s+as\s+(?:you\s+)?(?:can|possible)\b"
            r"|\b(?:thirty|forty|fifty|sixty|[3-9]\d|\d{3,})\s+(?:fast|quick|rapid|sharp)\s+(?:breaths|exhales|exhalations)\b",
            re.IGNORECASE,
        ),
        "Rapid or forced breathing. Hyperventilation can trigger panic, dizziness "
        "or seizures. Keep the pace slow and within the listener's own capacity.",
    ),
    (
        "PAIN_ELIMINATION",
        re.compile(
            r"\b(?:breathe|send|move|direct)\b(?:\W+\w+){0,4}?\W+(?:into|toward)\s+(?:the|your)\s+pain\b"
            r"(?:\W+\w+){0,6}?\W+until\s+it\s+(?:disappears|vanishes|goes\s+away|is\s+gone|dissolves)\b"
            r"|\bmake\s+(?:the|your)\s+pain\s+(?:disappear|vanish|go\s+away)\b"
            r"|\b(?:the|your)\s+pain\s+will\s+(?:disappear|vanish|go\s+away|dissolve)\b",
            re.IGNORECASE,
        ),
        "Promises to remove pain. That is a medical outcome claim. Invite the "
        "listener to notice and soften around sensation instead.",
    ),
    (
        "ENERGY_HEALING_CLAIM",
        re.compile(
            r"\b(?:chakras?|energy|energies|vibrations?|frequenc(?:y|ies)|aura)\s+(?:will\s+)?(?:heal|cure|cleanse|repair)s?\b",
            re.IGNORECASE,
        ),
        "Energetic-healing claim. Describe the practice, never a healing power.",
    ),
```

`check_safety` runs on tag-stripped prose, where `_strip_tags` replaces a tag with a space. The `(?:\W+\w+){0,N}?\W+` gaps tolerate the extra whitespace. Confirm with the tests. If `test_tags_and_curly_quotes_do_not_defeat_new_rules` fails because of the spacing, widen `\s+` to `\W+` in the failing rule only.

Then edit `docs/prompting_guides/content_safety_rules.md`. Under "Hard blocks", add rules 6–8, renumbering the "Required qualities" items to follow:
- **No rapid or forced breathing.** Never instruct hyperventilation, "breath of fire", breathing "as fast as you can", or counted bursts of thirty or more fast breaths. Energising practices stay slow-paced and within the listener's capacity.
- **No pain-elimination promises.** Never say breathing will make pain disappear. Invite noticing and softening around sensation.
- **No energetic-healing claims.** Never claim chakras, energy, vibrations or frequencies heal, cure or cleanse.
- Under required qualities, add: **Breathwork disclaimer.** In breathwork sessions, say once in the opening: if you are pregnant, have a heart or respiratory condition, or feel lightheaded at any point, return to natural breathing.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_script_linter.py tests/unit/test_script_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/script_gen/linter.py docs/prompting_guides/content_safety_rules.md tests/unit/test_script_linter.py
git commit -m "feat(linter): block rapid breathing, pain-elimination and energy-healing claims"
```

---

### Task 5: Pack-scoped rules, new pack fields and pack data

**Files:**
- Modify: `core/genres.py` (`GenrePack` fields, `load_pack` validation, banned limit 10)
- Modify: `core/script_gen/linter.py` (`check_pack_rules`; breath-hold limit parameter)
- Modify: `core/auto_generate.py` (call `check_pack_rules` instead of `check_banned_phrases`)
- Modify: `core/script_gen/__init__.py` (export `check_pack_rules`)
- Modify packs:
  - `docs/genre_packs/{panic_sos,anxiety_relief,overwhelm}.toml`: `max_breath_hold_sec = 4`, plus banned additions.
  - `docs/genre_packs/{box_breathing,breath_awareness,calming_breath_4_7_8,coherent_breathing,energising_breath}.toml`: `requires_disclaimer = true`.
  - `docs/genre_packs/{grief_and_loss,loneliness,self_compassion}.toml`: banned additions.
  - `docs/genre_packs/{deep_sleep,fall_asleep,back_to_sleep,power_nap}.toml`: `pause_ratio = 0.20`.
  - `docs/genre_packs/README.md`: field contract.
- Test: `tests/unit/test_genres.py` (add), `tests/unit/test_script_linter.py` (add)

**Interfaces:**
- Consumes: `check_banned_phrases(script, banned)` and `_BREATH_HOLD` (existing).
- Produces:
  - `GenrePack.max_breath_hold_sec: int = 7` and `GenrePack.requires_disclaimer: bool = False`, as dataclass fields with defaults at the END of the dataclass.
  - `MAX_BANNED = 10`.
  - `linter.check_safety(script, *, max_hold_sec: int = MAX_BREATH_HOLD_SEC)`.
  - `linter.check_pack_rules(script: str, pack) -> list[Violation]`. It returns the banned-phrase violations, a stricter `BREATH_HOLD` when `pack.max_breath_hold_sec < MAX_BREATH_HOLD_SEC`, and `MISSING_DISCLAIMER` when `pack.requires_disclaimer`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_genres.py` (use that file's temp-pack-dir helpers or conftest fixtures to write a minimal valid pack TOML):

```python
class TestNewPackFields(unittest.TestCase):
    # use the file's existing helper that writes a valid pack with overrides
    def test_defaults(self):
        pack = load_pack("body_scan")
        self.assertEqual(pack.max_breath_hold_sec, 7)
        self.assertFalse(pack.requires_disclaimer)

    def test_real_packs_have_new_values(self):
        self.assertEqual(load_pack("panic_sos").max_breath_hold_sec, 4)
        self.assertTrue(load_pack("energising_breath").requires_disclaimer)
        self.assertEqual(load_pack("deep_sleep").pause_ratio, 0.20)
        self.assertIn("you are safe", load_pack("grief_and_loss").banned)

    def test_hold_out_of_range_rejected(self):
        ...  # write a pack with max_breath_hold_sec = 9 -> GenrePackError naming the field

    def test_disclaimer_must_be_bool(self):
        ...  # requires_disclaimer = "yes" -> GenrePackError

    def test_banned_over_ten_rejected(self):
        ...  # 11 banned phrases -> GenrePackError mentioning 10

    def test_every_shipped_pack_loads(self):
        from core.genres import load_all_packs
        self.assertEqual(len(load_all_packs()), 46)
```

Fill in the three `...` tests concretely with the file's existing temp-pack helper. If none exists, write a valid TOML string modelled on `docs/genre_packs/body_scan.toml` into a `tempfile.TemporaryDirectory()` and call `load_pack(slug, packs_dir=Path(tmp))`, checking the `load_pack` signature first.

Add to `tests/unit/test_script_linter.py`:

```python
class TestPackRules(unittest.TestCase):
    def _pack(self, **kw):
        from types import SimpleNamespace
        base = dict(banned=(), max_breath_hold_sec=7, requires_disclaimer=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_stricter_hold_for_pack(self):
        from core.script_gen.linter import check_pack_rules
        text = "Hold your breath for five seconds."
        self.assertIn("BREATH_HOLD", [v.code for v in check_pack_rules(text, self._pack(max_breath_hold_sec=4))])
        self.assertEqual(check_pack_rules(text, self._pack()), [])

    def test_missing_disclaimer(self):
        from core.script_gen.linter import FATAL, check_pack_rules
        body = "Breathe in slowly. " * 40
        v = check_pack_rules(body, self._pack(requires_disclaimer=True))
        self.assertEqual([(x.code, x.severity) for x in v], [("MISSING_DISCLAIMER", FATAL)])

    def test_disclaimer_present_early(self):
        from core.script_gen.linter import check_pack_rules
        text = ("If you are pregnant or feel lightheaded at any point, return to natural breathing. "
                + "Breathe in slowly. " * 40)
        self.assertEqual(check_pack_rules(text, self._pack(requires_disclaimer=True)), [])

    def test_disclaimer_too_late(self):
        from core.script_gen.linter import check_pack_rules
        text = ("Breathe in slowly. " * 40
                + "If you feel lightheaded, return to natural breathing.")
        self.assertIn("MISSING_DISCLAIMER", [v.code for v in check_pack_rules(text, self._pack(requires_disclaimer=True))])

    def test_banned_still_checked(self):
        from core.script_gen.linter import check_pack_rules
        self.assertIn("BANNED_PHRASE", [v.code for v in check_pack_rules("You are safe here.", self._pack(banned=("you are safe",)))])
```

Add to `tests/unit/test_auto_generate.py`: the genre path calls `check_pack_rules`. A pack with `requires_disclaimer=True` and a clean script lacking a disclaimer makes the judge get a repair call (`MISSING_DISCLAIMER` is FATAL). Use that file's genre-path fixtures and patch `load_pack` to return a pack with the flag.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_genres.py tests/unit/test_script_linter.py -v -k "NewPackFields or PackRules"`
Expected: FAIL

- [ ] **Step 3: Implement the genres changes**

In `core/genres.py`:
- Add `MAX_BANNED = 10`.
- Add the two dataclass fields with defaults (last in the dataclass).
- In `load_pack`:
  - After banned parsing: `if len(banned) > MAX_BANNED: raise GenrePackError(f"{path} has {len(banned)} banned phrases; at most {MAX_BANNED}.")`.
  - `max_breath_hold_sec`: if present it must be an `int` (not bool) in 1–7, else `GenrePackError(f"{path}: field 'max_breath_hold_sec' must be an integer 1-7, got {value!r}.")`. The default is 7.
  - `requires_disclaimer`: if present it must be a `bool`, else `GenrePackError`. The default is False.
  - Pass both into `GenrePack(...)`.

- [ ] **Step 4: Implement the linter changes**

In `core/script_gen/linter.py`:
- Change the signature to `def check_safety(script: str, *, max_hold_sec: int = MAX_BREATH_HOLD_SEC)`. Use `max_hold_sec` in place of `MAX_BREATH_HOLD_SEC` in the breath-hold comparison and message (`f"Holds over {max_hold_sec}s are a risk..."`).
- Add:

```python
_DISCLAIMER_CONDITION = re.compile(r"\b(?:pregnan\w*|heart|respiratory|light-?headed|dizzy)\b", re.IGNORECASE)
_DISCLAIMER_ACTION = re.compile(r"\b(?:natural\s+breath\w*|normal\s+breath\w*|breathe\s+normally)\b", re.IGNORECASE)
DISCLAIMER_WINDOW = 0.25  # fraction of the script's words the disclaimer must appear within


def check_pack_rules(script: str, pack) -> list[Violation]:
    """Genre-pack rules: banned phrases, a stricter hold limit, a required disclaimer."""
    violations = check_banned_phrases(script, getattr(pack, "banned", ()))
    limit = getattr(pack, "max_breath_hold_sec", MAX_BREATH_HOLD_SEC)
    if limit < MAX_BREATH_HOLD_SEC:
        violations += [v for v in check_safety(script, max_hold_sec=limit) if v.code == "BREATH_HOLD"]
    if getattr(pack, "requires_disclaimer", False):
        prose = _normalize_apostrophes(_strip_tags(script))
        words = prose.split()
        opening = " ".join(words[: max(1, int(len(words) * DISCLAIMER_WINDOW))])
        if not (_DISCLAIMER_CONDITION.search(opening) and _DISCLAIMER_ACTION.search(opening)):
            violations.append(Violation(
                code="MISSING_DISCLAIMER", severity=FATAL,
                message=("Breathwork session without an early safety line. In the opening, say once: "
                         "if you are pregnant, have a heart or respiratory condition, or feel "
                         "lightheaded at any point, return to natural breathing."),
            ))
    return violations
```

  The global `check()` still uses the default 7 s limit, so a pack limit of 4 adds a stricter `BREATH_HOLD` only for holds of 5–7 s. That avoids duplicates, since `check()` already reports holds over 7 s. To make that exact, filter the pack-scoped hold violations to holds `> limit` and `<= MAX_BREATH_HOLD_SEC`. Implement it by parsing the seconds out the same way `check_safety` does, or by comparing messages. Prefer refactoring the hold-seconds extraction into a helper `_breath_hold_seconds(prose) -> list[int]` that both functions use.
- In `core/auto_generate.py`, replace `violations = violations + check_banned_phrases(script, pack.banned)` with `violations = violations + check_pack_rules(script, pack)`, and update the import. `check_banned_phrases` stays exported.
- Export `check_pack_rules` from `core/script_gen/__init__.py`.

- [ ] **Step 5: Edit the pack data**
- `panic_sos`, `anxiety_relief`, `overwhelm`: add `max_breath_hold_sec = 4` after `pause_ratio`. Append `"you are safe"` to `banned`.
- `grief_and_loss`: append `"you are safe"`, `"better place"`, `"everything happens for a reason"`, `"time heals"` to `banned`. Skip any already present, since some packs already ban "in a better place"; don't duplicate a phrase that's a substring of an existing one.
- `loneliness`, `self_compassion`: append `"better place"`, `"everything happens for a reason"`, `"time heals"` (same rule).
- The 5 Breathwork packs: add `requires_disclaimer = true` after `pause_ratio`.
- `deep_sleep`, `fall_asleep`, `back_to_sleep`, `power_nap`: set `pause_ratio = 0.20`.
- `docs/genre_packs/README.md`:
  - Change `banned` to "3–10 … enforced by the loader".
  - Add rows for `max_breath_hold_sec` (optional int 1–7, default 7: stricter hold limit for this genre) and `requires_disclaimer` (optional bool: requires an early breathwork contraindication line, FATAL `MISSING_DISCLAIMER`).
- Verify every banned list ends at 10 or fewer.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_genres.py tests/unit/test_script_linter.py tests/unit/test_auto_generate.py tests/unit/test_eval_testset.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/genres.py core/script_gen/linter.py core/script_gen/__init__.py core/auto_generate.py docs/genre_packs/ tests/unit/test_genres.py tests/unit/test_script_linter.py tests/unit/test_auto_generate.py
git commit -m "feat(genres): pack-scoped hold limits, breathwork disclaimer, sleep pause_ratio and banned additions"
```

---

### Task 6: Guides rendered from code constants

**Files:**
- Create: `core/script_gen/guide_context.py`
- Modify: `core/script_gen/rules.py` (`load_guide` renders; planner and safety rules unchanged)
- Modify: `core/script_gen/linter.py` (add `IDEAL_SENTENCE_MIN_WORDS = 8`, `IDEAL_SENTENCE_MAX_WORDS = 20`; use them in the `SENTENCE_TOO_LONG` message)
- Modify the four guides: `docs/prompting_guides/vocal_{meditation,sleep_story}_{f5,kokoro}_instructions.md`
- Test: `tests/unit/test_guide_rendering.py`

**Interfaces:**
- Consumes:
  - `core.f5_tts.preprocessor.MAX_CHUNK_CHARS` (250) and `core.kokoro_tts.preprocessor.MAX_CHUNK_TOKENS` (150).
  - `core.content_profiles.get_profile(ct)` with keys `speed`, `f5_paragraph_pause_sec`, `kokoro_paragraph_pause_sec`.
  - `core.script_gen.duration.DEFAULT_WPM`.
  - `linter.MIN_PAUSE_SEC`, `MAX_PAUSE_SEC`, `MAX_BREATH_HOLD_SEC`, `MAX_SENTENCE_WORDS`, and the new `IDEAL_SENTENCE_*`.
  - `core.script_gen.slop.slop_guidance()` (Task 3).
- Produces:
  - `guide_context(engine: str, content_type: str) -> dict[str, str]`, with keys `CHUNK_CHARS` (F5 only), `CHUNK_TOKENS` (Kokoro only), `SPEED`, `PARA_PAUSE`, `WPM`, `PAUSE_MIN`, `PAUSE_MAX`, `MAX_HOLD`, `SENT_IDEAL_MIN`, `SENT_IDEAL_MAX`, `SENT_MAX`, `SLOP_GUIDANCE`.
  - `render_guide(text: str, context: dict[str, str]) -> str`.
  - `load_guide(engine, content_type, guides_dir=None)` returns the rendered text.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_guide_rendering.py
import tempfile
import unittest
from pathlib import Path

from core.script_gen.guide_context import guide_context, render_guide
from core.script_gen.rules import (
    build_generator_system_prompt, build_judge_system_prompt, build_planner_system_prompt, load_guide,
)

COMBOS = [(e, c) for e in ("f5", "kokoro") for c in ("meditation", "sleep_story")]
STALE = ["300 char", "300-char", "0.88", "100–130", "100-130", "1.5s in Sleep Story", "1.5s in sleep story"]


class TestRenderGuide(unittest.TestCase):
    def test_renders_known_placeholder(self):
        self.assertEqual(render_guide("max {{WPM}} wpm", {"WPM": "85"}), "max 85 wpm")

    def test_unknown_placeholder_raises_before_llm(self):
        with self.assertRaisesRegex(KeyError, "CHUNK_CHAR"):
            render_guide("limit {{CHUNK_CHAR}}", {"CHUNK_CHARS": "250"})

    def test_unknown_placeholder_in_guide_file_raises_at_prompt_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "vocal_meditation_f5_instructions.md").write_text("bad {{NOPE}}")
            (d / "content_safety_rules.md").write_text("rules")
            with self.assertRaisesRegex(KeyError, "NOPE"):
                build_generator_system_prompt("f5", "meditation", 360, 600, guides_dir=d)


class TestContext(unittest.TestCase):
    def test_f5_values_come_from_code(self):
        from core.content_profiles import get_profile
        from core.f5_tts.preprocessor import MAX_CHUNK_CHARS
        from core.script_gen.duration import DEFAULT_WPM
        ctx = guide_context("f5", "sleep_story")
        self.assertEqual(ctx["CHUNK_CHARS"], str(MAX_CHUNK_CHARS))
        self.assertEqual(ctx["SPEED"], f"{get_profile('sleep_story')['speed']:.2f}")
        self.assertEqual(ctx["PARA_PAUSE"], f"{get_profile('sleep_story')['f5_paragraph_pause_sec']:g}")
        self.assertEqual(ctx["WPM"], f"{DEFAULT_WPM['f5']:.0f}")
        self.assertNotIn("CHUNK_TOKENS", ctx)

    def test_kokoro_uses_tokens(self):
        ctx = guide_context("kokoro", "meditation")
        self.assertIn("CHUNK_TOKENS", ctx)
        self.assertNotIn("CHUNK_CHARS", ctx)


class TestShippedGuides(unittest.TestCase):
    def test_every_guide_renders_clean_and_current(self):
        for engine, ct in COMBOS:
            text = load_guide(engine, ct)
            ctx = guide_context(engine, ct)
            self.assertNotIn("{{", text, (engine, ct))
            for key in ("WPM", "PARA_PAUSE", "SPEED"):
                self.assertIn(ctx[key], text, (engine, ct, key))
            for stale in STALE:
                self.assertNotIn(stale, text, (engine, ct, stale))

    def test_full_prompts_build(self):
        for engine, ct in COMBOS:
            build_generator_system_prompt(engine, ct, 360, 600)
            build_judge_system_prompt(engine, ct, 360, 600)
        build_planner_system_prompt("meditation", 360, 600)

    def test_example_scripts_pass_the_linter_safety_and_format(self):
        """Every fenced example script in the F5 guides must itself obey the rules."""
        import re
        from core.script_gen.linter import FATAL, check_format, check_safety
        for engine, ct in [("f5", "meditation"), ("f5", "sleep_story")]:
            text = load_guide(engine, ct)
            examples = [b for b in re.findall(r"```\n(.*?)```", text, re.S) if "[pause:" in b]
            self.assertTrue(examples, (engine, ct))
            for block in examples:
                fatal = [v.code for v in check_format(block) + check_safety(block) if v.severity == FATAL]
                self.assertEqual(fatal, [], (engine, ct, block[:80]))
                self.assertNotIn("...", block)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_guide_rendering.py -v`
Expected: FAIL (`ModuleNotFoundError: core.script_gen.guide_context`)

- [ ] **Step 3: Implement `guide_context.py` and rendering**

```python
# core/script_gen/guide_context.py
"""Values the prompting guides quote, taken from the code that enforces them.

Guides are templates: {{NAME}} is replaced at load time, so a guide can never
again tell the writer a chunk limit, speed or speaking rate the renderer does
not use.
"""

import re

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Z_]+)\s*\}\}")


def guide_context(engine: str, content_type: str) -> dict[str, str]:
    from core.content_profiles import get_profile
    from core.script_gen import linter
    from core.script_gen.duration import DEFAULT_WPM
    from core.script_gen.slop import slop_guidance

    profile = get_profile(content_type)
    ctx = {
        "SPEED": f"{profile['speed']:.2f}",
        "PARA_PAUSE": f"{profile[f'{engine}_paragraph_pause_sec']:g}",
        "WPM": f"{DEFAULT_WPM[engine]:.0f}",
        "PAUSE_MIN": f"{linter.MIN_PAUSE_SEC:g}",
        "PAUSE_MAX": f"{linter.MAX_PAUSE_SEC:g}",
        "MAX_HOLD": str(linter.MAX_BREATH_HOLD_SEC),
        "SENT_IDEAL_MIN": str(linter.IDEAL_SENTENCE_MIN_WORDS),
        "SENT_IDEAL_MAX": str(linter.IDEAL_SENTENCE_MAX_WORDS),
        "SENT_MAX": str(linter.MAX_SENTENCE_WORDS),
        "SLOP_GUIDANCE": slop_guidance(20),
    }
    if engine == "kokoro":
        from core.kokoro_tts.preprocessor import MAX_CHUNK_TOKENS
        ctx["CHUNK_TOKENS"] = str(MAX_CHUNK_TOKENS)
    else:
        from core.f5_tts.preprocessor import MAX_CHUNK_CHARS
        ctx["CHUNK_CHARS"] = str(MAX_CHUNK_CHARS)
    return ctx


def render_guide(text: str, context: dict[str, str]) -> str:
    def _sub(match: re.Match) -> str:
        name = match.group(1)
        if name not in context:
            raise KeyError(f"Unknown guide placeholder {{{{{name}}}}}; known: {sorted(context)}")
        return context[name]

    rendered = _PLACEHOLDER.sub(_sub, text)
    if "{{" in rendered:
        raise ValueError("Unrendered '{{' left in guide after substitution.")
    return rendered
```

In `rules.py`, change `load_guide` to `return render_guide(_read(root / filename), guide_context(engine, content_type))`. Import both from `core.script_gen.guide_context`. The planner prompt takes no engine guide, so it is unchanged. `load_safety_rules` is unchanged.

In `linter.py`, add `IDEAL_SENTENCE_MIN_WORDS = 8` and `IDEAL_SENTENCE_MAX_WORDS = 20` next to `MAX_SENTENCE_WORDS`. Use them in the `SENTENCE_TOO_LONG` message text instead of the literal "8-20".

- [ ] **Step 4: Edit the four guides**

Each guide is a document the LLM reads. Keep its structure and voice and change only what's listed.

1. **Replace every number that has a code source with its placeholder:**
   - F5 chunk limit (any "300 chars", "≤300-character", "~300 chars", "250") → `{{CHUNK_CHARS}}`.
   - Kokoro "150 tokens" → `{{CHUNK_TOKENS}}`.
   - Default or recommended speech speed (0.88, 0.90, "slightly slower preset") → `{{SPEED}}`. Say it's set by the Content Type preset, not the script.
   - The paragraph-break pause (3.0s, 6.5s, 2.5s, 1.5s, 2.0s, 4.0s) → `{{PARA_PAUSE}}s`.
   - The speaking rate (85–95, ~95–100, 100–130 words per minute) → "about `{{WPM}}` words per minute". Remove the "Target length examples" word counts that contradict it, replacing them with: "Budget words as minutes × `{{WPM}}`, then add your explicit pauses."
   - Pause bounds (0.5s minimum and so on) → `{{PAUSE_MIN}}`/`{{PAUSE_MAX}}`.
   - Breath hold limit → `{{MAX_HOLD}}`.
   - Sentence length ideal and maximum ("8–20 words", "under 25") → `{{SENT_IDEAL_MIN}}–{{SENT_IDEAL_MAX}}` / `{{SENT_MAX}}`.
2. **Remove stale claims:** every `STALE` string in the test, plus the "Speed | Approximate WPM" table that maps speeds to 85–130 wpm. Replace that table with one sentence: "Pace is set by the Content Type preset (speed `{{SPEED}}`); the voice reads at about `{{WPM}}` words per minute."
3. **Add a "Style" subsection to each guide** containing exactly `{{SLOP_GUIDANCE}}` on its own line.
4. **Sleep-story guides:** replace the pause table and frequency text with progressive guidance: "1.5–2.5 s in the first third, 2–4 s in the middle, 4–8 s in the final third, never more than 10 s; place a pause every 2–4 sentences, lengthening toward the end."
5. **Rewrite the F5 meditation "Complete Example Script"** to the following, verbatim (no ellipses, no outcome promises, an opt-out, pauses within bounds):

```
Welcome. This is a few quiet minutes, just for you. [pause:3s]

You can keep your eyes open or let them close, whichever feels easier, and you can shift your position at any time. [pause:4s]

Let your body settle into whatever is holding you. [pause:3s] Notice where you are touching the chair, the floor, or the bed. [pause:5s]

[breath]

If it feels right, take a slow breath in through your nose. [pause:4s] And let it go through your mouth. [pause:6s]

Notice the weight of your hands. [pause:3s] Notice the air on your face. [pause:5s]

Thoughts may come and go. That is fine. When one pulls you away, you can return to the feeling of your breath. [pause:8s]

[breath]

With each exhale, see if your shoulders can soften a little. [pause:6s]

Rest here for a few breaths. [pause:10s]

When you are ready, let your breathing deepen. [pause:4s] Notice the sounds in the room around you. [pause:4s] Move your fingers and your toes. [pause:3s]

Open your eyes when it suits you. Thank you for taking this time. [pause:3s]
```

   Also fix the F5 meditation "Breath Markers"/"Pause Duration Reference" and Kokoro sections so no example uses `...`. The F5 guides already say ellipses are normalised away.
6. **The F5 sleep-story worked example:** replace any `[pause:1s]`/`[pause:1.5s]` with values from the progressive guidance (e.g. `[pause:2s]` early, `[pause:4s]` at the end), and keep it free of ellipses.

After editing, render each guide (`load_guide`) and read the output once end-to-end. Confirm it reads naturally and has no contradictions: for example, no leftover sentence that still names a different chunk limit, speed or speaking rate.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_guide_rendering.py tests/unit/test_script_rules.py tests/unit/test_script_linter.py tests/unit/test_auto_generate.py -v`
Expected: PASS. If an existing `test_script_rules.py` test asserts on a now-removed literal from a guide, update it to assert on the rendered value instead, and say so in your report.

- [ ] **Step 6: Commit**

```bash
git add core/script_gen/guide_context.py core/script_gen/rules.py core/script_gen/linter.py docs/prompting_guides/ tests/unit/test_guide_rendering.py tests/unit/test_script_rules.py
git commit -m "feat(script_gen): render prompting guides from code constants and fix their contradictions"
```

---

### Task 7: Per-family voice selection

**Files:**
- Create: `core/voice_selection.py`
- Modify: `core/auto_generate.py` (`AutoConfig.voice_slug`, `from_genre`, `run()` passes `f5_voice_slug`, `meta.json["voice"]`, `AutoResult.voice`)
- Modify: `core/auto_tab.py` (status line shows the voice)
- Test: `tests/unit/test_voice_selection.py`, `tests/unit/test_auto_generate.py` (add)

**Interfaces:**
- Consumes: `core.f5_tts.voice_registry.scan() -> {slug: {phase: {"audio": Path, "transcript": Path | str}}}`.
- Produces:
  - `@dataclass(frozen=True) class VoiceTraits: wpm: float; f0_median: float; f0_range: float; centroid: float`.
  - `measure_voice(audio_path: Path, transcript: str) -> VoiceTraits`.
  - `voice_traits(registry: dict | None = None, cache_path: Path | None = None) -> dict[str, VoiceTraits]`.
  - `FAMILY_PROFILE: dict[str, str]`.
  - `profile_scores(traits: dict[str, VoiceTraits]) -> dict[str, dict[str, float]]`.
  - `voice_for_family(family: str, *, traits=None, overrides_path: Path | None = None) -> str`.
  - `AutoConfig.voice_slug: str | None = None`, `AutoResult.voice: str = ""`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_voice_selection.py
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.voice_selection import (
    FAMILY_PROFILE, VoiceTraits, measure_voice, profile_scores, voice_for_family, voice_traits,
)

# Measured 2026-10-07 from assets/speakers (see the R1 spec).
TRAITS = {
    "Brittney": VoiceTraits(109, 185, 175, 1412),
    "Clara": VoiceTraits(106, 175, 75, 1615),
    "Delilah": VoiceTraits(97, 177, 139, 1303),
    "Eryn": VoiceTraits(106, 191, 86, 1864),
}


class TestMapping(unittest.TestCase):
    def test_expected_assignment_for_todays_voices(self):
        expected = {"Sleep & Rest": "Delilah", "Stress & Anxiety": "Delilah",
                    "Morning & Energy": "Eryn", "Body & Movement": "Eryn",
                    "Emotional": "Brittney", "Presence": "Brittney",
                    "Focus & Work": "Clara", "Breathwork": "Clara"}
        for family, voice in expected.items():
            self.assertEqual(voice_for_family(family, traits=TRAITS, overrides_path=Path("/nonexistent")), voice, family)

    def test_every_shipped_family_has_a_profile(self):
        from core.genres import load_all_packs
        families = {p.family for p in load_all_packs().values()}
        self.assertTrue(families <= set(FAMILY_PROFILE), families - set(FAMILY_PROFILE))

    def test_unknown_family_uses_steady(self):
        self.assertEqual(voice_for_family("Nope", traits=TRAITS, overrides_path=Path("/nonexistent")), "Clara")

    def test_single_voice_pool(self):
        self.assertEqual(voice_for_family("Sleep & Rest", traits={"Solo": TRAITS["Eryn"]},
                                          overrides_path=Path("/nonexistent")), "Solo")

    def test_ties_break_alphabetically(self):
        same = {"B": VoiceTraits(100, 180, 100, 1500), "A": VoiceTraits(100, 180, 100, 1500)}
        self.assertEqual(voice_for_family("Sleep & Rest", traits=same, overrides_path=Path("/nonexistent")), "A")

    def test_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice_map.toml"
            p.write_text('[families]\n"Sleep & Rest" = "Eryn"\n')
            self.assertEqual(voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=p), "Eryn")

    def test_override_unknown_slug_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice_map.toml"
            p.write_text('[families]\n"Sleep & Rest" = "Nobody"\n')
            with self.assertRaisesRegex(ValueError, "Nobody"):
                voice_for_family("Sleep & Rest", traits=TRAITS, overrides_path=p)

    def test_scores_are_relative(self):
        s = profile_scores(TRAITS)
        self.assertGreater(s["Delilah"]["calm"], s["Eryn"]["calm"])
        self.assertGreater(s["Brittney"]["expressive"], s["Clara"]["expressive"])


class TestMeasurement(unittest.TestCase):
    def _tone(self, path, freq, seconds=3.0, sr=24000):
        t = np.arange(int(sr * seconds)) / sr
        sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)

    def test_measure_pitch_and_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "v.wav"
            self._tone(wav, 200.0)
            tr = measure_voice(wav, "one two three four five six")
            self.assertAlmostEqual(tr.f0_median, 200.0, delta=10.0)
            self.assertAlmostEqual(tr.wpm, 6 / 3.0 * 60, delta=1.0)

    def test_cache_hits_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            wav = d / "A.wav"
            self._tone(wav, 200.0)
            reg = {"A": {"default": {"audio": wav, "transcript": "one two three"}}}
            cache = d / "traits.json"
            first = voice_traits(reg, cache_path=cache)
            data = json.loads(cache.read_text())
            data["A"]["traits"]["f0_median"] = 999.0  # poison the cache
            cache.write_text(json.dumps(data))
            self.assertEqual(voice_traits(reg, cache_path=cache)["A"].f0_median, 999.0)  # cache hit
            os.utime(wav, (wav.stat().st_atime, wav.stat().st_mtime + 10))
            self.assertAlmostEqual(voice_traits(reg, cache_path=cache)["A"].f0_median, first["A"].f0_median, delta=1.0)

    def test_unreadable_clip_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bad = d / "Bad.wav"
            bad.write_bytes(b"not audio")
            good = d / "Good.wav"
            self._tone(good, 180.0)
            reg = {"Bad": {"default": {"audio": bad, "transcript": "x"}},
                   "Good": {"default": {"audio": good, "transcript": "one two"}}}
            self.assertEqual(set(voice_traits(reg, cache_path=d / "c.json")), {"Good"})


if __name__ == "__main__":
    unittest.main()
```

Add to `tests/unit/test_auto_generate.py`, using that file's fixtures:
- `AutoConfig.from_genre(load_pack("deep_sleep"), band="short")` has `voice_slug` set from `voice_for_family`. Patch `core.voice_selection.voice_for_family` (or the name imported into `auto_generate`) to return `"Delilah"`, and assert it.
- An explicit `AutoConfig.from_genre(pack, voice_slug="Eryn")` keeps `"Eryn"`.
- `run()` passes `f5_voice_slug="Delilah"` to the fake pipeline's `generate`, and `meta.json["voice"] == "Delilah"`.
- If `voice_for_family` raises a non-`ValueError` exception (measurement failure), `from_genre` logs a warning and sets `voice_slug=None`, so the job still runs. A `ValueError` (bad override) propagates.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_voice_selection.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `core/voice_selection.py`**

```python
# core/voice_selection.py
"""Choose a narrator voice per genre family from measured reference-clip traits.

Each reference clip is measured once (speaking rate, median pitch, pitch range,
spectral brightness) and cached by (filename, size, mtime). Families map to a
profile -- calm, energy, expressive, steady -- and the voice scoring highest on
that profile (z-scored across the pool) narrates the family. An optional
assets/speakers/voice_map.toml overrides any family.
"""

import json
import logging
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = _ROOT / "var" / "voice_traits.json"
DEFAULT_OVERRIDES = _ROOT / "assets" / "speakers" / "voice_map.toml"

FAMILY_PROFILE: dict[str, str] = {
    "Sleep & Rest": "calm",
    "Stress & Anxiety": "calm",
    "Morning & Energy": "energy",
    "Body & Movement": "energy",
    "Emotional": "expressive",
    "Presence": "expressive",
    "Focus & Work": "steady",
    "Breathwork": "steady",
}
FALLBACK_PROFILE = "steady"


@dataclass(frozen=True)
class VoiceTraits:
    wpm: float
    f0_median: float
    f0_range: float
    centroid: float


def measure_voice(audio_path: Path, transcript: str) -> VoiceTraits:
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(audio_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    duration = len(y) / sr
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
    f0v = f0[voiced & ~np.isnan(f0)]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    loud = rms > 0.1 * rms.max() if rms.size and rms.max() > 0 else np.ones_like(centroid, bool)
    n = min(len(centroid), len(loud))
    return VoiceTraits(
        wpm=len(transcript.split()) / duration * 60.0 if duration else 0.0,
        f0_median=float(np.median(f0v)) if f0v.size else 0.0,
        f0_range=float(np.percentile(f0v, 90) - np.percentile(f0v, 10)) if f0v.size else 0.0,
        centroid=float(np.median(centroid[:n][loud[:n]])) if n else 0.0,
    )


def _transcript_text(value) -> str:
    return Path(value).read_text(encoding="utf-8").strip() if isinstance(value, Path) else str(value)


def voice_traits(registry: dict | None = None, cache_path: Path | None = None) -> dict[str, VoiceTraits]:
    if registry is None:
        from core.f5_tts import voice_registry
        registry = voice_registry.scan()
    cache_path = cache_path or DEFAULT_CACHE
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    out: dict[str, VoiceTraits] = {}
    changed = False
    for slug, phases in sorted(registry.items()):
        phase = phases.get("default") or next(iter(phases.values()))
        audio = Path(phase["audio"])
        try:
            stat = audio.stat()
            key = [audio.name, stat.st_size, stat.st_mtime]
            entry = cache.get(slug)
            if entry and entry.get("key") == key:
                out[slug] = VoiceTraits(**entry["traits"])
                continue
            traits = measure_voice(audio, _transcript_text(phase["transcript"]))
        except Exception as exc:  # unreadable clip: skip, never fail a job
            logger.warning("voice_selection: could not measure %s (%s); skipping", audio, exc)
            continue
        out[slug] = traits
        cache[slug] = {"key": key, "traits": asdict(traits)}
        changed = True
    if changed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
        except OSError as exc:
            logger.warning("voice_selection: could not write cache %s (%s)", cache_path, exc)
    return out


def _z(values: dict[str, float]) -> dict[str, float]:
    arr = np.array(list(values.values()), dtype=float)
    std = arr.std()
    return {k: 0.0 if std == 0 else float((v - arr.mean()) / std) for k, v in values.items()}


def profile_scores(traits: dict[str, VoiceTraits]) -> dict[str, dict[str, float]]:
    zr = _z({k: t.wpm for k, t in traits.items()})
    zc = _z({k: t.centroid for k, t in traits.items()})
    zf = _z({k: t.f0_median for k, t in traits.items()})
    zg = _z({k: t.f0_range for k, t in traits.items()})
    scores = {}
    for k in traits:
        calm = -zr[k] - zc[k] - zf[k]
        scores[k] = {"calm": calm, "energy": -calm, "expressive": zg[k], "steady": -zg[k]}
    return scores


def _overrides(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(tomllib.loads(path.read_text(encoding="utf-8")).get("families", {}))


def voice_for_family(family: str, *, traits: dict[str, VoiceTraits] | None = None,
                     overrides_path: Path | None = None) -> str:
    traits = traits if traits is not None else voice_traits()
    if not traits:
        raise RuntimeError("No measurable reference voices found in assets/speakers.")
    override = _overrides(overrides_path or DEFAULT_OVERRIDES).get(family)
    if override is not None:
        if override not in traits:
            raise ValueError(f"voice_map.toml maps {family!r} to unknown voice {override!r}; known: {sorted(traits)}")
        return override
    profile = FAMILY_PROFILE.get(family, FALLBACK_PROFILE)
    scores = profile_scores(traits)
    return sorted(traits, key=lambda k: (-round(scores[k][profile], 9), k))[0]
```

`voice_registry.scan()` returns `"transcript"` as either a `Path` or a `str`; `_transcript_text` handles both.

- [ ] **Step 4: Wire it into auto-generation**

In `core/auto_generate.py`:
- Add `voice_slug: str | None = None` to `AutoConfig`, with a comment: "F5/Chatterbox/Qwen3 reference voice; set per genre family by from_genre".
- In `from_genre`, after `pack_values` is built and before applying caller overrides:

```python
        engine = overrides.get("tts_engine", config.tts_engine)
        if engine in ("f5", "chatterbox", "qwen3") and "voice_slug" not in overrides:
            try:
                from core.voice_selection import voice_for_family
                pack_values["voice_slug"] = voice_for_family(pack.family)
            except ValueError:
                raise
            except Exception as exc:
                logger.warning("voice selection failed for %s (%s); using engine default voice", pack.family, exc)
```

  `core/auto_tab.py` passes `tts_engine` as an override, so the engine check must read `overrides.get("tts_engine", config.tts_engine)`. Add a test that `from_genre(pack, tts_engine="kokoro")` leaves `voice_slug` None.
- In `run()`, when `config.voice_slug` is set, add `generate_kwargs["f5_voice_slug"] = config.voice_slug` before `generate_kwargs.update(pipeline_kwargs)`, so an explicit pipeline kwarg still wins.
- Add `voice: str = ""` to `AutoResult`. Set it to `config.voice_slug or ""` in `run()`'s return, and add `"voice": config.voice_slug` to `meta.json`.

In `core/auto_tab.py`, change the status line to include `f"Voice: {result.voice or 'default'}. "` after the Background part.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_voice_selection.py tests/unit/test_auto_generate.py tests/unit/test_auto_tab.py tests/unit/test_streaming_run.py -v`
Expected: PASS

- [ ] **Step 6: Real-clip sanity check (cheap, no model)**

Run: `.venv/bin/python -c "from core.voice_selection import FAMILY_PROFILE, voice_for_family; print({f: voice_for_family(f) for f in FAMILY_PROFILE})"`
Expected: Sleep & Rest/Stress & Anxiety → Delilah; Morning & Energy/Body & Movement → Eryn; Emotional/Presence → Brittney; Focus & Work/Breathwork → Clara. This also writes `var/voice_traits.json`, which is gitignored. If the real mapping differs, report the measured traits rather than changing the scoring.

- [ ] **Step 7: Commit**

```bash
git add core/voice_selection.py core/auto_generate.py core/auto_tab.py tests/unit/test_voice_selection.py tests/unit/test_auto_generate.py
git commit -m "feat(auto_generate): choose the narrator voice per genre family from measured clip traits"
```

---

### Task 8: Recording spec and documentation

**Files:**
- Create: `docs/setup_and_execution/reference_voice_recording_spec.md`
- Create: `assets/speakers/voice_map.example.toml`
- Modify: `docs/auto_generation/README.md`, `CLAUDE.md`, `docs/GOTCHAS.md`

**Interfaces:**
- Consumes: everything above (documentation only).

- [ ] **Step 1: Write the recording spec**

Write `docs/setup_and_execution/reference_voice_recording_spec.md` with these sections:
- **Purpose:** better F5 cloning; multi-phase sets to break the repeated intonation contour.
- **Capture:** 48 kHz/24-bit WAV, delivered as 24 kHz mono 16-bit (`ffmpeg -i in.wav -ar 24000 -ac 1 -sample_fmt s16 out.wav`). Noise floor ≤ −60 dBFS, RMS ≈ −20 dBFS, peaks ≤ −6 dBFS.
- **Mic and room:** large-diaphragm or dynamic mic at 15–20 cm, slightly off-axis, pop filter, treated or dead room. No bathrooms or bare walls.
- **Clip content:**
  - 8–11 s (F5 clips references at 12 s).
  - Two complete sentences ending on a falling cadence, with at least 0.5 s of room tone after.
  - One natural inhale allowed at the start, none mid-clip. Avoid dense sibilants.
  - A verbatim transcript in `assets/speakers/reference_text/<slug>.txt`.
- **Phases:** opening (≈100 WPM, warm, slightly brighter), body (≈90 WPM, neutral calm), closing (≈80 WPM, softer, lower, still fully voiced; F5 cannot whisper reliably). Three takes per phase. Wire them up with `assets/speakers/voices.toml` (show a phase table example) and `[voice:phase]` markers.
- **How a new voice is assigned:** `core/voice_selection.py` measures every clip. Lower WPM, darker and lower-pitched voices score `calm`; wider pitch range scores `expressive`. A voice lands where it scores highest. Override in `assets/speakers/voice_map.toml`; see the example file.
- **Licensing:** record only voices you have the rights to clone, with written consent.

- [ ] **Step 2: Write `assets/speakers/voice_map.example.toml`**

```toml
# Copy to voice_map.toml to override the automatic per-family voice choice.
# Keys are genre families (as in docs/genre_packs/*.toml); values are voice
# slugs from assets/speakers/reference_audio/. Unlisted families keep the
# measured choice (core/voice_selection.py).
[families]
# "Sleep & Rest" = "Delilah"
# "Morning & Energy" = "Eryn"
```

- [ ] **Step 3: Update the docs**
- **`docs/auto_generation/README.md`:**
  - A "Sampling" subsection: the table plus env overrides, and why `/v1` is enough (Modelfile `top_k`).
  - A "Voices" subsection: profiles, the mapping, the override file, the cache.
  - Add the new violation codes to the "Violation code reference" table, with severities: RAPID_BREATHING, PAIN_ELIMINATION, ENERGY_HEALING_CLAIM and MISSING_DISCLAIMER FATAL; SLOP_DENSITY and SLOP_HARD ADVISORY.
  - Note that slop advisories ride along in repairs but never trigger one.
- **`CLAUDE.md`:**
  - Folder Map: add `core/voice_selection.py` and the `core/script_gen/{sampling,slop,guide_context}.py` mentions.
  - Top Gotchas: add one bullet: **Prompting guides are templates.** `{{NAME}}` placeholders in `docs/prompting_guides/vocal_*` are rendered from code constants by `script_gen/guide_context.py`. Never hard-code a chunk limit, speed, WPM or pause length in a guide. An unknown placeholder fails the run before any LLM call.
- **`docs/GOTCHAS.md`:** under Auto-Generation, add the same templated-guides bullet, plus "Slop is ADVISORY by design: never make SLOP_* fatal; it would make local models fail on style."

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add docs/ assets/speakers/voice_map.example.toml CLAUDE.md
git commit -m "docs: R1 recording spec, sampling, voices and templated-guide gotcha"
```
