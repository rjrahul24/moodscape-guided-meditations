# Automated Meditation Generation — Design

**Date:** 2026-09-16
**Branch:** `dev-automate` (off `dev`)
**Status:** Approved design, pending implementation plan

---

## Goal

Automate MoodScape end to end. Today a script is hand-written and pasted into the
Gradio UI. The target flow is:

1. User enters a natural-language prompt ("I am feeling anxious, I need a relaxing
   meditation").
2. The app generates a full script suited to a 5–7 minute meditation.
3. A royalty-free background instrumental is picked at random.
4. Script plus instrumental render to a finished meditation.
5. The finished audio is returned to the user.

Steps 3–5 are wiring around `MeditationPipeline.generate()`, which already accepts
everything required. Steps 1–2 do not exist in any form.

## Context and constraints

- **Single user.** This is a content factory for the project owner, not a
  multi-tenant product. No auth, no horizontal scaling, no per-user cost control,
  no durable job queue. One job at a time.
- **Hardware: Apple Silicon M1 Max, 32 GB unified memory.** Note that `CLAUDE.md`
  and `docs/ARCHITECTURE.md` both state 36 GB; `hw.memsize` reports 32 GB, and
  `ARCHITECTURE.md:318` performs arithmetic on the wrong figure. Correcting those
  docs is out of scope here but tracked separately.
- **Fire-and-forget.** No human review gate between script generation and audio
  render. Every failure mode must have a decided-in-advance resolution.
- **Sequential model loading.** The pipeline already unloads TTS before loading the
  music engine. Script generation runs *before* any audio model loads, so a local
  LLM takes its turn in that existing sequence rather than contending for memory.

## Non-goals

- Multi-user support, authentication, hosted deployment.
- Durable job state, resume, or a persistent queue. A failed job is re-run by
  pressing the button again.
- Batch prompt submission. Single prompt per run.
- Changing `core/pipeline.py`, the mixer, or any audio path. The auto path calls
  the existing pipeline exactly as the manual tab does.
- Fine-tuning a model on a script corpus. Noted as a future possibility that
  favours staying on open weights; not built here.

## Prior decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scale | Solo content factory | Removes auth, scaling, and cost-control work entirely |
| Review gate | None — fully automatic | User's explicit choice; mitigated by deterministic validation |
| Interface | New tab in existing Gradio app | Reuses loaded models, progress streaming, existing sliders |
| Script generation | Two-pass: generator → independent judge | User's design; judge revises rather than scores |
| Rules | Split prose (LLM) from executable (linter) | Format rules should never cost a token |
| Backend | Pluggable `ScriptEngine` ABC | Mirrors existing `SpeechEngine` ABC; model becomes a config string |
| Model choice | Deferred to a benchmark | Leaderboard churn makes any named model stale by construction |
| Music source | Random from `assets/backgrounds/` | 21 royalty-free tracks already present |
| Render defaults | F5 + uploaded background (golden path) | Known-good combination; `tts_engine` stays a parameter |

### Cost basis for the model decision

Sizing from the repo's own files: the F5 meditation guide is ~5.5K tokens, Kokoro
~4.8K. A 5–7 minute meditation script is only ~350–500 words, because at F5 speed
0.88 the engine runs ~95–100 WPM and much of the runtime is pauses. Estimated
workload per meditation is ~17K input tokens and ~6K output (output dominated by
adaptive thinking).

At frontier API rates that is roughly $0.05–$0.24 per meditation depending on the
model pairing — a spread of about 19 cents. Each meditation also costs 5–10 minutes
of local audio compute. **Model cost is therefore not a deciding factor**; script
quality is. Open-weight models, run locally or on hosted inference, reduce the cost
to near zero but trade iteration speed (a local dense 32B runs ~12–15 tok/s, so
6–8 minutes for both passes versus well under a minute for a frontier API).

The pluggable interface defers this decision to measurement rather than argument.

---

## Architecture

```
core/script_gen/
├── __init__.py            # explicit public exports
├── engine.py              # ScriptEngine ABC + "provider:model" registry
├── adapters/
│   ├── __init__.py
│   ├── openai_compat.py   # Ollama AND hosted open-weight providers
│   ├── anthropic_api.py   # Claude (distinct wire protocol)
│   └── mlx_local.py       # optional: in-process mlx-lm
├── rules.py               # loads guide + safety files, assembles system prompts
├── generator.py           # pass 1: prompt → draft script
├── judge.py               # pass 2: draft → revised script + changelog; also repair
├── linter.py              # deterministic format + safety validation
├── duration.py            # runtime estimate without rendering
└── bench.py               # N models × M prompts → evidence table

core/background_picker.py               # random pick from assets/backgrounds/
core/auto_generate.py                   # orchestrator
docs/prompting_guides/content_safety_rules.md   # new, prose safety rules
scripts/bench_script_models.py          # CLI entry point for bench.py
```

### Adapter consolidation

Ollama, OpenRouter, Together, Fireworks and Groq all speak the OpenAI-compatible
`/v1/chat/completions` protocol. A single `openai_compat.py` with a configurable
`base_url` covers local *and* every hosted open-weight provider — local Ollama is
simply `http://localhost:11434/v1`. Only Claude requires a separate adapter.

Use `httpx` directly against these endpoints rather than adding another SDK
dependency. The Anthropic adapter uses the official `anthropic` package.

### Model naming

Models are configured as `provider:model` strings, with generator and judge set
independently:

```
MOODSCAPE_SCRIPT_GENERATOR=ollama:<model>
MOODSCAPE_SCRIPT_JUDGE=openrouter:<model>
MOODSCAPE_SCRIPT_MAX_REPAIRS=2
MOODSCAPE_TARGET_MIN_SEC=300
MOODSCAPE_TARGET_MAX_SEC=420
```

---

## Components

### `rules.py`

Reads the prompting guide **from disk at call time**, so edits to
`docs/prompting_guides/` take effect on the next generation with no code change.
Selects the file by `(engine, content_type)` — the existing four-way split of
`vocal_{meditation,sleep_story}_{kokoro,f5}_instructions.md` — and appends the new
`content_safety_rules.md`.

### `content_safety_rules.md` (new)

Prose rules fed to both LLM passes. Content covers, at minimum:

- **Invitational phrasing** over imperative: "you might", "if it feels right",
  "when you're ready", "allow" — rather than "you must", "you will".
- **Permission to opt out**: eyes open or closed, freedom to adjust posture or skip
  any instruction.
- **No outcome promises.** "You will be completely calm" sets the listener up to
  experience failure as their own.
- **Trauma-informed defaults**: choice and control throughout, no forced eye
  closing, no compulsory body scan of potentially distressing areas.
- **No clinical framing.** This is not therapy and must not present itself as
  treatment.

### `linter.py`

Deterministic validation in three families. This component is what makes a weaker
or cheaper model viable — format errors are caught in code and repaired, requiring
no model judgment.

**Format**
- `[pause:Xs]` well-formed, `0.5 <= X <= 60`
- Only known tags: `[pause:Xs]`, `[breath]`, `[inhale]`, `[exhale]`
- No markdown (headers, bold, bullets), no emoji
- No ALL-CAPS words longer than 3 characters
- Sentence length within the guide's 8–20 word band
- Chunks under Kokoro's 150-token limit

**Safety hard-blocks**
- Clinical claims: cure, treat, diagnose, "heal your <condition>", "replaces therapy"
- Outcome guarantees: "you will be completely", "this will eliminate"
- Invalidating imperatives: "don't feel", "stop feeling", "there's nothing wrong with you"
- Extended breath-holds (a genuine risk for panic and asthma sufferers)
- Dissociation-adjacent imagery: "leave your body", "float away from yourself"

**Duration**
- Estimate falls within the configured target window

Violations are returned as structured `{code, severity, span, message}` objects and
fed back to the judge as a **targeted repair instruction**, never as a vague
"try again".

### `duration.py`

Estimates runtime without rendering:

- Calls the engine's own `prepare_segments()` and sums pause `duration_sec` exactly
- Estimates speech as `word_count / wpm * 60` — the same formula used at
  `core/f5_tts/engine.py:463`
- Adds the inter-sentence room-tone gaps the engine inserts (0.8 s, or 1.2 s after
  an ellipsis)

Fades are deliberately **not** added: `apply_fades` shapes amplitude over audio that
already exists, so fade-in and fade-out do not extend total runtime.

After every real render it logs estimate-versus-actual, so the WPM constant can be
calibrated from data rather than remaining a guess.

### `background_picker.py`

Picks a random `.mp3` from `assets/backgrounds/`, excluding the last N used so a
batch does not land on the same instrumental repeatedly. Accepts a seed for
reproducibility.

### `auto_generate.py`

Orchestrates the whole flow and is the only module the Gradio tab calls.

---

## Data flow

```
prompt ─▶ rules.build_system_prompt(engine, content_type)
       ─▶ generator.draft()            [model A]
       ─▶ judge.review()               [model B, independent]
       ─▶ linter.check() + duration.estimate()
              │ violations? ─▶ judge.repair(violations) ─┐ (bounded, max 2)
              │ ◀───────────────────────────────────────┘
              ▼ clean
          background_picker.pick()      ─▶ random .mp3 from assets/backgrounds/
       ─▶ MeditationPipeline.generate(script=…, uploaded_music_path=…,
                                      tts_engine="f5", music_model="upload")
       ─▶ output.wav + output.script.txt + output.meta.json
```

Progress streams through the existing `queue.Queue` + `progress_cb` pattern at
`app.py:278`, so the new tab inherits the live progress bar unchanged.

The three artifacts are written as **siblings of the path `MeditationPipeline.generate()`
returns**, sharing its basename — so a run produces `<name>.wav`, `<name>.script.txt`
and `<name>.meta.json` in one place, with no new output-location concept.

`output.meta.json` records the prompt, both model identifiers, the chosen
background track, the judge changelog, any advisory violations, and the estimated
versus actual duration.

### Judge independence

An LLM asked to critique its own output ratifies it far more often than it should.
Two different model families — and especially two different providers — give
genuinely uncorrelated blind spots. The config keeps generator and judge
independent by default, and the bench harness measures whether it matters.

---

## Failure handling

The repair loop is bounded at 2 attempts (`MOODSCAPE_SCRIPT_MAX_REPAIRS`), which
bounds both spend and wall-clock time — the latter matters when a local model takes
3 minutes per pass.

Severities resolve differently. Treating every violation as fatal would make a
weaker model unusable; treating none as fatal would let a safety failure reach
audio.

| Failure | Resolution after repairs exhausted |
|---|---|
| Safety hard-block still present | **Fail the job. Do not render.** |
| Malformed markers or unknown tags | **Fail the job.** Would produce audibly broken audio. |
| Duration outside target window | **Render anyway, log a warning.** |
| Sentence-length or style drift | **Render anyway, log a warning.** Advisory only. |

**Artifacts are written on failure as well as success** — draft, revised script,
changelog, and the violation list. A failed job that can be read is debuggable; one
that vanishes is not.

**Adapter failures produce specific, actionable messages.** "Ollama unreachable at
localhost:11434 — is `ollama serve` running?" rather than a raw
`ConnectionRefusedError`. Network calls retry with capped backoff inside the
adapter; an exhausted retry fails the job cleanly.

---

## Testing

TDD per the project's normal workflow. The key design property: **every unit test
runs without a model or a network call.**

| Test file | Covers |
|---|---|
| `tests/unit/test_script_linter.py` | Table-driven: known-bad scripts → expected violation codes. Where the safety wordlist earns its coverage; extend whenever a new failure mode appears in a real script. |
| `tests/unit/test_duration_estimate.py` | Scripts with hand-computed pause sums; estimate within tolerance. Pure arithmetic. |
| `tests/unit/test_script_engine.py` | ABC contract against a `FakeEngine` returning canned scripts. Proves generator/judge/repair logic without spending a token. |
| `tests/unit/test_background_picker.py` | Picks from the directory, honours exclude-recent, reproducible under a seed. |
| `tests/integration/test_script_gen_live.py` | Opt-in, marked slow, hits a real configured model. Excluded from the default run. |

### Bench harness

`scripts/bench_script_models.py` is a tool, not a test. It runs ~10 fixed prompts
spanning the real range (anxious, sleepless, grieving, unfocused, overwhelmed)
through each candidate model pairing and emits a markdown table of lint pass rate,
duration accuracy, wall-clock time and estimated cost — alongside the scripts
themselves for reading. This is what settles the model question with evidence.

---

## UI

A new "Auto-Generate" tab in `app.py`, alongside the existing manual controls,
which remain available for hand-tuning.

**Inputs:** the natural-language prompt; content type (`meditation` / `sleep_story`,
reusing the existing dropdown); target duration min/max (default 300/420 s);
generator and judge model dropdowns populated from config.

**Outputs:** audio player; the final script, read-only; the judge changelog,
collapsible; the name of the background track chosen; any advisory warnings.

Showing the script *after* the render is not a review gate — it is a record, and it
costs nothing.

---

## Open items for the implementation plan

- Exact wording and initial coverage of `content_safety_rules.md`. The design fixes
  the categories; the prose is written during implementation and expected to grow.
- Initial WPM constants per engine and speed, pending calibration data.
- Whether `mlx_local.py` ships in the first cut or waits until the bench shows a
  local model worth using in-process. `mlx-lm` is installed but undeclared in
  `requirements.txt` — a leftover from the ACE-Step removal — so declaring it is a
  prerequisite if it ships.
