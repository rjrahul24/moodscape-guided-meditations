# Auto-Generation Subsystem

Prompt in, finished meditation out, with no human step in between. This is the
subsystem behind the "Auto-Generate" tab (`core/auto_tab.py`) and its
orchestrator, `core/auto_generate.py :: run()`.

It never touches the audio path. Once a script has been written and
validated, `run()` hands it to `MeditationPipeline.generate()` exactly as the
manual tab does — same arguments, same FX chains, same mix. Everything this
document covers happens *before* that call.

For how the tab itself is wired into `app.py`, see
[app_wiring.md](app_wiring.md). This document covers the generation
subsystem underneath the tab.

## What it does

1. The user types a natural-language request ("I'm feeling anxious and my
   chest is tight").
2. A generator model drafts a full script against the same formatting guide
   and safety rules a human writer would follow.
3. An independent judge model reviews that draft and returns a corrected
   version plus a changelog — it revises, it does not grade.
4. A deterministic linter and duration estimator check the result. Anything
   fatal goes back to the judge as a targeted repair instruction, bounded by
   a repair budget.
5. Once the script is clean (or advisory-only), a background track is picked
   at random from `assets/backgrounds/` and the whole thing renders through
   the existing, unmodified pipeline.
6. The audio, the final script, and a metadata file are written as siblings
   on disk.

No step here requires the user to look at anything until the finished file
is ready.

## The four layers

The subsystem is built as four layers, each catching a different class of
problem:

1. **Prose rules (LLM-enforced).** `docs/prompting_guides/content_safety_rules.md`
   plus the per-engine, per-content-type formatting guides
   (`vocal_{content_type}_{engine}_instructions.md`) are assembled into the
   generator's and judge's system prompts by `script_gen/rules.py`. These are
   read at call time, not import time — see the Gotchas note below.
2. **Linter (code-enforced).** `script_gen/linter.py` re-checks the same
   rules deterministically: markup, tags, pause bounds, and the mental-health
   hard-blocks. This never depends on the model having followed instructions
   correctly.
3. **Generator.** `script_gen/generator.py :: draft()` — one call, one draft.
4. **Judge.** `script_gen/judge.py :: review()` / `repair()` — an
   independent second model that revises the draft, and later repairs it
   against named violations.

**Why the split exists.** Format and safety rules living only in the prompt
would cost a token (and a chance of being ignored) on every single
generation, and a model that ignores them has no backstop. Putting the same
rules in `content_safety_rules.md` *and* in `linter.py` means the prompt does
the persuading — cheaply, since it's plain text with no extra inference — and
the linter does the enforcing, for free, without spending a single token.
That deterministic backstop is what makes a weaker or cheaper model usable
at all: a small local model that gets the format wrong occasionally is fine,
because the linter catches it and the judge repairs it, instead of a bad
script silently reaching the TTS engine.

## Configuration

All of these are read from the environment. `AutoConfig.from_env()`
(`core/auto_generate.py`) reads the numeric ones; `run()` reads the model
specs directly.

| Env var | Default | Read by |
|---|---|---|
| `MOODSCAPE_SCRIPT_GENERATOR` | `ollama:llama3.2:3b` | `auto_generate.py :: run()` |
| `MOODSCAPE_SCRIPT_JUDGE` | `ollama:llama3.2:3b` | `auto_generate.py :: run()` |
| `MOODSCAPE_SCRIPT_MAX_REPAIRS` | `2` | `AutoConfig.from_env()` |
| `MOODSCAPE_SCRIPT_MAX_RETRIES` | `3` (total attempts) | Both `script_gen/adapters/openai_compat.py` and `script_gen/adapters/anthropic_api.py` — see "Retrying a flaky provider" below |
| `MOODSCAPE_TARGET_MIN_SEC` | `300.0` (5 min) | `AutoConfig.from_env()` |
| `MOODSCAPE_TARGET_MAX_SEC` | `420.0` (7 min) | `AutoConfig.from_env()` |

A malformed numeric value (e.g. `MOODSCAPE_SCRIPT_MAX_REPAIRS=abc`) raises
`ScriptGenerationError` naming the variable, rather than silently falling
back to the default.

Provider API keys, from `script_gen/engine.py :: PROVIDER_KEY_ENV`:

| Provider | Env var |
|---|---|
| `openrouter` | `OPENROUTER_API_KEY` |
| `together` | `TOGETHER_API_KEY` |
| `fireworks` | `FIREWORKS_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |

`ollama` needs no key — it is absent from `PROVIDER_KEY_ENV` by design, since
it runs locally.

There are also two test-only environment gates, not part of normal
configuration: `MOODSCAPE_E2E=1` enables the real-audio end-to-end test
(`tests/integration/test_auto_generate_e2e.py`), and
`MOODSCAPE_LIVE_SCRIPT_TEST=1` enables the live-model test
(`tests/integration/test_script_gen_live.py`), which reads
`MOODSCAPE_SCRIPT_GENERATOR` / `_JUDGE` to know what to call.

## Model spec format

Every generator and judge is named by a `provider:model` string, parsed by
`script_gen/engine.py :: parse_engine_spec()`.

**Providers:**
- Five OpenAI-compatible providers, all served by one adapter
  (`OpenAICompatEngine`) because they all speak the same
  `/v1/chat/completions` protocol: `ollama`, `openrouter`, `together`,
  `fireworks`, `groq`.
- `anthropic`, served by its own adapter (`AnthropicEngine`) via the
  Anthropic Messages API, since the wire protocol differs.

**Parsing rule: split on the first colon only.** This matters because Ollama
model tags themselves contain colons — `ollama:qwen3:30b` must split into
provider `ollama` and model `qwen3:30b`, not fail or truncate the tag. A
spec with no colon, or an empty provider/model half, raises `ValueError`
with a message showing the expected shape.

Worked examples:

| Spec | Provider | Model |
|---|---|---|
| `ollama:qwen3:30b` | `ollama` | `qwen3:30b` |
| `anthropic:claude-opus-5` | `anthropic` | `claude-opus-5` |
| `openrouter:meta-llama/llama-3.3-70b` | `openrouter` | `meta-llama/llama-3.3-70b` |

## Retrying a flaky provider

Both adapters retry transient failures, but by deliberately different
mechanisms, and both read `MOODSCAPE_SCRIPT_MAX_RETRIES` (default `3`, total
attempts including the first) so their retry budgets stay in sync:

- **`script_gen/adapters/openai_compat.py`** talks to every OpenAI-compatible
  provider over raw `httpx`, which has no retry behaviour of its own, so the
  adapter hand-rolls a capped exponential-backoff loop: jittered into
  `[0.8, 1.0]` of the base delay so consecutive attempts don't overlap,
  capped at `BACKOFF_CAP_SEC` (30s) so a long backoff — or a hostile/mistaken
  `Retry-After` header — can't stall a job indefinitely, and honours a
  provider's `Retry-After` header when present.
- **`script_gen/adapters/anthropic_api.py`** does **not** add a second loop.
  The official `anthropic` SDK already retries connection errors, 429 and
  5xx internally with its own exponential backoff, governed by the client's
  `max_retries` constructor argument. The adapter just passes
  `max_retries` explicitly (derived from `MOODSCAPE_SCRIPT_MAX_RETRIES - 1`,
  since the SDK's `max_retries` counts retries, not total attempts) so that
  budget is an intentional, visible choice instead of an unexamined SDK
  default. Wrapping the SDK's own retrying client in a second retry loop
  would multiply attempts (up to `max_retries^2`) and double the backoff for
  no benefit — see the gotcha in `docs/GOTCHAS.md`.

**What counts as transient, in both adapters:** connection errors, timeouts,
HTTP 429, and HTTP 5xx. **Never retried:** any other 4xx (a 400 fails on the
first request), a missing API key (raised before any request is sent), or a
malformed/empty response body (the request succeeded; the content is wrong,
so retrying would just reproduce it).

## The failure severity table

This is the load-bearing design decision in the whole subsystem.

| Severity | Meaning | Consequence |
|---|---|---|
| **FATAL** | A safety hard-block or a malformed marker. | If it survives the repair budget, the job **fails and never renders**. Draft, revised script, changelog, and violations are still written to disk (see below) so the failure can be read and debugged. |
| **ADVISORY** | A style issue or duration drift. | Logged as a warning; **the job renders anyway**. |

Treating every violation as fatal would make a weaker or cheaper local model
unusable — it would never clear a strict bar on style alone. Treating no
violation as fatal would let a genuine safety failure (a clinical claim, an
outcome promise, dissociation-adjacent imagery, an unsafe breath-hold
instruction) reach rendered audio. The split is deliberate and should not be
flattened in either direction.

Artifacts are written on failure exactly as they would be on success — this
is intentional, not an oversight, so a rejected run can still be inspected.
`_write_failure_artifacts()` in `core/auto_generate.py` writes, under
`config.failure_dir` (defaults to `<tempdir>/moodscape_failures/`):

- `failed-<timestamp>.script.txt` — the still-fatal script
- `failed-<timestamp>.draft.txt` — the original draft, before any
  judge revision or repair
- `failed-<timestamp>.meta.json` — the prompt, the full changelog across
  every review/repair attempt, and the surviving violations

## Violation code reference

Every code `script_gen/linter.py` can emit. All safety codes and all format
codes except the two advisory ones below are **FATAL**.

| Code | Severity | Family | Trigger |
|---|---|---|---|
| `MARKER_MALFORMED` | FATAL | Format | A `[pause:...]` tag that doesn't match the exact `[pause:Xs]` shape (e.g. missing the `s`, non-numeric). |
| `PAUSE_OUT_OF_RANGE` | FATAL | Format | A well-formed `[pause:Xs]` outside 0.5–60 seconds. |
| `UNKNOWN_TAG` | FATAL | Format | Any bracketed tag that isn't `[pause:Xs]`, `[breath]`, `[inhale]`, or `[exhale]`. |
| `MARKDOWN_PRESENT` | FATAL | Format | Headings, `**bold**`, or bullet/numbered list syntax anywhere in the script — the TTS engine would read the markup aloud. |
| `EMOJI_PRESENT` | FATAL | Format | Any emoji character in the script. |
| `ALL_CAPS` | ADVISORY | Format | A run of 4+ capital letters — capitals change engine pronunciation, but this doesn't block a render. |
| `SENTENCE_TOO_LONG` | ADVISORY | Format | A sentence over 25 words (the guide's ideal band is 8–20). |
| `CHUNK_TOO_LONG` | ADVISORY | Format | Kokoro-only backstop, only emitted when `check_format()`/`check()` is called with `engine="kokoro"` (default `None` skips it, unchanged for every other caller). Runs the real `merge_sentences_to_chunks()` chunker over the script's speech segments and flags a chunk above `MAX_CHUNK_TOKENS` (150 tokens, ~115 words). |
| `CLINICAL_CLAIM` | FATAL | Safety | Language implying the practice cures, heals, treats, or diagnoses a condition, or replaces therapy/medication. |
| `OUTCOME_PROMISE` | FATAL | Safety | A guaranteed emotional outcome ("you will be completely calm", "this will eliminate your stress"). |
| `INVALIDATING` | FATAL | Safety | Instructions like "don't feel anxious" or "there's nothing wrong with you" that dismiss the listener's actual state. |
| `DISSOCIATION` | FATAL | Safety | Dissociation-adjacent imagery ("leave your body", "float away from yourself") — contraindicated for trauma survivors. |
| `BREATH_HOLD` | FATAL | Safety | An instructed breath hold longer than 7 seconds — a real risk for listeners with panic disorder or asthma. |
| `DURATION_OUT_OF_WINDOW` | ADVISORY | Duration | The estimated runtime falls outside `target_min_sec`–`target_max_sec`. Only emitted when an estimate is supplied. |

**`CHUNK_TOO_LONG` rarely fires in practice, by design.**
`merge_sentences_to_chunks()` flushes a chunk before it would exceed
`MAX_CHUNK_TOKENS`, so a multi-sentence chunk can never end up over the
limit — the only way a produced chunk can trip this check is a single
sentence already ~115+ words on its own, and any sentence that long has
already tripped the far cheaper `SENTENCE_TOO_LONG` check (25-word
threshold, roughly a third the size). In testing it took a 120-word single
sentence to trigger `CHUNK_TOO_LONG`, and `SENTENCE_TOO_LONG` fires on that
same script too. So `SENTENCE_TOO_LONG` is the stricter gate in every case
this check can reach; `CHUNK_TOO_LONG` exists as a backstop for a script
that would somehow slip past it, not as an independent detector.

**Wired into `auto_generate.generate_script()`.** It calls `check(..., engine=config.tts_engine)`, so this backstop is live in the production auto-generate flow — it fires for `AutoConfig(tts_engine="kokoro", ...)` and is a no-op for `tts_engine="f5"`, exactly as `check_format()`'s `engine` parameter intends. See
`tests/unit/test_auto_generate.py :: test_chunk_too_long_fires_for_kokoro_but_not_f5`
for an end-to-end check of both branches (also covered directly in
`tests/unit/test_script_linter.py`).

The safety patterns are matched case-insensitively against tag-stripped
prose, with typographic apostrophes (U+2019, U+2018, U+02BC, U+00B4, U+0060)
normalized to ASCII `'` first — a curly quote from an LLM would otherwise
slip `Don't feel anxious` past `INVALIDATING`.

The linter also rejects angle-bracket markup (`ANGLE_TAG`, e.g. a stray SSML
`<break time="2s"/>` or `<emphasis>`): by the time a judge's response reaches
`check_format`, `parse_judge_response` has already stripped the judge's own
`<script>`/`<changelog>` protocol tags, so any surviving `<...>` is genuinely
stray markup the TTS engine would otherwise read aloud.

## How to choose a model

The interface is pluggable on purpose — which generator/judge pairing is
actually good enough is an open, measured question, not a recommendation
this doc makes. `scripts/bench_script_models.py` runs the ten prompts in
`core/bench.py :: BENCH_PROMPTS` (deliberately spanning the harder emotional
cases — grief, overwhelm, numbness — where the safety rules matter most)
through every generator/judge pairing you give it, using the real
generate → judge → lint → repair loop.

Run it:

```bash
python scripts/bench_script_models.py \
    --pair ollama:qwen3:30b ollama:gemma3:27b \
    --pair anthropic:claude-opus-5 anthropic:claude-sonnet-5 \
    --out bench_results.md
```

Each `--pair GENERATOR JUDGE` is one pairing to test; pass it as many times
as you like. `--limit N` runs only the first N benchmark prompts (useful for
a quick smoke test before a full run). The script writes a markdown table
(one row per pairing × prompt) to `--out` and also prints it, followed by a
pass/fail summary and the error text for any failing row.

Each row (`BenchRow`) records: whether it passed, the estimated duration,
how many repairs were used, wall-clock elapsed time, and how many advisory
violations remained. A row failing does not stop the run — pairing
construction and script generation are both isolated per row
(`core/bench.py :: run_bench()`), so one bad spec or one model timeout
doesn't take down the rest of the benchmark.

Read the table for: pass rate, repair count (a pairing that needs 2 repairs
on every prompt is running at the edge of its budget), and elapsed time
(relevant for a local model on Apple Silicon). Read the actual generated
scripts — written alongside a real run via the normal artifact paths, or by
inspecting `outcome.script` if you call `generate_script()` directly — before
trusting a pairing that merely "passes."

## Calibrating `DEFAULT_WPM`

`script_gen/duration.py :: estimate_duration_sec()` estimates spoken runtime
without rendering anything: pauses are summed exactly from the same
`prepare_segments()` the real preprocessor uses, breath/inhale/exhale cues
add their measured sample duration, and prose is estimated as
`word_count / wpm * 60` plus the engine's own inter-sentence/inter-chunk
gaps — modeled per engine, since Kokoro gaps after every sentence while F5
only gaps between the ≤250-char chunks its preprocessor splits a paragraph
into (each chunk usually holds several sentences with no gap between them).
Fades are deliberately **not** added — `apply_fades` shapes amplitude on
audio that already exists, so they don't extend runtime.

`DEFAULT_WPM` currently holds:

| Engine | WPM | Basis |
|---|---|---|
| `f5` | `85.0` | Measured 2026-09-17 from a real render of `tests/integration/test_auto_generate_e2e.py :: REALISTIC_SCRIPT` (198 words, excluding `[pause:Xs]` markers) through the full pipeline. At the old value of 97.0, the estimate was 205.4s against an actual 226.0s — ratio 1.10. 85.0 reproduces the actual duration almost exactly (ratio 1.001). |
| `kokoro` | `105.0` | **Not verified.** No real-render measurement exists for this number yet — treat it as a placeholder, not a calibrated constant. |

**F5's number is voice-dependent.** F5 clones the pacing of its reference
audio, so a markedly faster or slower reference voice will drift from 85.0.
A short script will also mislead you: a 22-word measurement gave a wildly
different implied rate (36.8 WPM) because fixed per-chunk overhead
(reference-audio padding, leading/trailing silence) dominates a 3-chunk
script. Only a script long enough to amortize that overhead — roughly
200–250 words, several chunks — isolates the actual per-word speaking rate.

**The loop is now automatic.** `log_estimate_accuracy(estimated_sec,
actual_sec, engine)` in `script_gen/duration.py` logs (and returns, as a
string) a line of the form
`duration[f5] estimated=205.4s actual=226.0s error=+20.6s ratio=1.100`.
`auto_generate.py :: run()` calls it on every production render: after a
successful `MeditationPipeline.generate()` call, it reads the rendered
file's real duration with `soundfile.info(audio_path).duration` and calls
`log_estimate_accuracy(outcome.estimated_sec, actual_sec, config.tts_engine)`.
The measured `actual_sec` and the derived `estimate_ratio`
(`actual_sec / estimated_sec`) are written into `meta.json` alongside the
existing `estimated_sec`, so every render on disk already carries the data
needed to recalibrate — no manual comparison step required.

That duration read is wrapped in `try`/`except`: a calibration nicety must
never fail a job that already produced audio, so a read failure (an
unreadable file, or in tests a stub pipeline that writes a non-audio
placeholder) is logged at `DEBUG` and swallowed, and `meta.json` gets
`"actual_sec": null` and `"estimate_ratio": null` instead.

**To recalibrate `DEFAULT_WPM` from this data**, pull the `actual_sec` /
`estimated_sec` pairs out of a batch of `meta.json` files for the engine in
question (favor scripts in the 200–250-word range — see the F5 caveat
above), look at whether the ratio is consistently off from 1.0, and adjust
`DEFAULT_WPM[engine]` accordingly. This is exactly how the old F5 value of
97.0 (ratio 1.10, i.e. the estimate ran 10% long) was replaced with 85.0,
which reproduces the actual duration almost exactly (ratio 1.001). You can
still call `log_estimate_accuracy()` by hand for an ad-hoc comparison (or
see `test_duration_estimate.py`) — production renders just no longer require
that manual step.

## Where the pieces live

| Responsibility | File |
|---|---|
| Provider registry + `ScriptEngine` ABC | `core/script_gen/engine.py` |
| Prompt assembly from the on-disk guides | `core/script_gen/rules.py` |
| Draft generation (pass 1) | `core/script_gen/generator.py` |
| Review and repair (pass 2) | `core/script_gen/judge.py` |
| Format + safety linting | `core/script_gen/linter.py` |
| Duration estimation | `core/script_gen/duration.py` |
| OpenAI-compatible adapter (ollama, openrouter, together, fireworks, groq) | `core/script_gen/adapters/openai_compat.py` |
| Anthropic adapter | `core/script_gen/adapters/anthropic_api.py` |
| Orchestrator | `core/auto_generate.py` |
| Random background selection | `core/background_picker.py` |
| Model-pairing benchmark harness | `core/bench.py`, `scripts/bench_script_models.py` |
| Threaded progress streaming for the UI | `core/streaming_run.py` |
| Gradio "Auto-Generate" tab | `core/auto_tab.py` |
| Safety rules text | `docs/prompting_guides/content_safety_rules.md` |
| Per-engine, per-content-type formatting guides | `docs/prompting_guides/vocal_{content_type}_{engine}_instructions.md` |

For app.py wiring details, see [app_wiring.md](app_wiring.md). For the full
pipeline this subsystem renders through unmodified, see
[../ARCHITECTURE.md](../ARCHITECTURE.md).
