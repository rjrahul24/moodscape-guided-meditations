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

## What it does — Genre path

The "Auto-Generate" tab is **not** a prompt box. Instead:

1. User picks a **genre** from a dropdown (46 total, grouped by family: Sleep & Rest, Stress & Anxiety, Breathwork, etc.).
2. User picks a **length band**: 3–6 min, 6–10 min, or 10–15 min.
3. User clicks **Generate**.

Behind the scenes:

- Preflight check: verify all LLM models exist (Ollama `/api/tags` check, fail in 2s not 5 min).
- Load the genre pack (deterministic TOML file in `docs/genre_packs/<genre_slug>.toml`), validate it.
- Rotate an angle: pick one of 3 distinct metaphorical frames (imagery, pacing) from the pack, excluding the last N used.
- Extract an avoid-list: scan the 5 most recent same-genre scripts in `var/originality/`, pull out distinctive 1–3-grams, TF-IDF weighted (proactive protection). Also load 100 same-genre scripts for cosine comparison and 500 all-genre scripts for IDF weighting (reactive, during validation).
- **Planner** (Pass 0): generator model reads the pack's technique, arc, angle, music tags, pause ratio, safety caveats, plus the avoid-list and duration band. Outputs a prose creative brief (no JSON, no parsing).
- **Writer** (Pass 1): same generator model (already resident) reads the brief, the formatting guide, safety rules, and duration band. Outputs a full script with tags (`[pause:Xs]`, `[breath]`, etc.), formatted and ready to parse.
- **Judge** (Pass 2): independent judge model reviews the script, returns a revised version plus a changelog.
- **Validate** → `linter.check()` (format + safety + originality + banned phrases per genre) + `duration.estimate_duration_sec()`.
- **Repair loop** (if needed, up to 2 repairs): fatal violations go back to the judge with targeted instructions.
- **Pick music** by genre tags: `background_picker.pick_background(prefer_tags=pack.music_tags)` filters the background pool, excluding recently used tracks.
- **Render**: `MeditationPipeline.generate()` unchanged (F5 + uploaded background + full FX stack).
- **Persist**: audio, script, metadata (including `actual_sec` duration for future calibration).
- **Append corpus**: record the rendered script in `var/originality/` for the next run's avoid-list.

The audio, the final script, and a metadata file are written as siblings on disk. No step requires the user to intervene until the finished file is ready.

## Genre Packs

A genre pack is a TOML file in `docs/genre_packs/<slug>.toml` that curates all the creative and deterministic material for one genre. Read it at call time (not import time), so edits take effect on the next generation with no restart. Full contract: [docs/genre_packs/README.md](../genre_packs/README.md).

**Fields that do work downstream:**
- `content_type` ("meditation" or "sleep_story") → routed to `content_profiles.py`
- `music_tags` (exactly 2, one measured + one measured, or measured + declared) → passed to `background_picker.pick_background(prefer_tags=…)`
- `pause_ratio` (0.0–0.6) → tells the planner how to split runtime between words and silence
- `technique`, `arc`, `safety` → rendered into the planner's prompt to shape the brief
- `angles` (exactly 3) → rotation pool; one per run, excluding the last N
- `banned` (3–8 phrases, optional) → matched by substring in the linter; hard-fail if found

The pack decides `content_type` and `music_tags` deterministically, so the planner's output stays plain prose (no structured schema, no parse failures).

### Music Tagging — Two Tiers

**Measured tags** (feature-extracted, regenerable): `dark` / `warm` / `bright`, `drone` / `evolving`, `sparse` / `busy`, `tonal` / `textured`, `struck` / `sustained`, `steady` / `dynamic`. Extracted from the 25%-in point of each background track using `librosa` spectral/temporal analysis (`core/background_tags.py`). Cached by `(filename, size, mtime)` so a newly dropped file is tagged once on first use (~2s), unchanged files are never re-analysed, and replacements are re-tagged automatically.

**Declared tags** (human-written, never overwritten): `piano`, `flute`, `strings`, `nature`, `voice`, `bells`. Keyed by filename in `assets/backgrounds/tags.toml` and survive every re-analysis.

**Matching** (conjunctive): a genre's two tags must both match the background for it to rank. If no background matches both, the full pool is used ("variety is a preference, not a reason to fail"). The manual tab's dropdown uses the unfiltered pool and stays fast.

---

## The Originality Layer

Two-tier system: proactive (prevents bad stories from being written) + reactive (catches regeneration). No two meditations from the same genre are verbatim duplicates. Repeated storylines are defended by angle rotation and avoid-lists, not by a lexical ceiling (cosine similarity cannot distinguish a reworded storyline from a different story at scale).

### Proactive

1. **Angle rotation**: Each genre has 3 angles (distinct metaphorical frames). Each run picks one while excluding the last N (default 3) used for this genre. Two runs therefore use different imagery.
2. **Avoid-list**: Extract from the 5 most recent same-genre scripts. Pull 1–3-grams (word sequences), weight them by TF-IDF *over the entire corpus* (so "notice your breath" gets near-zero weight, distinctive phrases keep full weight), render into the planner's prompt as "do not use".

### Reactive

`linter.check_originality()` measures:

1. **Cosine similarity** (1–3-gram TF-IDF) against the same genre's 100 most recent scripts:
   - **Cosine ≥ 0.80 = FATAL**: Near-verbatim regeneration (0.936 at "lightly edited repeat"). Repair loop.
   - **Cosine 0.65–0.80 = ADVISORY**: Close enough to concern (feeds the next run's avoid-list). Renders anyway.
   - **Cosine < 0.65**: No signal (genuine stories score 0.467, reworded storylines score 0.409 — the bands overlap).

2. **Rare-n-gram overlap**: Longest shared run of 5-grams with document frequency ≤ 2 across the entire corpus. Catches a lifted passage inside an otherwise-different script.

**What cosine cannot do** (calibrated 2026-09-18 against 158-word same-genre scripts): Distinguish a reworded storyline (cosine 0.409) from a genuinely different meditation (0.467). The ordering inverts — rewording destroys n-gram overlap; different stories share stock openings/closings. This is a lexical-vs-semantic limit, not a tuning problem. Answer: defend storyline repetition **proactively** (angle rotation + avoid-list), not reactively. Embedding similarity is deferred.

### Corpus

`var/originality/` (gitignored): On success, `add_to_corpus(script, genre, angle)` records:
- `scripts/{script_id}.txt` — full rendered script (~6 KB each; 1,000 meditations is 6 MB)
- `index.json` — metadata: genre, angle, timestamp, max-similarity (for threshold calibration)

Cold start: With < ~10 scripts, IDF is meaningless; cosine check disabled, only rare-n-gram overlap runs.

---

## The Five Validation Layers

The subsystem is built as five layers, each catching a different class of problem:

0. **Genre pack validation** (`genres.load_pack`): Required fields present, `content_type` is valid, `music_tags` are in the known vocabulary, `angles` count and distinctness, exact 2–3 tags. Malformed pack raises at call time.
1. **Prose rules (LLM-enforced).** `docs/prompting_guides/content_safety_rules.md` plus the per-engine, per-content-type formatting guides (`vocal_{content_type}_{engine}_instructions.md`) are assembled into the planner's and writer's and judge's system prompts by `script_gen/rules.py`. These are read at call time, not import time — see the Gotchas note below.
2. **Linter (code-enforced).** `script_gen/linter.py` re-checks the same rules deterministically: markup, tags, pause bounds, mental-health hard-blocks, originality scoring, banned phrases. This never depends on the model having followed instructions correctly.
3. **Generator + Planner.** `script_gen/planner.py :: plan()` (pass 0: brief only), then `script_gen/generator.py :: draft()` (pass 1: full script).
4. **Judge.** `script_gen/judge.py :: review()` / `repair()` — an independent second model that revises the draft, and later repairs it against named violations.

**Why the split exists.** Format and safety rules living only in the prompt would cost a token (and a chance of being ignored) on every generation, and a model that ignores them has no backstop. Putting the same rules in `content_safety_rules.md` *and* in `linter.py` means the prompt does the persuading — cheaply, since it's plain text with no extra inference — and the linter does the enforcing, for free, without spending a single token. That deterministic backstop is what makes a weaker or cheaper model usable at all: a small local model that gets the format wrong occasionally is fine, because the linter catches it and the judge repairs it, instead of a bad script silently reaching the TTS engine. Genre packs add a fourth layer before any inference: if the pack is malformed, the job fails before the LLM budget is spent.

## Configuration

All of these are read from the environment. `AutoConfig.from_env()` (`core/auto_generate.py`) reads the numeric ones; `run()` reads the model specs directly.

| Env var | Default | Read by |
|---|---|---|
| `MOODSCAPE_SCRIPT_PLANNER` | `ollama:qwen3.8:27b` | `auto_generate.py :: run()` |
| `MOODSCAPE_SCRIPT_GENERATOR` | `ollama:qwen3.8:27b` | `auto_generate.py :: run()` (same model as planner) |
| `MOODSCAPE_SCRIPT_JUDGE` | `ollama:gemma4:31b` | `auto_generate.py :: run()` |
| `MOODSCAPE_ORIGINALITY` | `1` | `linter.check_originality()` kill-switch |
| `MOODSCAPE_ORIGINALITY_FATAL` | `0.80` | `linter.check_originality()` cosine threshold (≥ = FATAL violation) |
| `MOODSCAPE_ORIGINALITY_ADVISORY` | `0.65` | `linter.check_originality()` cosine threshold (≥ = ADVISORY, feeds next avoid-list) |
| `MOODSCAPE_GENRE_PACKS_DIR` | `docs/genre_packs` | `genres.load_pack()` directory |
| `MOODSCAPE_SCRIPT_MAX_REPAIRS` | `2` | `AutoConfig.from_env()` |
| `MOODSCAPE_SCRIPT_MAX_RETRIES` | `3` (total attempts) | Both `script_gen/adapters/openai_compat.py` and `script_gen/adapters/anthropic_api.py` — see "Retrying a flaky provider" below |
| `MOODSCAPE_TARGET_MIN_SEC` (for a given band) | `180` / `360` / `600` | `AutoConfig.from_env()` duration band → seconds mapping |
| `MOODSCAPE_TARGET_MAX_SEC` (for a given band) | `360` / `600` / `900` | `AutoConfig.from_env()` duration band → seconds mapping |

A malformed numeric value (e.g. `MOODSCAPE_SCRIPT_MAX_REPAIRS=abc`) raises `ScriptGenerationError` naming the variable, rather than silently falling back to the default.

### Default models and why they matter

The genre path ships with better defaults than the old prompt path, and they **must be pulled** on first run (unlike the prompt path, which fell back to `llama3.2:3b` that was already in Ollama's default model list). A **preflight check** runs before the planner, querying Ollama's `/api/tags` and failing immediately with an actionable message (`ollama pull qwen3.8:27b`) if a model is missing. Failing in 2 seconds beats failing 5 minutes into generation.

| Model | Size | Rationale | Source |
|---|---|---|---|
| Planner + Writer: `ollama:qwen3.8:27b` | 18 GB | EQ-Bench Creative Writing v3: slop **1.7** (best-in-class), rubric **77.50**, Elo **1668.4**. Judgemark **67.44**. Apache-2.0 license, 256K context. One load, reused for both stages. | Read 2026-09-17 |
| Judge: `ollama:gemma4:31b` | 19 GB | Judgemark **72.31** — 13th overall, above gpt-5.4 and claude-sonnet-5. Best local judge by 5 points. Google Gemma Terms of Use (mutable, unlike Apache-2.0; see section 13.1 of the design spec). | Read 2026-09-17 |

**Two-model independence is load-bearing:** If both planner and judge point to the same model, the judge is grading its own homework. Defaults ensure the property the design is built on. You can swap either one independently via its env var (e.g. `MOODSCAPE_SCRIPT_JUDGE=ollama:qwen3.8:27b` gives an all-Apache-2.0 stack at Judgemark 67.44, or `MOODSCAPE_SCRIPT_GENERATOR=groq:mixtral-8x7b` cuts LLM time to ~15 seconds and frees 18 GB).

**Originality thresholds** (0.80 fatal, 0.65 advisory) are provisional and calibrated on 158-word same-genre scripts. Every run logs the max-similarity score; after a batch of renders, recalibrate if measured values consistently drift from 1.0. The advisory band closes the loop: a near-miss today (0.65–0.80) becomes tomorrow's proactive constraint (avoided in the next run's brief).

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
  The official `anthropic` SDK already retries connection errors, 408, 409,
  429 and 5xx internally with its own exponential backoff, governed by the
  client's `max_retries` constructor argument. The adapter just passes
  `max_retries` explicitly (derived from `MOODSCAPE_SCRIPT_MAX_RETRIES - 1`,
  since the SDK's `max_retries` counts retries, not total attempts) so that
  budget is an intentional, visible choice instead of an unexamined SDK
  default. Wrapping the SDK's own retrying client in a second retry loop
  would multiply attempts (up to `max_retries^2`) and double the backoff for
  no benefit — see the gotcha in `docs/GOTCHAS.md`.

**What counts as transient, in both adapters:** connection errors, timeouts,
HTTP 408, HTTP 409, HTTP 429, and HTTP 5xx. **Never retried:** any other 4xx
(a 400 fails on the first request), a missing API key (raised before any
request is sent), or a malformed/empty response body (the request
succeeded; the content is wrong, so retrying would just reproduce it).

**Latency multiplier at the default budget.** A fully hung provider stalls
roughly `DEFAULT_TIMEOUT_SEC` (300s) x 3 attempts per `openai_compat` call,
and `600s` x 3 attempts per `anthropic` call (the anthropic SDK's own
per-request timeout), and a single `run()` can make up to four model calls
(draft, review, up to two repairs). Anyone raising
`MOODSCAPE_SCRIPT_MAX_RETRIES` should weigh that multiplier against the 30s
backoff cap above.

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
| `CLINICAL_CLAIM` | FATAL | Safety | Language implying the practice cures, heals, treats, or diagnoses a condition, or replaces therapy/medication. |
| `OUTCOME_PROMISE` | FATAL | Safety | A guaranteed emotional outcome ("you will be completely calm", "this will eliminate your stress"). |
| `INVALIDATING` | FATAL | Safety | Instructions like "don't feel anxious" or "there's nothing wrong with you" that dismiss the listener's actual state. |
| `DISSOCIATION` | FATAL | Safety | Dissociation-adjacent imagery ("leave your body", "float away from yourself") — contraindicated for trauma survivors. |
| `BREATH_HOLD` | FATAL | Safety | An instructed breath hold longer than 7 seconds — a real risk for listeners with panic disorder or asthma. |
| `ORIGINALITY_COSINE_HIGH` | FATAL | Originality | Cosine similarity to same-genre corpus ≥ `MOODSCAPE_ORIGINALITY_FATAL` (default 0.80) — near-verbatim regeneration. |
| `ORIGINALITY_RARE_NGRAM` | FATAL | Originality | A lifted 5-gram passage (document frequency ≤ 2 across entire corpus) found in the script. |
| `ORIGINALITY_COSINE_MID` | ADVISORY | Originality | Cosine similarity to same-genre corpus ≥ `MOODSCAPE_ORIGINALITY_ADVISORY` (default 0.65, < fatal threshold) — close enough to concern, seeded into next run's avoid-list. |
| `BANNED_PHRASE` | FATAL | Originality | A genre's `banned` list contains this substring (case-insensitive match). |
| `DURATION_OUT_OF_WINDOW` | ADVISORY | Duration | The estimated runtime falls outside the selected duration band's range. |

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
| Genre pack loader, validator, angle rotation | `core/genres.py` |
| Corpus management, TF-IDF scoring, avoid-list extraction | `core/originality.py` |
| Lazy background music tagging (measured + declared) | `core/background_tags.py` |
| Planner: genre pack + angle → prose brief (pass 0) | `core/script_gen/planner.py` |
| Provider registry + `ScriptEngine` ABC | `core/script_gen/engine.py` |
| Prompt assembly from the on-disk guides | `core/script_gen/rules.py` |
| Draft generation (pass 1) | `core/script_gen/generator.py` |
| Review and repair (pass 2) | `core/script_gen/judge.py` |
| Format + safety + originality + banned-phrase linting | `core/script_gen/linter.py` |
| Duration estimation | `core/script_gen/duration.py` |
| OpenAI-compatible adapter (ollama, openrouter, together, fireworks, groq) | `core/script_gen/adapters/openai_compat.py` |
| Anthropic adapter | `core/script_gen/adapters/anthropic_api.py` |
| Orchestrator: genre + band → finished audio | `core/auto_generate.py` |
| Tag-filtered background selection | `core/background_picker.py` |
| Model-pairing benchmark harness | `core/bench.py`, `scripts/bench_script_models.py` |
| Background tagger (bulk or forced re-tag) | `scripts/tag_backgrounds.py` |
| Genre evaluation harness (matrix of genres × model configs) | `scripts/eval_genres.py` |
| Threaded progress streaming for the UI | `core/streaming_run.py` |
| Gradio "Auto-Generate" tab (genre dropdown, band radio, steer accordion) | `core/auto_tab.py` |
| 46 genre packs (TOML) | `docs/genre_packs/*.toml` |
| Genre pack field contract, tag vocabulary, prose guidelines | `docs/genre_packs/README.md` |
| Safety rules text | `docs/prompting_guides/content_safety_rules.md` |
| Per-engine, per-content-type formatting guides | `docs/prompting_guides/vocal_{content_type}_{engine}_instructions.md` |
| Planner system prompt builder | `core/script_gen/rules.py :: build_planner_system_prompt()` |
| Originality corpus (machine-local, gitignored) | `var/originality/` |

For app.py wiring details, see [app_wiring.md](app_wiring.md). For the full
pipeline this subsystem renders through unmodified, see
[../ARCHITECTURE.md](../ARCHITECTURE.md).
