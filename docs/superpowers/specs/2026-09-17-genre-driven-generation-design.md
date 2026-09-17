# Genre-Driven Meditation Generation — Design

**Date:** 2026-09-17
**Status:** Approved for implementation planning
**Supersedes:** nothing. Extends the existing auto-generation path.

---

## 1. Problem

The Auto-Generate tab takes a free-text prompt. The goal is to reduce input to
**two clicks**: pick a genre from a dropdown, pick a duration band, press
Generate. Everything downstream — creative brief, script, review, music
selection, audio profile — is derived.

Three constraints shape the design:

1. **Open-source models by default**, running locally on an M1 Max / 32 GB.
2. **Do not rebuild what exists.** `MeditationPipeline.generate()`, the
   linter's fatal/advisory split, the judge-that-revises, `duration.py`'s
   calibration loop and `background_picker`'s exclude-recent behaviour all
   carry over unchanged in spirit and mostly unchanged in code.
3. **No two meditations may be the same.** Repeated runs of one genre must
   differ in storyline, imagery and phrasing, while still being allowed to
   share generic meditation language.

## 2. Non-goals

- Replacing or modifying the audio pipeline. Steps 1-12 of
  `MeditationPipeline.generate()` are untouched.
- Replacing the manual tab.
- Removing the prompt-driven path. It remains, both as an API and as an
  optional "steer this one" field.
- A web service, queue, or multi-user concerns. Single-user, local.

---

## 3. Genre taxonomy

Eight families, 46 leaf genres, derived from the published category taxonomies
of Insight Timer, Headspace and Calm — the best available proxy for what people
actually listen to.

| Family | Genres |
|---|---|
| **Sleep & Rest** (6) | Fall Asleep · Deep Sleep · Racing Mind at Night · Yoga Nidra (NSDR) · Power Nap · Back to Sleep |
| **Stress & Anxiety** (6) | Stress Relief · Anxiety Relief · Panic SOS · Overwhelm · Burnout Recovery · Worry & Rumination |
| **Focus & Work** (6) | Deep Work · Study · Pre-Meeting Calm · Work Break Reset · Creative Flow · Decision Clarity |
| **Body & Movement** (7) | Workout Warm-Up · Workout Cool-Down · Running · Walking · Stretching & Yoga · Body Scan · Pain & Discomfort |
| **Breathwork** (5) | Box Breathing · 4-7-8 Calming Breath · Coherent Breathing · Energising Breath · Breath Awareness |
| **Emotional** (7) | Self-Compassion · Grief & Loss · Loneliness · Anger & Frustration · Confidence · Letting Go · Forgiveness |
| **Morning & Energy** (4) | Morning Start · Intention Setting · Energy Boost · Commute |
| **Presence** (5) | Mindfulness Basics · Loving-Kindness · Gratitude · Open Awareness · Evening Wind-Down |

The dropdown groups by family so the list stays scannable.

---

## 4. Genre packs

One TOML file per genre in `docs/genre_packs/<slug>.toml`, read at call time so
edits take effect on the next run with no restart — the same contract
`rules.py` already uses for the prompting guides.

```toml
# docs/genre_packs/grief_and_loss.toml
label        = "Grief & Loss"
family       = "Emotional"
content_type = "meditation"          # -> selects a content_profiles.py profile
music_tags   = ["piano", "warm", "sparse"]
pause_ratio  = 0.28                  # share of runtime that is silence

technique = """
RAIN, held loosely: Recognise what is present, Allow it without fixing,
Investigate with kindness, Nurture. Never resolve the grief.
"""

arc = ["arrival & permission", "body's weather", "one memory, held lightly",
       "self-kindness", "return"]

safety = """
No stage models. Do not imply closure or timelines. Never instruct the
listener to 'let go' of the person. Offer an exit at every turn.
"""

banned = ["everything happens for a reason", "in a better place",
          "time heals", "move on"]

[[angles]]
name    = "the empty chair"
imagery = ["a chair by a window", "afternoon light", "a cup gone cold"]

[[angles]]
name    = "tidal"
imagery = ["a shoreline at dusk", "waves that arrive and withdraw", "wet sand"]
```

### Fields that do work beyond prompting

| Field | Consumed by | Effect |
|---|---|---|
| `content_type` | `content_profiles.py` | Selects the existing audio profile (pauses, duck, fades) |
| `music_tags` | `background_picker.pick_background(prefer_tags=...)` | Filters the background pool |
| `pause_ratio` | planner prompt | Tells the writer how to split the duration budget between words and silence |
| `angles` | `core/genres.py` | Rotation pool for variety |
| `banned` | `linter.check_banned_phrases()` | Deterministic phrase check, **per-genre only** |

There is deliberately **no global slop list**. Per-genre `banned` lists cover
the phrasings that actually hurt ("in a better place" for grief, "push
through the pain" for workout), and a shared list is one more thing to
maintain for a benefit the writer's slop score of 1.7 already delivers. If
measurement later shows judge-injected slop is a real problem, a shared
`banned` block can be added to the packs directory without a code change.

### Validation

`core/genres.py` validates every pack at load: required fields present,
`content_type` is a key of `CONTENT_PROFILES`, `music_tags` are drawn from the
known vocabulary, `pause_ratio` in [0, 0.6], at least two `angles`. A malformed
pack raises at load time, not mid-run.

---

## 5. Music tagging

### Vocabulary, two tiers

- **`measured`** — written by feature extraction, regenerable:
  `dark` / `warm` / `bright`, `drone` / `evolving`, `sparse` / `busy`,
  `tonal` / `textured`, `struck` / `sustained`, `steady` / `dynamic`.
- **`declared`** — human, never overwritten by the tagger:
  `piano`, `flute`, `strings`, `nature`, `voice`, `bells`.

### Extraction

`librosa` (already a dependency; no new packages). Per track, a 60-second
excerpt taken 25% in, at 22.05 kHz mono:

| Feature | Maps to |
|---|---|
| spectral centroid | `dark` / `warm` / `bright` |
| onset-strength flux | `drone` / `evolving` |
| onset rate | `sparse` / `busy` |
| spectral flatness | `tonal` / `textured` |
| HPSS percussive energy ratio | `struck` / `sustained` |
| RMS p95/p5 ratio | `steady` / `dynamic` |

Thresholds are calibrated against the measured distribution of the existing 20
tracks (recorded in `docs/genre_packs/README.md` so they can be re-derived).

**Known hazard:** onset rate misfires on near-silent drones — one existing
track reports 5.47 onsets/s despite flux of 0.29 and zero detected beats,
because the detector fires on noise floor. `busy` / `sparse` **must** be gated
on flux exceeding a floor, or the calmest beds get labelled busiest.

### Application — automatic and incremental

`core/background_tags.py` tags **lazily**, cached by `(filename, size, mtime)`:
a newly dropped track is analysed once on first use (~2 s) and never again;
a replaced file is re-analysed; `declared` tags are keyed by filename and
survive every re-analysis. `scripts/tag_backgrounds.py` exists for bulk or
forced re-tagging but is never required. **The user's workflow is: drop files
into `assets/backgrounds/`. Nothing else.**

The manual tab's dropdown keeps using the plain `scan_backgrounds()` and does
not get slower.

Matching: `measured` tags are a hard filter, `declared` a soft preference. An
untagged library still works. When nothing matches, the full pool is used —
"variety is a preference, not a reason to fail a job", as `background_picker`
already documents.

---

## 6. Pipeline

### Stages

```
genre + duration band + optional steer
 |- [deterministic] load pack, pick angle (excluding recent), build avoid-list
 |- PLANNER   qwen3.8:27b  -> prose creative brief          )  one load,
 |- WRITER    qwen3.8:27b  -> draft script                  )  no swap
 |                                                          -> unload
 |- JUDGE     gemma4:31b   -> revised script + changelog    )  one load,
 |- [deterministic] lint: format / safety / duration / ORIGINALITY / banned
 |- repair loop (judge, <= max_repairs)                     )  no swap
 |                                                          -> unload
 |- [deterministic] pick background by genre tags
 |- MeditationPipeline.generate()   <- UNCHANGED, golden path (F5 + upload)
 |- persist .wav / .script.txt / .meta.json + append to originality corpus
```

Peak resident memory ~19 GB; **zero** by the time F5-TTS loads.

### Three roles, two models

Research 1 proposes four stages (Planner, Writer, Critic, Finalizer). This
design keeps three, deliberately:

- The judge already **revises** rather than scores. Research 1's own argument —
  "a score does not help a fire-and-forget pipeline" — is the reasoning already
  written into `judge.py`. A Critic/Finalizer split re-introduces the scoring
  step both rejected.
- A separate finalizer means swapping back to the writer: an extra 18 GB load
  per run for no measured gain.
- The `<changelog>` is already the critique artifact.

Planner and writer share one model, so planning costs no swap. The independence
that matters — a model not grading its own homework — is **writer vs judge**,
which is preserved across two families.

### Model assignment and the evidence for it

| Role | Model | Size | Evidence |
|---|---|---|---|
| Planner + Writer | `ollama:qwen3.8:27b` | 18 GB | CW v3: slop **1.7** (best-in-class, tied), repetition 4.2, rubric 77.50, Elo 1668.4. Judgemark 67.44. Apache-2.0, 256K ctx |
| Judge + Repair | `ollama:gemma4:31b` | 19 GB | Judgemark **72.31** — 13th overall, above `gpt-5.4` and `claude-sonnet-5`; best local judge by 5 points |

Benchmarks: EQ-Bench Creative Writing v3 and Judgemark v4, read 2026-09-17.

Alternatives considered and their measured tradeoffs:

| Model | Slop | Repetition | Rubric | Elo | Judgemark |
|---|---|---|---|---|---|
| Muse-Glimmer-30B | 1.7 | **4.0** | **81.30** | **1789.7** | 40.67 |
| Qwen3.8-27B | 1.7 | 4.2 | 77.50 | 1668.4 | 67.44 |
| gemma-4-31B-it | 4.1 | 5.7 | 80.05 | 1366.0 | **72.31** |
| gemma-4-26B-A4B | 4.4 | 6.4 | 80.15 | 1301.6 | 53.02 |
| Nemotron-3.5-Lightning-30B-A3B | 3.1 | **3.5** | 71.95 | 1276.5 | — |

Notes for a future reader:
- **Creative-writing Elo is the wrong benchmark for the judge seat.**
  Muse-Glimmer leads every writing axis and scores 40.67 on Judgemark — a
  distill inherits generative style but loses evaluative discrimination.
- `qwen3.5:9b` and every other current ~9B model have **no published
  creative-writing data at all**. That is why there is no separate small
  planner: the choice would be a guess.
- **Gemma is open-weight, not OSI open source**, unlike Qwen (Apache-2.0).
  This does not affect output ownership or commercial use — see §13.1 for what
  it does and does not mean. The fallback is one env var:
  `MOODSCAPE_SCRIPT_JUDGE=ollama:qwen3.8:27b` gives an all-Apache-2.0 stack at
  Judgemark 67.44, saving 19 GB and one model load.
- Nemotron-3.5-Lightning has the lowest repetition of any local model, which is
  interesting given the originality goal. Harness candidate, not a default.

### Estimated cost per run

Typical, no repairs, on M1 Max (400 GB/s, ~14 tok/s generation, ~250 tok/s
prefill, ~15 s per 18 GB load): **~5.3 minutes of LLM time** before the audio
render. Worst case with two repairs: ~9 minutes. Token budget is dominated by
the 4,700-token system prompt (formatting guide + safety rules) re-sent on
every call.

### Unloading — a hard requirement

`keep_alive` is **silently ignored** on Ollama's `/v1/chat/completions`; it is
honoured only on the native `/api/*` endpoints. Since `engine.py` is built
entirely on the OpenAI-compatible protocol, models would otherwise never
unload, and an 18 GB resident model plus F5-TTS plus Demucs on a 32 GB machine
means swap — which is precisely where this project's MPS deallocation bus
errors live.

Therefore `ScriptEngine` gains:

```python
def unload(self) -> None:
    """Release backend resources. Default: no-op."""
```

No-op for Anthropic and every hosted provider. `OpenAICompatEngine` overrides
it **only when `provider == "ollama"`**, POSTing `{"model": ..., "keep_alive": 0}`
to the native `/api/generate`. The orchestrator calls `unload()` after each
stage. One code path; switching a stage to `groq:` makes the unload vanish.

### The brief is prose, not JSON

Research 1 wants structured planner output because it assumes the planner
decides everything. Here it does not: `content_type`, `music_tags` and
`pause_ratio` come deterministically from the pack. The planner only does
creative variation, so the brief is **plain prose** — no JSON schema, no
Pydantic, no parse-failure path.

### Hosted override

Every stage is a `provider:model` spec. `engine.py`'s existing registry
(ollama / openrouter / together / fireworks / groq / anthropic) already covers
hosted open-weight providers. Setting `MOODSCAPE_SCRIPT_GENERATOR=groq:...`
cuts the script stage from ~5 minutes to ~15 seconds and frees all 32 GB for
the audio pipeline, at roughly $0.004 per run. Local remains the default:
free, private, offline.

---

## 7. Originality

Two layers, because avoidance and detection fail differently.

### Layer 1 — proactive, shapes the brief

The planner receives, with the pack:
- an **angle**, chosen at random while excluding the last few used for this
  genre (the `background_picker` exclude-recent pattern, applied to content);
- an **avoid-list** of distinctive imagery and phrasing harvested from recent
  scripts in this genre, rendered into the prompt as an explicit
  "do not use" list.

### Layer 2 — reactive, a deterministic check

`linter.check_originality()` emits `Violation`s into the existing repair loop.

**The central problem is not flagging generic meditation language.** Every
script legitimately says "notice your breath". TF-IDF self-calibrates against
exactly this: terms appearing across all past scripts get near-zero weight,
distinctive ones keep full weight. No hand-maintained stoplist to rot.

This gives the key split:

- **IDF weights are computed over the entire corpus, all genres** — that is
  what learns which phrases are generic meditation vocabulary.
- **Similarity is measured against the same genre only** (100 most recent) —
  that is where collisions actually occur.

Two signals:

1. **TF-IDF cosine** over 1-3-grams — wholesale similarity, same storyline
   reworded.
2. **Rare-n-gram overlap** — the longest shared run of 5-grams whose document
   frequency is <= 2. Catches a lifted passage inside an otherwise-different
   script, which cosine can miss.

### Severity

Following the fatal/advisory principle CLAUDE.md says not to flatten:

| Signal | Severity | Effect |
|---|---|---|
| cosine > 0.72, or a lifted rare passage | **FATAL** | Repair loop; job fails if unfixable within budget |
| cosine 0.55 - 0.72 | **ADVISORY** | Renders, logs, **and feeds the next run's avoid-list** |

The advisory band closes the loop: a near-miss today becomes tomorrow's
proactive constraint.

### Implementation

Pure Python — `collections.Counter` and `math`. No scikit-learn, no new
dependency. A hundred 1,000-word scripts is milliseconds.

### Corpus

Gitignored `var/originality/`: scripts copied in on success (~6 KB each; 1,000
meditations is 6 MB) plus a small index of genre, angle, timestamp and max
similarity. Self-contained, so it survives output files being moved or deleted.

### Cold start

With fewer than ~10 scripts, IDF is meaningless and everything looks similar.
Below that threshold the cosine check is disabled and only rare-n-gram overlap
runs — that signal needs no corpus statistics.

### Calibration

Thresholds above are **provisional**. Every run logs its max-similarity score
so they can be replaced with measured values — the same pattern
`duration.py :: log_estimate_accuracy()` established for `DEFAULT_WPM`.

### Dependency on an existing fix

Commit `c372b18` stopped the repair loop poisoning itself by echoing
`<problems>` back into the script. An originality violation **must** quote the
offending phrases so the judge can remove them, which rides straight into that
hazard. The mitigation already exists: `parse_judge_response()` strips
`<problems>` before anything else. **This behaviour is load-bearing for
originality repair and must not be removed.**

---

## 8. UI

The Auto-Generate tab's free-text box is replaced by:

- **Genre** — dropdown grouped by the eight families, 46 entries.
- **Length** — radio, three options:

| Band | `target_min_sec` | `target_max_sec` |
|---|---|---|
| 3-6 min | 180 | 360 |
| 6-10 min | 360 | 600 |
| 10-15 min | 600 | 900 |

Content Type and Voice Engine become **overrides**, pre-filled from the genre
pack's `content_type` and still user-changeable. The prompt box survives in a
collapsed "Steer this one" accordion, empty by default, appended to the brief
when used.

### Backward compatibility

`auto_generate.run()` keeps its current signature and gains an optional
`genre=`. With a genre the planner stage runs; without one the existing
prompt path behaves exactly as today. `scripts/generate.py` and every existing
test keep working untouched.

---

## 9. Configuration

```
MOODSCAPE_SCRIPT_PLANNER        ollama:qwen3.8:27b
MOODSCAPE_SCRIPT_GENERATOR      ollama:qwen3.8:27b    # was ollama:llama3.2:3b
MOODSCAPE_SCRIPT_JUDGE          ollama:gemma4:31b     # was ollama:llama3.2:3b
MOODSCAPE_ORIGINALITY           1                     # kill switch
MOODSCAPE_ORIGINALITY_FATAL     0.72
MOODSCAPE_ORIGINALITY_ADVISORY  0.55
MOODSCAPE_GENRE_PACKS_DIR       docs/genre_packs
```

Existing `MOODSCAPE_TARGET_MIN_SEC` / `_MAX_SEC` / `_SCRIPT_MAX_REPAIRS` /
`_SCRIPT_MAX_RETRIES` are unchanged.

### Preflight

Changing the defaults off `llama3.2:3b` means the first run fails for anyone
without the models pulled. A preflight check runs **before the planner**,
querying Ollama's `/api/tags` and failing immediately with an actionable
message (`ollama pull qwen3.8:27b`). Failing in two seconds beats failing five
minutes in. The preflight is skipped for non-Ollama providers.

---

## 10. File map

**New**

```
docs/genre_packs/*.toml          46 packs + README (incl. tagger thresholds)
core/genres.py                   load/validate packs, angle rotation
core/originality.py              corpus, TF-IDF, similarity
core/background_tags.py          lazy tagging + (name,size,mtime) cache
core/script_gen/planner.py       pass 0: pack + angle -> prose brief
scripts/tag_backgrounds.py       bulk / forced re-tag
scripts/eval_genres.py           model-config comparison harness
assets/backgrounds/tags.toml     generated; declared tags hand-edited
var/originality/                 corpus (see .gitignore note below)
```

`var/` must be added to `.gitignore`. The corpus is machine-local state, and
committing generated scripts would both bloat the repo and make the
originality corpus differ between clones.

**Changed**

| File | Change |
|---|---|
| `core/script_gen/engine.py` | `unload()` on the ABC |
| `core/script_gen/adapters/openai_compat.py` | Ollama `unload()` implementation |
| `core/script_gen/rules.py` | `build_planner_system_prompt()` |
| `core/script_gen/linter.py` | `check_originality()`, banned-phrase check |
| `core/auto_generate.py` | planner stage, genre path, originality wiring, corpus append, preflight |
| `core/background_picker.py` | `prefer_tags=` argument |
| `core/auto_tab.py` | genre dropdown, duration radio, steer accordion |
| `CLAUDE.md`, `docs/auto_generation/README.md` | document the genre path |

---

## 11. Testing

Unit, all with `FakeScriptEngine`, no network, no model weights:

- **Every pack parses**, has required fields, references only known music tags,
  and names a valid `content_type` — a loop over the directory, so a malformed
  pack cannot reach production.
- Angle rotation excludes recent and degrades to the full pool rather than
  failing.
- Originality math against three fixtures: near-duplicates flagged, genuinely
  different passes, **and a pair sharing only generic meditation language
  passes**. The third is the fixture that matters.
- Cold-start behaviour: with a corpus below threshold, only rare-n-gram
  overlap runs.
- `unload()` is called once per stage, verified with a spy engine.
- Tagger thresholds against synthetic signals (pure tone, white noise, pulse
  train) — deterministic, no audio fixtures. Includes the near-silent-drone
  case that must not be labelled `busy`.
- Duration band -> seconds mapping.
- Preflight fails fast and names the missing model.

Integration, in a new `tests/integration/test_genre_e2e.py` with its own stub
pipeline. It does not extend `test_auto_generate_e2e.py`: that file is gated
behind `MOODSCAPE_E2E=1` and drives the real pipeline, so anything added there
is skipped by default. The stub-pipeline pattern to copy lives in
`tests/unit/test_auto_generate.py`.

- genre -> audio end to end with a stub pipeline;
- **two runs of one genre with an engine returning identical text must produce
  a fatal originality violation** — the test that encodes the core requirement.

---

## 12. Evaluation harness

`scripts/eval_genres.py` renders a matrix of genres x model configurations,
writing scripts, audio and per-run metrics (slop-phrase hits, max originality
cosine, estimate-vs-actual duration) into one comparison directory.

This settles, with listening rather than benchmark tables:
- writer: `qwen3.8:27b` vs `muse-glimmer:30b` vs `Nemotron-3.5-Lightning`;
- judge: `gemma4:31b` vs an all-Apache-2.0 `qwen3.8:27b`;
- originality thresholds.

Research 1's closing advice — build the harness before trusting any
configuration — is correct and is adopted here.

---

## 13. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **No energetic music.** All 20 tracks are ambient drones; Workout / Running / Energy Boost have nowhere to land. | Falls back to calm beds. Fixed by dropping files in — tags apply themselves. Data change, not code. |
| 2 | **46 packs is real authoring work**, and pack quality caps output quality. | Drafted up front, plain TOML, read at call time, editable without restart. |
| 3 | **Gemma's terms are mutable** in a way Apache-2.0's are not, and distribution obligations attach if this is ever bundled and shipped. | Low impact today; reversible with one env var. Full analysis and revisit triggers in §13.1. |
| 4 | **37 GB of models** on a volume at 91% capacity. | Preflight names what is missing; config C saves 19 GB. |
| 5 | **Originality thresholds are provisional.** | Every run logs its score; calibrate as `DEFAULT_WPM` was. |
| 6 | **The judge is the sloppiest local writer** (slop 4.1) and holds the pen last. | The judge prompt already says "preserve what works"; the banned-phrase and slop checks catch injected slop regardless of source. |
| 7 | **~5.3 min of LLM time per run** may frustrate iteration. | Per-stage hosted override drops it to ~15 s without a code change. |

### 13.1 On Gemma's licence

Gemma ships under Google's Gemma Terms of Use, not an OSI-approved licence.
That phrase overstates the exposure on its own, so this records what it
actually means for this project. Clause references are to the terms as read
2026-09-17.

**What it does not mean**

- **Output ownership is not affected.** §3.3: "Google claims no rights in
  Outputs you generate using Gemma." Generated scripts and audio are
  unencumbered.
- **Commercial use is permitted.** Nothing in the terms prevents selling what
  this produces.
- **The distribution obligations in §3.1** — pass the terms through, include a
  notice file, bind downstream recipients, mark modified files — attach *when
  distributing the model or a derivative*. This project distributes neither. It
  ships code that instructs Ollama to pull weights from Google, so the user
  accepts Google's terms directly.

**What is real**

1. **The terms are mutable; Apache-2.0 is not.** §4.1 lets Google update Gemma,
   and §3.2 incorporates a Prohibited Use Policy Google can revise. Apache-2.0
   is irrevocable, so nothing can be added to Qwen's terms retroactively. This
   is the one structural difference, and it concerns the future rather than the
   present.
2. **Distribution is a cliff, not a slope.** Bundling this as a Docker image,
   a packaged desktop app, or a Gemma fine-tune makes §3.1 apply at once,
   binding downstream users to a policy Google controls.
3. **Enterprise friction.** Apache-2.0 clears legal review unread; Gemma's
   terms require reading, and some organisations refuse non-OSI licences
   outright. Relevant only if someone other than the author adopts this.
4. **§3.2's remote-restriction reservation has little force over local
   weights.** There is no mechanism reaching a local GGUF. It matters for
   hosted Gemma.

**Commercial use is permitted**

Confirmed 2026-09-17 against both documents. Terms 3.3 — "Google claims no
rights in Outputs you generate using Gemma" — and neither the Terms nor the
Prohibited Use Policy carries any blanket commercial restriction. Meditations
generated with Gemma-4 may be sold.

**Where the Prohibited Use Policy brushes this project**

Two clauses are relevant, both quoted verbatim because the paraphrase
("unqualified medical advice") is broader than the real wording and would
mislead a future reader. Under misinformation:

> "Misleading claims of expertise or capability made particularly in sensitive
> areas (e.g. health, finance, government services, or legal)"

and separately:

> "Engaging in the unauthorized or unlicensed practice of any profession
> including, but not limited to, financial, legal, medical/health, or related
> professional practices."

These are **conduct restrictions, not content-category bans**. Guided
meditation is not the practice of medicine and is not a licensed profession in
any relevant jurisdiction. What would cross the line is framing: claiming
clinical efficacy ("cures anxiety", "clinically proven"), positioning the
product as a substitute for therapy, or implying the narrator is a licensed
clinician. That is a marketing constraint, not a script-generation one.

Critically, **these obligations exist independently of Gemma.** A commercial
wellness product making clinical claims is an advertising-law problem whether
its scripts came from Gemma, Qwen, or a human writer. The PUP adds essentially
nothing to what already applies.

This is already mitigated for reasons that predate the licence question:
`content_safety_rules.md` and the linter's safety **hard-blocks** exist to stop
clinical claims reaching audio. Those checks must not be weakened.

This section is a reading of the terms, not legal advice. Before a commercial
launch involving revenue or an app-store listing, have a lawyer read the Gemma
Terms once.

**Decision (2026-09-17): Gemma-4-31B stays the judge.** Commercial use is
confirmed permitted, the PUP clauses are narrower than first assumed and are
already covered by existing safety checks, and the Judgemark advantage is
worth keeping. The revisit triggers below still stand.

**Why this is not a blocking risk**

The cost of being wrong is one environment variable. Switching the judge to
`ollama:qwen3.8:27b` yields an all-Apache-2.0 stack the same day, trading 5
Judgemark points (72.31 -> 67.44) for 19 GB of disk and one fewer model load.
This is a future-optionality concern, not a licence hazard.

**Revisit if any of these becomes true**

- The project is packaged for distribution with weights included, or a Gemma
  derivative is fine-tuned and shared.
- Anyone outside the author adopts it, particularly inside an organisation.
- Google revises the Prohibited Use Policy in a way that touches wellbeing or
  mental-health content.
- The judge's measured advantage narrows — the §12 harness may show Qwen close
  enough that the licence question settles itself.

---

## 14. Rejected alternatives

- **Four stages (Critic + Finalizer split).** Re-introduces the scoring step
  the existing judge deliberately replaced, and adds an 18 GB reload per run.
- **Structured/JSON creative brief.** Unnecessary once the pack owns the
  deterministic fields; removes a parse-failure path.
- **A separate small planner model.** No current ~9B model has published
  creative-writing data; the choice would be unevidenced, and the writer model
  is already resident.
- **Muse-Glimmer-30B as judge.** Highest local creative Elo but Judgemark
  40.67. Creative-writing score is the wrong benchmark for the judge seat.
- **LangGraph / CrewAI orchestration** (Research 2/3). The existing
  `auto_generate.run()` is a linear function with a bounded repair loop; a
  graph framework adds a dependency and indirection for no capability gained.
- **Loading MLX in-process.** This codebase already fights Metal deallocation
  bugs; a second Metal-allocating framework in the Gradio process invites a new
  class of crash. Out-of-process (Ollama, which already speaks the
  OpenAI-compatible protocol) costs no new code. Ollama's MLX-optimised model
  variants give most of the speed benefit anyway.
- **scikit-learn for TF-IDF.** ~60 lines of pure Python avoids a heavy
  dependency, in keeping with this project's dependency hygiene.
