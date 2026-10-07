# R1: Script-Generation Fixes Design

**Date:** 2026-10-07
**Program:** [Generation Quality Roadmap](2026-10-06-generation-quality-roadmap.md)
**Status:** Design approved in conversation; awaiting spec review

## Purpose

Better scripts going into TTS, by fixing three things:

1. The contradictory instructions the writer and judge receive today.
2. Sampling settings that are wrong for the writer.
3. Gaps in the deterministic safety and style checks.

R1 adds no pipeline stages. Deterministic timing is R3, and best-of-N selection is R4. R1 also gives each genre family a voice suited to it, instead of the alphabetically first voice.

## Success criteria (measured with the R0 harness)

- **Baseline:** a fresh 24-job baseline, run back-to-back with the challenger.
- **Challenger:** the R1 changes, run with `--corpus-from <baseline_out>`.
- **Ship if:**
  - Pass rate does not drop (no new FATAL-driven failures beyond noise).
  - Median slop density per 150 words drops.
  - Median |1 − duration ratio| does not grow.
  - The blind A/B is SHIP or INCONCLUSIVE-leaning-positive. It is never NO-SHIP.
- **Attribution arm:** a second challenger with the voice forced to Brittney for every family (via `voice_map.toml`), to separate the writing effect from the voice effect.

## Non-goals

- Ollama native `/api/chat`. The installed Modelfiles already carry the card's `top_k`/`top_p`/`min_p`, so the `/v1` endpoint suffices.
- LLM safety classifier (R4).
- Exemplar-based prompting (R4).
- Progressive pause curves enforced by code (R3).
- Multi-phase reference recordings. Only the spec for recording them is in R1.
- Any change to the audio path.

## Verified facts this design rests on

- Ollama 0.34.2 serves the models with a 32 768-token context, so there is no truncation today.
- `qwen3.8:27b` Modelfile: `temperature 1`, `top_k 20`, `top_p 0.95`, `min_p 0`, `presence_penalty 0`.
- `gemma4:31b` Modelfile: `top_k 64`, `top_p 0.95`, `temperature 1`.
- Modelfile parameters apply as defaults on `/v1` requests. The request may override `temperature`, `top_p` and `presence_penalty`.
- Reference-clip traits, measured 2026-10-07 with librosa pyin/centroid:

  | Voice | Words/min | F0 median | F0 p90−p10 | Centroid |
  |---|---|---|---|---|
  | Brittney | 109 | 185 Hz | 175 Hz | 1412 Hz |
  | Clara | 106 | 175 Hz | 75 Hz | 1615 Hz |
  | Delilah | 97 | 177 Hz | 139 Hz | 1303 Hz |
  | Eryn | 106 | 191 Hz | 86 Hz | 1864 Hz |

## Components

### 1. Guides rendered from code constants

**`core/script_gen/guide_context.py`**
- `guide_context(engine: str, content_type: str) -> dict[str, str]` returns formatted values from the real constants:

| Placeholder | Source |
|---|---|
| `CHUNK_CHARS` | `core.f5_tts.preprocessor.MAX_CHUNK_CHARS` (F5 guides only) |
| `CHUNK_TOKENS` | `core.kokoro_tts.preprocessor.MAX_CHUNK_TOKENS` (Kokoro guides only; Kokoro chunks by tokens) |
| `SPEED` | `content_profiles.get_profile(ct)["speed"]` |
| `PARA_PAUSE` | `get_profile(ct)[f"{engine}_paragraph_pause_sec"]` |
| `WPM` | `script_gen.duration.DEFAULT_WPM[engine]` |
| `PAUSE_MIN`, `PAUSE_MAX` | `linter.MIN_PAUSE_SEC`, `linter.MAX_PAUSE_SEC` (0.5, 60) |
| `MAX_HOLD` | `linter.MAX_BREATH_HOLD_SEC` (7) |
| `SENT_IDEAL_MIN`, `SENT_IDEAL_MAX`, `SENT_MAX` | 8, 20 (new linter constants `IDEAL_SENTENCE_MIN/MAX_WORDS`), and `linter.MAX_SENTENCE_WORDS` (25) |
| `SLOP_GUIDANCE` | a rendered line plus the top 20 lexicon items (component 3) |

- Pause bounds, max sentence words and the breath-hold limit are already module constants. Only the 8–20 ideal band is new: add it as `IDEAL_SENTENCE_MIN_WORDS`/`IDEAL_SENTENCE_MAX_WORDS` and use it in the `SENTENCE_TOO_LONG` message.

**`rules.load_guide()`**
- Renders `{{NAME}}` from `guide_context`.
- A placeholder missing from the context raises `KeyError` naming it.
- Any `{{` left after rendering raises `ValueError`.
- Guides are still read at call time, so editing the `.md` takes effect on the next run.

**Guide edits** (all four `vocal_*_{f5,kokoro}_instructions.md`):
- Every number with a code source becomes a placeholder.
- Stale claims are removed: the 300-char limit, speed 0.88, the 100–130 WPM rule of thumb, the 1.5 s vs 2.0 s sleep paragraph pause, and the WPM/word-count targets that conflict with `DEFAULT_WPM`.
- Example scripts are rewritten to follow every rule: no ellipses, no outcome promises such as "You are completely at peace", invitational phrasing, an early opt-out, and pauses inside bounds.
- Sleep guide pause guidance becomes progressive: 1.5–2.5 s in the first third, 2–4 s in the middle, 4–8 s in the final third, never more than 10 s.
- Pause guidance is aligned with `pause_ratio` (component 5).

**Drift test** (`tests/unit/test_guide_rendering.py`)
- For each engine × content type:
  - The rendered guide contains no `{{`.
  - It contains the rendered `CHUNK_CHARS`/`WPM`/`PARA_PAUSE` values.
  - It contains none of the stale literals: `"300 char"`, `"300-char"`, `"0.88"`, `"100–130"`, `"100-130"`, `"1.5s in Sleep Story"`.
- The planner prompt (which takes no engine guide) also renders cleanly.

### 2. Per-stage sampling

**`core/script_gen/sampling.py`**
- `@dataclass(frozen=True) SamplingParams(temperature: float, top_p: float | None = None, presence_penalty: float | None = None)`.
- `sampling_for(stage: str, model: str) -> SamplingParams`:
  - `stage` is one of `planner`, `writer`, `judge`, `repair`.
  - The model-family prefix is matched against the bare model name (`qwen3`, `gemma4`).
  - The table:

    | Family | Stages | temperature | top_p | presence_penalty |
    |---|---|---|---|---|
    | qwen3* | planner, writer | 0.7 | 0.80 | 1.5 |
    | qwen3* | judge, repair (if a qwen model judges) | 0.7 | 0.80 | 0.0 |
    | gemma4* | any | 1.0 | 0.95 | None |
    | other | any | 1.0 | None | None (today's behaviour) |

  - Env overrides: `MOODSCAPE_SAMPLING_<STAGE>_<TEMPERATURE|TOP_P|PRESENCE_PENALTY>`. A malformed value raises `ScriptGenerationError` naming the variable, matching the existing `_parse_env_*` convention.

**Plumbing**
- `ScriptEngine.complete(system, user, *, max_tokens, temperature, sampling: SamplingParams | None = None)`. When `sampling` is given, it wins over `temperature`.
- `OpenAICompatEngine` sends `temperature`, plus `top_p` and `presence_penalty` when they are not None.
- `AnthropicEngine` ignores `sampling` (thinking-enabled models reject sampling params) and logs it once at DEBUG.
- `FakeScriptEngine` records `sampling` in `calls` for tests.
- `planner.plan`, `generator.draft`, `judge.review` and `judge.repair` accept and forward `sampling`. `generate_script` computes each stage's params from the engine's model name.
- `meta.json` gains `"sampling": {stage: {...}}`.

### 3. Slop check (ADVISORY)

- The lexicon moves to `core/script_gen/slop.py`: the same data and functions as `core/eval/slop.py`, which becomes a thin re-export so R0 consumers are unchanged.
- **Linter** `check_slop(script) -> list[Violation]`:
  - `SLOP_DENSITY` (ADVISORY) when `slop_density > SLOP_DENSITY_MAX = 2.0`. The message lists the top offending terms.
  - `SLOP_HARD` (ADVISORY) for each distinct hard-tier phrase found.
  - Called from `check()`.
- **Repair integration:** ADVISORY violations never trigger a repair. When a repair happens for FATAL reasons, slop advisories are appended to the problem list under a "style (lower priority)" heading. `<problems>` stripping is unchanged.
- **Writer guidance:** `SLOP_GUIDANCE` renders "Prefer concrete, specific nouns and verbs. Use gently/softly/slowly at most once per 150 words. Avoid: <top 20 hard-tier + rate-limited items>".

### 4. Safety additions

**New global FATAL codes in `linter.check_safety`** (tag-stripped, apostrophe-normalised prose, case-insensitive, as existing):
- `RAPID_BREATHING`: hyperventilat\w*; "breath of fire"; "as fast as (you )?can" within 6 words of breath/breathe/exhale/inhale; a counted burst of ≥ 30 ("thirty|forty|fifty|sixty|[3-9]\d" + "fast|quick|rapid|sharp" + "breaths|exhales"). Kapalabhati phrasing such as "a sharp exhale through the nose" stays allowed.
- `PAIN_ELIMINATION`: "(breathe|send|move|direct) … (into|toward) (the|your) pain … until it (disappears|vanishes|goes away|is gone|dissolves)"; "make (the|your) pain (disappear|vanish|go away)"; "(the|your) pain will (disappear|vanish|go away|dissolve)".
- `ENERGY_HEALING_CLAIM`: "(chakras?|energy|energies|vibrations?|frequenc(y|ies)|aura) (will )?(heal|cure|cleanse|repair)s?".

**New optional genre-pack fields** (validated by `genres.load_pack`):
- `max_breath_hold_sec`: int, default `MAX_BREATH_HOLD_SEC` (7), must be 1–7. The breath-hold check uses the pack's value when a pack is in play. Set to 4 in `panic_sos`, `anxiety_relief` and `overwhelm`.
- `requires_disclaimer`: bool, default false. When true, the script's first 25% of words must contain a contraindication line matching `pregnan|heart|respiratory|lightheaded|light-headed|dizzy` **and** `natural breath|normal breath|breathe normally`. Otherwise FATAL `MISSING_DISCLAIMER`. Set true on all 5 Breathwork packs: box_breathing, breath_awareness, calming_breath_4_7_8, coherent_breathing, energising_breath.
- The pack-scoped checks run via a new `check_pack_rules(script, pack)`. This replaces `check_banned_phrases` at the call site but keeps it as a function.

**Pack data** (no new code):
- Add `"you are safe"` to `banned` in panic_sos, anxiety_relief, overwhelm and grief_and_loss.
- Add `"better place"`, `"everything happens for a reason"`, `"time heals"` to grief_and_loss, loneliness and self_compassion.
- Resulting sizes: panic_sos 10 (already 9 today, so the README's 3–8 limit is not enforced by the validator), grief_and_loss 10, loneliness 10, self_compassion 9, anxiety_relief 7, overwhelm 7. The README limit becomes 3–10, and `genres.load_pack` starts enforcing it (raises above 10).

**Safety rules doc:** `content_safety_rules.md` gains the three new hard blocks, plus the breathwork-disclaimer rule. Prompts and linter stay in sync.

### 5. Sleep pacing alignment

- `pause_ratio` changes from 0.28 to 0.20 in deep_sleep, fall_asleep and back_to_sleep, and from 0.26 to 0.20 in power_nap.
- Sleep guides carry the progressive pause guidance (component 1).

### 6. Per-family voice selection

**`core/voice_selection.py`**
- `measure_voice(audio_path, transcript) -> VoiceTraits(wpm, f0_median, f0_range, centroid)`, using librosa `pyin` (70–400 Hz) and `spectral_centroid` over voiced frames.
- `voice_traits(registry=None) -> dict[slug, VoiceTraits]`, cached in `var/voice_traits.json` keyed by `(filename, size, mtime)`. A changed or new clip is re-measured; unchanged clips never are.
- Profile scores, computed from z-scores across the voice pool:
  - `calm = z(−wpm) + z(−centroid) + z(−f0_median)`
  - `energy = −calm`
  - `expressive = z(f0_range)`
  - `steady = −z(f0_range)`
- `FAMILY_PROFILE`:
  - `calm`: Sleep & Rest, Stress & Anxiety
  - `energy`: Morning & Energy, Body & Movement
  - `expressive`: Emotional, Presence
  - `steady`: Focus & Work, Breathwork
- `voice_for_family(family) -> str`:
  1. An override in `assets/speakers/voice_map.toml` (`[families] "Sleep & Rest" = "Delilah"`) wins. An unknown slug in the override raises.
  2. Otherwise the best-scoring voice for the family's profile, ties broken alphabetically.
  3. An unknown family falls back to `steady`.
  4. A single-voice pool returns that voice.
- With today's clips: Sleep & Rest and Stress & Anxiety → Delilah; Morning & Energy and Body & Movement → Eryn; Emotional and Presence → Brittney; Focus & Work and Breathwork → Clara.

**Wiring**
- `AutoConfig` gains `voice_slug: str | None`.
- `AutoConfig.from_genre` sets it from `voice_for_family(pack.family)` for F5, Chatterbox and Qwen3. An explicit override wins.
- `run()` passes `f5_voice_slug=config.voice_slug` to `pipeline.generate`.
- `meta.json` gains `"voice"`.
- The Auto tab status shows the chosen voice.

**Failure:** if measurement fails (unreadable clip), log a warning and fall back to the alphabetically first voice. A voice problem never fails a job.

### 7. Reference-clip recording spec (doc only)

`docs/setup_and_execution/reference_voice_recording_spec.md`, from the research's artifact (f):
- Capture: 48 kHz/24-bit, delivered as 24 kHz mono.
- Levels: noise floor ≤ −60 dBFS, RMS ≈ −20 dBFS, peaks ≤ −6 dBFS.
- Mic: 15–20 cm, off-axis, treated room.
- Clips: 8–11 s, two complete sentences, a falling final cadence, ≥ 0.5 s of room tone after.
- Transcript: verbatim.
- Phases: opening (~100 WPM) / body (~90) / closing (~80), 3 takes each, wired up via `voices.toml`.
- Profile targets: how a new voice lands in each `voice_selection` profile.

## Error handling summary

- Guide rendering errors (unknown or unrendered placeholder) raise at prompt build, before any LLM call.
- A malformed sampling env var raises `ScriptGenerationError` naming it.
- Invalid new pack fields raise at `load_pack`, like existing validation.
- Voice measurement failure logs a warning and falls back. A bad `voice_map.toml` slug raises at config build.

## Testing

Unit tests, all without real models:
- Guide rendering and the drift test.
- `sampling_for` (table, prefix matching, env overrides, bad env).
- Adapter payloads: the OpenAI-compat request body contains `top_p`/`presence_penalty`; the Anthropic adapter omits them.
- `generate_script` passes per-stage sampling (`FakeScriptEngine.calls`).
- Each new FATAL/ADVISORY code: positive and negative examples, including "a sharp exhale through the nose" passing, and Kapalabhati pack fixtures.
- Pack field validation and the per-pack hold limit.
- `MISSING_DISCLAIMER` first-25% rule.
- Slop advisories added to repair prompts only when a repair runs.
- `voice_selection` with fixture traits (scoring, ties, overrides, unknown slug, single voice, cache invalidation by mtime) and `measure_voice` on synthetic tones.
- `AutoConfig.from_genre` sets `voice_slug`, and `run()` forwards it.

The full suite must stay green.

**Evaluation:** a fresh baseline plus two challengers (R1 full, and R1 with voice forced to Brittney), run overnight under the RAM lock with the user's go-ahead. Then `summary.md` comparison and blind A/B for both arms.

## Deliverables

- Code: `core/script_gen/{guide_context,sampling,slop}.py`, `core/voice_selection.py`, plus edits to the linter, rules, engine adapters, planner/generator/judge, auto_generate, genres and auto_tab.
- Data: the four guides, the safety rules doc, 4 sleep packs, 5 Breathwork packs, and the banned lists of 6 packs.
- Docs: the recording spec, `docs/genre_packs/README.md` (new fields), `docs/auto_generation/README.md` (sampling, voices, new codes), and CLAUDE.md (folder map and gotchas: guides are templated).
