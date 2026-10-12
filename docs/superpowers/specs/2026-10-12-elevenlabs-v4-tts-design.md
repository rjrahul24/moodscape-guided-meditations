# ElevenLabs v4 TTS Provider — Design Spec

**Date:** 2026-10-12 · **Branch:** `dev-test-e11labs` · **Status:** approved in brainstorming, awaiting spec review

## 1. Goal

Add ElevenLabs (`eleven_v4`) as a cloud TTS provider beside the existing local
engines (F5, Kokoro, Chatterbox). A top-of-app toggle — **ElevenLabs | Local
Models** — switches the TTS stage for both the Manual and Auto-Generate tabs.
Everything other than TTS (planner, writer, judge, linter, music, mix, master,
export) is unchanged.

Two repo-wide changes ship with it, requested during design:

1. **No vocal sounds in any output.** No breath/inhale/exhale samples, no
   sighs, no reaction tags — for every engine.
2. **Post-generation Remix.** Voice-level and music-level trims applied after a
   run, re-running only mix → master → export from saved stems (both providers,
   Manual tab).

### Success criteria

- ElevenLabs renders flow through the same music/mix/master/export path and
  sound at least as good as the old `moodscape-mix-lib` output.
- **Local Models is the default** and the F5 + uploaded-background golden path
  is unchanged except for the intended breath-sound removal (LUFS baseline
  verified).
- A changed paragraph re-bills only that paragraph; a full re-run with identical
  inputs bills 0 characters.
- Remix at 0 / 0 dB reproduces the original mix; it never overwrites the
  original file.

### Decisions made in brainstorming

| Topic | Decision |
|---|---|
| v4 audio tags | Injected deterministically at TTS time; scripts stay engine-neutral; script gen untouched |
| Tag content | **Voice-quality directions only** (e.g. `[soft, slow, calm voice]`); never reaction/sound tags |
| Voices | Seed with the old repo's curated list; user supplies v4 voice IDs later; family map deferred |
| Voice FX | Light clean chain: no DeepFilterNet, no mastering engine; HPF + gentle de-ess + Reverb slider (default 0) |
| Cost guard | Per-chunk cache + whole-render cache + character estimate in UI/meta; no hard cap |
| Model | `eleven_v4` only (env override for experiments) |
| UI controls | Voice, Stability, Similarity, Delivery direction; Speed/WPM/Expressiveness hidden |
| Remix | Both providers, Manual tab only |
| Integration | Approach A — native 48 kHz `SpeechEngine` |

## 2. Research findings (Eleven v4, released 2026-09-28)

Sources: [What is Eleven v4](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/what-is-eleven-v4),
[Audio tags v3/v4](https://elevenlabs.io/docs/help-center/product/core-capabilities/text-to-speech/how-do-audio-tags-work-with-eleven-v3-and-v4),
[Eleven v4 blog](https://elevenlabs.io/blog/eleven-v4),
[TTS best practices](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices),
[Models](https://elevenlabs.io/docs/overview/models),
[TTS convert API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert),
[Morphic v4 guide (3rd party)](https://morphic.com/resources/how-to/eleven-v4-guide).

- Model IDs: `eleven_v4` (long-form/narration) and `eleven_v4_turbo` (real-time).
- **10,000 chars/request** (v3: 5,000). 90+ languages.
- **Only Stability and Similarity apply.** "Style and Speed sliders are not
  available in Eleven v4, and SSML is not supported." → all pacing comes from
  text (direction tags, punctuation, ellipses) and programmatic silence.
- Tags: free-text in square brackets, placed where delivery should change;
  descriptive voice-quality phrasing (`[low, gravelly voice]`) avoids being
  misread as a sound effect. Best-effort, voice-dependent ("A meditative voice
  shouldn't shout").
- **Request stitching is back** (v3 rejected it): "significantly more reliable
  in Eleven v4". API accepts `previous_request_ids` (max 3) and
  `previous_text`/`next_text` (ignored when request ids are sent).
- `seed` 0–4294967295, best-effort determinism. `output_format` includes
  `pcm_48000`. `apply_text_normalization` auto/on/off.
- IPA in `/slashes/` is understood (not used in v1 of this feature).
- One secondary source says older IVC/PVC clones may need retraining for v4 —
  preflight voice validation (§6) surfaces this per voice.

Carried over from the old repo (`moodscape-mix-lib/meditation_mixer/tts.py`):
programmatic silence for long pauses, 30 ms equal-power crossfades,
per-chunk LUFS normalization, content-addressed per-chunk PCM cache
(request ids excluded from the key), retry with backoff on transient errors,
curated voice list. **Dropped:** `speed=0.80`, `style`, `[slowly]`/`[soft]`
tone-preset tags as separate tokens, `_inject_sentence_pauses` ellipsis
injection (re-evaluate by ear later), Rubber Band time-stretch.

## 3. Architecture

### 3.1 New package `core/elevenlabs_tts/`

```
core/elevenlabs_tts/
├── __init__.py        # explicit exports: ElevenLabsEngine, prepare_segments, list_voices, ...
├── client.py          # SDK wrapper: auth, retries, request-id capture, voice validation
├── voices.py          # voices.toml loader → ElevenLabsVoice dataclass, default, family_map
├── preprocessor.py    # prepare_segments(): shared parser → API-ready chunks + pauses
├── direction.py       # delivery-direction sanitizer (rejects vocal-sound words)
├── chunk_cache.py     # per-chunk PCM cache in var/elevenlabs_cache/
├── engine.py          # ElevenLabsEngine(SpeechEngine), sample_rate = 48000
└── postprocessor.py   # build_elevenlabs_voice_chain()
assets/elevenlabs/voices.toml
```

**`voices.toml`** (tracked):

```toml
default_voice = "pjcYQlDFKMbcOUp6F5GD"   # Brittney (matches local default narrator)

[[voice]]
id = "pMsXgVXv3BLzUgSXRplE"
name = "Serena"
notes = "female · warm, meditative · stock"
# ... Clara, Valory, Brittney, Priyanka, Eryn, AImee, Drew, Archer (IDs from old repo)

[family_map]   # genre family → voice id; empty until the user's v4 review
```

`voices.py` exposes `list_voices()`, `default_voice_id()`,
`voice_for_family(family) -> str` (returns `family_map[family]` if present, else
the default). Unknown ids in `family_map` raise at load with the known list.

### 3.2 `client.py`

- Loads `ELEVENLABS_API_KEY` (via existing `.env` loading). Missing →
  `ElevenLabsConfigError("ELEVENLABS_API_KEY is not set …")`.
- `convert(text, voice_id, *, model_id, stability, similarity, seed,
  previous_request_ids) -> (pcm_bytes, request_id)` using
  `text_to_speech.with_raw_response.convert(output_format="pcm_48000", …)`;
  request id from headers `request-id` / `x-request-id` / `x-trace-id`.
- Retries: status `None` (network/timeout), `429`, `>=500` → exponential
  backoff (1, 2, 4 s), `MOODSCAPE_ELEVENLABS_MAX_RETRIES` (default 3). Other
  4xx raise `ElevenLabsAPIError(status, message)` immediately.
- `validate_voices(ids) -> dict[id, bool]` via `GET /v1/voices/{id}`; results
  memoized per process.
- `apply_text_normalization="auto"`; `use_speaker_boost=True`.

### 3.3 `preprocessor.py`

`prepare_segments(script, content_type="meditation", direction=None) -> list[dict]`

1. Reuse the shared bracket grammar by calling
   `core.f5_tts.preprocessor.parse_script(script, paragraph_pause_sec=profile["elevenlabs_paragraph_pause_sec"])`
   (pauses, `[voice:…]`, breath markers, paragraph breaks — identical
   semantics to F5).
2. Text normalization: `core.text_utils.expand_text` only (no F5 colon/dash
   rewrites — v4 handles punctuation; ellipses and em-dashes are kept as
   pacing cues).
3. **Breath segments → `{"type": "pause", "duration_sec": BREATH_SEC[subtype]}`**
   (merged with adjacent pauses by max, as the parser already does).
4. Merge consecutive speech segments (between pauses) into chunks up to
   `MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS` (default 1500), splitting oversize
   speech on sentence boundaries, never inside `[...]`; hard ceiling 9,000
   chars (< 10,000 API limit) including the direction prefix.
5. Prefix **every** speech chunk with the sanitized direction
   (`direction or env DIRECTION or "[soft, slow, calm voice]"`).
6. Return segments of the standard shape: `{"type": "speech", "text": str}` and
   `{"type": "pause", "duration_sec": float}`. No `"breath"` type is emitted.

Also exports `estimate_billed_chars(segments) -> int` (sum of speech `text`
lengths, i.e. what is sent).

### 3.4 `direction.py`

`sanitize_direction(text) -> str`: trims, wraps in `[...]` if unbracketed,
collapses to a single bracket group, max 80 chars. Raises
`ValueError` if it contains any vocal-sound word (case-insensitive, word
boundary): `breath, breathe, breathes, breathing, inhale, inhales, exhale,
exhales, sigh, sighs, gasp, gasps, laugh, laughs, chuckle, yawn, yawns, hum,
hums, cough, sniff, gulp, swallow, whimper, sob, crying, cries`. The UI shows
the error inline; env value errors fail at preflight.

### 3.5 `chunk_cache.py`

- Dir `var/elevenlabs_cache/` (override `MOODSCAPE_ELEVENLABS_CACHE_DIR`).
- Key = SHA-256 of JSON `{text, voice_id, model_id, stability, similarity,
  seed, output_format}` (sorted keys). **Request ids are not in the key.**
- Files: `<key>.pcm` (atomic write via `.part` + rename), `<key>.json` (payload
  + `request_id` + `created_at`).
- `load(payload) -> (pcm, request_id | None) | None`, `store(payload, pcm, request_id)`,
  `is_cached(payload) -> bool`.

### 3.6 `engine.py` — `ElevenLabsEngine(SpeechEngine)`

```python
class ElevenLabsEngine(SpeechEngine):
    sample_rate = 48000
    def __init__(self, voice_id=None, stability=None, similarity=None, direction=None, model_id=None, seed=None): ...
    def load_model(self): self._client = ElevenLabsClient()        # no RAM cost
    def unload_model(self): self._client = None
    def synthesize(self, segments, voice=None, speed=None, progress_cb=None, **kw) -> (audio48k, activity)
    def get_available_voices(self): ...
    last_manifest: dict   # chars_billed, chars_cached, cache_hits, cache_misses, request_ids, model_id, voice_id, settings
    last_qa = None
```

`synthesize` (`speed` accepted and ignored — v4 has no speed):

1. For each speech segment: build cache payload; hit → reuse PCM and stored
   request id; miss → `client.convert(..., previous_request_ids=recent[-3:])`
   when `MOODSCAPE_ELEVENLABS_STITCH=1` (default), then store.
   `recent` is appended from both hits and misses so continuity survives
   partial re-renders.
2. PCM int16 → float32. Per-chunk LUFS normalize to
   `MOODSCAPE_ELEVENLABS_CHUNK_LUFS` (default −21; `off` disables); passthrough
   if < 400 ms or measured < −50 LUFS.
3. Activity mask per chunk: frame energy gate (20 ms frames, active where
   frame RMS > chunk-peak-frame − 35 dB, 150 ms hangover) so pauses *inside* a
   chunk (ellipses, sentence gaps) read as inactive and the bed can rise.
4. Pause segments → zeros (mask False).
5. Stitch all pieces with `MOODSCAPE_ELEVENLABS_XFADE_MS` (default 30)
   equal-power crossfades; the mask is concatenated with the same overlap
   accounting (OR across crossfade region) so `len(mask) == len(audio)`.
6. On API failure mid-run: raise `ElevenLabsAPIError` whose message includes
   "N of M chunks rendered before failure (cached; re-run resumes)".

### 3.7 `postprocessor.py`

Two pieces, applied at the points where local engines use their mastering
engine and voice chain respectively:

- `deess_elevenlabs(audio, sr) -> np.ndarray`: calls the existing numpy
  `core.f5_tts.postprocessor.split_band_deess` (6.5 kHz / 11 kHz bands). Gated
  by `MOODSCAPE_ELEVENLABS_DEESS` (default `1`; `0` = off). Runs in the
  pipeline slot where `mastering_engine.master_vocals` runs for F5.
- `build_elevenlabs_voice_chain(reverb_amount: float) -> Pedalboard`:
  HighpassFilter(70 Hz), plus — only when `reverb_amount > 0` — the same
  parallel convolution-reverb Mix block `build_f5_voice_chain` uses (factor that
  block into a shared helper rather than duplicating it). No compression, no EQ
  tilt, no limiter. Applied via the existing `apply_fx`.

### 3.8 Changes to existing modules

| File | Change |
|---|---|
| `core/speech_engine.py` | Add class attr `sample_rate: int = SAMPLE_RATE` (24000) + docstring: engines may override; pipeline honours it. |
| `core/engine_registry.py` | `BUILTIN_ENGINES["elevenlabs"]` (`base_engine="elevenlabs"`, `provider="cloud"`, presets `reverb_amount=0.0`); add `provider="local"` to the other three; `get_engine("elevenlabs", **kw)`; `list_engines(provider=None)`; `list_all_engines()` keeps returning local engines only (it feeds the Local Models radio). |
| `core/pipeline.py` | (a) `base_engine` accepts `"elevenlabs"`; (b) preprocessor branch → `elevenlabs_tts.prepare_segments(script, content_type, direction=elevenlabs_direction)`; (c) new kwargs `elevenlabs_voice_id`, `elevenlabs_stability`, `elevenlabs_similarity`, `elevenlabs_direction` passed to the engine and into the cache key; (d) `tts_sr = getattr(tts, "sample_rate", SAMPLE_RATE)` or cached `sample_rate`; skip upsample when `tts_sr == mix_sr`, otherwise resample from `tts_sr` (mask repeat factor derived, not hard-coded 2); (e) DeepFilterNet skipped and `mastering_engine=None` for elevenlabs, with `deess_elevenlabs` in the mastering slot; (f) voice chain = `build_elevenlabs_voice_chain`; (g) expose `self.last_tts_manifest`; (h) save mix session (§5). |
| `core/tts_cache.py` | `_TTS_ENV_PREFIXES += ("MOODSCAPE_ELEVENLABS_",)`; `compute_cache_key(..., extra: dict | None = None)` folds engine-specific params (voice id, stability, similarity, direction, model) into the key; `save_cached_tts` already stores `sample_rate` — callers pass the true rate. |
| `core/content_profiles.py` | `elevenlabs_paragraph_pause_sec`: meditation 3.0, sleep_story 2.0 (= F5). |
| `core/script_gen/duration.py` | `DEFAULT_WPM["elevenlabs"] = 85.0` (seed = F5; calibrated from `log_estimate_accuracy` after first renders). `estimate_duration_sec` uses the elevenlabs preprocessor for that engine. |
| `core/script_gen/guide_context.py` + `core/script_gen/rules.py` (`GUIDE_TEMPLATE`) | Map `elevenlabs` → the **f5 guides** and F5's `CHUNK_CHARS` context (writer sees an engine-neutral format). Unknown-placeholder behaviour unchanged. |
| `core/auto_generate.py` | `"elevenlabs"` handled: voice from `elevenlabs_tts.voices.voice_for_family(pack.family)` (default until the map is filled) → `generate_kwargs["elevenlabs_voice_id"]`; `preflight()` checks key + voice validity + direction when `tts_engine == "elevenlabs"`; meta gains `"elevenlabs": manifest`. |
| `core/auto_tab.py` | Accepts the provider; hides its engine dropdown when ElevenLabs; shows chars billed in status. |
| `app.py` | §4. |
| `scripts/eval_genres.py`, `core/eval/*` | Accept `elevenlabs` as an engine in `--configs`; print estimated total chars before rendering. |
| `requirements.txt` | `elevenlabs` (pinned to the version installed during implementation). |
| `.env` (user) | `ELEVENLABS_API_KEY` — user adds; not committed. |

## 4. UI (`app.py`, `core/auto_tab.py`)

- **Provider toggle** `gr.Radio(["ElevenLabs", "Local Models"], value="Local Models", label="Voice Provider")`
  placed directly under the header HTML, above `gr.Tabs()`. Last choice
  persisted in browser localStorage (try/catch; falls back to default).
- **Manual tab, Local Models:** unchanged.
- **Manual tab, ElevenLabs:** hide `tts_engine_radio`, `kokoro_settings`,
  `f5_settings` (voice, WPM, Voice Expressiveness), `speed_slider`. Show
  `elevenlabs_settings` group:
  - Voice dropdown from `voices.toml` (label suffixed "(not in account)" when
    validation fails; validation runs lazily on first toggle to ElevenLabs).
  - Stability slider 0–1 (default 0.5), Similarity slider 0–1 (default 0.75).
  - Delivery direction textbox (default `[soft, slow, calm voice]`), validated
    on change via `sanitize_direction`.
  - Cost line (Markdown): "≈ 7,840 characters · 3/9 chunks cached", recomputed
    on script / voice / settings / direction change (pure function, no API).
  - Missing API key → warning banner in the group and Generate disabled while
    ElevenLabs is selected.
- Reverb defaults to 0.0 when ElevenLabs is selected (like Chatterbox today);
  content-profile presets otherwise apply as now (speed preset ignored).
- `generate_meditation` maps provider ElevenLabs → `tts_engine="elevenlabs"`
  and passes the four ElevenLabs params; status line includes chars billed /
  cached.
- **Remix panel** (both providers) under the output player, hidden until a run
  succeeds: "Voice level (dB)" and "Music level (dB)" sliders −12…+12 step 0.5
  default 0, info text "Changes the voice-to-music balance; final loudness is
  re-normalized", and a **Remix** button. Output replaces the player audio and
  shows the new file path. A `gr.State` holds the last `run_id`.
- **Auto-Generate tab:** receives the provider; ElevenLabs → engine dropdown
  hidden, `tts_engine="elevenlabs"`.

## 5. Remix (both providers)

### Mix session save (pipeline)

Immediately before `mixer.mix()` in `MeditationPipeline.generate()`, write
`var/mix_sessions/<run_id>/` (`run_id` = timestamp + short hash of output
name):

- `voice.npz` — processed voice (post voice-FX, post −18 LUFS normalize) +
  activity mask, float32/bool, at `mix_sr`.
- `music.npz` — processed music (post music FX + vocal pocket, post −16 LUFS).
- `mix_params.json` — `mix_sr`, effective `music_volume_db`, `duck_amount_db`,
  pre/post roll, fades, `target_lufs`, `mono_offset`, `stereo_output`,
  `bed_overrides`, `phrases`, spectral-duck/shared-reverb flag states, output
  format, output basename, content type, original output path.

Skipped for "Instrumental Only" / "Vocals Only" modes (Remix hidden). Prune to
the newest `MOODSCAPE_MIX_SESSIONS_KEEP` (default 20) after each save.
`pipeline.last_run_id` exposes the id.

### `core/remix.py`

`remix(run_id, voice_db=0.0, music_db=0.0, progress_cb=None) -> str`:
load session → `voice *= 10**(voice_db/20)` → `mix(..., music_volume_db=params.music_volume_db + music_db, phrases=params.phrases, …)`
→ `make_master_chain()` → `export_audio()` with the stored target/offset →
writes `<basename>_remix-v{+2}-m{-3}.<ext>` beside the original (never
overwrites). Voice trim scales only the voice; the duck timing comes from the
stored mask/phrases, so the bed relationship holds. Also writes a sibling
`.meta.json` noting source run and trims. QA (`run_qa_checks`) runs on the
result as in the pipeline.

Invariant tested: `remix(id, 0, 0)` output equals the original export
(sample-exact for WAV).

## 6. Repo-wide: no vocal sounds

- `core/f5_tts/engine.py` (~L746) and `core/kokoro_tts/engine.py` (~L321):
  `"breath"` segments emit **zeros** of `BREATH_SEC[subtype]` (mask False)
  instead of `load_breath()` samples.
- `core/chatterbox_tts/engine.py`: add explicit `"breath"` handling (currently
  falls into the speech branch with empty text and is dropped) → same silence.
- ElevenLabs: breath → pause in the preprocessor; direction sanitizer forbids
  sound words; no reaction tags are ever generated.
- Delete `core/breath_sounds.py`, `assets/breath_sounds/`,
  `scripts/generate_breath_samples.py`, and their tests/doc references.
  `BREATH_SEC` in `duration.py` stays (now "silent beat lengths"); update its
  comment.
- Linter, writer guides, and duration estimates are unchanged — `[breath]`
  etc. remain valid script markers and now mean a silent beat.
- `CLAUDE.md` Top Gotchas: add "No vocal sounds (breaths, sighs, reaction tags)
  in any output — breath markers render as silence."

## 7. Error handling

| Condition | Behaviour |
|---|---|
| No `ELEVENLABS_API_KEY` | UI banner + Generate disabled; CLI/auto-gen fail in `preflight()` before any LLM call. |
| Voice not in account | Preflight 404 → error naming voice + "add it from the Voice Library"; UI marks "(not in account)". |
| Invalid direction (env or UI) | `ValueError` with the offending word; UI inline, preflight for env. |
| 429 / 5xx / network | Retry with backoff (`MAX_RETRIES`, default 3). |
| 401 / 402 / 422 / other 4xx | Fail immediately; ElevenLabs message surfaced (402 → "out of credits"). |
| Mid-render failure | Rendered chunks remain cached; message reports N/M; re-run bills only the rest. |
| Silent/NaN/noise output | Existing vocal sanity check (pipeline) flags it. |
| Remix with missing/pruned session | Clear error "Mix session not found — generate again". |

Only the chunk text, voice id, and settings are sent to ElevenLabs.

## 8. Configuration (env, all in the TTS cache key via prefix)

| Var | Default | Purpose |
|---|---|---|
| `ELEVENLABS_API_KEY` | — | Auth (not a cache-key var) |
| `MOODSCAPE_ELEVENLABS_MODEL` | `eleven_v4` | Experiments only |
| `MOODSCAPE_ELEVENLABS_STABILITY` / `_SIMILARITY` | 0.5 / 0.75 | Defaults behind UI sliders |
| `MOODSCAPE_ELEVENLABS_DIRECTION` | `[soft, slow, calm voice]` | Default delivery direction |
| `MOODSCAPE_ELEVENLABS_SEED` | 42 | Best-effort determinism |
| `MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS` | 1500 | Target chunk size |
| `MOODSCAPE_ELEVENLABS_XFADE_MS` | 30 | Join crossfade |
| `MOODSCAPE_ELEVENLABS_CHUNK_LUFS` | −21 (`off`) | Per-chunk loudness match |
| `MOODSCAPE_ELEVENLABS_STITCH` | 1 | Send `previous_request_ids` |
| `MOODSCAPE_ELEVENLABS_MAX_RETRIES` | 3 | Transient retry budget |
| `MOODSCAPE_ELEVENLABS_DEESS` | 1 | Gentle split-band de-ess (post-TTS; in cache key by prefix though applied after cache) |
| `MOODSCAPE_ELEVENLABS_CACHE_DIR` | `var/elevenlabs_cache` | Chunk cache |
| `MOODSCAPE_MIX_SESSIONS_KEEP` | 20 | Remix session retention (not TTS) |

## 9. Testing

**Unit (`tests/unit/`, fake client, no network):**
`test_elevenlabs_preprocessor.py` (parity with F5 parsing; chunk ≤ max; no split
inside brackets; direction on every chunk; breath → silence of BREATH_SEC; no
sound/reaction tag in any API text), `test_elevenlabs_direction.py`,
`test_elevenlabs_engine.py` (48 kHz float32; `len(mask)==len(audio)`; pauses
False; crossfade length accounting; chunk-LUFS passthrough cases; ≤3 stitch ids
incl. from cache hits; mid-run failure message + cached prefix),
`test_elevenlabs_chunk_cache.py`, `test_elevenlabs_client.py` (retry
matrix), `test_elevenlabs_voices.py` (toml load, default, family_map
validation), `test_pipeline_elevenlabs.py` (no upsample/DF/mastering; cache key
varies with each ElevenLabs param and `MOODSCAPE_ELEVENLABS_*` env),
`test_no_breath_sounds.py` (F5/Kokoro/Chatterbox/ElevenLabs render breath as
zeros; `core.breath_sounds` not importable), `test_remix.py` (0/0 identity;
+6 dB voice raises voice-to-music ratio ≈6 dB; original never overwritten;
missing session error), `test_auto_generate_elevenlabs.py` (guides = F5,
voice from registry, preflight fails without key), plus updates to existing
tests that referenced breath samples.

**Integration (`tests/integration/`, `@pytest.mark.elevenlabs`, skipped
without key):** one-paragraph real render through the full pipeline (48 kHz,
sane LUFS, manifest chars).

**Regression:** F5 + upload golden path LUFS baseline unchanged (per
`mix-calibration-baseline` memory), allowing for breath removal.

**Manual:** Gradio app in the browser pane — toggle shows/hides correct
controls; short ElevenLabs render with Serena and Brittney; Remix ±4 dB;
Auto-Generate end-to-end with ElevenLabs. Listening judgement is the user's.

## 10. Docs

`CLAUDE.md` (engines, pipeline step 3 note, Top Gotchas: v4 has no speed /
pacing via text + silence; no vocal sounds; flag table rows), `docs/ARCHITECTURE.md`,
`docs/GOTCHAS.md`, `docs/COMPONENT_REGISTRY.md`, `docs/TASK_ROUTING.md`, new
`docs/model_implementation_guides/elevenlabs_v4.md` (research summary §2 +
tuning knobs).

## 11. Out of scope (follow-ups)

- v4 voice review and filling `[family_map]` (user supplies IDs).
- Optional flag-gated LLM tagging pass (adds voice-quality directions per line).
- Remix on the Auto-Generate tab.
- IPA pronunciation dictionary for tricky words.
- `eleven_v4_turbo` drafts → `eleven_v4` final workflow.
