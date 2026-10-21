# MoodScape Guided Meditations

AI-guided meditation audio generator (Gradio UI). Local TTS engines (Kokoro, F5-TTS, Chatterbox) plus a cloud provider (ElevenLabs v4) selected by a top-of-app Voice Provider toggle, and two music sources (Lyria RealTime or pre-existing background instrumentals). Target hardware: Apple Silicon M1 Max (32 GB unified RAM).

**Content types.** The app generates **guided meditations** (default) and **sleep stories**, selected via the "Content Type" UI dropdown (`--content-type` on the CLI). A single `content_type` ("meditation" | "sleep_story") threads UI → pipeline → preprocessors → mixer; `"meditation"` is the default at every layer, so the meditation path is unchanged. Profiles live in [core/content_profiles.py](core/content_profiles.py): slider-backed values (speed, duck, fades, reverb) are *UI preset defaults the user can still override*; non-slider behaviours are driven by `content_type` — shorter paragraph-break pauses (`{engine}_paragraph_pause_sec`, passed into `prepare_segments`) and a softer, more-constant sleep bed (gentler `apply_breathing_duck` kwargs + narrower `calibrate_music_bed` offsets via `mix(bed_overrides=…)`). Prompting guides are per-engine **and** per-content-type: `vocal_meditation_{kokoro,f5}_instructions.md` and `vocal_sleep_story_{kokoro,f5}_instructions.md`.

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
brew install espeak-ng                 # Kokoro G2P dependency
python app.py                          # Gradio UI at http://localhost:7860
```

**Model weights** live in the default user-level HF cache `~/.cache/huggingface/hub` (no `HF_HOME` override; there is no project-local `models/` or `checkpoints/` dir). Models actually used: `hexgrad/Kokoro-82M`, `SWivid/F5-TTS`, `charactr/vocos-mel-24khz`, `ResembleAI/chatterbox`, `mlx-community/DeepFilterNet-mlx`, `openai/whisper-large-v3-turbo` (F5 ref-transcription fallback), plus HTDemucs weights in `~/.cache/torch/hub/checkpoints`. Anything else in that cache is a leftover from a removed engine and safe to delete.

`.env` must define `HF_TOKEN` and `GOOGLE_API_KEY`, and (for the ElevenLabs provider) `ELEVENLABS_API_KEY` (`ELEVEN_LABS_API_KEY` is accepted as a fallback). `app.py` sets `TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=1` at startup.

## Build & Test

```bash
.venv/bin/python -m pytest tests/unit/ -v                                    # all unit tests
.venv/bin/python -m pytest tests/unit/test_mixer.py -v                       # single file
.venv/bin/python -m pytest tests/integration/ -v                             # full pipeline (slow)
python scripts/generate.py <script_file> --voice <voice_name> --output <out.wav>
.venv/bin/python scripts/eval_genres.py --testset eval/testset.toml --metrics --export-stems --configs "baseline=ollama:qwen3.8:27b|ollama:gemma4:31b" --out var/eval/base
python scripts/ab_chain.py <spec.toml> --write && python scripts/ab_review.py <spec.toml>   # eval chain + REVIEW.md from a TOML spec (never runs the chain)
python scripts/ab_export.py --a <base>/<config> --b <new>/<config> --out var/eval/ab-001 && python scripts/ab_score.py --ab var/eval/ab-001
```

## Folder Map

```
.
├── app.py                            # Gradio UI entry point
├── core/
│   ├── pipeline.py                   # MeditationPipeline orchestrator
│   ├── speech_engine.py              # SpeechEngine ABC (TTS contract)
│   ├── audio_processor.py            # Pedalboard FX chains
│   ├── mixer.py                      # ducking · overlay · loudness · export
│   ├── qa_monitor.py                 # output validation
│   ├── stem_separator.py             # Demucs source separation
│   ├── speech_qa.py                  # engine-agnostic per-chunk speech QA (gaps, truncation, rate, CER)
│   ├── text_utils.py · stereo_upmix.py · deepfilter_enhancer.py
│   ├── genres.py                     # Genre pack loader, angle rotation, GenrePack dataclass
│   ├── originality.py                # Corpus management, TF-IDF similarity scoring, avoid-lists
│   ├── eval/                         # eval harness: script/speech/mix metrics, test set, blind A/B; render_only.py (same-script re-render), listen.py (listening page + answer import), chain.py (TOML eval spec → zsh chain), review.py (REVIEW.md + measurement card)
│   ├── background_tags.py            # Lazy background music tagging (measured + declared)
│   ├── elevenlabs_tts/                # ElevenLabs v4 cloud engine (client, chunking, direction, stitching, chunk cache, light postproc)
│   ├── mix_session.py · remix.py · provider_ui.py   # saved mix inputs · Remix (mix→master→export only) · testable Voice Provider/Remix UI logic
│   ├── kokoro_tts/  f5_tts/             # TTS engines (engine + preproc + postproc + voices; f5_tts/chunking.py = breath-group chunker)
│   ├── lyria/                             # Lyria RealTime music generation (Google API)
│   ├── upload_music/                      # Background instrumental (engine + arrange/length-fit)
│   ├── script_gen/                    # prompt → validated script (planner + generator + judge + linter)
│   │   └── planner.py                 # Pass 0: genre pack + angle → prose creative brief
│   │   # also: sampling.py (per-stage sampling) · slop.py (slop lexicon) · guide_context.py (guide placeholders) · content_check.py (LLM sensitive-wording check)
│   ├── auto_generate.py               # orchestrator: genre → planner → script → music → pipeline
│   ├── voice_selection.py             # per-genre-family narrator voice from measured clip traits
│   ├── background_picker.py           # tag-filtered pick from assets/backgrounds/
│   ├── streaming_run.py               # threaded progress streaming for the UI
│   └── auto_tab.py                    # Gradio "Auto-Generate" tab (peer of "Manual" under app.py's gr.Tabs())
├── eval/testset.toml               # fixed 24-job evaluation set
├── scripts/                          # generate.py · separate_worker.py · tag_backgrounds.py · eval_genres.py · ab_export.py · ab_listen_page.py · ab_import_answers.py · ab_score.py · ab_chain.py (spec → chain.sh) · ab_review.py (spec → REVIEW.md)
├── tests/unit/  tests/integration/   # tests/__init__.py exists so site-packages `tests` packages don't shadow the repo's
│   └── conftest.py                   # pytest fixtures for genre tests
├── assets/                           # tracked in git
│   ├── impulse_responses/            # convolution reverb IRs
│   ├── backgrounds/                  # instrumental background tracks + tags.toml
│   ├── elevenlabs/voices.toml        # ElevenLabs voice list (id, name, notes)
│   └── speakers/                     # F5-TTS voice pool
│       ├── reference_audio/*.wav     #   speaker reference clips
│       ├── reference_text/*.txt      #   transcripts (paired by slug)
│       └── voices.toml               #   F5 multi-phase definitions
├── var/                              # gitignored; machine-local state
│   └── originality/                  # corpus index + scripts for originality checks
├── docs/
│   ├── genre_packs/                  # 46 TOML pack definitions + README (field contract, tag vocabulary, prose guidelines)
│   └── ...                           # other docs (see Where to Look below)
```

## Pipeline Flow (`core/pipeline.py :: MeditationPipeline.generate()`)

1. **Parse script** → `{tts}/preprocessor.py :: prepare_segments()`
2. **TTS synth** → 24 kHz mono float32 (local) or 48 kHz (ElevenLabs: no upsample, no DeepFilterNet)
3. **Unload TTS right after synthesis** (before upsample/denoise), then load music engine (sequential — 32 GB RAM constraint)
4. **Music gen** → 48 kHz mono float32 (Lyria or `upload_music` — decode + resample + loop/trim-fit the uploaded file to the same contract)
5. **Stem separation** (optional; skipped for uploads) → `stem_separator.remove_drums_and_vocals()`
6. **TTS upsample** (local only; ElevenLabs is already 48 kHz) 24 → 48 kHz via `audio_processor.upsample_audio(high_accuracy=True)`; then per-chunk humanize (Kokoro)
7. **Voice FX** → `build_voice_chain()` + `apply_fx()`; then `mixer.normalize_loudness()` to −18 LUFS
8. **Music FX** → `make_{engine}_music_chain()` + `make_vocal_pocket_chain()`
9. **Mix** → `mixer.mix()` (breathing sidechain duck — deep gradual S-curve, rises in pauses; exponential fades). Saves a mix session (`var/mix_sessions/`; voice + music renders only, not Vocals Only / Instrumental Only) so Remix can re-run mix → export without TTS. Bed + duck levels auto-calibrated per session from measured stem LUFS (`mixer.calibrate_music_bed`; disable with `MOODSCAPE_ADAPTIVE_BED=0`)
10. **Master** → `make_master_chain()` (HPF → gentle bus comp → +1 dB air shelf; **no** limiter)
11. **QA** → `qa_monitor.run_qa_checks()`
12. **Export** → `mixer.export_audio()` (LUFS-normalize to playback −16 (sleep −18); mono files sit 3 dB lower so dual-mono playback hits the target → `true_peak_limit()` to −1 dBTP → WAV/MP3)

Full breakdown with parameters: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Auto-Generation Flow (`core/auto_generate.py :: run()`)

Genre and duration band in, finished meditation out, with no human step.

0. **Preflight** → `preflight()` verifies all LLM models exist (Ollama
   `/api/tags`) and names any missing model before planning starts (fail in 2s,
   not 5 min)
1. **Load pack + rotate angle** → `genres.load_pack()` validates the genre
   pack (required fields, known tags, valid `content_type`), then
   `genres.pick_angle()` selects one of 3 distinct angles, excluding recently
   used ones for this genre; `originality.avoid_terms()` extracts a do-not-reuse
   list from the 5 most recent same-genre scripts (avoid-list), 100 same-genre scripts
   (cosine comparison), and 500 all-genre scripts (IDF weighting)
2. **Plan** (Pass 0) → `script_gen/planner.py :: plan()` takes the pack's
   technique, arc, imagery angle, pause_ratio, and safety rules, plus the
   avoid-list and duration band, and returns a **prose creative brief**
   (plain text, no JSON) that shapes what the writer will generate
3. **Write** (Pass 1) → `script_gen/generator.py :: draft()` ingests the brief,
   the formatting guide, and safety rules, and returns a full script using
   generator model (`MOODSCAPE_SCRIPT_GENERATOR` — shared load with planner)
4. **Review** (Pass 2) → an *independent* judge model (`MOODSCAPE_SCRIPT_JUDGE`)
   (`script_gen/judge.py :: review()`) returns a revised script plus a
   changelog — it revises, it does not score
5. **Validate** → `script_gen/linter.py :: check()` (format + mental-health
   safety incl. global `SENSITIVE_TOPIC` + originality + banned phrases) and
   `script_gen/duration.py` (runtime estimate, no rendering)
6. **Repair loop** → fatal violations go back to the judge as targeted
   instructions, bounded by `MOODSCAPE_SCRIPT_MAX_REPAIRS` (default 2); advisory
   violations render anyway and seed the next run's avoid-list. Then, sharing
   that budget: **content check** (`content_check.py`, judge flags sensitive
   wording → one `SENSITIVE_CONTENT` repair, ADVISORY after) and **duration
   control** (one length repair when >15% outside the window, then pause
   scaling ×0.85–1.15; drift never fails a job)
7. **Pick music** → `background_picker.pick_background(prefer_tags=pack.music_tags)`
   filters by the pack's two music tags, excluding recently used tracks
8. **Render** → `MeditationPipeline.generate()`, unchanged, on the golden path
   (F5 + uploaded background)
9. **Calibrate** → measures the rendered file's actual duration and calls
   `script_gen/duration.py :: log_estimate_accuracy()`, closing the loop that
   calibrates `DEFAULT_WPM`; a failed duration read never fails a job that
   already produced audio
10. **Append corpus** → `originality.add_to_corpus(script, genre=pack.slug, angle=selected_angle)`
    records the rendered script for use in future avoid-lists
11. **Persist** → `<name>.wav`, `<name>.script.txt`, `<name>.meta.json`
    (including `actual_sec` and `estimate_ratio`) as siblings

The UI exposes this as an "Auto-Generate" tab (`core/auto_tab.py`), a peer
of the "Manual" tab under one `gr.Tabs()` in `app.py` (Manual first, so it
stays the default). Its generator/judge model calls retry transient
failures (connection errors, timeouts, 429, 5xx) via
`MOODSCAPE_SCRIPT_MAX_RETRIES` — see the adapter-asymmetry gotcha in
[docs/GOTCHAS.md](docs/GOTCHAS.md).

Full detail: [docs/auto_generation/README.md](docs/auto_generation/README.md).

## Code Conventions

- `PascalCase` classes · `snake_case` functions · `UPPER_SNAKE_CASE` constants · `_leading_underscore` private
- Imports: `from core.module import Class` (relative to project root)
- Every subpackage has `__init__.py` with explicit public exports
- TTS engines inherit `SpeechEngine` ABC from `core/speech_engine.py`

## Git Workflow

- Conventional commits: `feat:` `fix:` `refactor:` `docs:` `test:` `chore:`
- Commit directly to `main` (solo developer)
- **Never push** unless the user explicitly asks

## Top Gotchas

The ones that bite most often. Full list in [docs/GOTCHAS.md](docs/GOTCHAS.md).

- **MPS bus error on exit** → `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove.
- **Kokoro forced to CPU** → MPS causes deallocation bus errors. British voices (`bf_*`, `bm_*`) need `KPipeline(lang_code="b")`.
- **No pedalboard `Limiter`** → pedalboard 0.9.23's `Limiter` inflates sub-threshold signals ~+4.75 dB and adds broadband "static". It was removed from all music + master chains. Peak control is `mixer.true_peak_limit()` at export (LUFS-normalize → true-peak limit to −1 dBTP).
- **Breathing duck** → `mixer.mix()` uses `apply_breathing_duck` (deep gradual S-curve, rises in pauses). Bed/duck levels are auto-calibrated per session (`calibrate_music_bed`, targets: bed 14.5 LU under voice in pauses, 30.5 LU under during speech); `MOODSCAPE_ADAPTIVE_BED=0` restores the fixed −16/−16 constants. The old multiband/`hold_ms` reactive ducker has been removed.
- **Intro fade-in** → default **1.5 s**, gentle exponential (rising steepness **2.0**, not 4.0). A 3 s steepness-4 fade read as "2–3 s of silence then music". `apply_fades` uses steepness 2.0 for the rising curve only (fade-out stays 4.0). Slider in `app.py`, default in `pipeline.generate`.
- **Prompting guides are templates.** `{{NAME}}` placeholders in `docs/prompting_guides/vocal_*` are rendered from code constants by `script_gen/guide_context.py`. Never hard-code a chunk limit, speed, WPM or pause length in a guide. An unknown placeholder fails the run before any LLM call.
- **TTS cache keys include every `MOODSCAPE_F5_*` / `MOODSCAPE_CHATTERBOX_*` env var** → a new TTS-affecting env var must use that prefix or the cache serves a stale voice. `MOODSCAPE_ELEVENLABS_*` vars count only for ElevenLabs keys (`tts_env_fingerprint(engine="elevenlabs")`) and never touch F5/Kokoro/Chatterbox keys. The prefix is necessary but not sufficient: `_TTS_ENV_EXCLUDE` in `core/tts_cache.py` leaves out `MOODSCAPE_ELEVENLABS_CACHE_DIR` and `_MAX_RETRIES`.
- **No vocal sounds** → breath markers (`[breath]`/`[inhale]`/`[exhale]`) render as silence in every engine (the breath-sample feature is gone). ElevenLabs strips every user-typed `[...]` tag from speech, and the direction sanitizer (`core/elevenlabs_tts/direction.py`) rejects sound words (breath/sigh/laugh/yawn/hum etc., incl. inflections and "mm-hmm"). Don't reintroduce sound samples or sound-word directions.
- **Eleven v4 has no speed/style** → pacing = bracketed direction + punctuation + programmatic silence; the UI hides Speed for ElevenLabs and speed is not in its cache key. A whole-render cache hit reports `chars_billed` 0. Stitching sends up to 3 `previous_request_ids`. Details: [docs/model_implementation_guides/elevenlabs_v4.md](docs/model_implementation_guides/elevenlabs_v4.md).
- **`var/elevenlabs_cache` grows without bound** → the per-chunk PCM cache (48 kHz int16) costs ~5.5 MB per minute of speech for every voice/settings/direction combination rendered, and nothing prunes it. It is what makes re-runs free; to reclaim space, delete the directory (`rm -rf var/elevenlabs_cache`) — the next render re-bills those chunks.
- **Mix sessions are big** → each is ~350 MB for 15 min; only the newest `MOODSCAPE_MIX_SESSIONS_KEEP` (5) are kept. Remix and Generate share one Gradio concurrency group.
- **Fatal vs advisory violations** → `script_gen/linter.py` fails the job for
  safety hard-blocks and malformed markers, but renders anyway (with a warning)
  for duration drift and style issues. Treating every violation as fatal makes
  a weaker local model unusable; treating none as fatal lets a safety failure
  reach audio. Do not flatten this distinction.
- **Content rules must never REQUIRE sensitive wording.** The old R1 disclaimer forced scripts to name medical conditions/pregnancy, which a user rule forbids. Now breathwork gets only a gentle "breathe naturally" line, and `SENSITIVE_TOPIC` (FATAL) is **global**: no medical conditions, pregnancy, illness/treatment, hospitals, or children/babies/younger self ("child's pose" is fine). Any new rule that mandates wording must be checked against it.
- **Ollama ignores `keep_alive` on `/v1`** → The `/v1/chat/completions` endpoint
  silently disregards the `keep_alive` parameter; it is honoured only on
  `/api/*` endpoints. `ScriptEngine.unload()` exists specifically to call the
  native `/api/generate` with `keep_alive=0` for Ollama after each stage,
  freeing 18 GB. Without this, an 18 GB planner/writer model stays resident
  while F5-TTS (14 GB) loads, forcing 32 GB → swap, which is where the MPS bus
  errors live. **Call `unload()` after each LLM stage.**
- **Originality repair depends on `<problems>` stripping** → The judge repair
  loop passes `check_originality()` violations into the prompt wrapped in
  `<problems>…</problems>` so the model can see which phrases failed. But a weak
  model frequently echoes that block verbatim, poisoning the script with the
  violation report itself. `parse_judge_response()` strips `<problems>` *before*
  anything else (commit `c372b18`). Removing that strip makes repair recursive
  — violations compound, budget exhausts, and a fixable script fails. **This
  behaviour is load-bearing and must not be removed.**

## Research Experiment Flags

From the 2026-06-13 research pass ([Research Execution Process Outline.md]). Bug fixes ship **ON** with kill-switch flags (F5 ref pad, short-phrase pacing, `LOUDNESS_DUAL_MONO`); experiments default **OFF** for A/B listening tests (some with Gradio toggles, the newer ones env-only).

| Flag (default) | Effect | Code |
|---|---|---|
| `MOODSCAPE_F5_REF_PAD` (**1**) | Appends ~1 s of −55 dBFS noise to reference audio so F5 leaks silence (not a stray syllable) on short phrases. `_SEC`/`_DBFS` tune it. | `core/f5_tts/engine.py::_condition_reference_audio` |
| `MOODSCAPE_F5_SHORT_PHRASE_PACING` (**1**) | Slows genuine fragments (≤ `_MAX_CHARS` non-space chars, default **12**) to `_SPEED` (0.5) in natural-rhythm mode only. **Char-based, not word-based** — a word count (≤6) caught normal short sentences and forced them so slow that F5 stretched + inserted mid-word gaps. | `core/f5_tts/engine.py::synthesize` |
| `MOODSCAPE_SPECTRAL_DUCK` (**0**) | Ducks only the 250–4000 Hz mid band (LR crossover, subtractive mid) instead of fullband — keeps bass warmth + air. `_DEPTH`/`_LO`/`_HI` tune it. | `mixer.apply_breathing_duck_multiband`, branch in `mix()` |
| `MOODSCAPE_SHARED_REVERB` (**0**) | Sends the bed through the voice's IR (HPF300/LPF6k) at `_SEND_DB` (−26) so both share one room. | `mixer.add_shared_reverb`, summed in `mix()` |
| `MOODSCAPE_F5_CFG`/`_SWAY`/`_NFE` (2.0/−1.0/32) | Override F5 latent params (research A/B: cfg ~1.2 warmer/less identical; sway/nfe best left alone). `cfg` also has the **"Voice Expressiveness" UI slider** (1.0–2.5). | `core/f5_tts/engine.py::synthesize` |
| `MOODSCAPE_F5_MICROPROSODY` (**0**) | Per-phrase WORLD pass, peak-preserving: pitch declination (`_DECLINE_TAIL_MS` 600 / `_DECLINE_CENTS` 120) + pitch-range widening (`_PITCH_SCALE` 1.15) + formant warmth (`_FORMANT_SHIFT` 0.98) + aperiodicity (`_AP_SCALE` 1.05) + phrase-final taper (`_TAPER_MS` 300 / `_TAPER_FLOOR` 0.6). Risky on clean Vocos. | `core/f5_tts/postprocessor.py::apply_microprosody` |
| `MOODSCAPE_F5_REF_PRESERVE_DYNAMICS` (**0**) | Skip the −20 dBFS RMS-normalisation of reference audio so its natural dynamics (expressive contour) survive. | `core/f5_tts/engine.py::_condition_reference_audio` |
| `MOODSCAPE_LOUDNESS_DUAL_MONO` (**1**) | Export targets are *playback* loudness: −16 LUFS meditation / −18 sleep (profile `export_lufs`); mono files are normalised 3.01 dB lower so dual-mono playback hits the target. Stereo sleep stories therefore export at −18 LUFS (the profile target applies regardless of channel count; only mono gets the −3.01 dB offset) — intended. `0` = legacy −16 on the mono reading. | `mixer.export_lufs_target`, `export_audio(mono_playback_offset_db)` |
| `MOODSCAPE_BED_RISE_LIMIT` (**0**) | Caps bed rise in pauses/intro/outro at `_DB_PER_SEC` (3.0) dB/s and `_MAX_DB` (12.0) dB total. When on it forces `lift_db` to 0 for every content type (incl. sleep stories). | `mixer.mix` |
| `MOODSCAPE_BED_SPEECH_OFFSET_LU` / `_PAUSE_OFFSET_LU` (unset) | Override the profile's `calibrate_music_bed` offsets (env wins). Setting either widens the duck clamp to (−20, −4) for all content types. Malformed value raises at the mix step. | `mixer.bed_offset_overrides()`, merged into the `calibrate_music_bed` kwargs in `core/pipeline.py`; applies only when `MOODSCAPE_ADAPTIVE_BED` is on (ignored with `=0`) |
| `MOODSCAPE_F5_CHUNKING` (**0**) | Renders breath groups sized to F5's own budget (upstream never re-splits) with punctuation-length gaps: `_GAP_SENTENCE` 0.45 / `_GAP_CLAUSE` 0.35 / `_GAP_COMMA` 0.20 s. Frees MPS cache per group. | `core/f5_tts/chunking.py`, `engine.py` |
| `MOODSCAPE_F5_CHUNK_QA` (**0**) | Needs `CHUNKING`. Per-chunk speech QA (gaps, truncation, rate outliers, CER); failing chunks re-rendered best-of-3; result in `meta.json` `"speech_qa"`. Never fails a job. `MOODSCAPE_F5_QA_DEVICE` (default `mps`; `cpu` = escape hatch) picks the Whisper device. | `core/speech_qa.py`, `engine.py` |
| `MOODSCAPE_F5_TRIM_LEADIN` (**0**) | Trims the ~1.5–2.2 s ref-pad lead-in at the start of each F5 chunk (onset = first frame within `_DB` −35 dB of the chunk's peak frame, i.e. relative not absolute; keeps `_LEAD_MS` 60 ms). Env-only; TTS-affecting so it is in the cache key. | `core/f5_tts/engine.py::synthesize` |
| `MOODSCAPE_DURATION_CONTROL` (**1**) | Script-stage length control: one judge length repair when the estimate is >15% outside the window (shared repair budget), then pause scaling ×0.85–1.15; falls back to the last fatal-free script; never fails a job. Result in meta `duration_control`. `0` = off. | `core/auto_generate.py::generate_script` |
| `MOODSCAPE_CONTENT_CHECK` (**1**) | Judge-model check for sensitive wording (illness/medical, pregnancy, medication/surgery, medical settings, children/younger self, non-grief death, self-harm, weight); the pack's own subject is allowed; one `SENSITIVE_CONTENT` repair, ADVISORY after; result in meta `content_check`. `0` = off. | `core/script_gen/content_check.py`, `core/auto_generate.py` |
| `MOODSCAPE_VOICE_BY_FAMILY` (**0**) | `1` picks the narrator per genre family from measured clip traits; `0` keeps Brittney. `voice_map.toml` overrides apply either way. | `core/voice_selection.py` |
| `MOODSCAPE_DF_STREAMING` (**0**) | `1` streams DeepFilterNet in 4 s chunks (bounded MLX graph, numerically equivalent, ~21× slower); default offline. MLX cache freed after every call. | `core/deepfilter_enhancer.py` |
| `MOODSCAPE_ELEVENLABS_MODEL` (`eleven_v4`) | ElevenLabs model override (experiments only). All `MOODSCAPE_ELEVENLABS_*` vars except `_CACHE_DIR` and `_MAX_RETRIES` are fingerprinted, and only into ElevenLabs keys. | `core/elevenlabs_tts/client.py` |
| `MOODSCAPE_ELEVENLABS_STABILITY` / `_SIMILARITY` (0.5/0.75) | Defaults behind the ElevenLabs UI sliders. | `core/elevenlabs_tts/engine.py` |
| `MOODSCAPE_ELEVENLABS_DIRECTION` (`[soft, slow, calm voice]`) | Default delivery direction prefixed to every chunk; sound words rejected. | `core/elevenlabs_tts/direction.py` |
| `MOODSCAPE_ELEVENLABS_SEED` (42) | Best-effort determinism. | `core/elevenlabs_tts/engine.py` |
| `MOODSCAPE_ELEVENLABS_CHUNK_MAX_CHARS` (1500) / `_XFADE_MS` (30) / `_CHUNK_LUFS` (−21, `off`) | Chunk size, join crossfade, per-chunk loudness match. | `core/elevenlabs_tts/engine.py` (`_XFADE_MS`, `_STITCH`), `preprocessor.py`, `assembly.py` |
| `MOODSCAPE_ELEVENLABS_STITCH` (**1**) / `_MAX_RETRIES` (3) | Send up to 3 `previous_request_ids`; transient retry budget. | `core/elevenlabs_tts/engine.py` (`_STITCH`); `client.py` caps at 3 ids and reads `_MAX_RETRIES` |
| `MOODSCAPE_ELEVENLABS_DEESS` (**1**) | Gentle split-band de-ess on the ElevenLabs voice (applied after the TTS cache, but its prefix still puts it in the ElevenLabs cache key; harmless). | `core/elevenlabs_tts/postprocessor.py` |
| `MOODSCAPE_ELEVENLABS_CACHE_DIR` (`var/elevenlabs_cache`) | Per-chunk PCM cache location. | `core/elevenlabs_tts/chunk_cache.py` |
| `MOODSCAPE_MIX_SESSIONS_KEEP` (**5**) | Newest mix sessions kept for Remix (~350 MB each for 15 min); not a TTS var. | `core/mix_session.py::prune_sessions` |

Rejected research items (don't revisit): true-peak-limiter removal (false premise — `true_peak_limit` is transparent below ceiling), SRC reorder (already correct — soxr_vhq before FX; the 48 kHz path has no extra audio resampling beyond the limiter's oversampling), full LR crossover for "harshness" (the 14 kHz LPF is already a gentle 2nd-order min-phase filter), look-ahead ducker (the 600 ms predictive pre-descent already beats it), colon→comma / ALL-CAPS text fixes (already in the preprocessor), dithering relocation (there is no dithering). Parked for a later "Issue-1 polish" round: upload LPF→high-shelf, `true_peak_limit` gain-envelope refactor, master bus saturation.

## Where to Look

| Need | Go to |
|------|-------|
| Full pipeline, FX params, QA thresholds, memory patterns | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Engine internals (Kokoro, F5, Lyria, Pedalboard) | [docs/model_implementation_guides/](docs/model_implementation_guides/) |
| Genre taxonomy, packs, music tagging, two-tier originality | [docs/auto_generation/README.md](docs/auto_generation/README.md) |
| Genre pack field contract, tag vocabulary, prose guidelines | [docs/genre_packs/README.md](docs/genre_packs/README.md) |
| Prompt writing per engine | [docs/prompting_guides/](docs/prompting_guides/) |
| Mix / post-processing details | [docs/optimization_and_processing/](docs/optimization_and_processing/) |
| Component & class map | [docs/COMPONENT_REGISTRY.md](docs/COMPONENT_REGISTRY.md) |
| File for a given task | [docs/TASK_ROUTING.md](docs/TASK_ROUTING.md) |
| All gotchas | [docs/GOTCHAS.md](docs/GOTCHAS.md) |
| Setup / checkpoints | [docs/setup_and_execution/](docs/setup_and_execution/) |
