# Common Gotchas

Hard-won lessons. The most load-bearing entries are repeated in `CLAUDE.md`; the full list lives here.

## Apple Silicon / MPS

- **MPS bus error on exit**: `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove.
- **PYTORCH_ENABLE_MPS_FALLBACK=1**: Set in `app.py` so ops without MPS kernels fall back to CPU instead of crashing.

## Kokoro TTS

- **Forced to CPU**: MPS causes deallocation bus errors with Kokoro. Do not switch device.
- **British voices need `lang_code="b"`**: Voices prefixed `bf_*` / `bm_*` require a separate `KPipeline(lang_code="b")`.
- **`trf=True`**: Transformer G2P produces better phonemization than the default for meditation prosody.

## F5-TTS

- **Requires verbatim transcript**: Each reference audio in `assets/speakers/` needs a matching `.txt` of the same slug under `assets/speakers/transcripts/`.
- **VAD on**: Silero VAD is loaded via `torch.hub.load('snakers4/silero-vad', 'silero_vad')` to trim silence at chunk boundaries.

- **Cache fingerprint**: TTS cache keys include every `MOODSCAPE_F5_*` / `MOODSCAPE_CHATTERBOX_*` env var. A new TTS-affecting env var must use that prefix or the cache serves a stale voice. Mix-only flags (`MOODSCAPE_BED_*`, `MOODSCAPE_LOUDNESS_*`) are deliberately outside it, so mix-only eval arms reuse cached TTS.
- **F5 chunking/QA are opt-in**: `MOODSCAPE_F5_CHUNKING=1` renders breath groups with punctuation-length gaps; `MOODSCAPE_F5_CHUNK_QA=1` (requires chunking) re-renders failing chunks best-of-3, records `speech_qa` in `meta.json` and never fails a job.

## Mixing & Mastering

- **Active ducking function**: `mixer.mix()` calls `apply_breathing_duck()` — a deep, gradual, script/VAD-aware sidechain duck. Phrases are detected from the voice (`detect_phrases`), a predictive S-curve descends ~600 ms before each phrase, holds at `duck_amount_db` during speech, releases over ~1.5 s, and lifts slightly during pauses ≥1.5 s so the bed "breathes". Applied **fullband** so the whole bed drops.
- **Adaptive bed calibration**: `pipeline` auto-derives `music_volume_db` / `duck_amount_db` from measured stem short-term LUFS (`mixer.calibrate_music_bed`, golden-path targets 14.5/30.5 LU under voice). `MOODSCAPE_ADAPTIVE_BED=0` forces the legacy fixed (−16, −16). A user-moved duck slider still overrides the calibrated duck.
- **No pedalboard `Limiter` anywhere**: pedalboard 0.9.23's `Limiter` inflates sub-threshold signals by ~+4.75 dB and adds broadband distortion ("static"). It was removed from `make_{upload,lyria}_music_chain()` and `make_master_chain()`. Don't reintroduce it.
- **Limiting is true-peak at export**: `export_audio()` does master EQ/glue → LUFS-normalize to playback −16 (sleep −18; mono files 3.01 dB lower, `MOODSCAPE_LOUDNESS_DUAL_MONO=0` restores plain −16) → `mixer.true_peak_limit()` to −1 dBTP (order matters: normalize first, then limit). `true_peak_limit` is a vectorized 4×-oversampled brickwall (no per-sample loop) and is transparent below threshold.
- **Bed overrides**: `MOODSCAPE_BED_SPEECH_OFFSET_LU` / `_PAUSE_OFFSET_LU` override profile offsets (env wins) and widen the duck clamp to (−20, −4) for all content types; `MOODSCAPE_BED_RISE_LIMIT=1` forces `lift_db` to 0 for every content type (sleep stories too) and caps intro/outro bed rise.
- **Default levels**: `duck_amount_db=-16` (`pipeline.py` / UI slider) = how low the bed sits under speech; `music_volume_db=-16` baseline in `mix()`. Lower (more negative) duck = quieter under speech.
- **Mix sample rate**: All music-source paths use `mix_sr = 48000` (`pipeline.py`). The 44.1 kHz fallback applies only when no music source is active.
- **Export target**: playback `−16 LUFS` (meditation) / `−18` (sleep story), `−1 dBTP` ceiling; mono files are written 3.01 dB lower so dual-mono playback hits the target. Matches Apple Music and avoids platform re-limiting.

## Uploaded Instrumental (`music_model="upload"`)

- **Engine output contract is load-bearing**: `UploadMusicEngine.generate()` must return **mono float32 @ 48 kHz, exactly `round(total_duration_sec*48000)` samples** — that is what lets the upload reuse the Lyria mix/duck/master path unchanged. Don't return stereo or a differently-sized array.
- **Stem separation is skipped for uploads**: guarded by `if stem_separation and not use_upload:` in `pipeline.py`. The user's file is already an instrumental; running Demucs on it is wasteful and not wanted. Don't remove the guard.
- **Don't apply fades in the upload engine**: `fit_to_length()` returns a bare fitted array. Pre/post-roll and fades are added later by `mixer.mix()` — applying them in the engine would double-fade.
- **Decoding uses `pedalboard.io.AudioFile`** (libsndfile/ffmpeg) so mp3/m4a/etc. work; UI validates the extension against `{.wav,.mp3,.flac,.ogg,.m4a,.aiff,.aif}` before the pipeline runs.

## Auto-Generation

- **The two script-gen adapters retry by different mechanisms — on purpose, don't "fix" it**: `script_gen/adapters/openai_compat.py` hand-rolls a jittered exponential-backoff loop around raw `httpx`, because `httpx` itself has no retry behaviour. `script_gen/adapters/anthropic_api.py` adds **no** second loop — the official `anthropic` SDK already retries connection errors, 408, 409, 429 and 5xx internally, governed by the client's `max_retries`. Wrapping the SDK's own retrying client in a second retry loop would multiply attempts (up to `max_retries^2`) and double the backoff for no benefit. The adapter instead constructs the client with an explicit `max_retries` so the budget is intentional. Both read `MOODSCAPE_SCRIPT_MAX_RETRIES` (default `3`, total attempts) so the two budgets stay in sync — the anthropic adapter converts it to the SDK's retries-only count (`total - 1`). Only connection errors, timeouts, 408, 409, 429, and 5xx are retried; a 400, a missing API key, or a malformed/empty response body fails on the first attempt.
- **Never let the repair prompt's `<problems>` block survive into the script**: `judge.py :: repair()` wraps the linter's findings in `<problems>…</problems>`, and a weak model frequently echoes that block back verbatim. `parse_judge_response()` strips it before anything else. This is not tidiness — it makes the repair loop *self-poisoning*, because the linter's own messages name tags and show marker examples ("Unknown tag '[UNK]'", "Use exactly [pause:Xs]"). An echoed problem list therefore re-triggers `UNKNOWN_TAG`, `MARKER_MALFORMED` and `ANGLE_TAG` on the text that *reports* those violations, so each repair round adds more violations than it fixes and the budget is exhausted on a script whose prose was fine. Observed with `ollama:llama3.2:3b`, the default engine: a draft with 2 real defects reached 20 violations after 2 repairs.
- **Fatal vs advisory violations**: `script_gen/linter.py` fails the job (no render) for safety hard-blocks and malformed markers, but renders anyway — with a logged warning — for duration drift and style issues. Treating every violation as fatal makes a weaker local model unusable; treating none as fatal lets a safety failure reach audio. Do not flatten this distinction.
- **Prompting guides are templates**: `{{NAME}}` placeholders in `docs/prompting_guides/vocal_*` are rendered from code constants by `script_gen/guide_context.py`. Never hard-code a chunk limit, speed, WPM or pause length in a guide. An unknown placeholder fails the run before any LLM call.
- **Slop is ADVISORY by design**: never make `SLOP_*` fatal; it would make local models fail on style.
- **Engine spec parsing splits on the first colon only**: `parse_engine_spec()` must handle Ollama model tags that themselves contain a colon — `ollama:qwen3:30b` splits into provider `ollama` and model `qwen3:30b`, not three pieces and not a truncated tag.
- **Prompting guides are read at call time, not import time**: `script_gen/rules.py :: load_guide()` / `load_safety_rules()` read the files under `docs/prompting_guides/` on every call. Editing a guide takes effect on the next generation — no code change, no restart.
- **`app.py` cannot be imported in a test**: it loads `torch`/Gradio and registers `atexit.register(lambda: os._exit(0))`, which would hijack pytest's exit code. This is why `core/streaming_run.py` exists as a separate, test-covered module — it runs `auto_generate.run()` on a background thread and streams progress without needing `app.py` at all; `core/auto_tab.py` only wires the two together for the real UI.
- **Fades are excluded from duration estimates**: `script_gen/duration.py :: estimate_duration_sec()` deliberately does not add fade time — `apply_fades` shapes amplitude over audio that already exists, so fades never extend runtime.

---

## Genre-Driven Generation

- **Ollama ignores `keep_alive` on `/v1/chat/completions`**: The OpenAI-compatible endpoint treats the `keep_alive` parameter as a no-op. It is only honoured on the native `/api/generate` endpoint. `ScriptEngine.unload()` and `OpenAICompatEngine.unload()` exist to force unload by calling `/api/generate` with `keep_alive=0` after each LLM stage (planner, writer, judge). Without this, an 18 GB model stays resident while the next model loads (F5-TTS: 14 GB) on a 32 GB machine, forcing swap — which is precisely where the MPS deallocation bus errors live. The orchestrator calls `unload()` after each stage. **Do not skip `unload()` calls.**
- **Originality repair depends on `<problems>` stripping being load-bearing**: The judge's repair prompt wraps linter violations in `<problems>…</problems>` so the model can see what failed. A weak model frequently echoes the block verbatim, poisoning the script with the violation report itself. `parse_judge_response()` in `script_gen/judge.py` strips `<problems>…</problems>` **before anything else** (a change locked in at commit `c372b18`). This is not cleanup — it prevents a self-poisoning repair loop. If the judge echoes `<problems>`, the linter reads them as stray angle-bracket markup (`ANGLE_TAG`, `UNKNOWN_TAG`), repair wraps fresh violations into the prompt, the next repair echoes those too, and the budget exhausts on a script whose prose was fine. Observed with weaker models: a draft with 2 real defects reached 20 violations after 2 repairs. **Do not remove this strip.**
- **`document_frequencies()` defaults to `max_n=3` while `longest_rare_run()` uses `n=5`**: A caller using both defaults (currently none do, but future code might) scores every 5-gram as "rare" and silently defeats the filter. When adding new originality checks, ensure both functions use the same n-gram size or explicitly control it.
- **`httpx` does not raise on 4xx/5xx automatically**: `script_gen/adapters/openai_compat.py` checks status codes explicitly in `_send_once()` because bare `httpx` tolerates any status code. Transient errors (408, 409, 429, 5xx) are wrapped in `_Transient` for retry; fatal errors (400, other 4xx) raise `RuntimeError` directly. A missing status-code check logs nothing on an HTTP error (200 OK assumed), and the "response" is an error page. Currently mitigated by explicit checks; do not remove them.
- **A banned phrase in a genre pack is a case-insensitive substring and is fatal**: A pack with `banned = ["relax"]` will reject any script containing "relax", "Relax", "RELAX", or as part of a larger word. This blocks otherwise-good drafts. Prefer multi-word phrases ("let it go", "breathe deeply") over single words; a word like "relax" is too common. Fatal violations go to repair; without a repair-loop fix, the script fails. Test a pack's banned list against a few generations before shipping it.

## Content Safety & Duration

- **Rules must never require sensitive wording**: the old R1 disclaimer forced scripts to name medical conditions and pregnancy. Breathwork now gets only a gentle "breathe naturally" line. `SENSITIVE_TOPIC` (FATAL) is global across all genres: no medical conditions, pregnancy, illness/treatment, hospitals, or children/babies/younger self ("child's pose" is allowed).
- **Content check is judge-based and bounded**: `MOODSCAPE_CONTENT_CHECK=0` disables it; it shares the repair budget, caps judge calls (`CONTENT_CHECK_MAX_CALLS`), and findings after the budget are ADVISORY. It allows the pack's own subject; do not tighten its class list without checking pain/grief packs.
- **Duration control never fails a job**: one length repair (>15% outside the window), then pause scaling x0.85-1.15; `MOODSCAPE_DURATION_CONTROL=0` restores plain ADVISORY drift.
- **`MOODSCAPE_F5_TRIM_LEADIN`** (default 0) trims the ref-pad lead-in per F5 chunk; it is TTS-affecting (in the cache key via the `MOODSCAPE_F5_` prefix).
- **Eval chains are generated, not run**: `scripts/ab_chain.py` only prints/writes `chain.sh`; start it only with the user's go-ahead.
