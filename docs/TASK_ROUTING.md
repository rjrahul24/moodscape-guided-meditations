# Task Routing Guide

When you need to change something, start here. Locate the row that matches your task and open the **Primary file** first.

| Task | Primary file | Secondary |
|------|-------------|-----------|
| TTS chunk splitting | `core/kokoro_tts/preprocessor.py` | `core/f5_tts/preprocessor.py` |
| Voice blend presets (Kokoro) | `core/kokoro_tts/voice_manager.py` | — |
| Add / edit F5 voice | `assets/speakers/` (audio + transcript) | `core/f5_tts/voice_registry.py` |
| Kokoro prosody / punctuation | `core/kokoro_tts/preprocessor.py` | — |
| Voice FX chain (EQ, reverb, compression) | `core/kokoro_tts/postprocessor.py :: build_voice_chain()` | `core/f5_tts/postprocessor.py` |
| Music FX chain (per engine) | `core/audio_processor.py :: make_{engine}_music_chain()` | — |
| Vocal pocket / intelligibility EQ | `core/audio_processor.py :: make_vocal_pocket_chain()` | — |
| Ducking behavior | `core/mixer.py :: apply_breathing_duck()` / `compute_breathing_gain_db()` | `core/pipeline.py` (`duck_amount_db`) |
| Music bed level (automated) | `core/mixer.py :: calibrate_music_bed()` / `adaptive_vad_threshold()` | `core/pipeline.py` (`MOODSCAPE_ADAPTIVE_BED`) |
| LUFS target | `core/pipeline.py` | `core/mixer.py :: export_audio()` |
| Uploaded-instrumental music source | `core/upload_music/engine.py :: UploadMusicEngine` | `core/pipeline.py` (`uploaded_music_path`, `music_model="upload"`), `app.py` (upload widget) |
| How an upload is looped/trimmed to length | `core/upload_music/arrange.py :: fit_to_length()` | — |
| Uploaded-instrumental FX chain | `core/audio_processor.py :: make_upload_music_chain()` | — |
| QA checks / thresholds | `core/qa_monitor.py` | `docs/ARCHITECTURE.md#qa-checks` |
| Stem separation behavior | `core/stem_separator.py` | `scripts/separate_worker.py` |
| Export format / sample rate | `core/mixer.py :: export_audio()` | `core/pipeline.py` (`export_sr`) |
| Master chain (final limiter/EQ) | `core/audio_processor.py :: make_master_chain()` | — |
| Text normalization (digits, abbrevs) | `core/text_utils.py` | — |
| DeepFilter voice enhancement | `core/deepfilter_enhancer.py` | `core/pipeline.py` (toggle) |
| Stereo upmix (Haas) | `core/stereo_upmix.py` | `core/pipeline.py` |
| Breath sound loading | `core/breath_sounds.py` | `scripts/generate_breath_samples.py` |
| Change how scripts are written | `docs/prompting_guides/` | `core/script_gen/rules.py` |
| Add a safety rule | `docs/prompting_guides/content_safety_rules.md` | `core/script_gen/linter.py` **and** a case in `tests/unit/test_script_linter.py` |
| Add a model provider | `core/script_gen/engine.py` (provider registry) | An adapter under `core/script_gen/adapters/` |
| Tune duration accuracy | `core/script_gen/duration.py :: DEFAULT_WPM` | `core/auto_generate.py :: run()` (auto-logs `actual_sec`/`estimate_ratio` into `meta.json` via `log_estimate_accuracy()`) |
| Tune script-gen retry behaviour | `core/script_gen/adapters/openai_compat.py` (hand-rolled backoff loop) | `core/script_gen/adapters/anthropic_api.py` (SDK `max_retries`, no second loop) — both read `MOODSCAPE_SCRIPT_MAX_RETRIES` |
| Add/adjust a script-format check | `core/script_gen/linter.py :: check_format()` | A case in `tests/unit/test_script_linter.py` |
| Add a genre | `docs/genre_packs/<slug>.toml` | Read [docs/genre_packs/README.md](../genre_packs/README.md); validate with `.venv/bin/python -c "from core.genres import load_all_packs; load_all_packs()"` |
| Change a genre's angles or imagery | `docs/genre_packs/<slug>.toml` | Affects `genres.pick_angle()` and the planner's brief on next run (no restart needed) |
| Retune originality thresholds | `MOODSCAPE_ORIGINALITY_FATAL`, `MOODSCAPE_ORIGINALITY_ADVISORY` env vars | Provisional; calibrate from `max_similarity` scores logged on every render (stored in `meta.json`) |
| Retune genre music tags | `docs/genre_packs/<slug>.toml :: music_tags` | Affects `background_picker.pick_background(prefer_tags=…)` on next run |
| Tag new background music | Drop `.wav` / `.mp3` etc. into `assets/backgrounds/` | Measured tags (dark/warm/bright, drone/evolving, sparse/busy, etc.) are auto-extracted on first use (~2s); declared tags keyed by filename in `assets/backgrounds/tags.toml` |
| Force re-tag all backgrounds | `python scripts/tag_backgrounds.py --report` | Re-extract measured features, preserve declared tags |
| Change the planner model | `MOODSCAPE_SCRIPT_PLANNER` env var | Planner + writer are loaded once; switching planner changes only the brief writer. Must be same provider protocol as writer (both Ollama, or both Anthropic, etc.). |
| Change the writer model | `MOODSCAPE_SCRIPT_GENERATOR` env var | Shared load with planner; check `MOODSCAPE_SCRIPT_PLANNER` is compatible (or leave it to share the value from `_GENERATOR`) |
| Change the judge model | `MOODSCAPE_SCRIPT_JUDGE` env var | **Must be independent** — a different model family than the writer, not the same weights. Affects quality of review/repair. |
| Change the Manual / Auto-Generate tab layout | `app.py` (`gr.Tabs()` container) | `core/auto_tab.py` (Auto-Generate tab content + `elem_classes` styling) — see [app_wiring.md](auto_generation/app_wiring.md) |
| Change the Auto-Generate UI controls (dropdown, radio, etc.) | `core/auto_tab.py` | Wiring lives in `auto_generate_handler()` callback; genre on-change handler calls `genres.load_pack()` to pre-fill voice engine and content type |
| Benchmark a model pairing | `python scripts/eval_genres.py --pair writer:model judge:model ...` | Renders matrix of genres × model configs, writes metrics to comparison dir |
