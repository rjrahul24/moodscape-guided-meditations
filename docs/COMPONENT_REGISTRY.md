# Component Registry

Authoritative map of every class and module in `core/`. Use this when you need to find an entry point but the [Task Routing Guide](TASK_ROUTING.md) doesn't list your exact task.

## TTS

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| TTS contract | `core/speech_engine.py` | `SpeechEngine(ABC)` | `load_model`, `unload_model`, `synthesize`, `get_available_voices` |
| Kokoro engine | `core/kokoro_tts/engine.py` | `KokoroEngine` | `load_model()`, `synthesize()` |
| Kokoro preproc | `core/kokoro_tts/preprocessor.py` | — | `parse_script()`, `prepare_segments()`, `merge_sentences_to_chunks()` |
| Kokoro postproc | `core/kokoro_tts/postprocessor.py` | — | `process_chunk()`, `crossfade_chunks()`, `build_voice_chain()`, `apply_fx()` |
| Kokoro voices | `core/kokoro_tts/voice_manager.py` | — | `MEDITATION_PRESETS`, `blend_voices()`, `slerp_blend()`, `BRITISH_VOICES` |
| F5 engine | `core/f5_tts/engine.py` | `F5Engine` | `load_model()`, `synthesize()` |
| F5 preproc | `core/f5_tts/preprocessor.py` | — | `parse_script()`, `normalize_for_f5()`, `split_into_chunks()` |
| F5 voice registry | `core/f5_tts/voice_registry.py` | `VoiceRegistry` | `scan()`, `get_voice()` |

## Music & Pipeline

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| Pipeline | `core/pipeline.py` | `MeditationPipeline` | `generate()` |
| Lyria | `core/lyria/engine.py` | `LyriaEngine` | `load_model()`, `generate()`, `_run_session()` |
| Lyria prompts | `core/lyria/prompts.py` | — | `parse_weighted_prompts()` |
| Uploaded instrumental | `core/upload_music/engine.py` | `UploadMusicEngine` | `load_model()`, `unload_model()`, `generate()` |
| Upload length-fit | `core/upload_music/arrange.py` | `FitReport` | `fit_to_length()`, `_equal_power_curves()` |
| Audio FX | `core/audio_processor.py` | — | `make_{engine}_music_chain()` (incl. `make_upload_music_chain()`), `make_vocal_pocket_chain()`, `make_master_chain()`, `upsample_audio()` |
| Mixer | `core/mixer.py` | — | `apply_breathing_duck()`, `detect_phrases()`, `adaptive_vad_threshold()`, `calibrate_music_bed()`, `overlay_tracks()`, `mix()`, `normalize_loudness()`, `export_audio()` |
| QA monitor | `core/qa_monitor.py` | — | `run_qa_checks()`, `compute_composite_score()`, `check_voice_music_ratio()`, `check_ducking_smoothness()` |
| Stem separator | `core/stem_separator.py` | `StemSeparator` | `remove_drums_and_vocals()` |
| Text utils | `core/text_utils.py` | — | `expand_text()`, `ABBREV_MAP` |
| Breath sounds | `core/breath_sounds.py` | — | `load_breath()` |
| DeepFilter enhancer | `core/deepfilter_enhancer.py` | — | `enhance_voice_deepfilter()` |
| Stereo upmix | `core/stereo_upmix.py` | — | `haas_stereo()`, `center_pan_voice()` |

## Auto-Generation

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| Genre pack loader | `core/genres.py` | `GenrePack`, `Angle`, `GenrePackError` | `load_pack()`, `load_all_packs()`, `pick_angle()`, `genre_choices()` — reads + validates TOML packs at call time |
| Originality corpus + scoring | `core/originality.py` | `CorpusEntry` | `load_corpus()`, `add_to_corpus()`, `recent_angles()`, `avoid_terms()`, `assess()` — TF-IDF + rare-n-gram overlap |
| Background music tagging | `core/background_tags.py` | — | `extract_features()`, `tags_from_features()`, `tags_for()` — lazy librosa analysis, cached by (filename, size, mtime) |
| Script engine ABC + registry | `core/script_gen/engine.py` | `ScriptEngine(ABC)`, `FakeScriptEngine` | `complete()`, `unload()`, `build_engine()`, `parse_engine_spec()`, `preflight()` — provider registry `PROVIDER_BASE_URLS` / `PROVIDER_KEY_ENV` |
| Prompt assembly | `core/script_gen/rules.py` | — | `build_planner_system_prompt()`, `build_generator_system_prompt()`, `build_judge_system_prompt()`, `load_guide()`, `load_safety_rules()` |
| Planner (pass 0) | `core/script_gen/planner.py` | — | `plan()` — genre pack + angle → prose creative brief |
| Script generator (pass 1) | `core/script_gen/generator.py` | — | `draft()`, `strip_wrapper()` |
| Script judge (pass 2) | `core/script_gen/judge.py` | — | `review()`, `repair()`, `parse_judge_response()` |
| Script linter | `core/script_gen/linter.py` | `Violation` | `check()`, `check_format()`, `check_safety()`, `check_originality()`, `check_banned_phrases()`, `fatal_violations()`, `format_for_repair()` |
| Duration estimator | `core/script_gen/duration.py` | — | `estimate_duration_sec()`, `log_estimate_accuracy()` (now called automatically by `auto_generate.py :: run()`), `DEFAULT_WPM` |
| OpenAI-compatible adapter | `core/script_gen/adapters/openai_compat.py` | `OpenAICompatEngine` | `complete()`, `unload()` — covers ollama, openrouter, together, fireworks, groq; hand-rolled retry-with-backoff (`MOODSCAPE_SCRIPT_MAX_RETRIES`) |
| Anthropic adapter | `core/script_gen/adapters/anthropic_api.py` | `AnthropicEngine` | `complete()` — no hand-rolled retry; passes `max_retries` to the SDK client, derived from `MOODSCAPE_SCRIPT_MAX_RETRIES` |
| Auto-generate orchestrator | `core/auto_generate.py` | `AutoConfig`, `ScriptOutcome`, `AutoResult`, `ScriptGenerationError`, `DURATION_BANDS` | `generate_script()`, `run()` — genre → planner → writer → judge → linter → render → persist |
| Background picker (tag-filtered) | `core/background_picker.py` | — | `pick_background()`, `prefer_tags=` argument |
| Model benchmark harness | `core/bench.py` | `BenchRow` | `run_bench()`, `format_bench_table()`, `BENCH_PROMPTS` |
| Streaming progress runner | `core/streaming_run.py` | `StreamingRun`, `ProgressUpdate` | `__iter__()` — yields progress, exposes `.result` / `.error` |
| Auto-Generate UI tab | `core/auto_tab.py` | — | `build_auto_tab()`, `auto_generate_handler()` — genre dropdown, band radio, steer accordion |
