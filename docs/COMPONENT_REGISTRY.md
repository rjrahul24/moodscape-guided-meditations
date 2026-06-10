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
| IndexTTS engine | `core/index_tts/engine.py` | `IndexTTSEngine` | `load_model()`, `synthesize()` |
| IndexTTS preproc | `core/index_tts/preprocessor.py` | — | `parse_script()`, `normalize_for_indextts()`, `split_into_chunks()` |
| IndexTTS postproc | `core/index_tts/postprocessor.py` | `IndexTTSMasteringEngine` | `master_vocals()`, `build_index_voice_chain()` |
| IndexTTS voice reg | `core/index_tts/voice_registry.py` | — | `scan_voices()`, `scan_emotions()`, `get_voice()`, `get_emotion()` |

## Music & Pipeline

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| Pipeline | `core/pipeline.py` | `MeditationPipeline` | `generate()`, `_enhance_acestep_prompt()` |
| ACE-Step | `core/acestep/engine.py` | `AceStepEngine` | `load_model()`, `generate()`, `_generate_infinite()`, `_enhance_prompt()` |
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
