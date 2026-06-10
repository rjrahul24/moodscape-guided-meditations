# Task Routing Guide

When you need to change something, start here. Locate the row that matches your task and open the **Primary file** first.

| Task | Primary file | Secondary |
|------|-------------|-----------|
| TTS chunk splitting | `core/kokoro_tts/preprocessor.py` | `core/f5_tts/preprocessor.py`, `core/index_tts/preprocessor.py` |
| Voice blend presets (Kokoro) | `core/kokoro_tts/voice_manager.py` | — |
| Add / edit F5 voice | `assets/speakers/` (audio + transcript) | `core/f5_tts/voice_registry.py` |
| Add / edit IndexTTS voice | `assets/speakers/` (WAV only) | `core/index_tts/voice_registry.py` |
| Add / edit IndexTTS emotion | `assets/emotions/` (WAV only) | `core/index_tts/voice_registry.py` |
| Kokoro prosody / punctuation | `core/kokoro_tts/preprocessor.py` | — |
| Voice FX chain (EQ, reverb, compression) | `core/kokoro_tts/postprocessor.py :: build_voice_chain()` | `core/f5_tts/postprocessor.py`, `core/index_tts/postprocessor.py` |
| Music FX chain (per engine) | `core/audio_processor.py :: make_{engine}_music_chain()` | — |
| Vocal pocket / intelligibility EQ | `core/audio_processor.py :: make_vocal_pocket_chain()` | — |
| Ducking behavior | `core/mixer.py :: apply_breathing_duck()` / `compute_breathing_gain_db()` | `core/pipeline.py` (`duck_amount_db`) |
| Music bed level (automated) | `core/mixer.py :: calibrate_music_bed()` / `adaptive_vad_threshold()` | `core/pipeline.py` (`MOODSCAPE_ADAPTIVE_BED`) |
| ACE-Step long-form strategy | `core/acestep/engine.py :: _generate_looped()` / `_generate_infinite()` | `pipeline.generate(acestep_long_form_mode=…)` |
| LUFS target | `core/pipeline.py` | `core/mixer.py :: export_audio()` |
| ACE-Step generation params | `core/acestep/engine.py` (module-level constants) | — |
| ACE-Step reference audio (melody conditioning) | `core/pipeline.py` (`melody_audio_path` param) | `core/acestep/engine.py :: _prepare_reference_audio()` |
| Uploaded-instrumental music source | `core/upload_music/engine.py :: UploadMusicEngine` | `core/pipeline.py` (`uploaded_music_path`, `music_model="upload"`), `app.py` (upload widget) |
| How an upload is looped/trimmed to length | `core/upload_music/arrange.py :: fit_to_length()` | — |
| Uploaded-instrumental FX chain | `core/audio_processor.py :: make_upload_music_chain()` | — |
| Prompt enhancement logic | `core/pipeline.py :: _enhance_acestep_prompt()` | `core/acestep/engine.py :: _enhance_prompt()` |
| QA checks / thresholds | `core/qa_monitor.py` | `docs/ARCHITECTURE.md#qa-checks` |
| Stem separation behavior | `core/stem_separator.py` | `scripts/separate_worker.py` |
| Export format / sample rate | `core/mixer.py :: export_audio()` | `core/pipeline.py` (`export_sr`) |
| Master chain (final limiter/EQ) | `core/audio_processor.py :: make_master_chain()` | — |
| Text normalization (digits, abbrevs) | `core/text_utils.py` | — |
| DeepFilter voice enhancement | `core/deepfilter_enhancer.py` | `core/pipeline.py` (toggle) |
| Stereo upmix (Haas) | `core/stereo_upmix.py` | `core/pipeline.py` |
| Breath sound loading | `core/breath_sounds.py` | `scripts/generate_breath_samples.py` |
