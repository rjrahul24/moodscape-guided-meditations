# MoodScape Guided Meditations

AI-guided meditation audio generator using Gradio, multiple TTS engines, and AI music generation. Target hardware: Apple Silicon M1 Max (36 GB unified RAM).

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # Gradio UI at http://localhost:7860
```

**System dependency:** `brew install espeak-ng` (required by Kokoro TTS)

**Environment variables:** `.env` file (gitignored) must contain `HF_TOKEN` and `GOOGLE_API_KEY`. App also sets `TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=1` at startup.

## Task Workflow

IMPORTANT: Follow these steps for every task.

1. **Use the Task-Routing Guide below** — find the relevant file(s) before scanning broadly
2. **Read docs first** — each doc file has a QUICK-REF header; read that before the full doc
   - `docs/model_implementation_guides/` — engine internals (ACE-Step, Kokoro, F5-TTS, HeartMuLa, Lyria, Pedalboard)
   - `docs/optimization_and_processing/` — audio pipeline and post-processing
   - `docs/prompting_guides/` — prompt engineering per engine
   - `docs/ARCHITECTURE.md` — full technical deep-dive (FX params, QA thresholds, memory patterns)
3. **Implement changes**
4. **Update documentation** — keep `docs/` in sync; if a function signature or engine behavior changes, update the corresponding doc file
5. **Update README.md** — only for significant changes (new feature, removed capability, changed setup)
6. **Run tests** — unit tests for small changes, integration tests for large changes
7. **Do NOT push to GitHub** unless explicitly asked
8. **Update this CLAUDE.md** if you discover new development patterns worth noting

## Build & Test

```bash
.venv/bin/python -m pytest unit-tests/ -v                                    # all unit tests
.venv/bin/python -m pytest unit-tests/test_mixer.py -v                       # single file
.venv/bin/python -m pytest unit-tests/test_meditation_mastering.py::test_mastering -v
.venv/bin/python -m pytest integration-tests/ -v                             # slower, full pipeline
python scripts/generate.py <script_file> --voice <voice_name> --output <output.wav>
```

## Git Workflow

- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- **Commit directly to main** — no feature branches (solo developer)
- **Never push** unless the user explicitly asks

---

## Architecture

**Pipeline flow** (`core/pipeline.py` — `MeditationPipeline.generate()`):

```
app.py / scripts/generate.py
  → core/pipeline.py :: MeditationPipeline.generate()
     1. parse script     kokoro_tts/preprocessor.py :: prepare_segments()
                         f5_tts/preprocessor.py :: prepare_segments()      [if f5]
     2. TTS synthesis    KokoroEngine.synthesize()  → 24 kHz mono float32
                         F5Engine.synthesize()       → 24 kHz mono float32
     3. unload TTS, load music engine
     4. music gen        AceStepEngine.generate()   → 48 kHz mono float32
                           [optional] melody_audio_path → loaded as numpy + sr, passed as melody_audio/melody_sample_rate kwargs
                         HeartMulaEngine.generate()  → 48 kHz mono float32
                         LyriaEngine.generate()      → 48 kHz mono float32
     5. stem sep         StemSeparator.remove_drums_and_vocals()            [optional]
     6. TTS upsample     audio_processor.upsample_audio() 24 kHz → 48 kHz
     7. voice FX         kokoro_tts/postprocessor :: build_voice_chain() + apply_fx()
    7b. voice norm       mixer.normalize_loudness() → −18 LUFS pre-mix
     8. music FX         audio_processor :: make_{engine}_music_chain() + make_vocal_pocket_chain()
     9. mix              mixer.mix() → apply_multiband_ducking() + overlay + exponential fades
    10. master           audio_processor :: make_master_chain() [HPF → bus comp → limiter@−1.5dBTP]
    11. QA              qa_monitor :: run_qa_checks()
    12. export           mixer.export_audio() → WAV/MP3 at −16 LUFS, −1.5 dBTP ceiling
```

**Sequential engine loading**: Load TTS → synthesize → unload TTS → load music → generate → unload music. Never load two engines simultaneously (36 GB unified RAM constraint).

## Audio Contracts

| Engine | Native SR | Mix SR | Output |
|--------|-----------|--------|--------|
| Kokoro TTS | 24 kHz | → upsampled to 48 kHz | mono float32 |
| F5-TTS | 24 kHz | → upsampled to 48 kHz | mono float32 |
| ACE-Step 1.5 | 48 kHz | 48 kHz | mono float32 |
| HeartMuLa | 48 kHz | 48 kHz | mono float32 |
| Lyria RealTime | 48 kHz | 48 kHz | mono float32 |

Upsample method: `audio_processor.upsample_audio(high_accuracy=True)` = `librosa soxr_vhq`.

## Component Registry

**TTS:**

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

**Music & Pipeline:**

| Component | File | Class | Key Methods |
|-----------|------|-------|-------------|
| Pipeline | `core/pipeline.py` | `MeditationPipeline` | `generate()`, `_enhance_heartmula_prompt()`, `_enhance_acestep_prompt()` |
| ACE-Step | `core/acestep_engine.py` | `AceStepEngine` | `load_model()`, `generate()`, `_generate_infinite()`, `_enhance_prompt()` |
| HeartMuLa | `core/heart_mula/engine.py` | `HeartMulaEngine` | `load_model()`, `generate()`, `_generate_mlx()`, `_generate_mps()` |
| Lyria | `core/lyria/engine.py` | `LyriaEngine` | `load_model()`, `generate()`, `_run_session()` |
| Lyria prompts | `core/lyria/prompts.py` | — | `parse_weighted_prompts()` |
| Audio FX | `core/audio_processor.py` | — | `make_{engine}_music_chain()`, `make_vocal_pocket_chain()`, `make_master_chain()`, `upsample_audio()` |
| Mixer | `core/mixer.py` | — | `apply_envelope_ducking()`, `apply_multiband_ducking()`, `overlay_tracks()`, `mix()`, `normalize_loudness()`, `export_audio()` |
| QA monitor | `core/qa_monitor.py` | — | `run_qa_checks()`, `compute_composite_score()`, `check_voice_music_ratio()`, `check_ducking_smoothness()` |
| Stem separator | `core/stem_separator.py` | `StemSeparator` | `remove_drums_and_vocals()` |
| Session config | `core/session_config.py` | `SessionConfig` | `to_json()`, `from_json()` |
| Text utils | `core/text_utils.py` | — | `expand_text()`, `ABBREV_MAP` |
| Breath sounds | `core/breath_sounds.py` | — | `load_breath()` |
| Neural enhancer | `core/neural_enhancer.py` | — | `enhance_with_apollo()` |
| Stereo upmix | `core/stereo_upmix.py` | — | `haas_stereo()`, `center_pan_voice()` |

## Key Constants

| Constant | Value | File | Note |
|----------|-------|------|------|
| TTS output SR | 24 000 Hz | `core/speech_engine.py` | All TTS engines |
| Mix SR (all engines) | 48 000 Hz | `core/pipeline.py:213` | ACE-Step / HeartMuLa / Lyria / F5 paths |
| Fallback mix SR | 44 100 Hz | `core/pipeline.py:213` | Only when no music engine selected |
| Export LUFS target | −16.0 | `core/pipeline.py` | Matches Apple Music; avoids platform re-limiting |
| Kokoro crossfade | 7 200 samples | `core/kokoro_tts/postprocessor.py` | 300ms at 24 kHz |
| Kokoro max tokens | 150 | `core/kokoro_tts/preprocessor.py` | Per chunk |
| F5 max chars | 300 | `core/f5_tts/preprocessor.py` | Per chunk |
| ACE-Step CFG | 5.0 | `core/acestep_engine.py` | `_GUIDANCE_SCALE` — SFT sweet spot |
| ACE-Step steps | 50 | `core/acestep_engine.py` | `_INFERENCE_STEPS` |
| ACE-Step LM temp | 0.4 | `core/acestep_engine.py` | `_LM_TEMPERATURE` — conservative for meditation |
| HeartMuLa CFG | 2.5 | `core/heart_mula/engine.py` | `_LM_CFG_SCALE` |
| HeartMuLa temp | 0.75 | `core/heart_mula/engine.py` | `_LM_TEMPERATURE` |
| HeartMuLa top_k | 30 | `core/heart_mula/engine.py` | `_LM_TOP_K` |
| HeartMuLa codec guidance | 1.25 | `core/heart_mula/engine.py` | `_CODEC_GUIDANCE_SCALE` |
| HeartMuLa codec steps | 12 | `core/heart_mula/engine.py` | `_CODEC_NUM_STEPS` |
| HeartMuLa max seg | 240 s | `core/heart_mula/engine.py` | `MAX_SEGMENT_SEC` |
| Lyria max session | 570 s | `core/lyria/engine.py` | `_MAX_SESSION_SEC` (9.5 min cap) |
| MPS watermark | 0.7 | `core/heart_mula/engine.py` | `PYTORCH_MPS_HIGH_WATERMARK_RATIO` |

## Checkpoint Locations

| Engine | Backend | Path |
|--------|---------|------|
| ACE-Step 1.5 | MLX + MPS | `./ACE-Step-1.5/checkpoints/` |
| HeartMuLa LM | MPS | `./ckpt/HeartMuLa-oss-3B/` |
| HeartCodec | MPS | `./ckpt/HeartCodec-oss/` |
| HeartMuLa LM | MLX | `./ckpt-mlx/heartmula/` |
| HeartCodec | MLX | `./ckpt-mlx/heartcodec/` |
| Kokoro-82M | HF hub (auto) | `~/.cache/huggingface/` |
| F5-TTS | HF hub (auto) | `~/.cache/huggingface/` |
| HT Demucs | torch hub (auto) | `~/.cache/torch/hub/` |
| Convolution IRs | local | `assets/impulse_responses/{warm_studio,wooden_hall,stone_chapel}.wav` |
| F5 voice assets | local | `core/f5_tts/assets/reference_audio/` · `.../reference_transcript/` · `.../voices.toml` |

## Task-Routing Guide

| Task | Primary file | Secondary |
|------|-------------|-----------|
| TTS chunk splitting | `core/kokoro_tts/preprocessor.py` | `core/f5_tts/preprocessor.py` |
| Voice blend presets | `core/kokoro_tts/voice_manager.py` | — |
| Add / edit F5 voice | `core/f5_tts/assets/` (audio + transcript) | `core/f5_tts/voice_registry.py` |
| Kokoro prosody / punctuation | `core/kokoro_tts/preprocessor.py` | — |
| Voice FX chain (EQ, reverb, compression) | `core/kokoro_tts/postprocessor.py :: build_voice_chain()` | `core/f5_tts/postprocessor.py` |
| Music FX chain (per engine) | `core/audio_processor.py :: make_{engine}_music_chain()` | — |
| Vocal pocket / intelligibility EQ | `core/audio_processor.py :: make_vocal_pocket_chain()` | — |
| Ducking behavior (fullband) | `core/mixer.py :: apply_envelope_ducking()` | `core/pipeline.py` (`duck_amount_db`) |
| Ducking behavior (multiband) | `core/mixer.py :: apply_multiband_ducking()` | `core/mixer.py :: mix(multiband=True)` |
| LUFS target | `core/pipeline.py` | `core/mixer.py :: export_audio()` |
| ACE-Step generation params | `core/acestep_engine.py` (module-level constants) | — |
| ACE-Step reference audio (melody conditioning) | `core/pipeline.py` (`melody_audio_path` param) | `core/acestep_engine.py :: _prepare_reference_audio()` |
| HeartMuLa generation params | `core/heart_mula/engine.py` (module-level constants) | — |
| Prompt enhancement logic | `core/pipeline.py :: _enhance_heartmula_prompt()` | `core/acestep_engine.py :: _enhance_prompt()` |
| QA checks / thresholds | `core/qa_monitor.py` | `docs/ARCHITECTURE.md#qa-checks` |
| Stem separation behavior | `core/stem_separator.py` | `scripts/separate_worker.py` |
| Export format / sample rate | `core/mixer.py :: export_audio()` | `core/pipeline.py` (`export_sr`) |
| Master chain (final limiter/EQ) | `core/audio_processor.py :: make_master_chain()` | — |
| Session reproducibility | `core/session_config.py` | — |
| Text normalization (digits, abbrevs) | `core/text_utils.py` | — |

## FX Chain Summary

| Chain function | Applied to | Key plugins (in order) |
|----------------|-----------|------------------------|
| `build_voice_chain()` | Kokoro voice | NoiseGate(−42dB) → HPF(80Hz) → LowShelf(+2dB@200Hz) → Peak(−2dB@350Hz) → Compressor(2:1@−18dB) → Peak(−2.5dB@3kHz) → HiShelf(−4dB@7.5kHz) → ConvReverb(IR) → LPF(9.5kHz) → Limiter(−1dB) |
| `build_f5_voice_chain()` | F5-TTS voice | De-esser(4–8kHz) → NoiseGate → HPF → EQ chain → Compressor → Limiter — see `core/f5_tts/postprocessor.py` |
| `make_heartmula_music_chain()` | HeartMuLa 48kHz | NoiseGate(−52dB) → HPF(60Hz) → LowShelf(+1.5dB@100Hz) → LowShelf(+2.0dB@150Hz) → Peak(−1dB@220Hz) → Peak(−2dB@4kHz,Q=0.5) → HiShelf(−2dB@9.5kHz) → LPF(14kHz) → ConvReverb(stone_chapel,15%wet) → Compressor(2:1@−20dB) → Limiter(−0.5dB) |
| `make_acestep_music_chain()` | ACE-Step 48kHz | NoiseGate(−50dB) → HPF(60Hz) → LowShelf(+2.5dB@200Hz) → Peak(−2dB@3kHz) → Peak(−1.5dB@4kHz) → Peak(−2dB@6kHz) → HiShelf(+0.5dB@8kHz) → HiShelf(−2.5dB@10kHz) → HiShelf(+1dB@12kHz) → LPF(16kHz) → Compressor(2.0:1@−20dB) → Limiter(−0.5dB) |
| `make_lyria_music_chain()` | Lyria 48kHz | HPF(60Hz) → Peak(−1.5dB@250Hz) → HiShelf(−2.5dB@9kHz) → Compressor(2:1@−18dB) → Limiter(−0.5dB) |
| `make_vocal_pocket_chain()` | Music (all engines) | HPF(30Hz) → Peak(−3dB@300Hz) → Peak(−2dB@1kHz) → Peak(−2dB@3kHz) → LPF(12kHz) |
| `make_master_chain()` | Final mix | HPF(30Hz) → Compressor(1.5:1@−22dB, 40ms/300ms) → Limiter(−1.5dB, 400ms) |

All chains in `core/audio_processor.py`. IRs in `assets/impulse_responses/` (default: `warm_studio`). Full parameter tables → `docs/ARCHITECTURE.md#fx-chains`.

---

## Code Conventions

- Classes: `PascalCase` — Functions/methods: `snake_case` — Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`
- Imports: `from core.module import Class` (relative to project root)
- Each subpackage has `__init__.py` with explicit public API exports
- Abstract base class pattern: `SpeechEngine` ABC → concrete engine implementations

## Common Gotchas

- **MPS bus error on exit**: `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove
- **ACE-Step timeout**: Always pass `compile_model=True` to `initialize_service()` — without it, generation takes ~9s/step and times out (~135s JIT overhead on first run, then ~4× faster per step)
- **transformers version**: Pinned to `>=4.51.0,<4.58.0` for ACE-Step compatibility — do not upgrade
- **Kokoro on CPU**: Intentionally forced to CPU — MPS causes deallocation bus errors. British voices (`bf_*`, `bm_*`) require `KPipeline(lang_code="b")`
- **HeartMuLa simultaneous loading (MLX)**: Both LM (bf16) + codec (fp32) loaded together (~12 GB). `mx.set_memory_limit(30GB)`, `mx.set_cache_limit(4GB)`. MPS: `lazy_load=True` still handles lifecycle.
- **HeartCodec dtype**: Always fp32 — bf16 causes metallic artifacts. MLX passes `mx.float32`; MPS passes `{"codec": torch.float32}`.
- **HeartMuLa MPS watermark**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` (set in `engine.py`). Values below 0.5 cause OOM during 3B LM generation.
- **HeartMuLa checkpoints**: Must be present before selecting HeartMuLa — see Checkpoint Locations table above.
- **Active ducking function**: `mixer.mix()` calls `apply_multiband_ducking()` by default (`multiband=True`). Falls back to `apply_envelope_ducking()` when `multiband=False`. `apply_rms_ducking()` exists but is not used in production.
- **Mix sample rate**: All music engine paths use `mix_sr = 48000` (`pipeline.py:213`). The 44.1 kHz fallback is only used when no music engine is active.
