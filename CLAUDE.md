# MoodScape Guided Meditations

AI-guided meditation audio generator (Gradio UI). Three TTS engines (Kokoro, F5-TTS, IndexTTS-2) and three music sources (ACE-Step 1.5, Lyria RealTime, or a user-uploaded instrumental). Target hardware: Apple Silicon M1 Max (36 GB unified RAM).

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
brew install espeak-ng                 # Kokoro G2P dependency
brew install rubberband                # IndexTTS-2 pacing time-stretch (REQUIRED for slow pacing; without it chunks stay unstretched)
python app.py                          # Gradio UI at http://localhost:7860
```

`.env` must define `HF_TOKEN` and `GOOGLE_API_KEY`. `app.py` sets `TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=1` at startup.

## Build & Test

```bash
.venv/bin/python -m pytest tests/unit/ -v                                    # all unit tests
.venv/bin/python -m pytest tests/unit/test_mixer.py -v                       # single file
.venv/bin/python -m pytest tests/integration/ -v                             # full pipeline (slow)
python scripts/generate.py <script_file> --voice <voice_name> --output <out.wav>
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
│   ├── text_utils.py · breath_sounds.py · stereo_upmix.py · deepfilter_enhancer.py
│   ├── kokoro_tts/  f5_tts/  index_tts/   # TTS engines (engine + preproc + postproc + voices)
│   ├── acestep/  lyria/                   # Generative music engines
│   └── upload_music/                      # User-uploaded instrumental (engine + arrange/length-fit)
├── scripts/                          # generate.py · separate_worker.py · generate_breath_samples.py
├── tests/unit/  tests/integration/
├── assets/                           # tracked in git
│   ├── breath_sounds/                # [breath]/[inhale]/[exhale] samples
│   ├── impulse_responses/            # convolution reverb IRs
│   ├── speakers/                     # shared voice pool (F5 + IndexTTS)
│   │   ├── reference_audio/*.wav     #   speaker reference clips (both engines)
│   │   ├── reference_text/*.txt      #   F5-only transcripts (paired by slug)
│   │   └── voices.toml               #   F5 multi-phase definitions
│   └── emotions/                     # IndexTTS-only emotion references
├── models/                           # gitignored; all model weights
│   ├── acestep/                      # ACE-Step 1.5 (source + checkpoints/)
│   ├── indextts2/                    # IndexTTS-2 (manual HF download)
│   └── hf_cache/                     # project-local HF cache
└── docs/                             # see Where to Look below
```

## Pipeline Flow (`core/pipeline.py :: MeditationPipeline.generate()`)

1. **Parse script** → `{tts}/preprocessor.py :: prepare_segments()`
2. **TTS synth** → 24 kHz mono float32
3. **Unload TTS**, load music engine (sequential — 36 GB RAM constraint)
4. **Music gen** → 48 kHz mono float32 (ACE-Step, Lyria, or `upload_music` — decode + resample + loop/trim-fit the uploaded file to the same contract)
5. **Stem separation** (optional; skipped for uploads) → `stem_separator.remove_drums_and_vocals()`
6. **TTS upsample** 24 → 48 kHz via `audio_processor.upsample_audio(high_accuracy=True)`; then per-chunk humanize (Kokoro)
7. **Voice FX** → `build_voice_chain()` + `apply_fx()`; then `mixer.normalize_loudness()` to −18 LUFS
8. **Music FX** → `make_{engine}_music_chain()` + `make_vocal_pocket_chain()`
9. **Mix** → `mixer.mix()` (breathing sidechain duck — deep gradual S-curve, rises in pauses; exponential fades). Bed + duck levels auto-calibrated per session from measured stem LUFS (`mixer.calibrate_music_bed`; disable with `MOODSCAPE_ADAPTIVE_BED=0`)
10. **Master** → `make_master_chain()` (HPF → gentle bus comp → +1 dB air shelf; **no** limiter)
11. **QA** → `qa_monitor.run_qa_checks()`
12. **Export** → `mixer.export_audio()` (LUFS-normalize to −16 → `true_peak_limit()` to −1 dBTP → WAV/MP3)

Full breakdown with parameters: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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

The six that bite most often. Full list in [docs/GOTCHAS.md](docs/GOTCHAS.md).

- **MPS bus error on exit** → `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove.
- **ACE-Step timeout** → always pass `compile_model=True` to `initialize_service()`; first run has ~135s JIT overhead.
- **transformers pin** → `>=4.51.0,<4.58.0` for ACE-Step compatibility — do not upgrade.
- **Kokoro forced to CPU** → MPS causes deallocation bus errors. British voices (`bf_*`, `bm_*`) need `KPipeline(lang_code="b")`.
- **IndexTTS-2 NaN clamp** → BigVGANv2 may emit NaN on MPS. Use `torch.clamp(mel, -10, 10)`; force `use_fp16=False, use_deepspeed=False, use_cuda_kernel=False`.
- **No pedalboard `Limiter`** → pedalboard 0.9.23's `Limiter` inflates sub-threshold signals ~+4.75 dB and adds broadband "static". It was removed from all music + master chains. Peak control is `mixer.true_peak_limit()` at export (LUFS-normalize → true-peak limit to −1 dBTP).
- **Breathing duck** → `mixer.mix()` uses `apply_breathing_duck` (deep gradual S-curve, rises in pauses). Bed/duck levels are auto-calibrated per session (`calibrate_music_bed`, targets: bed 14.5 LU under voice in pauses, 30.5 LU under during speech); `MOODSCAPE_ADAPTIVE_BED=0` restores the fixed −16/−16 constants. The old multiband/`hold_ms` reactive ducker has been removed.
- **IndexTTS pacing needs Rubber Band** → without the `rubberband` CLI, chunks are returned **unstretched** (the librosa phase-vocoder fallback metallicises voice; re-enable only via `MOODSCAPE_INDEXTTS_PV_FALLBACK=1`).
- **ACE-Step >5 min defaults to loop mode** → one ~4-min piece looped via `fit_to_length` (long_form_mode="auto"). Pick "Evolve" in the UI for continuously generated (slower, more seam risk).

## Where to Look

| Need | Go to |
|------|-------|
| Full pipeline, FX params, QA thresholds, memory patterns | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Engine internals (Kokoro, F5, IndexTTS, ACE-Step, Lyria, Pedalboard) | [docs/model_implementation_guides/](docs/model_implementation_guides/) |
| Prompt writing per engine | [docs/prompting_guides/](docs/prompting_guides/) |
| Mix / post-processing details | [docs/optimization_and_processing/](docs/optimization_and_processing/) |
| Component & class map | [docs/COMPONENT_REGISTRY.md](docs/COMPONENT_REGISTRY.md) |
| File for a given task | [docs/TASK_ROUTING.md](docs/TASK_ROUTING.md) |
| All gotchas | [docs/GOTCHAS.md](docs/GOTCHAS.md) |
| Setup / checkpoints | [docs/setup_and_execution/](docs/setup_and_execution/) |
