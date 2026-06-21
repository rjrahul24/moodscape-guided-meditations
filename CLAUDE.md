# MoodScape Guided Meditations

AI-guided meditation audio generator (Gradio UI). Two TTS engines (Kokoro, F5-TTS) and two music sources (Lyria RealTime or pre-existing background instrumentals). Target hardware: Apple Silicon M1 Max (36 GB unified RAM).

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
brew install espeak-ng                 # Kokoro G2P dependency
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
│   ├── kokoro_tts/  f5_tts/             # TTS engines (engine + preproc + postproc + voices)
│   ├── lyria/                             # Lyria RealTime music generation (Google API)
│   └── upload_music/                      # Background instrumental (engine + arrange/length-fit)
├── scripts/                          # generate.py · separate_worker.py · generate_breath_samples.py
├── tests/unit/  tests/integration/
├── assets/                           # tracked in git
│   ├── breath_sounds/                # [breath]/[inhale]/[exhale] samples
│   ├── impulse_responses/            # convolution reverb IRs
│   └── speakers/                     # F5-TTS voice pool
│       ├── reference_audio/*.wav     #   speaker reference clips
│       ├── reference_text/*.txt      #   transcripts (paired by slug)
│       └── voices.toml               #   F5 multi-phase definitions
├── models/                           # gitignored; all model weights
│   └── hf_cache/                     # project-local HF cache
└── docs/                             # see Where to Look below
```

## Pipeline Flow (`core/pipeline.py :: MeditationPipeline.generate()`)

1. **Parse script** → `{tts}/preprocessor.py :: prepare_segments()`
2. **TTS synth** → 24 kHz mono float32
3. **Unload TTS**, load music engine (sequential — 36 GB RAM constraint)
4. **Music gen** → 48 kHz mono float32 (Lyria or `upload_music` — decode + resample + loop/trim-fit the uploaded file to the same contract)
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
- **Kokoro forced to CPU** → MPS causes deallocation bus errors. British voices (`bf_*`, `bm_*`) need `KPipeline(lang_code="b")`.
- **No pedalboard `Limiter`** → pedalboard 0.9.23's `Limiter` inflates sub-threshold signals ~+4.75 dB and adds broadband "static". It was removed from all music + master chains. Peak control is `mixer.true_peak_limit()` at export (LUFS-normalize → true-peak limit to −1 dBTP).
- **Breathing duck** → `mixer.mix()` uses `apply_breathing_duck` (deep gradual S-curve, rises in pauses). Bed/duck levels are auto-calibrated per session (`calibrate_music_bed`, targets: bed 14.5 LU under voice in pauses, 30.5 LU under during speech); `MOODSCAPE_ADAPTIVE_BED=0` restores the fixed −16/−16 constants. The old multiband/`hold_ms` reactive ducker has been removed.
- **Intro fade-in** → default **1.5 s**, gentle exponential (rising steepness **2.0**, not 4.0). A 3 s steepness-4 fade read as "2–3 s of silence then music". `apply_fades` uses steepness 2.0 for the rising curve only (fade-out stays 4.0). Slider in `app.py`, default in `pipeline.generate`.

## Research Experiment Flags

From the 2026-06-13 research pass ([Research Execution Process Outline.md]). Two F5 fixes ship **ON** (kill-switch flags); four experiments ship **OFF** and have Gradio toggles for A/B listening tests.

| Flag (default) | Effect | Code |
|---|---|---|
| `MOODSCAPE_F5_REF_PAD` (**1**) | Appends ~1 s of −55 dBFS noise to reference audio so F5 leaks silence (not a stray syllable) on short phrases. `_SEC`/`_DBFS` tune it. | `core/f5_tts/engine.py::_condition_reference_audio` |
| `MOODSCAPE_F5_SHORT_PHRASE_PACING` (**1**) | Slows genuine fragments (≤ `_MAX_CHARS` non-space chars, default **12**) to `_SPEED` (0.5) in natural-rhythm mode only. **Char-based, not word-based** — a word count (≤6) caught normal short sentences and forced them so slow that F5 stretched + inserted mid-word gaps. | `core/f5_tts/engine.py::synthesize` |
| `MOODSCAPE_SPECTRAL_DUCK` (**0**) | Ducks only the 250–4000 Hz mid band (LR crossover, subtractive mid) instead of fullband — keeps bass warmth + air. `_DEPTH`/`_LO`/`_HI` tune it. | `mixer.apply_breathing_duck_multiband`, branch in `mix()` |
| `MOODSCAPE_SHARED_REVERB` (**0**) | Sends the bed through the voice's IR (HPF300/LPF6k) at `_SEND_DB` (−26) so both share one room. | `mixer.add_shared_reverb`, summed in `mix()` |
| `MOODSCAPE_F5_CFG`/`_SWAY`/`_NFE` (2.0/−1.0/32) | Override F5 latent params (research A/B: cfg ~1.2 warmer/less identical; sway/nfe best left alone). `cfg` also has the **"Voice Expressiveness" UI slider** (1.0–2.5). | `core/f5_tts/engine.py::synthesize` |
| `MOODSCAPE_F5_MICROPROSODY` (**0**) | Per-phrase WORLD pass, peak-preserving: pitch declination (`_DECLINE_TAIL_MS` 600 / `_DECLINE_CENTS` 120) + pitch-range widening (`_PITCH_SCALE` 1.15) + formant warmth (`_FORMANT_SHIFT` 0.98) + aperiodicity (`_AP_SCALE` 1.05) + phrase-final taper (`_TAPER_MS` 300 / `_TAPER_FLOOR` 0.6). Risky on clean Vocos. | `core/f5_tts/postprocessor.py::apply_microprosody` |
| `MOODSCAPE_F5_REF_PRESERVE_DYNAMICS` (**0**) | Skip the −20 dBFS RMS-normalisation of reference audio so its natural dynamics (expressive contour) survive. | `core/f5_tts/engine.py::_condition_reference_audio` |

Rejected research items (don't revisit): true-peak-limiter removal (false premise — `true_peak_limit` is transparent below ceiling), SRC reorder (already correct — soxr_vhq before FX; the 48 kHz path has no extra audio resampling beyond the limiter's oversampling), full LR crossover for "harshness" (the 14 kHz LPF is already a gentle 2nd-order min-phase filter), look-ahead ducker (the 600 ms predictive pre-descent already beats it), colon→comma / ALL-CAPS text fixes (already in the preprocessor), dithering relocation (there is no dithering). Parked for a later "Issue-1 polish" round: upload LPF→high-shelf, `true_peak_limit` gain-envelope refactor, master bus saturation.

## Where to Look

| Need | Go to |
|------|-------|
| Full pipeline, FX params, QA thresholds, memory patterns | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Engine internals (Kokoro, F5, Lyria, Pedalboard) | [docs/model_implementation_guides/](docs/model_implementation_guides/) |
| Prompt writing per engine | [docs/prompting_guides/](docs/prompting_guides/) |
| Mix / post-processing details | [docs/optimization_and_processing/](docs/optimization_and_processing/) |
| Component & class map | [docs/COMPONENT_REGISTRY.md](docs/COMPONENT_REGISTRY.md) |
| File for a given task | [docs/TASK_ROUTING.md](docs/TASK_ROUTING.md) |
| All gotchas | [docs/GOTCHAS.md](docs/GOTCHAS.md) |
| Setup / checkpoints | [docs/setup_and_execution/](docs/setup_and_execution/) |
