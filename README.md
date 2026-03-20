# MoodScape — Guided Meditation Audio Generator

> Generate professional-quality guided meditation audio entirely on your local machine. Write a script, choose your AI engines, and get a fully-mixed audio file — narration, music, reverb, ducking, and all.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20CUDA-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

MoodScape is a locally-run AI pipeline that synthesizes professional guided meditation audio entirely on-device. It combines multiple AI models for narration, music generation, source separation, and audio mastering — all orchestrated through a Gradio web UI or a CLI runner.

**AI engines included:**
- **Kokoro TTS** — fast, lightweight narration (82M params, preset voice blends, CPU)
- **F5-TTS** — zero-shot voice cloning from any 10s reference recording (MPS)
- **HeartMuLa** — high-quality text-to-music with structural section control (3B params, MLX/MPS, 44.1 kHz native)
- **ACE-Step 1.5** — high-fidelity text-to-music with LM planning (MLX, 48 kHz native)
- **Lyria RealTime** — Google DeepMind cloud music generation (48 kHz, no local GPU needed)

For engine-specific deep dives, see the [docs/](#documentation-guide) directory.

---

## Pipeline Architecture

```
Script text
    │
    ▼
[Preprocessor]    kokoro_tts/preprocessor.py  or  f5_tts/preprocessor.py
    │             → Parses [pause:Xs], [breath], [voice:phase] tags into segments
    ▼
[TTS Engine]      core/kokoro_tts/  (CPU, 24 kHz mono float32)
                  core/f5_tts/      (MPS, 24 kHz mono float32)
    │             → TTS unloaded; memory freed before music loads
    ▼
[Music Engine]    core/heart_mula/engine.py (HeartMuLa, MLX/MPS, 44.1 kHz)
                  core/acestep_engine.py    (ACE-Step 1.5, MLX, 48 kHz)
                  core/lyria/engine.py      (Lyria RealTime, cloud, 48 kHz)
    │             → Music engine unloaded; memory freed
    ▼
[Stem Separation] core/stem_separator.py   (HT Demucs, optional)
    │             → Removes drums/vocals that leaked past prompting
    ▼
[Pedalboard FX]   core/audio_processor.py
    │             → Separate FX chains for voice, music, and master bus
    ▼
[Mixer]           core/mixer.py
    │             → Lookahead sidechain ducking + crossfades + LUFS normalization
    ▼
[QA Monitor]      core/qa_monitor.py
    │             → 8 automated checks (clipping, LUFS, spectral balance, etc.)
    ▼
WAV / MP3  (44.1 kHz or 48 kHz)
```

### Module Map

| File | Role |
|------|------|
| `app.py` | Gradio UI entry point |
| `core/pipeline.py` | End-to-end orchestration |
| `core/kokoro_tts/` | Kokoro TTS — preprocessor, engine, postprocessor, voice manager |
| `core/f5_tts/` | F5-TTS — preprocessor, engine, postprocessor, voice registry |
| `core/heart_mula/` | HeartMuLa wrapper (3B LM + HeartCodec, segment-and-crossfade) |
| `core/acestep_engine.py` | ACE-Step 1.5 wrapper (MLX backend, 48 kHz) |
| `core/lyria/` | Lyria RealTime API — engine, weighted prompt parser |
| `core/stem_separator.py` | HT Demucs source separation (4-source model) |
| `core/audio_processor.py` | Pedalboard FX chains (voice / music / master) |
| `core/mixer.py` | Ducking, overlay, fades, normalization, export |
| `core/qa_monitor.py` | Output quality assurance and composite scoring |
| `core/session_config.py` | Immutable reproducible generation config |
| `core/speech_engine.py` | Abstract TTS interface (24 kHz mono float32 contract) |
| `scripts/generate.py` | CLI batch runner |

---

## Prerequisites

### Hardware

| Platform | Supported Engines | Notes |
|----------|-------------------|-------|
| Apple Silicon (M1/M2/M3) | All 5 engines | Recommended — MLX backend for ACE-Step, MLX/MPS for HeartMuLa, MPS for F5-TTS |
| CUDA GPU (8+ GB VRAM) | Kokoro, F5-TTS, HeartMuLa, ACE-Step | ACE-Step uses PyTorch MPS fallback on CUDA |
| CPU-only | Kokoro | HeartMuLa, ACE-Step, and Lyria unavailable; F5-TTS is very slow |

### Python & System Dependency

- **Python 3.10+** required
- **espeak-ng** — system package required by Kokoro's G2P fallback

```bash
# macOS
brew install espeak-ng

# Ubuntu / Debian
sudo apt-get install espeak-ng

# Fedora / RHEL
sudo dnf install espeak-ng
```

### ACE-Step Model Weights

ACE-Step requires local model weights in `ACE-Step-1.5/checkpoints/`. The weights are not included in this repository. See [`docs/model_implementation_guides/ace-step.md`](docs/model_implementation_guides/ace-step.md) for download instructions. Without the weights, selecting ACE-Step in the UI will fail silently.

### Environment Variables

Create a `.env` file at the project root (already in `.gitignore`):

```bash
HF_TOKEN=hf_...           # Required — downloads Kokoro, F5-TTS, Demucs models from Hugging Face
GOOGLE_API_KEY=AIza...    # Optional — only needed for Lyria RealTime engine
```

Get `HF_TOKEN` at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Get `GOOGLE_API_KEY` at [aistudio.google.com](https://aistudio.google.com) (for Lyria only).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rjrahul24/moodscape-guided-meditations.git
cd moodscape-guided-meditations

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Create the .env file with your tokens
echo "HF_TOKEN=your_token_here" > .env

# 4. Install Python dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `ace-step` installed directly from GitHub (`git+https://...`), which takes longer than a standard pip install. This is expected.

> **Note:** `transformers` is pinned to `>=4.51.0,<4.58.0` for ACE-Step compatibility. Do not upgrade past 4.57.x.

---

## Usage — Web UI

```bash
python app.py
# Opens at http://localhost:7860
```

### Step-by-step

1. **Choose Generation Mode** — `Instrumental + Vocal` (default), `Vocals Only`, or `Instrumental Only`
2. **Write your meditation script** in the left panel — use [pause tags](#script-format) for timed silences
3. **Choose TTS Engine** — `Kokoro` for preset voices, `F5-TTS` for zero-shot voice cloning
4. **Select a Voice** — Kokoro preset from the dropdown, or F5-TTS voice scanned from `core/f5_tts/assets/`
5. **Choose Music Engine** — `ACE-Step 1.5`, `HeartMuLa`, or `Lyria RealTime`
6. **Write a Music Prompt** — describe the background music style (see [Music Engines](#music-engines) for prompt tips per engine)
7. **Configure ACE-Step or Lyria settings** if selected (BPM, key, density, quality mode)
8. **Expand Audio Settings** to tune ducking, reverb, fade durations, reverb IR, and stem export
9. Click **Generate Meditation** and watch the progress bar
10. **Preview** in the browser player, then **download** the file

---

## Usage — CLI

```bash
python scripts/generate.py my_script.txt \
  --voice golden_hour \
  --output session.wav \
  --music-prompt "warm analog pads, sine drones, beatless" \
  --speed 0.78 \
  --format wav \
  --seed 42 \
  --stems \
  --upsample
```

| Flag | Default | Description |
|------|---------|-------------|
| `script_file` | *(required)* | Path to meditation script `.txt` file |
| `--voice` | `golden_hour` | Kokoro voice ID or preset name |
| `--music-prompt` | `"warm ambient pads, no drums"` | Music description |
| `--speed` | `0.78` | Speaking speed (0.5–1.0) |
| `--output` | `meditation.wav` | Output file path |
| `--format` | `wav` | `wav` or `mp3` |
| `--seed` | `0` (auto) | Integer seed for reproducible output; 0 = random |
| `--stems` | off | Save separate `voice.wav` and `music.wav` alongside the mix |
| `--upsample` | off | Export at 48 kHz instead of 44.1 kHz |

> **Note:** The CLI currently uses Kokoro TTS and HeartMuLa (default). F5-TTS, ACE-Step, and Lyria are available via the web UI.

---

## Script Format

### Pause Tags

| Syntax | Effect |
|--------|--------|
| `[pause:3s]` | 3-second silence |
| `[pause:2.5s]` | 2.5-second silence |
| `\n\n` (blank line) | ~1.5-second automatic paragraph pause |

### Breath Tags

Insert realistic breath sounds at any point in the script:

| Tag | Effect |
|-----|--------|
| `[breath]` | Neutral breath (1.2s) |
| `[inhale]` | Inhale breath (1.5s) |
| `[exhale]` | Exhale breath (1.8s) |

### F5-TTS Voice Phase Tags

For F5-TTS voices that define multiple phases in `voices.toml`, switch the active reference mid-script:

```
[voice:opening]
Welcome to this meditation. [pause:3s]

[voice:body]
Allow your breath to settle... [pause:5s]

[voice:closing]
Gently return your awareness to the room.
```

If no phase tag is present, the default phase is used throughout.

### Example Script

```
Welcome to this meditation. [pause:3s]

Take a slow, deep breath in... [inhale] [pause:4s] and release. [exhale] [pause:6s]

Notice the stillness around you. [pause:10s]

[pause:3s] When you are ready, gently open your eyes.
```

---

## TTS Engines & Voices

### Kokoro TTS

Ultra-fast narration synthesis (82M params, StyleTTS2 + ISTFTNet). Runs on CPU only on Apple Silicon (MPS causes deallocation bus errors — this is intentional).

| Preset | Voice ID | Character |
|--------|----------|-----------|
| Balanced Calm *(default)* | `balanced_calm` | Natural blend — general-purpose |
| Deep Rest | `deep_rest` | Intimate, breathy |
| Soft Whisper | `soft_whisper` | ASMR relaxation |
| Golden Hour | `golden_hour` | Warm, airy |
| Earth Root | `earth_root` | Grounding, centered |
| Heart | `af_heart` | US Female — warm |
| Nicole | `af_nicole` | US Female — calm, ASMR |
| Emma | `bf_emma` | UK Female — wise |
| Adam | `am_adam` | US Male — grounding |
| George | `bm_george` | UK Male — warm |

Speed range: 0.65–1.0. UI default: **0.85** (Kokoro). Meditation sweet spot: 0.80–0.88.

British voices (`bf_*`, `bm_*`) use a separate language pipeline internally and require no extra setup.

See [`docs/model_implementation_guides/kokoro_tts.md`](docs/model_implementation_guides/kokoro_tts.md) for voice blending, G2P details, and script writing guide.

### F5-TTS (Zero-Shot Voice Cloning)

Clones any voice from a 10–12 second `.wav` recording. No fine-tuning required.

**To add a custom voice:**
1. Record a clean 10–12s `.wav` at 24 kHz mono (or any rate — it will be resampled)
2. Write a verbatim transcript of exactly what was said
3. Place the `.wav` in `core/f5_tts/assets/reference_audio/`
4. Place the `.txt` transcript in `core/f5_tts/assets/reference_transcript/`
5. Restart the app — the voice appears in the dropdown automatically

Multi-phase voices (opening / body / closing) are configured in `core/f5_tts/assets/voices.toml`. Speed default: **0.80** for F5-TTS.

See [`docs/model_implementation_guides/f5_tts.md`](docs/model_implementation_guides/f5_tts.md) for reference audio requirements, multi-phase configuration, and synthesis parameters.

---

## Music Engines

### HeartMuLa

| Property | Value |
|----------|-------|
| Architecture | HeartMuLa LM (3B params) + HeartCodec (12.5 Hz neural codec) |
| Backend | MLX primary (heartlib-mlx), PyTorch MPS fallback (heartlib) |
| License | Apache 2.0 |
| Native sample rate | 44.1 kHz stereo → downmixed to mono |
| Long-form strategy | Segment-and-crossfade (240s segments + 4s cosine crossfades) |
| Dtype | LM: bf16, Codec: fp32 (never bf16 for codec) |

Prompts use comma-separated style tags (not sentences). Structural lyrics with `[intro]`, `[verse]`, `[bridge]`, `[outro]` markers guide tonal arc. Tags are enhanced with meditation base tags automatically.

See [`docs/prompting_guides/heartmula_instructions.md`](docs/prompting_guides/heartmula_instructions.md) for prompt engineering and [`docs/model_implementation_guides/heartmula.md`](docs/model_implementation_guides/heartmula.md) for implementation details.

---

### ACE-Step 1.5

| Property | Value |
|----------|-------|
| Architecture | LM planner (Qwen3 4B) + Diffusion Transformer |
| Backend | MLX (Apple Silicon Metal native) |
| Native sample rate | 48 kHz stereo |
| Long-form strategy | Three-phase pipeline: genesis → continuation → boundary smoothing |
| Quality modes | Draft (Turbo / 8-step) vs Studio (SFT / 50-step) |
| BPM control | 40–120 (default 50) |
| Key control | Auto or specific key |

> **First-run JIT compilation**: ACE-Step uses `compile_model=True` to prevent generation timeouts. The first generation after app startup takes ~135 extra seconds for MLX JIT compilation. Subsequent runs are ~4× faster. This is expected — do not cancel.

> **Local model weights required**: Weights must be present at `ACE-Step-1.5/checkpoints/` before selecting this engine. See [`docs/model_implementation_guides/ace-step.md`](docs/model_implementation_guides/ace-step.md).

Prompts use the **MESA framework** (Mood + Elements + Structure + Application). Structural section tags (`[Intro]`, `[Verse]`, `[Bridge]`, `[Outro]`) are auto-inserted based on duration.

See [`docs/prompting_guides/ace_step_instructions.md`](docs/prompting_guides/ace_step_instructions.md) for the MESA framework and prompt examples.

---

### Lyria RealTime

| Property | Value |
|----------|-------|
| Provider | Google DeepMind (cloud API) |
| Requires | `GOOGLE_API_KEY` in `.env` |
| Native sample rate | 48 kHz stereo |
| Session cap | 10 minutes per connection (auto-split with crossfade) |
| Local GPU | None — zero VRAM consumption |

Weighted prompt syntax: `"Hang Drum: 1.5, Piano: 0.8, Ambient Pads: 1.0"` — weights scale relative influence. Standard text prompts also work.

Controls: BPM (40–140, default 70), Density (0–1, default 0.2), Brightness (0–1, default 0.3).

> Audio generated via Lyria contains an embedded SynthID watermark as required by Google's terms of service. Do not time-stretch or pitch-shift Lyria output.

See [`docs/model_implementation_guides/lyria.md`](docs/model_implementation_guides/lyria.md).

---

## Settings Reference

### Generation Settings

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| Generation Mode | Instrumental + Vocal / Vocals Only / Instrumental Only | Instrumental + Vocal | What the pipeline produces |
| TTS Engine | Kokoro / F5-TTS | Kokoro | Voice synthesis engine |
| Music Engine | ACE-Step 1.5 / HeartMuLa / Lyria RealTime | ACE-Step 1.5 | Background music engine |
| Output Format | WAV / MP3 | WAV | WAV = lossless; MP3 = compressed |

### Voice Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Speaking Speed | 0.65–1.0 | 0.85 (Kokoro) / 0.80 (F5) | Lower = slower, more meditative pace |
| Voice Reverb | 0.0–0.5 | 0.15 | Wet level of the voice reverb effect |
| Reverb IR | warm_studio / wooden_hall / stone_chapel | warm_studio | Room character — intimate / natural / expansive |

### Music Settings

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Music Duration | 1–30 min | 3 min | Used for Instrumental Only mode |
| Music Ducking | -30 to -5 dB | -20 dB | How far music drops under speech |
| ACE-Step Quality | Draft (Turbo/8-step) / Studio (SFT/50-step) | Studio | Speed vs. fidelity |
| ACE-Step BPM | 40–120 | 50 | Target tempo |
| ACE-Step Key | Auto + all keys | Auto | Target musical key |
| Lyria BPM | 40–140 | 70 | Target tempo |
| Lyria Density | 0.0–1.0 | 0.2 | Textural density (sparse → lush) |
| Lyria Brightness | 0.0–1.0 | 0.3 | Harmonic brightness (warm → bright) |

### Output Settings

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| Fade In | 0–10 sec | 3 sec | Opening fade duration |
| Fade Out | 0–10 sec | 5 sec | Closing fade duration |
| Upsample to 48 kHz | checkbox | off | Export at 48 kHz (auto-enabled for ACE-Step / Lyria) |
| Export Stems | checkbox | off | Save `voice.wav` + `music.wav` alongside the mix |
| Stem Separation | checkbox | on | Run HT Demucs to strip drums/vocals from generated music |
| Seed | integer | 0 (auto) | Deterministic seed — same seed + same inputs = same output |

---

## Audio Processing

A brief summary of the quality pipeline under the hood:

- **Lookahead sidechain ducking** — Music begins fading 75ms before voice onset; recovers over a 500ms release envelope. Broadcast-grade — never reactive.
- **Vocal pocket EQ** — Music spectrum is carved at 300 Hz, 1 kHz, and 3 kHz to preserve voice intelligibility without reducing overall music volume.
- **Convolution reverb** — Three impulse responses: Warm Studio (intimate, short decay), Wooden Hall (natural warmth, medium space), Stone Chapel (ethereal, long decay).
- **HT Demucs separation** — 4-source model (drums, bass, vocals, other). Keeps bass + other; discards drums + vocals. Corrects percussion and vocal artifacts that leaked past prompting.
- **LUFS normalization** — ITU-R BS.1770-4 standard. Target: -19 LUFS (meditation broadcast reference).
- **QA Monitor** — 8 checks run before export: clipping ratio, LUFS accuracy, spectral warmth vs. presence balance, silence ratio, spectral rolloff, onset strength (transient detection), spectral flatness (noise detection), and long-silence gaps.

See [`docs/optimization_and_processing/audio_processing.md`](docs/optimization_and_processing/audio_processing.md) and [`docs/optimization_and_processing/post-processing-pipeline.md`](docs/optimization_and_processing/post-processing-pipeline.md).

---

## Documentation Guide

All engine-specific guides, prompt engineering references, and processing details live in `docs/`:

```
docs/
├── model_implementation_guides/
│   ├── kokoro_tts.md           ← Kokoro internals, voice blending, preprocessing, FX chain
│   ├── f5_tts.md               ← F5-TTS zero-shot cloning, multi-phase voices, chained reference
│   ├── heartmula.md            ← HeartMuLa 3B LM + HeartCodec, MLX/MPS, lazy loading, segment-and-crossfade
│   ├── ace-step.md             ← ACE-Step architecture, MLX backend, MESA framework, weight download
│   ├── lyria.md                ← Lyria RealTime API, weighted prompts, session limits, SynthID
│   └── pedalboard.md           ← Pedalboard FX chain design, all plugin parameters
├── prompting_guides/
│   ├── vocal_kokoro_instructions.md   ← How to write scripts for Kokoro TTS
│   ├── vocal_f5_instructions.md       ← How to write scripts for F5-TTS + phase guide
│   ├── heartmula_instructions.md      ← HeartMuLa tag-based prompting (meditation presets + examples)
│   └── ace_step_instructions.md       ← ACE-Step MESA framework + story mode prompting
├── optimization_and_processing/
│   ├── audio_processing.md            ← Ducking, FX chains, sample rate strategy, stereo-to-mono
│   └── post-processing-pipeline.md    ← Export, LUFS, master chain, streaming export
└── setup_and_execution/
    ├── app-running-instructions.md    ← Environment setup walkthrough
    └── local-target-env.md            ← Apple Silicon hardware specs, ML stack, memory management
```

---

## Testing

```bash
# Run all unit tests (fast, no models loaded)
.venv/bin/python -m pytest unit-tests/ -v

# Run a single test file
.venv/bin/python -m pytest unit-tests/test_mixer.py -v

# Run a single test function
.venv/bin/python -m pytest unit-tests/test_meditation_mastering.py::test_mastering -v

# Integration tests (loads models — much slower)
.venv/bin/python -m pytest integration-tests/ -v
```

Unit test coverage: mixer, audio_processor, qa_monitor, stem_separator, TTS engines (Kokoro + F5), script preprocessor, F5 pacing/chained-reference/params, ACE-Step long-form. Integration tests cover full pipeline mode combinations and stress scenarios.

---

## Troubleshooting

### Installation

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'kokoro'` | `pip install kokoro>=0.9.4` — package name is `kokoro`, not `kokoro-tts` |
| `espeak-ng not found` | `brew install espeak-ng` (macOS) or `sudo apt install espeak-ng` (Linux) |
| `transformers` version conflict | `pip install "transformers>=4.51.0,<4.58.0"` — ACE-Step is incompatible with ≥4.58.0 |
| `ace-step` build fails | Requires `git` in PATH; the `git+https://...` entry in `requirements.txt` clones from GitHub |

### Generation Errors

| Problem | Solution |
|---------|----------|
| Lyria error: `GOOGLE_API_KEY not set` | Add `GOOGLE_API_KEY=...` to `.env` and restart the app |
| ACE-Step generation times out | First run after app start takes ~135s for JIT compilation — wait it out. Subsequent runs are fast. |
| ACE-Step: `FileNotFoundError` for checkpoints | Model weights missing from `ACE-Step-1.5/checkpoints/` — see `docs/model_implementation_guides/ace-step.md` |
| F5-TTS: no voices in dropdown | Add `.wav` + `.txt` pairs to `core/f5_tts/assets/reference_audio/` and `reference_transcript/` |
| Output is completely silent | Check Generation Mode — "Vocals Only" produces no music; "Instrumental Only" produces no voice |

### macOS / Apple Silicon

| Problem | Solution |
|---------|----------|
| Bus error on app exit | Expected behavior — suppressed by `atexit.register(os._exit(0))` in `app.py`. Do not remove this line. |
| HeartMuLa `FileNotFoundError` | Download weights: `huggingface-cli download HeartMuLa/HeartMuLa-RL-oss-3B-20260123 --local-dir ./ckpt/HeartMuLa-oss-3B` and `huggingface-cli download HeartMuLa/HeartCodec-oss-20260123 --local-dir ./ckpt/HeartCodec-oss` |
| HeartMuLa generation very slow | Expected on MPS (~5-10x slower than CUDA). Use MLX backend (`pip install git+https://github.com/Acelogic/heartlib-mlx.git`) or ACE-Step for faster results. |
| HeartMuLa audio quality degradation | Verify `codec_dtype="fp32"` in `core/heart_mula/engine.py` — never use `bf16` for HeartCodec |
| Kokoro is running on CPU, not MPS | Intentional — Kokoro on MPS causes deallocation bus errors. CPU is stable and fast enough. |
| High peak RAM usage (~12–16 GB) | Expected — TTS and music models are large. Sequential loading (TTS → unload → music) keeps peak usage manageable. |

### Audio Quality

| Problem | Solution |
|---------|----------|
| Music contains drums or percussion | Enable **Stem Separation** — HT Demucs will strip drums from the generated music |
| Music is too quiet under narration | Increase Ducking value (e.g., from -20 dB to -12 dB) — less negative = louder music during speech |
| Voice reverb is too heavy | Reduce Voice Reverb slider; switch from `stone_chapel` to `warm_studio` IR |
| ACE-Step music sounds harsh or thin | Use **Studio (SFT/50-step)** mode instead of Draft — 50 diffusion steps yields significantly better quality |
| Narration sounds robotic or rushed | Reduce Speaking Speed to 0.80–0.85; use a vocal blend preset instead of a raw voice ID |

---

## Memory Management

TTS and music engines are loaded **sequentially**, never simultaneously:

1. TTS engine loads → narration is synthesized → TTS unloads (memory freed)
2. Music engine loads → music is generated → music engine unloads (memory freed)

This allows the 36 GB M1 Max to run all engines comfortably at peak usage of ~12–16 GB. **Lyria RealTime** is an exception — it runs in Google's cloud and consumes zero local GPU memory.

---

## Tech Stack

| Library | Role |
|---------|------|
| [Kokoro](https://github.com/hexgrad/kokoro) | TTS narration (82M params, StyleTTS2, CPU) |
| [F5-TTS](https://github.com/SWivid/F5-TTS) | Zero-shot voice cloning (Vocos vocoder, MPS) |
| [HeartMuLa / heartlib](https://github.com/HeartMuLa/heartlib) | AI music generation (3B LM + HeartCodec, MLX/MPS, 44.1 kHz) |
| [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) | Music generation (LM + DiT, MLX, 48 kHz) |
| [Google Lyria RealTime](https://deepmind.google/technologies/lyria/) | Cloud music generation (48 kHz, no local GPU) |
| [HT Demucs](https://github.com/facebookresearch/demucs) | AI source separation (4-source model) |
| [Pedalboard](https://github.com/spotify/pedalboard) | Audio FX — EQ, compression, convolution reverb, limiting |
| [Gradio](https://gradio.app) | Web UI |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS normalization (ITU-R BS.1770-4) |
| [MLX / mlx-lm](https://github.com/ml-explore/mlx) | Apple Silicon native ML acceleration |
| [silero-vad](https://github.com/snakers4/silero-vad) | Voice activity detection for F5-TTS chunking |
| [torchaudio](https://pytorch.org/audio) | Audio resampling |
| [soundfile](https://python-soundfile.readthedocs.io) | WAV I/O |
| [scipy](https://scipy.org) | Spectral analysis, Butterworth filters |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` file loading |

---

## License

MIT

Audio generated using Lyria RealTime contains an embedded SynthID watermark as required by Google's terms of service.
