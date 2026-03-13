# MoodScape — Guided Meditation Audio Generator

> Generate professional-quality guided meditation audio entirely on your local machine. Write a script, describe the music, and get a fully-mixed audio file in minutes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

MoodScape combines AI models into a single pipeline:

1. **Kokoro TTS** synthesizes calm, natural narration from your script
2. **MusicGen (Meta)** generates custom ambient background music from a text prompt
3. **Pedalboard (Spotify)** applies studio-quality audio effects — compression, reverb, EQ, limiting

The final mix features **automatic ducking** (music volume lowers when the narrator speaks), linear fades, and LUFS normalization — all rendered locally with no API keys or subscriptions required.

---

## Speech Engine

MoodScape uses **Kokoro TTS** for narration synthesis:

- Ultra-fast, lightweight (82M parameters)
- Named voice presets (Heart, Sky, Nova, Nicole, etc.) with voice blending
- Numeric speed control (0.5–1.0)
- Outputs at 24,000 Hz mono, integrating seamlessly with MusicGen and the Pedalboard FX chain

---

## Features

| Feature | Detail |
|---------|--------|
| AI Narration | Kokoro TTS — fast, lightweight, preset voices |
| AI Music | Meta MusicGen — generates from text prompt, fills exact duration needed |
| Auto Ducking | Music automatically lowers when narrator speaks; smoothly recovers in silence |
| Voice FX | Warmth EQ, gentle compression, configurable reverb |
| Music FX | Low-end warmth, high-frequency softening, limiting |
| Master Chain | Final limiter + LUFS normalization to -16 LUFS (broadcast standard) |
| Fade Control | Configurable fade-in and fade-out on the final mix |
| Output Formats | WAV (lossless) or MP3 (compressed) |
| Web UI | Gradio interface — no code needed to operate |
| Progress Bar | Real-time status for each pipeline stage |

---

## Architecture

```
app.py                    ← Gradio UI entry point
core/
  speech_engine.py        ← Abstract TTS engine interface (24 kHz mono contract)
  kokoro_tts/             ← Kokoro TTS wrapper (82M params, preset voices)
  script_parser.py        ← Parses [pause:Xs] markers into segment list
  music_engine.py         ← MusicGen wrapper (32 kHz → resampled to 24 kHz)
  audio_processor.py      ← Pedalboard FX chains (voice / music / master)
  mixer.py                ← Ducking, overlay, fades, normalization, export
  pipeline.py             ← Orchestrates full pipeline with progress callbacks
```

### Pipeline Flow

```
Script text
    │
    ▼
[script_parser]  →  list of speech/pause segments
    │
    ▼
[TTS engine]     →  voice audio (float32, 24 kHz) + voice activity mask
    │                 (TTS model unloaded, GPU memory freed)
    ▼
[music_engine]   →  music audio (float32, 24 kHz, auto-looped to length)
    │                 (MusicGen unloaded, GPU memory freed)
    ▼
[audio_processor]→  FX applied to voice + music separately
    │
    ▼
[mixer]          →  ducked music + voice summed, fades applied, normalized
    │
    ▼
WAV / MP3 file
```

---

## Prerequisites

- **Python 3.10+**
- **Apple Silicon Mac** (M1/M2/M3) with MPS backend, or **CUDA-capable GPU** with at least 8 GB VRAM
- **espeak-ng** — system-level dependency required by Kokoro TTS

### Install espeak-ng

```bash
# macOS
brew install espeak-ng

# Ubuntu / Debian
sudo apt-get install espeak-ng

# Fedora / RHEL
sudo dnf install espeak-ng

# Windows (WSL recommended, or download from espeak-ng releases)
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/rjrahul24/moodscape-guided-meditations.git
cd moodscape-guided-meditations

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Note:** `audiocraft` (AudioCraft / MusicGen) requires PyTorch 2.1+. If you have an existing PyTorch installation, verify compatibility before installing.

---

## Usage

```bash
python app.py
```

The Gradio UI opens automatically at **http://localhost:7860**.

### Step-by-step

1. **Write your script** in the left panel (or edit the pre-filled example)
2. **Describe the music** style in the Music Prompt box
3. **Configure voice** — choose a Kokoro preset from the Voice dropdown
4. **Open "Audio Settings"** to tune speed, ducking, reverb, and fades
5. **Choose output format** — WAV for lossless, MP3 for sharing
6. Click **Generate Meditation** and wait for the progress bar to complete
7. **Preview** the audio in the browser, then **download** it

---

## Script Format

Use `[pause:Xs]` markers for timed silences anywhere in your script. `X` can be an integer or a decimal number.

```
Welcome to this meditation. [pause:3s]

Take a slow, deep breath in... [pause:4.5s] and release. [pause:6s]

Notice the stillness around you. [pause:10s]

When you are ready, gently open your eyes. [pause:3s]
```

### Pause rules

| Syntax | Effect |
|--------|--------|
| `[pause:3s]` | 3-second silence |
| `[pause:2.5s]` | 2.5-second silence |
| `\n\n` (blank line) | Automatic 1.5-second pause between paragraphs |

---

## Available Voices

### Kokoro TTS Voices

| Display Name | Voice ID | Gender |
|--------------|----------|--------|
| Heart (default) | `af_heart` | Female — warm, calm |
| Bella | `af_bella` | Female |
| Nicole | `af_nicole` | Female |
| Sarah | `af_sarah` | Female |
| Sky | `af_sky` | Female |
| Nova | `af_nova` | Female |
| Adam | `am_adam` | Male |
| Michael | `am_michael` | Male |

---

## Settings Reference

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Speaking Speed | 0.5 – 1.0 | 0.80 | Lower = slower, more meditative pace |
| Music Ducking (dB) | -2 to -20 | -4 | How much music drops when voice speaks |
| Voice Reverb | 0.0 – 0.5 | 0.15 | Wet level of the voice reverb effect |
| Fade In (sec) | 0 – 10 | 3 | Duration of the opening fade-in |
| Fade Out (sec) | 0 – 10 | 5 | Duration of the closing fade-out |

---

## GPU Memory Management

TTS and MusicGen models are loaded **sequentially**, not simultaneously:

1. TTS engine is loaded → narration is synthesized → TTS is unloaded + memory freed
2. MusicGen is loaded → music is generated → MusicGen is unloaded + memory freed

This means even hardware with limited VRAM can run both models without out-of-memory errors.

---

## Tech Stack

| Library | Role |
|---------|------|
| [Kokoro TTS](https://github.com/hexgrad/kokoro) | Text-to-speech narration (fast, preset voices) |
| [AudioCraft / MusicGen](https://github.com/facebookresearch/audiocraft) | AI music generation |
| [Pedalboard](https://github.com/spotify/pedalboard) | Audio effects (EQ, compression, reverb, limiting) |
| [Gradio](https://gradio.app) | Web UI |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS loudness normalization |
| [soundfile](https://python-soundfile.readthedocs.io) | WAV I/O |
| [torchaudio](https://pytorch.org/audio) | Audio resampling |
| [scipy](https://scipy.org) | Butterworth filter for ducking smoothing |

---

## License

MIT
