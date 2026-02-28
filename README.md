# MoodScape — Guided Meditation Audio Generator

> Generate professional-quality guided meditation audio entirely on your local machine. Write a script, describe the music, and get a fully-mixed audio file in minutes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

MoodScape combines three AI models into a single pipeline:

1. **Kokoro TTS** synthesizes calm, natural narration from your script
2. **MusicGen (Meta)** generates custom ambient background music from a text prompt
3. **Pedalboard (Spotify)** applies studio-quality audio effects — compression, reverb, EQ, limiting

The final mix features **automatic ducking** (music volume lowers when the narrator speaks), linear fades, and LUFS normalization — all rendered locally with no API keys or subscriptions required.

---

## Features

| Feature | Detail |
|---------|--------|
| AI Narration | Kokoro TTS — 6 female voices, 2 male voices, adjustable speed |
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
  script_parser.py        ← Parses [pause:Xs] markers into segment list
  tts_engine.py           ← Kokoro TTS wrapper (24 kHz mono output)
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
[tts_engine]     →  voice audio (float32, 24 kHz) + voice activity mask
    │                 (Kokoro model unloaded, GPU memory freed)
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
- **CUDA-capable GPU** with at least **8 GB VRAM**
  - 8 GB: uses `musicgen-small` (300M params)
  - 16 GB+: uses `musicgen-melody` (1.5B params, better quality) — fallback is automatic
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
3. **Open "Voice Settings"** to select a voice and adjust speaking speed
4. **Open "Audio Settings"** to tune ducking, reverb, and fades
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
| Speaking Speed | 0.5 – 1.0 | 0.85 | Lower = slower, more meditative pace |
| Music Ducking (dB) | -4 to -15 | -8 | How much music drops when voice speaks |
| Voice Reverb | 0.0 – 0.5 | 0.15 | Wet level of the voice reverb effect |
| Fade In (sec) | 0 – 10 | 3 | Duration of the opening fade-in |
| Fade Out (sec) | 0 – 10 | 5 | Duration of the closing fade-out |

---

## GPU Memory Management

Both models (Kokoro TTS and MusicGen) are loaded **sequentially**, not simultaneously:

1. Kokoro is loaded → narration is synthesized → Kokoro is unloaded + GPU cache cleared
2. MusicGen is loaded → music is generated → MusicGen is unloaded + GPU cache cleared

This means even a GPU with 8 GB VRAM can run both models without out-of-memory errors.

---

## Tech Stack

| Library | Role |
|---------|------|
| [Kokoro TTS](https://github.com/hexgrad/kokoro) | Text-to-speech narration |
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
