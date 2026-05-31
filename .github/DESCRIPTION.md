# MoodScape Guided Meditations

AI-powered meditation audio generator that synthesizes guided voice, adaptive music, and professional audio processing into coherent, personalized meditation experiences.

## Features

**Three TTS Engines**
- **Kokoro**: Fast, expressive voice synthesis with multiple languages
- **F5-TTS**: Natural voice cloning with minimal reference audio
- **IndexTTS-2**: Zero-shot voice cloning with emotion control

**Two Music Generation Engines**
- **ACE-Step 1.5**: Controlled, mood-aware music composition via MESA framework
- **Lyria RealTime**: Real-time music generation (cloud API)

**Production-Grade Audio Processing**
- Multiband ducking and dynamic mixing
- Voice-guided reverb and FX chains
- Loudness normalization (−18 LUFS dialogue, −16 LUFS export)
- Optional stem separation via Demucs

**Flexible Meditation Scripts**
- Structured YAML-based meditation definitions
- Mood, pace, duration, and music style controls
- Long-form audio support (90+ minutes via seamless two-phase generation)
- Breath sound integration and stereo upmixing

## Hardware Target

Optimized for **Apple Silicon (M1 Max, 36 GB unified RAM)** with MPS acceleration and memory-conscious sequential loading of models.

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt
brew install espeak-ng  # Kokoro G2P dependency
python app.py          # Gradio UI at http://localhost:7860
```

Requires `.env` with `HF_TOKEN` and `GOOGLE_API_KEY`.

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline: script parsing → TTS synthesis → music generation → stem separation → FX processing → multiband mixing → QA validation → export.
