# MoodScape Guided Meditations

AI-guided meditation audio generator using Gradio, multiple TTS engines, and AI music generation. Target hardware: Apple Silicon M1 Max (36 GB unified RAM).

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # Gradio UI at http://localhost:7860
```

**System dependency:** `brew install espeak-ng` (required by Kokoro TTS)

**Environment variables:** `.env` file (gitignored) must contain `HF_TOKEN` and `GOOGLE_API_KEY`. App also sets `TOKENIZERS_PARALLELISM=false`, `AUDIOCRAFT_DISABLE_MPS_AUTOCAST=1`, and `PYTORCH_ENABLE_MPS_FALLBACK=1` at startup.

## Task Workflow

IMPORTANT: Follow these steps for every task.

1. **Read docs first** — Before touching any code, review relevant documentation in `docs/` to understand the current implementation context:
   - `docs/model_implementation_guides/` — Engine-specific details (ACE-Step, Kokoro, F5-TTS, MusicGen, Lyria, Pedalboard)
   - `docs/optimization_and_processing/` — Audio pipeline and post-processing
   - `docs/prompting_guides/` — Prompt engineering for each engine (Kokoro, F5-TTS, ACE-Step, MusicGen)
   - `docs/setup_and_execution/` — Environment and runtime setup
2. **Explore codebase** — Find the relevant code sections before making changes
3. **Implement changes**
4. **Update documentation** — Keep `docs/` in sync with any code changes. If a function signature, pipeline step, or engine behavior changes, update the corresponding doc file
5. **Update README.md** — Only if the change is significant (new feature, removed capability, changed setup steps)
6. **Run tests** — Unit tests for small changes, integration tests for large changes (see Build & Test below)
7. **Do NOT push to GitHub** unless explicitly asked
8. **Update this CLAUDE.md** if you discover new development patterns worth noting

## Build & Test

```bash
# Run all unit tests
.venv/bin/python -m pytest unit-tests/ -v

# Run a specific test file
.venv/bin/python -m pytest unit-tests/test_mixer.py -v

# Run a specific test
.venv/bin/python -m pytest unit-tests/test_meditation_mastering.py::test_mastering -v

# Integration tests (slower, use for large changes)
.venv/bin/python -m pytest integration-tests/ -v

# CLI batch generation
python scripts/generate.py <script_file> --voice <voice_name> --output <output.wav>
```

## Git Workflow

- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- **Commit directly to main** — no feature branches (solo developer)
- **Never push** unless the user explicitly asks

## Architecture

**Pipeline flow** (`core/pipeline.py`):
```
Script text → TTS (Kokoro/F5) → Music (MusicGen/ACE-Step/Lyria) → Audio FX (Pedalboard) → Mixer → WAV/MP3
```

- **Sequential engine loading**: Each engine is loaded, used, then unloaded to fit within 36 GB memory. Never load two engines simultaneously.
- **Audio contract**: All TTS engines output 24,000 Hz mono float32 (enforced by `core/speech_engine.py` ABC). Music is generated at native sample rate and resampled to 24 kHz in the pipeline.
- **Key modules**: `core/pipeline.py` (orchestration), `core/mixer.py` (ducking/fades/export), `core/audio_processor.py` (Pedalboard FX chains), `core/session_config.py` (immutable generation config), `core/qa_monitor.py` (output validation)

**TTS engines** (`core/kokoro_tts/`, `core/f5_tts/`):
- Kokoro: Forced to CPU on Apple Silicon (MPS causes bus errors). Uses `trf=True` for transformer G2P.
- F5-TTS: Zero-shot voice cloning with Silero VAD (15% gain floor). Voice assets in `core/f5_tts/assets/`. WPM-based pacing via `fix_duration` (default 110 WPM). 300-char chunk limit. Multi-phase voices via `voices.toml`.

**Music engines** (`core/music_engine.py`, `core/acestep_engine.py`, `core/lyria/`):
- ACE-Step: Must use `compile_model=True` to avoid timeouts. Uses MLX backend on Apple Silicon.
- MusicGen: CPU only, `musicgen-small` (300M params) primary. MPS disabled — EnCodec ELU corrupts audio on Apple Silicon. Small model is ~3.5× faster than stereo-medium on CPU (~13 min vs ~45 min for 5-min track); quality difference is imperceptible at −17 dB background level.

## Code Conventions

- Classes: `PascalCase` — Functions/methods: `snake_case` — Constants: `UPPER_SNAKE_CASE`
- Private functions: `_leading_underscore`
- Imports: `from core.module import Class` (relative to project root)
- Each subpackage has `__init__.py` with explicit public API exports
- Abstract base class pattern: `SpeechEngine` ABC → concrete engine implementations

## Common Gotchas

- **MPS bus error on exit**: `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove
- **ACE-Step timeout**: Always pass `compile_model=True` to `initialize_service()` — without it, generation takes ~9s/step and times out
- **transformers version**: Pinned to `>=4.51.0,<4.58.0` for ACE-Step compatibility — do not upgrade
- **Kokoro on CPU**: Intentionally forced to CPU — MPS causes deallocation bus errors
- **AUDIOCRAFT_DISABLE_MPS_AUTOCAST=1**: Required env var to prevent MPS float16 autocast crashes
