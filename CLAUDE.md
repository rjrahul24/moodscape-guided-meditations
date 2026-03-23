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

1. **Read docs first** — Before touching any code, review relevant documentation in `docs/` to understand the current implementation context:
   - `docs/model_implementation_guides/` — Engine-specific details (ACE-Step, Kokoro, F5-TTS, HeartMuLa, Lyria, Pedalboard)
   - `docs/optimization_and_processing/` — Audio pipeline and post-processing
   - `docs/prompting_guides/` — Prompt engineering for each engine (Kokoro, F5-TTS, ACE-Step, HeartMuLa)
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
Script text → TTS (Kokoro/F5) → Music (HeartMuLa/ACE-Step/Lyria) → Audio FX (Pedalboard) → Mixer → WAV/MP3
```

- **Sequential engine loading**: Each engine is loaded, used, then unloaded to fit within 36 GB memory. Never load two engines simultaneously.
- **Audio contract**: All TTS engines output 24,000 Hz mono float32 (enforced by `core/speech_engine.py` ABC). All music engines output 48 kHz natively. TTS audio is upsampled to 48 kHz for mixing.
- **Key modules**: `core/pipeline.py` (orchestration), `core/mixer.py` (ducking/fades/export), `core/audio_processor.py` (Pedalboard FX chains), `core/session_config.py` (immutable generation config), `core/qa_monitor.py` (output validation)

**TTS engines** (`core/kokoro_tts/`, `core/f5_tts/`):
- Kokoro: Forced to CPU on Apple Silicon (MPS causes bus errors). Uses `trf=True` for transformer G2P.
- F5-TTS: Zero-shot voice cloning with Silero VAD (15% gain floor). Voice assets in `core/f5_tts/assets/`. WPM-based pacing via `fix_duration` (default 110 WPM). 300-char chunk limit. Multi-phase voices via `voices.toml`.

**Music engines** (`core/heart_mula/`, `core/acestep_engine.py`, `core/lyria/`):
- ACE-Step **(default, recommended for Apple Silicon)**: Must use `compile_model=True` to avoid timeouts. Uses MLX backend on Apple Silicon. ~5–10 min for a 5-min track at 48 kHz native. Music post-processing: noisereduce spectral repair → tape saturation → 12-stage Pedalboard EQ (Fletcher-Munson compensated) → organic noise floor. Story mode uses STFT crossfades for seamless transitions.
- HeartMuLa: 3B LM + HeartCodec (12.5 Hz codec), MLX primary with MPS fallback. Manual lazy loading (load LM → generate → unload → load codec → detokenize → unload). 48 kHz native. Max 240s per call; segment-and-crossfade for long-form. ~8–20 min per 4-min segment on MPS.
- Lyria RealTime: Cloud API (Google), fastest (~1–3 min), requires `GOOGLE_API_KEY`.

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
- **HeartMuLa lazy loading**: The engine manually loads/unloads LM and codec sequentially to avoid OOM. MLX `from_pretrained()` loads both models at once — never use it directly; the engine handles the lifecycle. MPS uses `lazy_load=True`. Peak usage ~6-8 GB per phase.
- **HeartCodec dtype**: Always use fp32 — bf16 degrades audio quality with metallic artifacts. MLX path passes `mx.float32` to HeartCodec; MPS path passes `{"codec": torch.float32}`.
- **HeartMuLa MPS watermark**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` (set in engine.py). Lower values (e.g. 0.4) cause OOM during 3B LM generation.
- **HeartMuLa checkpoints**: Must be present at `./ckpt/HeartMuLa-oss-3B/` and `./ckpt/HeartCodec-oss/` (MPS) or `./ckpt-mlx/heartmula/` and `./ckpt-mlx/heartcodec/` (MLX) before selecting HeartMuLa in the UI
