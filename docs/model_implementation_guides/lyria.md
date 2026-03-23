<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/lyria/engine.py` · `core/lyria/prompts.py`
**Class:** `LyriaEngine` — `load_model()` / `generate()` / `_run_session()` · cloud WebSocket API
**Constants:** `_MAX_SESSION_SEC=570.0` (9.5 min) · `_DEFAULT_BPM=70` · `_DEFAULT_DENSITY=0.2` · `_DEFAULT_BRIGHTNESS=0.3` · `_DEFAULT_GUIDANCE=4.0`
**Contract:** Output — 48 kHz mono float32 · Requires `GOOGLE_API_KEY` in `.env`
**SynthID watermark** embedded — do not strip or time-stretch (Google ToS)
**Weighted prompt syntax:** `"Hang Drum: 1.5, Piano: 0.8, Ambient Pads: 1.0"` → `prompts.py :: parse_weighted_prompts()`
**Tasks:**
- Tune default BPM/density/brightness → module-level constants in `engine.py`
- Change session crossfade → `_CROSSFADE_SEC` constant
- Adjust FX chain → `core/audio_processor.py :: make_lyria_music_chain()`
- Multi-session splitting → `generate()` (auto-splits durations > 570s)
**See also:** `docs/ARCHITECTURE.md#lyriaengine` · `docs/prompting_guides/` (no dedicated guide; use weighted syntax)
<!-- ────────────────────────────────────────────────────────────────── -->

# Lyria RealTime — Implementation Guide

## What Is Lyria RealTime?

[Lyria RealTime](https://deepmind.google/models/lyria/lyria-realtime/) (`models/lyria-realtime-exp`) is Google DeepMind's experimental streaming music generation model, accessed via the Gemini API. Unlike HeartMuLa and ACE-Step 1.5, which run fully locally, Lyria:

- Runs as a **cloud API** (no local GPU/NPU time for music generation)
- Streams **48 kHz stereo 16-bit PCM** over a persistent WebSocket
- Embeds a **SynthID watermark** in all output (must not be stripped)
- Has a **10-minute session cap** per WebSocket connection

Because Lyria uses no local VRAM, it is the most memory-efficient music option for MoodScape on Apple Silicon.

---

## Architecture

All Lyria-specific code lives in `core/lyria/` to keep it completely isolated from the existing engines.

```
core/lyria/
├── __init__.py       # package marker
├── engine.py         # LyriaEngine class (main integration)
└── prompts.py        # WeightedPrompt building + meditation presets
```

The engine exposes the same public interface as `HeartMulaEngine` and `AceStepEngine`:

```python
engine = LyriaEngine()
engine.load_model()   # validates API key, initialises genai.Client
audio = engine.generate(prompt, duration_sec, ...)  # returns mono f32 @ 48 kHz
engine.unload_model() # releases client reference (no GPU memory to free)
```

---

## API Key Setup

Lyria requires a Google AI Studio API key. Steps:

1. Go to [aistudio.google.com](https://aistudio.google.com) and create an API key.
2. Add it to the `.env` file at the project root:
   ```
   GOOGLE_API_KEY=AIza...
   ```
3. Restart the app. The key is read at engine load time via `os.environ.get("GOOGLE_API_KEY")`.

The `.env` file is already in `.gitignore` — the key is never committed.

If the key is missing when Lyria is selected, the app shows a clear error message before calling the pipeline.

---

## Audio Path: 48 kHz End-to-End

Lyria's native output is **48 kHz stereo 16-bit PCM**. MoodScape preserves this quality rather than downsampling.

```
Lyria API (48 kHz stereo int16)
  → deinterleave + average to mono float32     [core/lyria/engine.py: _pcm_to_numpy()]
  → returned at 48 kHz (no resampling)

TTS voice (24 kHz mono, Kokoro)
  → upsample_audio(24000 → 48000)              [core/audio_processor.py: upsample_audio()]
  → voice_activity mask repeated ×2 (exact 2:1 ratio)

Mix at 48 kHz                                  [core/mixer.py: mix()]
  → Pedalboard FX applied at 48 kHz
  → export at 48 kHz
```

When Lyria is selected, the pipeline sets `mix_sr = 48000` and all downstream operations (voice FX, music FX, ducking, export) use that rate. No resampling of music occurs — it is already at `mix_sr`.

For context, the other engines' paths are:
- **HeartMuLa**: 44.1 kHz (engine) → 44.1 kHz (pipeline)
- **ACE-Step 1.5**: 48 kHz → 24 kHz (engine) → 44.1 kHz (pipeline)
- **Lyria**: 48 kHz (API) → 48 kHz (engine) → 48 kHz (pipeline, TTS upsampled to match)

---

## Session Duration & Multi-Session Stitching

Each Lyria WebSocket session supports up to ~10 minutes of audio. MoodScape uses a safe cap of **570 seconds (9.5 minutes)**.

If the requested duration exceeds this:
1. The engine splits the total into multiple ≤570s chunks.
2. Each chunk is a separate `client.aio.live.music.connect()` session.
3. Chunks are stitched with a **3-second equal-power cosine crossfade**:
   - `fade_out = cos(t)`, `fade_in = sin(t)` for `t ∈ [0, π/2]`
   - Satisfies `fade_out² + fade_in² = 1` — no energy dip at seams.

The same crossfade applies in story mode, where one session is opened per stage.

---

## Prompt Engineering

### Weighted Prompt Syntax

Lyria uses weighted text prompts via `WeightedPrompt(text, weight)`. The user can write:

```
Ambient: 1.5, Piano: 0.8, Hang Drum: 1.2
```

This is parsed by `core/lyria/prompts.py:parse_weighted_prompt_string()` into individual weighted prompts. Plain text without colons is treated as a single prompt at weight 1.0.

### Meditation Base Tags

`core/lyria/prompts.py:build_lyria_prompts()` automatically prepends base tags at weight 0.6:

```python
MEDITATION_BASE_TAGS = [
    "Ambient",
    "Ethereal Ambience",
    "Subdued Melody",
    "Sustained Chords",
]
```

Tags already present in the user's input are skipped to avoid diluting the attention budget.

### What to Avoid

- **Artist names**: Lyria safety filters may block or alter output.
- **Very high guidance** (>5.0): Can produce abrupt tonal shifts.
- **High density + high brightness together**: Creates busy, non-meditative textures.

### Recommended Meditation Settings

| Parameter  | Default | Guidance |
|------------|---------|----------|
| BPM        | 70      | 60–80 suits slow breathing rhythms |
| Density    | 0.2     | Low = sparse, minimal; high = busy |
| Brightness | 0.3     | Low = warm/dark; high = bright/airy |
| Guidance   | 4.0     | Fixed in engine; midpoint of [0, 6] range |

---

## LiveMusicGenerationConfig Parameters

| Parameter    | Type  | Range       | Notes |
|--------------|-------|-------------|-------|
| `bpm`        | int   | 60–200      | Requires `reset_context()` to take effect mid-session (not used here — each session starts fresh) |
| `density`    | float | 0.0–1.0     | Musical density; can change smoothly on-the-fly |
| `brightness` | float | 0.0–1.0     | Spectral brightness; can change smoothly on-the-fly |
| `guidance`   | float | 0.0–6.0     | Prompt adherence strength; hardcoded to 4.0 |
| `scale`      | Enum  | Various     | Not exposed in MoodScape UI (left to model) |

---

## FX Chain

`core/audio_processor.py:make_lyria_music_chain()` is applied to Lyria output at 48 kHz before mixing:

| Stage | Plugin | Setting | Rationale |
|-------|--------|---------|-----------|
| 1 | HighpassFilter | 60 Hz | Remove sub-bass energy from API stream |
| 2 | PeakFilter | 250 Hz, −1.5 dB | Mud notch — Lyria's harmonic-rich textures can accumulate warmth here |
| 3 | PeakFilter | 4500 Hz, −2.0 dB | Soften upper-mid presence; keeps ambient bed behind narration |
| 4 | HighShelfFilter | 9000 Hz, −2.5 dB | Gentle HF rolloff for warmth; Lyria extends above 12 kHz unlike other engines |
| 5 | Compressor | 2:1, 80ms attack, 500ms release | Slow glue — no pumping on sustained pads |
| 6 | Limiter | −0.5 dBFS | Extra headroom before master chain |

**Pre-mix LUFS target**: −22 LUFS (slightly quieter than HeartMuLa's −20, slightly louder than ACE-Step's −24).

---

## Pipeline Routing

`core/pipeline.py` routes to Lyria when `music_model == "lyria"`:

```
generate_meditation()              [app.py]
  → music_model_choice == "Lyria RealTime"
  → music_model = "lyria"
  → validates GOOGLE_API_KEY env var

MeditationPipeline.generate()     [core/pipeline.py]
  → use_lyria = music_model == "lyria"
  → mix_sr = 48000

TTS synthesis → 24 kHz            [kokoro engine]
→ upsample 24k → 48k              [upsample_audio()]
→ voice_activity repeated ×2

LyriaEngine.load_model()          [validates API key, creates genai.Client]
LyriaEngine.generate()            [returns mono f32 @ 48 kHz]
LyriaEngine.unload_model()

→ stem separation (optional)      [Demucs, accepts 48 kHz]
→ make_lyria_music_chain() FX     [at 48 kHz]
→ normalize to −22 LUFS
→ mix() at 48 kHz                 [ducking, fades]
→ export at 48 kHz                [always, regardless of upsample_48k toggle]
```

---

## SynthID Watermark

All Lyria audio is embedded with Google DeepMind's SynthID watermark at the API level. **Do not add any post-processing that would strip or modify this watermark.** The watermark survives:
- Loudness normalisation
- EQ and compression (Pedalboard)
- Stereo-to-mono averaging

It does not survive:
- Time-stretching / pitch-shifting
- Aggressive codec compression (low-bitrate MP3)

MoodScape's pipeline does not apply any watermark-destroying operations.

---

## Known Limitations

| Limitation | Notes |
|------------|-------|
| No melody conditioning | Lyria does not accept reference audio for melody/timbre guidance (HeartMuLa-only feature) |
| No musical key control | Scale parameter exists in the API but is not exposed in the UI (left to model) |
| 10-minute session cap | Handled automatically via multi-session stitching with crossfade |
| Cloud dependency | Requires internet + valid API key; no offline fallback |
| Experimental API | `models/lyria-realtime-exp` is a preview model subject to change |
| Free tier quota | Currently free; quota limits not publicly documented |

---

## Dependencies

```
google-genai>=1.16.0   # Google GenAI Python SDK with Lyria RealTime support
```

Already in `requirements.txt`. Install with:

```bash
pip install 'google-genai>=1.16.0'
```

The SDK is imported lazily inside `LyriaEngine.load_model()` to avoid import-time errors if the package is somehow missing (fails clearly at engine load time, not at app startup).
