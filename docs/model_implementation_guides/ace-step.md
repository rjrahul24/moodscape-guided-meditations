# ACE-Step 1.5 Implementation Guide for Meditation Audio
### A Complete Reference for Moodscape

---

## 1. Purpose & Scope

This document is the complete reference for the **ACE-Step 1.5** music generation backend in the MoodScape meditation audio generator. It covers:

- What ACE-Step 1.5 is and why it fits meditation audio generation
- Hardware setup for **Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)** via **MLX**
- Model architecture overview (LM planner + DiT decoder)
- Prompt engineering using the MESA framework
- Complete implementation walkthrough of `core/acestep_engine.py`
- Three-phase long-form generation pipeline
- Pipeline integration with the existing workflow
- Parameter tuning, memory management, and troubleshooting

---

## 2. What is ACE-Step 1.5?

ACE-Step 1.5 is an open-source **text-to-music foundation model** with a two-brain architecture:

| Stage | Component | Role |
|-------|-----------|------|
| **Planning** | Language Model (LM) — 4B Qwen3 (primary) / 1.7B fallback | Reads the text prompt and uses Chain-of-Thought reasoning to plan musical structure — tempo, key, section layout, dynamics, and 5Hz semantic audio codes |
| **Synthesis** | Diffusion Transformer (DiT) — ~2B params | Takes the LM plan and renders 48 kHz stereo audio through iterative denoising using a 1D VAE with 1920x compression |

**Key insight:** The LM planner's output quality is the single biggest determinant of final audio quality. A bad plan produces bad audio regardless of DiT settings.

### Why ACE-Step for Meditation?

| Feature | Benefit for Meditation |
|---------|----------------------|
| **48 kHz native output** | Higher fidelity than MusicGen's 32 kHz — richer harmonics in singing bowls, pads |
| **Structural coherence up to 10 min** | LM planner ensures the track evolves meaningfully over long durations without drifting |
| **Chain-of-Thought planning** | Can be directed to plan slow, meditative arcs with controlled BPM (40–60) |
| **`instrumental=True` mode** | Hard-codes no vocal generation — critical for meditation backgrounds |
| **MLX backend** | Native Apple Silicon Metal GPU acceleration — fully utilizes the M1 Max's 24-core GPU |

### ACE-Step vs MusicGen — Which to Choose?

| Criteria | MusicGen | ACE-Step 1.5 |
|----------|----------|--------------|
| **Native sample rate** | 32 kHz | 48 kHz |
| **Long-form approach** | Sliding window continuation (30s segments stitched) | Three-phase pipeline (genesis + cover continuation + boundary smoothing) |
| **Coherence** | Depends on context overlap; can drift | LM-planned structure; naturally coherent |
| **Backend** | MPS / CPU via AudioCraft | MLX (Metal native) |
| **Best for** | Reliable ambient, fast prototyping | Long-form coherent journeys, high fidelity |

---

## 3. Hardware Requirements — Apple Silicon M1 Max

### MLX Backend

| Component | Requirement |
|-----------|-------------|
| **Chip** | Apple Silicon (M1, M1 Pro, M1 Max, M2, M3, etc.) |
| **RAM** | 36 GB unified recommended (DiT + LM loaded simultaneously) |
| **macOS** | 13.0+ (Ventura or later for MLX support) |
| **Python** | 3.10+ |

---

## 4. Installation

### Prerequisites
```bash
python3 --version  # 3.10+ required
python3 -c "import platform; print(platform.machine())"  # Should print 'arm64'
```

### Python Dependencies
```bash
source .venv/bin/activate
pip install mlx
pip install git+https://github.com/ace-step/ACE-Step-1.5.git
python -c "from acestep.handler import AceStepHandler; print('ACE-Step OK')"
python -c "from acestep.llm_inference import LLMHandler; print('LLM Handler OK')"
```

### Model Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| **DiT config** | `./ACE-Step-1.5/` | Diffusion Transformer with `acestep-v15-sft` config |
| **LLM checkpoint** | `./ACE-Step-1.5/checkpoints/` | Language Model `acestep-5Hz-lm-4B` (preferred) or `acestep-5Hz-lm-1.7B` (fallback) |

---

## 5. Model Configuration

### DiT (Diffusion Transformer)

| Config | Use Case | Quality | Speed |
|--------|----------|---------|-------|
| **`acestep-v15-sft`** | Production — maximum detail | Highest | Slower |
| `turbo` | Fast prototyping | Lower | Fastest (8 steps) |

### Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `_GUIDANCE_SCALE` | `5.0` | SFT optimal range 5.0–7.0; balances prompt adherence with smooth textures |
| `_INFERENCE_STEPS` | `50` | SFT max without error accumulation |
| `_INFERENCE_STEPS_REPAINT` | `50` | Matches base steps for consistency |
| `_LM_TEMPERATURE` | `0.85` | Richer tonal palettes while maintaining coherence |
| `_USE_ADG` | `True` | Adaptive Dual Guidance reduces spectral noise |
| `instrumental` | `True` | Always — meditation tracks must never have vocals |
| `thinking` | `True` | Enables Chain-of-Thought planning for coherent structure |
| `vocal_language` | `"unknown"` | Best practice for instrumental tracks |
| `enable_normalization` | `True` | Consistent output levels from the VAE decoder |

---

## 6. Prompt Engineering — MESA Framework

### Overview

The prompt system uses the **MESA** framework:
- **M**ood: Emotional context (serene, contemplative, peaceful)
- **E**lements: Instruments and textures (tibetan bowls, warm pads, soft piano)
- **S**tructure: Song architecture via structural lyrics tags
- **A**pplication: Use case context (meditation background, sleep journey)

### Caption Construction

`_enhance_prompt()` builds the caption from:
1. **Base tags** (filtered to avoid duplicating user words): `ambient, meditation, calm, peaceful, warm, spacious, soft dynamics, gentle, soothing, high fidelity, studio quality, clean production`
2. **User prompt** — appended verbatim to preserve creative intent
3. **Instrumental constraint** — `no vocals, instrumental` (if not already present)

**Important:** The caption does NOT contain contradictory directives like "no melody" or "no chord changes". These fight the LM planner and produce incoherent output.

### Structural Lyrics

Even for instrumental tracks, structural section tags guide the model's section-aware generation. The tags scale with duration:

**Short tracks (≤90s):**
```
[Instrumental]
[Intro - Gentle ambient texture emerging softly]
[Verse - Main theme, warm and meditative, slowly developing]
[Outro - Gradual fade, dissolving into stillness]
```

**Medium tracks (90s–300s):** Adds Bridge and second Verse

**Long tracks (>300s):** Adds Interlude and second Bridge for expanded meditation arc

### Curated Presets

**Deep Sleep / Theta Waves**
```
Deep sub-bass drone, extremely slow evolution, warm analog synth pads,
healing 432Hz tones, dark and warm tonal balance, very spacious reverb
```

**Mindfulness / Breath Focus**
```
Single sustained cello pad, occasional soft singing bowl strike, vast silence,
warm room reverb, meditative and grounding, minimal movement
```

**Zen Garden / Clarity**
```
Soft shakuhachi flute with long decays, warm synth drone, occasional distant gong,
silence between notes, dry reverb, high clarity, introspective
```

### Anti-Patterns

- Do NOT include BPM or key in the caption — use dedicated metadata parameters
- Do NOT use contradictory terms ("no melody", "static harmony")
- Do NOT use non-standard lyrics tags (`[Drone]`, `[Static Pad]`, `[Beatless]`)
- Do NOT include `upbeat`, `energetic`, `dance` — triggers rhythmic planning

---

## 7. Implementation — `core/acestep_engine.py`

### Class Interface

`AceStepEngine` mirrors `MusicEngine`'s interface:

| Method | Signature | Returns |
|--------|-----------|---------|
| `load_model()` | `(model_type="sft") -> None` | Initializes DiT + LLM on MLX |
| `unload_model()` | `() -> None` | Shuts down services, frees memory |
| `generate()` | `(prompt, total_duration_sec, ...) -> np.ndarray` | Mono float32 at 24 kHz |

### Audio Post-Processing

The postprocess chain is minimal — the 1D VAE produces near-lossless quality:

```
ACE-Step output (48 kHz stereo tensor)
    │
    ├── Stereo → Mono: tensor.mean(dim=0)
    │
    ├── Resample 48 kHz → 24 kHz: torchaudio.functional.resample()
    │       (rolloff=0.9475 provides proper Nyquist filtering)
    │
    ├── Peak normalize to -1 dBFS (consistent output level)
    │
    └── Output: float32 numpy array at 24 kHz
```

All spectral shaping is handled by the downstream Pedalboard FX chain (`make_acestep_music_chain()`) at 44.1 kHz.

### Three-Phase Long-Form Pipeline (>90s)

For tracks exceeding 90 seconds, `_generate_infinite()` uses a three-phase approach:

**Phase 1 — Genesis:**
Generate the initial 60s anchor via `text2music`. This sets the harmonic DNA, timbre palette, and tonal baseline.

**Phase 2 — Continuation via Cover Task:**
For each subsequent segment, use the `cover` task with `audio_cover_strength` modulation:
- Segment 2: strength=0.85 (close to anchor)
- Segment 3: strength=0.80
- Segment 4+: strength=0.75 (floor 0.70)

The cover task preserves the source's harmonic skeleton while allowing controlled tonal evolution. All intermediate audio stays at native 48 kHz stereo.

**Phase 3 — Boundary Smoothing:**
Repaint 5-second windows centered on each seam between segments. The DiT automatically smooths rhythm, harmony, and timbre across the boundary.

**Final:** Single postprocess converts the complete 48 kHz stereo track to 24 kHz mono.

### Output Validation

`_validate_output()` checks for:
- NaN/Inf values (model failure)
- Near-silence (RMS < -50 dBFS)
- Excessive clipping (>5% samples at ±1.0)
- Short output (<50% of requested duration)

Failed generations are retried up to 3 times automatically.

### A/B Selection on Retry

Rather than keeping the last successful attempt, the retry loops collect all candidates (both single-stage and continuation segments) and score each via `core.qa_monitor.compute_composite_score()`. The candidate with the **highest composite score** is selected. The composite score combines:

| Check | Weight |
|---|---|
| Spectral warmth dominance | 0.25 |
| Spectral rolloff within 8 kHz | 0.20 |
| Onset smoothness (no transient spikes) | 0.20 |
| Clipping-free | 0.20 |
| LUFS proximity to -16 dBFS | 0.15 |

**Early exit**: if any candidate scores **> 0.8**, generation stops immediately — further retries are skipped. This avoids wasting compute when the first attempt is already high-quality.

### Memory Management

```python
def unload_model(self):
    self._llm.unload()
    self._dit = None
    self._llm = None
    gc.collect()
    torch.mps.empty_cache()
```

Explicit `gc.collect()` + `torch.mps.empty_cache()` ensures the 36 GB pool is clear before the next TTS model loads.

---

## 8. Pipeline Integration

### Sequential Loading Pattern

```
1. Load TTS → Synthesize narration → Unload TTS
2. Load ACE-Step → Generate music → Unload ACE-Step
3. Mix, apply FX, export (CPU-only)
```

### Model Routing in `pipeline.py`

```python
if use_acestep:
    from core.acestep_engine import AceStepEngine
    music_engine = AceStepEngine()
music_engine.load_model()
music_audio = music_engine.generate(enhanced_prompt, duration, ...)
music_engine.unload_model()
```

### Prompt Enhancement Routing

- **MusicGen:** `_enhance_music_prompt()` — caps at 45 words
- **ACE-Step:** `_enhance_acestep_prompt(prompt, duration_hint)` — MESA framework with duration-aware lyrics

### Pre-Mix Loudness

ACE-Step output is normalized to **-20 LUFS** before mixing, matching MusicGen's reference level.

---

## 9. Gradio UI Integration

### ACE-Step Controls

- `music_model_dropdown`: Choices = ["MusicGen", "ACE-Step 1.5", "Lyria RealTime"]
- `acestep_quality`: Radio button for "Full Quality (SFT)" vs "Turbo (Fast)"
- `acestep_bpm`: Slider 40–100, default 50
- `acestep_key`: Dropdown for musical key, default "Auto"

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'acestep'"
```bash
pip install git+https://github.com/ace-step/ACE-Step-1.5.git
```

### OOM / Memory Pressure During Generation
- Ensure TTS model is fully unloaded before ACE-Step loads
- Close other GPU-intensive applications
- Monitor: `sudo memory_pressure`

### Generated Audio Has Vocals Despite instrumental=True
- Verify `lyrics` starts with `[Instrumental]`
- Ensure prompt doesn't contain vocal-adjacent terms

### Audio Quality Issues
- Verify MLX backend is active (check logs for "MLX" or "mps")
- Increase `_GUIDANCE_SCALE` if output doesn't match prompt (max 7.0)
- Decrease `_LM_TEMPERATURE` if output is too unpredictable

### Long-Form Seam Artifacts
- The three-phase pipeline (cover + boundary smoothing) should eliminate seams
- If artifacts persist, increase `BOUNDARY_WINDOW_SEC` in `_generate_infinite()`

---

## 11. Quick Reference — Default Values

```python
# Model configuration
DIT_CONFIG = "acestep-v15-sft"
LLM_MODEL = "acestep-5Hz-lm-4B" # Falls back to acestep-5Hz-lm-1.7B if 4B checkpoint is absent
BACKEND = "mlx"
COMPILE_MODEL = True

# Generation parameters
GUIDANCE_SCALE = 5.0
INFERENCE_STEPS = 50
LM_TEMPERATURE = 0.85
USE_ADG = True
INSTRUMENTAL = True
ENABLE_NORMALIZATION = True
VOCAL_LANGUAGE = "unknown"

# Audio format
NATIVE_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 24000
OUTPUT_FORMAT = "mono float32 numpy"

# Long-form pipeline
GENESIS_LENGTH = 60.0         # Initial anchor segment
CONTINUATION_LENGTH = 60.0    # Cover continuation segments
CONTEXT_LENGTH = 30.0         # Source audio for cover task
CROSSFADE_SEC = 2.0           # Crossfade between segments
BOUNDARY_WINDOW_SEC = 5.0     # Repaint window at seams
STORY_CROSSFADE_SEC = 6.0     # Crossfade between story stages
```
