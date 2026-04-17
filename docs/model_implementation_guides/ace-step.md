<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/acestep_engine.py`
**Class:** `AceStepEngine` — `load_model()` / `generate()` / `_generate_infinite()` / `_enhance_prompt()`
**Constants:** `_GUIDANCE_SCALE=5.5` · `_INFERENCE_STEPS=50` · `_LM_TEMPERATURE=0.65` · `_USE_ADG=False` · `_CFG_INTERVAL_END=0.8` · `_STORY_CROSSFADE_SEC=6.0`
**Contract:** Output — 48 kHz mono float32 · Checkpoints — `./ACE-Step-1.5/checkpoints/`
**MANDATORY:** `compile_model=True` to `initialize_service()` — without it, ~9s/step → timeout
**Tasks:**
- Tune generation params → module-level constants at top of `acestep_engine.py`
- Change prompt enhancement → `_enhance_prompt()` (MESA framework: Mood/Elements/Structure/Application)
- Adjust FX chain → `core/audio_processor.py :: make_acestep_music_chain()`
- Long-form behavior → `_generate_infinite()` (two-phase: genesis [90s] + repaint continuation [20s overlap, 60s new per iteration])
**See also:** `docs/ARCHITECTURE.md#acestepenginecoreacestepenginepy` · `docs/prompting_guides/ace_step_instructions.md`
<!-- ────────────────────────────────────────────────────────────────── -->

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
- Two-phase long-form generation pipeline (genesis + repaint continuation)
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
| **48 kHz native output** | Higher fidelity than HeartMuLa's 44.1 kHz — richer harmonics in singing bowls, pads |
| **Structural coherence up to 10 min** | LM planner ensures the track evolves meaningfully over long durations without drifting |
| **Chain-of-Thought planning** | Can be directed to plan slow, meditative arcs with controlled BPM (40–60) |
| **`instrumental=True` mode** | Hard-codes no vocal generation — critical for meditation backgrounds |
| **MLX backend** | Native Apple Silicon Metal GPU acceleration — fully utilizes the M1 Max's 24-core GPU |

### ACE-Step vs HeartMuLa — Which to Choose?

| Criteria | HeartMuLa | ACE-Step 1.5 |
|----------|-----------|--------------|
| **Native sample rate** | 44.1 kHz | 48 kHz |
| **Long-form approach** | Sliding window continuation (30s segments stitched) | Two-phase pipeline (90s genesis + overlapping repaint continuation, 20s context per call) |
| **Coherence** | Tag-based prompting; depends on context overlap | LM-planned structure; naturally coherent |
| **Backend** | MPS / CPU via HeartLib | MLX (Metal native) |
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
| `_GUIDANCE_SCALE` | `5.5` | Upper SFT sweet spot (4–6); strong prompt adherence for ambient texture control without rigidity |
| `_INFERENCE_STEPS` | `50` | SFT max without error accumulation |
| `_INFERENCE_STEPS_REPAINT` | `50` | Matches base steps for consistency |
| `_LM_TEMPERATURE` | `0.65` | Balances harmonic variety with calm predictability; prevents unexpected timbral jumps while avoiding harmonic stagnation |
| `_USE_ADG` | `False` | ADG disabled — SFT model has strong prompt adherence baked in; ADG doubles forward passes without quality benefit on SFT |
| `_CFG_INTERVAL_END` | `0.8` | Release CFG at 80% — only final 20% of micro-detail steps run free; 0.6 allowed too much drift from prompt |
| `_SHIFT` | `3.0` | Higher timestep shift = stronger semantic conditioning, cleaner harmonics |
| `_INFER_METHOD` | `"ode"` | Deterministic ODE for smooth, reproducible output |
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

**Medium tracks (90s–300s):** Adds `[Verse]` + `[Chorus]` around a central `[Instrumental]` block

**Long tracks (>300s):** Adds `[Bridge]` + a second `[Verse]`/`[Chorus]` pass for expanded meditation arc

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

`AceStepEngine` mirrors `HeartMulaEngine`'s interface:

| Method | Signature | Returns |
|--------|-----------|---------|
| `load_model()` | `(model_type="sft") -> None` | Initializes DiT + LLM on MLX |
| `unload_model()` | `() -> None` | Shuts down services, frees memory |
| `generate()` | `(prompt, total_duration_sec, ...) -> np.ndarray` | Mono float32 at 48 kHz |

### Audio Post-Processing

The postprocess chain is minimal — the 1D VAE produces near-lossless quality:

```
ACE-Step output (48 kHz stereo tensor)
    │
    ├── Stereo → Mono: tensor.mean(dim=0)
    │
    ├── Peak normalize to -1 dBFS (consistent output level)
    │
    └── Output: float32 numpy array at 48 kHz (native rate preserved)
```

All spectral shaping is handled by the downstream Pedalboard FX chain (`make_acestep_music_chain()`) at 48 kHz.

### Two-Phase Long-Form Pipeline (>90s)

For tracks exceeding 90 seconds, `_generate_infinite()` uses a two-phase approach:

**Phase 1 — Genesis:**
Generate the initial 90s anchor via `text2music`. This sets the harmonic DNA, timbre palette, and tonal baseline.

**Phase 2 — Repaint Continuation:**
For each subsequent segment, use `task_type="repaint"` with a 20-second overlap window:
- Extract the last 20s of accumulated audio as context (written to a temp WAV file)
- ACE-Step repaints from `repainting_start=20.0s` forward, generating up to 60s of new audio
- Append only the newly generated tail (after the 20s overlap region) to the accumulation

The repaint task produces seamless transitions at the model level — no post-hoc STFT crossfades required. All intermediate audio stays at native 48 kHz mono.

**Final:** Single postprocess converts the complete 48 kHz mono track to float32 numpy output.

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

- **HeartMuLa:** `_enhance_music_prompt()` — caps at 45 words
- **ACE-Step:** `_enhance_acestep_prompt(prompt, duration_hint)` — MESA framework with duration-aware lyrics

### Pre-Mix Loudness

ACE-Step output is normalized to **-14 LUFS** before mixing (matching the final streaming standard). HeartMuLa uses -20 LUFS; Lyria uses -16 LUFS.

---

## 9. Gradio UI Integration

### ACE-Step Controls

- `music_model_dropdown`: Choices = ["HeartMuLa", "ACE-Step 1.5", "Lyria RealTime"]
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
- Increase `_GUIDANCE_SCALE` if output doesn't match prompt (max 7.0; currently 5.5)
- Decrease `_LM_TEMPERATURE` if output is too unpredictable (currently 0.65; minimum ~0.2)

### Long-Form Seam Artifacts
- The two-phase repaint pipeline produces seamless transitions at the model level — no post-hoc STFT crossfades
- If artifacts persist, increase `CONTINUATION_OVERLAP` (currently 20s) in `_generate_infinite()` to provide more context per repaint call

---

## 11. Quick Reference — Default Values

```python
# Model configuration
DIT_CONFIG = "acestep-v15-sft"
LLM_MODEL = "acestep-5Hz-lm-4B" # Falls back to acestep-5Hz-lm-1.7B if 4B checkpoint is absent
BACKEND = "mlx"
COMPILE_MODEL = True

# Generation parameters
GUIDANCE_SCALE = 5.5
INFERENCE_STEPS = 50
LM_TEMPERATURE = 0.65
USE_ADG = False              # Disabled — SFT has strong adherence baked in; ADG doubles passes with no benefit
CFG_INTERVAL_END = 0.8       # Release guidance at 80% of denoising steps
INSTRUMENTAL = True
ENABLE_NORMALIZATION = True
VOCAL_LANGUAGE = "unknown"

# Audio format
NATIVE_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 48000
OUTPUT_FORMAT = "mono float32 numpy at 48 kHz"

# Long-form pipeline (two-phase)
GENESIS_LENGTH = 90.0         # Initial anchor segment (text2music)
CONTINUATION_LENGTH = 60.0    # New audio per repaint continuation call
CONTINUATION_OVERLAP = 20.0   # Context window passed to each repaint call
STORY_CROSSFADE_SEC = 6.0     # Crossfade between story stages
```
