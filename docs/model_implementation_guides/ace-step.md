# ACE-Step 1.5 Implementation Guide for Meditation Audio
### A Complete Reference for Moodscape

---

## 1. Purpose & Scope

This document is the complete reference for the **ACE-Step 1.5** music generation backend in the MoodScape meditation audio generator. It covers:

- What ACE-Step 1.5 is and why it fits meditation audio generation
- Hardware setup for **Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)** via **MLX**
- Model architecture overview (LM planner + DiT decoder)
- Prompt engineering vocabulary specific to ACE-Step + meditation
- Complete implementation walkthrough of `core/acestep_engine.py`
- Pipeline integration with the existing MusicGen-based workflow
- Parameter tuning, memory management, and troubleshooting

---

## 2. What is ACE-Step 1.5?

ACE-Step 1.5 is an open-source **text-to-music foundation model** that combines two stages:

| Stage | Component | Role |
|-------|-----------|------|
| **Planning** | Language Model (LM) | Reads the text prompt and uses Chain-of-Thought reasoning to plan the musical structure — tempo, key, section layout, dynamics |
| **Synthesis** | Diffusion Transformer (DiT) | Takes the LM plan and generates high-fidelity audio at **48 kHz stereo** using iterative denoising |

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
| **Long-form approach** | Sliding window continuation (30s segments stitched) | Single-pass generation (up to 10 min) |
| **Coherence** | Depends on context overlap; can drift | LM-planned structure; naturally coherent |
| **Backend** | MPS / CPU via AudioCraft | MLX (Metal native) |
| **Generation speed** | Fast for short segments | Slower per-call but fewer calls needed |
| **Maturity** | Well-established (Meta, large community) | Newer, smaller community |
| **Best for** | Reliable ambient, fast prototyping | Long-form coherent journeys, high fidelity |

---

## 3. Hardware Requirements — Apple Silicon M1 Max

### MLX Backend

ACE-Step 1.5 runs on Apple's **MLX** framework, which provides native Metal GPU acceleration on Apple Silicon. This is the recommended backend for M1 Max:

| Component | Requirement |
|-----------|-------------|
| **Chip** | Apple Silicon (M1, M1 Pro, M1 Max, M2, M3, etc.) |
| **RAM** | 36 GB unified recommended (DiT + LM loaded simultaneously) |
| **macOS** | 13.0+ (Ventura or later for MLX support) |
| **Python** | 3.10+ |

> **Note:** Unlike MusicGen (which uses MPS/CPU via PyTorch), ACE-Step uses MLX which accesses Metal GPU directly without the MPS compatibility issues that AudioCraft faces.

---

## 4. Installation

### Prerequisites
```bash
# Confirm Python version
python3 --version  # 3.10+ required

# MLX requires Apple Silicon — verify
python3 -c "import platform; print(platform.machine())"  # Should print 'arm64'
```

### Python Dependencies
```bash
# Activate virtual environment
source .venv/bin/activate

# Install MLX (Apple's native ML framework)
pip install mlx

# Install ACE-Step 1.5 from the official repository
pip install git+https://github.com/ace-step/ACE-Step-1.5.git

# Verify installation
python -c "from acestep.handler import AceStepHandler; print('ACE-Step OK')"
python -c "from acestep.llm_inference import LLMHandler; print('LLM Handler OK')"
```

### Model Artifacts

ACE-Step requires two model artifacts to be present locally:

| Artifact | Path | Description |
|----------|------|-------------|
| **DiT config** | `./ACE-Step-1.5/` | Diffusion Transformer with `acestep-v15-sft` config |
| **LLM checkpoint** | `./ACE-Step-1.5/checkpoints/` | Language Model `acestep-5Hz-lm-1.7B` |

These are downloaded automatically on first run via the `AceStepHandler` and `LLMHandler` initialization, or can be cloned manually:
```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
```

---

## 5. Model Configuration

### DiT (Diffusion Transformer)

| Config | Use Case | Quality | Speed |
|--------|----------|---------|-------|
| **`acestep-v15-sft`** ✅ | Production — maximum detail | Highest | Slower |
| `base` | General use | Good | Moderate |
| `turbo` | Fast prototyping | Lower | Fastest |

**Recommendation:** Use `acestep-v15-sft` — with 36 GB RAM there's no need to compromise on quality.

### LLM (Language Model Planner)

| Model | Parameters | Use Case |
|-------|-----------|----------|
| **`acestep-5Hz-lm-1.7B`** ✅ | 1.7B | Production — good balance of planning quality and speed |
| `acestep-5Hz-lm-4B` | 4B | Higher quality planning, slower |

### Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `instrumental` | `True` | **Always** — meditation tracks must never have vocals |
| `lyrics` | `"[Instrumental]"` | Reinforces instrumental mode at the token level |
| `inference_steps` | `32` | Good quality/speed balance; increase to 50 for maximum fidelity |
| `thinking` | `True` | Enables Chain-of-Thought planning for coherent long-form structure |
| `duration` | User-specified | Passed directly from the UI; supports up to ~600s (10 min) |

---

## 6. Prompt Engineering for Meditation Audio

### ACE-Step Prompt Strategy

ACE-Step's LM planner responds well to **explicit structural guidance**. Unlike MusicGen (which has a ~45 word attention window), ACE-Step can process longer, more descriptive prompts because the LM stage has a full transformer context.

### Automatic Prompt Enhancement

The `AceStepEngine._enhance_prompt()` method automatically augments the user's prompt:

```python
# User provides: "Soft piano, warm pads, singing bowls"
# Engine sends: "Meditation, ambient, minimalist, Soft piano, warm pads, singing bowls, 
#                slow ambient pads, minimal motifs, no percussion, gentle, warm drone, 
#                calm, spacious, no drums, slow tempo, high fidelity, lush reverb"
```

The enhancer only adds keywords the user hasn't already included, to avoid diluting prompt attention with duplicates.

### Keyword Vocabulary

**Always-included base:** `Meditation, ambient, minimalist`

**Ambient textures (auto-appended if not present):**
`slow ambient pads` · `minimal motifs` · `warm drone` · `calm` · `spacious` · `gentle` · `no percussion`

**Tail constraints (always appended):**
`no drums` · `slow tempo` · `high fidelity` · `lush reverb`

### Curated Presets for ACE-Step

**Deep Sleep / Theta Waves**
```
Deep sub-bass drone, extremely slow evolution, warm analog synth pads, 
healing 432Hz tones, dark and warm tonal balance, very spacious reverb,
constant volume throughout
```

**Mindfulness / Breath Focus**
```
Single sustained cello pad, occasional soft singing bowl strike, vast silence, 
60 BPM feel, warm room reverb, meditative and grounding, minimal movement
```

**Zen Garden / Clarity**
```
Soft shakuhachi flute with long decays, warm synth drone, occasional distant gong,
silence between notes, dry reverb, high clarity, introspective
```

**Healing / Chakra Work**
```
Tibetan singing bowls, pure harmonic overtones, resonant metal vibrations,
432Hz tuning, distant meditative gong, healing frequencies, sacred warmth
```

**Morning Awakening**
```
Light shimmering bells, gentle piano chords with long sustain, birdsong texture,
slowly rising warm pads, peaceful and hopeful, very slow and gentle
```

### Anti-Patterns (What NOT to Put in ACE-Step Prompts)

- ❌ `upbeat`, `energetic`, `dance`, `groove` — triggers rhythmic planning in the LM
- ❌ `vocals`, `singing`, `rap`, `lyrics` — contradicts `instrumental=True`
- ❌ `fast tempo`, `120 BPM` — fights the meditation intent
- ❌ Contradictory terms (`beatless with a groove`) — confuses the LM planner

### BPM Control

ACE-Step's LM planner understands BPM constraints. For meditation:
- **Target BPM:** 40–60 (very slow, meditative)
- This can be passed via `use_cot_metas=True` or by including `"40 BPM"` in the prompt
- The LM will plan sections and transitions around this tempo

---

## 7. Implementation — `core/acestep_engine.py`

### Class Interface

`AceStepEngine` mirrors `MusicEngine`'s interface exactly:

| Method | Signature | Returns |
|--------|-----------|---------|
| `load_model()` | `() -> None` | Initializes DiT + LLM on MLX |
| `unload_model()` | `() -> None` | Shuts down services, frees memory |
| `generate()` | `(prompt, total_duration_sec, progress_cb) -> np.ndarray` | Mono float32 at 24 kHz |

### Audio Post-Processing Pipeline

ACE-Step outputs **48 kHz stereo**. The Moodscape pipeline expects **24 kHz mono**. The conversion happens inside `generate()`:

```
ACE-Step output (48 kHz stereo tensor)
    │
    ├── Stereo → Mono: tensor.mean(dim=0, keepdim=True)
    │
    ├── Resample 48 kHz → 24 kHz: torchaudio.functional.resample()
    │       (lowpass_filter_width=64, rolloff=0.9475 for high-quality sinc interpolation)
    │
    └── Output: float32 numpy array at 24 kHz (same as MusicEngine)
```

This ensures the downstream pipeline (`mixer.py`, `audio_processor.py`, `export_audio()`) works identically regardless of which music engine was used.

### Memory Management

```python
def unload_model(self):
    """Release ACE-Step models and aggressively free memory."""
    if self._dit is not None:
        self._dit.shutdown_service()  # Clean MLX shutdown
    
    self._dit = None
    self._llm = None
    self.initialized = False
    
    gc.collect()                      # Force Python garbage collection
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()       # Clear any MPS-side allocations
```

> **Why aggressive teardown?** PyTorch/MLX memory on unified RAM doesn't free immediately after `del`. Explicit `gc.collect()` + `torch.mps.empty_cache()` ensures the 36 GB pool is clear before the next TTS model loads. This is critical for the sequential loading pattern (TTS → Music → unload).

---

## 8. Pipeline Integration

### Sequential Loading Pattern

The pipeline enforces strict sequential model loading to prevent OOM:

```
1. Load TTS (Kokoro or Parler) → Synthesize narration → Unload TTS
2. Load Music Engine (MusicGen or ACE-Step) → Generate music → Unload Music
3. Mix, apply FX, export (CPU-only, no model needed)
```

### Model Routing in `pipeline.py`

The `music_model` parameter (`"musicgen"` or `"acestep"`) controls which engine is instantiated:

```python
if use_acestep:
    from core.acestep_engine import AceStepEngine
    music_engine = AceStepEngine()
else:
    music_engine = self.music  # Existing MusicEngine instance

music_engine.load_model()
music_audio = music_engine.generate(enhanced_prompt, duration, progress_cb=...)
music_engine.unload_model()
```

### Prompt Enhancement Routing

Each model gets its own prompt enhancer:
- **MusicGen:** `_enhance_music_prompt()` — caps at 45 words, adds `no drums/vocals/beatless`
- **ACE-Step:** `_enhance_acestep_prompt()` → delegates to `AceStepEngine._enhance_prompt()` — appends meditation ambient keywords

### Sample Rate Routing

Since MusicGen outputs at 24 kHz and ACE-Step also outputs at 24 kHz (after internal resampling), the pipeline uses the correct `TARGET_SAMPLE_RATE` constant from whichever engine was used:

```python
if use_acestep:
    from core.acestep_engine import TARGET_SAMPLE_RATE as MUSIC_SR
else:
    from core.music_engine import TARGET_SAMPLE_RATE as MUSIC_SR
```

Both resolve to `24000` — the pipeline's standard rate.

---

## 9. Gradio UI Integration

### Background Music Model Dropdown

A new `gr.Dropdown` in the settings column:

```python
music_model_dropdown = gr.Dropdown(
    choices=["MusicGen", "ACE-Step 1.5"],
    value="MusicGen",
    label="Background Music Model",
)
```

### Visibility Behavior

| Generation Mode | Music Dropdown Visible? |
|----------------|------------------------|
| Instrumental Only | ✅ Yes |
| Instrumental + Vocal | ✅ Yes |
| Vocals Only | ❌ Hidden (no music generated) |

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'acestep'"
```bash
pip install git+https://github.com/ace-step/ACE-Step-1.5.git
```

### "ModuleNotFoundError: No module named 'mlx'"
```bash
pip install mlx
# MLX only works on Apple Silicon — verify with:
python -c "import platform; assert platform.machine() == 'arm64', 'Not Apple Silicon'"
```

### OOM / Memory Pressure During Generation
- Ensure TTS model is fully unloaded before ACE-Step loads
- Close other GPU-intensive applications (Safari tabs, video players)
- Consider using `inference_steps=32` instead of 50 to reduce peak memory
- Monitor memory: `sudo memory_pressure` in another terminal

### Generated Audio Has Vocals Despite instrumental=True
- Verify `lyrics="[Instrumental]"` is set in `GenerationParams`
- Ensure the prompt doesn't contain vocal-adjacent terms (`singing`, `choir`, `voice`)

### Audio Quality Issues (Artifacts, Metallic Sound)
- Increase `inference_steps` from 32 to 50 for higher fidelity
- Add `"high fidelity, warm, smooth texture"` to the prompt
- Verify MLX backend is active (`device="mlx"`) — CPU fallback produces lower quality

### Seams or Clicks in Long Tracks
- ACE-Step generates in a single pass (no stitching needed for tracks ≤10 min)
- If duration exceeds the model's single-pass limit, the issue may be in downstream processing — check the mixer's crossfade settings

---

## 11. Quick Reference — Default Values

```python
# Model configuration
DIT_CONFIG = "acestep-v15-sft"           # Full quality DiT
LLM_MODEL = "acestep-5Hz-lm-1.7B"       # 1.7B parameter LM planner
DEVICE = "mlx"                           # Metal GPU via MLX
BACKEND = "mlx"                          # Native Apple Silicon

# Generation parameters
INSTRUMENTAL = True                       # Always, for meditation
LYRICS = "[Instrumental]"                # Token-level vocal suppression
INFERENCE_STEPS = 32                     # Quality/speed balance (max: 50)
THINKING = True                          # Chain-of-Thought planning enabled

# Audio format
NATIVE_SAMPLE_RATE = 48000               # ACE-Step native output
TARGET_SAMPLE_RATE = 24000               # Pipeline standard (matches MusicEngine)
OUTPUT_FORMAT = "mono float32 numpy"     # Pipeline contract
```
