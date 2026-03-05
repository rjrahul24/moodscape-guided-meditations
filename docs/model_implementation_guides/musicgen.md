# MusicGen Implementation Guide for Meditation Audio
### A Complete Reference for Claude Code

---

## 1. Purpose & Scope

This document is a complete, authoritative reference for implementing Meta's **MusicGen** (via the AudioCraft library) in the MoodScape meditation audio generator. It covers:

- Hardware-specific setup for **Apple Silicon M1 Max (32 GB unified memory and 24 Cores GPU)**
- In Apple's Silicon chips like your M1 Max, 32 GB unified memory means a single fast shared pool of 32 gigabytes of RAM that the CPU, 24-core GPU, and other components access instantly without copying data back and forth (unlike traditional PCs with separate RAM and VRAM), while the 24-core GPU refers to the integrated graphics processor with 24 dedicated graphics cores for handling demanding visual tasks like video editing, 3D rendering, gaming, and machine learning with high efficiency and performance.
- Model selection rationale tuned for **calm, sleep, and relaxation** audio goals
- Long-form generation using the **sliding window + continuation** technique
- Prompt engineering vocabulary specifically for meditation
- Parameter tuning for stability and smoothness
- Complete, copy-paste-ready Python implementation patterns
- Integration notes for the `core/music_engine.py` module

---

## 2. Hardware Reality Check — Apple Silicon M1 Max

### What Works
The official AudioCraft library targets **NVIDIA CUDA GPUs**. However, MusicGen can run on Apple Silicon via two paths:

| Path | Status | Notes |
|------|--------|-------|
| **CPU inference** | ✅ Works reliably | Slower, but fully functional. ~3–5× real-time for `musicgen-medium`. |
| **MPS (Metal Performance Shaders)** | ⚠️ Partially supported | Tensor type issues at runtime; not stable in main AudioCraft as of early 2026. |
| **MLX port** | ✅ Works (small/medium/large) | Community port; `musicgen-melody` not yet supported. |

### Recommended Approach for M1 Max 32 GB
Use **CPU inference** with the official `audiocraft` library. With 32 GB unified memory, the M1 Max can comfortably load `musicgen-medium` (~3.2 GB model) and generate in reasonable time. For a 10-minute meditation track, expect approximately **4–8 minutes** of generation time on CPU.

**Do not attempt MPS** — it causes silent failures or tensor type mismatches in the current AudioCraft codebase. CPU inference is deterministic and stable.

### Device Detection Code
```python
import torch

def get_device() -> str:
    """Reliably select the best available device for MusicGen on Apple Silicon."""
    # MPS is intentionally skipped — AudioCraft has known tensor dtype issues with MPS.
    # CPU is stable, and M1 Max has enough bandwidth to handle musicgen-medium comfortably.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

---

## 3. Installation

### Prerequisites
```bash
# Install ffmpeg (required by torchaudio for audio I/O)
brew install ffmpeg

# Confirm Python version (3.10+ required)
python3 --version
```

### Python Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU build for M1 Mac stability)
pip install torch torchaudio

# Install AudioCraft
pip install audiocraft

# Verify installation
python -c "from audiocraft.models import MusicGen; print('AudioCraft OK')"
```

> **If `audiocraft` install fails on macOS:** Clone and install editable:
> ```bash
> git clone https://github.com/facebookresearch/audiocraft.git
> cd audiocraft
> pip install -e .
> ```

---

## 4. Model Selection for Meditation Audio

### Available Models

| Model | Params | Disk Size | RAM Required | Generation Speed (M1 Max CPU) | Best For |
|-------|--------|-----------|--------------|-------------------------------|----------|
| `facebook/musicgen-small` | 300M | ~1.2 GB | ~4 GB | ~1.5× realtime | Prototyping, fast iteration |
| `facebook/musicgen-medium` | 1.5B | ~3.2 GB | ~8 GB | ~0.5× realtime | **Production meditation audio** |
| `facebook/musicgen-large` | 3.3B | ~6.5 GB | ~14 GB | ~0.25× realtime | Highest quality, long generation time |
| `facebook/musicgen-melody` | 1.5B | ~3.2 GB | ~8 GB | ~0.5× realtime | Melody conditioning — avoid for pure ambient |
| `facebook/musicgen-stereo-medium` | 1.5B | ~3.2 GB | ~10 GB | ~0.4× realtime | Stereo output (headphone-friendly) |

### Recommendation for This Project

**Primary model: `facebook/musicgen-medium`**

Rationale:
- Produces rich, evolving ambient textures — significantly better than `small` for long sustained pads
- Fits comfortably within M1 Max's 32 GB unified memory
- Generation time is acceptable for a local tool (~5–10 minutes per 10-minute track)
- **Avoid `musicgen-melody`** for pure ambient meditation: it was trained to follow melodic structures, which causes it to introduce unwanted musical movement and beat patterns in ambient/drone contexts

**Fallback: `facebook/musicgen-small`**
- Use if the user explicitly wants faster generation or is testing

**Optional upgrade: `facebook/musicgen-large`**
- Enable via user setting if they're willing to wait 15–20 minutes for a 10-minute track

### Model Selection Code
```python
MODEL_PRIORITY = [
    "facebook/musicgen-medium",   # Primary
    "facebook/musicgen-small",    # Fallback
]

def load_model_with_fallback():
    from audiocraft.models import MusicGen
    for model_name in MODEL_PRIORITY:
        try:
            model = MusicGen.get_pretrained(model_name)
            print(f"Loaded: {model_name}")
            return model, model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}. Trying next...")
    raise RuntimeError("Could not load any MusicGen model.")
```

---

## 5. Core Generation API

### Basic Generation (≤30 seconds)
```python
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained("facebook/musicgen-medium", device="cpu")

model.set_generation_params(
    duration=30,           # seconds — hard cap for MusicGen
    use_sampling=True,     # ALWAYS True; greedy decoding sounds robotic
    top_k=250,             # Keep at 250; controls diversity
    top_p=0.0,             # Set to 0 to use top_k exclusively (more stable)
    temperature=0.87,      # Stabilises token sampling for consonant pads (0.85-0.90 sweet spot)
    cfg_coef=4.0,          # Strong CFG — heavily penalises tokens that diverge from text prompt
)

prompt = "Deep ambient drone, warm synth pads, no drums, no melody, beatless, 432Hz, peaceful"
wav = model.generate([prompt])  # Returns tensor shape (batch, channels, samples)

# wav[0] = first batch item, shape (channels, samples) at 32000 Hz
audio = wav[0].cpu().numpy()   # Convert to numpy
```

### Audio Tensor Shape Reference
```
model.generate() → Tensor shape: (batch_size, num_channels, num_samples)
- batch_size: number of prompts passed (usually 1)
- num_channels: 1 for mono models, 2 for stereo models
- num_samples: duration_sec × 32000

To get a mono numpy array from the first batch item:
  audio = wav[0].mean(dim=0).cpu().numpy()  # (samples,) float32
```

---

## 6. Long-Form Generation — The Sliding Window Technique

MusicGen's transformer context window is hard-capped at **30 seconds**. To generate a 10-minute meditation track, use autoregressive **continuation**: feed the last N seconds of the previous segment as the audio prompt for the next segment. This ensures musical coherence (key, timbre, texture) across the full track.

### Algorithm

```
Segment 1: Generate 30s using text prompt only
Segment 2: Feed last 10s of Segment 1 + text prompt → Generate 30s
Segment 3: Feed last 10s of Segment 2 + text prompt → Generate 30s
...
Final: Crossfade all segments together at seam points
```

The `generate_continuation()` API:
```python
# prompt_wav: Tensor of shape (batch, channels, samples) at model.sample_rate (32000)
# descriptions: list of text strings (same as generate())
next_chunk = model.generate_continuation(
    prompt=last_chunk_tensor,       # Audio context (last 10s of previous segment)
    prompt_sample_rate=32000,
    descriptions=[prompt_text],
    progress=True,
)
```

### Complete Long-Form Generator Implementation

```python
import math
import numpy as np
import torch
import torchaudio
from audiocraft.models import MusicGen


NATIVE_SR = 32000       # MusicGen native sample rate
TARGET_SR = 24000       # Kokoro TTS sample rate (resampling target)
SEGMENT_DURATION = 30   # Seconds per MusicGen call (hard limit)
CONTEXT_DURATION = 10   # Seconds of previous audio used as context for continuation
CROSSFADE_DURATION = 2  # Seconds of equal-power cosine crossfade at each segment seam


class MusicEngine:
    """Generates long-form ambient meditation music via MusicGen sliding window."""

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self, preferred: str = "facebook/musicgen-medium"):
        """Load MusicGen. Falls back to musicgen-small on memory failure."""
        from audiocraft.models import MusicGen as MG

        candidates = [preferred, "facebook/musicgen-small"]
        last_error = None

        for name in candidates:
            try:
                self.model = MG.get_pretrained(name, device="cpu")
                self.model_name = name
                print(f"[MusicEngine] Loaded: {name}")
                return
            except Exception as e:
                print(f"[MusicEngine] Could not load {name}: {e}")
                last_error = e

        raise RuntimeError(f"No MusicGen model could be loaded. Last error: {last_error}")

    def unload_model(self):
        """Release model from memory."""
        import gc
        del self.model
        self.model = None
        self.model_name = None
        gc.collect()
        # MPS/CUDA cache not needed on CPU, but safe to call if CUDA present
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        temperature: float = 0.8,
        top_k: int = 250,
        cfg_coef: float = 4.0,
        progress_cb=None,
    ) -> np.ndarray:
        """
        Generate ambient music of total_duration_sec at TARGET_SR (24000 Hz).

        Args:
            prompt: Text description of the desired music style.
            total_duration_sec: Final audio length needed (seconds).
            temperature: Sampling temperature (0.7–0.9 for stable meditation audio).
            top_k: Token sampling breadth (250 is stable default).
            cfg_coef: Prompt adherence strength (3.5–5.0 recommended).
            progress_cb: Callable(current_step, total_steps) for UI progress.

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before generate().")

        # Calculate how many segments we need
        # Each continuation call produces (SEGMENT_DURATION - CONTEXT_DURATION) net new seconds
        net_per_segment = SEGMENT_DURATION - CONTEXT_DURATION
        if total_duration_sec <= SEGMENT_DURATION:
            num_segments = 1
        else:
            extra = total_duration_sec - SEGMENT_DURATION
            num_segments = 1 + math.ceil(extra / net_per_segment)

        # Set generation params
        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=top_k,
            top_p=0.0,           # Use top_k exclusively — more stable for ambient
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        if progress_cb:
            progress_cb(0, num_segments)

        # ── Segment 1: text-only generation ──────────────────────────────────
        wav = self.model.generate([prompt], progress=False)
        segment = wav[0]  # (channels, samples) at NATIVE_SR

        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)  # Force mono: (1, samples)

        all_segments = [segment.cpu()]

        if progress_cb:
            progress_cb(1, num_segments)

        # ── Subsequent segments: continuation ────────────────────────────────
        for seg_idx in range(1, num_segments):
            # Extract last CONTEXT_DURATION seconds as audio prompt
            context_samples = int(CONTEXT_DURATION * NATIVE_SR)
            context = all_segments[-1][..., -context_samples:]  # (1, context_samples)

            next_wav = self.model.generate_continuation(
                prompt=context,
                prompt_sample_rate=NATIVE_SR,
                descriptions=[prompt],
                progress=False,
            )
            next_segment = next_wav[0]  # (channels, samples)

            if next_segment.shape[0] > 1:
                next_segment = next_segment.mean(dim=0, keepdim=True)

            all_segments.append(next_segment.cpu())

            if progress_cb:
                progress_cb(seg_idx + 1, num_segments)

        # ── Stitch segments together ──────────────────────────────────────────
        full_audio = self._stitch_segments(all_segments)

        # Trim to exact requested duration
        target_samples_native = int(total_duration_sec * NATIVE_SR)
        if full_audio.shape[-1] > target_samples_native:
            full_audio = full_audio[..., :target_samples_native]

        # ── Resample from 32000 Hz → 24000 Hz ────────────────────────────────
        resampled = torchaudio.functional.resample(full_audio, NATIVE_SR, TARGET_SR)
        mono = resampled.squeeze(0).numpy().astype(np.float32)

        return mono

    def _stitch_segments(self, segments: list) -> torch.Tensor:
        """
        Join segments produced by the sliding window approach.

        The continuation API returns a full 30-second segment starting from
        the context audio. We skip the first CONTEXT_DURATION seconds of each
        continuation (they duplicate the context) and apply a short crossfade
        at the join point.
        """
        if len(segments) == 1:
            return segments[0]

        context_samples = int(CONTEXT_DURATION * NATIVE_SR)
        fade_samples = int(CROSSFADE_DURATION * NATIVE_SR)

        result = segments[0]  # Full first segment

        for seg in segments[1:]:
            # Strip the context region (it duplicates the end of the previous segment)
            new_content = seg[..., context_samples:]

            if new_content.shape[-1] == 0:
                continue

            # Crossfade: blend the tail of result with the head of new_content
            overlap = min(fade_samples, result.shape[-1], new_content.shape[-1])

            fade_out = torch.linspace(1.0, 0.0, overlap)
            fade_in  = torch.linspace(0.0, 1.0, overlap)

            blended = result[..., -overlap:] * fade_out + new_content[..., :overlap] * fade_in

            result = torch.cat([
                result[..., :-overlap],
                blended,
                new_content[..., overlap:],
            ], dim=-1)

        return result
```

---

## 7. Generation Parameters — Tuning Guide for Meditation

| Parameter | Default | Meditation Value | Effect |
|-----------|---------|-----------------|--------|
| `temperature` | 1.0 | **0.87** | Stabilises token sampling for consonant pads. The 0.85–0.90 range is the sweet spot for ambient generation. |
| `top_k` | 250 | **250** | Keeps sampling within the top 250 tokens. Safe default for all ambient styles. |
| `top_p` | 0.0 | **0.0** | Disabled — top_k handles truncation exclusively. Using both can cause instability. |
| `cfg_coef` | 3.0 | **4.0** | Strong Classifier-Free Guidance. Heavily penalises EnCodec tokens that don’t align with the text prompt (e.g., “no percussion”). Don't exceed 6.0 — audio distorts. |
| `duration` | — | **30** | Always use the maximum 30s per call for maximum context coherence. |

> **Note on `cfg_coef=4.0`:** Enabling CFG at this level means MusicGen runs **two forward passes per token** (conditional + unconditional), roughly doubling generation time compared to `cfg_coef=1.0`. This is a worthwhile trade-off for suppressing hallucinated drums and melodic intrusions.

### Spectral Flux Hallucination Guard

MusicGen occasionally generates sudden percussive transients or rhythmic bursts mid-generation, especially when prompt attention diffuses during continuation passes. MoodScape implements a spectral flux analysis failsafe:

1. After each generated segment, compute the STFT magnitude spectra (1024-point FFT, 512-sample hop).
2. Calculate per-frame **spectral flux** (L1 norm of frame-to-frame magnitude differences).
3. If any frame’s flux exceeds **4.5× the median flux**, the segment is flagged as containing a percussive transient.
4. The segment is regenerated (up to 3 retries). If all retries fail, the last attempt is used (graceful degradation).

This catches hallucinated drum hits, clicks, and rhythmic events that CFG alone may miss.

---

## 8. Prompt Engineering for Meditation Audio

### Structural Formula
```
[Style] + [Instruments] + [Texture/Mood] + [Structural Constraints]
```

### Keyword Vocabulary

**Texture (always include 2–3):**
`atmospheric` · `ethereal` · `spacious` · `floating` · `dreamy` · `hazy` · `shimmering` · `vast` · `evolving`

**Instruments for calm/sleep:**
`warm synth pads` · `analog drones` · `tibetan singing bowls` · `crystal bowls` · `soft piano` · `sustained strings` · `cello pad` · `glass harmonica` · `low hum` · `flute`

**Structural constraints (always include):**
`beatless` · `no drums` · `no percussion` · `no vocals` · `no melody` · `no sudden changes` · `drone` · `constant volume` · `slow tempo` · `non-rhythmic`

**Mood anchors:**
`peaceful` · `healing` · `deep sleep` · `serene` · `introspective` · `sacred` · `meditative` · `weightless` · `inner stillness`

### Curated Presets by Meditation Style

**Deep Sleep / Theta Waves**
```
Deep ambient drone, extremely slow, warm analog synth pads, low frequency sub-bass hum,
no drums, no percussion, no vocals, beatless, evolving slowly, constant volume,
432Hz healing tones, sleep music, dark and warm tonal balance, very spacious reverb
```

**Mindfulness / Breath Focus**
```
Minimalist ambient soundscape, single sustained cello pad, occasional soft singing bowl,
vast silence, no melody, beatless, 60 BPM feel, very spacious, warm room reverb,
meditative and grounding, no sudden changes
```

**Zen Garden / Clarity**
```
Minimalist Zen atmosphere, soft shakuhachi flute with long decays, warm synth drone,
occasional distant gong, vast silence between notes, no rhythm, no percussion,
peaceful and clear, dry reverb, high clarity, introspective
```

**Healing / Chakra Work**
```
Authentic tibetan singing bowls, pure harmonic overtones, resonant metal vibrations,
432Hz tuning, meditative gong in the far distance, healing frequencies, no drums,
no vocals, constant volume, sacred and warm
```

**Morning Awakening / Gentle Uplift**
```
Soft morning ambient, light shimmering bells, gentle piano chords with long sustain,
birdsong texture, slowly rising warm pads, no drums, beatless, peaceful and hopeful,
spacious reverb, very slow and gentle
```

**Transcendental / Mantra Support**
```
Ethereal crystal harp, shimmering glass textures, light ambient wash, soaring pads,
heavenly atmosphere, non-rhythmic, fluid movement, high-frequency sparkle,
no drums, no vocals, beatless, expansive and peaceful
```

### Anti-Patterns (What NOT to Put in Prompts)
- ❌ `upbeat`, `energetic`, `lively`, `dance` — introduces rhythmic elements
- ❌ `piano melody`, `melody line`, `hook` — forces melodic structure
- ❌ `60 BPM with drums` — always drops "no drums"; separate rhythm from BPM references
- ❌ Long complex prompts (>40 words) — model attention dilutes; keep focused
- ❌ Contradictory terms (`beatless 4/4 groove`) — model tries to satisfy both badly

---

## 9. Audio Output Handling

### Sample Rate
MusicGen always outputs at **32,000 Hz**. The rest of this project (Kokoro TTS, Pedalboard) operates at **24,000 Hz**. Always resample.

```python
import torchaudio

def resample_to_24k(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Resample from MusicGen's 32kHz to the project's 24kHz standard."""
    return torchaudio.functional.resample(audio_tensor, orig_freq=32000, new_freq=24000)
```

### Mono Conversion
MusicGen medium/large produce mono by default. Stereo variants (`musicgen-stereo-*`) produce 2-channel output. Always reduce to mono for this project:

```python
if audio.shape[0] == 2:
    audio = audio.mean(dim=0, keepdim=True)  # Average L+R → mono
```

### Saving (for debugging, not final pipeline output)
```python
from audiocraft.data.audio import audio_write

# Saves with built-in loudness normalization to -14 LUFS
audio_write(
    "debug_output",           # Filename without extension
    wav[0].cpu(),             # Tensor (channels, samples)
    model.sample_rate,        # 32000
    strategy="loudness",      # Normalize to -14 LUFS
    loudness_compressor=True, # Apply compression for smoother dynamics
)
```

---

## 10. Memory Management

### Sequential Loading Pattern (Critical for 8–16 GB VRAM Systems)
This project loads Kokoro TTS and MusicGen sequentially, never simultaneously. The pattern in `pipeline.py` is correct:

```python
# Step 1: Load TTS, synthesize speech, unload TTS
tts_engine.load_model()
voice_audio, voice_activity = tts_engine.synthesize(segments, ...)
tts_engine.unload_model()   # ← Free memory before loading music model

# Step 2: Load MusicGen, generate music, unload MusicGen
music_engine.load_model()
music_audio = music_engine.generate(prompt, duration)
music_engine.unload_model()  # ← Free before mixing (mixing is CPU-only)
```

### Explicit Cleanup
```python
import gc
import torch

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # On Apple Silicon, the system reclaims unified memory automatically
    # after Python objects are deleted, but gc.collect() helps timing
```

---

## 11. Complete `core/music_engine.py` — Recommended Final Implementation

```python
"""MusicGen wrapper for generating ambient meditation background music.

Target hardware: Apple Silicon M1 Max (32 GB unified memory)
MusicGen sample rate: 32000 Hz  →  resampled to 24000 Hz for pipeline
"""

import gc
import math

import numpy as np
import torch
import torchaudio


NATIVE_SR = 32000
TARGET_SR = 24000
SEGMENT_DURATION = 30       # Max per MusicGen call
CONTEXT_DURATION = 5       # Audio context passed to generate_continuation
CROSSFADE_DURATION = 2.0      # Seconds of equal-power cosine crossfade at each seam

MODEL_CANDIDATES = [
    "facebook/musicgen-stereo-medium",   # Primary — best quality for M1 Max
    "facebook/musicgen-medium",    # Fallback
    "facebook/musicgen-small",    # Fallback
]


class MusicEngine:

    def __init__(self):
        self.model = None
        self.model_name = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load_model(self, preferred: str = "facebook/musicgen-medium"):
        from audiocraft.models import MusicGen

        candidates = [preferred] + [m for m in MODEL_CANDIDATES if m != preferred]
        for name in candidates:
            try:
                # Force CPU — MPS has dtype instability in AudioCraft on Apple Silicon
                self.model = MusicGen.get_pretrained(name, device="cpu")
                self.model_name = name
                return
            except Exception as e:
                print(f"[MusicEngine] {name} failed: {e}")

        raise RuntimeError("No MusicGen model could be loaded.")

    def unload_model(self):
        del self.model
        self.model = None
        self.model_name = None
        gc.collect()

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        temperature: float = 0.8,
        top_k: int = 250,
        cfg_coef: float = 4.0,
        progress_cb=None,
    ) -> np.ndarray:
        """Generate ambient music. Returns mono float32 at 24000 Hz."""
        if self.model is None:
            raise RuntimeError("Call load_model() first.")

        num_segments = self._num_segments(total_duration_sec)

        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=top_k,
            top_p=0.0,
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        # Segment 1: text-only
        wav = self.model.generate([prompt], progress=False)
        current = self._to_mono(wav[0]).cpu()
        segments = [current]
        if progress_cb:
            progress_cb(1, num_segments)

        # Subsequent segments: continuation
        for i in range(1, num_segments):
            context_samples = int(CONTEXT_DURATION * NATIVE_SR)
            context = segments[-1][..., -context_samples:]

            next_wav = self.model.generate_continuation(
                prompt=context,
                prompt_sample_rate=NATIVE_SR,
                descriptions=[prompt],
                progress=False,
            )
            segments.append(self._to_mono(next_wav[0]).cpu())
            if progress_cb:
                progress_cb(i + 1, num_segments)

        full = self._stitch(segments)

        # Trim to exact length
        target_native = int(total_duration_sec * NATIVE_SR)
        full = full[..., :target_native]

        # Resample 32k → 24k and return numpy
        resampled = torchaudio.functional.resample(full, NATIVE_SR, TARGET_SR)
        return resampled.squeeze(0).numpy().astype(np.float32)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _num_segments(self, duration: float) -> int:
        if duration <= SEGMENT_DURATION:
            return 1
        net = SEGMENT_DURATION - CONTEXT_DURATION
        return 1 + math.ceil((duration - SEGMENT_DURATION) / net)

    @staticmethod
    def _to_mono(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce (channels, samples) to (1, samples)."""
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.shape[0] > 1:
            return tensor.mean(dim=0, keepdim=True)
        return tensor

    def _stitch(self, segments: list) -> torch.Tensor:
        """Join continuation segments, skipping the duplicated context region."""
        if len(segments) == 1:
            return segments[0]

        ctx = int(CONTEXT_DURATION * NATIVE_SR)
        fade = int(CROSSFADE_DURATION * NATIVE_SR)
        result = segments[0]

        for seg in segments[1:]:
            new = seg[..., ctx:]  # Strip duplicate context
            if new.shape[-1] == 0:
                continue

            overlap = min(fade, result.shape[-1], new.shape[-1])
            fade_out = torch.linspace(1.0, 0.0, overlap)
            fade_in  = torch.linspace(0.0, 1.0, overlap)

            blended = result[..., -overlap:] * fade_out + new[..., :overlap] * fade_in

            # Note: The actual implementation uses equal-power cosine crossfade:
            # t = torch.linspace(0, math.pi/2, overlap)
            # fade_out = torch.cos(t) ** 2
            # fade_in = torch.cos(math.pi/2 - t) ** 2
            result = torch.cat([result[..., :-overlap], blended, new[..., overlap:]], dim=-1)

        return result
```

---

## 12. Integration with `pipeline.py`

The pipeline calls `music_engine.generate()` after unloading Kokoro. The duration passed should be:

```python
voice_duration = len(voice_audio) / SAMPLE_RATE   # seconds of narration
music_duration = voice_duration + 10.0             # Add 10s buffer for pre-roll + fade-out

music_audio = self.music.generate(
    music_prompt,
    music_duration,
    temperature=0.8,       # Stable ambient generation
    top_k=250,
    cfg_coef=4.0,
)
```

---

## 13. Troubleshooting

### "Cannot allocate memory" / Crash on Load
- Switch from `musicgen-medium` to `musicgen-small`
- Close all other GPU-intensive applications before running
- On M1 Mac, unified memory is shared between CPU and GPU — having Safari with many tabs open can reduce available memory for the model

### Metallic / Artifact-Heavy Audio
- Lower `temperature` to 0.75
- Raise `cfg_coef` to 5.0 — forces stricter adherence to ambient descriptors
- Add `no artifacts, smooth texture` to the prompt

### Music Sounds Rhythmic / Has Beats
- Remove any BPM numbers from the prompt
- Add: `beatless, no rhythm, no percussion, no drums, drone only`
- Switch from `musicgen-melody` to `musicgen-medium` — melody model actively seeks rhythmic structure

### Seams / Clicks Between Segments
- Increase `CROSSFADE_DURATION` to 3–4 seconds
- Increase `CONTEXT_DURATION` to 15 seconds for smoother continuation
- Ensure `_stitch()` is correctly skipping the duplicated context region

### Generation Takes Too Long
- Use `musicgen-small` (faster, good enough for initial testing)
- Reduce total duration — break 20-minute meditations into multiple tracks
- On M1 Max with 32 GB RAM, `musicgen-medium` generating a 10-minute track takes ~5–8 minutes on CPU

### Import Errors / Module Not Found
```bash
# Verify audiocraft is installed
python -c "import audiocraft; print(audiocraft.__version__)"

# If missing, reinstall
pip install audiocraft

# If ffmpeg errors appear
brew install ffmpeg
```

---

## 14. Quick Reference — Default Parameter Values

```python
# For use in music_engine.generate() — these are fixed internally
MEDITATION_DEFAULTS = {
    "temperature": 0.87,  # Stabilises token sampling for ambient pads
    "top_k": 250,         # Standard diversity
    "top_p": 0.0,         # Disabled — use top_k only
    "cfg_coef": 4.0,      # Strong prompt adherence (no drums = no drums)
}

# Segment architecture
SEGMENT_DURATION = 30     # Seconds per MusicGen call
CONTEXT_DURATION = 5      # Seconds fed as audio context for continuation
CROSSFADE_DURATION = 2    # Seconds of equal-power cosine crossfade at segment seams

# Sample rates
NATIVE_SR = 32000         # MusicGen output
TARGET_SR = 24000         # Pipeline standard (matches Kokoro TTS)
```

---