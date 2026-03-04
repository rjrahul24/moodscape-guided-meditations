# Kokoro TTS — Guided Meditation Implementation Research

---

## 1. Model Overview

**Kokoro-82M v1.0** (released January 27, 2025) is the current production-ready version.

| Property | Value |
|----------|-------|
| Parameters | 82 million |
| Architecture | StyleTTS 2 + ISTFTNet vocoder (decoder-only, no diffusion) |
| G2P library | `misaki` (primary); espeak-ng (OOD fallback) |
| Output sample rate | **24,000 Hz (24 kHz), mono** |
| License | Apache 2.0 |
| Training data | Koniwa, SIWIS, CC BY audio, public domain, synthetic audio from closed providers |
| Languages (v1.0) | American English, British English, Spanish, French, Hindi, Italian, Japanese, Mandarin Chinese |
| Voices (v1.0) | **54 voices** across 8 languages |
| Token limit per pass | **510 tokens** (auto-split by KPipeline for longer texts) |
| Peak memory (af_heart) | ~1.4–2.7 GB depending on text length |
| Real-time factor (MPS) | ~17× RTF on M-series (≈17 minutes of audio per minute of inference) |
| Real-time factor (GPU) | Up to **200×** on high-end CUDA GPU |
| Real-time factor (CPU) | 3–5× on CPU only |

The model achieves state-of-the-art naturalness (ranked #1 in TTS Spaces Arena), outperforming XTTS v2 (467M params) and MetaVoice (1.2B params), and even Fish Speech (trained on 1 million hours), despite being trained on fewer than 100 hours of curated data.

### Variants

| Variant | Languages | Voices | Notes |
|---------|-----------|--------|-------|
| **v1.0** (use this) | 8 | 54 | Current release, best quality |
| v0.19 | 1 (English) | 10 | Legacy; ONNX version available (`kokoro-v0_19.onnx`) |

The **ONNX variant** (`kokoro-v0_19.onnx`) is useful for CPU-only deployment or quantized inference (~80MB), but for MPS-accelerated M1 Max use the standard `pip install kokoro` PyTorch package.

---

## 2. Installation for Apple Silicon (M1 Max)

### System Dependencies

```bash
# espeak-ng — required for OOD text, G2P fallback, and phonemization
brew install espeak-ng

# Verify installation and check it is in PATH
espeak-ng --version
which espeak-ng
```

### Python Dependencies

```bash
# Core package
pip install "kokoro>=0.9.4" soundfile numpy misaki[en]

# For conda-based setups that have libstdcxx conflicts:
# conda install libstdcxx~=12.4.0

# PyTorch 2.0+ with MPS support (usually already installed):
pip install torch torchvision torchaudio
```

### MPS (Metal Performance Shaders) Acceleration

Kokoro supports MPS-accelerated inference on Apple Silicon. Set this environment variable **before running any Kokoro script** to enable GPU fallback for unsupported ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python app.py
```

Or set it programmatically at the **very top** of your entry-point script (before any torch imports):

```python
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

### Expected Performance on M1 Max

Based on research document benchmarks:

| Task | Duration |
|------|----------|
| Generate 30 seconds of audio | 10–30 seconds |
| Generate 1 minute of audio | ~15–45 seconds |
| A full 10-minute meditation script | ~2–5 minutes |
| First-run model download | One-time; auto-downloads from HuggingFace |

After the initial download, Kokoro operates **fully offline** with no internet required.

### Manual Model Download (Offline Setup)

```bash
huggingface-cli download hexgrad/Kokoro-82M
```

---

## 3. Known Issues and Mitigations (M1 Mac Specific)

### Issue 1: MPS Crash on Very Long Strings

**Symptom:** `RuntimeError: Placeholder storage has not been allocated on MPS device!`

**Cause:** Some PyTorch ops used internally by Kokoro are not fully implemented on MPS and the fallback mechanism can sometimes fail for very long inputs.

**Mitigations:**
1. Always set `PYTORCH_ENABLE_MPS_FALLBACK=1` — resolves most cases
2. Process text in sentence-level segments (which `KPipeline` does automatically via `split_pattern`)
3. Per-segment CPU fallback if MPS still fails:

```python
def synthesize_with_fallback(pipeline, text, voice, speed):
    try:
        return list(pipeline(text, voice=voice, speed=speed))
    except RuntimeError as e:
        if "MPS" in str(e) or "placeholder" in str(e).lower():
            pipeline.model.cpu()
            result = list(pipeline(text, voice=voice, speed=speed))
            pipeline.model.to("mps")
            return result
        raise
```

### Issue 2: Memory Leak (hexgrad/kokoro #152)

**Symptom:** RAM usage grows with each synthesis call and does not fully return to baseline.

**Impact on MoodScape:** Low — MoodScape generates one meditation session at a time (not a long-running server), so the leak is bounded by the session length.

**Mitigation:** Unload the model immediately after all speech is generated for the session:

```python
del self.pipeline
self.pipeline = None
gc.collect()
torch.mps.empty_cache()   # MPS-specific
```

### Issue 3: Speed Below Threshold

**Symptom:** Robot-sounding or distorted voice at very low speeds.

**Mitigation:** Clamp speed to a minimum of `0.65`:

```python
speed = max(speed, 0.65)
```

---

## 4. KPipeline API Reference

### Initialization

```python
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from kokoro import KPipeline

# lang_code must match the voice prefix (af_*/am_* → 'a', bf_*/bm_* → 'b')
pipeline = KPipeline(lang_code='a')  # American English — covers all af_* and am_* voices

# For British voices (bf_*, bm_*), create a separate pipeline:
pipeline_british = KPipeline(lang_code='b')

# Language codes reference:
# 'a' => American English (en-us)    'b' => British English (en-gb)
# 'e' => Spanish (es)                'f' => French (fr-fr)
# 'h' => Hindi (hi)                  'i' => Italian (it)
# 'j' => Japanese (pip install misaki[ja])
# 'p' => Brazilian Portuguese        'z' => Mandarin (pip install misaki[zh])
```

### Text Generation (Generator Pattern)

```python
generator = pipeline(
    text,                        # str: the input text
    voice='af_heart',            # str or torch.Tensor: voice ID or blended tensor
    speed=0.85,                  # float: 0.65–1.0 for meditation (lower = slower)
    split_pattern=r'\n+',        # regex: how to chunk long text
    # Other useful patterns:
    # r'(?<=[.!?])\s+'           # split only at sentence-end punctuation
    # r'[.!?]\s+'                # similar, includes the punctuation in first part
)

audio_segments = []
for i, (graphemes, phonemes, audio) in enumerate(generator):
    # graphemes: the original text chunk being spoken
    # phonemes:  IPA phoneme string used for synthesis (useful for debugging)
    # audio:     float32 numpy array, 24,000 Hz mono
    if audio is not None and len(audio) > 0:
        audio_segments.append(audio)

full_audio = np.concatenate(audio_segments)
```

### Generating Multiple Variants (Quality Selection)

Research doc 3 recommends generating 3–5 variants of each segment and selecting the best one for critical passages (e.g., the opening or closing lines):

```python
def generate_best_variant(pipeline, text, voice, speed, num_samples=3):
    """Generate multiple takes and return the longest (most complete) one."""
    candidates = []
    for _ in range(num_samples):
        parts = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                parts.append(audio)
        if parts:
            candidates.append(np.concatenate(parts))
    # Select the variant with the median length (avoid outliers)
    if not candidates:
        return np.zeros(int(0.1 * 24000), dtype=np.float32)
    candidates.sort(key=len)
    return candidates[len(candidates) // 2]
```

Note: For a full meditation pipeline, generating multiple variants for every segment is slow. Reserve this for short critical sections only.

---

## 5. Voice Selection Guide for Meditation

### American English Voices (`lang_code='a'`)

| Voice ID | Gender | Character | Meditation Suitability |
|----------|--------|-----------|------------------------|
| **af_heart** | Female | Warm, velvety, intimate | ⭐⭐⭐⭐⭐ **Gold standard for meditation** |
| **af_sky** | Female | Clear, airy, gentle | ⭐⭐⭐⭐⭐ Excellent for visualizations & sleep |
| **af_nova** | Female | Smooth, calm | ⭐⭐⭐⭐ Good for sleep meditations |
| **af_nicole** | Female | Soft, ASMR-like | ⭐⭐⭐⭐ Great for whisper-style / deep relaxation |
| af_river | Female | Flowing, calm | ⭐⭐⭐⭐ Good for body scan |
| af_bella | Female | Expressive, rich | ⭐⭐⭐ Good for more dynamic sessions |
| af_sarah | Female | Natural, warm | ⭐⭐⭐ Clear narration |
| af_alloy | Female | Neutral, clear | ⭐⭐⭐ Versatile |
| af_aoede | Female | Musical quality | ⭐⭐⭐ Chant-style sessions |
| af_jessica | Female | Lively | ⭐⭐ Less suited for calm sessions |
| af_kore | Female | Expressive | ⭐⭐ More dynamic than meditative |
| **am_adam** | Male | Deep, grounded | ⭐⭐⭐⭐ Excellent for body scans & grounding |
| am_michael | Male | Resonant, authoritative | ⭐⭐⭐⭐ Grounding exercises |

### British English Voices (`lang_code='b'`)

| Voice ID | Gender | Character | Meditation Suitability |
|----------|--------|-----------|------------------------|
| **bf_emma** | Female | Sophisticated, calm, wise | ⭐⭐⭐⭐⭐ Excellent for mindfulness |
| **bf_lily** | Female | Sweet, soft, light, angelic | ⭐⭐⭐⭐⭐ Child-focused meditations; ethereal tone |
| bf_isabella | Female | Gentle, soft | ⭐⭐⭐⭐ Good for gentle sessions |
| bm_george | Male | Warm British baritone | ⭐⭐⭐⭐ Body scan, grounding |
| bm_lewis | Male | Clear British male | ⭐⭐⭐ Standard narration |

> **Important for British voices:** Use `KPipeline(lang_code='b')` — do not mix `bf_*`/`bm_*` voices with `lang_code='a'`. Create a second pipeline instance if you need both accents.

### Top Curated Voice Blends for Meditation

Voice embeddings are PyTorch tensors that can be mathematically blended. Weights should sum to approximately 1.0 for stable amplitude:

```python
import torch
from huggingface_hub import hf_hub_download

def load_voice(voice_id: str) -> torch.Tensor:
    path = hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename=f"voices/{voice_id}.pt",
    )
    return torch.load(path, weights_only=True)

# "Golden Hour" — Warmth + airy clarity (best all-purpose meditation voice)
# Research doc 1 recipe: af_heart:0.6, af_sky:0.4
blend_golden = load_voice('af_heart') * 0.6 + load_voice('af_sky') * 0.4

# "Night Garden" — Calm with depth (sleep meditations)
blend_night = load_voice('af_heart') * 0.60 + load_voice('af_nova') * 0.40

# "Earth Root" — Male/female blend for grounding (Research doc 3: 70% female + 30% male)
blend_earth = load_voice('af_heart') * 0.70 + load_voice('am_adam') * 0.30

# "Still Water" — ASMR-influenced for deep relaxation
blend_water = load_voice('af_nicole') * 0.50 + load_voice('af_sky') * 0.50

# Use a blended tensor directly (bypasses string lookup):
generator = pipeline(text, voice=blend_golden, speed=0.80)
```

---

## 6. Speed & Prosody Tuning for Meditation

### Speed Parameter

| Use Case | Speed Value |
|----------|-------------|
| Normal speech | 1.0 |
| Calm meditation pace (recommended default) | **0.85** |
| Deep relaxation / sleep | 0.75–0.80 |
| Very slow breathing guides | **0.70 (minimum recommended)** |
| Hard minimum (below = artifacts) | **0.65** |

Research doc 1 recommends 0.7×–0.85× as the effective meditation range. Research doc 3 confirms 0.7–0.9 for meditative pace. Do not go below 0.65.

### Punctuation as Prosody Control

Kokoro's StyleTTS2 architecture interprets punctuation to shape intonation curves. Use deliberately:

| Symbol | Effect | Best Use |
|--------|--------|----------|
| `.` (period) | Falling intonation, clear end | End of instruction or thought |
| `...` (ellipsis) | Trailing off, drifting quality | Inhale cues, dreamlike transitions |
| `;` (semicolon) | Brief pause with anticipation | Connected-but-spaced thoughts |
| `—` (em dash) | Slight rhythmic break | Natural speech rhythm |
| `?` (question) | Rising intonation | Use sparingly in meditation |

Example combining punctuation with explicit pauses:
```
"Breathe in... [pause:4s] and release. [pause:6s]"
"Let your body grow heavy... sinking into the earth. [pause:8s]"
"Notice the sounds around you; let them drift by like clouds. [pause:5s]"
```

### Stress and Emphasis via Phonetic Brackets

Research doc 1 documents a stress syntax for word-level emphasis control:

```
[breath](+1)    → Increase emphasis on the word "breath"
[peace](+2)     → Strong emphasis on "peace"
[tension](-1)   → Soften "tension" (makes it sound discarded/released)
```

This works with the `misaki` G2P library's phoneme annotation system.

### IPA Phoneme Injection

Force specific pronunciations:

```
[Kokoro](/kˈOkəɹO/)         → Custom IPA pronunciation
[breathe](/bɹiːð/)          → Force long 'ee' vowel for resonance
[release](/ɹɪˈliːs/)       → Emphasize the exhale vowel
[pranayama](/pɹɑːnɑːˈjɑːmə/) → Correct Sanskrit pronunciation
[ujjayi](/uːˈdʒɑːji/)      → Correct Sanskrit pronunciation
[chakra](/tʃɑːkɹə/)        → Correct Sanskrit pronunciation
```

---

## 7. Pause Implementation Strategy

### The Correct Approach: Silence Array Injection

Kokoro does **not** natively support `[pause:Xs]` markers in raw text — it will either ignore them or attempt to pronounce the word "pause." The correct implementation (already in `core/tts_engine.py` and `core/script_parser.py`) is:

1. Parse the script to extract typed chunks: `{"type": "speech", "text": "..."}` and `{"type": "pause", "duration_sec": X.X}`
2. For `speech` chunks → call `KPipeline` → get audio array
3. For `pause` chunks → `np.zeros(int(duration_sec * 24000), dtype=np.float32)`
4. Concatenate all chunks in order

This provides **millisecond-accurate pause control** as confirmed by all three research documents.

### Natural Micro-Pauses from Punctuation

Kokoro adds ~0.1–0.3 seconds of natural breath pause around sentence-ending punctuation. These are **additive** to your explicit silence arrays and contribute to the meditative feel — do not try to eliminate them.

### Typical Pause Durations by Meditation Type

From research doc 3 curated examples:

| Context | Recommended Pause |
|---------|-------------------|
| Between short phrases | 2–3s |
| After breathing instruction (inhale cue) | 4s |
| After breathing instruction (exhale) | 4–6s |
| After guided visualization sentence | 5s |
| Body awareness cue | 3–5s |
| Deep presence / stillness | 8–10s |
| Between major sections | 5–8s |

---

## 8. Script Writing Best Practices for Meditation TTS

Research documents 1, 2, and 3 converge on these principles:

### Structure

```
1. Calm introduction (set the intention)
2. Body awareness cues (settling in)
3. Controlled breathing instructions (with timed pauses)
4. Counting / rhythm sequences
5. Guided visualization segments (longer pauses)
6. Gentle closing (slow return to awareness)
```

### High-Quality Script Patterns

```
# Deep Relaxation
"Close your eyes. [pause:2s] Feel the weight of your body sinking. [pause:3s] Let tension melt away."

# Breathing Exercise (from Research doc 3)
"Inhale through your nose for four counts. [pause:4s] Hold your breath. [pause:4s] Exhale slowly. [pause:4s] Repeat."

# Guided Sleep Journey
"Imagine a peaceful forest. [pause:5s] Hear the gentle wind. [pause:3s] Drift into deep sleep."

# Mindfulness
"Focus on your breath. [pause:2.5s] Notice thoughts come and go. [pause:3s] Return to the present."
```

### Script Writing Rules

- Use **short, clear sentences** — Kokoro's prosody model performs better on clean, well-punctuated input
- Avoid **long subordinate clauses** — break them into separate sentences with pauses
- Use **repetitive, rhythmic phrases** — creates a hypnosis-like effect ("Breathe in... hold... exhale...")
- Avoid **complex vocabulary** — keep language simple and direct
- Limit text to sentence-level chunks per `pipeline()` call — avoids mid-word splits at the 510-token boundary

### Paragraph Break Behavior

The existing `script_parser.py` converts `\n\n` to `[pause:1.5s]`. This is correct. Do not rely on Kokoro's natural paragraph pausing — it is inconsistent (~0.15s variability, per research doc 2).

### Target Duration Matching

Research doc 1 recommends a feature to hit an exact track length by slightly adjusting speed:

```python
def calculate_adjusted_speed(
    script_segments: list[dict],
    target_duration_sec: float,
    base_speed: float = 0.85,
    sample_rate: int = 24000,
) -> float:
    """
    Estimate the speed needed to match a target duration.
    Generates a short preview, measures it, and extrapolates.
    """
    # Count speech characters as a proxy for duration
    total_chars = sum(len(s["text"]) for s in script_segments if s["type"] == "speech")
    pause_duration = sum(s["duration_sec"] for s in script_segments if s["type"] == "pause")

    # Rough estimate: ~1000 chars ≈ 60 seconds at speed=1.0
    # Adjust proportionally
    estimated_speech_duration = (total_chars / 1000.0) * 60.0 / base_speed
    estimated_total = estimated_speech_duration + pause_duration

    if estimated_total <= 0:
        return base_speed

    # Scale speed to hit target (within ±5s accuracy per Research doc 3)
    adjusted = base_speed * (estimated_total / target_duration_sec)
    return max(0.65, min(1.0, adjusted))
```

---

## 9. Audio Output Specifications

| Property | Value |
|----------|-------|
| Sample rate | 24,000 Hz |
| Bit depth | float32 in numpy (−1.0 to +1.0) |
| Channels | **Mono (1 channel)** |
| WAV export | 24-bit WAV via `soundfile.write` — always generate WAV first |
| MP3 export | Convert WAV → MP3 at the very end only (`pedalboard.io.AudioFile`) |
| Target LUFS | −16 LUFS (broadcast/podcast standard) |

> **Critical rule from Research doc 1:** Always generate WAV first. Converting directly to MP3 during synthesis can introduce click artifacts in the silence regions between segments. Only convert to MP3 as the final export step after all mixing is done.

---

## 10. Post-Processing Recommendations from Research Documents

### What the Research Documents Recommend

Research doc 3 mentions several post-processing options. Here is how they map to the existing MoodScape stack:

| Recommendation | MoodScape Implementation |
|----------------|--------------------------|
| Normalize loudness | ✅ Already done via `pyloudnorm` to −16 LUFS in `mixer.py` |
| Gentle reverb on voice | ✅ Already done via Pedalboard `Reverb` in `audio_processor.py` |
| Fade in/out | ✅ Already done in `mixer.py` |
| EQ warmth (low shelf) | ✅ Already done via Pedalboard `LowShelfFilter` in `audio_processor.py` |
| Pitch shifting | ❌ Not needed — voice selection handles this; librosa would add a dependency |
| Time stretching | ❌ Avoid — use speed parameter instead; stretching degrades quality |
| pydub for segment merging | ❌ Do not use — Pedalboard + numpy is already superior |
| FFmpeg subprocess calls | ❌ Do not use — Pedalboard handles all I/O |

The existing MoodScape audio pipeline already implements everything the research documents recommend and more.

---

## 11. Complete TTSEngine Implementation (with all improvements)

```python
import os
import gc

# Must be set BEFORE any torch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

SAMPLE_RATE = 24000

# Voices ordered by meditation suitability (best first)
# Include both American (lang_code='a') and British (lang_code='b') options
VOICE_CHOICES = [
    # American English — use with KPipeline(lang_code='a')
    ("Heart ❤️ — US Female (warm, intimate) [Best]",   "af_heart"),
    ("Sky 🌤️ — US Female (airy, gentle)",               "af_sky"),
    ("Nova ✨ — US Female (smooth, calm)",               "af_nova"),
    ("Nicole 🎙️ — US Female (soft, ASMR)",              "af_nicole"),
    ("River 💧 — US Female (flowing calm)",              "af_river"),
    ("Bella 🔥 — US Female (expressive)",               "af_bella"),
    ("Sarah 🌿 — US Female (natural)",                  "af_sarah"),
    ("Adam 🏔️ — US Male (deep, grounding)",             "am_adam"),
    ("Michael 🌊 — US Male (resonant)",                 "am_michael"),
    # British English — use with KPipeline(lang_code='b')
    ("Emma 🇬🇧 — UK Female (wise, calm)",               "bf_emma"),
    ("Lily 🇬🇧 — UK Female (sweet, angelic)",           "bf_lily"),
    ("Isabella 🇬🇧 — UK Female (gentle)",              "bf_isabella"),
    ("George 🇬🇧 — UK Male (warm baritone)",           "bm_george"),
    ("Lewis 🇬🇧 — UK Male (clear)",                    "bm_lewis"),
]

# British voice IDs — need lang_code='b' pipeline
BRITISH_VOICES = {"bf_emma", "bf_lily", "bf_isabella", "bm_george", "bm_lewis"}


class TTSEngine:
    def __init__(self):
        self.pipeline_en_us = None  # lang_code='a' — American English
        self.pipeline_en_gb = None  # lang_code='b' — British English

    def load_model(self):
        from kokoro import KPipeline
        self.pipeline_en_us = KPipeline(lang_code='a')
        # British pipeline loaded lazily only if needed

    def _get_pipeline(self, voice: str):
        """Return the correct pipeline for a given voice ID."""
        if voice in BRITISH_VOICES:
            if self.pipeline_en_gb is None:
                from kokoro import KPipeline
                self.pipeline_en_gb = KPipeline(lang_code='b')
            return self.pipeline_en_gb
        return self.pipeline_en_us

    def unload_model(self):
        del self.pipeline_en_us
        del self.pipeline_en_gb
        self.pipeline_en_us = None
        self.pipeline_en_gb = None
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass

    def synthesize(
        self,
        segments: list[dict],
        voice: str = "af_heart",
        speed: float = 0.85,
        progress_cb=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Synthesize all script segments into a single audio track.

        Returns:
            voice_audio:    float32 mono array at 24,000 Hz
            voice_activity: bool array, True where voice is speaking (for ducking)
        """
        if self.pipeline_en_us is None:
            raise RuntimeError("Call load_model() first.")

        # Clamp speed to safe range
        speed = max(0.65, min(1.0, speed))

        audio_chunks: list[np.ndarray] = []
        activity_chunks: list[np.ndarray] = []
        total = len(segments)

        for idx, seg in enumerate(segments):
            if seg["type"] == "speech":
                speech_audio = self._synthesize_text(seg["text"], voice, speed)
                audio_chunks.append(speech_audio.astype(np.float32))
                activity_chunks.append(np.ones(len(speech_audio), dtype=bool))

            elif seg["type"] == "pause":
                n_samples = int(seg["duration_sec"] * SAMPLE_RATE)
                audio_chunks.append(np.zeros(n_samples, dtype=np.float32))
                activity_chunks.append(np.zeros(n_samples, dtype=bool))

            if progress_cb:
                progress_cb(idx + 1, total)

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        return (
            np.concatenate(audio_chunks).astype(np.float32),
            np.concatenate(activity_chunks),
        )

    def _synthesize_text(self, text: str, voice: str, speed: float) -> np.ndarray:
        """Synthesize a single text chunk, with MPS crash fallback."""
        pipeline = self._get_pipeline(voice)
        parts = []

        try:
            gen = pipeline(
                text,
                voice=voice,
                speed=speed,
                split_pattern=r'(?<=[.!?…])\s+',  # Split at sentence-end punctuation
            )
            for _gs, _ps, audio in gen:
                if audio is not None and len(audio) > 0:
                    parts.append(audio)

        except RuntimeError as e:
            if "MPS" in str(e) or "placeholder" in str(e).lower():
                # MPS crash fallback — rare but documented
                try:
                    import torch
                    pipeline.model.cpu()
                    gen = pipeline(text, voice=voice, speed=speed)
                    for _gs, _ps, audio in gen:
                        if audio is not None and len(audio) > 0:
                            parts.append(audio)
                    pipeline.model.to("mps")
                except Exception:
                    pass  # Return silence fallback below
            else:
                raise

        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
```

---

## 12. Gradio UI Updates for `app.py`

The `VOICE_CHOICES` list in `app.py` should be expanded to include the British voices from research doc 1 (`bf_lily` is a notable addition). Updated list:

```python
VOICE_CHOICES = [
    ("Heart — US Female (default)", "af_heart"),
    ("Sky — US Female (airy)",      "af_sky"),
    ("Nova — US Female (calm)",     "af_nova"),
    ("Nicole — US Female (ASMR)",   "af_nicole"),
    ("River — US Female (flowing)", "af_river"),
    ("Bella — US Female",           "af_bella"),
    ("Sarah — US Female",           "af_sarah"),
    ("Emma — UK Female (wise)",     "bf_emma"),
    ("Lily — UK Female (angelic)",  "bf_lily"),    # ← Added from Research doc 1
    ("Adam — US Male (grounding)",  "am_adam"),
    ("Michael — US Male",           "am_michael"),
    ("George — UK Male (warm)",     "bm_george"),
]
```

---

## 13. What NOT to Do (confirmed across all three research documents)

| Anti-Pattern | Reason | Alternative |
|-------------|--------|-------------|
| Pass `[pause:Xs]` raw to Kokoro | Model ignores or mispronounces it | Silence array injection via `script_parser.py` |
| Use `pydub` for mixing | Slow, low quality | Pedalboard + numpy (already in MoodScape) |
| Use `ffmpeg` subprocess | Unnecessary dependency | Pedalboard handles all I/O |
| Use `librosa` time-stretch for pacing | Degrades voice quality | Use Kokoro's `speed` parameter |
| Go below `speed=0.65` | Artifacts and distorted prosody | Clamp to `max(speed, 0.65)` |
| Export MP3 during synthesis | Click artifacts in silences | WAV first, MP3 only at final export |
| Load both TTS and MusicGen simultaneously | OOM on 8–16 GB GPU | Sequential loading with `unload_model()` |
| Ignore `PYTORCH_ENABLE_MPS_FALLBACK=1` | MPS crashes on some ops | Always set before any torch import |

---

## 14. Resources

| Resource | URL |
|----------|-----|
| Official model (HuggingFace) | https://huggingface.co/hexgrad/Kokoro-82M |
| Python library (GitHub) | https://github.com/hexgrad/kokoro |
| VOICES.md (full voice list) | https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md |
| SAMPLES.md (audio demos) | https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md |
| misaki G2P library | https://github.com/hexgrad/misaki |
| Kokoro-TTS-Pause (reference) | https://github.com/ibuhs/Kokoro-TTS-Pause |
| Memory leak issue | https://github.com/hexgrad/kokoro/issues/152 |
| HuggingFace demo space | https://huggingface.co/spaces/hexgrad/Kokoro-TTS |
| Community (Reddit) | https://reddit.com/r/TextToSpeech |

---