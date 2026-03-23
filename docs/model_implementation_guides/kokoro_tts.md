<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/kokoro_tts/engine.py` · `core/kokoro_tts/preprocessor.py` · `core/kokoro_tts/postprocessor.py` · `core/kokoro_tts/voice_manager.py`
**Class:** `KokoroEngine` — `load_model()` / `synthesize()` · forced CPU on Apple Silicon (MPS → SIGBUS)
**Constants:** `MAX_CHUNK_TOKENS=150` · `CROSSFADE_SAMPLES=7200` (300ms@24kHz) · `INTER_SENTENCE_PAUSE_SEC=0.8` · `MIN_SPEED=0.65`
**Contract:** Output — 24 kHz mono float32 · Models from HF hub `hexgrad/Kokoro-82M`
**British voices** (`bf_*`, `bm_*`): require separate `KPipeline(lang_code="b")`
**6 presets:** `balanced_calm`, `deep_rest`, `soft_whisper`, `golden_hour`, `earth_root`, `serene_sky`
**Tasks:**
- Tune chunking/pauses → `preprocessor.py` (constants at top)
- Tune prosody/punctuation rules → `preprocessor.py :: enhance_prosody_punctuation()`
- Tune voice FX → `postprocessor.py :: build_voice_chain()`
- Add/edit voice blends → `voice_manager.py :: MEDITATION_PRESETS`
**See also:** `docs/ARCHITECTURE.md#phase-2--tts-synthesis` · `docs/prompting_guides/vocal_kokoro_instructions.md`
<!-- ────────────────────────────────────────────────────────────────── -->

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
    speed=0.90,                  # float: 0.65–1.0 for meditation (lower = slower)
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
generator = pipeline(text, voice=blend_golden, speed=0.90)
```

---

## 6. Speed & Prosody Tuning for Meditation

### Speed Parameter

| Use Case | Speed Value |
|----------|-------------|
| Normal speech | 1.0 |
| Calm meditation pace (recommended default) | **0.90** |
| Ideal meditation range | **0.85–0.95** |
| Deep relaxation / sleep | 0.80–0.85 |
| Very slow breathing guides | **0.70 (minimum recommended)** |
| Hard minimum (below = artifacts) | **0.65** |

Research doc 1 recommends 0.85×–0.95× as the effective meditation range. Research doc 3 confirms 0.7–0.9 for meditative pace. Do not go below 0.65.

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

Kokoro does **not** natively support `[pause:Xs]` markers in raw text — it will either ignore them or attempt to pronounce the word "pause." The correct implementation (in `core/kokoro_tts/engine.py` and `core/kokoro_tts/preprocessor.py`) is:

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

The existing `core/kokoro_tts/preprocessor.py` converts `\n\n` to `[pause:6.5s]` — a spacious paragraph break tuned for meditation pacing. Do not rely on Kokoro's natural paragraph pausing — it is inconsistent (~0.15s variability).

### Target Duration Matching

Research doc 1 recommends a feature to hit an exact track length by slightly adjusting speed:

```python
def calculate_adjusted_speed(
    script_segments: list[dict],
    target_duration_sec: float,
    base_speed: float = 0.90,
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
| Target LUFS | −14 LUFS (streaming distribution standard — Spotify, Apple Music, YouTube) |

> **Critical rule from Research doc 1:** Always generate WAV first. Converting directly to MP3 during synthesis can introduce click artifacts in the silence regions between segments. Only convert to MP3 as the final export step after all mixing is done.

---

## 10. Post-Processing Pipeline

### Kokoro-Specific Signal Chain

MoodScape uses a multi-stage postprocessing pipeline tailored to Kokoro's ISTFTNet vocoder characteristics. All processing lives in `core/kokoro_tts/postprocessor.py`:

#### Stage 1 — Per-Chunk Cleanup (`process_chunk()`)
- DC offset removal (`audio -= np.mean(audio)`)
- Hard-clip guard (clamp to [-1, 1])
- Trailing silence trim (RMS windowing at -45 dBFS, 20ms windows)
- Spectral flatness detection for repetition-loop artifacts (Wiener entropy > 0.85)
- RMS normalization to -23 dBFS (EBU R128 speech reference)

#### Stage 2 — Segment Assembly
- **22ms cosine-squared crossfade** between chunks (528 samples at 24 kHz)
- **25ms pre-roll silence** + **100ms cosine fade-in** (masks cold-start prosody drift)
- **50ms fade-out** at segment boundaries
- **0.8s inter-sentence pause** (1.2s after ellipsis)

#### Stage 2b — Spectral Gating Noise Reduction (`reduce_synthesis_noise()`)
- Lightweight stationary spectral gating via `noisereduce` library
- Replaces the disabled neural denoiser (resemble-enhance) which is unstable on Apple Silicon
- Conservative parameters: `prop_decrease=0.6`, `n_std_thresh=2.0`, `freq_mask_smooth_hz=500`
- Reduces ISTFTNet vocoder hiss by ~6 dB without damaging soft consonants

#### Stage 3 — Unified Voice FX Chain (`build_voice_chain()`, single-pass at mix sample rate)
1. NoiseGate (-42 dB) — cleanup, mutes inter-chunk silence
2. HighpassFilter (80 Hz) — removes sub-bass rumble and plosives
3. LowShelfFilter (+2 dB @ 200 Hz) — warmth / proximity effect
4. PeakFilter (-2 dB @ 350 Hz, Q=1.0) — mud cut
5. Compressor (2:1, -18 dB threshold, 10ms/100ms) — dynamics control
6. PeakFilter (-2.5 dB @ 3 kHz, Q=0.8) — anti-harshness subtractive cut at the primary metallic resonance zone
7. HighShelfFilter (-4.0 dB @ 7.5 kHz) — de-harsh shelf enforcing steep spectral tilt
8. Convolution reverb — space (IR: warm_studio / wooden_hall / stone_chapel)
9. LowpassFilter (9.5 kHz) — Nyquist masking (after reverb to catch reverb HF)
10. Limiter (-1 dBFS) — protection

Tape saturation (tanh soft-clipping, drive=1.05) is applied before the Pedalboard chain, adding subtle 2nd/3rd harmonics for perceived warmth.

### Quality Assurance (`core/qa_monitor.py`)
Automated checks run after the master chain:
- **Clipping detection** — flags if >0.1% of samples exceed ±0.99
- **LUFS compliance** — verifies within ±2 dB of -14 LUFS target
- **Long silence detection** — flags silence gaps > 15 seconds
- **Spectral balance** — warns if presence (2-5 kHz) exceeds warmth (100-300 Hz)
- **Silence ratio** — warns if outside 15-60% range for meditation

### What NOT to Do

| Anti-Pattern | Reason | Alternative |
|-------------|--------|-------------|
| Pitch shifting | Degrades quality | Use voice selection / blending |
| Time stretching | Degrades quality | Use Kokoro's `speed` parameter |
| pydub for segment merging | Slow, low quality | Pedalboard + numpy (already in MoodScape) |
| FFmpeg subprocess calls | Unnecessary dependency | Pedalboard handles all I/O |

---

## 11. Implementation File Map

The Kokoro TTS implementation is a self-contained package at `core/kokoro_tts/`:

| File | Purpose |
|------|---------|
| `core/kokoro_tts/engine.py` | `KokoroEngine` — model loading, dual-pipeline (US/British), synthesis with per-sentence speed variation, per-sentence voice jitter, room-tone pauses, spectral gating noise reduction |
| `core/kokoro_tts/preprocessor.py` | Script parsing (`[pause:Xs]`, `[breath]`, `\n\n`), text expansion, contraction conversion, IPA phoneme injection (30+ Sanskrit/yoga terms), prosody punctuation enhancement, sensory ellipsis injection, sentence length variation, per-sentence speed annotation, token-aware chunking |
| `core/kokoro_tts/postprocessor.py` | Per-chunk cleanup (artifact trim, RMS norm), crossfade assembly (300ms cos²), segment fades, spectral gating noise reduction, room-tone pause generation, unified voice FX chain (`build_voice_chain()`) |
| `core/kokoro_tts/voice_manager.py` | Voice tensor loading from HuggingFace, SLERP blending, per-sentence voice micro-variation (jitter, amount=0.001), 6 meditation presets (`balanced_calm`, `deep_rest`, `soft_whisper`, `golden_hour`, `earth_root`, `serene_sky`), British voice detection |
| `core/pipeline.py` | End-to-end orchestration: script → TTS → upsampling (soxr_vhq) → voice FX → music gen → mixing → master → export |
| `core/qa_monitor.py` | Quality validation: clipping, LUFS, silence gaps, spectral balance, silence ratio |

---

## 12. Gradio UI Voice Options

The `app.py` voice dropdown includes both premium blends and individual voices:

```python
KOKORO_VOICE_CHOICES = [
    # Premium Blends (Recommended)
    ("Balanced Calm — natural & human (default)",  "balanced_calm"),
    ("Deep Rest — intimate & breathy",             "deep_rest"),
    ("Soft Whisper — ASMR relaxation",             "soft_whisper"),
    ("Golden Hour — warm & airy",                  "golden_hour"),
    ("Earth Root — grounding blend",               "earth_root"),
    # High-Quality Individual Voices
    ("Heart — US Female (warm)",                 "af_heart"),
    ("Nicole — US Female (calm/ASMR)",           "af_nicole"),
    # British & Male Voices
    ("Emma — UK Female (wise)",                  "bf_emma"),
    ("Adam — US Male (grounding)",               "am_adam"),
    ("George — UK Male (warm)",                  "bm_george"),
]
```

Blend presets are resolved by `core/kokoro_tts/voice_manager.py` using SLERP interpolation.

---

## 13. Expressiveness Enhancements

Kokoro-82M has no runtime emotion conditioning, SSML, or prosody parameters beyond `speed` and `voice`. Expressiveness is achieved through two primary layers: **text structure** (shapes the model's pitch/duration predictions at generation time — highest impact) and **voice blending** (interpolation in the convex style space). DSP post-processing is kept minimal to preserve Kokoro's neural vocoder quality.

### Text Preprocessing (`preprocessor.py`)

The `preprocess_for_meditation()` pipeline applies these transforms in order:

1. **Text expansion** — digits/abbreviations → spoken words
2. **Contraction conversion** — formal phrasing → contractions (`"you are"→"you're"`, `"do not"→"don't"`, etc.). Kokoro produces warmer, more conversational prosody with contractions.
3. **IPA phoneme injection** — Sanskrit/yoga terms → misaki IPA syntax
4. **Prosody punctuation** — comma insertion at breath-group boundaries (gerund phrases, meditation verbs, long clauses)
5. **Sensory ellipsis injection** — `...` before sensory words (`warmth`, `peace`, `stillness`, etc.) when preceded by a determiner. Kokoro treats ellipsis as a ~200ms trailing pause with suspended pitch.
6. **Sentence length variation** — promotes one long-clause comma to a period per block. Short/long alternation exploits Kokoro's length-dependent style vector selection (shorter utterances receive different prosodic characteristics).

### Per-Sentence Speed Variation (`preprocessor.py` + `engine.py`)

`annotate_speed(sentence, base_speed)` adjusts speed per sentence:
- Short phrases (<6 words): `base_speed × 0.88` (deliberate, intimate)
- Questions: `base_speed × 0.95` (slower for reflection)
- Ellipsis endings: `base_speed × 0.92` (contemplative trailing)
- Default: `base_speed` unchanged
- Always clamped to ≥ 0.65 (Kokoro distortion floor)

### Voice Blending (`voice_manager.py`)

- **SLERP interpolation** replaces linear blending. Spherical linear interpolation preserves embedding norms on the style space hypersphere, producing smoother blends (linear interpolation shrinks norms by ~29% at midpoint for orthogonal vectors).
- **Per-sentence voice jitter** (`add_voice_jitter`, amount=0.001) adds subtle Gaussian noise to the voice tensor per sentence, creating natural micro-variation in timbre. Kept very low (0.001) to avoid audible timbre shifts at pause boundaries.

### Pitch Humanization & Formant Warmth (`postprocessor.py` — DISABLED)

`humanize_voice(audio, sr)` exists in the codebase but is **not called** in the active pipeline. pyworld's WORLD vocoder resynthesis degrades Kokoro's neural vocoder output (ISTFTNet), causing both flatness and harshness. The function is retained for potential future use with a higher-quality resynthesis approach.

### Voice FX Chain (`postprocessor.py`)

The voice FX chain uses gentle glue compression (2:1 @ -18 dB) rather than aggressive parallel compression. The EQ is subtractive: +2.0 dB low shelf @ 200 Hz for warmth, -2.5 dB cut @ 3 kHz (Q=0.8) targeting metallic resonance, -4.0 dB high shelf @ 7.5 kHz for de-harshness. Tape saturation (drive=1.05) adds subtle harmonic warmth before the Pedalboard chain. This preserves the natural smoothness of Kokoro's neural vocoder output while actively removing harshness.

### Room-Tone Pauses (`postprocessor.py` + `engine.py`)

`generate_room_tone(duration_sec, sr, level_db=-55)` replaces dead-silence pauses with barely perceptible bandpass-filtered noise (100–800 Hz, -55 dBFS). Cosine fade-in/out prevents clicks. Used for both inter-sentence pauses and explicit `[pause:Xs]` segments.

---

## 14. What NOT to Do (confirmed across all three research documents)

| Anti-Pattern | Reason | Alternative |
|-------------|--------|-------------|
| Pass `[pause:Xs]` raw to Kokoro | Model ignores or mispronounces it | Silence array injection via `core/kokoro_tts/preprocessor.py` |
| Use `pydub` for mixing | Slow, low quality | Pedalboard + numpy (already in MoodScape) |
| Use `ffmpeg` subprocess | Unnecessary dependency | Pedalboard handles all I/O |
| Use `librosa` time-stretch for pacing | Degrades voice quality | Use Kokoro's `speed` parameter |
| Go below `speed=0.65` | Artifacts and distorted prosody | Clamp to `max(speed, 0.65)` |
| Export MP3 during synthesis | Click artifacts in silences | WAV first, MP3 only at final export |
| Load both TTS and HeartMuLa simultaneously | OOM on 8–16 GB GPU | Sequential loading with `unload_model()` |
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