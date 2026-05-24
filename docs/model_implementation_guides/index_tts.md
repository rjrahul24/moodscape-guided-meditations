<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/index_tts/engine.py` · `core/index_tts/preprocessor.py` · `core/index_tts/voice_registry.py` · `core/index_tts/postprocessor.py`
**Class:** `IndexTTSEngine` — `load_model()` / `synthesize()` · zero-shot voice cloning with emotion control · voice resolved at construction
**Constants:** `MAX_CHUNK_CHARS=250` · `_VAD_GAIN_FLOOR=0.15` · `_DEFAULT_SPEED=1.0`
**Contract:** Output — 24 kHz mono float32 · Speaker assets `reference_audio/vocals/` · Emotion assets `reference_audio/instrumental/`
**Checkpoints:** `model_checkpoints/indextts2/` (manual download from HuggingFace `IndexTeam/IndexTTS-2`)
**Tasks:**
- Add new voice → drop `.wav` in `reference_audio/vocals/` (auto-discovered by `voice_registry.scan_voices()`)
- Add emotion → drop `.wav` in `reference_audio/instrumental/` (auto-discovered by `voice_registry.scan_emotions()`)
- Tune chunking → `preprocessor.py :: MAX_CHUNK_CHARS`
- Tune VAD behavior → `engine.py :: _apply_silero_vad()`
- Tune voice FX → `postprocessor.py :: build_index_voice_chain()`
**See also:** `docs/prompting_guides/vocal_indextts_instructions.md` · `CLAUDE.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# IndexTTS-2 Implementation Guide

IndexTTS-2 is the third TTS engine available in MoodScape, offering zero-shot voice cloning with emotion control and precise duration management. It is an autoregressive model using the BigVGANv2 neural vocoder, producing high-quality 24 kHz speech with natural prosody.

---

## 1. Overview

| Property | Value |
|---|---|
| Model | IndexTTS2 (autoregressive) |
| Vocoder | BigVGANv2 |
| Native sample rate | 24 000 Hz (matches Kokoro / F5-TTS / pipeline contract) |
| Device | MPS (Apple Silicon) with CPU fallback |
| Precision | float32 (prevents MPS NaN errors) |
| Voice input | Speaker reference audio (via VoiceRegistry) — no transcript required |
| Emotion input | Optional emotion reference audio or text description |
| Generation mode | "Free" (uncontrolled) — natural prosodic timing |
| VRAM usage | ~3–4 GB (fits within M1 Max 36 GB budget) |

**Why IndexTTS-2?**

- **Zero-shot cloning with emotion control** — uniquely decouples speaker identity from emotional expression, allowing "calm" or "warm" emotion with any voice
- **Autoregressive architecture** — produces more naturally variable prosody than diffusion-based models (F5-TTS), avoiding the repetitive intonation curve issue
- **BigVGANv2 vocoder** — high-quality neural waveform synthesis at 24 kHz
- **No transcript required** — unlike F5-TTS, IndexTTS-2 does not need a verbatim transcript of the reference audio
- **Duration control** — supports both free-form and fixed-duration generation modes
- **SpeechEngine compliance** — implements the same ABC interface as Kokoro and F5, so all downstream pipeline stages are engine-agnostic

---

## 2. Module Structure

```
core/index_tts/
├── __init__.py              — package marker
├── engine.py                — IndexTTSEngine(SpeechEngine): synthesis, VAD, assembly
├── preprocessor.py          — character-aware script chunker (250-char limit)
├── postprocessor.py         — IndexTTSMasteringEngine, split-band de-esser, voice FX chain
└── voice_registry.py        — voice and emotion asset discovery

reference_audio/             — (project root) shared reference audio directory
├── vocals/                  — speaker reference WAV files (for voice cloning)
└── instrumental/            — emotion reference WAV files (for emotion control)

model_checkpoints/           — (project root) model weights
└── indextts2/               — IndexTTS-2 checkpoints (manual download)
    ├── config.yaml
    ├── bigvgan_v2_24khz_100band_256x/
    ├── bpe_69.json
    ├── dvae.pth
    ├── gpt.pth
    └── s2m.ckpt
```

---

## 3. Voice Selection (VoiceRegistry)

IndexTTS-2 voice resolution is simpler than F5-TTS because no transcript is required:

```
reference_audio/vocals/calm_meditation.wav  → slug: "calm_meditation"
reference_audio/vocals/gentle_guide.wav     → slug: "gentle_guide"
```

Each `.wav` file in `reference_audio/vocals/` is automatically discovered and appears in the Gradio UI dropdown. The filename (without extension) becomes the voice slug, which is transformed to a human-readable label in the UI.

### Voice Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24,000 Hz |
| Channels | Mono |
| Duration | 5–10 seconds |
| Bit depth | 16-bit or 32-bit float |

### Emotion Registry

Emotion reference audio works similarly:

```
reference_audio/instrumental/calm.wav     → slug: "calm"
reference_audio/instrumental/warm.wav     → slug: "warm"
```

Users can also upload custom emotion audio directly in the Gradio UI.

---

## 4. Reference Audio Guide

### Recording for Voice Cloning

- **Pace:** Speak at meditation pace (~100–120 WPM)
- **Tone:** Warm, calm, compassionate — not whispered, not monotone
- **Content:** 2–3 complete meditation sentences (IndexTTS-2 analyzes speech patterns from the reference)
- **Duration:** 5–10 seconds (IndexTTS-2 works well with shorter clips than F5-TTS)
- **Environment:** Quiet room, no HVAC/fan noise, pop filter recommended
- **No transcript needed:** Unlike F5-TTS, IndexTTS-2 does not require a verbatim `.txt` transcript

### Converting Audio

```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 voice_name.wav
```

### Recording for Emotion Reference

- Record 3–5 seconds of speech exhibiting the target emotion
- Keep the same environment requirements as voice references
- Use a different speaker if desired — IndexTTS-2 decouples emotion from identity
- Examples: calm meditation delivery, warm nurturing tone, soothing sleep-preparation voice

---

## 5. Preprocessing

The preprocessor follows the exact same pattern as F5-TTS with one key difference:

| Setting | F5-TTS | IndexTTS-2 | Rationale |
|---|---|---|---|
| Chunk limit | 300 chars | **250 chars** | Autoregressive models hallucinate on longer inputs |
| Paragraph pause | 3.0s | **3.5s** | BigVGANv2 acoustic tails benefit from more separation |
| Text normalization | Same | Same | Both do their own G2P from raw text |
| Pause/voice markers | Same | Same | Shared parsing logic |

### Pipeline Flow

```
parse_script() → same pause/voice/breath parser as F5
normalize_for_indextts() → shared expand_text() + punctuation normalization
split_into_chunks() → sentence-boundary splitting at 250 chars
prepare_segments() → full pipeline: parse → normalize → chunk
```

---

## 6. Synthesis Engine

### Chunk-and-Stitch Pipeline

```
Input: list[dict] segments from preprocessor
  │
  ▼ For each "speech" segment:
  │   1. Normalize text (collapse whitespace, lowercase ALL_CAPS)
  │   2. Synthesize via IndexTTS2.infer() to temp WAV file
  │   3. Read output WAV → float32 numpy array
  │   4. Trim trailing silence (_TRIM_THRESHOLD_DB=-45 dB)
  │   5. Apply Silero VAD (crop trailing + attenuate interior gaps to 15%)
  │   6. Build voice activity mask
  │
  ▼ For each "pause" segment:
  │   Generate room-tone noise (shared from kokoro_tts.postprocessor)
  │
  ▼ For each "breath" segment:
  │   Load breath sample (shared from core.breath_sounds)
  │
  ▼ Assembly (type-aware boundaries):
  │   speech → speech: 0.6s room-tone gap + 300ms cosine crossfade
  │   anything → pause: direct concatenation (preserves exact duration)
  │   pause → anything: direct concatenation
  │
  ▼ Output: voice_audio (float32, 24 kHz, mono) + voice_activity (bool mask)
```

### Inference Pattern

IndexTTS-2's `infer()` method writes output to a file path rather than returning an array. The engine passes the **full meditation-tuned parameter set** on every call:

```python
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tts.infer(
        spk_audio_prompt=ref_audio_path,
        text=chunk_text,
        output_path=tmp.name,
        emo_vector=INDEXTTS_CALM_VECTOR,    # or emo_audio_prompt=<path>
        emo_alpha=INDEXTTS_EMO_ALPHA,        # 0.70
        top_p=INDEXTTS_TOP_P,                # 0.85
        temperature=INDEXTTS_TEMPERATURE,    # 0.70
        interval_silence=INDEXTTS_INTERVAL_SILENCE_MS,        # 200ms
        max_text_tokens_per_segment=INDEXTTS_MAX_TOKENS_PER_SEG,  # 120
        do_sample=True,
        use_random=False,
        verbose=False,
    )
    audio, sr = sf.read(tmp.name, dtype="float32")
```

### Emotion Control

Two paths, with `emo_audio_prompt` winning when both are available:

| Mode | Trigger | Behavior |
|---|---|---|
| **Calm vector** (default) | `emotion_audio_path is None` | `emo_vector = [0,0,0,0,0,0,0,1.0]` — pure calm dimension. Deterministic, immune to LLM mis-mapping. After the API's internal bias scaling this yields a normalized value of ~0.5625, safely under the 0.8 sum-penalty threshold. |
| **Audio reference** | `emotion_audio_path` set (UI dropdown or uploaded WAV) | `emo_audio_prompt=<path>` — projects the emotion of that clip onto the cloned voice. Use for emotions outside the 8D vector space, or to match a specific source recording. |

`emo_alpha=0.70` blends 70% emotion override with 30% speaker-timbre preservation. Below 0.3 fails to suppress reference arousal; at 1.0 timbre flattens.

We intentionally **do not** use `use_emo_text` — IndexTTS-2's Qwen3 path conflates "calm/serene" with "sad/melancholic" (see Issue research note in `vocal_indextts_instructions.md`).

### Why no speed parameter

The IndexTTS-2 v2 API does not expose reliable time-stretching — `use_speed`/`target_dur` are documented but Issue [#422](https://github.com/index-tts/index-tts/issues/422) confirms the token-rate calculation is unreliable for outputs longer than ~10s. The engine accepts `speed` for `SpeechEngine` ABC compatibility but logs a one-time warning and ignores any non-1.0 value. Pacing is shaped by:

1. The calm emotion vector → naturally slower, breathier prosody.
2. `_PARAGRAPH_PAUSE_SEC = 3.5` in the preprocessor → long inter-paragraph beats.
3. `interval_silence=200ms` → API-internal silence between micro-segments.
4. The engine's own 600ms room-tone gap + 300ms cosine crossfade between speech chunks.

### Apple Silicon MPS Strategy

| Setting | Value | Rationale |
|---|---|---|
| `use_fp16` | `False` | float32 prevents MPS NaN errors |
| `use_deepspeed` | `False` | CUDA-only; crashes on MPS |
| `use_cuda_kernel` | `False` | CUDA-only; crashes on MPS |
| Device detection | MPS → CPU fallback | `PYTORCH_ENABLE_MPS_FALLBACK=1` set globally |
| Mel clamping | `torch.clamp(mel, -10, 10)` | Prevents NaN propagation in BigVGANv2 |

### Memory Management

- Sequential engine loading: TTS loads → synthesizes → unloads before music engine loads
- `gc.collect()` + `torch.mps.empty_cache()` on unload
- IndexTTSEngine is instantiated per-call (like F5Engine) and `del`-ed after synthesis

---

## 7. Postprocessing

### Mastering Chain (IndexTTSMasteringEngine)

Tuned specifically for BigVGANv2 vocoder characteristics:

```
Phase A — split_band_deess():  Dynamic sibilance control (5.5 kHz center)
Tape saturation:               tanh(x*1.08)/1.08 for subtle harmonic warmth

Phase B — EQ / dynamics:

    NoiseGate(-42 dB, 2:1)          — catch autoregressive residual noise
    HighpassFilter(75 Hz)           — remove sub-bass rumble
    PeakFilter(350 Hz, -2.0 dB)    — anti-boxiness
    LowShelfFilter(180 Hz, +1.5 dB) — warmth / proximity effect
    PeakFilter(2.8 kHz, -1.5 dB)   — reduce autoregressive metallic resonance
    HighShelfFilter(8 kHz, -2.0 dB) — de-harsh shelf
    HighShelfFilter(10 kHz, +1.5 dB)— air shelf for breathiness/intimacy
    LowpassFilter(11 kHz)           — BigVGANv2 bandwidth cap
    Compressor(-22 dB, 2.5:1)       — gentle meditation-paced leveling
    Limiter(-1.5 dB)                — safe ceiling
```

### Key Differences from F5-TTS Chain

| Parameter | F5-TTS (Vocos) | IndexTTS-2 (BigVGANv2) | Rationale |
|---|---|---|---|
| HPF | 80 Hz | 75 Hz | BigVGANv2 has cleaner bass reproduction |
| Boxiness cut | 300 Hz, -2 dB | 350 Hz, -2 dB | Different low-mid resonance peak |
| Metallic cut | 3.0 kHz, -2 dB | 2.8 kHz, -1.5 dB | Autoregressive models produce different resonance |
| Air shelf | 10 kHz, +1 dB | 10 kHz, +1.5 dB | More air for BigVGANv2's warmer output |
| LPF | 12 kHz | 11 kHz | BigVGANv2 at 24 kHz has slightly different bandwidth |
| Noise gate | -45 dB | -42 dB | Autoregressive noise floor is slightly different |
| De-esser center | 6 kHz | 5.5 kHz | BigVGANv2 sibilance peak is slightly lower |

### Voice FX Chain

Same pattern as F5-TTS: convolution reverb (warm_studio default) + limiter only.
All EQ/dynamics handled upstream by IndexTTSMasteringEngine.

---

## 8. Pipeline Integration

IndexTTS-2 is integrated at the same branching points as F5-TTS in `core/pipeline.py`:

| Step | Branch Point | IndexTTS-2 Action |
|---|---|---|
| Parse script | `tts_engine == "indextts"` | Import `core.index_tts.preprocessor.prepare_segments` |
| Load TTS | `tts_engine == "indextts"` | Instantiate `IndexTTSEngine(voice_slug, emotion_slug, emotion_audio_path)` |
| Synthesize | `tts_engine == "indextts"` | Call `tts.synthesize(segments, speed, seed, emotion_audio_path)` |
| Mastering | `tts_engine == "indextts"` | Instantiate `IndexTTSMasteringEngine(sample_rate)` |
| Voice FX | `tts_engine == "indextts"` | Import `build_index_voice_chain()` |
| Mix SR | `tts_engine in ("f5", "indextts")` | Use 48 kHz mix rate |
| TTS unload | `tts_engine in ("f5", "indextts")` | `del tts` after unloading |

---

## 9. Gradio UI Integration

IndexTTS-2 is exposed in the Gradio UI via `app.py`:

| Widget | Type | Description |
|---|---|---|
| Voice Engine radio | `gr.Radio` | Added "IndexTTS-2" as third choice |
| Voice dropdown | `gr.Dropdown` | Populated from `reference_audio/vocals/*.wav` |
| Emotion dropdown | `gr.Dropdown` | Populated from `reference_audio/instrumental/*.wav` + "None (neutral)" |
| Emotion audio upload | `gr.Audio` | Optional user-uploaded emotion reference WAV |
| Speed slider | `gr.Slider` | 0.70–1.30, default 1.0 |

Visibility callbacks ensure only the active engine's controls are shown.

---

## 10. Apple Silicon Notes

### Known MPS Issues and Mitigations

| Issue | Mitigation |
|---|---|
| NaN in mel-spectrogram processing | `torch.clamp(mel, -10, 10)` before BigVGANv2 |
| Unsupported MPS operations | `PYTORCH_ENABLE_MPS_FALLBACK=1` (set globally) |
| `torch.compile` instability | Not used (IndexTTS-2 doesn't require it) |
| float64 unsupported | Force float32 throughout |
| DeepSpeed CUDA-only | `use_deepspeed=False` |
| CUDA kernel CUDA-only | `use_cuda_kernel=False` |
| Memory fragmentation | `gc.collect()` + `torch.mps.empty_cache()` on unload |

### Recommended PyTorch Version

- PyTorch 2.4+ for improved MPS kernel support
- PyTorch 2.5+ preferred for better MPS stability with autoregressive models

---

## 11. Checkpoint Setup

### Download

```bash
# Using huggingface-cli (recommended)
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=model_checkpoints/indextts2

# Or using Git LFS
git lfs install
git clone https://huggingface.co/IndexTeam/IndexTTS-2 model_checkpoints/indextts2
```

### Verification

```bash
# Check that config.yaml exists
ls model_checkpoints/indextts2/config.yaml
```

If checkpoints are missing, `IndexTTSEngine.load_model()` raises a `FileNotFoundError` with download instructions.

---

## 12. Known Limitations

1. **MPS NaN risk** — rare but possible in BigVGANv2 mel processing; mitigated by tensor clamping
2. **DeepSpeed disabled** — ~30% slower inference than CUDA with DeepSpeed enabled
3. **Qwen3 text encoder** — the embedded text encoder may have specific dtype requirements; float32 is the safe choice
4. **Autoregressive hallucination** — longer inputs (>250 chars) can cause repetition; mitigated by shorter chunk limit
5. **No multi-phase voices** — unlike F5-TTS, IndexTTS-2's voice registry currently does not support `voices.toml` multi-phase definitions (can be added later)
6. **Manual checkpoint download** — unlike Kokoro/F5 which auto-download, IndexTTS-2 checkpoints must be manually downloaded to `model_checkpoints/indextts2/`
