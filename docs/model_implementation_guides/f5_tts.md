# F5-TTS Implementation Guide

F5-TTS is the second TTS engine available in MoodScape, offering zero-shot voice cloning. Unlike Kokoro (which uses a fixed set of pre-trained voice embeddings), F5-TTS clones any voice from a short reference audio clip at inference time. This makes it ideal for studios or creators who want the meditation narrated in a specific human voice.

---

## 1. Overview

| Property | Value |
|---|---|
| Model | F5TTS_v1_Base |
| Vocoder | Vocos |
| Native sample rate | 24 000 Hz (matches Kokoro / pipeline contract) |
| Device | MPS (Apple Silicon) with CPU fallback |
| Voice input | Reference audio + verbatim transcript (via VoiceRegistry) |
| Key inference params | `nfe_step=32`, `sway_sampling_coef=-1.0`, `cfg_strength=2.0` |

**Why F5-TTS?**

- **Zero-shot cloning** -- no fine-tuning required; one 10-12s clip is enough
- **Vocos vocoder** -- produces a cleaner signal than Kokoro's ISTFTNet; no hiss artefacts, no spectral noise gating needed
- **Native 24 kHz output** -- matches the pipeline's `SAMPLE_RATE` constant exactly; no resampling within the engine
- **SpeechEngine compliance** -- implements the same `load_model / unload_model / synthesize / get_available_voices` interface as KokoroEngine, so all downstream pipeline stages (mixing, mastering, FX) are engine-agnostic

---

## 2. Module Structure

```
core/f5_tts/
├── __init__.py              -- package marker
├── engine.py                -- F5Engine(SpeechEngine): synthesis, chaining, VAD, WPM normalization
├── preprocessor.py          -- character-aware script chunker (400-char limit)
├── postprocessor.py         -- F5MasteringEngine, split-band de-esser, voice FX chain
├── voice_registry.py        -- voice asset discovery and multi-phase resolution
└── assets/
    ├── README.md            -- reference audio format requirements
    ├── voices.toml          -- multi-phase voice definitions
    ├── reference_audio/     -- .wav files (24 kHz, mono, 16-bit PCM)
    └── reference_transcript/ -- .txt transcripts (verbatim)
```

---

## 3. Voice Selection (VoiceRegistry)

Voice identity is resolved at construction time via a **voice slug** that maps to registered asset pairs:

```python
engine = F5Engine(voice_slug="female_meditative_warm")
```

### VoiceRegistry System

The `voice_registry.py` module discovers voices by scanning `core/f5_tts/assets/`:

1. Scans `reference_audio/*.wav` for audio files
2. Matches each `.wav` with a corresponding `.txt` transcript in `reference_transcript/`
3. Validates transcript is non-empty
4. Layers multi-phase definitions from `voices.toml` (TOML format with phase names as keys)

### Multi-Phase Voices

A single voice can have multiple **phases** for different sections of a meditation:

```toml
# voices.toml example
[female_meditative_warm]
default = { audio = "female_meditative_warm.wav", transcript = "female_meditative_warm.txt" }
closing = { audio = "female_meditative_warm_closing.wav", transcript = "female_meditative_warm_closing.txt" }
```

Phase switching is triggered by `[voice:phase_name]` markers in the meditation script:

```
Welcome to this meditation session...

[voice:closing]
Thank you for joining. Carry this peace with you...
```

### Gradio UI Integration

- Voice registry is scanned at startup, populating a dropdown in the UI
- Selecting "F5-TTS" shows a voice personality dropdown; selecting "Kokoro" hides it
- Both TTS accordions are hidden in "Instrumental Only" mode

---

## 4. Reference Audio Recording Guide

The quality of zero-shot cloning depends almost entirely on the reference audio.

### Format Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24 000 Hz |
| Channels | Mono |
| Duration | 10-12 seconds |
| Bit depth | 16-bit or 32-bit float |

Convert any recording:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 voice_name.wav
```

### Recording Tips

- **Room**: Soft furnishings absorb reflections. Avoid bathrooms or bare-walled rooms.
- **Microphone**: Any clean condenser or large-diaphragm USB mic, 20-30 cm from mouth, slightly off-axis.
- **Pop filter**: Use one. Plosive bursts cause alignment failures.
- **Background noise**: Turn off HVAC, fans, and notifications.
- **Speaking style**: Calm, compassionate, slightly slower than conversational (~120 WPM). Do not whisper.
- **Avoid**: Strong emotion, vocal fry, and uptalk -- these are difficult to clone consistently.

### The Verbatim Transcript Requirement

F5-TTS aligns reference audio against its transcript at the character level. A single word mismatch shifts the alignment window, causing metallic artefacts, pitch drift, or stutter. Rules:

1. Type exactly what the speaker says, including filler words
2. Match punctuation as spoken
3. Do **not** add IPA or phonetic spelling -- F5-TTS does its own G2P

---

## 5. Reference Audio Conditioning

At model load time (`load_model()`), each reference audio phase is automatically conditioned:

1. **F5's `preprocess_ref_audio_text()`** clips to ~12s and performs Whisper transcription
2. **RMS normalization** to -20 dBFS -- ensures consistent reference levels across voices
3. **1.0s trailing silence** (low-level noise at -55 dBFS) -- prevents phrase leakage where reference words bleed into generated output. The noise level is above F5's internal -42 dBFS edge trimmer threshold so it survives preprocessing.

This runs once per voice phase load (not per chunk), adding zero per-chunk overhead.

---

## 6. Preprocessing

F5-TTS preprocessing ([preprocessor.py](../../core/f5_tts/preprocessor.py)) performs **character-count chunking** at a 300-character limit.

### Why Character-Count Chunking?

F5-TTS has a hard ~30-second context window per `infer()` call. At a meditative pace of 0.80 speed, 300 characters = ~8-10 seconds of audio, safely within the window.

### What Preprocessing Does NOT Include

Unlike Kokoro's preprocessor, F5-TTS preprocessing does **not**:
- Expand digits to words
- Inject IPA phonemes
- Insert prosody commas

F5-TTS performs its own G2P from raw prose. Write natural meditation prose.

### Pause Markers

| Marker | Pause duration |
|---|---|
| `[pause:Xs]` | X seconds |
| `[N second pause]` | N seconds |
| `[breath]` / `[inhale]` / `[exhale]` | 1.2 seconds (real breath samples) |
| Double newline | 6.5 seconds |

---

## 7. Synthesis Engine

### Chained Reference System

Instead of using the same static reference for every chunk, F5Engine maintains a **quasi-autoregressive** chain:

- Each speech chunk (after the first) seeds the model from the **tail of the previous chunk's audio**
- This provides acoustic context for natural pitch and energy continuity across sentences
- The static reference is restored after:
  - **Long pauses (>= 3.0s)** -- section boundaries warrant a fresh voice anchor
  - **Voice phase changes** -- different reference audio for the new phase
  - **Every 6 consecutive speech chunks** -- prevents voice drift on long sessions
- Short pauses (< 3.0s) and breath segments **maintain** the chain for prosodic continuity

Constants:
- `_CHAIN_RESET_EVERY = 6` -- reset to static ref every N consecutive speech chunks
- `_CHAIN_RESET_PAUSE_SEC = 3.0` -- only long pauses trigger reset
- `_CHAIN_MAX_REF_SEC = 5.0` -- maximum seconds of previous chunk tail to use
- `_CHAIN_MAX_REF_FRAC = 0.50` -- cap chain ref at 50% of previous chunk length

### Text Normalization

Before passing text to F5's `infer()`:
- Collapse whitespace and newlines
- Lower-case ALL_CAPS words (prevents letter-by-letter G2P spelling)
- Append trailing ellipsis (cues natural decay tail rather than abrupt cutoff)

### Inference Parameters

| Parameter | Value | Notes |
|---|---|---|
| `nfe_step` | 32 | Production quality. Use 16 for fast iteration. |
| `cfg_strength` | 2.0 | Paper-validated optimal for stable generation |
| `sway_sampling_coef` | -1.0 | Enables sway sampling for smoother meditative prosody |
| `speed` | 0.80 (default) | Duration predictor generates correct mel frame count |
| `remove_silence` | False | Engine handles silence trimming explicitly |

### Post-Inference Processing

1. **Trailing silence trimming**: Remove samples below -45 dBFS, keep 50ms decay tail
2. **WPM normalization**: Measure each chunk's words-per-minute, time-stretch outliers to within +/-15% of session median via librosa
3. **Silero VAD**: Probability-based gain envelope with 15% gain floor (`_VAD_GAIN_FLOOR=0.15`) — attenuates non-speech segments to 15% instead of zeroing them, preserving natural breath, resonance, and room tone
4. **Assembly**: 0.8s room tone gap + 300ms equal-power cosine crossfade between speech chunks; direct concatenation at pause boundaries

### Room Tone

Pauses and breath segments use low-level noise (1e-3 amplitude, ~-60 dBFS) instead of digital silence. This prevents the unnatural "dead air" artefact and keeps reverb tails active through pauses.

---

## 8. Postprocessing (Mastering)

### F5MasteringEngine

Two-phase mastering engine ([postprocessor.py](../../core/f5_tts/postprocessor.py)):

**Phase A -- `restore_vocals()`**: Stub. Vocos output is pre-clean; no neural denoising needed.

**Phase B -- `master_vocals()`**: EQ / dynamics at the mix sample rate (48 kHz for F5).

#### Split-Band De-Esser (Preprocessing)

- Isolates 4-8 kHz sibilant band via 4th-order Butterworth bandpass
- Compresses sibilant band: threshold -18 dB, ratio 4:1, attack 0.5ms, release 10ms
- Recombines with untouched non-sibilant signal

#### Mastering Chain

```
NoiseGate(-50 dB, 1.5:1, 5ms/250ms)    -- safety gate for clean noise floor
HighpassFilter(80 Hz)                    -- remove sub-bass rumble
PeakFilter(300 Hz, -2 dB, Q=1.5)        -- anti-boxiness (cut low-mid mud)
LowShelfFilter(200 Hz, +2 dB)           -- warmth
PeakFilter(3.2 kHz, +1.5 dB, Q=0.8)    -- presence / intelligibility
HighShelfFilter(10 kHz, +1.5 dB)        -- air shelf for clarity
HighShelfFilter(8 kHz, -1.0 dB)         -- tame brightness without killing air
LowpassFilter(13 kHz)                   -- remove ultrasonic content
Compressor(-20 dB, 2.5:1, 15ms/100ms)  -- gentle, transparent leveling
Limiter(-1.5 dB, 80ms)                  -- safe ceiling
```

**Why 13 kHz lowpass (vs. Kokoro's 9.5 kHz)?** Vocos has broader native bandwidth than Kokoro's ISTFTNet. 13 kHz preserves natural "air."

**No algorithmic reverb in mastering.** Reverb is applied downstream via the user-controlled convolution reverb in `build_f5_voice_chain()`.

### Voice FX Chain

`build_f5_voice_chain(reverb_amount, ir_name)` -- convolution reverb + limiter only:

```
Convolution(impulse_response, mix=reverb_amount)  -- real IR reverb (0-50% wet)
Limiter(-1.0 dB)                                  -- safety ceiling
```

Available impulse responses: `warm_studio`, `wooden_hall`, `stone_chapel`.

---

## 9. Pipeline Integration

### Branching Points in pipeline.py

1. **Script parsing**: imports `core.f5_tts.preprocessor.prepare_segments`
2. **Model loading**: instantiates `F5Engine(voice_slug=...)` locally per call
3. **Synthesis**: `tts.synthesize(segments, speed=speed, progress_cb=...)`
4. **Mastering init**: instantiates `F5MasteringEngine(sample_rate=24000)`
5. **Upsampling**: F5 forces 48 kHz mixing with `high_accuracy=True` (soxr_vhq)
6. **Voice FX**: `build_f5_voice_chain(reverb_amount, ir_name)`

All downstream stages (mixing, sidechain ducking, master chain, export) are engine-agnostic.

### Memory Management

F5Engine is instantiated locally per `generate()` call. After synthesis: `tts.unload_model()` + `del tts` before the music engine loads. Only one neural model is resident at a time.

---

## 10. Performance Notes

| Setting | Value | Notes |
|---|---|---|
| `nfe_step` | 32 | Production quality. Use 16 for fast script iteration. |
| `sway_sampling_coef` | -1.0 | Smoother, more natural prosody. Disable (set to 0) only for timing artefacts. |
| `speed` | 0.80 | Default meditation pace. |
| Device | MPS | Apple Silicon. CPU fallback for non-Apple hardware. |
| Precision | fp16 | `ema_model.to(torch.float16)` -- prevents distortion artefacts seen in bf16. |
| First-run | Slow | Model weights (~1.5 GB) downloaded from HuggingFace on first use. |

### Expected Generation Speed (Apple M1 Max, MPS)

| Segment length | NFE=32 | NFE=16 |
|---|---|---|
| Short (1-2 sentences, ~150 chars) | ~3-6s | ~2-3s |
| Medium (3-4 sentences, ~400 chars) | ~6-12s | ~4-6s |

Long scripts are chunked into <=400-character segments; generation time scales linearly with chunk count.

---

## 11. Known Limitations

1. **No Sanskrit / yoga term phoneme injection** -- Kokoro's preprocessor injects IPA for terms like "chakra". F5-TTS does its own G2P; these terms are pronounced per the model's training data.

2. **No deterministic seed support** -- F5-TTS does not expose a seed parameter. The `seed=` kwarg from the pipeline is silently ignored. Generation is not reproducible across runs.

3. **30-second context ceiling** -- F5-TTS has a hard inference limit of ~30s per call. The 400-character chunking keeps segments safely under this. If truncation occurs, add punctuation to the script.

4. **Voice consistency over very long scripts** -- Zero-shot cloning is stochastic. The chained reference system and periodic reset maintain consistency, but subtle drift may occur on very long meditations (>30 minutes).
