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
├── engine.py                -- F5Engine(SpeechEngine): synthesis, VAD, assembly
├── preprocessor.py          -- character-aware script chunker (300-char limit)
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

F5-TTS preprocessing ([preprocessor.py](../../core/f5_tts/preprocessor.py)) performs **text normalization** and **character-count chunking** at a 300-character limit.

### Why Character-Count Chunking?

F5-TTS has a hard ~30-second context window per `infer()` call. At a meditative pace of 0.88 speed, 300 characters = ~8-10 seconds of audio, safely within the window.

### Text Normalization (`normalize_for_f5`)

Shared digit/abbreviation expansion (via `core.text_utils`) plus F5-specific punctuation fixes:

- **Digits → words**: `10` → `ten`, `4-7-8` → `four, seven, eight`
- **Abbreviations**: `sec` → `seconds`, `min` → `minutes`
- **Colons → commas**: F5 ignores colons, producing no pause
- **Ellipses → periods**: F5 does not create longer pauses from ellipses
- **Em/en-dashes → commas**: not reliably handled by F5
- **Compound hyphens removed**: `well-being` → `wellbeing` (hyphens cause mispronunciation)

### What Preprocessing Does NOT Include

Unlike Kokoro's preprocessor, F5-TTS preprocessing does **not**:
- Inject IPA phonemes
- Insert prosody commas

F5-TTS performs its own G2P from normalized prose.

### Pause Markers

| Marker | Pause duration |
|---|---|
| `[pause:Xs]` | X seconds |
| `[N second pause]` | N seconds |
| `[breath]` / `[inhale]` / `[exhale]` | 1.2 seconds (real breath samples) |
| Double newline | 3.0 seconds |

---

## 7. Synthesis Engine

### Static Reference System

Every speech chunk uses the same static reference audio. This ensures consistent voice identity across all chunks without compounding drift. Multi-phase voices allow switching to different reference clips mid-script via `[voice:phase_name]` markers.

### Text Normalization

Before passing text to F5's `infer()`:
- Collapse whitespace and newlines
- Lower-case ALL_CAPS words (prevents letter-by-letter G2P spelling)

### Inference Parameters

| Parameter | Value | Notes |
|---|---|---|
| `nfe_step` | 32 | Production quality. Use 16 for fast iteration. |
| `cfg_strength` | 2.0 | F5-TTS official default. Enables classifier-free guidance for expressive output. Split-band de-esser handles any HF diffusion artefacts. |
| `sway_sampling_coef` | -1.0 | Enables sway sampling for smoother meditative prosody |
| `speed` | 0.88 (default) | Primary pacing control. ~95-100 WPM meditation pace with natural prosodic timing. |
| `target_wpm` | None (default) | Natural rhythm mode (recommended). Set 90-150 for fixed WPM pacing. |
| `fix_duration` | Only when `target_wpm` is set | Per-chunk: `ref_duration + (word_count / target_wpm * 60)` |
| `remove_silence` | False | Engine handles silence trimming explicitly |

### Pacing: Natural Rhythm vs. Fixed WPM

**Default: Natural rhythm (`target_wpm=None`, `speed=0.88`)**

F5-TTS's core design advantage is implicit duration assignment — the paper states that avoiding explicit phoneme-level duration modelling "results in improved prosody and rhythm." The `speed` parameter gives the model ~14% more temporal room than natural speech, targeting ~95-100 WPM for meditation delivery while letting the model assign natural internal rhythm, pauses, and emphasis.

**Optional: Fixed WPM pacing (`target_wpm=90-150`)**

When `target_wpm` is set (via the Pacing slider in the UI), the engine calculates `fix_duration` per chunk:

```python
word_count = len(gen_text.split())
target_speech_sec = word_count / target_wpm * 60.0
fix_duration = ref_audio_duration_sec + target_speech_sec
```

This provides precise pacing control but constrains the model's prosodic freedom. Use when consistent timing is more important than expressiveness (e.g., timed breathing exercises).

### Post-Inference Processing

1. **Trailing silence trimming**: Remove samples below -45 dBFS, keep 50ms decay tail
2. **Silero VAD**: Two-pass — crop trailing non-speech (100ms safety tail), attenuate interior gaps to 15% gain floor (`_VAD_GAIN_FLOOR=0.15`)
3. **Assembly**: 0.4s room-tone gap + 300ms equal-power cosine crossfade between speech chunks; direct concatenation at pause boundaries

### Pauses and Silence

Pauses use room-tone noise (bandpass 100-800 Hz, -55 dBFS, with cosine fades) for natural acoustic continuity between speech segments. The room-tone generator is shared with the Kokoro engine (`generate_room_tone()` from `core.kokoro_tts.postprocessor`). Downstream convolution reverb provides additional natural tails on voice segments.

---

## 8. Postprocessing (Mastering)

### F5MasteringEngine

Two-phase mastering engine ([postprocessor.py](../../core/f5_tts/postprocessor.py)):

**Phase A -- `restore_vocals()`**: Stub. Vocos output is pre-clean; no neural denoising needed.

**Phase B -- `master_vocals()`**: EQ / dynamics at the mix sample rate (48 kHz for F5).

#### Split-Band De-Esser (Preprocessing)

- Isolates 4-8 kHz sibilant band via 4th-order Butterworth bandpass
- Compresses sibilant band: threshold -20 dB, ratio 4:1, attack 0.5ms, release 10ms
- Recombines with untouched non-sibilant signal

#### Tape Saturation

After de-essing, subtle tape-style saturation (`np.tanh(audio * 1.08) / 1.08`) adds 2nd/3rd harmonics for perceived warmth without audible distortion. This bridges the gap between clinical TTS output and the warm, intimate quality of professional meditation narration.

#### Mastering Chain

```
NoiseGate(-45 dB, 2:1, 5ms/250ms)       -- catch diffusion residual noise
HighpassFilter(80 Hz)                    -- remove sub-bass rumble
PeakFilter(300 Hz, -2 dB, Q=1.5)        -- anti-boxiness (cut low-mid mud)
LowShelfFilter(200 Hz, +2 dB)           -- warmth
PeakFilter(3.0 kHz, -2.0 dB, Q=0.8)    -- anti-harshness subtractive cut
HighShelfFilter(7.5 kHz, -3.0 dB)       -- de-harsh shelf (steep spectral tilt)
HighShelfFilter(10 kHz, +1.0 dB)        -- air shelf for breathiness/intimacy
LowpassFilter(12 kHz)                   -- preserve Vocos native bandwidth
Compressor(-20 dB, 2.5:1, 15ms/200ms)  -- gentle, meditation-paced leveling
Limiter(-1.5 dB, 80ms)                  -- safe ceiling
```

**Why 12 kHz lowpass (vs. Kokoro's 9.5 kHz)?** Vocos has broader native bandwidth than Kokoro's ISTFTNet. 12 kHz preserves natural "air" and breathiness while cutting ultrasonic content.

**Anti-harshness philosophy:** The 3 kHz zone is the primary locus of metallic resonance from the diffusion-based Vocos vocoder. The -2.0 dB subtractive cut (vs. the previous +1.5 dB presence boost) removes the core harshness, while the -3.0 dB shelf at 7.5 kHz enforces the steep spectral tilt characteristic of calm, breathy speech.

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
| `cfg_strength` | 2.0 | Enables classifier-free guidance for expressive output. |
| `sway_sampling_coef` | -1.0 | Smoother, more natural prosody. Disable (set to 0) only for timing artefacts. |
| `target_wpm` | None | Natural rhythm (recommended). Set 90-150 for fixed pacing. |
| `speed` | 0.88 | Primary pacing control. ~95-100 WPM meditation pace. |
| Device | MPS | Apple Silicon. CPU fallback for non-Apple hardware. |
| Precision | fp16 | `ema_model.to(torch.float16)` -- prevents distortion artefacts seen in bf16. |
| First-run | Slow | Model weights (~1.5 GB) downloaded from HuggingFace on first use. |

### Expected Generation Speed (Apple M1 Max, MPS)

| Segment length | NFE=32 | NFE=16 |
|---|---|---|
| Short (1-2 sentences, ~150 chars) | ~3-6s | ~2-3s |
| Medium (3-4 sentences, ~400 chars) | ~6-12s | ~4-6s |

Long scripts are chunked into <=300-character segments; generation time scales linearly with chunk count.

---

## 11. Known Limitations

1. **No Sanskrit / yoga term phoneme injection** -- Kokoro's preprocessor injects IPA for terms like "chakra". F5-TTS does its own G2P; these terms are pronounced per the model's training data.

2. **30-second context ceiling** -- F5-TTS has a hard inference limit of ~30s per call. The 300-character chunking keeps segments safely under this. If truncation occurs, add punctuation to the script.

3. **Voice consistency over very long scripts** -- Zero-shot cloning is stochastic. The static reference system maintains consistency, but the same intonation curve repeats on every chunk. Use multi-phase voices to introduce natural variation across sections.

4. **Expression depends on reference audio** -- At speed 1.0, F5-TTS faithfully reproduces the expression from the reference. A monotone reference produces monotone output. Record reference clips with natural, expressive meditation delivery.
