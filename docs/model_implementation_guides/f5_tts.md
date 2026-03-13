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
| Voice input | Reference audio + verbatim transcript |
| Key inference param | `nfe_step=32`, `sway_sampling_coef=-1.0` |

**Why F5-TTS?**

- **Zero-shot cloning** — no fine-tuning required; one 10–12s clip is enough
- **Vocos vocoder** — produces a cleaner signal than Kokoro's ISTFTNet; no hiss artefacts, no spectral noise gating needed
- **Native 24 kHz output** — matches the pipeline's `SAMPLE_RATE` constant exactly; no resampling within the engine
- **SpeechEngine compliance** — implements the same `load_model / unload_model / synthesize / get_available_voices` interface as KokoroEngine, so all downstream pipeline stages (mixing, mastering, FX) are engine-agnostic

---

## 2. Module Structure

```
core/f5_tts/
├── __init__.py          — package marker
├── engine.py            — F5Engine(SpeechEngine)
├── preprocessor.py      — character-aware script chunker
├── postprocessor.py     — crossfade assembly + F5MasteringEngine
└── assets/
    ├── README.md        — reference audio format requirements
    └── ref_meditation.wav  ← YOU MUST PLACE THIS FILE HERE
```

---

## 3. Voice Selection

The `voice` argument to `F5Engine.synthesize()` accepts two forms:

| Form | Usage |
|---|---|
| `"default"` | Uses the bundled `core/f5_tts/assets/ref_meditation.wav` + the hardcoded transcript in `engine.py` |
| `(ref_audio_path, ref_text)` | Custom zero-shot cloning from any uploaded audio file |

In the Gradio UI:
- Leave the F5-TTS settings accordion empty → uses the bundled default voice
- Upload a reference clip and type the transcript → clones the uploaded voice

The pipeline passes the user's UI selections to `pipeline.generate()` as `f5_ref_audio` and `f5_ref_text`. The engine resolves these into a `(path, text)` tuple internally.

---

## 4. Reference Audio Recording Guide

The quality of zero-shot cloning depends almost entirely on the reference audio. Follow these guidelines:

### Format Requirements

| Property | Value |
|---|---|
| Format | WAV (PCM), uncompressed |
| Sample rate | 24 000 Hz |
| Channels | Mono |
| Duration | 10–12 seconds |
| Bit depth | 16-bit or 32-bit float |

You can convert any recording to the correct format with:
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 ref_meditation.wav
```

### Recording Setup

- **Room**: Soft furnishings absorb reflections. A bedroom, walk-in wardrobe, or carpeted studio works well. Avoid bathrooms, kitchens, and rooms with bare walls.
- **Microphone**: Any clean condenser or large-diaphragm USB mic. Position it 20–30 cm from your mouth, slightly off-axis to reduce plosives.
- **Pop filter**: Use one. Plosive bursts ("p", "b", "t") cause alignment failures in the transcript matching.
- **Background noise**: Turn off HVAC, fans, and notifications. Even subtle noise bleeds into every generated segment at scale.

### Speaking Style

- Match the tone you want in the final meditation: calm, compassionate, slightly slower than conversational pace.
- Do not whisper — F5-TTS needs a full voiced signal for reliable cloning.
- Avoid strong emotion or vocal fry; both are difficult to clone consistently.
- A pace of ~120 words per minute works well (standard meditation narration is 100–130 wpm).

### Reference Content

The content of the reference clip does not matter — it will not appear in the output. Choose any text that lets you speak naturally and demonstrates your full vocal range.

**Example reference script** (used as default in `engine.py`):
```
Allow yourself to settle here, right where you are.
There is nothing to do, nowhere to go.
Simply rest in this moment, breathing gently.
```

---

## 5. The Verbatim Transcript Requirement

F5-TTS aligns the reference audio against its transcript to extract speaker timbre, prosody, and rhythm. The model performs forced alignment at the character level. A single word mismatch shifts the alignment window, causing:

- Metallic or robotic artefacts
- Pitch drift mid-sentence
- Stutter or repetition at chunk boundaries

**Rules for the transcript:**
1. Type exactly what the speaker says, including filler words ("um", "uh")
2. Match punctuation as spoken (commas → micro-pause, full stop → longer pause)
3. Do not add phonetic spelling or IPA — unlike Kokoro, F5-TTS does not use IPA injection
4. If using the default bundled voice, the transcript is already set in `_DEFAULT_REF_TEXT` in [engine.py](../core/f5_tts/engine.py)

---

## 6. Preprocessing

F5-TTS preprocessing ([preprocessor.py](../core/f5_tts/preprocessor.py)) performs **character-count chunking** rather than Kokoro's token-count chunking.

### Why Different?

F5-TTS has a hard ~30-second context window per `infer()` call. At a meditative pace of 0.75, 200 characters ≈ 7–10 seconds of audio, which safely fits within the window.

Kokoro's preprocessor merges sentences into 100–150-token chunks (targeting its optimal batch size). That logic is not applicable here.

### What Preprocessing Does NOT Include

Unlike Kokoro's preprocessor, F5-TTS preprocessing does **not**:
- Expand digits to words (`42` → "forty two")
- Inject IPA phonemes (`chakra` → `[chakra](/tʃɑːkɹə/)`)
- Insert prosody commas at phrasing boundaries

F5-TTS performs its own G2P (grapheme-to-phoneme) from raw prose. Passing Kokoro's IPA injection format would corrupt F5's input, since the brackets and IPA characters would be read as literal text.

**Write natural, unpunctuated meditation prose. F5-TTS handles it.**

### Pause Markers

All pause marker formats supported by Kokoro are also supported by F5-TTS preprocessing:

| Marker | Pause duration |
|---|---|
| `[pause:Xs]` | X seconds |
| `[N second pause]` | N seconds |
| `[breath]` / `[inhale]` / `[exhale]` | 1.2 seconds |
| Double newline | 6.5 seconds |

---

## 7. Postprocessing

F5-TTS postprocessing ([postprocessor.py](../core/f5_tts/postprocessor.py)) provides two components:

### `crossfade_chunks()`

Stitches audio chunks with a 20 ms linear crossfade at each boundary. This is simpler than Kokoro's cosine-squared crossfade because F5-TTS produces clean sentence boundaries without the cold-start prosody drift that requires Kokoro's pre-roll and fade-in.

### `F5MasteringEngine`

Implements the same two-method interface as `KokoroMasteringEngine`:

| Method | Purpose |
|---|---|
| `restore_vocals(audio, sr)` | Stub — Vocos is pre-clean, returns audio unchanged |
| `master_vocals(audio, sr)` | EQ / de-ess / limiting at the mix sample rate |

**Why no spectral gating?** Kokoro's ISTFTNet vocoder produces a characteristic broadband hiss above 12 kHz, which requires spectral noise gating before mastering. Vocos produces a clean signal without this artefact — no gating is needed.

**Why 13 kHz lowpass (vs. Kokoro's 9.5 kHz)?** Kokoro's 12 kHz Nyquist limit means all spectral content above 12 kHz is aliasing artefact. Masking it at 9.5 kHz removes the artefact. Vocos has a broader native bandwidth; 13 kHz preserves its natural "air" without passing ultrasonic content.

The mastering chain:
```
HighpassFilter(80 Hz)         — remove sub-bass rumble
LowShelfFilter(200 Hz, +2 dB) — warmth
PeakFilter(400 Hz, -1.5 dB)   — mud cut
PeakFilter(3.5 kHz, +1.5 dB)  — presence / intelligibility
PeakFilter(7 kHz, +4 dB)      — air / clarity boost
Compressor(-20 dB, 3:1)       — gentle dynamic control
PeakFilter(7 kHz, -4 dB)      — de-ess (cancel the air boost)
LowpassFilter(13 kHz)         — remove ultrasonic content
Limiter(-0.5 dB)              — hard ceiling
```

---

## 8. Pipeline Integration

### New Parameters on `pipeline.generate()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tts_engine` | `str` | `"kokoro"` | `"kokoro"` or `"f5"` |
| `f5_ref_audio` | `str \| None` | `None` | Path to user-uploaded reference WAV |
| `f5_ref_text` | `str` | `""` | Verbatim transcript of the reference clip |

### Branching Points in pipeline.py

There are two branch points where the pipeline dispatches to the correct engine:

1. **Step 1 — Script parsing**: imports `core.f5_tts.preprocessor.prepare_segments` for F5, `core.kokoro_tts.preprocessor.prepare_segments` for Kokoro
2. **Step 2 — Model loading**: instantiates `F5Engine()` locally for F5 (never stored in `self.tts`), uses `self.tts` (KokoroEngine) otherwise
3. **Step 3 — Synthesis**: passes `voice=f5_voice` tuple for F5, `voice=voice` string for Kokoro
4. **Phase A — Mastering init**: instantiates `F5MasteringEngine` for F5, `KokoroMasteringEngine` for Kokoro

All downstream stages (upsample, Phase B mastering, voice FX, mixing, export) are **unchanged** — both mastering engines share the same `master_vocals(audio, sr)` interface.

### Memory Management

For F5-TTS, the engine is instantiated **locally per call** (not stored in `self`). After synthesis, `tts.unload_model()` + `del tts` are called explicitly before the music engine is loaded. This maintains the sequential VRAM allocation pattern — only one neural model is resident at a time.

---

## 9. UI Integration

The Gradio UI adds:

1. **TTS Voice Engine radio** — appears in the right column between the music quality radio and the Kokoro settings accordion. Default: `"Kokoro"`.

2. **F5-TTS Voice Settings accordion** — hidden by default; visible when F5-TTS is selected. Contains:
   - `gr.Audio` (upload) — reference voice clip (optional; leave empty for bundled default)
   - `gr.Textbox` — verbatim transcript (optional; leave empty for bundled default)

3. **Conditional visibility**: selecting F5-TTS hides the Kokoro settings accordion and shows F5-TTS settings, and vice versa. Both are hidden in "Instrumental Only" mode (no TTS needed).

---

## 10. Performance Notes

| Setting | Value | Notes |
|---|---|---|
| `nfe_step` | 32 | Production quality. Use 16 for fast script iteration (lower prosodic detail). |
| `sway_sampling_coef` | -1.0 | Enables sway sampling. Produces smoother, more natural prosody for monotone narration. Disable (set to 0) only if you notice timing artefacts. |
| `speed` | 0.75 | Default meditation pace. Kokoro default is 0.70; F5-TTS at 0.75 produces comparable real-time duration due to different timing models. |
| Device | MPS | Apple Silicon M-series. CPU fallback for non-Apple hardware. F5-TTS does NOT support MLX natively (unlike the ACE-Step LLM planner). |
| First-run | Slow | Model weights (~1.5 GB) are downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/`. |

### Expected Generation Speed (Apple M1 Max, MPS)

| Segment length | NFE=32 | NFE=16 |
|---|---|---|
| Short (1–2 sentences, ~150 chars) | ~3–6s | ~2–3s |
| Medium (3–4 sentences, ~400 chars after chunking) | ~6–12s | ~4–6s |

Long scripts are chunked into ≤200-character segments, so generation time scales linearly with the number of chunks.

---

## 11. Known Limitations

1. **No Sanskrit / yoga term phoneme injection** — Kokoro's preprocessor injects IPA pronunciation hints for terms like "chakra", "pranayama", and "namaste". F5-TTS does its own G2P; these terms will be pronounced according to the model's training data, which may not match the intended Sanskrit pronunciation.

2. **Fresh inference per chunk** — Each ≤200-character chunk is a separate `infer()` call with the same reference audio. There is no cross-chunk prosodic context; pacing and intonation may vary slightly between chunks on long scripts.

3. **Voice consistency over long scripts** — Zero-shot cloning is stochastic. The cloned voice is highly consistent within a single session but may drift subtly on very long meditations (>30 minutes). This is expected behaviour.

4. **30-second context ceiling** — F5-TTS has a hard inference limit of ~30 seconds per call. The 200-character chunking keeps every segment safely under this limit at meditative pace (0.75), but unusually long sentences without punctuation may still approach the ceiling. If you observe truncation, add punctuation to the script.

5. **No deterministic seed support** — F5-TTS does not expose a seed parameter in its public API. The `seed=` kwarg passed by the pipeline is silently ignored (absorbed by `**kwargs`). Generation is not reproducible across runs.
