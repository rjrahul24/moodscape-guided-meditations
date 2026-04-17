<!-- QUICK-REF ──────────────────────────────────────────────────────── -->
**Files:** `core/chatterbox_tts/engine.py`
**Class:** `ChatterboxEngine(SpeechEngine)` — `load_model()` / `synthesize()` / `unload_model()`
**Constants:** `_DEFAULT_EXAGGERATION=0.45` · `_DEFAULT_CFG_WEIGHT=0.2` · `_DEFAULT_SPEED=0.90` · `INTER_SENTENCE_PAUSE_SEC=1.0` · `ELLIPSIS_PAUSE_SEC=1.8` · `_TRIM_THRESHOLD_DB=-45.0`
**Contract:** Output — 24 kHz mono float32 · Model auto-downloads from HuggingFace (~2 GB, first run only)
**Script parsing:** Reuses `core/kokoro_tts/preprocessor.py :: prepare_segments()` (same segment format)
**Voice FX:** Uses `core/kokoro_tts/postprocessor.py :: build_chatterbox_voice_chain()` (Chatterbox-tuned chain)
**Tasks:**
- Tune emotion params → `engine.py` module-level constants (`_DEFAULT_EXAGGERATION`, `_DEFAULT_CFG_WEIGHT`)
- Add reference voices → drop `.wav` file in `core/chatterbox_tts/assets/reference_audio/`
- Tune noise reduction → `core/kokoro_tts/postprocessor.py :: reduce_synthesis_noise()`
- Tune inter-sentence pauses → `engine.py` constants (`INTER_SENTENCE_PAUSE_SEC`, `ELLIPSIS_PAUSE_SEC`)
**See also:** `docs/prompting_guides/chatterbox_tts_instructions.md` · `docs/ARCHITECTURE.md`
<!-- ────────────────────────────────────────────────────────────────── -->

# Chatterbox TTS Implementation Guide

### Emotion-Controlled Meditation Narration for MoodScape

---

## 1. Purpose

Chatterbox TTS is the third TTS engine available in MoodScape, alongside Kokoro and F5-TTS. It adds two capabilities that the other engines do not provide simultaneously: a continuous emotion exaggeration dial and zero-shot voice cloning from a reference clip that requires no verbatim transcript.

For meditation audio, the engine is used at moderate exaggeration (`0.45`) and low `cfg_weight` (`0.2`) to produce warm, deliberate delivery with natural expressiveness — warmer than Kokoro's fixed-voice style, and simpler to configure than F5-TTS voice cloning (which requires a paired transcript). Per-chunk RMS normalization, segment fades, and pyworld pitch humanization (shared with Kokoro) further enhance naturalness.

---

## 2. What is Chatterbox?

Chatterbox is an open-weights TTS model released by Resemble AI under the MIT license.

| Property | Value |
|----------|-------|
| Parameters | 0.5B |
| License | MIT |
| Developer | Resemble AI |
| Output sample rate | 24 000 Hz (matches pipeline `SAMPLE_RATE`) |
| Output channels | Mono |
| Output dtype | float32 |
| Model size (download) | ~2 GB (auto-downloads from HuggingFace on first run) |
| Runtime memory | ~4–6 GB |
| Paralinguistic tags | `[laugh]`, `[chuckle]`, `[cough]` |
| Emotion control | `exaggeration` dial (0.0–1.0) |
| Voice cloning | Zero-shot from 5–15s reference `.wav`; no transcript required |
| Package | `chatterbox-tts` (see `requirements.txt`) |

### Why Chatterbox fits meditation narration

- **Emotion dial** — `exaggeration=0.25` produces natural micro-expressiveness without drama. Kokoro's 82M model has no runtime emotion conditioning; F5-TTS has no emotion dial either.
- **Zero-shot cloning without a transcript** — F5-TTS cloning requires a verbatim transcript of the reference clip. Chatterbox clones from a bare `.wav` file, making it far easier to onboard new reference voices.
- **24 kHz native output** — matches `SAMPLE_RATE` in `core/speech_engine.py` exactly; no resampling within the engine under normal conditions.
- **SpeechEngine compliance** — implements `load_model / unload_model / synthesize / get_available_voices`, making all downstream pipeline stages (FX, mixing, mastering) fully engine-agnostic.

### Engine comparison

| Feature | Kokoro (82M) | F5-TTS | Chatterbox (0.5B) |
|---------|-------------|--------|-------------------|
| Voice blending presets | Yes (6 presets, SLERP) | No | No |
| Zero-shot voice cloning | No | Yes (needs transcript) | Yes (no transcript needed) |
| Emotion intensity dial | No | No | Yes (0.0–1.0) |
| Paralinguistic tags | No | No | Yes ([laugh], [chuckle], [cough]) |
| Native sample rate | 24 kHz | 24 kHz | 24 kHz |
| Runtime memory | ~1.4–2.7 GB | ~4–6 GB | ~4–6 GB |
| HF transcript required for cloning | N/A | Yes | No |

---

## 3. Hardware Requirements

| Requirement | Value |
|-------------|-------|
| Preferred device | MPS (Apple Silicon) |
| CPU fallback | Yes (automatic) |
| CUDA support | Yes (automatic if available) |
| Runtime memory (MPS) | ~4–6 GB |
| torch.load MPS patch | Applied at load time — required for Apple Silicon compatibility |

The engine detects the device at `load_model()` time using this priority: MPS → CUDA → CPU. No manual device configuration is needed.

A `torch.load` monkey-patch is applied during model loading (from the official Chatterbox `example_for_mac.py`) to force `map_location` to the active device. The original `torch.load` is always restored in a `finally` block regardless of whether loading succeeds or fails.

---

## 4. Installation

```bash
# Install via pip (already listed in requirements.txt)
pip install chatterbox-tts

# Model weights download automatically from HuggingFace on first run (~2 GB).
# No manual download step required.
```

The model is cached in `~/.cache/huggingface/` after the first download and loads from cache on subsequent runs.

---

## 5. Model Configuration

### Emotion and pacing parameters

| Parameter | Default | Range | Rationale |
|-----------|---------|-------|-----------|
| `exaggeration` | `0.45` | 0.0–1.0 | 0.0 = monotone; 0.45 = warm expressiveness ideal for meditation (warmth + care); 1.0 = dramatic narration |
| `cfg_weight` | `0.2` | 0.0–1.0 | Lower values produce slower, more deliberate pacing. `0.2` produces deliberate meditation delivery. Hardcoded in `app.py` (`chatterbox_cfg_weight=0.2`). |
| `speed` | `0.90` | 0.5–1.5 | Accepted by `synthesize()` for ABC compliance; Chatterbox pacing is controlled primarily via `cfg_weight`, not a direct speed multiplier |
| `reference_audio` | `None` | path or None | Optional `.wav` path for zero-shot voice cloning. When `None`, Chatterbox uses its built-in default voice |

### Meditation-optimized recommendations

| Use case | `exaggeration` | `cfg_weight` |
|----------|---------------|-------------|
| Sleep / deep relaxation | 0.25–0.35 | 0.15–0.20 |
| General guided meditation (default) | 0.45 | 0.20 |
| Body scan / awareness | 0.40–0.50 | 0.20–0.25 |
| Energetic morning / motivation | 0.55–0.70 | 0.25–0.30 |
| Dramatic / theatrical narration | 0.75–1.00 | 0.35–0.50 |

---

## 6. Voice Cloning

### Reference audio requirements

| Requirement | Value |
|-------------|-------|
| Duration | 5–15 seconds |
| Format | `.wav` (mono recommended) |
| Sample rate | Any (Chatterbox resamples internally) |
| Transcript | Not required |

### Assets directory

Reference audio files for named voices are placed in:

```
core/chatterbox_tts/assets/reference_audio/
```

Any `.wav` file placed in this directory is automatically discovered by `get_available_voices()`. The file stem becomes the voice ID, formatted as a title-case display name.

Example: `core/chatterbox_tts/assets/reference_audio/calm_female.wav` is exposed as:

```python
{"id": "calm_female", "name": "Calm Female", "description": "Reference: calm_female.wav"}
```

### Built-in default voice

When no reference audio is provided, `get_available_voices()` always returns the built-in Chatterbox voice as the first entry:

```python
{"id": "default", "name": "Default (Chatterbox)", "description": "Built-in Chatterbox voice, no reference needed."}
```

### Passing reference audio

Reference audio can be set at construction time or overridden per synthesize call:

```python
# At construction
engine = ChatterboxEngine(reference_audio="core/chatterbox_tts/assets/reference_audio/calm_female.wav")

# Per call (overrides init value)
voice_audio, voice_activity = engine.synthesize(segments, reference_audio="path/to/clip.wav")
```

If the path does not exist at construction time, a warning is logged and `reference_audio` is silently set to `None` (falls back to the default voice).

---

## 7. Implementation Walkthrough

### Class interface

| Method | Signature | Notes |
|--------|-----------|-------|
| `__init__` | `(exaggeration=0.25, cfg_weight=0.3, reference_audio=None)` | Validates reference audio path; clips params to [0.0, 1.0] |
| `load_model` | `() -> None` | Loads `ChatterboxTTS.from_pretrained(device=device)`; applies torch.load MPS patch; no-op if already loaded |
| `unload_model` | `() -> None` | `del self._model`; `gc.collect()`; clears MPS/CUDA cache |
| `synthesize` | `(segments, voice="", speed=0.90, progress_cb=None, seed=None, exaggeration=None, cfg_weight=None, reference_audio=None, **kwargs) -> tuple[ndarray, ndarray]` | Per-call params override init defaults; returns `(voice_audio, voice_activity)` |
| `get_available_voices` | `() -> list[dict]` | Scans `_REF_AUDIO_DIR`; always includes `"default"` entry |

### Synthesis flow

`synthesize()` processes each segment from the parsed script in order:

1. **Speech segments** — split into individual sentences via `_split_text_into_sentences()`, which preserves ellipses (`...` and `…`) intact and splits on `.`, `!`, `?` boundaries followed by whitespace.
2. **Per-sentence generation** — calls `self._model.generate(text=sentence, exaggeration=exag, cfg_weight=cfg)`. If `reference_audio` is set, also passes `audio_prompt_path=ref_audio`.
3. **Tensor conversion** — converts the model output to `float32` numpy array (handles both torch tensors and raw arrays).
4. **Trailing silence trim** — `_trim_trailing_silence(arr, sr)` uses `_TRIM_THRESHOLD_DB=-45.0` with a `_TRIM_TAIL_MS=50.0` preservation tail. Chatterbox appends silence padding to every generation; this step removes it.
5. **Sample rate guard** — if `self._model.sr` differs from `SAMPLE_RATE` (24000), resamples using `torchaudio.functional.resample`.
6. **Per-chunk processing** — each sentence is RMS-normalized to -23 dBFS (`normalize_chunk_rms`) and given segment fades (25ms pre-roll + 100ms fade-in + 50ms fade-out via `apply_segment_fades`) to mask cold-start artifacts and prevent volume jumps.
7. **Inter-sentence pauses** — after each sentence (except the final sentence of the final segment when the next segment is already a pause), appends room-tone generated by `generate_room_tone()` from `core/kokoro_tts/postprocessor.py`. Pause duration is `ELLIPSIS_PAUSE_SEC=1.8` if the sentence ends with `...` or `…`, else `INTER_SENTENCE_PAUSE_SEC=1.0`.
8. **Breath segments** — replaced with 0.6s room-tone pauses (Chatterbox generates naturalistic breathing via exaggeration; explicit breath WAVs sound artificial).
7. **Pause segments** — handled by `generate_room_tone(segment["duration_sec"], sr=SAMPLE_RATE)`.
9. **Assembly** — all audio chunks are concatenated with matching boolean `voice_activity` arrays.
10. **Spectral gating** — `reduce_synthesis_noise(voice_audio, sr=SAMPLE_RATE)` from `core/kokoro_tts/postprocessor.py` runs on the assembled audio for a stable noise profile.
11. **Pitch humanization** — `humanize_voice(voice_audio, sr=SAMPLE_RATE)` adds 3-layer micro-pitch variation (drift ±6 cents, vibrato ±3 cents, jitter ±2 cents) and 3% formant warmth shift via pyworld, transforming flat output into natural expressiveness.

### Deterministic seed

When `seed` is provided, all relevant random states are set before synthesis:

```python
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

### Memory management

```python
# unload_model() pattern
del self._model
self._model = None
gc.collect()
torch.mps.empty_cache()   # Apple Silicon
torch.cuda.empty_cache()  # CUDA (if available)
```

The pipeline performs an explicit `del tts` after `unload_model()` for Chatterbox (same as F5-TTS) because both are instantiated locally within `generate()` rather than held as persistent instance attributes:

```python
if tts_engine in ("f5", "chatterbox"):
    del tts
```

---

## 8. Voice FX Chain

Chatterbox uses a dedicated voice FX chain (`build_chatterbox_voice_chain()`) tuned for its flow-matching vocoder, which has different spectral characteristics than Kokoro's ISTFTNet.

The chain is built by `build_chatterbox_voice_chain()` in `core/kokoro_tts/postprocessor.py`.

### Signal flow

| Step | Plugin | Parameters | Purpose |
|------|--------|------------|---------|
| 1 | `NoiseGate` | threshold=−45 dB, ratio=2.5, attack=1ms, release=100ms | Less aggressive than Kokoro (−40 dB) — Chatterbox output is cleaner |
| 2 | `HighpassFilter` | cutoff=80 Hz | Removes sub-bass rumble and plosive energy |
| 3 | `PeakFilter` | −1.5 dB @ 400 Hz, Q=1.0 | Reduced mud cut (no ISTFTNet boxiness to correct) |
| 4 | `LowShelfFilter` | +2.0 dB @ 180 Hz | More warmth / proximity effect (counteracts Chatterbox's cooler character) |
| 5 | `Compressor` | 2.5:1 @ −22 dB, attack=15ms, release=150ms | Gentle dynamics control; 3–6 dB max gain reduction |
| 6 | `HighShelfFilter` | +1.5 dB @ 10 kHz | More air/presence for clarity and intimacy |
| 7 | `Convolution` | IR: `warm_studio`, mix=15% wet | Intimate room presence |
| 8 | `Limiter` | threshold=−1.0 dBTP | True-peak protection |

**Soft saturation** (tanh formula, drive=1.5, 12% wet) is applied by `apply_fx()` before the Pedalboard chain. This adds even harmonics for perceived warmth and runs before EQ so the harmonics are shaped by subsequent filtering.

The chain is applied at the upsampled mix sample rate (48 kHz) after the pipeline upsamples TTS audio. `apply_fx()` accepts audio of any length and does not truncate the reverb tail.

### Selecting a different IR

The convolution reverb IR can be changed via the `reverb_ir` parameter passed from `pipeline.generate()`:

```python
voice_chain = build_chatterbox_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
```

Available IRs: `warm_studio` (default), `wooden_hall`, `stone_chapel`. All IRs are in `assets/impulse_responses/`.

---

## 9. Pipeline Integration

### Where Chatterbox fits in `core/pipeline.py`

Chatterbox is a peer of Kokoro and F5-TTS in the pipeline. The engine is selected via `tts_engine="chatterbox"` and follows the same sequential loading pattern.

#### Step 1: Script parsing (progress 0.0)

Chatterbox reuses the Kokoro preprocessor:

```python
elif tts_engine == "chatterbox":
    from core.kokoro_tts.preprocessor import prepare_segments as _prepare
segments = _prepare(script)
```

The Kokoro preprocessor handles `[pause:Xs]`, `[breath]`, `\n\n` paragraph breaks, IPA injection, prosody punctuation, and token-aware chunking — all compatible with Chatterbox's synthesis interface.

#### Step 2: TTS engine instantiation (progress 0.05)

```python
elif tts_engine == "chatterbox":
    from core.chatterbox_tts.engine import ChatterboxEngine
    tts = ChatterboxEngine(
        exaggeration=chatterbox_exaggeration,
        cfg_weight=chatterbox_cfg_weight,
        reference_audio=chatterbox_reference_audio,
    )
    _progress(progress_cb, 0.05, "Loading Chatterbox TTS (0.5B)...")
tts.load_model()
```

#### Mix sample rate

Chatterbox forces `mix_sr = 48000` regardless of music engine selection:

```python
mix_sr = 48000 if (use_lyria or use_acestep or use_heartmula or tts_engine in ("f5", "chatterbox")) else TARGET_SR
```

`TARGET_SR` (44 100 Hz) is only used when no music engine is active and neither F5 nor Chatterbox is selected.

#### Step 3: Synthesis (progress 0.05 → 0.40)

```python
elif tts_engine == "chatterbox":
    voice_audio, voice_activity = tts.synthesize(
        segments, speed=speed, progress_cb=tts_progress,
        seed=seed,
    )
```

Note: `exaggeration` and `cfg_weight` are set at construction time, not passed to `synthesize()` from the pipeline. The pipeline only passes `speed` and `seed`.

#### Post-synthesis steps (before voice FX)

After synthesis and before the voice FX chain, the pipeline applies:

1. **Upsample** — `audio_processor.upsample_audio(voice_audio, from_sr=24000, to_sr=48000, high_accuracy=True)` via `librosa soxr_vhq`. The 24→48 kHz ratio is an exact integer (×2), making it aliasing-free.
2. **Neural denoising** — `enhance_voice_deepfilter(voice_audio, sr=48000)` from `core/deepfilter_enhancer.py`. DeepFilterNet MLX (2.1M params) removes synthesis noise at 48 kHz using learned spectral masks. Falls back to `noisereduce` spectral gating if `mlx-audio` is not installed. Applied to all 48 kHz TTS paths.

#### Step 4: Unload (progress 0.40)

```python
tts.unload_model()
if tts_engine in ("f5", "chatterbox"):
    del tts
```

Explicit `del tts` frees the local reference immediately rather than waiting for garbage collection.

#### Step 7: Voice FX (progress 0.72)

```python
elif tts_engine == "chatterbox":
    from core.kokoro_tts.postprocessor import build_chatterbox_voice_chain, apply_fx
    voice_chain = build_chatterbox_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
```

`build_chatterbox_voice_chain()` is used (tuned for Chatterbox's flow-matching vocoder). `apply_fx()` runs soft saturation before the Pedalboard chain and does not truncate the reverb tail.

---

## 10. UI Integration

### Gradio controls

| Control | Type | Default | Range / Options |
|---------|------|---------|----------------|
| Voice Engine radio | `gr.Radio` | `"Kokoro"` | `["Kokoro", "F5-TTS", "Chatterbox"]` |
| Emotion Intensity slider | `gr.Slider` | `0.25` | 0.0–1.0, step 0.05 |
| Voice Reference | `gr.Audio` | None | Upload, type=filepath, `.wav` |

The Chatterbox settings group (`elem_id="chatterbox-group"`) is visible only when the Voice Engine radio is set to `"Chatterbox"` and the generation mode is not `"Instrumental Only"`.

When Chatterbox is selected, the speed slider label updates to `"Speaking Speed (pacing controlled by emotion intensity)"` and defaults to `0.90`.

### cfg_weight handling

`cfg_weight` is hardcoded at `0.3` in `app.py` and is not exposed as a UI control:

```python
chatterbox_cfg_weight=0.3,
```

### Mapping from UI to pipeline

```python
if tts_engine_choice == "Chatterbox":
    tts_engine = "chatterbox"
```

The `chatterbox_exaggeration` float and `chatterbox_ref_audio` filepath are passed directly to `pipeline.generate()`. `chatterbox_cfg_weight=0.3` is always passed regardless of the selected engine (it is ignored by non-Chatterbox paths).

---

## 11. Troubleshooting

| Problem | Symptom | Solution |
|---------|---------|---------|
| `torch.load` MPS crash | RuntimeError or unexpected tensor device mismatch on Apple Silicon | The engine patches `torch.load` before calling `from_pretrained()` and restores it in a `finally` block. This is automatic — do not call `ChatterboxTTS.from_pretrained()` directly outside `load_model()` on Apple Silicon. |
| Reference audio not found | Warning logged; synthesis continues with default voice | `ChatterboxEngine.__init__` validates the path at construction. If missing, `self._reference_audio` is set to `None`. Check the path is relative to the project root or absolute. |
| High exaggeration (> 0.5) sounds unnatural | Overdramatic delivery incompatible with meditation | Reduce `exaggeration` to 0.20–0.35. Values above 0.5 are for energetic/theatrical content only. |
| OOM during synthesis | Memory error on first run | Chatterbox loads after TTS unload in the pipeline (sequential loading). If loading Chatterbox standalone, ensure no music engine is loaded simultaneously. |
| DeepFilterNet fallback active | Log message: "Not available (install mlx-audio). Falling back to noisereduce spectral gating." | Install `mlx-audio` for the full neural denoiser. The fallback (`noisereduce`) still produces acceptable results. |
| Chatterbox outputs unexpected sample rate | `model.sr` differs from 24000 | The engine handles this automatically: `torchaudio.functional.resample` resamples to `SAMPLE_RATE` when `model.sr != SAMPLE_RATE`. |

---

## 12. Quick Reference

```python
# ── Module-level constants (core/chatterbox_tts/engine.py) ──────────────
_DEFAULT_EXAGGERATION    = 0.45    # Warm meditation delivery (warmth + care)
_DEFAULT_CFG_WEIGHT      = 0.2    # Slower, more deliberate pacing
_DEFAULT_SPEED           = 0.90   # Meditation pace (ABC compliance only)

INTER_SENTENCE_PAUSE_SEC = 1.0    # Room-tone pause between sentences
ELLIPSIS_PAUSE_SEC       = 1.8    # Contemplative pause after "..." or "…"

_TRIM_THRESHOLD_DB       = -45.0  # Trailing silence detection floor
_TRIM_TAIL_MS            = 50.0   # Preserved tail after trim point

# ── Shared constants from core/speech_engine.py ─────────────────────────
SAMPLE_RATE              = 24000  # All TTS engines output at 24 kHz

# ── Asset paths ──────────────────────────────────────────────────────────
_ASSETS_DIR    = Path("core/chatterbox_tts/assets")
_REF_AUDIO_DIR = Path("core/chatterbox_tts/assets/reference_audio")

# ── Pipeline defaults (app.py / pipeline.py) ────────────────────────────
# chatterbox_cfg_weight hardcoded to 0.3 in app.py (not a UI control)
# mix_sr = 48000 (forced for chatterbox paths in pipeline.py)
```

---

## 13. What NOT to Do

| Anti-pattern | Reason | Correct approach |
|-------------|--------|-----------------|
| Call `ChatterboxTTS.from_pretrained()` directly on Apple Silicon without the `torch.load` patch | Device mismatch crash | Always use `ChatterboxEngine.load_model()` which applies and restores the patch |
| Set `exaggeration > 0.5` for sleep or relaxation meditation | Over-dramatic delivery; breaks the calm atmosphere | Use 0.15–0.30 for sleep/relaxation, 0.25 default for general meditation |
| Load Chatterbox while a music engine is still in memory | ~4–6 GB + music engine RAM exceeds 36 GB budget for long-form generation | Follow the sequential loading pattern: `tts.unload_model()` then load music |
| Pass `reference_audio` for a file path that does not exist | Silent fallback to default voice with a warning | Validate paths before passing; use `get_available_voices()` to enumerate discovered voices |
| Bypass `synthesize()` and call `self._model.generate()` directly in production | Bypasses trailing silence trim, pause injection, room-tone generation, and spectral gating | Always use `synthesize()` |
| Use F5-TTS preprocessor with Chatterbox | Unnecessary — F5 preprocessor uses character-aware chunking (300-char limit) tuned for F5's flow-matching architecture | Use the Kokoro preprocessor (`prepare_segments`) which the pipeline selects automatically |

---

## 14. Resources

| Resource | URL / Path |
|----------|-----------|
| Chatterbox model (HuggingFace) | https://huggingface.co/ResembleAI/chatterbox |
| Chatterbox GitHub | https://github.com/resemble-ai/chatterbox |
| Package (PyPI) | `chatterbox-tts` |
| Apple Silicon example | `example_for_mac.py` in the Chatterbox repo (source of the `torch.load` patch) |
| Voice FX chain | `core/kokoro_tts/postprocessor.py :: build_chatterbox_voice_chain()` |
| Noise reduction | `core/kokoro_tts/postprocessor.py :: reduce_synthesis_noise()` |
| Neural denoiser | `core/deepfilter_enhancer.py :: enhance_voice_deepfilter()` |
| Pipeline integration | `core/pipeline.py` (search "chatterbox") |
| Prompting guide | `docs/prompting_guides/chatterbox_tts_instructions.md` |
| Architecture deep-dive | `docs/ARCHITECTURE.md` |

---
