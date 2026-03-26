# MoodScape — Architecture Deep-Dive

Full technical reference. For quick navigation use `CLAUDE.md` tables first; come here for parameter-level detail.

---

## Pipeline Phase Breakdown

### Phase 1 — Script Parsing

**Kokoro path:** `core/kokoro_tts/preprocessor.py`
- `parse_script(script)` → raw segments with `[pause:Xs]`, `[breath]`, `\n\n` markers
- `preprocess_for_meditation(text)` — digits/abbrevs → `text_utils.expand_text()`, contractions, IPA phoneme injection (30+ Sanskrit/yoga terms), prosody punctuation, sensory ellipses
- `merge_sentences_to_chunks(sentences)` — token-aware chunking: target 100–150 tokens, hard ceiling 400
- `annotate_speed(sentence, base_speed)` — short phrases / questions / trailing clauses get slightly slower speed

**F5 path:** `core/f5_tts/preprocessor.py`
- `normalize_for_f5(text)` — shared expansion + F5-specific: colons → commas, ellipses → periods, em-dashes → commas, hyphenated compounds joined
- `split_into_chunks(text)` — sentence-boundary split respecting 300-char limit
- `[voice:phase_name]` tag supported for multi-phase voice switching

**Segment format:** `{"type": "speech"|"pause"|"breath", "text": str, "duration_sec": float, "voice_phase": str}`

**Pause durations (Kokoro):**
- Explicit `[pause:Xs]` → Xs
- `[breath]` tag → 1.2s
- Paragraph break `\n\n` → 6.5s (`_PARAGRAPH_PAUSE_SEC`)
- Inter-sentence → 0.8s (`INTER_SENTENCE_PAUSE_SEC`)
- After `...` → 1.2s (`ELLIPSIS_PAUSE_SEC`)

---

### Phase 2 — TTS Synthesis

**SpeechEngine ABC** (`core/speech_engine.py`):
```python
SAMPLE_RATE = 24000  # enforced contract

class SpeechEngine(ABC):
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def synthesize(
        self, segments: list[dict], voice: str, speed: float, progress_cb=None
    ) -> tuple[np.ndarray, np.ndarray]:
        # Returns: (voice_audio float32 mono 24kHz, voice_activity bool same length)
    def get_available_voices(self) -> list[dict]: ...
```

**KokoroEngine specifics:**
- CPU-forced on Apple Silicon: `device = "cpu"` (MPS causes SIGBUS deallocation errors)
- Two KPipeline instances: `lang_code="a"` (default) + `lang_code="b"` (British voices: `bf_*`, `bm_*`)
- `trf=True` — transformer G2P for better phonemization
- Voice blending: `voice_manager.blend_voices()` (linear) or `slerp_blend()` (spherical, norm-preserving)
- Voice jitter: `add_voice_jitter(tensor, amount=0.001)` — per-sentence micro-variation
- 6 meditation presets: `balanced_calm`, `deep_rest`, `soft_whisper`, `golden_hour`, `earth_root`, `serene_sky`

**F5Engine specifics:**
- Vocos vocoder + F5TTS_v1_Base diffusion model on MPS
- Voice resolved at construction from `VoiceRegistry.scan()` — raises `FileNotFoundError` if missing
- Reference audio RMS-normalized to −20 dBFS before conditioning
- `_NFE_STEPS = 32` (production quality)
- Silero VAD: crop trailing non-speech + attenuate interior non-speech to 15% gain floor (`_VAD_GAIN_FLOOR = 0.15`)
- `_SWAY_COEF = -1.0` — sway sampling for smoother prosody
- Multi-phase: `[voice:phase_name]` selects reference audio phase from `voices.toml`

---

### Phase 3 — Music Generation

**Sequential unload pattern:** TTS must fully unload (+ `gc.collect()` + cache clear) before music engine loads.

**AceStepEngine** (`core/acestep_engine.py`):
- `load_model(model_type="sft")` — patches `ACESTEP_GENERATION_TIMEOUT=7200`, `DURATION_MAX=1200` before import
- `compile_model=True` — mandatory; one-time JIT ~135s, then ~4× faster per step
- Device: `"auto"` → MPS on Apple Silicon; DiT uses `use_mlx_dit=True`
- Long-form (>90s): `_generate_infinite()` — three phases:
  1. **Genesis** (60s): initial anchor segment
  2. **Cover continuation** (60s segments): decaying `audio_cover_strength` (0.85 → 0.80 → 0.75 floor); last 30s of previous as context
  3. **Boundary smoothing**: 5s ACE-Step repaint windows at each seam
- Story mode: `_generate_story()` — per-stage prompt + 6s equal-power crossfades
- `_enhance_prompt(user_prompt, duration_hint)` — MESA framework → `(caption, lyrics)` tuple

**HeartMulaEngine** (`core/heart_mula/engine.py`):
- Detects MLX vs. MPS backend at `load_model()` time
- **MLX simultaneous loading:** Both LM (bf16) + codec (fp32) loaded together (~12 GB). `mx.set_memory_limit(30GB)`, `mx.set_cache_limit(4GB)`.
- **Best-of-N selection:** Multiple candidates generated per segment; best selected via `compute_composite_score()` QA ranking
- **Scheduling:** `cfg_scale=1.8`, `temperature=0.75`, `top_k=30`, `codec_guidance_scale=1.25`, `codec_num_steps=12`
- **Token continuation:** Long-form segments reuse trailing tokens as context for seamless transitions
- **MPS path:** heartlib's `lazy_load=True` + `__call__` API (writes to temp file internally)
- HeartCodec dtype: **always fp32** — `mx.float32` (MLX) / `{"codec": torch.float32}` (MPS)
- Long-form (>240s): `_generate_segments()` — 240s segments with 8s cosine crossfades
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` set in engine module scope (not just `__init__`)

**LyriaEngine** (`core/lyria/engine.py`):
- Async WebSocket via `client.aio.live.music.connect(model="models/lyria-realtime-exp")`
- PCM bytes (int16 stereo 48kHz) → deinterleave → mono average → float32
- Sessions capped at 570s (9.5 min); longer durations split + 3s crossfade
- SynthID watermark embedded — do not strip or time-stretch

---

### Phase 4 — Stem Separation (optional)

**StemSeparator** (`core/stem_separator.py`):
- HT Demucs, 42M params, ~168 MB, CPU-only
- **Subprocess isolation** via `scripts/separate_worker.py`:
  1. Flush MLX/MPS caches in main process
  2. Write audio to `.npy` temp file (memory-mapped IPC)
  3. Launch: `python scripts/separate_worker.py <input.npy> <sr> <output.npy>`
  4. Load result; subprocess memory reclaimed by OS on exit
- Demucs native SR: 44.1 kHz (input resampled from 48kHz, output resampled back)
- Output: bass + other stems summed; drums + vocals discarded
- `_SEGMENT_SEC = 5.0` — chunked processing for low peak memory

---

### Phase 5 — TTS Upsampling

- 24 kHz → 48 kHz: `upsample_audio(audio, 24000, 48000, high_accuracy=True)` = `librosa soxr_vhq`
- Integer ratio (2×): `voice_activity` mask extended via `np.repeat(mask, 2)`
- Non-integer ratio (e.g. 24 → 44.1): `resample_highly_accurate()` (same soxr_vhq backend)
- Do not downsample then re-upsample — always upsample from the lower-rate source

---

### Phase 6 — Voice FX

**Kokoro:** `build_voice_chain(reverb_amount, ir_name)` + `apply_fx(audio, chain, sr)` — single unified Pedalboard pass. Tape saturation (drive=1.05, tanh soft-clip) applied before chain.

**F5-TTS:** `build_f5_voice_chain()` in `core/f5_tts/postprocessor.py` — adds split-band de-esser (4–8 kHz, −20dB threshold, 4:1 ratio, 0.5/10ms) before the standard chain.

Voice activity mask realigned after FX (reverb tail can alter array length).

---

### Phase 7 — Music FX + Vocal Pocket

Pre-mix LUFS normalization per engine (`pipeline.py`, step 8):
- Lyria: −16 LUFS
- ACE-Step: −14 LUFS
- HeartMuLa: −17 LUFS

Then engine-specific chain applied, followed by vocal pocket:
```
make_{engine}_music_chain()  →  apply_audio_fx()
make_vocal_pocket_chain()    →  apply_audio_fx()   # carves 300Hz/1kHz/3kHz lane
```

---

### Phase 8 — Mix

`mixer.mix()` sequence:
1. `overlay_tracks(voice, music, sr)` — `music_pre_roll_sec=8.0`, `music_post_roll_sec=15.0`, `music_volume_db=−17.0`
2. `apply_multiband_ducking(voice, music, ...)` — default when `multiband=True`:
   - 3-band Linkwitz-Riley crossover: low (<250 Hz, 25% duck), mid (250–4 kHz, full duck), high (>4 kHz, 50% duck)
   - `duck_amount_db` (configurable, default −12.0 in pipeline)
   - `attack_ms=80.0`, `release_ms=1000.0`, `hold_ms=1200.0`, `lookahead_ms=60.0`, `window_ms=50.0`
   - Hold time bridges inter-phrase gaps — prevents per-sentence pumping
   - Falls back to `apply_envelope_ducking()` when `multiband=False`
3. `apply_fades(audio, sr, fade_in_sec, fade_out_sec, curve="exponential")` — natural DAW-style curves

`apply_rms_ducking()` also exists in `mixer.py` (vectorized offline lookahead, 75ms shift) but is NOT called by `mix()`.

---

### Phase 9 — Master Chain

`make_master_chain()` applied inside `export_audio()` via 20-second chunk streaming:
- `HighpassFilter(30 Hz)` — subsonic removal
- `Compressor(1.5:1 @ −22 dB, 40ms attack, 300ms release)` — gentle bus glue (~1-2 dB GR)
- `Limiter(−1.5 dBTP, 400ms release)` — true-peak safety
- True-peak clip at ±0.841 linear (−1.5 dBTP) applied after chain

---

### Phase 10 — QA Checks

`run_qa_checks(audio, sr)` runs 11 checks (was 7 originally; added spectral smoothness, harmonic stability, onset density, dynamic range); `compute_composite_score()` used by music engines for retry/A-B selection (up to 3 attempts). Two additional stem-level checks available: `check_voice_music_ratio()` (≥15 dB during speech) and `check_ducking_smoothness()` (≤30 dB/s envelope rate).

---

### Phase 11 — Export

`export_audio(audio, sample_rate, output_format, target_sample_rate)`:
1. Pre-compute LUFS gain scalar (target −16.0 LUFS)
2. Stream in 20s chunks: apply master chain → resample to `target_sample_rate` → normalize → clip at ±0.841 linear (−1.5 dBTP) → write via Pedalboard `AudioFile`
3. WAV (lossless) or MP3; `target_sample_rate` = `export_sr` from `pipeline.py`

---

## FX Chains — Full Parameter Tables

### `build_voice_chain()` — Kokoro (`core/kokoro_tts/postprocessor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| pre | Tape saturation | tanh(audio × 1.05) before chain |
| 1 | NoiseGate | threshold=−42 dB, ratio=20:1 |
| 2 | HighpassFilter | cutoff=80 Hz |
| 3 | LowShelfFilter | cutoff=200 Hz, gain=+2.0 dB |
| 4 | PeakFilter | freq=350 Hz, gain=−2.0 dB, Q=1.0 (mud cut) |
| 5 | Compressor | threshold=−18 dB, ratio=2:1 |
| 6 | PeakFilter | freq=3 000 Hz, gain=−2.5 dB, Q=0.8 (presence control) |
| 7 | HighShelfFilter | cutoff=7 500 Hz, gain=−4.0 dB (de-harsh) |
| 8 | Convolution | IR file, wet=reverb_amount (0.0–0.5) |
| 9 | LowpassFilter | cutoff=9 500 Hz (Nyquist mask after reverb) |
| 10 | Limiter | threshold=−1.0 dBFS |

### `make_heartmula_music_chain()` (`core/audio_processor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | NoiseGate | threshold=−52 dB, ratio=2:1, attack=5ms, release=200ms |
| 2 | HighpassFilter | cutoff=60 Hz |
| 3 | LowShelfFilter | cutoff=100 Hz, gain=+1.5 dB |
| 4 | LowShelfFilter | cutoff=150 Hz, gain=+2.0 dB |
| 5 | PeakFilter | freq=220 Hz, gain=−1.0 dB, Q=0.7 |
| 6 | PeakFilter | freq=4 000 Hz, gain=−2.0 dB, Q=0.5 |
| 7 | HighShelfFilter | cutoff=9 500 Hz, gain=−2.0 dB |
| 8 | LowpassFilter | cutoff=14 000 Hz |
| 9 | Convolution | IR=stone_chapel, wet=0.15 (15%) |
| 10 | Compressor | threshold=−20 dB, ratio=2:1, attack=100ms, release=900ms |
| 11 | Limiter | threshold=−0.5 dBFS |

### `make_acestep_music_chain()` (`core/audio_processor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | NoiseGate | threshold=−50 dB, ratio=2:1, attack=1ms, release=100ms |
| 2 | HighpassFilter | cutoff=60 Hz |
| 3 | LowShelfFilter | cutoff=200 Hz, gain=+2.0 dB |
| 4 | PeakFilter | freq=3 000 Hz, gain=−2.0 dB, Q=1.5 (vocal pocket adds −2 dB more = −4 dB combined) |
| 5 | PeakFilter | freq=4 000 Hz, gain=−1.5 dB, Q=0.8 |
| 6 | PeakFilter | freq=6 000 Hz, gain=−2.0 dB, Q=1.0 (5–7 kHz gap fill) |
| 7 | HighShelfFilter | cutoff=8 000 Hz, gain=+0.5 dB |
| 8 | HighShelfFilter | cutoff=10 000 Hz, gain=−2.5 dB |
| 9 | LowpassFilter | cutoff=16 000 Hz |
| 10 | Compressor | threshold=−20 dB, ratio=2.0:1, attack=80ms, release=800ms |
| 11 | Limiter | threshold=−0.5 dBFS |

### `make_lyria_music_chain()` (`core/audio_processor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=60 Hz |
| 2 | PeakFilter | freq=250 Hz, gain=−1.5 dB, Q=0.8 (mud notch) |
| 3 | HighShelfFilter | cutoff=9 000 Hz, gain=−2.5 dB |
| 4 | Compressor | threshold=−18 dB, ratio=2:1, attack=80ms, release=500ms |
| 5 | Limiter | threshold=−0.5 dBFS |

### `make_vocal_pocket_chain()` (`core/audio_processor.py`)

Applied to music after engine-specific chain to carve spectral room for voice.

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=30 Hz |
| 2 | PeakFilter | freq=300 Hz, gain=−3.0 dB, Q=0.8 |
| 3 | PeakFilter | freq=1 000 Hz, gain=−2.0 dB, Q=0.7 |
| 4 | PeakFilter | freq=3 000 Hz, gain=−2.0 dB, Q=1.0 (presence pocket; combined −4 dB with music chain) |
| 5 | LowpassFilter | cutoff=12 000 Hz |

### `make_master_chain()` (`core/audio_processor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=30 Hz |
| 2 | Compressor | threshold=−22 dB, ratio=1.5:1, attack=40ms, release=300ms (gentle bus glue, ~1-2 dB GR) |
| 3 | Limiter | threshold=−1.5 dBTP, release=400ms |

### Convolution Reverb IR Catalog (`assets/impulse_responses/`)

| IR name | Character | Typical use |
|---------|-----------|-------------|
| `warm_studio` | Short decay, intimate | Body scans, breathwork |
| `wooden_hall` | Medium space, natural warmth | Visualizations, journeys |
| `stone_chapel` | Long decay, ethereal | Deep relaxation, expansive |

Default: `warm_studio`. Selectable in UI via "Reverb Room" dropdown.

---

## QA Checks

`core/qa_monitor.py` — `run_qa_checks(audio, sr)` runs all checks; `compute_composite_score()` returns weighted float ∈ [0, 1].

| Check function | Pass condition | Notes |
|----------------|---------------|-------|
| `check_silence_gaps(audio, sr, max_silence_sec=15.0)` | No gap > 15s | RMS threshold=0.001 |
| `check_lufs(audio, sr, target=-16.0, tolerance=2.0)` | Within ±2 dB of −16 LUFS | Via pyloudnorm |
| `check_clipping(audio, threshold=0.99)` | < 0.1% clipped | Samples at/near ±1.0 |
| `check_spectral_balance(audio, sr)` | warmth(100–300Hz) ≥ presence(2–5kHz) | Meditation should be warm |
| `check_silence_ratio(audio, sr)` | 15–70% silence | RMS threshold=0.001 |
| `check_spectral_rolloff(audio, sr, percentile=0.85, max_hz=8000)` | 85th-pct rolloff ≤ 8kHz | High values = metallic artifacts |
| `check_onset_strength(audio, sr, peak_mult=5.0)` | peak/median ratio < 5.0 | Detects harsh transients |
| `check_spectral_flatness(audio, sr, low=4000, high=12000, max=0.3)` | flatness in 4–12kHz < 0.3 | Values near 1.0 = white noise |
| `check_spectral_smoothness(audio, sr)` | centroid variance < 50 | Detects erratic spectral changes |
| `check_harmonic_stability(audio, sr)` | chroma autocorr > 0.85 | Ensures tonal consistency |
| `check_onset_density(audio, sr, max_per_sec=0.5)` | < 0.5 onsets/sec | Meditation should have sparse events |
| `check_dynamic_range(audio, sr)` | RMS std < 0.01 | Ensures smooth, even dynamics |

**`compute_composite_score()` weights:**

| Metric | Weight |
|--------|--------|
| Spectral balance (warmth) | 0.20 |
| Spectral rolloff | 0.20 |
| Clipping-free | 0.20 |
| Spectral flatness | 0.15 |
| Onset smoothness | 0.15 |
| LUFS proximity | 0.10 |

---

## Memory Management Patterns

### Sequential Engine Loading

```
Load TTS → synthesize() → unload_model() → gc.collect() → [cache clear]
Load Music → generate() → unload_model() → gc.collect() → [cache clear]
[Optional] Load Demucs subprocess → separate → subprocess exits (memory reclaimed by OS)
```

Never load two engines simultaneously. Peak memory per phase:
- Kokoro: ~200 MB (CPU RAM only)
- F5-TTS: ~1.5 GB (MPS)
- ACE-Step: ~8–12 GB (MLX unified RAM, with compile)
- HeartMuLa (LM + codec simultaneous): ~12 GB (MLX bf16 LM + fp32 codec)
- HT Demucs: ~168 MB (CPU, subprocess)

### HeartMuLa Simultaneous Loading (MLX)

Both LM (bf16) and codec (fp32) are loaded together (~12 GB total). Memory limits set before loading:

```python
mx.set_memory_limit(30 * 1024**3)  # 30 GB
mx.set_cache_limit(4 * 1024**3)    # 4 GB

# Load both models simultaneously
lm_model = load_lm_from_pretrained("./ckpt-mlx/heartmula/", dtype=mx.bfloat16)
codec_model = load_codec_from_pretrained("./ckpt-mlx/heartcodec/", dtype=mx.float32)

# Best-of-N generation with scheduling
tokens = lm_model.generate(tags, lyrics, cfg_scale=1.8, temperature=0.75, top_k=30)
audio = codec_model.decode(tokens, num_steps=12, guidance_scale=1.25)

# Token continuation for long-form: reuse last N tokens as context
# for seamless multi-segment generation

del lm_model, codec_model
gc.collect()
mx.clear_cache()
```

MPS path uses heartlib's `lazy_load=True` — two-phase lifecycle still managed internally.

### Demucs Subprocess Isolation

```python
# Main process
mx.clear_cache()          # MLX
torch.mps.empty_cache()   # MPS
np.save(input_npy, audio)
result = subprocess.run(["python", "scripts/separate_worker.py", input_npy, sr, output_npy])
audio_separated = np.load(output_npy)
```

The worker process holds Demucs weights; when it exits, all its memory is reclaimed by the OS — no manual unloading needed.

### MPS Memory Ceiling

`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` set as `os.environ` in `core/heart_mula/engine.py` module scope (before any torch import in that file). 0.7 × 36 GB = 25.2 GB ceiling. Values below 0.5 cause OOM during 3B LM generation.

### atexit Hook

`atexit.register(lambda: os._exit(0))` in `app.py` and `test_modes.py`. Suppresses MPS deallocation bus error (SIGBUS) on Python interpreter shutdown. Do not remove.

---

## Audio Contract Reference

### Full Sample Rate Chain

```
Kokoro TTS (24 kHz mono float32)
  → upsample_audio(24000 → 48000, soxr_vhq)
  → build_voice_chain() @ 48 kHz
  → mix() @ 48 kHz

F5-TTS (24 kHz mono float32)
  → upsample_audio(24000 → 48000, soxr_vhq)
  → build_f5_voice_chain() @ 48 kHz
  → mix() @ 48 kHz

ACE-Step 1.5 (48 kHz mono float32, native)
  → normalize_loudness(premix_lufs=-14)
  → make_acestep_music_chain() @ 48 kHz
  → make_vocal_pocket_chain() @ 48 kHz
  → mix() @ 48 kHz

HeartMuLa (48 kHz mono float32, native)
  → normalize_loudness(premix_lufs=-17)
  → make_heartmula_music_chain() @ 48 kHz
  → make_vocal_pocket_chain() @ 48 kHz
  → mix() @ 48 kHz

Lyria RealTime (48 kHz stereo int16 PCM → mono float32)
  → normalize_loudness(premix_lufs=-16)
  → make_lyria_music_chain() @ 48 kHz
  → make_vocal_pocket_chain() @ 48 kHz
  → mix() @ 48 kHz

mix() output (48 kHz mono float32)
  → export_audio() streaming: master_chain + resample to export_sr + LUFS normalize
  → WAV or MP3 at export_sr (44.1 or 48 kHz, user-selectable)
```

### Voice Activity Mask

- Type: `np.ndarray(dtype=bool)`, same length as `voice_audio`
- `True` = speech frame; `False` = silence/pause
- Used by `apply_envelope_ducking()` to detect voice onset
- After upsample (2×): extended via `np.repeat(mask, 2)`
- After voice FX (reverb tail can change length): mask trimmed/padded to match new length

---

## Prompt Engineering Summary

### ACE-Step — MESA Framework (`core/acestep_engine.py :: _enhance_prompt()`)

- **M**ood: emotional context (e.g. "peaceful, introspective, warm")
- **E**lements: instruments + textures (e.g. "singing bowls, soft piano, ambient pads")
- **S**tructure: song-form labels `[Intro]` `[Verse]` `[Bridge]` `[Outro]` (standard Qwen3 training vocab)
- **A**pplication: use case (e.g. "meditation background, sleep journey")
- Auto-prepended base tags: `ambient, meditation, calm, peaceful, warm, spacious, soft dynamics, gentle, soothing, high fidelity, studio quality, clean production`
- Auto-appended negatives: `no vocals, instrumental`

### HeartMuLa — Eight Pillars (`core/pipeline.py :: _enhance_heartmula_prompt()`)

Tag priority (high → low): Genre (95%) → Timbre (50%) → Mood (32%) → Instrument (25%) → Scene
- Max 7–9 tags total
- Temporal descriptor scales with duration: ≤90s → "extremely slow"; ≤300s → "long soft pads that barely move"; >300s → "soundscape stays flat"
- Structural lyrics: `[interlude]` only (suppresses vocal bias); `[intro]`/`[outro]` for structure
- Auto-appended: `"no drums, instrumental"`

### Lyria — Weighted Prompts (`core/lyria/prompts.py :: parse_weighted_prompts()`)

Syntax: `"Label: weight, Label2: weight2"` e.g. `"Hang Drum: 1.5, Piano: 0.8, Ambient Pads: 1.0"`
Controls: BPM (40–140), Density (0.0–1.0), Brightness (0.0–1.0), Guidance (default 4.0)

---

## Test Coverage Map

| Test file | What it covers |
|-----------|---------------|
| `unit-tests/test_mixer.py` | `apply_rms_ducking`, `apply_envelope_ducking`, `overlay_tracks`, `apply_fades`, `normalize_loudness`, `resample_for_export` |
| `unit-tests/test_audio_processor.py` | All 5 Pedalboard chains + `apply_fx()` + `upsample_audio()` |
| `unit-tests/test_qa_monitor.py` | All 11 QA checks + composite score |
| `unit-tests/test_meditation_mastering.py` | Full mastering chain (ducking → FX → export) |
| `unit-tests/test_stem_separator.py` | Demucs subprocess isolation |
| `unit-tests/test_voice_manager.py` | Kokoro voice blending, presets, British voice detection |
| `unit-tests/test_kokoro_postprocessor.py` | `process_chunk`, `crossfade_chunks`, `build_voice_chain` |
| `unit-tests/test_tts_engines.py` | `KokoroEngine` + `F5Engine` load/unload/synthesize |
| `unit-tests/test_text_preprocessor.py` | `expand_text`, `inject_phonemes`, `enhance_prosody_punctuation` |
| `unit-tests/test_script_parser.py` | `parse_script` — pause/breath/speech segments |
| `unit-tests/test_f5_preprocessor.py` | `normalize_for_f5`, `split_into_chunks`, VAD |
| `unit-tests/test_f5_params.py` | F5 parameter validation |
| `unit-tests/test_f5_phases.py` | Multi-phase voice switching |
| `unit-tests/test_f5_pacing.py` | WPM-based pacing, `fix_duration` |
| `unit-tests/test_heartmula_engine.py` | HeartMuLa lazy loading, segment generation |
| `unit-tests/test_acestep_engine.py` | ACE-Step generation, MESA prompt enhancement |
| `unit-tests/test_acestep_infinite.py` | Three-phase long-form generation |
| `unit-tests/test_stitch_client.py` | `StitchClient.generate_design_concept()` |
| `integration-tests/test_integration_modes.py` | Full pipeline (all mode combinations) |
| `integration-tests/test_stress.py` | Load testing, memory management across sessions |
