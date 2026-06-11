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

**AceStepEngine** (`core/acestep/engine.py`):
- `load_model(model_type="sft")` — patches `ACESTEP_GENERATION_TIMEOUT=7200`, `DURATION_MAX=1200` before import
- `compile_model=True` — mandatory; one-time JIT ~135s, then ~4× faster per step
- Device: `"auto"` → MPS on Apple Silicon; DiT uses `use_mlx_dit=True`
- Long-form (>90s) — two strategies, selected via `long_form_mode` ("auto" | "loop" | "evolve"; pipeline param `acestep_long_form_mode`, UI radio):
  - **Loop** (auto default above 300s): `_generate_looped()` — one ~4-min piece (genesis + 2-3 repaints, whole-piece composite-QA retry ≤2 attempts), then looped to target with `fit_to_length()` equal-power crossfades (8s). Few seams, deterministic, ~4× faster for 10-15 min beds.
  - **Evolve**: `_generate_infinite()` — genesis (90s) + chained repaint continuations (20s context, 60s new per call), hardened with per-segment seed pinning (`seed + seg_num`), per-segment composite-QA retry (threshold 0.6, one offset-seed retry), and seam validation (`_seam_discontinuity_db` log-band energy check → 3s STFT crossfade fallback when > 6 dB)
- Seed pinned end-to-end via `GenerationConfig(use_random_seed=False, seeds=[seed])` — ACE-Step is seed-sensitive; the pipeline forwards its session seed
- Story mode: `_generate_story()` — per-stage prompt + 6s equal-power crossfades; per-stage seed = `seed + stage_index`
- `_enhance_prompt(user_prompt, duration_hint)` — MESA framework → `(caption, lyrics)` tuple

**LyriaEngine** (`core/lyria/engine.py`):
- Async WebSocket via `client.aio.live.music.connect(model="models/lyria-realtime-exp")`
- PCM bytes (int16 stereo 48kHz) → deinterleave → mono average → float32
- Sessions capped at 570s (9.5 min); longer durations split + 3s crossfade
- SynthID watermark embedded — do not strip or time-stretch

**UploadMusicEngine** (`core/upload_music/engine.py`) — `music_model="upload"`:
- No model/weights; `load_model()`/`unload_model()` are no-ops (contract symmetry)
- `generate(prompt, total_duration_sec, …)` ignores `prompt` and all gen kwargs — it
  decodes the uploaded file via `pedalboard.io.AudioFile.resampled_to(48000)`, downmixes
  to mono float32, then fits to exactly `round(total_duration_sec*48000)` samples.
- **Length fitting** (`arrange.fit_to_length`): equal length → used as-is; longer → trim
  tail (master fade-out hides the cut); shorter → seamless loop with 500 ms equal-power
  crossfades (auto-shrinks to ≤25% of source, hard-tile fallback under ~4 ms).
- Output is byte-identical in contract to ACE-Step/Lyria (mono float32 @ 48 kHz, exact
  length), so the rest of the pipeline treats it uniformly. **Stem separation is skipped**
  for uploads (the file is already an instrumental). A `FitReport` is surfaced in the
  pipeline status message.

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
- Uploaded instrumental: −16 LUFS (uses `make_upload_music_chain()`; no noise reduction)
- : −17 LUFS

Then engine-specific chain applied, followed by vocal pocket:
```
make_{engine}_music_chain()  →  apply_audio_fx()
make_vocal_pocket_chain()    →  apply_audio_fx()   # carves 300Hz/1kHz/3kHz lane
```

---

### Phase 8 — Mix

`mixer.mix()` sequence:
1. `overlay_tracks(voice, music, sr)` — `music_pre_roll_sec=8.0`, `music_post_roll_sec=15.0`, `music_volume_db=−16.0`
2. `apply_breathing_duck(voice, music, sr, duck_depth_db)` — deep, gradual, script/VAD-aware sidechain duck:
   - `detect_phrases()` finds speech phrases from the voice (RMS-envelope VAD; merge gaps <250 ms, drop <150 ms)
   - `_script_gain_db()`: predictive cubic-S-curve descent starting ~600 ms before each phrase, holds at `duck_depth_db` (default −16) during speech, S-curve release over ~1.5 s, and lifts +1.5 dB during pauses ≥1.5 s so the bed "breathes"; zero-phase smoothed (~6 Hz)
   - `_reactive_gain_db()`: vectorized envelope-follower safety net (200 Hz–4 kHz detector), combined via `combine_script_with_reactive()` (script lift wins; else the deeper of the two)
   - Applied **fullband** (whole bed drops, not just mids), so speech sits "very low" while pauses recover
3. `apply_fades(audio, sr, fade_in_sec, fade_out_sec, curve="exponential")` — natural DAW-style curves

---

### Phase 9 — Master Chain

`make_master_chain()` applied inside `export_audio()`:
- `HighpassFilter(30 Hz)` — subsonic removal
- `Compressor(−12 dB, 1.5:1, 50/200 ms)` — gentle glue (engages lightly)
- `HighShelfFilter(12 kHz, +1 dB)` — master air
- **No `Limiter`** — pedalboard 0.9.23's Limiter inflates level (~+4.75 dB) and adds broadband distortion. True-peak limiting to −1 dBTP is done by `mixer.true_peak_limit()` after LUFS normalization (see Phase 11).

---

### Phase 10 — QA Checks

`run_qa_checks(audio, sr)` runs 11 checks (was 7 originally; added spectral smoothness, harmonic stability, onset density, dynamic range); `compute_composite_score()` used by music engines for retry/A-B selection (up to 3 attempts). Two additional stem-level checks available: `check_voice_music_ratio()` (≥15 dB during speech) and `check_ducking_smoothness()` (≤30 dB/s envelope rate).

---

### Phase 11 — Export

`export_audio(audio, sample_rate, output_format, target_sample_rate)`:
1. Apply master chain (EQ/glue, no limiter) to the whole array
2. LUFS-normalize to target (−16.0 LUFS) **first**, then `true_peak_limit()` to −1 dBTP (order is critical — normalizing after limiting would re-exceed the ceiling)
3. Stream in 20s chunks: resample to `target_sample_rate` → final safety clip at ±0.891 (−1 dBTP) → write via Pedalboard `AudioFile`
4. WAV (lossless) or MP3; `target_sample_rate` = `export_sr` from `pipeline.py`

---

## FX Chains — Full Parameter Tables

### `build_voice_chain()` — Kokoro (`core/kokoro_tts/postprocessor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | NoiseGate | threshold=−40 dB, ratio=2.5 |
| 2 | HighpassFilter | cutoff=80 Hz |
| 3 | PeakFilter | freq=400 Hz, gain=−2.5 dB, Q=1.0 (mud cut) |
| 4 | LowShelfFilter | cutoff=200 Hz, gain=+1.5 dB (warmth) |
| 5 | Compressor | threshold=**−28 dB**, ratio=2.5:1, attack=15ms, release=150ms |
| 6 | HighShelfFilter | cutoff=10 000 Hz, gain=+1.0 dB (air) |
| 7 | Convolution | IR file, wet=**0.18** default (18%) |
| 8 | Limiter | threshold=−1.0 dBFS |

### `make_acestep_music_chain()` (`core/audio_processor.py`)

Minimal chain — ACE-Step's VAE output is clean and doesn't need heavy processing.

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | NoiseGate | threshold=−55 dB, ratio=2:1, attack=1ms, release=100ms |
| 2 | HighpassFilter | cutoff=60 Hz |
| 3 | LowShelfFilter | cutoff=200 Hz, gain=+1.5 dB |
| 4 | PeakFilter | freq=3 000 Hz, gain=−1.5 dB, Q=1.0 (vocal pocket adds −1.5 dB more = −3 dB combined) |
| 5 | LowpassFilter | cutoff=16 000 Hz |
| 6 | Compressor | threshold=−20 dB, ratio=2.0:1, attack=80ms, release=800ms |
| — | (Convolution warm_studio reverb @ 8% wet appended when the IR file exists; **no Limiter** — true-peak limiting is at export) |

### `make_lyria_music_chain()` (`core/audio_processor.py`)

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=60 Hz |
| 2 | PeakFilter | freq=250 Hz, gain=−1.5 dB, Q=0.8 (mud notch) |
| 3 | PeakFilter | freq=4 500 Hz, gain=−2.0 dB, Q=0.7 (upper-mid presence control) |
| 4 | HighShelfFilter | cutoff=9 000 Hz, gain=−2.5 dB |
| 5 | Compressor | threshold=−18 dB, ratio=2:1, attack=80ms, release=500ms |
| — | (**no Limiter** — true-peak limiting is at export) |

### `make_upload_music_chain()` (`core/audio_processor.py`)

Deliberately light — an uploaded file is already a finished production, so only
protection + a small speech pocket are applied (the ducker and vocal-pocket chain do
the rest). No noise reduction.

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=30 Hz (subsonic) |
| 2 | PeakFilter | freq=2 000 Hz, gain=−2.0 dB, Q=0.7 (static speech pocket) |
| 3 | LowpassFilter | cutoff=14 000 Hz (safety) |
| — | (**no Limiter** — true-peak limiting is at export) |

### `make_vocal_pocket_chain()` (`core/audio_processor.py`)

Applied to music after engine-specific chain to carve spectral room for voice.

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=30 Hz |
| 2 | PeakFilter | freq=300 Hz, gain=−2.0 dB, Q=0.8 |
| 3 | PeakFilter | freq=1 000 Hz, gain=−1.0 dB, Q=0.7 |
| 4 | PeakFilter | freq=3 000 Hz, gain=−1.5 dB, Q=1.0 (presence pocket; combined −3 dB with ACE-Step chain) |
| 5 | LowpassFilter | cutoff=12 000 Hz |

### `make_master_chain()` (`core/audio_processor.py`)

EQ + gentle glue only. The peak limiter is **not** in this chain — `export_audio()`
LUFS-normalizes then applies `mixer.true_peak_limit()` (−1 dBTP). The pedalboard
`Limiter` was removed (it inflated level ~+4.75 dB and added broadband distortion).

| # | Plugin | Key params |
|---|--------|-----------|
| 1 | HighpassFilter | cutoff=30 Hz |
| 2 | Compressor | threshold=−12 dB, ratio=1.5:1, attack=50ms, release=200ms (gentle glue) |
| 3 | HighShelfFilter | cutoff=12 000 Hz, gain=+1.0 dB (air) |

True-peak limiting (`mixer.true_peak_limit`, 4×-oversampled vectorized brickwall, −1 dBTP) runs in `export_audio()` after normalization.

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
- IndexTTS-2: ~6 GB (MPS, fp32)
- ACE-Step: ~8–12 GB (MLX unified RAM, with compile)
- HT Demucs: ~168 MB (CPU, subprocess)

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

`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` set as `os.environ` in `core//engine.py` module scope (before any torch import in that file). 0.7 × 36 GB = 25.2 GB ceiling. Values below 0.5 cause OOM during 3B LM generation.

### atexit Hook

`atexit.register(lambda: os._exit(0))` in `app.py`. Suppresses MPS deallocation bus error (SIGBUS) on Python interpreter shutdown. Do not remove.

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

Lyria RealTime (48 kHz stereo int16 PCM → mono float32)
  → normalize_loudness(premix_lufs=-16)
  → make_lyria_music_chain() @ 48 kHz
  → make_vocal_pocket_chain() @ 48 kHz
  → mix() @ 48 kHz

Uploaded instrumental (any file → decode → resample 48 kHz → mono float32)
  → fit_to_length() loop/trim to exact duration
  → normalize_loudness(premix_lufs=-16)
  → make_upload_music_chain() @ 48 kHz
  → make_vocal_pocket_chain() @ 48 kHz
  → mix() @ 48 kHz   (stem separation skipped)

mix() output (48 kHz mono float32)
  → export_audio() streaming: master_chain + resample to export_sr + LUFS normalize
  → WAV or MP3 at export_sr (44.1 or 48 kHz, user-selectable)
```

### Voice Activity Mask

- Type: `np.ndarray(dtype=bool)`, same length as `voice_audio`
- `True` = speech frame; `False` = silence/pause
- Carried through the pipeline for alignment; ducking itself detects phrases from the voice signal (`detect_phrases()`)
- After upsample (2×): extended via `np.repeat(mask, 2)`
- After voice FX (reverb tail can change length): mask trimmed/padded to match new length

---

## Prompt Engineering Summary

### ACE-Step — MESA Framework (`core/acestep/engine.py :: _enhance_prompt()`)

- **M**ood: emotional context (e.g. "peaceful, introspective, warm")
- **E**lements: instruments + textures (e.g. "singing bowls, soft piano, ambient pads")
- **S**tructure: song-form labels `[Intro]` `[Verse]` `[Bridge]` `[Outro]` (standard Qwen3 training vocab)
- **A**pplication: use case (e.g. "meditation background, sleep journey")
- Auto-prepended base tags: `ambient, meditation, calm, peaceful, warm, spacious, soft dynamics, gentle, soothing, high fidelity, studio quality, clean production`
- Auto-appended negatives: `no vocals, instrumental`

### Lyria — Weighted Prompts (`core/lyria/prompts.py :: parse_weighted_prompts()`)

Syntax: `"Label: weight, Label2: weight2"` e.g. `"Hang Drum: 1.5, Piano: 0.8, Ambient Pads: 1.0"`
Controls: BPM (40–140), Density (0.0–1.0), Brightness (0.0–1.0), Guidance (default 4.0)

---

## Test Coverage Map

| Test file | What it covers |
|-----------|---------------|
| `tests/unit/test_mixer.py` | `overlay_tracks`, `apply_fades`, `normalize_loudness`, `resample_for_export` |
| `tests/unit/test_audio_processor.py` | All 5 Pedalboard chains + `apply_fx()` + `upsample_audio()` |
| `tests/unit/test_qa_monitor.py` | All 11 QA checks + composite score |
| `tests/unit/test_meditation_mastering.py` | Full mastering chain (ducking → FX → export) |
| `tests/unit/test_stem_separator.py` | Demucs subprocess isolation |
| `tests/unit/test_voice_manager.py` | Kokoro voice blending, presets, British voice detection |
| `tests/unit/test_kokoro_postprocessor.py` | `process_chunk`, `crossfade_chunks`, `build_voice_chain` |
| `tests/unit/test_tts_engines.py` | `KokoroEngine` + `F5Engine` load/unload/synthesize |
| `tests/unit/test_text_preprocessor.py` | `expand_text`, `inject_phonemes`, `enhance_prosody_punctuation` |
| `tests/unit/test_script_parser.py` | `parse_script` — pause/breath/speech segments |
| `tests/unit/test_f5_preprocessor.py` | `normalize_for_f5`, `split_into_chunks`, VAD |
| `tests/unit/test_f5_params.py` | F5 parameter validation |
| `tests/unit/test_f5_phases.py` | Multi-phase voice switching |
| `tests/unit/test_f5_pacing.py` | WPM-based pacing, `fix_duration` |
| `tests/unit/test_acestep_engine.py` | ACE-Step generation, MESA prompt enhancement |
| `tests/unit/test_acestep_infinite.py` | Long-form generation: repaint chain hardening (seed/QA/seam), loop mode, routing |
| `tests/integration/test_integration_modes.py` | Full pipeline (all mode combinations) |
| `tests/integration/test_stress.py` | Load testing, memory management across sessions |
