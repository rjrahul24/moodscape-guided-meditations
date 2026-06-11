# Common Gotchas

Hard-won lessons. The most load-bearing entries are repeated in `CLAUDE.md`; the full list lives here.

## Apple Silicon / MPS

- **MPS bus error on exit**: `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove.
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7**: Set per-engine at startup (e.g. in IndexTTS engine module). Values below 0.5 cause OOM during large LM generation.
- **PYTORCH_ENABLE_MPS_FALLBACK=1**: Set in `app.py` so ops without MPS kernels fall back to CPU instead of crashing.

## ACE-Step

- **Timeout fix**: Always pass `compile_model=True` to `initialize_service()` — without it generation takes ~9s/step and times out. First run has ~135s JIT overhead, then ~4× faster per step.
- **transformers version pin**: `>=4.55.0,<4.58.0`. The ACE-Step HF-cached remote code (`configuration_acestep_v15.py`) imports `layer_type_validation`, added in transformers 4.55 — older 4.5x fails ACE-Step model load with an ImportError. 4.58+ breaks ACE-Step the other way. (Hit 2026-06-11 with 4.52.1; fixed by upgrading to 4.57.6.)
- **ADG disabled**: `_USE_ADG=False`. SFT adherence is baked in; ADG doubles forward passes without quality benefit.
- **Local checkpoint dir**: `models/acestep/checkpoints/` (relative — run from project root).

## Kokoro TTS

- **Forced to CPU**: MPS causes deallocation bus errors with Kokoro. Do not switch device.
- **British voices need `lang_code="b"`**: Voices prefixed `bf_*` / `bm_*` require a separate `KPipeline(lang_code="b")`.
- **`trf=True`**: Transformer G2P produces better phonemization than the default for meditation prosody.

## F5-TTS

- **Requires verbatim transcript**: Each reference audio in `assets/speakers/` needs a matching `.txt` of the same slug under `assets/speakers/transcripts/`.
- **VAD on**: Silero VAD is loaded via `torch.hub.load('snakers4/silero-vad', 'silero_vad')` to trim silence at chunk boundaries.

## IndexTTS-2

- **MPS NaN**: BigVGANv2 vocoder may produce NaN values on MPS — mitigated by `torch.clamp(mel, -10, 10)`. Force `use_fp16=False, use_deepspeed=False, use_cuda_kernel=False` on Apple Silicon.
- **Checkpoints — manual download**: `huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=models/indextts2`. Engine raises `FileNotFoundError` with instructions if missing.
- **No transcript needed**: Unlike F5-TTS, IndexTTS-2 only needs a speaker reference WAV.
- **Calm vector is the default**: When no emotion audio is selected, `infer()` is called with `emo_vector=[0,…,0,1.0]` + `emo_alpha=0.70`. Do NOT use `use_emo_text` — the Qwen3 text path conflates "calm/serene" with "sad/melancholic".
- **Speed slider is a no-op**: v2 API does not expose reliable time-stretching (Issue #422). Pacing comes from the calm vector + preprocessor pause durations + 600ms inter-chunk room-tone gap.
- **Per-chunk VRAM cleanup**: Engine calls `torch.mps.empty_cache()` (or CUDA equivalent) after every chunk to mitigate the known BigVGAN/CFM memory leak (upstream Issue #364) on long sessions.
- **Meditation lexicon**: `INDEX_MEDITATION_LEXICON` in the preprocessor phoneticizes sanskrit/pali terms (Om, pranayama, savasana, …) AFTER the compound-hyphen stripper so the syllabification hyphens survive. Add new terms there.
- **Emotion is decoupled from speaker**: Emotion reference clips go in `assets/emotions/`. The Gradio UI also accepts a custom emotion upload.

## Mixing & Mastering

- **Active ducking function**: `mixer.mix()` calls `apply_breathing_duck()` — a deep, gradual, script/VAD-aware sidechain duck. Phrases are detected from the voice (`detect_phrases`), a predictive S-curve descends ~600 ms before each phrase, holds at `duck_amount_db` during speech, releases over ~1.5 s, and lifts slightly during pauses ≥1.5 s so the bed "breathes". Applied **fullband** so the whole bed drops.
- **Adaptive bed calibration**: `pipeline` auto-derives `music_volume_db` / `duck_amount_db` from measured stem short-term LUFS (`mixer.calibrate_music_bed`, golden-path targets 14.5/30.5 LU under voice). `MOODSCAPE_ADAPTIVE_BED=0` forces the legacy fixed (−16, −16). A user-moved duck slider still overrides the calibrated duck.
- **IndexTTS-2 pacing requires Rubber Band**: `_apply_meditation_pace` shells out to the `rubberband` CLI via pyrubberband. If missing, chunks are returned **unstretched** (no librosa phase-vocoder fallback — it smears voice into the "robotic" sound; opt back in only via `MOODSCAPE_INDEXTTS_PV_FALLBACK=1`). Install: `brew install rubberband` + `pip install pyrubberband`.
- **ACE-Step long-form modes**: `long_form_mode="auto"` uses loop mode above 300s (one ~4-min piece + `fit_to_length` crossfade looping) and the hardened evolve/repaint chain at 90–300s. Seeds are pinned per segment; seams are QA-checked with an STFT-crossfade fallback.
- **MLX lazy graphs cannot cross threads** (mlx 0.31): evaluating a graph built on another thread raises `There is no Stream(gpu, 0) in current thread`. ACE-Step's `generate_music_execute` ran diffusion in a watchdog `threading.Thread`, so every MLX DiT run silently fell back to the ~10× slower PyTorch path (the package's loguru `%s` warning swallows the exception). `AceStepEngine._patch_inline_generation_thread()` runs the target inline — keep it; without it ACE-Step is drastically slower.
- **No pedalboard `Limiter` anywhere**: pedalboard 0.9.23's `Limiter` inflates sub-threshold signals by ~+4.75 dB and adds broadband distortion ("static"). It was removed from `make_{upload,acestep,lyria}_music_chain()` and `make_master_chain()`. Don't reintroduce it.
- **Limiting is true-peak at export**: `export_audio()` does master EQ/glue → LUFS-normalize to −16 → `mixer.true_peak_limit()` to −1 dBTP (order matters: normalize first, then limit). `true_peak_limit` is a vectorized 4×-oversampled brickwall (no per-sample loop) and is transparent below threshold.
- **Default levels**: `duck_amount_db=-16` (`pipeline.py` / UI slider) = how low the bed sits under speech; `music_volume_db=-16` baseline in `mix()`. Lower (more negative) duck = quieter under speech.
- **Mix sample rate**: All music-source paths use `mix_sr = 48000` (`pipeline.py`). The 44.1 kHz fallback applies only when no music source is active.
- **Export target**: `−16 LUFS`, `−1 dBTP` ceiling. Matches Apple Music and avoids platform re-limiting.

## Uploaded Instrumental (`music_model="upload"`)

- **Engine output contract is load-bearing**: `UploadMusicEngine.generate()` must return **mono float32 @ 48 kHz, exactly `round(total_duration_sec*48000)` samples** — that is what lets the upload reuse the ACE-Step/Lyria mix/duck/master path unchanged. Don't return stereo or a differently-sized array.
- **Stem separation is skipped for uploads**: guarded by `if stem_separation and not use_upload:` in `pipeline.py`. The user's file is already an instrumental; running Demucs on it is wasteful and not wanted. Don't remove the guard.
- **Don't apply fades in the upload engine**: `fit_to_length()` returns a bare fitted array. Pre/post-roll and fades are added later by `mixer.mix()` — applying them in the engine would double-fade.
- **Decoding uses `pedalboard.io.AudioFile`** (libsndfile/ffmpeg) so mp3/m4a/etc. work; UI validates the extension against `{.wav,.mp3,.flac,.ogg,.m4a,.aiff,.aif}` before the pipeline runs.
