# Common Gotchas

Hard-won lessons. The most load-bearing entries are repeated in `CLAUDE.md`; the full list lives here.

## Apple Silicon / MPS

- **MPS bus error on exit**: `atexit.register(lambda: os._exit(0))` in `app.py` — do not remove.
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7**: Set per-engine at startup (e.g. in IndexTTS engine module). Values below 0.5 cause OOM during large LM generation.
- **PYTORCH_ENABLE_MPS_FALLBACK=1**: Set in `app.py` so ops without MPS kernels fall back to CPU instead of crashing.

## ACE-Step

- **Timeout fix**: Always pass `compile_model=True` to `initialize_service()` — without it generation takes ~9s/step and times out. First run has ~135s JIT overhead, then ~4× faster per step.
- **transformers version pin**: `>=4.51.0,<4.58.0` is required for ACE-Step compatibility — do not upgrade.
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

- **Active ducking function**: `mixer.mix()` calls `apply_multiband_ducking()` by default (`multiband=True`). Falls back to `apply_envelope_ducking()` when `multiband=False`. `apply_rms_ducking()` exists but is not used in production.
- **Don't reduce `hold_ms` below 800**: Default `hold_ms=1200` in `mix()` `_duck_kwargs` bridges slow meditation phrase gaps. Default `duck_amount_db=-12.0` lives in `pipeline.py`.
- **Mix sample rate**: All music-engine paths use `mix_sr = 48000` (`pipeline.py:213`). The 44.1 kHz fallback applies only when no music engine is active.
- **Export targets**: `−16 LUFS`, `−1.5 dBTP` ceiling. Matches Apple Music and avoids platform re-limiting.

## Uploaded Instrumental (`music_model="upload"`)

- **Engine output contract is load-bearing**: `UploadMusicEngine.generate()` must return **mono float32 @ 48 kHz, exactly `round(total_duration_sec*48000)` samples** — that is what lets the upload reuse the ACE-Step/Lyria mix/duck/master path unchanged. Don't return stereo or a differently-sized array.
- **Stem separation is skipped for uploads**: guarded by `if stem_separation and not use_upload:` in `pipeline.py`. The user's file is already an instrumental; running Demucs on it is wasteful and not wanted. Don't remove the guard.
- **Don't apply fades in the upload engine**: `fit_to_length()` returns a bare fitted array. Pre/post-roll and fades are added later by `mixer.mix()` — applying them in the engine would double-fade.
- **Decoding uses `pedalboard.io.AudioFile`** (libsndfile/ffmpeg) so mp3/m4a/etc. work; UI validates the extension against `{.wav,.mp3,.flac,.ogg,.m4a,.aiff,.aif}` before the pipeline runs.
