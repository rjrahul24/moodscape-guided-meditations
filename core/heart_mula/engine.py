"""HeartMuLa engine — text-to-music via heartlib on Apple Silicon.

Model: HeartMuLa-RL-oss-3B (LM) + HeartCodec-oss (codec)
Backend: MLX primary (heartlib-mlx), PyTorch MPS fallback (heartlib).
Lazy loading: HeartMuLa LM loads -> generates tokens -> unloads; then HeartCodec
              loads -> decodes to waveform -> unloads.  Preserves unified memory budget.
Output: Mono float32 at 48,000 Hz (HeartCodec native rate downmixed to mono).
Long-form: Segment-and-crossfade pipeline.  Each segment is up to MAX_SEGMENT_SEC
           (240 s).  Multiple segments are joined with equal-power cosine crossfades.

Target hardware: Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)
"""

import gc
import logging
import math
import os
import tempfile
import time

# Allow MPS to use up to 70% of unified memory (25 GB on 36 GB M1 Max).
# The old 0.4 (14 GB) was too restrictive for the 3B LM + generation buffers.
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"

import numpy as np

logger = logging.getLogger("moodscape.heartmula")

# -- Constants ----------------------------------------------------------------

NATIVE_SAMPLE_RATE = 48_000     # HeartCodec native output rate
TARGET_SAMPLE_RATE = 48_000     # Exported for pipeline.py to import

MAX_SEGMENT_SEC = 240.0         # HeartMuLa OSS max per inference call (4 min)
CROSSFADE_SEC = 4.0             # Equal-power cosine crossfade between segments

# Checkpoint directories (relative to project root)
CHECKPOINT_DIR = "./ckpt"           # PyTorch MPS path
CHECKPOINT_DIR_MLX = "./ckpt-mlx"   # MLX converted weights path


# -- Engine -------------------------------------------------------------------

class HeartMulaEngine:
    """Generates music via HeartMuLa (3B LM + HeartCodec).

    Uses MLX backend (heartlib-mlx) for fastest Apple Silicon inference.
    Falls back to official PyTorch MPS backend (heartlib) if MLX is
    unavailable.

    CRITICAL memory strategy — lazy loading:
      MLX:  Load LM (bf16) -> generate tokens -> unload LM -> load codec (fp32)
            -> detokenize -> unload codec.  Never both models in memory at once.
      MPS:  heartlib's built-in lazy_load=True handles the same lifecycle.

    HeartCodec MUST use fp32 — bf16 causes metallic artifacts.

    Public API mirrors AceStepEngine / LyriaEngine so the pipeline can
    treat all engines uniformly::

        engine = HeartMulaEngine()
        engine.load_model()
        audio = engine.generate(prompt, duration_sec, ...)
        engine.unload_model()

    Returns:
        Mono float32 numpy array at 48,000 Hz.
    """

    def __init__(self):
        self._generator = None
        self._model = None    # MLX LM handle
        self._codec = None    # MLX codec handle
        self.device = None
        self.initialized = False
        self._backend = None  # "mlx" or "mps"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self):
        """Detect backend (MLX or MPS) and validate checkpoints.

        Tries the MLX backend first for ~2x faster inference on Apple
        Silicon.  Falls back to the official heartlib PyTorch MPS path
        if heartlib-mlx is not installed.

        Actual weight loading is deferred to generation time (lazy loading)
        to keep peak memory usage within the 36 GB unified memory budget.
        """
        t0 = time.time()

        # -- Try MLX backend first -------------------------------------------
        try:
            import mlx.core as _mx  # noqa: F401
            from heartlib_mlx.heartmula import HeartMuLa as _HM  # noqa: F401
            from heartlib_mlx.heartcodec import HeartCodec as _HC  # noqa: F401

            from pathlib import Path
            ckpt = Path(CHECKPOINT_DIR_MLX)
            if not (ckpt / "heartmula").exists() or not (ckpt / "heartcodec").exists():
                raise FileNotFoundError(
                    f"MLX checkpoints not found at {ckpt}/heartmula and "
                    f"{ckpt}/heartcodec.  Run the weight conversion:\n"
                    f"  python -m heartlib_mlx.utils.convert"
                )

            self._backend = "mlx"
            self.device = "mlx"
            logger.info(
                "[HeartMuLa] MLX backend selected (heartlib-mlx) — "
                "checkpoint dir: %s", ckpt
            )
        except (ImportError, FileNotFoundError) as mlx_err:
            logger.info(
                "[HeartMuLa] MLX backend unavailable (%s), trying PyTorch MPS...",
                mlx_err,
            )

            # -- Fall back to PyTorch MPS ------------------------------------
            import torch
            from pathlib import Path

            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            if self.device == "cpu":
                logger.warning(
                    "[HeartMuLa] MPS not available — falling back to CPU. "
                    "Generation will be very slow."
                )

            ckpt = Path(CHECKPOINT_DIR)
            if not (ckpt / "HeartMuLa-oss-3B").exists():
                raise FileNotFoundError(
                    f"HeartMuLa LM weights not found at {ckpt}/HeartMuLa-oss-3B.\n"
                    "Download with:\n"
                    "  huggingface-cli download HeartMuLa/HeartMuLa-RL-oss-3B-20260123 "
                    f"--local-dir {ckpt}/HeartMuLa-oss-3B"
                )
            if not (ckpt / "HeartCodec-oss").exists():
                raise FileNotFoundError(
                    f"HeartCodec weights not found at {ckpt}/HeartCodec-oss.\n"
                    "Download with:\n"
                    "  huggingface-cli download HeartMuLa/HeartCodec-oss-20260123 "
                    f"--local-dir {ckpt}/HeartCodec-oss"
                )

            self._backend = "mps"
            logger.info(
                "[HeartMuLa] PyTorch MPS backend selected — device: %s, "
                "checkpoint dir: %s", self.device, ckpt,
            )

        self.initialized = True
        logger.info(
            "[HeartMuLa] load_model() complete (backend=%s) in %.1fs",
            self._backend, time.time() - t0,
        )

    def unload_model(self):
        """Release model references and free memory."""
        logger.info("[HeartMuLa] Unloading HeartMuLa...")

        self._generator = None
        self._model = None
        self._codec = None
        self.initialized = False

        gc.collect()

        if self._backend == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
        elif self._backend == "mlx":
            try:
                import mlx.core as mx
                # Force MLX to release ALL cached metal buffers back to the OS.
                # Without this, MLX retains GB of metal buffers even after model
                # deletion, preventing subsequent steps (Demucs) from allocating.
                mx.set_cache_limit(0)
                mx.clear_cache()
                logger.info(
                    "[HeartMuLa] MLX metal memory after cleanup — active: %.1f MB, cache: %.1f MB",
                    mx.get_active_memory() / 1e6,
                    mx.get_cache_memory() / 1e6,
                )
            except Exception:
                pass

        self._backend = None
        self.device = None
        logger.info("[HeartMuLa] Unloaded")

    # ------------------------------------------------------------------
    # Public generation entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        prompt_stages: list[tuple[str, float]] | None = None,
        lyrics: str | None = None,
        **kwargs,  # absorb unused kwargs (seed, melody_audio, keyscale, bpm, etc.)
    ) -> np.ndarray:
        """Generate background music and return mono float32 at 48 kHz.

        Args:
            prompt: Comma-separated style tags (e.g. "ambient, warm pads, meditation").
            total_duration_sec: Target duration in seconds.
            progress_cb: Optional callback ``(current_step, total_steps)``.
            prompt_stages: Story-mode list of ``(tags, duration_sec)`` pairs.
                When provided, one HeartMuLa call is made per stage and the
                results are crossfaded together.
            lyrics: Optional structural lyrics with section markers
                (e.g. "[intro]\\n\\n[verse]\\n\\n[outro]").
                Empty string or None = instrumental.
            **kwargs: Ignored; present for API compatibility with other engines.

        Returns:
            Mono float32 numpy array at 48,000 Hz.
        """
        if not self.initialized:
            raise RuntimeError(
                "HeartMulaEngine: call load_model() before generate()."
            )

        if prompt_stages is not None:
            return self._generate_story(prompt_stages, progress_cb, lyrics)

        if total_duration_sec <= MAX_SEGMENT_SEC:
            return self._generate_single(
                prompt, total_duration_sec, lyrics, progress_cb
            )

        return self._generate_long_form(
            prompt, total_duration_sec, lyrics, progress_cb
        )

    # ------------------------------------------------------------------
    # Internal: single segment generation
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        tags: str,
        duration_sec: float,
        lyrics: str | None,
        progress_cb,
    ) -> np.ndarray:
        """Generate a single segment up to MAX_SEGMENT_SEC."""
        logger.info(
            "[HeartMuLa] Generating %.0fs segment (backend=%s)...",
            duration_sec, self._backend,
        )
        t0 = time.time()

        if self._backend == "mlx":
            audio = self._generate_mlx(tags, duration_sec, lyrics)
        else:
            audio = self._generate_mps(tags, duration_sec, lyrics)

        elapsed = time.time() - t0
        logger.info(
            "[HeartMuLa] Segment generated in %.1fs (%.1fx realtime)",
            elapsed, duration_sec / max(elapsed, 0.01),
        )
        return audio

    def _generate_mlx(
        self, tags: str, duration_sec: float, lyrics: str | None,
    ) -> np.ndarray:
        """Generate audio using the MLX backend (heartlib-mlx).

        Implements manual lazy loading to avoid OOM:
        1. Load HeartMuLa LM (bf16) — ~6 GB
        2. Preprocess text + generate discrete audio codes
        3. Unload LM — free ~6 GB
        4. Load HeartCodec (fp32) — ~1-2 GB
        5. Detokenize codes → waveform
        6. Unload codec
        """
        import mlx.core as mx
        from heartlib_mlx.heartmula import HeartMuLa
        from heartlib_mlx.heartcodec import HeartCodec
        from heartlib_mlx.pipelines.music_generation import HeartMuLaGenPipeline, HeartMuLaGenConfig
        from tokenizers import Tokenizer
        from pathlib import Path

        ckpt = Path(CHECKPOINT_DIR_MLX)

        # Load config and tokenizer (lightweight, stays in memory)
        config = HeartMuLaGenConfig.from_pretrained(ckpt)
        tokenizer_path = ckpt / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists() else None

        # ── Phase 1: Load LM (bf16), generate codes ────────────────────
        logger.info("[HeartMuLa/MLX] Loading HeartMuLa LM (bf16)...")
        heartmula_lm = HeartMuLa.from_pretrained(ckpt / "heartmula", dtype=mx.bfloat16)

        # Build a pipeline object with LM only (codec=None placeholder)
        # We need the pipeline for preprocess() and generate() methods.
        # Create a minimal HeartCodec placeholder to satisfy __init__
        pipeline = HeartMuLaGenPipeline.__new__(HeartMuLaGenPipeline)
        pipeline.heartmula = heartmula_lm
        pipeline.heartcodec = None  # Will load separately
        pipeline.tokenizer = tokenizer
        pipeline.config = config
        pipeline._parallel_number = heartmula_lm.num_codebooks + 1

        # Preprocess text inputs
        inputs = pipeline.preprocess(lyrics=lyrics or "", tags=tags, cfg_scale=1.5)

        # Generate discrete audio codes
        logger.info("[HeartMuLa/MLX] Generating audio codes (%.0fs)...", duration_sec)
        codes = pipeline.generate(
            inputs=inputs,
            duration=duration_sec,
            temperature=1.0,
            top_k=50,
            cfg_scale=1.5,
        )
        # Ensure codes are fully evaluated before we unload the LM
        mx.eval(codes)

        # ── Unload LM ──────────────────────────────────────────────────
        logger.info("[HeartMuLa/MLX] Unloading LM to free memory...")
        del pipeline.heartmula
        del heartmula_lm
        del pipeline
        del inputs
        gc.collect()
        mx.set_cache_limit(0)
        mx.clear_cache()
        logger.info(
            "[HeartMuLa/MLX] Post-LM cleanup — active: %.1f MB, cache: %.1f MB",
            mx.get_active_memory() / 1e6, mx.get_cache_memory() / 1e6,
        )

        # ── Phase 2: Load HeartCodec (fp32), detokenize ────────────────
        logger.info("[HeartMuLa/MLX] Loading HeartCodec (fp32)...")
        heartcodec = HeartCodec.from_pretrained(ckpt / "heartcodec", dtype=mx.float32)

        # Detokenize: codes shape is (num_codebooks, num_frames)
        # HeartCodec expects (batch, frames, num_quantizers)
        codes_input = mx.transpose(codes, axes=(1, 0))[None, :, :]  # (1, frames, codebooks)

        num_frames = codes_input.shape[1]
        codec_frame_rate = heartcodec.config.frame_rate or 50.0
        duration_for_codec = num_frames / codec_frame_rate

        logger.info("[HeartMuLa/MLX] Detokenizing %d frames...", num_frames)
        audio_mx = heartcodec.detokenize(
            codes=codes_input,
            duration=duration_for_codec,
            num_steps=10,
            guidance_scale=1.25,
        )
        # Evaluate to get the result before cleanup
        mx.eval(audio_mx)

        # Convert to numpy before cleanup
        audio_data = np.array(audio_mx, dtype=np.float32)

        # ── Unload codec ────────────────────────────────────────────────
        logger.info("[HeartMuLa/MLX] Unloading HeartCodec...")
        del heartcodec
        del codes
        del codes_input
        del audio_mx
        gc.collect()
        mx.set_cache_limit(0)
        mx.clear_cache()
        logger.info(
            "[HeartMuLa/MLX] Post-codec cleanup — active: %.1f MB, cache: %.1f MB",
            mx.get_active_memory() / 1e6, mx.get_cache_memory() / 1e6,
        )

        sr = config.sample_rate or NATIVE_SAMPLE_RATE
        return self._postprocess(audio_data, sr)

    def _generate_mps(
        self, tags: str, duration_sec: float, lyrics: str | None,
    ) -> np.ndarray:
        """Generate audio using the PyTorch MPS backend (official heartlib).

        Uses heartlib's built-in lazy_load=True which:
        1. Loads HeartMuLa LM on first .mula access
        2. _forward() generates tokens, then calls _unload() to free the LM
        3. postprocess() loads HeartCodec on .codec access, detokenizes, saves
        4. _unload() frees the codec

        We use the __call__ API (saves to temp file) because the internal API
        assumes the full lifecycle is managed by __call__.
        """
        import torch
        import torchaudio
        from heartlib import HeartMuLaGenPipeline

        device = torch.device(self.device)
        dtype = {
            "mula": torch.bfloat16 if self.device == "mps" else torch.float32,
            "codec": torch.float32  # CRITICAL: fp32 for codec, bf16 = metallic artifacts
        }

        logger.info("[HeartMuLa/MPS] Loading pipeline (lazy_load=True)...")
        pipeline = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=CHECKPOINT_DIR,
            device=device,
            dtype=dtype,
            version="3B",
            lazy_load=True,
        )

        # Use __call__ which handles the full LM -> unload -> codec -> save lifecycle
        # The library saves output to a file, so we use a temp file and read it back
        inputs = {
            "tags": tags,
            "lyrics": lyrics or "",
        }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            max_len_ms = int(duration_sec * 1000)
            logger.info("[HeartMuLa/MPS] Generating %.0fs of audio...", duration_sec)
            pipeline(
                inputs,
                save_path=tmp_path,
                max_audio_length_ms=max_len_ms,
                temperature=1.0,
                topk=50,
                cfg_scale=1.5,
            )

            # Read back the generated audio
            audio_data, sr = torchaudio.load(tmp_path)
            logger.info("[HeartMuLa/MPS] Audio loaded: shape=%s, sr=%d", audio_data.shape, sr)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Pipeline already called _unload(), but ensure full cleanup
        del pipeline
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return self._postprocess(audio_data, int(sr))

    # ------------------------------------------------------------------
    # Internal: long-form generation (>240s)
    # ------------------------------------------------------------------

    def _generate_long_form(
        self,
        tags: str,
        total_duration_sec: float,
        lyrics: str | None,
        progress_cb,
    ) -> np.ndarray:
        """Generate audio longer than MAX_SEGMENT_SEC via segment-and-crossfade.

        Divides total_duration_sec into N segments of at most MAX_SEGMENT_SEC
        each.  Generates each segment with the same tags prompt for style
        continuity.  Joins adjacent segments with equal-power cosine crossfades.
        """
        n_segments = math.ceil(total_duration_sec / MAX_SEGMENT_SEC)
        seg_duration = total_duration_sec / n_segments

        logger.info(
            "[HeartMuLa] Long-form: %.0fs total -> %d segments x %.0fs",
            total_duration_sec, n_segments, seg_duration,
        )

        if progress_cb:
            progress_cb(0, n_segments)

        segment_audios = []
        for i in range(n_segments):
            seg_lyrics = self._build_segment_lyrics(lyrics, i, n_segments, seg_duration)
            audio = self._generate_single(tags, seg_duration, seg_lyrics, progress_cb=None)
            segment_audios.append(audio)
            if progress_cb:
                progress_cb(i + 1, n_segments)

        return self._crossfade_segments(segment_audios, CROSSFADE_SEC)

    # ------------------------------------------------------------------
    # Internal: story mode
    # ------------------------------------------------------------------

    def _generate_story(
        self,
        prompt_stages: list[tuple[str, float]],
        progress_cb,
        lyrics: str | None,
    ) -> np.ndarray:
        """Generate story mode: one HeartMuLa call per stage, crossfade results."""
        n = len(prompt_stages)
        logger.info("[HeartMuLa] Story mode: %d stages", n)

        segment_audios = []
        for i, (stage_tags, stage_duration) in enumerate(prompt_stages):
            if progress_cb:
                progress_cb(i, n)

            if stage_duration > MAX_SEGMENT_SEC:
                audio = self._generate_long_form(
                    stage_tags, stage_duration, lyrics, progress_cb=None
                )
            else:
                stage_lyrics = self._build_segment_lyrics(
                    lyrics, i, n, stage_duration
                )
                audio = self._generate_single(
                    stage_tags, stage_duration, stage_lyrics, progress_cb=None
                )
            segment_audios.append(audio)

        if progress_cb:
            progress_cb(n, n)

        return self._crossfade_segments(segment_audios, CROSSFADE_SEC)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_segment_lyrics(
        user_lyrics: str | None,
        segment_idx: int,
        n_segments: int,
        duration_sec: float,
    ) -> str:
        """Build structural section tags per segment for heartlib lyrics field.

        HeartMuLa uses standard markers: [intro], [verse], [chorus], [bridge],
        [outro].  For instrumental meditation tracks (no lyrics), structural
        markers alone guide the model's tonal arc.

        Segment 0 of N:  [intro] + [verse]
        Middle segments:  [verse] + [bridge] + [verse]
        Last segment:     [verse] + [outro]
        Single segment:   [intro] + [verse] + [outro]
        """
        if user_lyrics:
            return user_lyrics

        if n_segments == 1:
            return "[intro]\n\n[verse]\n\n[outro]"
        elif segment_idx == 0:
            return "[intro]\n\n[verse]"
        elif segment_idx == n_segments - 1:
            return "[verse]\n\n[outro]"
        else:
            return "[verse]\n\n[bridge]\n\n[verse]"

    @staticmethod
    def _crossfade_segments(
        segments: list[np.ndarray],
        crossfade_sec: float,
    ) -> np.ndarray:
        """Join mono float32 segments with equal-power cosine-squared crossfades."""
        if len(segments) == 1:
            return segments[0]

        fade_samples = int(crossfade_sec * TARGET_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            overlap = min(fade_samples, len(result), len(seg))
            t = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
            fade_out = np.cos(t) ** 2
            fade_in = np.sin(t) ** 2
            blended = result[-overlap:] * fade_out + seg[:overlap] * fade_in
            result = np.concatenate([result[:-overlap], blended, seg[overlap:]])

        return result

    @staticmethod
    def _postprocess(audio_data, sample_rate: int) -> np.ndarray:
        """Convert HeartCodec output to mono float32 numpy at TARGET_SAMPLE_RATE.

        HeartCodec outputs stereo at 48,000 Hz.  Steps:
        1. CPU + float32 conversion
        2. Stereo -> mono (channel average)
        3. Resample if native SR differs from TARGET_SAMPLE_RATE
        4. Peak normalize to -1 dBFS
        5. Safety clip [-1, 1]
        """
        # Convert to numpy float32
        if hasattr(audio_data, "cpu"):
            arr = audio_data.cpu().float().numpy().astype(np.float32)
        elif hasattr(audio_data, "__array__"):
            arr = np.asarray(audio_data, dtype=np.float32)
        else:
            arr = np.array(audio_data, dtype=np.float32)

        # Handle batch dimension: (batch, channels, samples)
        if arr.ndim == 3:
            arr = arr[0]

        # Stereo -> mono
        # HeartCodec may output (channels, samples) or (samples, channels).
        # Detect orientation by finding the small axis (channels ≤ 8).
        if arr.ndim == 2:
            if arr.shape[0] <= 8:
                arr = arr.mean(axis=0)       # (channels, samples) → (samples,)
            elif arr.shape[1] <= 8:
                arr = arr.mean(axis=1)       # (samples, channels) → (samples,)
            else:
                arr = arr.flatten()          # fallback

        # Resample if needed (unlikely — both NATIVE and TARGET are 48 kHz)
        if sample_rate != TARGET_SAMPLE_RATE:
            import torch
            import torchaudio
            t = torch.from_numpy(arr).unsqueeze(0)
            t = torchaudio.functional.resample(
                t, sample_rate, TARGET_SAMPLE_RATE,
                lowpass_filter_width=64, rolloff=0.9475,
            )
            arr = t.squeeze(0).numpy().astype(np.float32)

        # Peak normalize to -1 dBFS
        peak = np.abs(arr).max()
        if peak > 1e-6:
            target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS ~ 0.891
            arr = arr * (target_peak / peak)

        return np.clip(arr, -1.0, 1.0).astype(np.float32)
