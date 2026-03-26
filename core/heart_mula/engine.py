"""HeartMuLa engine — text-to-music via heartlib on Apple Silicon.

Model: HeartMuLa-RL-oss-3B (LM) + HeartCodec-oss (codec)
Backend: MLX primary (heartlib-mlx), PyTorch MPS fallback (heartlib).
Memory: Both models loaded simultaneously — LM bf16 (~6 GB) + codec fp32 (~1.2 GB)
        + KV cache (~3-5 GB) ≈ 12 GB total.  36 GB system has 24+ GB headroom.
        MPS fallback still uses heartlib's built-in lazy_load=True.
Output: Mono float32 at 48,000 Hz (HeartCodec native rate downmixed to mono).
Long-form: Token-level continuation for coherent multi-segment generation.
           Segments are joined with STFT crossfades in log-magnitude domain.

Quality features:
  - Best-of-N generation (N=3) with QA composite scoring
  - Temperature annealing: 0.80 → 0.85 → 0.65 (establish → develop → resolve)
  - Dynamic CFG scheduling: 2.0 → 1.0 (strong start → organic drift)

Target hardware: Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)
"""

import gc
import logging
import math
import os
import random
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
CROSSFADE_SEC = 8.0             # Equal-power cosine crossfade between segments
                                # 8s (was 4s) — longer overlap masks tonal differences
                                # between independently generated segments.

# Checkpoint directories (relative to project root)
CHECKPOINT_DIR = "./ckpt"           # PyTorch MPS path
CHECKPOINT_DIR_MLX = "./ckpt-mlx"   # MLX converted weights path

# Generation tuning — optimised for calm meditation / sleep music
# cfg_scale: classifier-free guidance strength.  HeartMuLa-RL uses DPO training
#   (not GRPO) with MuQ tag similarity, AudioBox aesthetics, SongEval musicality,
#   and PER phoneme error rate.  The official default is 1.5 (too loose — LM wanders).
#   2.5 gives clear tag adherence without the mode collapse risk at 4.0+.
#   The research-suggested 1.8 was too close to default — prompt adherence was weak.
_LM_CFG_SCALE = 2.5
# temperature: token sampling temperature.  Lower values create more stable token
#   distributions essential for sustained drones and slowly evolving pad textures.
#   0.75 reduces random timbral jumps while preserving gentle tonal variety.
_LM_TEMPERATURE = 0.75
# top_k: sampling pool size.  Tighter pool (30) keeps the LM firmly in ambient
#   territory.  Range 25-35 is optimal per research; above 40 risks repetitive loops.
_LM_TOP_K = 30
# guidance_scale for HeartCodec flow-matching decoder.  HeartMuLa paper Table 2
#   confirms 1.25 yields "a more natural and balanced auditory experience, with
#   vocals and accompaniment sounding smoother and less harsh."  Lower codec
#   guidance reduces metallic artifacts alongside the fp32 requirement.
_CODEC_GUIDANCE_SCALE = 1.25
# HeartCodec underwent reflow distillation (50 → 10 steps).  12 steps gives
# marginal improvement over the 10-step design point with less waste than 16.
_CODEC_NUM_STEPS = 12


# -- Engine -------------------------------------------------------------------

class HeartMulaEngine:
    """Generates music via HeartMuLa (3B LM + HeartCodec).

    Uses MLX backend (heartlib-mlx) for fastest Apple Silicon inference.
    Falls back to official PyTorch MPS backend (heartlib) if MLX is
    unavailable.

    Memory strategy:
      MLX:  Load both LM (bf16, ~6 GB) and codec (fp32, ~1.2 GB)
            simultaneously.  Total ~12 GB on 36 GB system.
      MPS:  heartlib's built-in lazy_load=True handles lifecycle.

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

        Checkpoint validation only — actual weight loading happens during
        generation.  MLX loads both models simultaneously; MPS uses
        heartlib's lazy_load=True.
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
        quality_mode: bool = False,
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
                (e.g. "[intro-medium]\\n\\n[inst-medium]\\n\\n[outro-medium]").
                Empty string or None = instrumental.
            quality_mode: If True, generate N=3 candidates and select the best
                via QA composite scoring.  ~3x slower but 15-25% quality gain.
            **kwargs: Ignored; present for API compatibility with other engines.

        Returns:
            Mono float32 numpy array at 48,000 Hz.
        """
        if not self.initialized:
            raise RuntimeError(
                "HeartMulaEngine: call load_model() before generate()."
            )

        self._quality_mode = quality_mode

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

    # ------------------------------------------------------------------
    # Temperature and CFG scheduling for meditation music
    # ------------------------------------------------------------------

    @staticmethod
    def _meditation_temperature_schedule(step: int, total_steps: int) -> float:
        """Temperature annealing matching musical form.

        Establish structure → creative development → coherent resolution.
        """
        progress = step / max(total_steps, 1)
        if progress < 0.1:
            return 0.80   # Establish structure
        elif progress < 0.7:
            return 0.85   # Creative development
        else:
            return 0.65   # Coherent resolution

    @staticmethod
    def _meditation_cfg_schedule(step: int, total_steps: int) -> float:
        """Linear CFG annealing — strong initial direction, organic drift.

        2.0 → 1.0 over the sequence gives strong genre direction early
        while allowing natural variation as the piece develops.
        """
        cfg_start, cfg_end = 2.0, 1.0
        progress = step / max(total_steps, 1)
        return cfg_start + (cfg_end - cfg_start) * progress

    @staticmethod
    def _make_scheduled_generate(pipeline_cls):
        """Create a generate method with per-step temperature and CFG scheduling.

        Monkey-patches the heartlib-mlx pipeline's generate() to accept
        optional temperature_fn and cfg_fn callables.
        """
        import mlx.core as mx

        def scheduled_generate(
            self,
            inputs,
            duration=30.0,
            temperature=1.0,
            top_k=50,
            cfg_scale=1.5,
            temperature_fn=None,
            cfg_fn=None,
        ):
            prompt_tokens = inputs["tokens"]
            prompt_tokens_mask = inputs["tokens_mask"]
            continuous_segment = inputs["muq_embed"]
            starts = inputs["muq_idx"]
            prompt_pos = inputs["pos"]

            frames = []
            bs_size = 2 if cfg_scale != 1.0 else 1
            self.heartmula.setup_caches(bs_size)

            max_audio_frames = int(duration * self.config.frame_rate)

            # First frame uses initial temperature/cfg
            step_temp = temperature_fn(0, max_audio_frames) if temperature_fn else temperature
            step_cfg = cfg_fn(0, max_audio_frames) if cfg_fn else cfg_scale

            curr_token = self.heartmula.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=step_temp,
                topk=top_k,
                cfg_scale=step_cfg,
                continuous_segments=continuous_segment,
                starts=starts,
            )
            frames.append(curr_token[0:1, :])

            for i in range(max_audio_frames):
                step_temp = temperature_fn(i + 1, max_audio_frames) if temperature_fn else temperature
                step_cfg = cfg_fn(i + 1, max_audio_frames) if cfg_fn else cfg_scale

                curr_token_padded, curr_token_mask = self._pad_audio_token(curr_token)
                next_pos = prompt_pos[:, -1:] + i + 1

                curr_token = self.heartmula.generate_frame(
                    tokens=curr_token_padded,
                    tokens_mask=curr_token_mask,
                    input_pos=next_pos,
                    temperature=step_temp,
                    topk=top_k,
                    cfg_scale=step_cfg,
                    continuous_segments=None,
                    starts=None,
                )

                if mx.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                    break

                frames.append(curr_token[0:1, :])

            self.heartmula.reset_caches()

            codes = mx.concatenate(frames, axis=0)
            codes = codes.transpose(1, 0)
            return codes

        pipeline_cls.generate = scheduled_generate

    def _generate_mlx(
        self, tags: str, duration_sec: float, lyrics: str | None,
    ) -> np.ndarray:
        """Generate audio using the MLX backend (heartlib-mlx).

        Loads both HeartMuLa LM (bf16) and HeartCodec (fp32) simultaneously.
        Total memory: ~12 GB on a 36 GB system (24+ GB headroom).

        Features:
        - Simultaneous model loading (no load/unload cycle)
        - Temperature annealing (establish → develop → resolve)
        - Dynamic CFG scheduling (strong start → organic drift)
        - Best-of-N selection when quality_mode is enabled
        """
        import mlx.core as mx
        from heartlib_mlx.heartmula import HeartMuLa
        from heartlib_mlx.heartcodec import HeartCodec
        from heartlib_mlx.pipelines.music_generation import HeartMuLaGenPipeline, HeartMuLaGenConfig
        from tokenizers import Tokenizer
        from pathlib import Path

        ckpt = Path(CHECKPOINT_DIR_MLX)

        # ── Memory budget for simultaneous loading ──────────────────────
        mx.set_memory_limit(30 * 1024**3)   # 30 GB (leave 6 GB for OS)
        mx.set_cache_limit(4 * 1024**3)     # 4 GB operation cache

        # Load config and tokenizer (lightweight)
        config = HeartMuLaGenConfig.from_pretrained(ckpt)
        tokenizer_path = ckpt / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists() else None

        # ── Load BOTH models simultaneously ─────────────────────────────
        logger.info("[HeartMuLa/MLX] Loading HeartMuLa LM (bf16) + HeartCodec (fp32)...")
        t_load = time.time()
        heartmula_lm = HeartMuLa.from_pretrained(ckpt / "heartmula", dtype=mx.bfloat16)
        heartcodec = HeartCodec.from_pretrained(ckpt / "heartcodec", dtype=mx.float32)
        logger.info(
            "[HeartMuLa/MLX] Both models loaded in %.1fs — active: %.1f MB",
            time.time() - t_load, mx.get_active_memory() / 1e6,
        )

        # Build pipeline with both models
        pipeline = HeartMuLaGenPipeline.__new__(HeartMuLaGenPipeline)
        pipeline.heartmula = heartmula_lm
        pipeline.heartcodec = heartcodec
        pipeline.tokenizer = tokenizer
        pipeline.config = config
        pipeline._parallel_number = heartmula_lm.num_codebooks + 1

        # Patch generate() to support temperature and CFG scheduling
        self._make_scheduled_generate(HeartMuLaGenPipeline)

        # Preprocess text inputs
        inputs = pipeline.preprocess(lyrics=lyrics or "", tags=tags, cfg_scale=_LM_CFG_SCALE)

        # ── Generate (best-of-N if quality_mode) ────────────────────────
        n_candidates = 3 if getattr(self, "_quality_mode", False) else 1

        sr = config.sample_rate or NATIVE_SAMPLE_RATE

        if n_candidates > 1:
            # Best-of-N: returns already-postprocessed audio
            result = self._generate_best_of_n(
                pipeline, heartcodec, inputs, config, duration_sec,
                n_candidates, sr, mx,
            )
        else:
            # Single generation with scheduling
            logger.info("[HeartMuLa/MLX] Generating audio codes (%.0fs)...", duration_sec)
            codes = pipeline.generate(
                inputs=inputs,
                duration=duration_sec,
                temperature=_LM_TEMPERATURE,
                top_k=_LM_TOP_K,
                cfg_scale=_LM_CFG_SCALE,
                temperature_fn=self._meditation_temperature_schedule,
                cfg_fn=self._meditation_cfg_schedule,
            )
            mx.eval(codes)

            # Reset KV cache, keep weights
            heartmula_lm.reset_caches()
            mx.clear_cache()

            # Detokenize and postprocess
            audio_data = self._decode_codes(codes, heartcodec, config)
            result = self._postprocess(audio_data, sr)

        # ── Cleanup ─────────────────────────────────────────────────────
        logger.info("[HeartMuLa/MLX] Unloading models...")
        del pipeline, heartmula_lm, heartcodec, inputs
        gc.collect()
        mx.set_cache_limit(0)
        mx.clear_cache()
        logger.info(
            "[HeartMuLa/MLX] Post-cleanup — active: %.1f MB, cache: %.1f MB",
            mx.get_active_memory() / 1e6, mx.get_cache_memory() / 1e6,
        )

        return result

    def _generate_best_of_n(
        self, pipeline, heartcodec, inputs, config, duration_sec,
        n_candidates, sr, mx,
    ) -> np.ndarray:
        """Generate N candidates and select the best via QA composite scoring.

        Returns already-postprocessed audio (mono float32 at TARGET_SAMPLE_RATE).
        """
        from core.qa_monitor import compute_composite_score

        logger.info(
            "[HeartMuLa/MLX] Quality mode: generating %d candidates...",
            n_candidates,
        )

        candidates = []
        for i in range(n_candidates):
            # Different seed per candidate
            mx.random.seed(int(time.time() * 1000) % (2**31) + i * 42)

            codes = pipeline.generate(
                inputs=inputs,
                duration=duration_sec,
                temperature=_LM_TEMPERATURE,
                top_k=_LM_TOP_K,
                cfg_scale=_LM_CFG_SCALE,
                temperature_fn=self._meditation_temperature_schedule,
                cfg_fn=self._meditation_cfg_schedule,
            )
            mx.eval(codes)

            # Reset KV cache between candidates (weights stay loaded)
            pipeline.heartmula.reset_caches()
            mx.clear_cache()

            # Decode to audio and postprocess
            audio_data = self._decode_codes(codes, heartcodec, config)
            audio_np = self._postprocess(audio_data, sr)

            # Score using QA composite
            score = compute_composite_score(audio_np, TARGET_SAMPLE_RATE)
            candidates.append((audio_np, score))
            logger.info(
                "[HeartMuLa/MLX] Candidate %d/%d: QA score=%.4f",
                i + 1, n_candidates, score,
            )

            del codes, audio_data
            mx.clear_cache()

        # Select best
        best_audio, best_score = max(candidates, key=lambda x: x[1])
        logger.info(
            "[HeartMuLa/MLX] Selected best candidate (score=%.4f)",
            best_score,
        )
        return best_audio

    @staticmethod
    def _decode_codes(codes, heartcodec, config) -> np.ndarray:
        """Decode discrete audio codes to waveform via HeartCodec."""
        import mlx.core as mx

        # Detokenize: codes shape is (num_codebooks, num_frames)
        # HeartCodec expects (batch, frames, num_quantizers)
        codes_input = mx.transpose(codes, axes=(1, 0))[None, :, :]

        num_frames = codes_input.shape[1]
        codec_frame_rate = heartcodec.config.frame_rate or 50.0
        duration_for_codec = num_frames / codec_frame_rate

        logger.info("[HeartMuLa/MLX] Detokenizing %d frames...", num_frames)
        audio_mx = heartcodec.detokenize(
            codes=codes_input,
            duration=duration_for_codec,
            num_steps=_CODEC_NUM_STEPS,
            guidance_scale=_CODEC_GUIDANCE_SCALE,
        )
        mx.eval(audio_mx)

        audio_data = np.array(audio_mx, dtype=np.float32)
        del audio_mx, codes_input
        return audio_data

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
                temperature=_LM_TEMPERATURE,
                topk=_LM_TOP_K,
                cfg_scale=_LM_CFG_SCALE,
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

    # Musical keys suitable for meditation — minor keys and soft major keys
    # that pair well with ambient/drone textures.
    _MEDITATION_KEYS = [
        "key of C major", "key of D minor", "key of A minor",
        "key of F major", "key of E minor", "key of G major",
    ]

    def _generate_long_form(
        self,
        tags: str,
        total_duration_sec: float,
        lyrics: str | None,
        progress_cb,
    ) -> np.ndarray:
        """Generate audio longer than MAX_SEGMENT_SEC with token-level continuation.

        MLX backend: loads both models once, generates segments with token prefix
        continuation (last 20s of codes fed as context for the next segment).
        This produces musically coherent long-form output — far superior to
        independent segments.

        MPS backend: falls back to independent segment generation (heartlib's
        lazy_load=True doesn't support holding models across calls).
        """
        n_segments = math.ceil(total_duration_sec / MAX_SEGMENT_SEC)
        seg_duration = total_duration_sec / n_segments

        # Anchor all segments to the same musical key for harmonic continuity
        tonal_anchor = random.choice(self._MEDITATION_KEYS)
        anchored_tags = f"{tags}, {tonal_anchor}"
        logger.info(
            "[HeartMuLa] Long-form: %.0fs total -> %d segments x %.0fs "
            "(tonal anchor: %s, backend: %s)",
            total_duration_sec, n_segments, seg_duration, tonal_anchor,
            self._backend,
        )

        if self._backend == "mlx":
            return self._generate_long_form_mlx(
                anchored_tags, n_segments, seg_duration, lyrics, progress_cb,
            )

        # MPS fallback: independent segments
        if progress_cb:
            progress_cb(0, n_segments)

        segment_audios = []
        for i in range(n_segments):
            seg_lyrics = self._build_segment_lyrics(lyrics, i, n_segments, seg_duration)
            audio = self._generate_single(
                anchored_tags, seg_duration, seg_lyrics, progress_cb=None,
            )
            segment_audios.append(audio)
            if progress_cb:
                progress_cb(i + 1, n_segments)

        return self._crossfade_segments(segment_audios, CROSSFADE_SEC)

    def _generate_long_form_mlx(
        self,
        tags: str,
        n_segments: int,
        seg_duration: float,
        lyrics: str | None,
        progress_cb,
    ) -> np.ndarray:
        """MLX long-form: token-level continuation across segments.

        Keeps both models loaded for the entire multi-segment generation.
        Feeds the final 20s of generated tokens (250 tokens at 12.5 Hz)
        as prefix context for the next segment, producing continuous
        musical ideas across segment boundaries.
        """
        import mlx.core as mx
        from heartlib_mlx.heartmula import HeartMuLa
        from heartlib_mlx.heartcodec import HeartCodec
        from heartlib_mlx.pipelines.music_generation import HeartMuLaGenPipeline, HeartMuLaGenConfig
        from tokenizers import Tokenizer
        from pathlib import Path

        ckpt = Path(CHECKPOINT_DIR_MLX)

        # Memory budget
        mx.set_memory_limit(30 * 1024**3)
        mx.set_cache_limit(4 * 1024**3)

        # Load config, tokenizer, and both models
        config = HeartMuLaGenConfig.from_pretrained(ckpt)
        tokenizer_path = ckpt / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path.exists() else None

        logger.info("[HeartMuLa/MLX] Long-form: loading both models...")
        heartmula_lm = HeartMuLa.from_pretrained(ckpt / "heartmula", dtype=mx.bfloat16)
        heartcodec = HeartCodec.from_pretrained(ckpt / "heartcodec", dtype=mx.float32)

        pipeline = HeartMuLaGenPipeline.__new__(HeartMuLaGenPipeline)
        pipeline.heartmula = heartmula_lm
        pipeline.heartcodec = heartcodec
        pipeline.tokenizer = tokenizer
        pipeline.config = config
        pipeline._parallel_number = heartmula_lm.num_codebooks + 1

        # Patch generate() for scheduling support
        self._make_scheduled_generate(HeartMuLaGenPipeline)

        # Token continuation parameters
        overlap_sec = 20.0
        overlap_tokens = int(overlap_sec * config.frame_rate)  # 250 tokens at 12.5 Hz
        sr = config.sample_rate or NATIVE_SAMPLE_RATE

        if progress_cb:
            progress_cb(0, n_segments)

        segment_audios = []
        prefix_codes = None

        for i in range(n_segments):
            seg_lyrics = self._build_segment_lyrics(lyrics, i, n_segments, seg_duration)
            inputs = pipeline.preprocess(
                lyrics=seg_lyrics or "", tags=tags, cfg_scale=_LM_CFG_SCALE,
            )

            t0 = time.time()
            if prefix_codes is not None:
                # Token continuation: feed prefix through model to populate KV cache
                logger.info(
                    "[HeartMuLa/MLX] Segment %d/%d with %d-token prefix continuation...",
                    i + 1, n_segments, prefix_codes.shape[1],
                )
                codes = self._generate_with_prefix(
                    pipeline, inputs, prefix_codes, seg_duration, config, mx,
                )
            else:
                logger.info(
                    "[HeartMuLa/MLX] Segment %d/%d (initial, no prefix)...",
                    i + 1, n_segments,
                )
                codes = pipeline.generate(
                    inputs=inputs,
                    duration=seg_duration,
                    temperature=_LM_TEMPERATURE,
                    top_k=_LM_TOP_K,
                    cfg_scale=_LM_CFG_SCALE,
                    temperature_fn=self._meditation_temperature_schedule,
                    cfg_fn=self._meditation_cfg_schedule,
                )
            mx.eval(codes)

            # Save tail as prefix for next segment
            if codes.shape[1] > overlap_tokens:
                prefix_codes = codes[:, -overlap_tokens:]
                mx.eval(prefix_codes)
            else:
                prefix_codes = None

            # Reset KV cache but keep weights for next segment
            heartmula_lm.reset_caches()
            mx.clear_cache()

            # Decode to audio
            audio_data = self._decode_codes(codes, heartcodec, config)
            audio_np = self._postprocess(audio_data, sr)
            segment_audios.append(audio_np)

            elapsed = time.time() - t0
            logger.info(
                "[HeartMuLa/MLX] Segment %d/%d done in %.1fs (%.1fx RT)",
                i + 1, n_segments, elapsed, seg_duration / max(elapsed, 0.01),
            )

            del codes, audio_data, inputs
            mx.clear_cache()

            if progress_cb:
                progress_cb(i + 1, n_segments)

        # Cleanup
        del pipeline, heartmula_lm, heartcodec
        gc.collect()
        mx.set_cache_limit(0)
        mx.clear_cache()

        return self._crossfade_segments(segment_audios, CROSSFADE_SEC)

    @staticmethod
    def _generate_with_prefix(pipeline, inputs, prefix_codes, duration_sec, config, mx):
        """Generate codes with token prefix continuation.

        Feeds prefix_codes through the model to populate the KV cache,
        then continues autoregressive generation from established context.
        Returns only the NEW codes (not the prefix).
        """
        prompt_tokens = inputs["tokens"]
        prompt_tokens_mask = inputs["tokens_mask"]
        continuous_segment = inputs["muq_embed"]
        starts = inputs["muq_idx"]
        prompt_pos = inputs["pos"]

        bs_size = 2 if _LM_CFG_SCALE != 1.0 else 1
        pipeline.heartmula.setup_caches(bs_size)

        # Phase 1: Process text prompt to populate KV cache
        curr_token = pipeline.heartmula.generate_frame(
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=_LM_TEMPERATURE,
            topk=_LM_TOP_K,
            cfg_scale=_LM_CFG_SCALE,
            continuous_segments=continuous_segment,
            starts=starts,
        )

        # Phase 2: Feed prefix codes through the model (populates KV cache
        # with musical context from the previous segment)
        n_prefix = prefix_codes.shape[1]
        for j in range(n_prefix):
            # Create a token from the prefix codes at this frame
            prefix_frame = prefix_codes[:, j:j+1]  # (num_codebooks, 1)
            prefix_token = prefix_frame.transpose(1, 0)  # (1, num_codebooks)
            # Duplicate for CFG batch
            if bs_size == 2:
                prefix_token = mx.concatenate([prefix_token, prefix_token], axis=0)

            curr_token_padded, curr_token_mask = pipeline._pad_audio_token(prefix_token)
            next_pos = prompt_pos[:, -1:] + j + 1

            curr_token = pipeline.heartmula.generate_frame(
                tokens=curr_token_padded,
                tokens_mask=curr_token_mask,
                input_pos=next_pos,
                temperature=_LM_TEMPERATURE,
                topk=_LM_TOP_K,
                cfg_scale=_LM_CFG_SCALE,
                continuous_segments=None,
                starts=None,
            )

        # Phase 3: Continue generating NEW frames from established context
        max_audio_frames = int(duration_sec * config.frame_rate)
        frames = []

        for i in range(max_audio_frames):
            step_temp = HeartMulaEngine._meditation_temperature_schedule(i, max_audio_frames)
            step_cfg = HeartMulaEngine._meditation_cfg_schedule(i, max_audio_frames)

            curr_token_padded, curr_token_mask = pipeline._pad_audio_token(curr_token)
            next_pos = prompt_pos[:, -1:] + n_prefix + i + 1

            curr_token = pipeline.heartmula.generate_frame(
                tokens=curr_token_padded,
                tokens_mask=curr_token_mask,
                input_pos=next_pos,
                temperature=step_temp,
                topk=_LM_TOP_K,
                cfg_scale=step_cfg,
                continuous_segments=None,
                starts=None,
            )

            if mx.any(curr_token[0:1, :] >= config.audio_eos_id):
                break

            frames.append(curr_token[0:1, :])

        pipeline.heartmula.reset_caches()

        codes = mx.concatenate(frames, axis=0)
        codes = codes.transpose(1, 0)  # (num_codebooks, num_frames)
        return codes

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
        """Build structural section markers per segment for heartlib lyrics field.

        Uses the standard song-structure markers that HeartMuLa was trained on:
        [interlude] for sustained instrumental sections, [intro]/[outro] for
        bookends.  These are from the Llama-3 training data (song lyrics format).

        Without text lines between markers, the LM produces instrumental
        sustained pads — no phonetic tokens means no vocal generation.

        Segment 0 of N:  [intro] + N×[interlude]
        Middle segments:  N×[interlude]
        Last segment:     N×[interlude] + [outro]
        Single segment:   [intro] + N×[interlude] + [outro]
        """
        if user_lyrics:
            return user_lyrics

        # Scale [interlude] count to segment duration (~20s each)
        n_inst = max(1, round(duration_sec / 20.0))
        inst_block = "\n\n".join(["[interlude]"] * n_inst)

        if n_segments == 1:
            return f"[intro]\n\n{inst_block}\n\n[outro]"
        elif segment_idx == 0:
            return f"[intro]\n\n{inst_block}"
        elif segment_idx == n_segments - 1:
            return f"{inst_block}\n\n[outro]"
        else:
            return inst_block

    @staticmethod
    def _crossfade_segments(
        segments: list[np.ndarray],
        crossfade_sec: float,
    ) -> np.ndarray:
        """Join mono float32 segments with STFT crossfades in log-magnitude domain.

        Interpolates magnitude in log space and phase linearly for spectrally
        smooth transitions.  Falls back to cosine-squared if STFT fails.
        """
        if len(segments) == 1:
            return segments[0]

        import librosa

        fade_samples = int(crossfade_sec * TARGET_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            overlap = min(fade_samples, len(result), len(seg))
            try:
                blended = HeartMulaEngine._stft_crossfade(
                    result, seg, overlap,
                )
                result = blended
            except Exception as e:
                logger.warning(
                    "[HeartMuLa] STFT crossfade failed (%s), falling back to cosine²",
                    e,
                )
                t = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2
                blended = result[-overlap:] * fade_out + seg[:overlap] * fade_in
                result = np.concatenate([result[:-overlap], blended, seg[overlap:]])

        return result

    @staticmethod
    def _stft_crossfade(
        seg_a: np.ndarray,
        seg_b: np.ndarray,
        overlap_samples: int,
    ) -> np.ndarray:
        """STFT crossfade in log-magnitude domain for spectrally smooth transitions.

        Interpolates magnitude in log1p space (avoids log(0)) and phase linearly.
        Produces smoother transitions than cosine-squared because each frequency
        bin is interpolated independently — no broadband amplitude modulation.
        """
        import librosa

        n_fft = 4096
        hop = n_fft // 4

        # STFT of overlap regions
        S_a = librosa.stft(
            seg_a[-overlap_samples:], n_fft=n_fft, hop_length=hop,
        )
        S_b = librosa.stft(
            seg_b[:overlap_samples], n_fft=n_fft, hop_length=hop,
        )

        # Interpolation ramp (one value per STFT frame)
        n_frames = min(S_a.shape[1], S_b.shape[1])
        S_a = S_a[:, :n_frames]
        S_b = S_b[:, :n_frames]
        t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[None, :]

        # Interpolate in log-magnitude domain
        mag_a = np.abs(S_a)
        mag_b = np.abs(S_b)
        log_mag_a = np.log1p(mag_a)
        log_mag_b = np.log1p(mag_b)
        blended_mag = np.expm1((1.0 - t) * log_mag_a + t * log_mag_b)

        # Linear phase interpolation
        phase_a = np.angle(S_a)
        phase_b = np.angle(S_b)
        blended_phase = (1.0 - t) * phase_a + t * phase_b

        # Reconstruct
        blended = librosa.istft(
            blended_mag * np.exp(1j * blended_phase),
            hop_length=hop,
            length=overlap_samples,
        )

        return np.concatenate([
            seg_a[:-overlap_samples],
            blended.astype(np.float32),
            seg_b[overlap_samples:],
        ])

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
