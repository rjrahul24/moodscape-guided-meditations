"""ACE-Step 1.5 wrapper — text-to-music via MLX on Apple Silicon.

Model: ACE-Step 1.5 (DiT decoder + LM planner)
Device: MPS (PyTorch) with MLX-accelerated DiT and LLM
Output: Mono float32 at 48 kHz (native rate preserved; stereo downmixed to mono)
Story mode: generate each meditation stage as a separate inference call and
            crossfade the results — allows deliberate tonal evolution across
            the meditation arc (e.g. centering → depth → integration).

Target hardware: Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)
"""

import gc
import logging
import time
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 48000       # ACE-Step native output rate
TARGET_SAMPLE_RATE = 48000       # Preserve native 48 kHz through the pipeline

# ── Generation quality knobs ──────────────────────────────────────────────────
# guidance_scale: CFG strength for the SFT model.
# 5.5 sits in the upper SFT sweet spot (4–6) for ambient music — tighter
# prompt adherence for meditation without the rigidity and spectral artifacts
# that appear above 6.0.
_GUIDANCE_SCALE = 5.5

# inference_steps: SFT supports up to 50 steps. Going higher causes error
# accumulation that degrades output quality. 50 is maximum detail without
# degradation artifacts.
_INFERENCE_STEPS = 50
_INFERENCE_STEPS_REPAINT = 50

# lm_temperature: Controls LM planner creativity.
# 0.65 balances harmonic variety with calm predictability — low enough to
# prevent unexpected timbral jumps, high enough to avoid harmonic stagnation
# and repetitive loops in long meditations.
_LM_TEMPERATURE = 0.65

# ADG (Adaptive Dual Guidance) was designed for the base model.
# On SFT, prompt adherence is baked into weights via supervised fine-tuning,
# so ADG doubles forward passes without quality benefit and can conflict with
# the already-strong SFT guidance signal.  Disabled for SFT config.
# Note: _generate_single_repaint and _generate_cover_continuation already
# hardcode use_adg=False; this aligns the main generation path with those.
_USE_ADG = False

# cfg_interval_end: Release CFG guidance at 80% of denoising steps.
# Only the final 20% of micro-detail steps run free — enough organic softening
# while keeping the structural calm/meditation intent intact through most of
# the denoising process. 0.6 was too early, allowing 40% drift from prompt.
_CFG_INTERVAL_END = 0.8

# shift: Timestep shift factor for the DiT scheduler.
# 3.0 concentrates more denoising budget on high-noise (semantic) steps,
# producing cleaner harmonic structure and stronger tonal conditioning.
_SHIFT = 3.0

# infer_method: Diffusion inference method ("ode" or "sde").
# ODE is deterministic — smoother, more reproducible output for meditation.
# SDE adds stochastic micro-variations (organic but less predictable).
_INFER_METHOD = "ode"

# Story mode: crossfade duration between adjacent stages (seconds).
# 6 seconds gives a natural, unhurried transition between tonal worlds
# without cutting into the usable content of either stage.
_STORY_CROSSFADE_SEC = 6.0


class AceStepEngine:
    """Generates ambient background music via ACE-Step 1.5.

    Uses the full-quality ``acestep-v15-sft`` DiT config and the
    ``acestep-5Hz-lm-4B`` language model (falls back to 1.7B) for
    Chain-of-Thought planning.  Forces instrumental mode with no vocals.
    Output is 48 kHz mono (stereo downmixed to mono).

    Device strategy:
    - DiT handler: ``device="auto"`` resolves to ``"mps"`` on Apple Silicon.
      MLX acceleration is enabled via ``use_mlx_dit=True`` (default).
    - LLM handler: ``backend="mlx"`` loads weights natively via ``mlx-lm``.
      Falls back to PyTorch on MPS automatically if MLX load fails.

    Memory: explicitly calls gc.collect() and torch.mps.empty_cache()
    on unload to free unified RAM for subsequent TTS loading.
    """

    def __init__(self):
        self._dit = None
        self._llm = None
        self.initialized = False
        self.model_type = None  # "sft" or "turbo"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self, model_type="sft"):
        """Load ACE-Step DiT and LLM handlers.

        Args:
            model_type: "sft" (high fidelity, 50 steps) or
                        "turbo" (distilled, 8 steps).
        """
        # ── Patch ACE-Step package constants before first import ──────────────
        # The package hard-codes DURATION_MAX=600 and ACESTEP_GENERATION_TIMEOUT=600
        # at module-import time.  We need to override both before any acestep
        # module is loaded so the patched values are seen by all importers.
        #
        # DURATION_MAX: caps audio length in the LM constrained decoder and the
        # GPU-tier config.  Set to 1200 s (20 min) to match our target ceiling.
        #
        # ACESTEP_GENERATION_TIMEOUT: wall-clock deadline for the diffusion loop
        # thread.  7200 s (2 h) is generous enough for any realistic generation
        # length on M1 Max while still guarding against genuine hangs.
        os.environ["ACESTEP_GENERATION_TIMEOUT"] = "7200"

        import acestep.constants as _ac
        _ac.DURATION_MAX = 1200

        import acestep.gpu_config as _agc
        _agc.GPU_TIER_CONFIGS["unlimited"]["max_duration_with_lm"] = 1200
        _agc.GPU_TIER_CONFIGS["unlimited"]["max_duration_without_lm"] = 1200

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        logger.info("[AceStepEngine] Loading ACE-Step 1.5...")
        t0 = time.time()

        # ── DiT handler ──────────────────────────────────────────────────
        if self._dit is None:
            self._dit = AceStepHandler()
            
        config_path = f"acestep-v15-{model_type}"
        logger.info(f"[AceStepEngine] Initializing DiT with config: {config_path}")

        # device="auto" resolves to "mps" on Apple Silicon.
        # use_mlx_dit=True (default) activates MLX-accelerated DiT inference.
        status_msg, success = self._dit.initialize_service(
            project_root="./ACE-Step-1.5",
            config_path=config_path,
            device="auto",
            use_mlx_dit=True,
            # compile_model=True redirects to mx.compile on MPS, which fuses MLX
            # graph operations and reduces per-step time ~4x vs uncompiled dispatch.
            # First generation pays a one-time JIT cost; all subsequent runs are fast.
            compile_model=True,
        )
        if not success:
            raise RuntimeError(f"[AceStepEngine] DiT initialization failed: {status_msg}")
        logger.info("[AceStepEngine] DiT initialized: %s", status_msg[:200])

        # ── LLM handler ──────────────────────────────────────────────────
        self._llm = LLMHandler()
        # Try the 4B model first (Qwen3-4B, 12 GB); fall back to 1.7B (8 GB).
        # backend="mlx" uses native Apple MLX; device="auto" resolves to MPS.
        for lm_path in ("acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B"):
            try:
                llm_status, llm_success = self._llm.initialize(
                    checkpoint_dir="./ACE-Step-1.5/checkpoints",
                    lm_model_path=lm_path,
                    backend="mlx",
                    device="auto",
                )
                if llm_success:
                    logger.info("[AceStepEngine] LLM initialized (%s): %s", lm_path, llm_status[:200])
                    break
                logger.warning(
                    "[AceStepEngine] LLM init failed for %s: %s — trying fallback",
                    lm_path, llm_status[:200],
                )
            except Exception as exc:  # OOM, missing checkpoint, etc.
                logger.warning(
                    "[AceStepEngine] LLM load error for %s: %s — trying fallback",
                    lm_path, exc,
                )
        else:
            logger.error("[AceStepEngine] All LLM variants failed; generation proceeds without CoT")
            llm_success = False

        self.initialized = True
        self.model_type = model_type
        logger.info(
            "[AceStepEngine] ACE-Step 1.5 (%s) loaded in %.1fs", model_type, time.time() - t0
        )

    def unload_model(self):
        """Release ACE-Step models and aggressively free memory."""
        logger.info("[AceStepEngine] Unloading ACE-Step 1.5...")

        if self._llm is not None:
            try:
                self._llm.unload()
            except Exception as exc:
                logger.warning("[AceStepEngine] LLM unload error: %s", exc)

        # DiT handler doesn't expose a shutdown_service — just delete refs
        self._dit = None
        self._llm = None
        self.initialized = False

        # Aggressive memory teardown for unified-RAM Apple Silicon
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("[AceStepEngine] ACE-Step 1.5 unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        prompt_stages: list[tuple[str, float]] | None = None,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        **kwargs,
    ) -> np.ndarray:
        """Generate instrumental meditation music via ACE-Step 1.5.

        Args:
            prompt: Text description of desired music style.
                    Used when prompt_stages is None.
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_step, total_steps).
            prompt_stages: Optional list of (prompt, duration_sec) tuples.
            lyrics: Optional structural tags or lyrics.
            bpm: Beats per minute (60-80). None means auto.
            keyscale: Musical key (e.g. "C Major"). "Auto" for detect.
                for story mode. Each stage is generated as a separate
                ACE-Step inference call and adjacent stages are blended
                with a 6-second equal-power cosine crossfade.
                Example:
                    [
                        ("calm breathing pads, soft sine waves", 90.0),
                        ("deep sleep ambient drones, very slow", 120.0),
                    ]

        Returns:
            Mono float32 numpy array at 48 000 Hz.
        """
        # Quality selection (Task 5)
        requested_model = kwargs.get("acestep_model_type", "sft")
        
        if not self.initialized or self.model_type != requested_model:
            logger.info(f"[AceStepEngine] Loading/Switching model to: {requested_model}")
            self.load_model(model_type=requested_model)

        # Handle reference audio if provided in kwargs (melody_audio from pipeline)
        melody_audio = kwargs.get("melody_audio")
        melody_sr = kwargs.get("melody_sample_rate")
        ref_path = None
        if melody_audio is not None and melody_sr is not None:
            ref_path = self._prepare_reference_audio(melody_audio, melody_sr)

        try:
            if prompt_stages is not None:
                return self._generate_story(
                    prompt_stages, progress_cb, bpm=bpm, keyscale=keyscale, 
                    reference_audio_path=ref_path
                )

            # ── Infinite Generation ───────────────────────────────────────────
            # For long tracks (> 90s), use three-phase pipeline (genesis +
            # cover continuation + boundary smoothing) at native 48 kHz.
            if total_duration_sec > 90.0:
                return self._generate_infinite(
                    prompt, total_duration_sec, progress_cb, lyrics=lyrics,
                    bpm=bpm, keyscale=keyscale, reference_audio_path=ref_path
                )

            # ── Single-stage generation ───────────────────────────────────────
            enhanced_prompt, enhanced_lyrics = self._enhance_prompt(
                prompt, duration_hint=total_duration_sec,
            )
            if lyrics:
                enhanced_lyrics = f"{enhanced_lyrics}, {lyrics}"

            logger.info(
                "[AceStepEngine] Generating %.0fs of music — caption: %s | lyrics: %s | bpm: %s | key: %s",
                total_duration_sec, enhanced_prompt[:60], enhanced_lyrics[:60], bpm, keyscale
            )

            if progress_cb is not None:
                progress_cb(0, 1)

            # Generate with validation and A/B selection (up to 3 attempts)
            from core.qa_monitor import compute_composite_score

            audio = None
            candidates: list[tuple[np.ndarray, float]] = []
            for attempt in range(3):
                audio = self._generate_single(
                    enhanced_prompt, total_duration_sec, lyrics=enhanced_lyrics,
                    bpm=bpm, keyscale=keyscale, reference_audio_path=ref_path
                )
                valid, reason = self._validate_output(audio, total_duration_sec)
                if valid:
                    score = compute_composite_score(audio, TARGET_SAMPLE_RATE)
                    candidates.append((audio, score))
                    if score > 0.8:
                        break  # good enough
                else:
                    logger.warning(
                        "[AceStepEngine] Attempt %d/3 failed validation: %s", attempt + 1, reason,
                    )
                    candidates.append((audio, 0.0))

            if candidates:
                audio = max(candidates, key=lambda c: c[1])[0]
                if len(candidates) > 1:
                    scores = [f"{s:.3f}" for _, s in candidates]
                    logger.info("[AceStepEngine] A/B selection — scores: %s", scores)
            if not valid and not any(s > 0 for _, s in candidates):
                logger.error("[AceStepEngine] All attempts failed: %s", reason)

            if progress_cb is not None:
                progress_cb(1, 1)

            return audio
        finally:
            if ref_path and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
                except Exception as e:
                    logger.warning(f"[AceStepEngine] Failed to remove temp ref audio: {e}")

    def _generate_story(
        self,
        prompt_stages: list[tuple[str, float]],
        progress_cb=None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        reference_audio_path: str | None = None,
    ) -> np.ndarray:
        """Generate story mode audio: one ACE-Step call per stage, then crossfade.

        Each stage's prompt is enhanced independently so the model receives
        full meditation-optimised guidance for each tonal world. Adjacent
        stages are blended with a 6-second equal-power cosine crossfade in
        the numpy domain (after postprocessing) to create a smooth, natural
        transition without abrupt tonal jumps.

        Args:
            prompt_stages: List of (raw_prompt, duration_sec) pairs.
            progress_cb: Called with (completed_stages, total_stages).

        Returns:
            Mono float32 numpy array at TARGET_SAMPLE_RATE (48 000 Hz).
        """
        n = len(prompt_stages)
        logger.info(
            "[AceStepEngine] Story mode: %d stage(s), total ~%.0fs",
            n, sum(d for _, d in prompt_stages),
        )

        if progress_cb is not None:
            progress_cb(0, n)

        stage_audios: list[np.ndarray] = []
        for i, (stage_prompt, stage_duration) in enumerate(prompt_stages):
            logger.info(
                "[AceStepEngine] Story stage %d/%d (%.0fs) — %s",
                i + 1, n, stage_duration, stage_prompt[:80],
            )
            enhanced_cap, enhanced_lyr = self._enhance_prompt(stage_prompt, duration_hint=stage_duration)
            audio = self._generate_single(
                enhanced_cap, stage_duration, lyrics=enhanced_lyr,
                bpm=bpm, keyscale=keyscale, reference_audio_path=reference_audio_path
            )
            stage_audios.append(audio)

            if progress_cb is not None:
                progress_cb(i + 1, n)

        return self._crossfade_stages(stage_audios, _STORY_CROSSFADE_SEC)

    def _generate_infinite(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        reference_audio_path: str | None = None,
    ) -> np.ndarray:
        """Two-phase long-form generation at native 48 kHz stereo.

        Phase 1 — Genesis: text2music for the initial 90-second anchor segment.
        Phase 2 — Repaint continuation: overlapping repaint calls that treat the
                  last 20 seconds of accumulated audio as anchor context, then
                  generate new audio from that point forward.  Repaint produces
                  seamless transitions at the model level — no post-hoc STFT
                  crossfades required.

        All intermediate audio stays at 48 kHz stereo.  Single postprocess
        at the end eliminates quality-degrading sample-rate round-trips.
        """
        import tempfile
        import soundfile as sf

        GENESIS_LEN = 90.0
        CONTINUATION_LEN = 60.0

        enhanced_prompt, enhanced_lyrics = self._enhance_prompt(
            prompt, duration_hint=total_duration_sec,
        )
        if lyrics:
            enhanced_lyrics = f"{enhanced_lyrics}, {lyrics}"

        logger.info(
            "[AceStepEngine] Three-phase infinite: %.0fs target", total_duration_sec,
        )

        # ── Phase 1: Genesis ─────────────────────────────────────────────
        genesis_len = min(GENESIS_LEN, total_duration_sec)
        logger.info("[AceStepEngine] Phase 1 (Genesis): %.0fs", genesis_len)

        if progress_cb:
            progress_cb(0, max(1, int(total_duration_sec / CONTINUATION_LEN)))

        full_tensor, sr = self._generate_single_raw(
            enhanced_prompt, genesis_len, lyrics=enhanced_lyrics,
            bpm=bpm, keyscale=keyscale, reference_audio_path=reference_audio_path,
        )
        seg_num = 1

        # Downmix genesis to mono (1, T) so that repaint continuations
        # (which return mono numpy → (1, T) tensor) can be concatenated
        # along dim=-1 without a channel-count mismatch.
        if full_tensor.ndim > 1 and full_tensor.shape[0] > 1:
            full_tensor = full_tensor.mean(dim=0, keepdim=True)

        # ── Phase 2: Overlapping repaint continuation ────────────────────
        # repaint (task_type="repaint") treats the first N seconds as anchor
        # and generates new audio from repainting_start onward — producing
        # seamless transitions without needing post-hoc STFT crossfades.
        CONTINUATION_OVERLAP = 20.0  # seconds of context passed to each repaint call

        while full_tensor.shape[-1] / sr < total_duration_sec - 1.0:
            seg_num += 1
            full_audio_len = full_tensor.shape[-1] / sr
            remaining = total_duration_sec - full_audio_len
            segment_new_len = min(60.0, remaining)
            repaint_total = CONTINUATION_OVERLAP + segment_new_len

            # Extract the last CONTINUATION_OVERLAP seconds as repaint context
            overlap_samples = int(CONTINUATION_OVERLAP * sr)
            overlap_samples = min(overlap_samples, full_tensor.shape[-1])
            src_chunk = full_tensor[:, -overlap_samples:]

            logger.info(
                "[AceStepEngine] Phase 2 seg %d: repaint continuation (overlap=%.0fs, new=%.0fs)",
                seg_num, CONTINUATION_OVERLAP, segment_new_len,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            # soundfile expects (samples, channels) — transpose from (C, T)
            sf.write(tmp_path, src_chunk.cpu().numpy().T, sr)

            continuation_audio = None
            try:
                continuation_audio = self._generate_single_repaint(
                    enhanced_prompt,
                    tmp_path,
                    CONTINUATION_OVERLAP,
                    repaint_total,
                    lyrics=enhanced_lyrics,
                    bpm=bpm,
                    keyscale=keyscale,
                    reference_audio_path=reference_audio_path,
                )
            except Exception as exc:
                logger.warning(
                    "[AceStepEngine] Repaint continuation seg %d error: %s — stopping early",
                    seg_num, exc,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if continuation_audio is None:
                logger.warning("[AceStepEngine] Repaint continuation returned None — stopping early")
                break

            # continuation_audio is mono float32 numpy from _generate_single_repaint
            # Convert to (1, T) torch tensor to match full_tensor shape
            cont_tensor = torch.from_numpy(continuation_audio).unsqueeze(0)

            # Append only the newly generated tail (after the overlap region)
            new_start_sample = int(CONTINUATION_OVERLAP * sr)
            new_audio = cont_tensor[:, new_start_sample:]
            full_tensor = torch.cat([full_tensor, new_audio], dim=-1)

            if progress_cb:
                done = full_tensor.shape[-1] / sr
                progress_cb(
                    int(done / CONTINUATION_LEN),
                    max(1, int(total_duration_sec / CONTINUATION_LEN)),
                )

        # ── Final: single postprocess to 24 kHz mono ────────────────────
        logger.info(
            "[AceStepEngine] Final postprocess: %.1fs at %d Hz",
            full_tensor.shape[-1] / sr, sr,
        )
        return self._postprocess(full_tensor, sr)

    def _get_inference_steps(self, is_repaint: bool = False) -> int:
        """Resolve inference steps based on current model type."""
        if self.model_type == "turbo":
            return 8
        return _INFERENCE_STEPS_REPAINT if is_repaint else _INFERENCE_STEPS

    def _generate_single(
        self, 
        enhanced_prompt: str, 
        duration_sec: float, 
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        reference_audio_path: str | None = None
    ) -> np.ndarray:
        """One ACE-Step inference call → 24 kHz mono float32 numpy array.

        Builds GenerationParams with the tuned quality knobs, calls
        generate_music(), extracts the audio tensor, and returns the
        postprocessed result.  The caller is responsible for prompt
        enhancement before passing here.
        """
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        t0 = time.time()

        params = GenerationParams(
            caption=enhanced_prompt,
            lyrics=lyrics or "[Instrumental]",
            instrumental=True,
            duration=duration_sec,
            reference_audio=reference_audio_path,
            bpm=bpm if bpm and bpm > 0 else None,
            keyscale=keyscale if keyscale and keyscale != "Auto" else "",
            vocal_language="unknown",
            inference_steps=self._get_inference_steps(is_repaint=False),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=_USE_ADG,
            thinking=True,
            lm_temperature=_LM_TEMPERATURE,
            use_cot_metas=True,
            use_cot_caption=True,
            enable_normalization=True,
            cfg_interval_end=_CFG_INTERVAL_END,
            shift=_SHIFT,
            infer_method=_INFER_METHOD,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
        )

        # ── Generate ─────────────────────────────────────────────────────
        result = generate_music(
            dit_handler=self._dit,
            llm_handler=self._llm,
            params=params,
            config=config,
            save_dir=None,
        )

        if not result.success:
            raise RuntimeError(
                f"[AceStepEngine] Generation failed: {result.error or result.status_message}"
            )

        if not result.audios:
            raise RuntimeError("[AceStepEngine] Generation returned no audio outputs")

        # Extract audio tensor from the first (and only) batch item
        audio_dict = result.audios[0]
        tensor = audio_dict["tensor"]           # 48 kHz stereo torch.Tensor
        sample_rate = audio_dict.get("sample_rate", NATIVE_SAMPLE_RATE)

        logger.info("[AceStepEngine] Generation done in %.1fs", time.time() - t0)

        return self._postprocess(tensor, sample_rate)

    def _generate_single_repaint(
        self, 
        enhanced_prompt: str, 
        src_audio_path: str,
        repaint_start: float,
        repaint_end: float,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        reference_audio_path: str | None = None
    ) -> np.ndarray:
        """One ACE-Step Repaint inference call."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        t0 = time.time()
        
        # In Repaint mode, 'duration' is derived from src_audio if not provided,
        # but we specify it to be sure.
        params = GenerationParams(
            task_type="repaint",
            caption=enhanced_prompt,
            lyrics=lyrics or "[Instrumental]",
            instrumental=True,
            src_audio=src_audio_path,
            reference_audio=reference_audio_path,
            bpm=bpm if bpm and bpm > 0 else None,
            keyscale=keyscale if keyscale and keyscale != "Auto" else "",
            vocal_language="unknown",
            repainting_start=repaint_start,
            repainting_end=repaint_end,
            duration=repaint_end,
            inference_steps=self._get_inference_steps(is_repaint=True),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=False,
            thinking=False,
            lm_temperature=_LM_TEMPERATURE,
            enable_normalization=True,
            cfg_interval_end=_CFG_INTERVAL_END,
            shift=_SHIFT,
            infer_method=_INFER_METHOD,
        )
        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
        )

        result = generate_music(
            dit_handler=self._dit,
            llm_handler=self._llm,
            params=params,
            config=config,
            save_dir=None,
        )

        if not result.success:
            raise RuntimeError(f"[AceStepEngine] Repaint failed: {result.error}")

        audio_dict = result.audios[0]
        tensor = audio_dict["tensor"]
        sample_rate = audio_dict.get("sample_rate", NATIVE_SAMPLE_RATE)

        logger.info("[AceStepEngine] Repaint done in %.1fs", time.time() - t0)

        return self._postprocess(tensor, sample_rate)

    # ------------------------------------------------------------------
    # Raw / cover / boundary helpers (three-phase pipeline)
    # ------------------------------------------------------------------

    def _generate_single_raw(
        self,
        enhanced_prompt: str,
        duration_sec: float,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
        reference_audio_path: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """One ACE-Step inference call → raw 48 kHz stereo tensor.

        Same as ``_generate_single`` but returns the native tensor before
        postprocessing.  Used by the three-phase infinite pipeline to avoid
        quality-degrading sample-rate round-trips.

        Returns:
            (tensor, sample_rate) where tensor is shape (C, T) at 48 kHz.
        """
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        t0 = time.time()

        params = GenerationParams(
            caption=enhanced_prompt,
            lyrics=lyrics or "[Instrumental]",
            instrumental=True,
            duration=duration_sec,
            reference_audio=reference_audio_path,
            bpm=bpm if bpm and bpm > 0 else None,
            keyscale=keyscale if keyscale and keyscale != "Auto" else "",
            vocal_language="unknown",
            inference_steps=self._get_inference_steps(is_repaint=False),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=_USE_ADG,
            thinking=True,
            lm_temperature=_LM_TEMPERATURE,
            use_cot_metas=True,
            use_cot_caption=True,
            enable_normalization=True,
            cfg_interval_end=_CFG_INTERVAL_END,
            shift=_SHIFT,
            infer_method=_INFER_METHOD,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")

        result = generate_music(
            dit_handler=self._dit, llm_handler=self._llm,
            params=params, config=config, save_dir=None,
        )

        if not result.success:
            raise RuntimeError(
                f"[AceStepEngine] Generation failed: {result.error or result.status_message}"
            )
        if not result.audios:
            raise RuntimeError("[AceStepEngine] Generation returned no audio outputs")

        audio_dict = result.audios[0]
        tensor = audio_dict["tensor"]
        sample_rate = audio_dict.get("sample_rate", NATIVE_SAMPLE_RATE)

        # Ensure (C, T) shape on CPU
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        logger.info("[AceStepEngine] Raw generation done in %.1fs", time.time() - t0)
        return tensor, sample_rate

    @staticmethod
    def _validate_output(audio: np.ndarray, duration_sec: float) -> tuple[bool, str]:
        """Validate generated audio quality.

        Checks for NaN/Inf, near-silence, excessive clipping, and short output.
        """
        if np.isnan(audio).any() or np.isinf(audio).any():
            return False, "NaN/Inf detected"

        if len(audio) < int(duration_sec * TARGET_SAMPLE_RATE * 0.5):
            actual = len(audio) / TARGET_SAMPLE_RATE
            return False, f"Output too short: {actual:.1f}s vs {duration_sec:.1f}s requested"

        rms = float(np.sqrt(np.mean(audio ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        if rms_db < -50.0:
            return False, f"Near-silent output ({rms_db:.1f} dBFS)"

        clip_ratio = float(np.sum(np.abs(audio) >= 0.99)) / max(len(audio), 1)
        if clip_ratio > 0.05:
            return False, f"Excessive clipping ({clip_ratio * 100:.1f}%)"

        return True, "OK"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stft_crossfade(
        tail: np.ndarray,
        head: np.ndarray,
    ) -> np.ndarray:
        """STFT overlap-add crossfade in log-magnitude domain.

        Interpolates magnitudes in dB (perceptually linear) and switches
        phase at the midpoint.  Produces smoother transitions than time-domain
        cosine² for sustained drones and singing bowls.

        Falls back to cosine² crossfade if STFT produces anomalous results.
        """
        from scipy.signal import stft, istft
        import math

        overlap = len(tail)
        n_fft = 2048
        hop = 512

        if overlap < n_fft:
            # Too short for STFT — use cosine² fallback
            t = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
            return tail * np.cos(t) ** 2 + head * np.sin(t) ** 2

        try:
            _, _, S1 = stft(tail, fs=TARGET_SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop)
            _, _, S2 = stft(head, fs=TARGET_SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop)

            # Align frame counts (stft may produce slightly different lengths)
            n_frames = min(S1.shape[1], S2.shape[1])
            S1 = S1[:, :n_frames]
            S2 = S2[:, :n_frames]

            mag1_db = 20.0 * np.log10(np.abs(S1) + 1e-8)
            mag2_db = 20.0 * np.log10(np.abs(S2) + 1e-8)

            # Linear fade in dB domain (perceptually smooth)
            fade = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[np.newaxis, :]
            mag_blend_db = mag1_db * (1.0 - fade) + mag2_db * fade
            mag_blend = 10.0 ** (mag_blend_db / 20.0)

            # Phase: use outgoing for first half, incoming for second half
            mid = n_frames // 2
            phase = np.empty_like(S1)
            phase[:, :mid] = np.angle(S1[:, :mid])
            phase[:, mid:] = np.angle(S2[:, mid:])

            S_blend = mag_blend * np.exp(1j * phase)
            _, blended = istft(S_blend, fs=TARGET_SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop)

            # Trim or pad to exact overlap length
            if len(blended) >= overlap:
                blended = blended[:overlap]
            else:
                blended = np.pad(blended, (0, overlap - len(blended)))

            # Energy anomaly check: >3 dB deviation from expected → fallback
            expected_rms = np.sqrt(
                0.5 * np.mean(tail ** 2) + 0.5 * np.mean(head ** 2)
            )
            actual_rms = np.sqrt(np.mean(blended ** 2))
            if expected_rms > 1e-8:
                db_diff = abs(20.0 * np.log10(max(actual_rms, 1e-10) / expected_rms))
                if db_diff > 3.0:
                    logger.warning(
                        "[AceStepEngine] STFT crossfade energy anomaly (%.1f dB) — "
                        "falling back to cosine²", db_diff
                    )
                    raise ValueError("energy anomaly")

            return blended.astype(np.float32)

        except Exception:
            # Fallback: cosine² crossfade
            t = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
            return (tail * np.cos(t) ** 2 + head * np.sin(t) ** 2).astype(np.float32)

    @staticmethod
    def _crossfade_stages(
        segments: list[np.ndarray],
        crossfade_sec: float,
    ) -> np.ndarray:
        """Join story mode segments with STFT crossfades in log-magnitude domain.

        STFT crossfading interpolates magnitudes in dB (perceptually linear)
        for smoother transitions on sustained drones and singing bowls compared
        to time-domain cosine² blending.  Falls back to cosine² automatically
        if the STFT produces energy anomalies.

        Args:
            segments:      List of mono float32 arrays at TARGET_SAMPLE_RATE.
            crossfade_sec: Blend duration in seconds.  Clamped to the
                           length of the shorter adjacent segment.

        Returns:
            Single mono float32 array — the stitched meditation track.
        """
        if len(segments) == 1:
            return segments[0]

        fade_samples = int(crossfade_sec * TARGET_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            overlap = min(fade_samples, len(result), len(seg))

            blended = AceStepEngine._stft_crossfade(
                result[-overlap:].copy(),
                seg[:overlap].copy(),
            )
            result = np.concatenate([result[:-overlap], blended, seg[overlap:]])

        return result

    @classmethod
    def _prepare_reference_audio(cls, audio_data: np.ndarray, sr: int) -> str:
        """Save a memory-hosted audio array to a temp file for ACE-Step.

        ACE-Step inference requires a file path for reference audio.
        """
        import tempfile
        import soundfile as sf
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            sf.write(path, audio_data, sr)
        finally:
            os.close(fd)
        logger.info("[AceStepEngine] Prepared temp reference audio: %s", path)
        return path

    # ── MESA Prompt Framework ─────────────────────────────────────────────
    # Mood + Elements + Structure + Application

    # Base caption tags that anchor output in meditation territory.
    # Filtered at runtime to avoid duplicating words the user already supplied.
    _MESA_BASE_TAGS = (
        "ambient, meditation, calm, peaceful, warm, spacious, "
        "soft dynamics, gentle, soothing, smooth texture, "
        "slow tempo, drone pads, harmonic layers, ethereal, "
        "no percussion, no drums, no beat, slow harmonic evolution, "
        "sustained pads, sine wave layers, breathing space, no lead instrument, "
        "high fidelity, deep warmth, Tuned to 432 Hz for natural resonance"
    )

    @classmethod
    def _enhance_prompt(
        cls, user_prompt: str, duration_hint: float = 120.0,
    ) -> tuple[str, str]:
        """Build (caption, lyrics) using the MESA framework.

        Caption (Mood + Elements + Application):
        - Prepend meditation base tags (filtered to avoid duplicating user words)
        - Append user prompt verbatim (preserving instrument/style choices)
        - Append "no vocals, instrumental" constraint
        - NO contradictory directives ("no melody", "no chord changes")

        Lyrics (Structure):
        - [Instrumental] as primary mode token
        - Structural section tags scaled by duration
        """
        user_lower = user_prompt.lower().strip()

        # Filter base tags to avoid duplicating what the user already said
        base_parts = [t.strip() for t in cls._MESA_BASE_TAGS.split(",") if t.strip()]
        filtered_base = []
        for part in base_parts:
            words = [w for w in part.split() if len(w) > 3]
            if not any(w in user_lower for w in words):
                filtered_base.append(part)

        # Build caption: filtered base + user prompt + instrumental constraint
        caption_parts = filtered_base + [user_prompt.strip()]
        if "no vocal" not in user_lower and "instrumental" not in user_lower:
            caption_parts.append("no vocals, instrumental")

        caption = ", ".join(p for p in caption_parts if p).strip().rstrip(",")
        lyrics = cls._build_structural_lyrics(user_prompt, duration_hint)

        return caption, lyrics

    @classmethod
    def _build_structural_lyrics(
        cls, user_prompt: str, duration_hint: float = 120.0,
    ) -> str:
        """Build structural lyrics with section tags.

        Uses [Instrumental] as the primary section tag — this is the proper
        training-vocabulary token for instrumental tracks.  [Verse]/[Bridge]
        carry implicit vocal bias from training data.  [Intro] and [Outro]
        are kept as they have clear semantic meaning for beginnings/endings.
        Prose cues after dashes were removed: they are out-of-distribution
        for the SFT LM planner and degrade prompt adherence.  Section count
        scales with duration.
        """
        # Short tracks (up to 90s): minimal structure
        if duration_hint <= 90.0:
            return (
                "[Instrumental]\n\n"
                "[Intro]\n\n"
                "[Instrumental]\n\n"
                "[Outro]"
            )

        # Medium tracks (90s - 300s): full journey structure with compositional arc
        if duration_hint <= 300.0:
            return (
                "[Instrumental]\n\n"
                "[Intro]\n\n"
                "[Verse]\n\n"
                "[Instrumental]\n\n"
                "[Chorus]\n\n"
                "[Instrumental]\n\n"
                "[Outro]"
            )

        # Long tracks (> 300s / 5 min): expanded arc for meditation journey
        return (
            "[Instrumental]\n\n"
            "[Intro]\n\n"
            "[Verse]\n\n"
            "[Instrumental]\n\n"
            "[Chorus]\n\n"
            "[Bridge]\n\n"
            "[Instrumental]\n\n"
            "[Verse]\n\n"
            "[Chorus]\n\n"
            "[Outro]"
        )

    @staticmethod
    def _postprocess(tensor: torch.Tensor, source_rate: int = NATIVE_SAMPLE_RATE) -> np.ndarray:
        """Convert ACE-Step 48 kHz stereo output to 48 kHz mono float32.

        Preserves the native 48 kHz sample rate to avoid discarding spectral
        content above 12 kHz (24 kHz Nyquist). The pipeline upsamples TTS to
        48 kHz to match before mixing.

        Chain:
        1. CPU / float32 conversion
        2. Stereo → mono (channel average)
        3. Peak normalization to -1 dBFS (consistent output level)
        4. Safety clip to [-1, 1]
        """
        # 1. CPU / float32
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.float()

        # 2. Stereo → mono
        if tensor.ndim > 1 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        # 3. Peak normalization to -1 dBFS
        peak = tensor.abs().max()
        if peak > 1e-6:
            target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS ≈ 0.891
            tensor = tensor * (target_peak / peak)

        # 4. Safety clip
        tensor = tensor.clamp(-1.0, 1.0)

        return tensor.squeeze().cpu().numpy().astype(np.float32)
