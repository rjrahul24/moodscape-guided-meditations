"""ACE-Step 1.5 wrapper — text-to-music via MLX on Apple Silicon.

Model: ACE-Step 1.5 (DiT decoder + LM planner)
Device: MPS (PyTorch) with MLX-accelerated DiT and LLM
Output: Mono float32 at 24 kHz (native 48 kHz stereo downmixed and resampled)
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
import torchaudio

logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 48000       # ACE-Step native output rate
TARGET_SAMPLE_RATE = 24000       # Pipeline standard rate (matches MusicEngine)

# ── Generation quality knobs ──────────────────────────────────────────────────
# guidance_scale: CFG strength for the SFT model.
# SFT optimal range is 5.0–7.0; 5.0 balances prompt adherence with smooth
# ambient textures without amplifying spectral roughness.
_GUIDANCE_SCALE = 5.0

# inference_steps: SFT supports up to 50 steps. Going higher causes error
# accumulation that degrades output quality. 50 is maximum detail without
# degradation artifacts.
_INFERENCE_STEPS = 50
_INFERENCE_STEPS_REPAINT = 50

# lm_temperature: Controls LM planner creativity.
# 0.85 allows richer tonal palettes for ambient while maintaining coherence.
_LM_TEMPERATURE = 0.85

# Enable Adaptive Dual Guidance for the base (non-turbo) model.
# ADG applies two complementary CFG branches that reinforce each other:
# this significantly reduces spectral noise without increasing inference time.
_USE_ADG = True

# Story mode: crossfade duration between adjacent stages (seconds).
# 6 seconds gives a natural, unhurried transition between tonal worlds
# without cutting into the usable content of either stage.
_STORY_CROSSFADE_SEC = 6.0


class AceStepEngine:
    """Generates ambient background music via ACE-Step 1.5.

    Uses the full-quality ``acestep-v15-sft`` DiT config and the
    ``acestep-5Hz-lm-1.7B`` language model for Chain-of-Thought planning.
    Forces instrumental mode with no vocals.  Output is converted from
    48 kHz stereo to 24 kHz mono to honour the pipeline contract.

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
        import os

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
        # backend="mlx" uses native Apple MLX for the language model.
        # device="auto" resolves to "mps" — MLX loads weights separately.
        llm_status, llm_success = self._llm.initialize(
            checkpoint_dir="./ACE-Step-1.5/checkpoints",
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="mlx",
            device="auto",
        )
        if not llm_success:
            logger.warning("[AceStepEngine] LLM initialization issue: %s", llm_status[:200])
            # LLM failure is non-fatal — generation can proceed without CoT
        else:
            logger.info("[AceStepEngine] LLM initialized: %s", llm_status[:200])

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
            Mono float32 numpy array at 24 000 Hz — same contract as
            ``MusicEngine.generate()``.
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

            # Generate with validation and retry (up to 3 attempts)
            audio = None
            for attempt in range(3):
                audio = self._generate_single(
                    enhanced_prompt, total_duration_sec, lyrics=enhanced_lyrics,
                    bpm=bpm, keyscale=keyscale, reference_audio_path=ref_path
                )
                valid, reason = self._validate_output(audio, total_duration_sec)
                if valid:
                    break
                logger.warning(
                    "[AceStepEngine] Attempt %d/3 failed validation: %s", attempt + 1, reason,
                )
            if not valid:
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
            Mono float32 numpy array at TARGET_SAMPLE_RATE (24 000 Hz).
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
        """Three-phase long-form generation at native 48 kHz stereo.

        Phase 1 — Genesis: text2music for the initial anchor segment.
        Phase 2 — Continuation: cover task with decaying audio_cover_strength
                  to generate harmonically coherent subsequent segments.
        Phase 3 — Boundary Smoothing: repaint on seam regions.

        All intermediate audio stays at 48 kHz stereo.  Single postprocess
        at the end eliminates quality-degrading sample-rate round-trips.
        """
        import tempfile
        import soundfile as sf

        GENESIS_LEN = 60.0
        CONTINUATION_LEN = 60.0
        CONTEXT_LEN = 30.0
        CROSSFADE_SEC = 2.0
        BOUNDARY_WINDOW_SEC = 5.0

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
        seam_positions: list[int] = []
        seg_num = 1

        # ── Phase 2: Cover Continuation ──────────────────────────────────
        while full_tensor.shape[-1] / sr < total_duration_sec - 1.0:
            seg_num += 1
            remaining = total_duration_sec - (full_tensor.shape[-1] / sr)
            next_len = min(CONTINUATION_LEN, remaining + CROSSFADE_SEC)

            # Cover strength decays: 0.85, 0.80, 0.75, 0.70 (floor)
            cover_strength = max(0.70, 0.90 - 0.05 * seg_num)

            # Use last CONTEXT_LEN seconds as cover source
            context_samples = min(
                int(CONTEXT_LEN * sr), full_tensor.shape[-1],
            )
            context_tensor = full_tensor[:, -context_samples:]

            logger.info(
                "[AceStepEngine] Phase 2 seg %d: cover (strength=%.2f, %.0fs)",
                seg_num, cover_strength, next_len,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                src_wav = os.path.join(tmpdir, "cover_src.wav")
                # soundfile expects (samples, channels) — transpose from (C, T)
                sf.write(src_wav, context_tensor.cpu().numpy().T, sr)

                new_tensor, _ = self._generate_cover_continuation(
                    enhanced_prompt, src_wav, next_len,
                    cover_strength=cover_strength,
                    lyrics=enhanced_lyrics, bpm=bpm, keyscale=keyscale,
                )

            # Crossfade at native 48 kHz stereo
            fade_samples = int(CROSSFADE_SEC * sr)
            fade_samples = min(fade_samples, full_tensor.shape[-1], new_tensor.shape[-1])

            t = torch.linspace(0.0, torch.pi / 2.0, fade_samples)
            fade_out = (torch.cos(t) ** 2).unsqueeze(0)   # (1, F) broadcasts over channels
            fade_in = (torch.sin(t) ** 2).unsqueeze(0)

            blended = full_tensor[:, -fade_samples:] * fade_out + new_tensor[:, :fade_samples] * fade_in
            seam_pos = full_tensor.shape[-1] - fade_samples
            full_tensor = torch.cat([
                full_tensor[:, :-fade_samples], blended, new_tensor[:, fade_samples:],
            ], dim=-1)
            seam_positions.append(seam_pos)

            if progress_cb:
                done = full_tensor.shape[-1] / sr
                progress_cb(
                    int(done / CONTINUATION_LEN),
                    max(1, int(total_duration_sec / CONTINUATION_LEN)),
                )

        # ── Phase 3: Boundary Smoothing ──────────────────────────────────
        for i, seam_pos in enumerate(seam_positions):
            logger.info(
                "[AceStepEngine] Phase 3: smoothing boundary %d/%d at %.1fs",
                i + 1, len(seam_positions), seam_pos / sr,
            )
            full_tensor = self._smooth_boundary(
                full_tensor, sr, seam_pos,
                window_sec=BOUNDARY_WINDOW_SEC,
                enhanced_prompt=enhanced_prompt,
                lyrics=enhanced_lyrics, bpm=bpm, keyscale=keyscale,
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

    def _generate_cover_continuation(
        self,
        enhanced_prompt: str,
        src_audio_path: str,
        duration_sec: float,
        cover_strength: float = 0.80,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
    ) -> tuple[torch.Tensor, int]:
        """Cover-task continuation → raw 48 kHz stereo tensor.

        Generates a new segment that inherits the harmonic DNA of the source
        audio.  ``audio_cover_strength`` controls how much structure is
        preserved (0.85 = close to source, 0.70 = gentle evolution).
        """
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        t0 = time.time()

        params = GenerationParams(
            task_type="cover",
            caption=enhanced_prompt,
            lyrics=lyrics or "[Instrumental]",
            instrumental=True,
            src_audio=src_audio_path,
            audio_cover_strength=cover_strength,
            duration=duration_sec,
            bpm=bpm if bpm and bpm > 0 else None,
            keyscale=keyscale if keyscale and keyscale != "Auto" else "",
            vocal_language="unknown",
            inference_steps=self._get_inference_steps(is_repaint=False),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=False,
            thinking=False,
            lm_temperature=_LM_TEMPERATURE,
            enable_normalization=True,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")

        result = generate_music(
            dit_handler=self._dit, llm_handler=self._llm,
            params=params, config=config, save_dir=None,
        )

        if not result.success:
            raise RuntimeError(
                f"[AceStepEngine] Cover continuation failed: {result.error or result.status_message}"
            )
        if not result.audios:
            raise RuntimeError("[AceStepEngine] Cover returned no audio outputs")

        audio_dict = result.audios[0]
        tensor = audio_dict["tensor"]
        sample_rate = audio_dict.get("sample_rate", NATIVE_SAMPLE_RATE)

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        logger.info("[AceStepEngine] Cover continuation done in %.1fs", time.time() - t0)
        return tensor, sample_rate

    def _smooth_boundary(
        self,
        full_tensor: torch.Tensor,
        sr: int,
        seam_sample_pos: int,
        window_sec: float = 5.0,
        enhanced_prompt: str | None = None,
        lyrics: str | None = None,
        bpm: int | None = 50,
        keyscale: str | None = "Auto",
    ) -> torch.Tensor:
        """Repaint a short window around a seam to smooth the boundary.

        Extracts a ~30s context window centered on the seam, repaints the
        middle ``window_sec`` seconds, and splices the result back.
        """
        import tempfile
        import soundfile as sf

        half_window = int(window_sec * sr / 2)
        context_pad = int(12.5 * sr)  # ~12.5s of context on each side

        # Define the extraction window (context + repaint region + context)
        extract_start = max(0, seam_sample_pos - half_window - context_pad)
        extract_end = min(full_tensor.shape[-1], seam_sample_pos + half_window + context_pad)
        window_tensor = full_tensor[:, extract_start:extract_end]
        window_duration = window_tensor.shape[-1] / sr

        # Repaint boundaries within the extracted window
        repaint_start = (seam_sample_pos - half_window - extract_start) / sr
        repaint_end = (seam_sample_pos + half_window - extract_start) / sr
        repaint_start = max(0.0, repaint_start)
        repaint_end = min(window_duration, repaint_end)

        if repaint_end - repaint_start < 0.5:
            return full_tensor  # window too small to repaint

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        with tempfile.TemporaryDirectory() as tmpdir:
            src_wav = os.path.join(tmpdir, "boundary_src.wav")
            sf.write(src_wav, window_tensor.cpu().numpy().T, sr)

            params = GenerationParams(
                task_type="repaint",
                caption=enhanced_prompt or "",
                lyrics=lyrics or "[Instrumental]",
                instrumental=True,
                src_audio=src_wav,
                repainting_start=repaint_start,
                repainting_end=repaint_end,
                duration=window_duration,
                bpm=bpm if bpm and bpm > 0 else None,
                keyscale=keyscale if keyscale and keyscale != "Auto" else "",
                vocal_language="unknown",
                inference_steps=self._get_inference_steps(is_repaint=True),
                guidance_scale=_GUIDANCE_SCALE,
                use_adg=False,
                thinking=False,
                lm_temperature=_LM_TEMPERATURE,
                enable_normalization=True,
            )
            config = GenerationConfig(batch_size=1, audio_format="wav")

            result = generate_music(
                dit_handler=self._dit, llm_handler=self._llm,
                params=params, config=config, save_dir=None,
            )

        if not result.success:
            logger.warning("[AceStepEngine] Boundary smoothing failed: %s", result.error)
            return full_tensor  # non-fatal — keep the crossfaded version

        repainted_tensor = result.audios[0]["tensor"]
        if repainted_tensor.device.type != "cpu":
            repainted_tensor = repainted_tensor.cpu()
        repainted_tensor = repainted_tensor.float()
        if repainted_tensor.ndim == 1:
            repainted_tensor = repainted_tensor.unsqueeze(0)

        # Splice the repainted region back into full_tensor.
        # Use a short (0.1s) crossfade at splice edges to avoid clicks.
        splice_fade = min(int(0.1 * sr), half_window)
        rp_start_sample = seam_sample_pos - half_window
        rp_end_sample = seam_sample_pos + half_window
        rp_start_in_window = rp_start_sample - extract_start
        rp_end_in_window = rp_end_sample - extract_start

        # Ensure indices are in bounds
        rp_start_in_window = max(0, rp_start_in_window)
        rp_end_in_window = min(repainted_tensor.shape[-1], rp_end_in_window)
        rp_start_sample = max(0, rp_start_sample)
        rp_end_sample = min(full_tensor.shape[-1], rp_end_sample)

        repainted_region = repainted_tensor[:, rp_start_in_window:rp_end_in_window]
        region_len = min(repainted_region.shape[-1], rp_end_sample - rp_start_sample)
        if region_len > 0:
            full_tensor[:, rp_start_sample:rp_start_sample + region_len] = repainted_region[:, :region_len]

        return full_tensor

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
    def _crossfade_stages(
        segments: list[np.ndarray],
        crossfade_sec: float,
    ) -> np.ndarray:
        """Join story mode segments with equal-power cosine crossfades.

        Args:
            segments:      List of mono float32 arrays at TARGET_SAMPLE_RATE.
            crossfade_sec: Blend duration in seconds.  Clamped to the
                           length of the shorter adjacent segment.

        Returns:
            Single mono float32 array — the stitched meditation track.
        """
        if len(segments) == 1:
            return segments[0]

        import math
        fade_samples = int(crossfade_sec * TARGET_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            overlap = min(fade_samples, len(result), len(seg))
            # Equal-power (cosine²) fade prevents energy dips at the seam
            t = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)
            fade_out = np.cos(t) ** 2
            fade_in  = np.cos(math.pi / 2.0 - t) ** 2

            blended = result[-overlap:] * fade_out + seg[:overlap] * fade_in
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
        "soft dynamics, gentle, soothing, "
        "high fidelity, studio quality, clean production"
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

        Uses standard training-vocabulary tags ([Intro], [Verse], [Bridge],
        [Outro]) with descriptors.  Section count scales with duration to
        maintain coherent structure.
        """
        # Short tracks (up to 90s): minimal structure
        if duration_hint <= 90.0:
            return (
                "[Instrumental]\n\n"
                "[Intro - Gentle ambient texture emerging softly]\n\n"
                "[Verse - Main theme, warm and meditative, slowly developing]\n\n"
                "[Outro - Gradual fade, dissolving into stillness]"
            )

        # Medium tracks (90s - 300s): full journey structure
        if duration_hint <= 300.0:
            return (
                "[Instrumental]\n\n"
                "[Intro - Soft ambient texture fading in from silence, establishing space]\n\n"
                "[Verse - Primary harmonic landscape, slow and contemplative]\n\n"
                "[Bridge - Subtle tonal shift, deeper warmth, new colors emerge]\n\n"
                "[Verse - Return to main theme with gentle enrichment]\n\n"
                "[Outro - Extended fade, all elements dissolving gently into silence]"
            )

        # Long tracks (> 300s / 5 min): expanded arc for meditation journey
        return (
            "[Instrumental]\n\n"
            "[Intro - Barely audible ambient wash fading in from pure silence]\n\n"
            "[Verse - Primary harmonic landscape establishes slowly, warm and grounding]\n\n"
            "[Bridge - Texture deepens, tonal palette shifts to richer colors]\n\n"
            "[Interlude - Spacious minimal passage, stillness and breath]\n\n"
            "[Verse - Main theme returns, enriched with new harmonic layers]\n\n"
            "[Bridge - Gentle upward shift, lighter textures, approaching resolution]\n\n"
            "[Outro - Long extended fade, everything dissolving into peaceful silence]"
        )

    @staticmethod
    def _postprocess(tensor: torch.Tensor, source_rate: int = NATIVE_SAMPLE_RATE) -> np.ndarray:
        """Convert ACE-Step 48 kHz stereo output to 24 kHz mono float32.

        Minimal processing — the 1D VAE produces near-lossless quality.
        All spectral shaping is handled by the downstream Pedalboard FX
        chain (make_acestep_music_chain) at 44.1 kHz.

        Chain:
        1. CPU / float32 conversion
        2. Stereo → mono (channel average)
        3. Resample 48 kHz → 24 kHz (Kaiser-windowed sinc interpolation)
        4. Peak normalization to -1 dBFS (consistent output level)
        5. Safety clip to [-1, 1]
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

        # 3. Resample 48 kHz → 24 kHz
        # The resampler's own anti-alias filter (rolloff=0.9475) provides
        # proper Nyquist filtering — no separate pre-filter needed.
        if source_rate != TARGET_SAMPLE_RATE:
            tensor = torchaudio.functional.resample(
                tensor,
                orig_freq=source_rate,
                new_freq=TARGET_SAMPLE_RATE,
                lowpass_filter_width=64,
                rolloff=0.9475,
            )

        # 4. Peak normalization to -1 dBFS
        # Ensures consistent output level for the downstream pipeline.
        peak = tensor.abs().max()
        if peak > 1e-6:
            target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS ≈ 0.891
            tensor = tensor * (target_peak / peak)

        # 5. Safety clip
        tensor = tensor.clamp(-1.0, 1.0)

        return tensor.squeeze().cpu().numpy().astype(np.float32)
