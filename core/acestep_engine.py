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
# guidance_scale: CFG strength. 7.0 (default) forces the DiT to chase the
# prompt so hard it amplifies any spectral roughness. 3.5–4.5 is the sweet spot
# for smooth ambient textures — still adheres to the prompt but lets the
# diffusion model settle into natural tonal continuity.
_GUIDANCE_SCALE = 3.5

# inference_steps: More steps = smoother diffusion trajectory. 60 steps
# gives noticeably cleaner textures vs 32 with only ~85% extra time.
_INFERENCE_STEPS = 60
_INFERENCE_STEPS_REPAINT = 60

# lm_temperature: Lower = more conservative LM planning. 0.7 avoids
# unexpected melodic "ideas" that cause rhythmic intrusions in ambient output.
_LM_TEMPERATURE = 0.7

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
            model_type: "sft" (high fidelity, 60 steps) or 
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
            # For long tracks (> 90s), use iterative repaint to avoid structural
            # collapse. 90s is the comfortable single-pass limit for DiT.
            if total_duration_sec > 90.0:
                return self._generate_infinite(
                    prompt, total_duration_sec, progress_cb, lyrics=lyrics,
                    bpm=bpm, keyscale=keyscale, reference_audio_path=ref_path
                )

            # ── Single-stage generation ───────────────────────────────────────
            enhanced_prompt, enhanced_lyrics = self._enhance_prompt(prompt)
            if lyrics:
                # If explicit lyrics provided (e.g. from pipeline), append them
                enhanced_lyrics = f"{enhanced_lyrics}, {lyrics}"

            logger.info(
                "[AceStepEngine] Generating %.0fs of music — caption: %s | lyrics: %s | bpm: %s | key: %s",
                total_duration_sec, enhanced_prompt[:60], enhanced_lyrics[:60], bpm, keyscale
            )

            if progress_cb is not None:
                progress_cb(0, 1)

            audio = self._generate_single(
                enhanced_prompt, total_duration_sec, lyrics=enhanced_lyrics,
                bpm=bpm, keyscale=keyscale, reference_audio_path=ref_path
            )

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
            enhanced_cap, enhanced_lyr = self._enhance_prompt(stage_prompt)
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
        """Iteratively generate long audio using Repaint mode.

        Algorithm:
        1. Generate initial 60s (Stage 1).
        2. Continue in 30s steps:
           - Use the last 30s of previous audio as context.
           - Generate next 30s using Repaint mode.
           - Stitch with a tiny (2s) crossfade for absolute safety.

        Repaint mode preserves the context audio and ensures the new segment
        starts with matching timbre and harmony.
        """
        import tempfile
        import os
        import soundfile as sf

        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Segment configuration
        # Segment 1: 60s (provides solid 30s context for Segment 2)
        # Segments after: 30s extension + 30s context = 60s total Repaint task
        SEG_LEN = 30.0
        CONTEXT_LEN = 30.0
        TOTAL_SEG_LEN = SEG_LEN + CONTEXT_LEN # 60s
        
        enhanced_prompt, enhanced_lyrics = self._enhance_prompt(prompt)
        if lyrics:
            enhanced_lyrics = f"{enhanced_lyrics}, {lyrics}"

        logger.info(
            "[AceStepEngine] Infinite Mode: Generating %.0fs in segments...",
            total_duration_sec
        )

        # ── Segment 1: The Anchor ───────────────
        current_len = min(TOTAL_SEG_LEN, total_duration_sec)
        logger.info("[AceStepEngine] Infinite Seg 1: base generation (%.0fs)", current_len)
        
        if progress_cb:
            progress_cb(0, int(total_duration_sec // SEG_LEN))

        full_audio = self._generate_single(
            enhanced_prompt, current_len, lyrics=enhanced_lyrics,
            bpm=bpm, keyscale=keyscale, reference_audio_path=reference_audio_path
        )
        
        # ── Iterative Extension ─────────────────
        while len(full_audio) / TARGET_SAMPLE_RATE < total_duration_sec - 1.0:
            remaining = total_duration_sec - (len(full_audio) / TARGET_SAMPLE_RATE)
            next_step = min(SEG_LEN, remaining)
            
            # We use the LAST 30 seconds of full_audio as context for the NEXT 30 seconds.
            # We create a 60s "skeleton" file: [context | silence]
            # Then tell ACE-Step to repaint the silence part.
            context_samples = int(CONTEXT_LEN * TARGET_SAMPLE_RATE)
            if len(full_audio) < context_samples:
                # Should not happen given Seg 1 is 60s, but for safety:
                context_audio = full_audio
                actual_context_sec = len(context_audio) / TARGET_SAMPLE_RATE
            else:
                context_audio = full_audio[-context_samples:]
                actual_context_sec = CONTEXT_LEN

            # Build the source file for Repaint
            silence_samples = int(next_step * TARGET_SAMPLE_RATE)
            src_audio_np = np.concatenate([context_audio, np.zeros(silence_samples, dtype=np.float32)])
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_wav = os.path.join(tmpdir, "context.wav")
                # ACE-Step expects 48kHz stereo for src_audio usually, but we are 
                # passing it back mono 24k. Let's see if its handler handles it.
                # Actually, handler.process_src_audio does:
                # audio = self._normalize_audio_to_stereo_48k(audio, sr)
                # So we can save at TARGET_SAMPLE_RATE mono and it will be upsampled.
                sf.write(tmp_wav, src_audio_np, TARGET_SAMPLE_RATE)
                
                logger.info(
                    "[AceStepEngine] Infinite Seg Extension: Repaint from %.1fs (adding %.1fs)", 
                    actual_context_sec, next_step
                )
                
                # We want to repaint starting from actual_context_sec to the end
                new_chunk = self._generate_single_repaint(
                    enhanced_prompt, 
                    src_audio_path=tmp_wav,
                    repaint_start=actual_context_sec,
                    repaint_end=actual_context_sec + next_step,
                    lyrics=enhanced_lyrics,
                    bpm=bpm,
                    keyscale=keyscale,
                    reference_audio_path=reference_audio_path
                )
                
                # new_chunk contains the WHOLE 60s (context + new part)
                # We only want the new part
                ext_start_idx = int(actual_context_sec * TARGET_SAMPLE_RATE)
                ext_audio = new_chunk[ext_start_idx:]
                
                # Stitch with a tiny crossfade (0.1s) to avoid any mathematical rounding clicks
                # even though Repaint is meant to be seamless.
                FADE_SEC = 0.1
                fade_samples = int(FADE_SEC * TARGET_SAMPLE_RATE)
                if len(ext_audio) > fade_samples:
                    # Apply crossfade at the seam
                    t = np.linspace(0.0, np.pi/2, fade_samples)
                    cos_fade = np.cos(t)**2
                    sin_fade = np.sin(t)**2
                    
                    overlap_area = full_audio[-fade_samples:] * cos_fade + ext_audio[:fade_samples] * sin_fade
                    full_audio = np.concatenate([full_audio[:-fade_samples], overlap_area, ext_audio[fade_samples:]])
                else:
                    full_audio = np.concatenate([full_audio, ext_audio])

            if progress_cb:
                completed = len(full_audio) / TARGET_SAMPLE_RATE
                progress_cb(int(completed // SEG_LEN), int(total_duration_sec // SEG_LEN))

        return full_audio

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

        # ── Build generation parameters ──────────────────────────────────
        # Key quality choices:
        # - guidance_scale=3.5: softer CFG avoids harsh, over-determined textures
        # - inference_steps=60: more diffusion steps → smoother frequency response
        # - use_adg=True: Adaptive Dual Guidance reduces spectral noise on base model
        # - enable_normalization=False: prevent ACE-Step from peak-normalising to
        #   -1 dBFS before our own loudness stage; the pipeline controls gain
        # - lm_temperature=0.7: conservative planning avoids unexpected melodic jolts
        # - bpm=50: force slow meditative tempo in the LM planner
        params = GenerationParams(
            caption=enhanced_prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            duration=duration_sec,
            reference_audio=reference_audio_path,
            bpm=bpm if bpm and bpm > 0 else None,
            keyscale=keyscale if keyscale and keyscale != "Auto" else "",
            inference_steps=self._get_inference_steps(is_repaint=False),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=_USE_ADG,
            thinking=True,
            lm_temperature=_LM_TEMPERATURE,
            use_cot_metas=True,
            use_cot_caption=True,
            # Disable ACE-Step's own peak normalization. The pipeline performs
            # loudness normalization via pyloudnorm at a later, controlled stage.
            enable_normalization=False,
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

        # ── Post-processing: denoise → stereo→mono → resample ────────────
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
            repainting_start=repaint_start,
            repainting_end=repaint_end,
            duration=repaint_end, # Total duration of the resulting file
            inference_steps=self._get_inference_steps(is_repaint=True),
            guidance_scale=_GUIDANCE_SCALE,
            use_adg=False, # ADG not supported for repaint
            thinking=False, # Skip LM for Repaint tasks
            lm_temperature=_LM_TEMPERATURE,
            # Disable ACE-Step's own peak normalization. The pipeline performs
            # loudness normalization via pyloudnorm at a later, controlled stage.
            enable_normalization=False,
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

    # Words in a user prompt that could push ACE-Step toward loud or
    # energetic output.  Each entry is (pattern_substring, replacement).
    # Replacements keep the semantic intent but redirect toward softness.
    _LOUD_KEYWORD_REPLACEMENTS: list[tuple[str, str]] = [
        ("epic", "ethereal"),
        ("powerful", "subtle"),
        ("intense", "gentle"),
        ("dramatic", "serene"),
        ("bold", "soft"),
        ("loud", "quiet"),
        ("energetic", "peaceful"),
        ("upbeat", "tranquil"),
        ("dynamic", "still"),
        ("pulsing", "floating"),
        ("driving", "drifting"),
        ("strong", "delicate"),
        ("full orchestra", "sparse strings"),
        ("orchestral", "minimal"),
        ("cinematic", "meditative"),
        ("triumphant", "peaceful"),
        ("exciting", "calming"),
    ]

    @classmethod
    def _sanitize_prompt(cls, user_prompt: str) -> str:
        """Replace loud/energetic keywords in the user prompt with soft alternatives.

        This prevents the LM planner from interpreting the user's creative
        intent as a license to generate loud transients or rhythmic content.
        Replacements are case-insensitive and preserve surrounding text.
        """
        import re
        result = user_prompt
        for loud_word, replacement in cls._LOUD_KEYWORD_REPLACEMENTS:
            result = re.sub(
                re.escape(loud_word), replacement, result, flags=re.IGNORECASE
            )
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
            # ACE-Step expects float32 mono or stereo
            sf.write(path, audio_data, sr)
        finally:
            os.close(fd)
        logger.info("[AceStepEngine] Prepared temp reference audio: %s", path)
        return path

    @classmethod
    def _extract_lyrics_tags(cls, prompt: str) -> str:
        """Scan prompt for keywords and wrap them in structural [tags].
        
        Returns a comma-separated string of tags starting with [Instrumental].
        """
        tags = ["[Instrumental]", "[Drone]", "[Static Pad]", "[No Transients]", "[Beatless]"]
        
        # Keywords that map well to structural tags in ACE-Step
        structural_keywords = [
            "dreamy", "ethereal", "low energy", "high energy", "fade out",
            "ambient", "cinematic", "serene", "soft", "warm", "evolving"
        ]
        
        prompt_lower = prompt.lower()
        for kw in structural_keywords:
            if kw in prompt_lower:
                tags.append(f"[{kw}]")
                
        return ", ".join(tags)

    @classmethod
    def _sanitize_prompt(cls, prompt: str) -> str:
        """Strip metadata commands and harsh sound references from user prompt."""
        import re
        s = prompt.lower()
        
        # Strip explicit BPM/Key commands that should be in metadata fields
        s = re.sub(r'\d+\s*(?:bpm|beats per minute)', '', s)
        s = re.sub(r'[a-g][#b♭♯]?\s*(?:major|minor|maj|min)', '', s)
        
        # Strip harsh sound requests
        harsh = ["sawtooth", "square wave", "distorted", "aggressive", "sharp", "bright synth"]
        for h in harsh:
            s = s.replace(h, "")
            
        return s.strip()

    @classmethod
    def _enhance_prompt(cls, user_prompt: str) -> tuple[str, str]:
        """Augment prompt into separate (caption, lyrics) fields.

        Caption: Strict minimalist wrapper forcing static-drone generation.
        Lyrics: Structural tags like [Instrumental].
        """
        sanitized = cls._sanitize_prompt(user_prompt)
        lyrics = cls._extract_lyrics_tags(user_prompt)

        caption = (
            f"Pure ambient drone. Single continuous texture, static harmony, "
            f"no chord changes, no melody, no percussion. "
            f"Extremely soft pianissimo background. {sanitized.strip()}"
        )
        return caption, lyrics

    @staticmethod
    def _postprocess(tensor: torch.Tensor, source_rate: int = NATIVE_SAMPLE_RATE) -> np.ndarray:
        """Convert ACE-Step output to clean 24 kHz mono float32.

        Processing chain:
        1. Move to CPU, cast to float32
        2. Stereo → mono (equal-power average)
        3. Soft-clip with tanh to round off hard transients / artifacts
           without hard distortion — subtler than np.clip()
        4. Spectral smoothing: apply a mild Hann-windowed moving average
           across frames to even out diffusion-noise hotspots in the
           high-frequency range (>8 kHz at 48 kHz SR)
        5. Resample 48 kHz → 24 kHz with a high-quality Kaiser window
        6. Final safety clip to [-1, 1]
        """
        # ── 1. CPU / float32 ─────────────────────────────────────────────
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        tensor = tensor.float()

        # ── 2. Stereo → mono (equal-power average) ───────────────────────
        if tensor.ndim > 1 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        # ── 3. Soft-clip with tanh ────────────────────────────────────────
        # Maps any value outside [-1, 1] through a smooth saturation curve
        # rather than hard-clipping. This softens rare transient spikes
        # (diffusion artifacts) without creating the digital "crackle" that
        # np.clip() introduces on true over-peaks.
        # Scale factor 0.95 gives ~0.5 dB headroom before the tanh knee.
        tensor = torch.tanh(tensor * 0.95) / 0.95

        # ── 4. High-frequency smoothing (stochastic noise reduction) ─────
        # ACE-Step's VAE decoder occasionally leaves small spectral-noise
        # artefacts above 14 kHz at 48 kHz SR.  A short triangular moving
        # average in the TIME domain attenuates these without touching the
        # all-important fundamental and harmonic content.
        #
        # Kernel: triangular window of 3 samples (≈0.06 ms at 48 kHz).
        # This is identical to two successive box-filters of length 2 and is
        # nearly indistinguishable from a brick-wall cut at ~11 kHz while
        # being far more efficient (no STFT needed).
        kernel = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float32)
        # Apply as a 1D depthwise convolution: pad=1 preserves length
        tensor_4d = tensor.unsqueeze(0)                     # (1, 1, T)
        kernel_4d = kernel.view(1, 1, 3)
        tensor_4d = torch.nn.functional.conv1d(
            tensor_4d, kernel_4d, padding=1
        )
        tensor = tensor_4d.squeeze(0)                       # (1, T)

        # ── 5. Resample 48 kHz → 24 kHz ────────────────────────────────
        if source_rate != TARGET_SAMPLE_RATE:
            tensor = torchaudio.functional.resample(
                tensor,
                orig_freq=source_rate,
                new_freq=TARGET_SAMPLE_RATE,
                lowpass_filter_width=64,
                rolloff=0.9475,
            )

        # ── 6. Safety clip ───────────────────────────────────────────────
        tensor = tensor.clamp(-1.0, 1.0)

        return tensor.squeeze().cpu().numpy().astype(np.float32)
