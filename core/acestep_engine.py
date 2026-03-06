"""ACE-Step 1.5 wrapper — text-to-music via MLX on Apple Silicon.

Model: ACE-Step 1.5 (DiT decoder + LM planner)
Device: MLX (Metal GPU on Apple Silicon)
Output: Mono float32 at 24 kHz (native 48 kHz stereo downmixed and resampled)

Target hardware: Apple Silicon M1 Max (24-Core GPU, 36 GB Unified RAM)
"""

import gc
import logging
import time

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 48000       # ACE-Step native output rate
TARGET_SAMPLE_RATE = 24000       # Pipeline standard rate (matches MusicEngine)


class AceStepEngine:
    """Generates ambient background music via ACE-Step 1.5 on MLX.

    Uses the full-quality ``acestep-v15-sft`` DiT config and the
    ``acestep-5Hz-lm-1.7B`` language model for Chain-of-Thought planning.
    Forces instrumental mode with no vocals.  Output is converted from
    48 kHz stereo to 24 kHz mono to honour the pipeline contract.

    Memory: explicitly calls gc.collect() and torch.mps.empty_cache()
    on unload to free unified RAM for subsequent TTS loading.
    """

    def __init__(self):
        self._dit = None
        self._llm = None
        self.initialized = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self):
        """Load ACE-Step DiT and LLM on the MLX backend."""
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        logger.info("[AceStepEngine] Loading ACE-Step 1.5 on MLX...")
        t0 = time.time()

        self._dit = AceStepHandler()
        self._llm = LLMHandler()

        self._dit.initialize_service(
            project_root="./ACE-Step-1.5",
            config_path="acestep-v15-sft",
            device="mlx",
        )
        self._llm.initialize(
            checkpoint_dir="./ACE-Step-1.5/checkpoints",
            lm_model_path="acestep-5Hz-lm-1.7B",
            backend="mlx",
            device="mlx",
        )

        self.initialized = True
        logger.info(
            "[AceStepEngine] ACE-Step 1.5 loaded in %.1fs", time.time() - t0
        )

    def unload_model(self):
        """Release ACE-Step models and aggressively free memory."""
        logger.info("[AceStepEngine] Unloading ACE-Step 1.5...")
        if self._dit is not None:
            try:
                self._dit.shutdown_service()
            except Exception as exc:
                logger.warning("[AceStepEngine] DiT shutdown error: %s", exc)

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
    ) -> np.ndarray:
        """Generate instrumental meditation music via ACE-Step 1.5.

        Args:
            prompt: Text description of desired music style (will be
                    enhanced internally with meditation keywords).
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_step, total_steps).

        Returns:
            Mono float32 numpy array at 24 000 Hz — same contract as
            ``MusicEngine.generate()``.
        """
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        if not self.initialized:
            self.load_model()

        # ── Enhance prompt for meditation ────────────────────────────────
        enhanced_prompt = self._enhance_prompt(prompt)
        logger.info(
            "[AceStepEngine] Generating %.0fs of music — prompt: %s",
            total_duration_sec, enhanced_prompt[:120],
        )

        if progress_cb is not None:
            progress_cb(0, 1)

        t0 = time.time()

        # ── Build generation parameters ──────────────────────────────────
        params = GenerationParams(
            caption=enhanced_prompt,
            lyrics="[Instrumental]",
            instrumental=True,
            duration=total_duration_sec,
            inference_steps=32,
            thinking=True,            # Full LM Chain‑of‑Thought planning
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")

        result = generate_music(
            self._dit, self._llm, params, config, save_dir=None,
        )
        tensor = result.audios["tensor"]  # 48 kHz stereo

        elapsed = time.time() - t0
        logger.info("[AceStepEngine] Generation done in %.1fs", elapsed)

        if progress_cb is not None:
            progress_cb(1, 1)

        # ── Post-processing: 48 kHz stereo → 24 kHz mono ────────────────
        return self._postprocess(tensor)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_prompt(user_prompt: str) -> str:
        """Augment the user's prompt with meditation-oriented keywords.

        ACE-Step benefits from explicit guidance away from transients and
        towards ambient textures.  This appends keywords that the
        implementation doc specifies without duplicating terms the user
        already provided.
        """
        ambient_keywords = [
            ("ambient", "slow ambient pads"),
            ("motifs", "minimal motifs"),
            ("percussion", "no percussion"),
            ("gentle", "gentle"),
            ("drone", "warm drone"),
            ("calm", "calm"),
            ("spacious", "spacious"),
        ]
        user_lower = user_prompt.lower()
        extras = [desc for key, desc in ambient_keywords if key not in user_lower]

        parts = [
            "Meditation, ambient, minimalist",
            user_prompt.strip(),
        ] + extras + [
            "no drums, slow tempo, high fidelity, lush reverb",
        ]

        return ", ".join(parts)

    @staticmethod
    def _postprocess(tensor: torch.Tensor) -> np.ndarray:
        """Convert ACE-Step output from 48 kHz stereo to 24 kHz mono float32.

        This ensures the output matches the MusicEngine contract that the
        downstream pipeline (mixer, FX, export) relies on.
        """
        # Ensure we are on CPU for numpy conversion
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Stereo → mono
        if tensor.ndim > 1 and tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)

        # 48 kHz → 24 kHz
        resampled = torchaudio.functional.resample(
            tensor,
            orig_freq=NATIVE_SAMPLE_RATE,
            new_freq=TARGET_SAMPLE_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475,
        )

        return resampled.squeeze().cpu().numpy().astype(np.float32)
