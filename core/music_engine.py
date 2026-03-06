"""MusicGen wrapper — stereo-medium with sliding window continuation + story mode.

Model: facebook/musicgen-stereo-medium (1.5B params), fallback to musicgen-small
Device: MPS (Apple Silicon GPU) with automatic CPU fallback
Strategy: 30s initial segment, then sliding window with 5s context overlap.
          Optional story mode: supply prompt_stages to evolve the text prompt
          across the timeline, matching meditation phase transitions.
Output: Mono float32 at 24 kHz (stereo generated internally, downmixed at boundary)

Target hardware: Apple Silicon with 36 GB unified memory
"""

import gc
import logging
import math
import time

import numpy as np
import torch
import torchaudio


logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 32000       # MusicGen native output rate
TARGET_SAMPLE_RATE = 24000       # Kokoro TTS / pipeline standard rate
SEGMENT_DURATION = 30            # Seconds per MusicGen call (hard limit)
CONTEXT_DURATION = 10            # Seconds of audio context for continuation
CROSSFADE_DURATION = 2.0         # Seconds of crossfade at each segment seam

MODEL_ID = "facebook/musicgen-stereo-medium"
FALLBACK_MODEL_ID = "facebook/musicgen-small"


class MusicEngine:
    """Generates ambient background music via MusicGen sliding window continuation.

    Uses musicgen-stereo-medium (1.5B params) on MPS (Apple Silicon GPU) for
    fast inference, with automatic CPU fallback if MPS is unstable. Generates
    a 30s initial segment, then extends with continuation calls using 5s of
    audio context per step. Stereo output is downmixed to mono at the boundary
    to preserve the pipeline's mono contract.

    Performance: MPS + CFG 4.0 for stable, prompt-adherent ambient output.
    """

    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        """Choose the fastest available device.

        Prefers MPS (Apple Silicon GPU) for ~4-5x faster inference.
        Falls back to CPU if MPS is not available.
        """
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _patch_mps_elu(self):
        """Monkeypatch PyTorch ELU to ensure contiguous inputs on MPS.
        
        Fixes a known bug in PyTorch where `F.elu` produces garbled static
        on Apple Silicon if the input tensor is non-contiguous. EnCodec
        relies heavily on ELU, causing random static bursts without this.
        """
        import torch.nn.functional as F
        
        if not hasattr(F, "_orig_elu"):
            F._orig_elu = F.elu
            
            def safe_elu(input, alpha=1.0, inplace=False):
                if input.device.type == "mps" and not input.is_contiguous():
                    input = input.contiguous()
                return F._orig_elu(input, alpha, inplace)
            
            F.elu = safe_elu

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self):
        """Load MusicGen on the best available device (MPS → CPU fallback)."""
        import warnings

        from audiocraft.models import MusicGen

        self.device = self._pick_device()
        if self.device == "mps":
            self._patch_mps_elu()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS autocast.*")

            candidates = [MODEL_ID, FALLBACK_MODEL_ID]
            for name in candidates:
                # Try preferred device first, fall back to CPU
                devices_to_try = [self.device] if self.device == "cpu" else [self.device, "cpu"]
                for dev in devices_to_try:
                    try:
                        logger.info("[MusicEngine] Loading %s on %s...", name, dev)
                        self.model = MusicGen.get_pretrained(name, device=dev)
                        self.model_name = name
                        self.device = dev
                        logger.info("[MusicEngine] Loaded %s on %s", name, dev)
                        return
                    except Exception as e:
                        logger.warning("[MusicEngine] %s on %s failed: %s", name, dev, e)
                        print(f"[MusicEngine] {name} on {dev} failed: {e}")

            raise RuntimeError("No MusicGen model could be loaded.")

    def unload_model(self):
        """Release model and free memory."""
        del self.model
        self.model = None
        self.model_name = None
        self.device = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        total_duration_sec: float,
        progress_cb=None,
        prompt_stages: list[tuple[str, float]] | None = None,
        melody_audio: np.ndarray | None = None,
        melody_sample_rate: int | None = None,
    ) -> np.ndarray:
        """Generate background music using sliding window continuation.

        Args:
            prompt: Text description of desired music style (used when
                    prompt_stages is None).
            total_duration_sec: Target duration in seconds.
            progress_cb: Called with (current_segment, total_segments).
            prompt_stages: Optional list of (prompt, duration_sec) tuples
                defining story mode. Each segment's text prompt is chosen
                from whichever stage contains that segment's elapsed time.
                Audio continuity is preserved via the sliding window
                regardless of prompt changes — the context audio bridges
                thematic transitions smoothly.
                Example:
                    [
                        ("calm breathing pads, soft sine waves", 90.0),
                        ("deep sleep ambient drones, very slow", 120.0),
                        ("peaceful gentle morning light, birds", 90.0),
                    ]
                Ignored if None (falls back to single ``prompt``).
            melody_audio: Optional reference audio for melody conditioning.
                Float32 mono numpy array. When provided, the first segment
                uses chroma-based melody conditioning to guide MusicGen's
                melodic/harmonic structure toward the reference. Subsequent
                continuation segments use standard audio context.
            melody_sample_rate: Sample rate of melody_audio (required when
                melody_audio is provided).

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Music model not loaded. Call load_model() first.")

        num_segments = self._num_segments(total_duration_sec)
        story_mode = prompt_stages is not None
        logger.info(
            "[MusicEngine] Generating %.0fs of music (%d segments) on %s%s",
            total_duration_sec, num_segments, self.device,
            f" — story mode ({len(prompt_stages)} stages)" if story_mode else "",
        )

        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=250,           # Keeps sampling within top-250 tokens
            top_p=0.0,           # Disabled — top_k handles truncation
            temperature=0.87,    # Stabilises token sampling for consonant pads (0.85-0.90 sweet spot)
            cfg_coef=4.0,        # Strong CFG — penalises tokens that diverge from text prompt
        )

        if progress_cb is not None:
            progress_cb(0, num_segments)

        t0 = time.time()

        # ── Segment 1: generation with optional melody conditioning ─────
        # In story mode the first segment uses the prompt for t=0.
        # When melody_audio is provided, use chroma-based conditioning so
        # MusicGen follows the reference's melodic/harmonic contour.
        MAX_RETRIES = 3
        seg0_prompt = (
            self._stage_prompt_for_time(0.0, prompt_stages) if story_mode else prompt
        )

        use_melody = melody_audio is not None and melody_sample_rate is not None
        melody_tensor = None
        if use_melody:
            melody_tensor = torch.from_numpy(melody_audio.astype(np.float32))
            if melody_tensor.ndim == 1:
                melody_tensor = melody_tensor.unsqueeze(0)
            # Trim or pad melody to SEGMENT_DURATION at the melody's native SR
            melody_samples = int(SEGMENT_DURATION * melody_sample_rate)
            if melody_tensor.shape[-1] > melody_samples:
                melody_tensor = melody_tensor[..., :melody_samples]
            if self.device != "cpu":
                melody_tensor = melody_tensor.to(self.device)
            logger.info(
                "[MusicEngine] Melody conditioning active — %.1fs reference at %d Hz",
                melody_tensor.shape[-1] / melody_sample_rate, melody_sample_rate,
            )

        for attempt in range(MAX_RETRIES):
            # Explicit memory cleanup against AudioCraft state leak on retry
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            if use_melody:
                wav = self.model.generate_with_chroma(
                    descriptions=[seg0_prompt],
                    melody_wavs=melody_tensor.unsqueeze(0),
                    melody_sample_rate=melody_sample_rate,
                    progress=False,
                )
            else:
                wav = self.model.generate([seg0_prompt], progress=False)
            current = wav[0].cpu()
            if self._check_spectral_flux(current):
                break
            logger.info("[MusicEngine] Segment 1 retry %d/%d", attempt + 1, MAX_RETRIES)
        segments = [current]

        elapsed = time.time() - t0
        logger.info("[MusicEngine] Segment 1/%d done in %.1fs", num_segments, elapsed)

        if progress_cb is not None:
            progress_cb(1, num_segments)

        # Net new audio per continuation (context is stripped during stitching)
        net_new_per_seg = SEGMENT_DURATION - CONTEXT_DURATION

        # ── Subsequent segments: continuation with audio context ────────
        for i in range(1, num_segments):
            seg_t0 = time.time()
            context_samples = int(CONTEXT_DURATION * NATIVE_SAMPLE_RATE)
            context = segments[-1][..., -context_samples:]

            # Story mode: pick the prompt for the elapsed time at this segment
            elapsed_sec = i * net_new_per_seg
            if story_mode:
                seg_prompt = self._stage_prompt_for_time(elapsed_sec, prompt_stages)
                prev_prompt = self._stage_prompt_for_time(
                    (i - 1) * net_new_per_seg, prompt_stages
                )
                if seg_prompt != prev_prompt:
                    logger.info(
                        "[MusicEngine] Story mode: stage transition at t=%.0fs → %s",
                        elapsed_sec, seg_prompt[:60],
                    )
            else:
                seg_prompt = prompt

            # Move context to model device for continuation
            if self.device != "cpu":
                context = context.to(self.device)

            for attempt in range(MAX_RETRIES):
                # Explicit memory cleanup against AudioCraft state leak on retry
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()

                next_wav = self.model.generate_continuation(
                    prompt=context,
                    prompt_sample_rate=NATIVE_SAMPLE_RATE,
                    descriptions=[seg_prompt],
                    progress=False,
                )
                candidate = next_wav[0].cpu()
                if self._check_spectral_flux(candidate):
                    break
                logger.info("[MusicEngine] Segment %d retry %d/%d", i + 1, attempt + 1, MAX_RETRIES)
            segments.append(candidate)
            
            # Explicit memory cleanup during large sliding string of generations
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            seg_elapsed = time.time() - seg_t0
            logger.info(
                "[MusicEngine] Segment %d/%d done in %.1fs",
                i + 1, num_segments, seg_elapsed,
            )

            if progress_cb is not None:
                progress_cb(i + 1, num_segments)

        total_elapsed = time.time() - t0
        logger.info("[MusicEngine] All %d segments done in %.1fs", num_segments, total_elapsed)

        # ── Stitch segments and trim to exact duration ──────────────────
        full_audio = self._stitch(segments)

        target_samples_native = int(total_duration_sec * NATIVE_SAMPLE_RATE)
        if full_audio.shape[-1] > target_samples_native:
            full_audio = full_audio[..., :target_samples_native]

        # ── Resample 32 kHz → 24 kHz ───────────────────────────────────
        resampled = torchaudio.functional.resample(
            full_audio, NATIVE_SAMPLE_RATE, TARGET_SAMPLE_RATE,
            lowpass_filter_width=64,
            rolloff=0.9475,
        )
        result = resampled.numpy().astype(np.float32)
        # Return unclipped audio: pipeline.py will safely LUFS-normalize it.
        # Hard clipping here causes massive distortion because MusicGen
        # frequently generates unnormalized peaks > 1.0.
        return result

    # ── Private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _stage_prompt_for_time(
        elapsed_sec: float,
        stages: list[tuple[str, float]],
    ) -> str:
        """Return the stage prompt that is active at *elapsed_sec*.

        Iterates through (prompt, duration) pairs and returns the prompt
        whose cumulative duration range contains elapsed_sec.  Falls back
        to the last stage's prompt if elapsed_sec exceeds total stage time.
        """
        cumulative = 0.0
        for stage_prompt, duration in stages:
            cumulative += duration
            if elapsed_sec < cumulative:
                return stage_prompt
        return stages[-1][0]

    @staticmethod
    def _check_spectral_flux(
        audio: torch.Tensor,
        sample_rate: int = NATIVE_SAMPLE_RATE,
        hop_size: int = 512,
        flux_threshold_multiplier: float = 4.5,
    ) -> bool:
        """Return True if audio passes the hallucination check (no sudden transients).

        Computes per-frame spectral flux (L1 norm of the difference between
        consecutive magnitude spectra). If any single frame's flux exceeds
        4.5x the median flux of the segment, it indicates a sudden percussive
        event — the segment is flagged for rejection and regeneration.

        Args:
            audio:                   1D or (1, N) float tensor at NATIVE_SAMPLE_RATE.
            sample_rate:             Sample rate (used for frame sizing only).
            hop_size:                STFT hop in samples.
            flux_threshold_multiplier: How many times the median flux triggers rejection.

        Returns:
            True  = segment is clean (accept).
            False = transient spike detected (reject and regenerate).
        """
        waveform = audio.squeeze()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        # Compute STFT magnitude frames
        n_fft = 1024
        window = torch.hann_window(n_fft, device=waveform.device)
        stft = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_size,
            window=window, return_complex=True
        )
        magnitude = stft.abs()  # (freq_bins, time_frames)

        if magnitude.shape[-1] < 2:
            return True  # Too short to evaluate — accept

        # Spectral flux: L1 norm of frame-to-frame magnitude difference
        flux = (magnitude[:, 1:] - magnitude[:, :-1]).abs().sum(dim=0)  # (time_frames-1,)

        median_flux = flux.median()
        if median_flux < 1e-8:
            return True  # Silent segment — accept

        peak_flux = flux.max()
        ratio = float(peak_flux / median_flux)

        is_clean = ratio < flux_threshold_multiplier
        if not is_clean:
            logger.warning(
                "[MusicEngine] Hallucination detected: peak/median flux ratio=%.1f (threshold=%.1f). "
                "Regenerating segment.", ratio, flux_threshold_multiplier
            )
        return is_clean

    def _num_segments(self, duration: float) -> int:
        """Calculate number of generation passes needed for the target duration."""
        if duration <= SEGMENT_DURATION:
            return 1
        net_new_per_segment = SEGMENT_DURATION - CONTEXT_DURATION
        return 1 + math.ceil((duration - SEGMENT_DURATION) / net_new_per_segment)

    def _to_mono(tensor: torch.Tensor) -> torch.Tensor:
        # Not used centrally anymore to support native stereo passing, but kept for legacy API compat
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.shape[0] > 1:
            return tensor.mean(dim=0, keepdim=True)
        return tensor

    def _stitch(self, segments: list[torch.Tensor]) -> torch.Tensor:
        """Join continuation segments, stripping the duplicated context region.

        Each continuation output starts with ~CONTEXT_DURATION seconds that echo
        the audio prompt. We strip that overlap and crossfade at the seam.
        """
        if len(segments) == 1:
            return segments[0]

        ctx_samples = int(CONTEXT_DURATION * NATIVE_SAMPLE_RATE)
        fade_samples = int(CROSSFADE_DURATION * NATIVE_SAMPLE_RATE)
        result = segments[0]

        for seg in segments[1:]:
            new = seg[..., ctx_samples:]  # strip duplicate context
            if new.shape[-1] == 0:
                continue

            overlap = min(fade_samples, result.shape[-1], new.shape[-1])
            # Equal-power cosine crossfade prevents volume dips at the seam
            t = torch.linspace(0, math.pi / 2, overlap, device=result.device)
            fade_out = torch.cos(t) ** 2
            fade_in = torch.cos(math.pi / 2 - t) ** 2

            blended = result[..., -overlap:] * fade_out + new[..., :overlap] * fade_in
            result = torch.cat(
                [result[..., :-overlap], blended, new[..., overlap:]], dim=-1
            )

        return result
