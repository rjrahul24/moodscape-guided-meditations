"""MusicGen wrapper — stereo-medium with sliding window continuation + story mode.

Model: facebook/musicgen-stereo-medium (1.5B params), fallback to musicgen-small
Device: CPU only (MPS disabled — AudioCraft EnCodec ELU corruption on Apple Silicon)
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

    Uses musicgen-stereo-medium (1.5B params) on CPU. MPS is disabled because
    AudioCraft's EnCodec decoder uses F.elu extensively, and Apple Silicon's MPS
    backend has a known tensor corruption bug in ELU that produces audible static
    ("broken radio" artifacts) even with contiguity patches. CPU inference is
    slower (~0.5x realtime) but produces artifact-free audio every time.

    Generates a 30s initial segment, then extends with continuation calls using
    5s of audio context per step. Stereo output is downmixed to mono at the
    boundary to preserve the pipeline's mono contract.

    Performance: CPU + CFG 4.5 for stable, prompt-adherent ambient output.
    """

    def __init__(self, use_mbd: bool = False):
        self.model = None
        self.model_name = None
        self.device = None
        self._mbd = None
        # MBD is only supported on CUDA; MPS support is untested by Meta and
        # conflicts with AUDIOCRAFT_DISABLE_MPS_AUTOCAST=1 already set in app.py.
        self.use_mbd = use_mbd

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_device() -> str:
        """Always returns CPU for MusicGen inference.

        AudioCraft's EnCodec decoder uses F.elu extensively, and Apple
        Silicon's MPS backend has a known tensor corruption bug in ELU that
        produces audible static even with contiguity patches applied. CPU
        inference is slower (~0.5x realtime for stereo-medium) but produces
        mathematically identical, artifact-free audio every time.
        """
        return "cpu"

    @staticmethod
    def _patch_multinomial():
        """Monkeypatch torch.multinomial to sanitize NaN/inf probability tensors.

        MusicGen's CFG sampling can produce NaN/inf logits that survive softmax
        and cause 'probability tensor contains inf/nan or element < 0' errors.
        This patch clamps/replaces bad values before multinomial sampling so
        generation degrades gracefully rather than crashing.
        """
        if hasattr(torch, "_orig_multinomial"):
            return  # Already patched

        torch._orig_multinomial = torch.multinomial

        def _safe_multinomial(input, num_samples, replacement=False, **kwargs):
            if not input.is_floating_point():
                return torch._orig_multinomial(input, num_samples, replacement=replacement, **kwargs)
            # Replace NaN/inf, then clamp negatives to 0
            input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=0.0)
            input = input.clamp(min=0.0)
            # If all weights are zero, use uniform distribution
            row_sums = input.sum(dim=-1, keepdim=True)
            zero_mask = (row_sums == 0.0)
            if zero_mask.any():
                input = input + zero_mask.float()
            return torch._orig_multinomial(input, num_samples, replacement=replacement, **kwargs)

        torch.multinomial = _safe_multinomial

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self):
        """Load MusicGen on CPU."""
        import warnings

        from audiocraft.models import MusicGen

        self.device = "cpu"
        self._patch_multinomial()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*MPS autocast.*")

            for name in [MODEL_ID, FALLBACK_MODEL_ID]:
                try:
                    logger.info("[MusicEngine] Loading %s on cpu...", name)
                    self.model = MusicGen.get_pretrained(name, device="cpu")
                    self.model_name = name
                    logger.info("[MusicEngine] Loaded %s on cpu", name)
                    return
                except Exception as e:
                    logger.warning("[MusicEngine] %s failed: %s", name, e)
                    print(f"[MusicEngine] {name} failed: {e}")

            raise RuntimeError("No MusicGen model could be loaded.")

        # Load Multi-Band Diffusion decoder if requested and on CUDA.
        # MBD uses a diffusion model (~12 steps) that requires CUDA; on MPS the
        # internal autocast conflicts with AUDIOCRAFT_DISABLE_MPS_AUTOCAST=1.
        self._mbd = None
        if self.use_mbd:
            if self.device == "cuda":
                try:
                    from audiocraft.models import MultiBandDiffusion
                    logger.info("[MusicEngine] Loading MultiBandDiffusion decoder...")
                    self._mbd = MultiBandDiffusion.get_mbd_musicgen()
                    logger.info("[MusicEngine] MultiBandDiffusion loaded.")
                except Exception as e:
                    logger.warning("[MusicEngine] MBD load failed, falling back to EnCodec: %s", e)
                    self._mbd = None
            else:
                logger.warning(
                    "[MusicEngine] use_mbd=True requested but device=%s — "
                    "MBD requires CUDA. Falling back to EnCodec decoder.",
                    self.device,
                )

    def unload_model(self):
        """Release model and free memory."""
        del self.model
        self.model = None
        self.model_name = None
        self.device = None
        if self._mbd is not None:
            del self._mbd
            self._mbd = None
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
        temperature: float = 0.87,   # Ambient sweet spot: 0.85–0.90 per docs
        top_k: int = 250,            # Wider vocabulary needed for evolving pads
        top_p: float = 0.0,
        cfg_coef: float = 4.5,
        extend_stride: float = 12.0,
        seed: int | None = None,
        prompt_schedule_interval: float | None = 60.0,
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
            temperature: Sampling temperature. 0.87 is the documented sweet spot
                for ambient meditation pads (0.85–0.90 range), balancing stability
                with enough variation for slow-evolving textures.
            top_k: Token sampling breadth. 250 keeps the full ambient vocabulary
                available. Low values (< 100) prevent slow-evolving pads and cause
                repetitive loops or random noise on long generations.
            top_p: Top-p (nucleus) sampling (keep 0.0 for k-only).
            cfg_coef: Classifier-free guidance strength.
            extend_stride: Seconds of new audio per segment (sets context window).

        Returns:
            Mono float32 numpy array at 24000 Hz.
        """
        if self.model is None:
            raise RuntimeError("Music model not loaded. Call load_model() first.")

        story_mode = prompt_stages is not None
        # In story mode, ensure segment boundaries (and thus prompt changes)
        # occur at least every prompt_schedule_interval seconds.
        if story_mode and prompt_schedule_interval is not None:
            extend_stride = min(extend_stride, prompt_schedule_interval)
        num_segments = self._num_segments(total_duration_sec, extend_stride)
        logger.info(
            "[MusicEngine] Generating %.0fs of music (%d segments) on %s%s",
            total_duration_sec, num_segments, self.device,
            f" — story mode ({len(prompt_stages)} stages)" if story_mode else "",
        )

        self.model.set_generation_params(
            duration=SEGMENT_DURATION,
            use_sampling=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
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

        from core.qa_monitor import compute_composite_score

        seg0_candidates: list[tuple[torch.Tensor, float]] = []
        for attempt in range(MAX_RETRIES):
            # Explicit memory cleanup against AudioCraft state leak on retry
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            # Deterministic generation: seed segment 0 directly, retries use seed+attempt
            if seed is not None:
                torch.manual_seed(seed + attempt)

            if use_melody:
                wav = self.model.generate_with_chroma(
                    descriptions=[seg0_prompt],
                    melody_wavs=melody_tensor.unsqueeze(0),
                    melody_sample_rate=melody_sample_rate,
                    progress=False,
                )
            elif self._mbd is not None:
                # MBD decoding: request tokens and decode via diffusion instead of EnCodec.
                # Only supported for generate() — generate_continuation() does not expose tokens.
                wav, tokens = self.model.generate([seg0_prompt], return_tokens=True, progress=False)
                wav = self._mbd.tokens_to_wav(tokens)
            else:
                wav = self.model.generate([seg0_prompt], progress=False)
            current = wav[0].cpu()
            if self._check_spectral_flux(current):
                arr = current.squeeze().numpy()
                score = compute_composite_score(arr, NATIVE_SAMPLE_RATE)
                seg0_candidates.append((current, score))
                if score > 0.8:
                    break
            else:
                seg0_candidates.append((current, 0.0))
                logger.info("[MusicEngine] Segment 1 retry %d/%d", attempt + 1, MAX_RETRIES)

        if seg0_candidates:
            current = max(seg0_candidates, key=lambda c: c[1])[0]
            if len(seg0_candidates) > 1:
                scores = [f"{s:.3f}" for _, s in seg0_candidates]
                logger.info("[MusicEngine] Seg 0 A/B selection — scores: %s", scores)
        segments = [current]

        elapsed = time.time() - t0
        logger.info("[MusicEngine] Segment 1/%d done in %.1fs", num_segments, elapsed)

        if progress_cb is not None:
            progress_cb(1, num_segments)

        # Net new audio per continuation (context is stripped during stitching)
        net_new_per_seg = extend_stride
        context_duration = SEGMENT_DURATION - extend_stride

        # ── Subsequent segments: continuation with audio context ────────
        for i in range(1, num_segments):
            seg_t0 = time.time()
            context_samples = int(context_duration * NATIVE_SAMPLE_RATE)
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

            cont_candidates: list[tuple[torch.Tensor, float]] = []
            for attempt in range(MAX_RETRIES):
                # Explicit memory cleanup against AudioCraft state leak on retry
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()

                # Each continuation segment gets a deterministic seed derived from
                # the base seed + segment index, so the full track is reproducible.
                if seed is not None:
                    torch.manual_seed(seed + i * MAX_RETRIES + attempt)

                next_wav = self.model.generate_continuation(
                    prompt=context,
                    prompt_sample_rate=NATIVE_SAMPLE_RATE,
                    descriptions=[seg_prompt],
                    progress=False,
                )
                candidate = next_wav[0].cpu()
                if self._check_spectral_flux(candidate):
                    arr = candidate.squeeze().numpy()
                    score = compute_composite_score(arr, NATIVE_SAMPLE_RATE)
                    cont_candidates.append((candidate, score))
                    if score > 0.8:
                        break
                else:
                    cont_candidates.append((candidate, 0.0))
                    logger.info("[MusicEngine] Segment %d retry %d/%d", i + 1, attempt + 1, MAX_RETRIES)

            if cont_candidates:
                candidate = max(cont_candidates, key=lambda c: c[1])[0]
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
        full_audio, seam_positions = self._stitch(segments, extend_stride=extend_stride)
        if seam_positions:
            full_audio = self._apply_micro_crossfades(full_audio, seam_positions)

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
        """Return True if audio passes quality checks (non-silent, no sudden transients).

        Computes per-frame spectral flux (L1 norm of the difference between
        consecutive magnitude spectra).  Two failure modes are detected:

        1. **Silent segment** — median flux < 1e-8 means the output is pure
           silence or an inaudible hum.  These are *rejected* (return False) so
           the retry loop regenerates the segment rather than feeding silence as
           the context for the next continuation window (which would cascade
           blank audio through the remainder of the track).

        2. **Transient spike** — if any single frame's flux exceeds
           ``flux_threshold_multiplier`` × the median flux, a sudden percussive
           event was detected and the segment is also rejected.

        Args:
            audio:                   1D or (1, N) float tensor at NATIVE_SAMPLE_RATE.
            sample_rate:             Sample rate (used for frame sizing only).
            hop_size:                STFT hop in samples.
            flux_threshold_multiplier: How many times the median flux triggers rejection.

        Returns:
            True  = segment is clean (accept).
            False = silent or transient spike detected (reject and regenerate).
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
            # Reject silent segments — feeding silence as continuation context
            # cascades blank audio through every subsequent window.
            logger.warning("[MusicEngine] Silent segment detected. Regenerating.")
            return False

        peak_flux = flux.max()
        ratio = float(peak_flux / median_flux)

        is_clean = ratio < flux_threshold_multiplier
        if not is_clean:
            logger.warning(
                "[MusicEngine] Hallucination detected: peak/median flux ratio=%.1f (threshold=%.1f). "
                "Regenerating segment.", ratio, flux_threshold_multiplier
            )
        return is_clean

    def _num_segments(self, duration: float, extend_stride: float = 20.0) -> int:
        """Calculate number of generation passes needed for the target duration."""
        if duration <= SEGMENT_DURATION:
            return 1
        return 1 + math.ceil((duration - SEGMENT_DURATION) / extend_stride)

    def _to_mono(tensor: torch.Tensor) -> torch.Tensor:
        # Not used centrally anymore to support native stereo passing, but kept for legacy API compat
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.shape[0] > 1:
            return tensor.mean(dim=0, keepdim=True)
        return tensor

    def _stitch(
        self, segments: list[torch.Tensor], extend_stride: float = 20.0,
    ) -> tuple[torch.Tensor, list[int]]:
        """Join continuation segments, stripping the duplicated context region.

        Each continuation output starts with a context region that echoes
        the audio prompt. We strip that overlap and crossfade at the seam.

        Returns:
            (stitched_audio, seam_positions) where seam_positions lists the
            sample index of each crossfade centre (for micro-crossfade pass).
        """
        if len(segments) == 1:
            return segments[0], []

        context_duration = SEGMENT_DURATION - extend_stride
        ctx_samples = int(context_duration * NATIVE_SAMPLE_RATE)
        fade_samples = int(CROSSFADE_DURATION * NATIVE_SAMPLE_RATE)
        result = segments[0]
        seam_positions: list[int] = []

        for seg in segments[1:]:
            new = seg[..., ctx_samples:]  # strip duplicate context
            if new.shape[-1] == 0:
                continue

            overlap = min(fade_samples, result.shape[-1], new.shape[-1])
            # Equal-power cosine crossfade prevents volume dips at the seam
            t = torch.linspace(0, math.pi / 2, overlap, device=result.device)
            fade_out = torch.cos(t) ** 2
            fade_in = torch.cos(math.pi / 2 - t) ** 2

            # Record seam centre position (midpoint of the blended region)
            seam_centre = result.shape[-1] - overlap + overlap // 2
            seam_positions.append(seam_centre)

            blended = result[..., -overlap:] * fade_out + new[..., :overlap] * fade_in
            result = torch.cat(
                [result[..., :-overlap], blended, new[..., overlap:]], dim=-1
            )

        return result, seam_positions

    @staticmethod
    def _apply_micro_crossfades(
        audio: torch.Tensor,
        seam_positions: list[int],
        window: int = 64,
    ) -> torch.Tensor:
        """Apply sub-100ms linear fade at zero-crossings near each seam.

        After the macro cosine crossfade, residual high-frequency clicks can
        remain at seam boundaries.  This pass finds the nearest zero-crossing
        within ±window samples of each seam centre and applies a tiny
        triangular (linear fade-out × fade-in) window to smooth it.

        Args:
            audio:          1-D or (1, N) tensor at NATIVE_SAMPLE_RATE.
            seam_positions: Sample indices of crossfade centres from _stitch.
            window:         Half-width in samples (~2 ms at 32 kHz).
        """
        waveform = audio.squeeze()
        mono = waveform.mean(dim=0) if waveform.ndim > 1 else waveform

        for pos in seam_positions:
            lo = max(0, pos - window)
            hi = min(len(mono) - 1, pos + window)
            if hi - lo < 4:
                continue

            region = mono[lo:hi]
            # Find zero-crossings (sign changes) in the search window
            signs = torch.sign(region)
            crossings = ((signs[1:] * signs[:-1]) < 0).nonzero(as_tuple=False)

            if len(crossings) == 0:
                zc = pos  # no crossing found — use centre
            else:
                # Pick crossing nearest to the seam centre
                zc = lo + int(crossings[torch.argmin(torch.abs(crossings - window))].item())

            # Apply tiny triangular window centred on zero-crossing
            half = min(window, zc, audio.shape[-1] - zc - 1)
            if half < 2:
                continue
            ramp = torch.linspace(0.0, 1.0, half, device=audio.device)
            audio[..., zc - half:zc] *= ramp
            audio[..., zc:zc + half] *= ramp.flip(0)

        return audio
