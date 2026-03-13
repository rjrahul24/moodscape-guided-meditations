"""Orchestrates the full meditation audio generation pipeline."""

import logging
import time

import numpy as np

from core.audio_processor import make_master_chain, make_music_chain
from core.mixer import export_audio, export_stems, mix, normalize_loudness
from core.music_engine import MusicEngine
from core.qa_monitor import run_qa_checks
from core.kokoro_tts.engine import KokoroEngine

logger = logging.getLogger("moodscape")

TARGET_SR = 44100
SAMPLE_RATE = 24000


def _progress(cb, fraction, message):
    if cb is not None:
        cb(fraction, message)


def _enhance_music_prompt(user_prompt: str) -> str:
    """Build a MusicGen-optimized prompt from the user's description.

    Keeps total under ~45 words for best MusicGen attention utilization.
    Always includes essential constraints (no drums/vocals). Adds ambient
    descriptors only if the user hasn't already mentioned them to avoid
    diluting the attention budget with duplicates.
    """
    constraints = "subliminal background, barely audible drone, single sustained note, pure ambient texture, no melody, no chord changes, no drums, no percussion, no vocals, beatless"

    # Add ambient descriptors only if the user hasn't already included them
    optional = [
        ("ambient", "ambient"),
        ("reverb", "spacious reverb"),
        ("evolving", "slow evolving"),
        ("warm", "warm"),
    ]
    user_lower = user_prompt.lower()
    extras = [desc for key, desc in optional if key not in user_lower][:2]

    parts = [constraints] + extras + [user_prompt]
    enhanced = ", ".join(parts)

    # Cap at 45 words to stay within MusicGen's effective attention window
    words = enhanced.split()
    if len(words) > 45:
        enhanced = " ".join(words[:45])

    return enhanced


def _enhance_acestep_prompt(user_prompt: str, duration_hint: float = 120.0) -> tuple[str, str]:
    """Build an ACE-Step-optimized prompt using the MESA framework.

    Delegates to AceStepEngine's prompt enhancer which returns
    (caption, lyrics) tuple with duration-aware structural tags.
    """
    from core.acestep_engine import AceStepEngine
    return AceStepEngine._enhance_prompt(user_prompt, duration_hint=duration_hint)


class MeditationPipeline:
    """End-to-end meditation audio generator."""

    def __init__(self):
        self.tts = KokoroEngine()
        self.music = MusicEngine()

    def generate(
        self,
        script: str,
        music_prompt: str,
        voice: str = "golden_hour",
        speed: float = 0.78,
        duck_amount_db: float = -21.0,
        reverb_amount: float = 0.15,
        fade_in_sec: float = 3.0,
        fade_out_sec: float = 5.0,
        output_format: str = "wav",
        progress_cb=None,
        seed: int | None = None,
        do_export_stems: bool = False,
        upsample_48k: bool = False,
        generation_mode: str = "Instrumental + Vocal",
        instrumental_duration_m: float = 3.0,
        music_model: str = "musicgen",
        music_prompt_stages: list[tuple[str, float]] | None = None,
        stem_separation: bool = True,
        melody_audio: np.ndarray | None = None,
        melody_sample_rate: int | None = None,
        bpm: int = 50,
        keyscale: str = "Auto",
        acestep_model_type: str = "sft",
        lyria_bpm: int = 70,
        lyria_density: float = 0.2,
        lyria_brightness: float = 0.3,
        tts_engine: str = "kokoro",
        f5_voice_slug: str | None = None,
    ) -> tuple[str, str | None]:
        """Run the full pipeline and return the path to the output audio file.

        Args:
            script: Meditation script with [pause:Xs] markers.
            music_prompt: Text description of desired background music.
                Used when music_prompt_stages is None.
            voice: Kokoro voice name, preset, or comma-separated blend.
            speed: Speaking speed (0.5–1.0).
            duck_amount_db: How much to reduce music during speech (negative dB).
                Combined with music_volume_db=-17 → total -38 dB during speech.
            reverb_amount: Voice reverb wet level (0.0–0.5).
            fade_in_sec: Fade-in duration for the final mix.
            fade_out_sec: Fade-out duration for the final mix.
            output_format: "wav" or "mp3".
            progress_cb: Called with (fraction: float, message: str).
            seed: Optional deterministic seed for reproducible generation.
            do_export_stems: If True, save voice/music stems alongside the mix.
            upsample_48k: If True, export at 48 kHz instead of 44.1 kHz.
            music_prompt_stages: Optional list of (prompt, duration_sec) pairs
                for story mode music generation. When provided, overrides
                music_prompt and instrumental_duration_m (total music duration
                is the sum of stage durations). Each stage prompt is enhanced
                with model-specific meditation guardrails. Example:
                    [
                        ("calm breathing pads, soft sine waves", 90.0),
                        ("deep sleep ambient drones, very slow", 180.0),
                        ("gentle awakening, morning light, birds", 90.0),
                    ]
            stem_separation: If True (default), run AI source separation
                (HT Demucs) on generated music to remove any unwanted drums
                or vocals that the model may have produced despite prompting.
            melody_audio: Optional reference audio for melody conditioning
                (MusicGen only). Float32 mono numpy array. Guides the model's
                melodic/harmonic structure toward the reference.
            melody_sample_rate: Sample rate of melody_audio.

        Returns:
            Tuple of (path_to_output_file, status_message).
        """
        status_message = ""

        # Generate a session seed if not provided
        if seed is None:
            seed = int(time.time()) % (2**31)

        # Unified LUFS target for all meditation sessions
        target_lufs = -14.0

        is_instrumental = generation_mode == "Instrumental Only"
        is_vocals = generation_mode == "Vocals Only"
        use_acestep = music_model == "acestep"
        use_lyria = music_model == "lyria"

        # Lyria mixes at its native 48 kHz; all other engines mix at 44.1 kHz.
        # TTS voice (24 kHz) is upsampled to match whichever rate is selected.
        mix_sr = 48000 if use_lyria else TARGET_SR

        logger.info("Starting generation — mode=%s, music_model=%s, voice=%s, speed=%s, seed=%s, lufs=%s",
                    generation_mode, music_model, voice, speed, seed, target_lufs)

        try:
            if not is_instrumental:
                # ── Step 1: Parse script ────────────────────────────────────────
                _progress(progress_cb, 0.0, "Parsing meditation script...")

                if tts_engine == "f5":
                    from core.f5_tts.preprocessor import prepare_segments as _prepare
                else:
                    from core.kokoro_tts.preprocessor import prepare_segments as _prepare
                segments = _prepare(script)

                if not segments:
                    raise ValueError("Script is empty or contains no content.")

                logger.info(
                    "Script: %d segments, %d speech blocks",
                    len(segments),
                    sum(1 for s in segments if s["type"] == "speech"),
                )

                # ── Step 2: Load TTS ────────────────────────────────────────────
                if tts_engine == "f5":
                    from core.f5_tts.engine import F5Engine
                    tts = F5Engine(voice_slug=f5_voice_slug)
                    _progress(progress_cb, 0.05, "Loading F5-TTS voice model...")
                else:
                    tts = self.tts
                    _progress(progress_cb, 0.05, "Loading Kokoro voice model...")
                tts.load_model()

                # ── Step 3: Synthesize narration ────────────────────────────────
                _progress(progress_cb, 0.10, "Synthesizing narration...")

                def tts_progress(current, total):
                    frac = 0.10 + 0.30 * (current / max(total, 1))
                    _progress(progress_cb, frac, f"Synthesizing segment {current}/{total}...")

                if tts_engine == "f5":
                    voice_audio, voice_activity = tts.synthesize(
                        segments, speed=speed, progress_cb=tts_progress,
                    )
                else:
                    voice_audio, voice_activity = tts.synthesize(
                        segments, voice=voice, speed=speed,
                        progress_cb=tts_progress, seed=seed,
                    )

                logger.info("TTS complete — %.1fs of audio", len(voice_audio) / SAMPLE_RATE)

                # ── Vocal sanity check ──────────────────────────────────
                # Catch broken TTS output BEFORE it reaches the post-processor.
                _progress(progress_cb, 0.36, "Validating vocal stem...")
                vocal_ok = True
                if len(voice_audio) == 0:
                    status_message += "WARNING: Vocal stem is empty. "
                    vocal_ok = False
                elif np.isnan(voice_audio).any():
                    status_message += "WARNING: Vocal stem contains NaN values. "
                    vocal_ok = False
                elif np.abs(voice_audio).max() < 1e-6:
                    status_message += "WARNING: Vocal stem is silent. "
                    vocal_ok = False
                else:
                    # Zero-crossing rate > 0.4 is characteristic of noise, not speech
                    zcr = float(
                        np.sum(np.abs(np.diff(np.sign(voice_audio))))
                        / (2 * max(len(voice_audio), 1))
                    )
                    if zcr > 0.4:
                        status_message += (
                            f"WARNING: Vocal stem looks like noise (ZCR={zcr:.3f}). "
                        )
                        vocal_ok = False

                if not vocal_ok:
                    print(f"[Pipeline] {status_message}")

                # ── Phase A: Mastering Engine Init ──────────────────────────────
                # Both engines' restore_vocals() stubs are bypassed — output from
                # Kokoro (ISTFTNet) and F5-TTS (Vocos) is pre-clean.
                # mastering_engine is still needed for Phase B (master_vocals).
                _progress(progress_cb, 0.38, "Preparing vocal mastering chain...")
                if tts_engine == "f5":
                    from core.f5_tts.postprocessor import F5MasteringEngine
                    mastering_engine = F5MasteringEngine(sample_rate=SAMPLE_RATE)
                else:
                    from core.kokoro_tts.postprocessor import KokoroMasteringEngine
                    mastering_engine = KokoroMasteringEngine(sample_rate=SAMPLE_RATE)

                logger.info("TTS complete — %.1fs of audio", len(voice_audio) / SAMPLE_RATE)

                # Upsample Voice to the mix sample rate:
                #   Lyria path → 48 kHz (TTS upsampled to match native Lyria quality)
                #   All other engines → 44.1 kHz (studio standard)
                if use_lyria:
                    from core.audio_processor import upsample_audio
                    _progress(progress_cb, 0.39, "Upsampling TTS audio to 48kHz (Lyria path)...")
                    voice_audio = upsample_audio(voice_audio, from_sr=SAMPLE_RATE, to_sr=48000)
                    # 48000 / 24000 = 2 exactly — clean integer repeat
                    voice_activity = np.repeat(voice_activity, 2)
                else:
                    from core.audio_processor import resample_to_44100
                    _progress(progress_cb, 0.39, "Upsampling TTS audio to 44.1kHz standard...")
                    voice_audio = resample_to_44100(voice_audio, SAMPLE_RATE)
                    voice_activity = np.repeat(voice_activity, TARGET_SR // SAMPLE_RATE)
                pad_diff = len(voice_audio) - len(voice_activity)
                if pad_diff > 0:
                     voice_activity = np.concatenate([voice_activity, np.zeros(pad_diff, dtype=bool)])
                else:
                     voice_activity = voice_activity[:len(voice_audio)]

                # ── Phase B: Mastering EQ / De-Ess / Limiting ────────────────────
                _progress(progress_cb, 0.39, "Mastering vocal stem (EQ/De-Ess)...")
                voice_audio = mastering_engine.master_vocals(voice_audio, sr=mix_sr)
            else:
                voice_audio = np.zeros(0, dtype=np.float32)
                voice_activity = np.zeros(0, dtype=bool)

            if not is_vocals:
                # ── Step 4: Unload TTS, load music model ────────────────────────
                if not is_instrumental:
                    tts.unload_model()
                    if tts_engine == "f5":
                        del tts

                music_model_label = "Lyria RealTime" if use_lyria else ("ACE-Step 1.5" if use_acestep else "MusicGen")
                _progress(progress_cb, 0.40, f"Switching to {music_model_label}...")

                # Instantiate the selected music engine
                if use_lyria:
                    from core.lyria.engine import LyriaEngine
                    music_engine = LyriaEngine()
                elif use_acestep:
                    from core.acestep_engine import AceStepEngine
                    music_engine = AceStepEngine()
                else:
                    music_engine = self.music
                music_engine.load_model()

                # ── Step 5: Generate background music ───────────────────────────
                story_mode = music_prompt_stages is not None
                if story_mode:
                    # Story mode: total duration is derived from the stages.
                    # Add a small buffer so the final fade-out has audio to work with.
                    music_duration = sum(d for _, d in music_prompt_stages) + 5.0
                elif is_instrumental:
                    music_duration = instrumental_duration_m * 60.0
                else:
                    voice_duration = len(voice_audio) / mix_sr
                    music_duration = voice_duration + 10  # extra for pre-roll + fade-out

                _progress(progress_cb, 0.45, f"Generating background music ({music_model_label})...")

                def music_progress(current, total):
                    frac = 0.45 + 0.25 * (current / max(total, 1))
                    label = f"story stage {current}/{total}" if story_mode else f"segment {current}/{total}"
                    _progress(progress_cb, frac, f"Generating music {label}...")

                # Melody conditioning is only supported by MusicGen
                melody_kwargs = {}
                if not use_acestep and not use_lyria and melody_audio is not None and melody_sample_rate is not None:
                    melody_kwargs = {
                        "melody_audio": melody_audio,
                        "melody_sample_rate": melody_sample_rate,
                    }
                    logger.info("Melody conditioning enabled — %.1fs reference audio",
                                len(melody_audio) / melody_sample_rate)

                if story_mode:
                    # Build engine-specific stages: enhance each stage prompt
                    # with its model's meditation guardrails.
                    if use_lyria:
                        # LyriaEngine handles prompt enhancement internally;
                        # pass raw stage prompts so they are not double-enhanced.
                        engine_stages = music_prompt_stages
                        enhanced_prompt = music_prompt
                        music_audio = music_engine.generate(
                            enhanced_prompt,
                            music_duration,
                            progress_cb=music_progress,
                            prompt_stages=engine_stages,
                            bpm=lyria_bpm,
                            density=lyria_density,
                            brightness=lyria_brightness,
                        )
                    elif use_acestep:
                        # AceStepEngine enhances prompts internally per stage;
                        # pass raw stage prompts so they are not double-enhanced.
                        engine_stages = music_prompt_stages
                        enhanced_prompt, enhanced_lyrics = _enhance_acestep_prompt(music_prompt, duration_hint=music_duration)
                        music_audio = music_engine.generate(
                            enhanced_prompt,
                            music_duration,
                            progress_cb=music_progress,
                            prompt_stages=engine_stages,
                            lyrics=enhanced_lyrics,
                            bpm=bpm,
                            keyscale=keyscale,
                            **melody_kwargs,
                        )
                    else:
                        # MusicEngine expects pre-enhanced prompts (pipeline enhances).
                        engine_stages = [
                            (_enhance_music_prompt(p), d)
                            for p, d in music_prompt_stages
                        ]
                        enhanced_prompt = _enhance_music_prompt(music_prompt)
                        music_audio = music_engine.generate(
                            enhanced_prompt,
                            music_duration,
                            progress_cb=music_progress,
                            prompt_stages=engine_stages,
                            **melody_kwargs,
                        )
                else:
                    # Single-prompt generation
                    if use_lyria:
                        # LyriaEngine handles prompt enhancement internally
                        music_audio = music_engine.generate(
                            music_prompt, music_duration, progress_cb=music_progress,
                            bpm=lyria_bpm, density=lyria_density, brightness=lyria_brightness,
                        )
                    elif use_acestep:
                        enhanced_prompt, lyrics = _enhance_acestep_prompt(music_prompt, duration_hint=music_duration)
                        music_audio = music_engine.generate(
                            enhanced_prompt, music_duration, progress_cb=music_progress,
                            lyrics=lyrics, bpm=bpm, keyscale=keyscale, 
                            acestep_model_type=acestep_model_type, **melody_kwargs,
                        )
                    else:
                        enhanced_prompt = _enhance_music_prompt(music_prompt)
                        music_audio = music_engine.generate(
                            enhanced_prompt, music_duration, progress_cb=music_progress,
                            **melody_kwargs,
                        )

                # ── Step 5b: Unload music model ─────────────────────────────────
                music_engine.unload_model()
                if use_acestep or use_lyria:
                    del music_engine

                # ── Step 5c: AI Source Separation (remove drums/vocals) ─────────
                if stem_separation:
                    _progress(progress_cb, 0.68, "Removing drums/vocals via AI source separation...")
                    if use_lyria:
                        from core.lyria.engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    elif use_acestep:
                        from core.acestep_engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    else:
                        from core.music_engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    from core.stem_separator import StemSeparator
                    separator = StemSeparator()
                    separator.load_model()
                    music_audio = separator.remove_drums_and_vocals(music_audio, MUSIC_SR)
                    separator.unload_model()
                    del separator

                # Resample music audio to the mix sample rate.
                # Lyria is already at 48 kHz (mix_sr) — no resampling needed.
                # ACE-Step and MusicGen output at 24 kHz and must be upsampled.
                if use_lyria:
                    _progress(progress_cb, 0.70, "Lyria audio at native 48kHz — no resampling needed.")
                    # music_audio is already at 48 kHz mono float32
                else:
                    _progress(progress_cb, 0.70, f"Upsampling {music_model_label} audio to 44.1kHz standard...")
                    from core.audio_processor import resample_to_44100
                    if use_acestep:
                        from core.acestep_engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    else:
                        from core.music_engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    music_audio = resample_to_44100(music_audio, MUSIC_SR)
            else:
                music_audio = np.zeros(0, dtype=np.float32)

            if not is_instrumental:
                # ── Step 7: Apply voice FX ──────────────────────────────────────
                _progress(progress_cb, 0.72, "Applying voice effects...")
                from core.kokoro_tts.postprocessor import build_voice_chain, apply_fx
                voice_chain = build_voice_chain(reverb_amount=reverb_amount)
                voice_audio = apply_fx(voice_audio, voice_chain, mix_sr)

                # Align voice_activity to post-FX voice length (reverb tail trim
                # may change length slightly)
                voice_activity = voice_activity[:len(voice_audio)]
                if len(voice_activity) < len(voice_audio):
                    pad = len(voice_audio) - len(voice_activity)
                    voice_activity = np.concatenate([
                        voice_activity, np.zeros(pad, dtype=bool)
                    ])

            if not is_vocals:
                # ── Step 8: Apply music FX ──────────────────────────────────────
                _progress(progress_cb, 0.77, "Applying music effects...")
                from core.audio_processor import apply_fx as apply_audio_fx
                # Pre-mix loudness targets per engine:
                #   Lyria:    -22 LUFS — slightly quieter, avoids dominating the mix
                #   ACE-Step: -14 LUFS — matches final streaming standard; ensures strong,
                #             audible presence during vocal pauses before the -17 dB
                #             music_volume_db offset and ducking are applied in mix()
                #   MusicGen: -20 LUFS — established reference
                if use_lyria:
                    premix_lufs = -22.0
                elif use_acestep:
                    premix_lufs = -14.0
                else:
                    premix_lufs = -20.0
                music_audio = normalize_loudness(music_audio, mix_sr, target_lufs=premix_lufs)
                if use_lyria:
                    from core.audio_processor import make_lyria_music_chain
                    music_chain = make_lyria_music_chain()
                elif use_acestep:
                    from core.audio_processor import make_acestep_music_chain
                    music_chain = make_acestep_music_chain()
                else:
                    music_chain = make_music_chain()
                music_audio = apply_audio_fx(music_audio, music_chain, mix_sr)

            # ── Optional: Export stems ──────────────────────────────────────
            stem_paths = None
            if do_export_stems and not is_instrumental and not is_vocals:
                _progress(progress_cb, 0.80, "Exporting stems...")
                stem_paths = export_stems(voice_audio, music_audio, mix_sr)
                logger.info("Stems exported: %s", stem_paths)

            # ── Step 9: Mix with ducking ────────────────────────────────────
            if is_instrumental:
                _progress(progress_cb, 0.82, "Applying final fades...")
                from core.mixer import apply_fades
                mixed = apply_fades(music_audio, mix_sr, fade_in_sec, fade_out_sec)
            elif is_vocals:
                _progress(progress_cb, 0.82, "Applying final fades...")
                from core.mixer import apply_fades
                mixed = apply_fades(voice_audio, mix_sr, fade_in_sec, fade_out_sec)
            else:
                _progress(progress_cb, 0.82, "Mixing voice and music...")
                mixed = mix(
                    voice_audio,
                    voice_activity,
                    music_audio,
                    sample_rate=mix_sr,
                    duck_amount_db=duck_amount_db,
                    fade_in_sec=fade_in_sec,
                    fade_out_sec=fade_out_sec,
                )

            # ── Step 10: Master processing ──────────────────────────────────
            _progress(progress_cb, 0.90, "Applying master processing...")
            master_chain = make_master_chain()
            # Mixed array is not processed here to prevent memory spikes.
            # Master FX, normalization, and resampling will be applied via
            # chunked streaming in `export_audio()`.

            # ── Step 10b: QA checks ─────────────────────────────────────────
            qa_results = run_qa_checks(mixed, mix_sr)
            if qa_results["silence"]:
                status_message += f"QA: {len(qa_results['silence'])} long silence(s) detected. "
            if not qa_results["clipping"]["passed"]:
                status_message += "QA: Pre-master clipping detected. "

            # ── Step 11: Export ─────────────────────────────────────────────
            _progress(progress_cb, 0.95, f"Exporting {output_format.upper()}...")

            # Lyria always exports at 48 kHz (native quality).
            # Other engines respect the user's upsample_48k preference.
            export_sr = 48000 if (use_lyria or upsample_48k) else TARGET_SR

            output_path = export_audio(
                mixed,
                sample_rate=mix_sr,
                output_format=output_format,
                target_sample_rate=export_sr,
                master_chain=master_chain,
                target_lufs=target_lufs,
            )

        finally:
            # Prevent PyTorch/MPS teardown segfaults
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        _progress(progress_cb, 1.0, "Done!")
        return output_path, status_message
