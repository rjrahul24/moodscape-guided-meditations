"""Orchestrates the full meditation audio generation pipeline."""

import gc
import logging
import time

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

from core.audio_processor import make_master_chain
from core.mixer import export_audio, export_stems, mix, normalize_loudness
from core.qa_monitor import run_qa_checks
from core.kokoro_tts.engine import KokoroEngine
from core.stitch_client import StitchClient

logger = logging.getLogger("moodscape")

TARGET_SR = 44100
SAMPLE_RATE = 24000


def _progress(cb, fraction, message):
    if cb is not None:
        cb(fraction, message)


def _enhance_heartmula_prompt(user_prompt: str, duration_hint: float = 120.0) -> tuple[str, str]:
    """Build (tags, lyrics) for HeartMuLa from the user's music style prompt.

    HeartMuLa uses comma-separated style tags (not natural-language sentences).
    The tags field corresponds to the 'tags' parameter in heartlib.
    The lyrics field uses structural section markers for instrumental tracks.

    Follows the **Eight Pillars** tag hierarchy from HeartMuLa research:
      GENRE (95%) > TIMBRE (50%) > MOOD (32%) > INSTRUMENT (25%) > SCENE (20%)
    with the "Less is More" principle.  The model is trained with Random Dimension
    Dropout (2 of 6 annotation dimensions masked per sample), so 5-6 focused tags
    are optimal.  Overloading beyond 8 tags causes *probability interference*.

    Tag construction (target: 5-6 tags):
      1. Core anchors: Genre + Instrument (2 tags — most influential pillars)
      2. Pacing descriptor (duration-scaled natural language — not "60bpm" which
         tokenizes as ['60','b','pm'] and is not in training vocabulary)
      3. Negative floor: "no drums, instrumental" (2 tags)
      4. User tags appended (deduplicated against core anchors)

    Structural lyrics:
      - [interlude] markers — the standard instrumental section marker from
        HeartMuLa's Llama-3 training data (song lyrics format).
      - [intro] / [outro] for bookends.
      - Without text lines between markers, the LM generates instrumental pads.
      - One [interlude] per ~20s of target duration.
    """
    user_lower = user_prompt.lower().strip()

    # ── Core anchors (only the highest-influence pillars) ──────────────
    # Keep tag count low (5-6 total) for maximum conditioning strength.
    # Genre and Instrument are the two strongest pillars; Timbre and Mood
    # are conveyed implicitly by sub-genre choice ("deep ambient" = warm + calm).
    _PILLAR_DEFAULTS = {
        # pillar: (default_tag, keywords_that_indicate_user_coverage)
        "genre":      ("deep ambient", {"ambient", "new age", "drone", "classical",
                                        "psychill", "chillout", "lo-fi"}),
        "instrument": ("synthesizer",  {"synthesizer", "piano", "strings", "bowl",
                                        "singing bowl", "flute", "harp", "chimes",
                                        "organ", "cello", "guitar", "pads", "pad"}),
    }
    core = []
    for _pillar, (default, keywords) in _PILLAR_DEFAULTS.items():
        if not any(kw in user_lower for kw in keywords):
            core.append(default)

    # ── Pacing descriptor (duration-scaled) ────────────────────────────
    # Use natural-language descriptors that appeared in HeartMuLa training data.
    # Avoid "60bpm" — it tokenizes as ['60','b','pm'] (non-natural) and the
    # model was not trained on numeric BPM notation in the tags field.
    if duration_hint <= 90.0:
        pacing = "extremely slow"
    elif duration_hint <= 300.0:
        pacing = "slow"
    else:
        pacing = "slow, peaceful"

    # ── Assemble tags ─────────────────────────────────────────────────
    tag_parts = core + [pacing]

    # Append user tags (deduplicated against defaults)
    user_stripped = user_prompt.strip()
    if user_stripped:
        tag_parts.append(user_stripped)

    # Negative constraints floor — minimal but essential.
    # "no drums" prevents percussion leaking into ambient output.
    # "instrumental" suppresses vocal generation.
    tag_parts.extend(["no drums", "instrumental"])

    tags = ", ".join(tag_parts)

    # ── Structural lyrics ([interlude] markers) ────────────────────────
    # [interlude] is the standard instrumental section marker from HeartMuLa's
    # Llama-3 training data (song lyrics format).  Without text lines between
    # markers, the LM sustains instrumental pads — no vocals generated.
    # Each [interlude] covers ~15-20 seconds.
    n_sections = max(1, round(duration_hint / 20.0))
    sections = "\n\n".join(["[interlude]"] * n_sections)
    lyrics = f"[intro]\n\n{sections}\n\n[outro]"

    return tags, lyrics


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
        self.stitch = StitchClient()

    def generate(
        self,
        script: str,
        music_prompt: str,
        voice: str = "golden_hour",
        speed: float = 0.90,
        duck_amount_db: float = -12.0,
        reverb_amount: float = 0.15,
        fade_in_sec: float = 3.0,
        fade_out_sec: float = 5.0,
        output_format: str = "wav",
        progress_cb=None,
        seed: int | None = None,
        do_export_stems: bool = False,
        upsample_48k: bool = True,
        generation_mode: str = "Instrumental + Vocal",
        instrumental_duration_m: float = 3.0,
        music_model: str = "heartmula",
        music_prompt_stages: list[tuple[str, float]] | None = None,
        stem_separation: bool = True,
        bpm: int = 50,
        keyscale: str = "Auto",
        acestep_model_type: str = "sft",
        lyria_bpm: int = 70,
        lyria_density: float = 0.2,
        lyria_brightness: float = 0.3,
        tts_engine: str = "kokoro",
        f5_voice_slug: str | None = None,
        f5_target_wpm: int | None = None,
        reverb_ir: str = "warm_studio",
        do_stitch: bool = False,
        quality_mode: bool = False,
        stereo_output: bool = False,
        melody_audio_path: str | None = None,
    ) -> tuple[str, str | None, dict | None]:
        """Run the full pipeline and return the path to the output audio file.

        Args:
            script: Meditation script with [pause:Xs] markers.
            music_prompt: Text description of desired background music.
                Used when music_prompt_stages is None.
            voice: Kokoro voice name, preset, or comma-separated blend.
            speed: Speaking speed (0.5–1.0).
            duck_amount_db: How much to reduce music during speech (negative dB).
                Combined with music_volume_db=-14 dB baseline offset, -12 dB
                ducking gives ~22 dB voice-music separation during speech.
            reverb_amount: Voice reverb wet level (0.0–0.5).
            fade_in_sec: Fade-in duration for the final mix.
            fade_out_sec: Fade-out duration for the final mix.
            output_format: "wav" or "mp3".
            progress_cb: Called with (fraction: float, message: str).
            seed: Optional deterministic seed for reproducible generation.
            do_export_stems: If True, save voice/music stems alongside the mix.
            upsample_48k: If True, export at 48 kHz instead of 44.1 kHz.
            melody_audio_path: Optional path to a reference audio file for
                ACE-Step melody/style conditioning. When provided, the audio is
                loaded and passed to AceStepEngine as melody_audio + melody_sample_rate
                kwargs. Has no effect for HeartMuLa or Lyria engines.
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

        Returns:
            Tuple of (path_to_output_file, status_message, stitch_design).
        """
        status_message = ""
        stitch_design = None

        # Generate a session seed if not provided
        if seed is None:
            seed = int(time.time()) % (2**31)

        # Unified LUFS target for all meditation sessions.
        # -16 LUFS matches Apple Music natively; Spotify applies minimal +2 dB
        # boost.  The previous -19 LUFS caused platforms to boost and apply their
        # own limiter, degrading quality.
        target_lufs = -16.0

        is_instrumental = generation_mode == "Instrumental Only"
        is_vocals = generation_mode == "Vocals Only"
        use_acestep = music_model == "acestep"
        use_lyria = music_model == "lyria"
        use_heartmula = music_model == "heartmula"

        # All music engines output at 48 kHz natively; mix at that rate for quality.
        # F5-TTS also mixes at 48 kHz.
        # TTS voice (24 kHz) is upsampled to match whichever rate is selected.
        mix_sr = 48000 if (use_lyria or use_acestep or use_heartmula or tts_engine == "f5") else TARGET_SR

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
                gc.collect()

                # ── Step 3: Synthesize narration ────────────────────────────────
                _progress(progress_cb, 0.10, "Synthesizing narration...")

                def tts_progress(current, total):
                    frac = 0.10 + 0.30 * (current / max(total, 1))
                    _progress(progress_cb, frac, f"Synthesizing segment {current}/{total}...")

                if tts_engine == "f5":
                    voice_audio, voice_activity = tts.synthesize(
                        segments, speed=speed, progress_cb=tts_progress,
                        target_wpm=f5_target_wpm if f5_target_wpm and f5_target_wpm > 0 else None,
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

                # ── Mastering Engine Init (F5-TTS only) ─────────────────────────
                # Kokoro uses a unified voice chain (build_voice_chain) applied in
                # Step 7, replacing the old two-chain master_vocals + voice_chain.
                # F5-TTS still uses its own mastering engine.
                mastering_engine = None
                if tts_engine == "f5":
                    _progress(progress_cb, 0.38, "Preparing vocal mastering chain...")
                    from core.f5_tts.postprocessor import F5MasteringEngine
                    mastering_engine = F5MasteringEngine(sample_rate=SAMPLE_RATE)

                logger.info("TTS complete — %.1fs of audio", len(voice_audio) / SAMPLE_RATE)

                # Upsample Voice to the mix sample rate:
                #   Lyria / ACE-Step path → 48 kHz (preserves native music resolution)
                #   HeartMuLa → 44.1 kHz (HeartCodec native rate)
                # Always use high_accuracy=True (soxr_vhq) — mathematically zero
                # aliasing artifacts, avoids subtle metallic shimmer on non-integer
                # ratio conversions (24 kHz → 44.1 kHz).
                if mix_sr == 48000:
                    from core.audio_processor import upsample_audio
                    _progress(progress_cb, 0.39, "Upsampling TTS audio to 48kHz (High Fidelity sinc)...")
                    voice_audio = upsample_audio(
                        voice_audio, from_sr=SAMPLE_RATE, to_sr=48000,
                        high_accuracy=True,
                    )
                    # 48000 / 24000 = 2 exactly — clean integer repeat
                    voice_activity = np.repeat(voice_activity, 2)
                else:
                    from core.audio_processor import resample_highly_accurate
                    _progress(progress_cb, 0.39, "Upsampling TTS audio to 44.1kHz (soxr_vhq)...")
                    voice_audio = resample_highly_accurate(voice_audio, SAMPLE_RATE, TARGET_SR)
                    voice_activity = np.repeat(voice_activity, TARGET_SR // SAMPLE_RATE)
                pad_diff = len(voice_audio) - len(voice_activity)
                if pad_diff > 0:
                     voice_activity = np.concatenate([voice_activity, np.zeros(pad_diff, dtype=bool)])
                else:
                     voice_activity = voice_activity[:len(voice_audio)]

                # F5-TTS Phase B mastering (Kokoro skips — unified voice chain in Step 7)
                if mastering_engine is not None:
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

                music_model_label = "Lyria RealTime" if use_lyria else ("ACE-Step 1.5" if use_acestep else "HeartMuLa")
                _progress(progress_cb, 0.40, f"Switching to {music_model_label}...")

                # Instantiate the selected music engine
                if use_lyria:
                    from core.lyria.engine import LyriaEngine
                    music_engine = LyriaEngine()
                elif use_acestep:
                    from core.acestep_engine import AceStepEngine
                    music_engine = AceStepEngine()
                elif use_heartmula:
                    from core.heart_mula.engine import HeartMulaEngine
                    music_engine = HeartMulaEngine()
                else:
                    raise ValueError(f"Unknown music model: {music_model}")
                music_engine.load_model()
                gc.collect()

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
                    # Music must cover: pre-roll (4s) + voice + post-roll (8s) + safety margin
                    music_duration = voice_duration + 15

                _progress(progress_cb, 0.45, f"Generating background music ({music_model_label})...")

                def music_progress(current, total):
                    frac = 0.45 + 0.25 * (current / max(total, 1))
                    label = f"story stage {current}/{total}" if story_mode else f"segment {current}/{total}"
                    _progress(progress_cb, frac, f"Generating music {label}...")

                # Load reference audio for ACE-Step melody conditioning (if provided).
                # Loaded once here and passed as kwargs to all generate calls below.
                ref_audio_kwargs: dict = {}
                if use_acestep and melody_audio_path:
                    import soundfile as sf
                    try:
                        _ref_data, _ref_sr = sf.read(melody_audio_path, dtype="float32", always_2d=False)
                        if _ref_data.ndim == 2:
                            _ref_data = _ref_data.mean(axis=1)  # stereo → mono
                        ref_audio_kwargs = {"melody_audio": _ref_data, "melody_sample_rate": _ref_sr}
                        logger.info(
                            "[Pipeline] Reference audio loaded: %s — %.1fs @ %d Hz",
                            melody_audio_path, len(_ref_data) / _ref_sr, _ref_sr,
                        )
                    except Exception as e:
                        logger.warning("[Pipeline] Failed to load reference audio %s: %s — ignoring", melody_audio_path, e)

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
                            **ref_audio_kwargs,
                        )
                    elif use_heartmula:
                        engine_stages = [
                            (_enhance_heartmula_prompt(p, duration_hint=d)[0], d)
                            for p, d in music_prompt_stages
                        ]
                        enhanced_tags, _ = _enhance_heartmula_prompt(music_prompt, duration_hint=music_duration)
                        music_audio = music_engine.generate(
                            enhanced_tags,
                            music_duration,
                            progress_cb=music_progress,
                            prompt_stages=engine_stages,
                            quality_mode=quality_mode,
                        )
                    else:
                        raise ValueError(f"No story mode path for music_model: {music_model}")
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
                            acestep_model_type=acestep_model_type,
                            **ref_audio_kwargs,
                        )
                    elif use_heartmula:
                        enhanced_tags, enhanced_lyrics = _enhance_heartmula_prompt(music_prompt, duration_hint=music_duration)
                        music_audio = music_engine.generate(
                            enhanced_tags, music_duration, progress_cb=music_progress,
                            lyrics=enhanced_lyrics, quality_mode=quality_mode,
                        )
                    else:
                        raise ValueError(f"No generation path for music_model: {music_model}")

                # ── Step 5b: Unload music model ─────────────────────────────────
                music_engine.unload_model()
                if use_acestep or use_lyria or use_heartmula:
                    del music_engine
                gc.collect()
                if mx:
                    # Force release ALL cached MLX metal buffers back to the OS
                    # so subsequent steps (Demucs subprocess) have enough memory.
                    mx.set_cache_limit(0)
                    mx.clear_cache()

                # ── Step 5c: AI Source Separation (remove drums/vocals) ─────────
                if stem_separation:
                    _progress(progress_cb, 0.68, "Removing drums/vocals via AI source separation...")
                    if use_lyria:
                        from core.lyria.engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    elif use_acestep:
                        from core.acestep_engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    elif use_heartmula:
                        from core.heart_mula.engine import TARGET_SAMPLE_RATE as MUSIC_SR
                    else:
                        raise ValueError(f"Unknown music model for stem separation: {music_model}")
                    from core.stem_separator import StemSeparator
                    separator = StemSeparator()
                    music_audio = separator.remove_drums_and_vocals(music_audio, MUSIC_SR)
                    del separator
                    gc.collect()

                # ── Step 5d: Neural enhancement (HeartMuLa only) ─────────────
                if use_heartmula:
                    _progress(progress_cb, 0.70, "Applying neural codec artifact removal...")
                    from core.neural_enhancer import enhance_with_apollo
                    music_audio = enhance_with_apollo(music_audio, mix_sr)

                # All engines (ACE-Step, Lyria, HeartMuLa) output at 48 kHz natively.
                _progress(progress_cb, 0.71, f"Ensuring {music_model_label} audio is at 48kHz mixing rate...")
            else:
                music_audio = np.zeros(0, dtype=np.float32)

            if not is_instrumental:
                # ── Step 7: Apply voice FX ──────────────────────────────────────
                _progress(progress_cb, 0.72, "Applying voice effects...")
                if tts_engine == "f5":
                    from core.f5_tts.postprocessor import build_f5_voice_chain
                    from core.kokoro_tts.postprocessor import apply_fx
                    voice_chain = build_f5_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
                else:
                    from core.kokoro_tts.postprocessor import build_voice_chain, apply_fx
                    voice_chain = build_voice_chain(reverb_amount=reverb_amount, ir_name=reverb_ir)
                voice_audio = apply_fx(voice_audio, voice_chain, mix_sr)

                # Align voice_activity to post-FX voice length (reverb tail trim
                # may change length slightly)
                voice_activity = voice_activity[:len(voice_audio)]
                if len(voice_activity) < len(voice_audio):
                    pad = len(voice_audio) - len(voice_activity)
                    voice_activity = np.concatenate([
                        voice_activity, np.zeros(pad, dtype=bool)
                    ])

                # Pre-normalize voice to -18 LUFS before mixing.
                # This ensures a consistent voice-to-music ratio regardless of
                # TTS engine output level.  We use -18 (not -16) because the
                # final mix targets -16 LUFS and we need headroom for the
                # music underneath.
                voice_audio = normalize_loudness(voice_audio, mix_sr, target_lufs=-18.0)

            if not is_vocals:
                # ── Step 8: Apply music FX ──────────────────────────────────────
                _progress(progress_cb, 0.77, "Applying music effects...")
                from core.audio_processor import apply_fx as apply_audio_fx
                # Per-engine pre-mix loudness calibration.
                # These targets are tuned so that music_volume_db=-14 dB in mix()
                # produces the right ambient presence during pauses, while the
                # duck_amount_db offset provides adequate separation during speech.
                #
                #   ACE-Step  -14 LUFS: VAE output is clean and benefits from the
                #                       extra headroom; gives -28 LUFS baseline in mix
                #                       and -40 LUFS during speech (22 dB separation).
                #   HeartMuLa -17 LUFS: noise reduction + tape saturation increase
                #                       perceived loudness; lower pre-norm prevents
                #                       the mix from feeling over-compressed.
                #   Lyria     -16 LUFS: cloud output is slightly brighter/denser;
                #                       moderate level balances presence vs. headroom.
                if use_acestep:
                    premix_lufs = -14.0
                elif use_heartmula:
                    premix_lufs = -17.0
                else:  # lyria
                    premix_lufs = -16.0
                music_audio = normalize_loudness(music_audio, mix_sr, target_lufs=premix_lufs)

                # Pre-EQ processing: spectral repair + tape saturation
                if use_acestep:
                    # Moderate noise reduction (prop_decrease=0.45) removes diffusion
                    # static without causing warbling on sustained pads. Matches HeartMuLa
                    # treatment — 0.25 was too conservative, leaving 75% of the noise floor.
                    from core.audio_processor import reduce_music_noise
                    music_audio = reduce_music_noise(music_audio, mix_sr, prop_decrease=0.45)
                if use_heartmula:
                    from core.audio_processor import reduce_music_noise, apply_tape_saturation
                    music_audio = reduce_music_noise(music_audio, mix_sr, prop_decrease=0.45)
                    music_audio = apply_tape_saturation(music_audio, drive=0.2, bias=0.10)

                if use_lyria:
                    from core.audio_processor import make_lyria_music_chain
                    music_chain = make_lyria_music_chain()
                elif use_acestep:
                    from core.audio_processor import make_acestep_music_chain
                    music_chain = make_acestep_music_chain()
                elif use_heartmula:
                    from core.audio_processor import make_heartmula_music_chain
                    music_chain = make_heartmula_music_chain()
                else:
                    raise ValueError(f"No music FX chain for: {music_model}")
                music_audio = apply_audio_fx(music_audio, music_chain, mix_sr)

                # Post-EQ: add organic noise floor for analog warmth (HeartMuLa only;
                # ACE-Step output doesn't benefit from added noise)
                if use_heartmula:
                    from core.audio_processor import add_organic_noise_floor
                    music_audio = add_organic_noise_floor(music_audio, mix_sr)

                # Carve vocal pocket for intelligibility
                from core.audio_processor import make_vocal_pocket_chain
                vocal_pocket_chain = make_vocal_pocket_chain()
                music_audio = apply_audio_fx(music_audio, vocal_pocket_chain, mix_sr)

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
                if stereo_output:
                    from core.stereo_upmix import haas_stereo
                    music_audio = haas_stereo(music_audio, mix_sr)
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
                    stereo_output=stereo_output,
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

            # Lyria and ACE-Step export at native 48 kHz.
            # HeartMuLa respects the user's upsample_48k preference.
            export_sr = 48000 if (use_lyria or use_acestep or upsample_48k) else TARGET_SR

            output_path = export_audio(
                mixed,
                sample_rate=mix_sr,
                output_format=output_format,
                target_sample_rate=export_sr,
                master_chain=master_chain,
                target_lufs=target_lufs,
            )

            # ── Step 12: Stitch Design Generation ───────────────────────────
            if do_stitch:
                _progress(progress_cb, 0.98, "Generating Stitch UI design concept...")
                metadata = {
                    "mood": "calm",  # This should be derived from the prompt/script
                    "theme": music_prompt,
                    "duration": music_duration / 60.0 if not is_vocals else 10.0
                }
                stitch_design = self.stitch.generate_design_concept(metadata)
                
                # Save stitch design to a file next to the audio
                design_path = os.path.splitext(output_path)[0] + "_design.json"
                with open(design_path, "w") as f:
                    json.dump(stitch_design, f, indent=2)
                logger.info("Stitch design saved to %s", design_path)

        finally:
            # Prevent PyTorch/MPS teardown segfaults
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Additional stability for MLX and general Python heap
            gc.collect()
            if mx:
                mx.set_cache_limit(0)
                mx.clear_cache()

        _progress(progress_cb, 1.0, "Done!")
        return output_path, status_message, stitch_design
