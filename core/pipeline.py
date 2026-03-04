"""Orchestrates the full meditation audio generation pipeline."""

import numpy as np

from core.audio_processor import apply_fx, make_master_chain, make_music_chain, make_voice_chain
from core.mixer import export_audio, mix, normalize_loudness
from core.music_engine import MusicEngine
from core.script_parser import parse_script
from core.tts_engine import TTSEngine

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
    constraints = "no drums, no percussion, no vocals, beatless"

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


class MeditationPipeline:
    """End-to-end meditation audio generator."""

    def __init__(self):
        self.tts = TTSEngine()
        self.music = MusicEngine()

    def generate(
        self,
        script: str,
        music_prompt: str,
        voice: str = "af_heart,af_nicole",
        speed: float = 0.80,
        tts_engine: str = "kokoro",
        parler_voice_preset: str = "Serene Female — warm, calm, breathy",
        parler_custom_description: str = "",
        duck_amount_db: float = -4.0,
        reverb_amount: float = 0.15,
        fade_in_sec: float = 3.0,
        fade_out_sec: float = 5.0,
        output_format: str = "wav",
        progress_cb=None,
    ) -> tuple[str, str]:
        """Run the full pipeline and return the path to the output audio file.

        Args:
            script: Meditation script with [pause:Xs] markers.
            music_prompt: Text description of desired background music.
            voice: Kokoro voice name.
            speed: Speaking speed (0.5–1.0).
            tts_engine: "kokoro" or "parler".
            parler_voice_preset: Parler voice preset label.
            parler_custom_description: Custom Parler voice description.
            duck_amount_db: How much to reduce music during speech (negative dB).
            reverb_amount: Voice reverb wet level (0.0–0.5).
            fade_in_sec: Fade-in duration for the final mix.
            fade_out_sec: Fade-out duration for the final mix.
            output_format: "wav" or "mp3".
            progress_cb: Called with (fraction: float, message: str).

        Returns:
            Tuple of (path_to_output_file, status_message).
        """
        status_message = ""
        try:
            # ── Step 1: Parse script ────────────────────────────────────────
            _progress(progress_cb, 0.0, "Parsing meditation script...")
            segments = parse_script(script)
            if not segments:
                raise ValueError("Script is empty or contains no content.")

            # ── Step 2: Load TTS ────────────────────────────────────────────
            if tts_engine == "parler":
                _progress(progress_cb, 0.05, "Loading Parler TTS engine...")
                from core.parler_engine import ParlerTTSEngine
                tts = ParlerTTSEngine()
                tts.load_model()
            else:
                _progress(progress_cb, 0.05, "Loading Kokoro voice model...")
                tts = self.tts
                tts.load_model()

            # ── Step 3: Synthesize narration ────────────────────────────────
            _progress(progress_cb, 0.10, "Synthesizing narration...")

            def tts_progress(current, total):
                frac = 0.10 + 0.30 * (current / max(total, 1))
                _progress(progress_cb, frac, f"Synthesizing segment {current}/{total}...")

            if tts_engine == "parler":
                voice_param = (
                    parler_custom_description
                    if parler_voice_preset == "Custom Description" and parler_custom_description
                    else parler_voice_preset
                )
                voice_audio, voice_activity = tts.synthesize(
                    segments, voice=voice_param, speed=speed, progress_cb=tts_progress
                )

                # ── Vocal sanity check ──────────────────────────────────
                # Catch broken TTS output BEFORE it reaches the post-processor.
                _progress(progress_cb, 0.36, "Validating vocal stem...")
                vocal_ok = True
                if np.isnan(voice_audio).any():
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

                # Apply neural restoration immediately on the raw TTS output
                _progress(progress_cb, 0.38, "Applying AI vocal restoration...")
                from core.post_processor import MasteringEngine
                mastering_engine = MasteringEngine(sample_rate=SAMPLE_RATE)
                voice_audio = mastering_engine.restore_vocals(voice_audio, sr=SAMPLE_RATE)
            else:
                voice_audio, voice_activity = tts.synthesize(
                    segments, voice=voice, speed=speed, progress_cb=tts_progress
                )

            # ── Step 4: Unload TTS, load MusicGen ───────────────────────────
            _progress(progress_cb, 0.40, "Switching to music model...")
            tts.unload_model()
            if tts_engine == "parler":
                del tts
            self.music.load_model()

            # ── Step 5: Generate background music ───────────────────────────
            voice_duration = len(voice_audio) / SAMPLE_RATE
            music_duration = voice_duration + 10  # extra for pre-roll + fade-out

            _progress(progress_cb, 0.45, "Generating background music...")

            def music_progress(current, total):
                frac = 0.45 + 0.25 * (current / max(total, 1))
                _progress(progress_cb, frac, f"Generating music segment {current}/{total}...")

            # Enhance the user's prompt with meditation guardrails
            enhanced_prompt = _enhance_music_prompt(music_prompt)

            music_audio = self.music.generate(
                enhanced_prompt, music_duration, progress_cb=music_progress
            )

            # ── Step 6: Unload MusicGen ─────────────────────────────────────
            self.music.unload_model()

            # ── Step 7: Apply voice FX ──────────────────────────────────────
            _progress(progress_cb, 0.72, "Applying voice effects...")
            voice_chain = make_voice_chain(reverb_amount=reverb_amount)
            voice_audio = apply_fx(voice_audio, voice_chain, SAMPLE_RATE)

            # Align voice_activity to post-FX voice length (reverb tail trim
            # may change length slightly)
            voice_activity = voice_activity[:len(voice_audio)]
            if len(voice_activity) < len(voice_audio):
                pad = len(voice_audio) - len(voice_activity)
                voice_activity = np.concatenate([
                    voice_activity, np.zeros(pad, dtype=bool)
                ])

            # ── Step 8: Apply music FX ──────────────────────────────────────
            _progress(progress_cb, 0.77, "Applying music effects...")
            music_audio = normalize_loudness(music_audio, SAMPLE_RATE, target_lufs=-20.0)
            music_chain = make_music_chain()
            music_audio = apply_fx(music_audio, music_chain, SAMPLE_RATE)

            # ── Step 9: Mix with ducking ────────────────────────────────────
            _progress(progress_cb, 0.82, "Mixing voice and music...")
            mixed = mix(
                voice_audio,
                voice_activity,
                music_audio,
                sample_rate=SAMPLE_RATE,
                duck_amount_db=duck_amount_db,
                fade_in_sec=fade_in_sec,
                fade_out_sec=fade_out_sec,
            )

            # ── Step 10: Master processing ──────────────────────────────────
            _progress(progress_cb, 0.90, "Applying master processing...")
            master_chain = make_master_chain()
            mixed = apply_fx(mixed, master_chain, SAMPLE_RATE)

            # ── Step 11: Export ─────────────────────────────────────────────
            _progress(progress_cb, 0.95, f"Exporting {output_format.upper()}...")
            output_path = export_audio(mixed, SAMPLE_RATE, output_format)

        finally:
            # Prevent PyTorch/MPS teardown segfaults
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        _progress(progress_cb, 1.0, "Done!")
        return output_path, status_message
