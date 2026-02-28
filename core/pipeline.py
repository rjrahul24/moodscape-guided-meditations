"""Orchestrates the full meditation audio generation pipeline."""

from core.audio_processor import apply_fx, make_master_chain, make_music_chain, make_voice_chain
from core.mixer import export_audio, mix
from core.music_engine import MusicEngine
from core.script_parser import parse_script
from core.tts_engine import TTSEngine

SAMPLE_RATE = 24000


def _progress(cb, fraction, message):
    if cb is not None:
        cb(fraction, message)


class MeditationPipeline:
    """End-to-end meditation audio generator."""

    def __init__(self):
        self.tts = TTSEngine()
        self.music = MusicEngine()

    def generate(
        self,
        script: str,
        music_prompt: str,
        voice: str = "af_heart",
        speed: float = 0.85,
        duck_amount_db: float = -8.0,
        reverb_amount: float = 0.15,
        fade_in_sec: float = 3.0,
        fade_out_sec: float = 5.0,
        output_format: str = "wav",
        progress_cb=None,
    ) -> str:
        """Run the full pipeline and return the path to the output audio file.

        Args:
            script: Meditation script with [pause:Xs] markers.
            music_prompt: Text description of desired background music.
            voice: Kokoro voice name.
            speed: Speaking speed (0.5–1.0).
            duck_amount_db: How much to reduce music during speech (negative dB).
            reverb_amount: Voice reverb wet level (0.0–0.5).
            fade_in_sec: Fade-in duration for the final mix.
            fade_out_sec: Fade-out duration for the final mix.
            output_format: "wav" or "mp3".
            progress_cb: Called with (fraction: float, message: str).

        Returns:
            Path to the exported audio file.
        """
        # ── Step 1: Parse script ────────────────────────────────────────
        _progress(progress_cb, 0.0, "Parsing meditation script...")
        segments = parse_script(script)
        if not segments:
            raise ValueError("Script is empty or contains no content.")

        # ── Step 2: Load TTS ────────────────────────────────────────────
        _progress(progress_cb, 0.05, "Loading voice model...")
        self.tts.load_model()

        # ── Step 3: Synthesize narration ────────────────────────────────
        _progress(progress_cb, 0.10, "Synthesizing narration...")

        def tts_progress(current, total):
            frac = 0.10 + 0.30 * (current / max(total, 1))
            _progress(progress_cb, frac, f"Synthesizing segment {current}/{total}...")

        voice_audio, voice_activity = self.tts.synthesize(
            segments, voice=voice, speed=speed, progress_cb=tts_progress
        )

        # ── Step 4: Unload TTS, load MusicGen ───────────────────────────
        _progress(progress_cb, 0.40, "Switching to music model...")
        self.tts.unload_model()
        self.music.load_model()

        # ── Step 5: Generate background music ───────────────────────────
        voice_duration = len(voice_audio) / SAMPLE_RATE
        music_duration = voice_duration + 10  # extra for pre-roll + fade-out

        _progress(progress_cb, 0.45, "Generating background music...")

        def music_progress(current, total):
            frac = 0.45 + 0.25 * (current / max(total, 1))
            _progress(progress_cb, frac, f"Generating music segment {current}/{total}...")

        music_audio = self.music.generate(
            music_prompt, music_duration, progress_cb=music_progress
        )

        # ── Step 6: Unload MusicGen ─────────────────────────────────────
        self.music.unload_model()

        # ── Step 7: Apply voice FX ──────────────────────────────────────
        _progress(progress_cb, 0.72, "Applying voice effects...")
        voice_chain = make_voice_chain(reverb_amount=reverb_amount)
        voice_audio = apply_fx(voice_audio, voice_chain, SAMPLE_RATE)

        # ── Step 8: Apply music FX ──────────────────────────────────────
        _progress(progress_cb, 0.77, "Applying music effects...")
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

        _progress(progress_cb, 1.0, "Done!")
        return output_path
