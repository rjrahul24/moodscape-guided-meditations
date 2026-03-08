"""MoodScape — Guided Meditation Audio Generator (Gradio UI)."""

import os
import warnings

# Prevent HuggingFace tokenizers from warning about fork-after-parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fix: MPS Autocast on Apple Silicon crashes with float16 on some PyTorch versions
# related to audiocraft/encodec. We force disable it here for stability.
os.environ.setdefault("AUDIOCRAFT_DISABLE_MPS_AUTOCAST", "1")

# Fix: MacOS Bus Error (SIGBUS) when Python garbage collects MPS tensors on process exit.
# We disable fallback to prevent unsupported ops from quietly failing, and register
# a hard exit hook to bypass the problematic PyTorch C++ destructor sequence.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import atexit
atexit.register(lambda: os._exit(0))

# Silence noisy third-party warnings that are not actionable from user code.
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
    module="transformers",
)
warnings.filterwarnings(
    "ignore",
    message="Possible clipped samples in output",
    category=UserWarning,
    module="pyloudnorm",
)

import numpy as np
import soundfile as sf
import gradio as gr
from dotenv import load_dotenv

# Compatibility shim: transformers >=4.47 renamed/removed SlidingWindowCache.
# parler_tts (pinned to transformers==4.46.1 API) imports it at module level.
# We backfill the name with StaticCache so the import succeeds.  parler_tts
# only uses SlidingWindowCache when cache_implementation=="sliding_window",
# a path never taken in our inference calls, so StaticCache is a safe alias.
try:
    from transformers.cache_utils import SlidingWindowCache as _swc  # noqa: F401
except ImportError:
    import transformers.cache_utils as _tcu
    from transformers.cache_utils import StaticCache
    _tcu.SlidingWindowCache = StaticCache

from core.kokoro_tts.engine import KokoroEngine
from core.parler_tts.engine import VOICE_PRESETS as PARLER_VOICE_PRESETS
from core.pipeline import MeditationPipeline

# Load environment variables (like HF_TOKEN) from .env file
load_dotenv()

# ── Voice choices for Kokoro engine ──────────────────────────────────────────

# (label, voice_id) pairs shown in the Kokoro dropdown.
# Presets resolve to blended voice tensors via core.kokoro_tts.voice_manager.
KOKORO_VOICE_CHOICES = [
    # Premium Blends (Recommended)
    ("Balanced Calm — natural & human (default)",  "balanced_calm"),
    ("Deep Rest — intimate & breathy",             "deep_rest"),
    ("Soft Whisper — ASMR relaxation",             "soft_whisper"),
    ("Golden Hour — warm & airy",                  "golden_hour"),
    ("Earth Root — grounding blend",               "earth_root"),
    # High-Quality Individual Voices
    ("Heart — US Female (warm)",                 "af_heart"),
    ("Nicole — US Female (calm/ASMR)",           "af_nicole"),
    # British & Male Voices
    ("Emma — UK Female (wise)",                  "bf_emma"),
    ("Adam — US Male (grounding)",               "am_adam"),
    ("George — UK Male (warm)",                  "bm_george"),
]

# Parler TTS preset labels for the dropdown
PARLER_PRESET_CHOICES = [label for label, _ in PARLER_VOICE_PRESETS]

pipeline = MeditationPipeline()

DEFAULT_SCRIPT = """\
Welcome to this guided meditation. [pause:3s]

Find a comfortable position... and gently close your eyes. [pause:5s]

Take a slow, deep breath in... [pause:4s] and release. [pause:6s]

Notice the sensations in your body. [pause:8s]

With each breath, allow yourself to sink deeper into relaxation. [pause:5s]

There is nowhere else you need to be. Nothing else you need to do. [pause:6s]

Simply be here, in this moment. [pause:10s]

When you are ready, gently bring your awareness back to the room. [pause:5s]

Slowly open your eyes. Thank you for practicing with me today. [pause:3s]\
"""

DEFAULT_MUSIC_PROMPT = (
    "Soft felt piano, warm analog synth pads, gentle singing bowls, "
    "sustained strings, slow and evolving, spacious atmosphere"
)


def _get_duration(path: str) -> float:
    info = sf.info(path)
    return info.duration


def generate_meditation(
    generation_mode,
    script,
    music_prompt,
    music_duration,
    engine_choice,
    music_model_choice,
    kokoro_voice,
    parler_preset,
    parler_custom_desc,
    speed,
    duck_amount,
    reverb_amount,
    fade_in,
    fade_out,
    output_format,
    seed_value,
    export_stems_flag,
    upsample_flag,
    stem_separation_flag,
    reference_audio_file,
    acestep_bpm,
    acestep_key,
    progress=gr.Progress(),
):
    def progress_cb(fraction, message):
        progress(fraction, desc=message)

    # Map radio label to engine key
    tts_engine = "parler" if engine_choice == "Parler TTS" else "kokoro"

    # Map music model label to engine key
    music_model = "acestep" if music_model_choice == "ACE-Step 1.5" else "musicgen"

    # Resolve seed: 0 means auto
    seed = int(seed_value) if seed_value and int(seed_value) != 0 else None

    # Load reference audio for melody conditioning if provided
    melody_audio = None
    melody_sample_rate = None
    if reference_audio_file is not None:
        try:
            audio_data, sr = sf.read(reference_audio_file)
            audio_data = audio_data.astype(np.float32)
            # Convert stereo to mono if needed
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            melody_audio = audio_data
            melody_sample_rate = sr
        except Exception as e:
            print(f"[App] Failed to load reference audio: {e}")

    try:
        output_path, status_msg = pipeline.generate(
            generation_mode=generation_mode,
            script=script,
            music_prompt=music_prompt,
            instrumental_duration_m=music_duration,
            voice=kokoro_voice,
            speed=speed,
            tts_engine=tts_engine,
            music_model=music_model,
            parler_voice_preset=parler_preset,
            parler_custom_description=parler_custom_desc,
            duck_amount_db=duck_amount,
            reverb_amount=reverb_amount,
            fade_in_sec=fade_in,
            fade_out_sec=fade_out,
            output_format=output_format,
            progress_cb=progress_cb,
            seed=seed,
            do_export_stems=export_stems_flag,
            upsample_48k=upsample_flag,
            stem_separation=stem_separation_flag,
            melody_audio=melody_audio,
            melody_sample_rate=melody_sample_rate,
            bpm=acestep_bpm,
            keyscale=acestep_key,
        )
        duration = _get_duration(output_path)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        base_status = f"Duration: {minutes}m {seconds}s"
        if status_msg:
            status = f"Warning: {status_msg}. {base_status}"
        else:
            engine_label = "Parler TTS" if tts_engine == "parler" else "Kokoro"
            music_label = music_model_choice
            status = f"Generated with {engine_label} + {music_label}. {base_status}"
        return output_path, status
    except Exception as e:
        return None, f"Error: {e}"


# ── Build the Gradio UI ────────────────────────────────────────────────────

with gr.Blocks(
    title="MoodScape — Guided Meditation Generator",
) as demo:
    gr.Markdown("# MoodScape — Guided Meditation Audio Generator")
    gr.Markdown(
        "Create personalized guided meditation audio with AI-generated "
        "narration and ambient music. Write your script using `[pause:Xs]` or "
        "`[N second pause]` markers for timed silences, and `[breath]` for breath pauses."
    )

    with gr.Row():
        # ── Left column: main inputs ───────────────────────────────────
        with gr.Column(scale=2):
            generation_mode = gr.Radio(
                choices=["Instrumental Only", "Vocals Only", "Instrumental + Vocal"],
                value="Instrumental + Vocal",
                label="Generation Mode",
            )
            script_input = gr.Textbox(
                label="Meditation Script",
                placeholder=(
                    "Enter your meditation script here.\n"
                    "Use [pause:Xs] or [N second pause] for timed pauses, e.g. [pause:5s] or [5 second pause]\n"
                    "Use [breath] / [inhale] / [exhale] for a 1.2s breath pause.\n"
                    "Double newlines create 6.5-second pauses automatically."
                ),
                value=DEFAULT_SCRIPT,
                lines=15,
            )
            music_prompt = gr.Textbox(
                label="Music Style Prompt",
                placeholder="Describe the background music you'd like...",
                value=DEFAULT_MUSIC_PROMPT,
                lines=3,
            )
            music_duration = gr.Slider(
                minimum=1.0,
                maximum=30.0,
                value=3.0,
                step=0.5,
                label="Instrumental Duration (minutes)",
                info="Only applies to 'Instrumental Only' mode.",
                visible=False,
            )
            reference_audio = gr.Audio(
                label="Reference Audio (Melody / Acoustic Style — ACE-Step: timbre, MusicGen: melody)",
                type="filepath",
                sources=["upload"],
            )

        # ── Right column: settings ─────────────────────────────────────
        with gr.Column(scale=1):
            # TTS Engine selector
            engine_radio = gr.Radio(
                choices=["Kokoro TTS", "Parler TTS"],
                value="Kokoro TTS",
                label="TTS Engine",
                info=(
                    "Kokoro: fast, lightweight, preset voices. "
                    "Parler: slower, richer, description-controlled voices."
                ),
            )

            # Background Music Model selector
            music_model_dropdown = gr.Dropdown(
                choices=["MusicGen", "ACE-Step 1.5"],
                value="MusicGen",
                label="Background Music Model",
                info=(
                    "MusicGen: Meta's established model, reliable ambient generation. "
                    "ACE-Step 1.5: newer DiT model, 48kHz native, coherent long-form via MLX."
                ),
            )

            # Kokoro settings (visible by default)
            with gr.Accordion("Kokoro Voice Settings", open=False, visible=True) as kokoro_settings:
                kokoro_voice_dropdown = gr.Dropdown(
                    choices=KOKORO_VOICE_CHOICES,
                    value="balanced_calm",
                    label="Voice",
                )

            # Parler TTS settings (hidden by default)
            with gr.Accordion("Parler TTS Settings", open=False, visible=False) as parler_settings:
                parler_preset_dropdown = gr.Dropdown(
                    choices=PARLER_PRESET_CHOICES,
                    value=PARLER_PRESET_CHOICES[0],
                    label="Voice Style",
                    info="Select a meditation-optimized voice preset, or choose 'Custom Description' to write your own.",
                )
                parler_custom_textbox = gr.Textbox(
                    label="Custom Voice Description",
                    placeholder=(
                        "Example: A warm, low female voice with slow, soothing delivery, "
                        "breathy tone, and crystal clear studio recording with no background noise."
                    ),
                    lines=3,
                    visible=False,
                    info=(
                        "Describe the voice you want. Key terms: warm/cold, breathy/clear, "
                        "slow/fast, low/high pitch, male/female, close-mic/distant, "
                        "reverb/no reverb. Use named speakers (Jon, Lea) for consistency."
                    ),
                )
 
            # ACE-Step Metadata (Collapsed by default)
            with gr.Accordion("ACE-Step Metadata (BPM / Key)", open=False, visible=False) as acestep_metadata:
                acestep_bpm = gr.Slider(
                    minimum=60,
                    maximum=100,
                    value=70,
                    step=1,
                    label="BPM (Beats Per Minute)",
                    info="ACE-Step only. 60-80 is ideal for meditation.",
                )
                acestep_key = gr.Dropdown(
                    choices=["Auto", "C Major", "C Minor", "C# Major", "C# Minor", "D Major", "D Minor", "Eb Major", "Eb Minor", "E Major", "E Minor", "F Major", "F Minor", "F# Major", "F# Minor", "G Major", "G Minor", "Ab Major", "Ab Minor", "A Major", "A Minor", "Bb Major", "Bb Minor", "B Major", "B Minor"],
                    value="Auto",
                    label="Musical Key",
                    info="ACE-Step only. Set to 'Auto' to let the model choose.",
                )

            # Common settings
            with gr.Accordion("Audio Settings", open=False):
                speed_slider = gr.Slider(
                    minimum=0.50,
                    maximum=1.0,
                    value=0.68,
                    step=0.01,
                    label="Speaking Speed (0.65-0.75 = meditation ideal)",
                )
                duck_slider = gr.Slider(
                    minimum=-30,
                    maximum=-5,
                    value=-20,
                    step=1,
                    label="Music Ducking During Speech (dB)",
                )
                reverb_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.15,
                    step=0.05,
                    label="Voice Reverb",
                )
                fade_in_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=3,
                    step=0.5,
                    label="Fade In (seconds)",
                )
                fade_out_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=0.5,
                    label="Fade Out (seconds)",
                )


            # Advanced settings (collapsed by default)
            with gr.Accordion("Advanced Settings", open=False):
                stem_separation_checkbox = gr.Checkbox(
                    label="AI Source Separation (remove drums/vocals from music)",
                    value=True,
                    info=(
                        "Runs HT Demucs to strip any unwanted drums or vocal artefacts "
                        "from generated music. Recommended for pure ambient output."
                    ),
                )
                seed_input = gr.Number(
                    label="Random Seed (0 = auto)",
                    value=0,
                    precision=0,
                )
                stems_checkbox = gr.Checkbox(
                    label="Export separate voice/music stems",
                    value=False,
                )
                upsample_checkbox = gr.Checkbox(
                    label="48 kHz output (higher fidelity, slower export)",
                    value=False,
                )

            format_radio = gr.Radio(
                choices=["wav", "mp3"],
                value="wav",
                label="Output Format",
            )

    # Toggle visibility of generation mode specific settings
    def toggle_mode_settings(mode, current_engine):
        is_inst = mode == "Instrumental Only"
        is_voc = mode == "Vocals Only"

        show_kokoro = (not is_inst) and (current_engine == "Kokoro TTS")
        show_parler = (not is_inst) and (current_engine == "Parler TTS")

        return (
            gr.update(visible=not is_inst), # script_input
            gr.update(visible=not is_voc),  # music_prompt
            gr.update(visible=is_inst),     # music_duration
            gr.update(visible=not is_inst), # engine_radio
            gr.update(visible=not is_voc),  # music_model_dropdown
            gr.update(visible=show_kokoro), # kokoro_settings
            gr.update(visible=show_parler), # parler_settings
            gr.update(visible=not is_inst), # speed_slider
            gr.update(visible=not is_voc),  # duck_slider
            gr.update(visible=not is_inst), # reverb_slider
            gr.update(visible=not is_voc),  # reference_audio
            gr.update(visible=(music_model_dropdown.value == "ACE-Step 1.5") and not is_voc), # acestep_metadata
        )

    generation_mode.change(
        fn=toggle_mode_settings,
        inputs=[generation_mode, engine_radio],
        outputs=[script_input, music_prompt, music_duration, engine_radio, music_model_dropdown, kokoro_settings, parler_settings, speed_slider, duck_slider, reverb_slider, reference_audio, acestep_metadata],
    )

    # Toggle visibility of engine-specific settings
    def toggle_engine_settings(engine_choice, mode):
        is_inst = mode == "Instrumental Only"
        is_kokoro = engine_choice == "Kokoro TTS"
        return (
            gr.update(visible=is_kokoro and not is_inst),       # kokoro_settings
            gr.update(visible=not is_kokoro and not is_inst),    # parler_settings
        )

    # Toggle ACE-Step specific metadata accordion
    def toggle_acestep_ui(model, mode):
        is_acestep = model == "ACE-Step 1.5"
        is_voc = mode == "Vocals Only"
        return gr.update(visible=is_acestep and not is_voc)

    music_model_dropdown.change(
        fn=toggle_acestep_ui,
        inputs=[music_model_dropdown, generation_mode],
        outputs=[acestep_metadata],
    )

    engine_radio.change(
        fn=toggle_engine_settings,
        inputs=[engine_radio, generation_mode],
        outputs=[kokoro_settings, parler_settings],
    )

    # Show/hide custom description textbox based on preset selection
    def toggle_custom_description(preset):
        return gr.update(visible=(preset == "Custom Description"))

    parler_preset_dropdown.change(
        fn=toggle_custom_description,
        inputs=[parler_preset_dropdown],
        outputs=[parler_custom_textbox],
    )

    generate_btn = gr.Button("Generate Meditation", variant="primary", size="lg")

    with gr.Row():
        audio_output = gr.Audio(label="Generated Meditation", type="filepath")

    status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_meditation,
        inputs=[
            generation_mode,
            script_input,
            music_prompt,
            music_duration,
            engine_radio,
            music_model_dropdown,
            kokoro_voice_dropdown,
            parler_preset_dropdown,
            parler_custom_textbox,
            speed_slider,
            duck_slider,
            reverb_slider,
            fade_in_slider,
            fade_out_slider,
            format_radio,
            seed_input,
            stems_checkbox,
            upsample_checkbox,
            stem_separation_checkbox,
            reference_audio,
            acestep_bpm,
            acestep_key,
        ],
        outputs=[audio_output, status_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
