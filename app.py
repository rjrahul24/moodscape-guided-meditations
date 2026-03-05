"""MoodScape — Guided Meditation Audio Generator (Gradio UI)."""

import os
import warnings

# Prevent HuggingFace tokenizers from warning about fork-after-parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fix: MPS Autocast on Apple Silicon crashes with float16 on some PyTorch versions
# related to audiocraft/encodec. We force disable it here for stability.
os.environ.setdefault("AUDIOCRAFT_DISABLE_MPS_AUTOCAST", "1")

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

import soundfile as sf
import gradio as gr
from dotenv import load_dotenv

from core.parler_engine import VOICE_PRESETS as PARLER_VOICE_PRESETS
from core.pipeline import MeditationPipeline

# Load environment variables (like HF_TOKEN) from .env file
load_dotenv()

# ── Voice choices for Kokoro engine ──────────────────────────────────────────

# (label, voice_id) pairs shown in the Kokoro dropdown.
# Presets resolve to blended voice tensors via core.voice_manager.
KOKORO_VOICE_CHOICES = [
    # Presets (blended)
    ("Deep Calm — very soft & whispery (default)", "deep_calm"),
    ("Golden Hour — warm meditation blend",        "golden_hour"),
    ("Heart + Nicole blend",                       "af_heart,af_nicole"),
    ("Still Water — ASMR relaxation",              "still_water"),
    ("Night Garden — sleep meditation",            "night_garden"),
    ("Earth Root — grounding blend",               "earth_root"),
    # Individual US voices
    ("Heart — US Female (warm)",                 "af_heart"),
    ("Nicole — US Female (calm/ASMR)",           "af_nicole"),
    ("Sky — US Female (airy)",                   "af_sky"),
    ("Nova — US Female (intimate)",              "af_nova"),
    ("Bella — US Female",                        "af_bella"),
    ("Sarah — US Female",                        "af_sarah"),
    # British voices
    ("Emma — UK Female (wise)",                  "bf_emma"),
    ("Lily — UK Female (angelic)",               "bf_lily"),
    # US Male voices
    ("Adam — US Male (grounding)",               "am_adam"),
    ("Michael — US Male",                        "am_michael"),
    # British Male
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
    script,
    music_prompt,
    engine_choice,
    kokoro_voice,
    parler_preset,
    parler_custom_desc,
    speed,
    duck_amount,
    reverb_amount,
    fade_in,
    fade_out,
    session_mode,
    output_format,
    seed_value,
    export_stems_flag,
    upsample_flag,
    progress=gr.Progress(),
):
    def progress_cb(fraction, message):
        progress(fraction, desc=message)

    # Map radio label to engine key
    tts_engine = "parler" if engine_choice == "Parler TTS" else "kokoro"

    # Resolve seed: 0 means auto
    seed = int(seed_value) if seed_value and int(seed_value) != 0 else None

    try:
        output_path, status_msg = pipeline.generate(
            script=script,
            music_prompt=music_prompt,
            voice=kokoro_voice,
            speed=speed,
            tts_engine=tts_engine,
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
            session_mode=session_mode,
        )
        duration = _get_duration(output_path)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        base_status = f"Duration: {minutes}m {seconds}s"
        if status_msg:
            status = f"Warning: {status_msg}. {base_status}"
        else:
            engine_label = "Parler TTS" if tts_engine == "parler" else "Kokoro"
            status = f"Generated with {engine_label}. {base_status}"
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
        "narration and ambient music. Write your script using `[pause:Xs]` "
        "markers for timed silences."
    )

    with gr.Row():
        # ── Left column: main inputs ───────────────────────────────────
        with gr.Column(scale=2):
            script_input = gr.Textbox(
                label="Meditation Script",
                placeholder=(
                    "Enter your meditation script here.\n"
                    "Use [pause:Xs] for timed pauses, e.g. [pause:5s]\n"
                    "Double newlines create 1.5-second pauses automatically."
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

            # Kokoro settings (visible by default)
            with gr.Accordion("Kokoro Voice Settings", open=False, visible=True) as kokoro_settings:
                kokoro_voice_dropdown = gr.Dropdown(
                    choices=KOKORO_VOICE_CHOICES,
                    value="deep_calm",
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

            # Common settings
            with gr.Accordion("Audio Settings", open=False):
                speed_slider = gr.Slider(
                    minimum=0.50,
                    maximum=1.0,
                    value=0.70,
                    step=0.01,
                    label="Speaking Speed (0.65-0.75 = meditation ideal)",
                )
                duck_slider = gr.Slider(
                    minimum=-20,
                    maximum=-2,
                    value=-4,
                    step=1,
                    label="Music Ducking (dB)",
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
                session_mode_radio = gr.Radio(
                    choices=["Daytime Meditation", "Sleep Journey"],
                    value="Daytime Meditation",
                    label="Session Mode",
                    info=(
                        "Daytime: standard loudness (-16 LUFS). "
                        "Sleep: quieter, softer master (-19 LUFS)."
                    ),
                )

            # Advanced settings (collapsed by default)
            with gr.Accordion("Advanced Settings", open=False):
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

    # Toggle visibility of engine-specific settings
    def toggle_engine_settings(engine_choice):
        is_kokoro = engine_choice == "Kokoro TTS"
        return (
            gr.update(visible=is_kokoro),       # kokoro_settings
            gr.update(visible=not is_kokoro),    # parler_settings
        )

    engine_radio.change(
        fn=toggle_engine_settings,
        inputs=[engine_radio],
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
            script_input,
            music_prompt,
            engine_radio,
            kokoro_voice_dropdown,
            parler_preset_dropdown,
            parler_custom_textbox,
            speed_slider,
            duck_slider,
            reverb_slider,
            fade_in_slider,
            fade_out_slider,
            session_mode_radio,
            format_radio,
            seed_input,
            stems_checkbox,
            upsample_checkbox,
        ],
        outputs=[audio_output, status_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
