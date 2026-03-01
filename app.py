"""MoodScape — Guided Meditation Audio Generator (Gradio UI)."""

import soundfile as sf
import gradio as gr
from dotenv import load_dotenv

from core.pipeline import MeditationPipeline

# Load environment variables (like HF_TOKEN) from .env file
load_dotenv()

# (label, voice_id) pairs shown in the dropdown.
# Comma-separated IDs create blended voices in Kokoro.
VOICE_CHOICES = [
    ("Heart + Nicole blend (meditation)", "af_heart,af_nicole"),
    ("Heart — Female",                    "af_heart"),
    ("Nicole — Female (calm/ASMR)",       "af_nicole"),
    ("Bella — Female",                    "af_bella"),
    ("Sarah — Female",                    "af_sarah"),
    ("Sky — Female",                      "af_sky"),
    ("Nova — Female (intimate)",          "af_nova"),
    ("Adam — Male",                       "am_adam"),
    ("Michael — Male",                    "am_michael"),
]

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
    "Slow meditation music, 60 BPM, soft piano and singing bowls, "
    "gentle strings, light melody, "
    "no drums, no vocals, no electronic beats, peaceful and calming"
)


def _get_duration(path: str) -> float:
    info = sf.info(path)
    return info.duration


def generate_meditation(
    script,
    music_prompt,
    voice,
    speed,
    duck_amount,
    reverb_amount,
    fade_in,
    fade_out,
    output_format,
    progress=gr.Progress(),
):
    def progress_cb(fraction, message):
        progress(fraction, desc=message)

    try:
        output_path = pipeline.generate(
            script=script,
            music_prompt=music_prompt,
            voice=voice,
            speed=speed,
            duck_amount_db=duck_amount,
            reverb_amount=reverb_amount,
            fade_in_sec=fade_in,
            fade_out_sec=fade_out,
            output_format=output_format,
            progress_cb=progress_cb,
        )
        duration = _get_duration(output_path)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        status = f"Meditation generated successfully! Duration: {minutes}m {seconds}s"
        return output_path, status
    except Exception as e:
        return None, f"Error: {e}"


# ── Build the Gradio UI ────────────────────────────────────────────────────

with gr.Blocks(
    title="MoodScape — Guided Meditation Generator",
    theme=gr.themes.Soft(),
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
            with gr.Accordion("Voice Settings", open=False):
                voice_dropdown = gr.Dropdown(
                    choices=VOICE_CHOICES,
                    value="af_heart,af_nicole",
                    label="Voice",
                )
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.80,
                    step=0.05,
                    label="Speaking Speed",
                )

            with gr.Accordion("Audio Settings", open=False):
                duck_slider = gr.Slider(
                    minimum=-20,
                    maximum=-4,
                    value=-10,
                    step=1,
                    label="Music Ducking (dB)",
                )
                reverb_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.12,
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

            format_radio = gr.Radio(
                choices=["wav", "mp3"],
                value="wav",
                label="Output Format",
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
            voice_dropdown,
            speed_slider,
            duck_slider,
            reverb_slider,
            fade_in_slider,
            fade_out_slider,
            format_radio,
        ],
        outputs=[audio_output, status_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False)
