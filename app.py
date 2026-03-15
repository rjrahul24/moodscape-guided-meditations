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

from core.kokoro_tts.engine import KokoroEngine
from core.pipeline import MeditationPipeline
from core.f5_tts import voice_registry as _f5_registry

# Load environment variables (like HF_TOKEN) from .env file
load_dotenv()

# ── F5-TTS voice registry (scanned at startup) ───────────────────────────────

_F5_VOICE_REGISTRY = _f5_registry.scan()
# Slug list for the dropdown; sorted alphabetically for consistent ordering.
F5_VOICE_SLUGS = sorted(_F5_VOICE_REGISTRY.keys())
# (label, value) pairs — label is human-readable, value is the slug for the pipeline.
F5_VOICE_CHOICES = [(slug.replace("_", " ").title(), slug) for slug in F5_VOICE_SLUGS]
F5_VOICE_DEFAULT = F5_VOICE_CHOICES[0][1] if F5_VOICE_CHOICES else None

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
    "Warm evolving synthesizer pads, gentle drone textures, soft sustained tones, "
    "spacious floating atmosphere, peaceful calm new age instrumental"
)


def _get_duration(path: str) -> float:
    info = sf.info(path)
    return info.duration


def generate_meditation(
    generation_mode,
    script,
    music_prompt,
    music_duration,
    music_model_choice,
    kokoro_voice,
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
    acestep_quality,
    acestep_bpm,
    acestep_key,
    lyria_bpm,
    lyria_density,
    lyria_brightness,
    tts_engine_choice,
    f5_voice_slug,
    progress=gr.Progress(),
):
    def progress_cb(fraction, message):
        progress(fraction, desc=message)

    # Map music model label to engine key
    if music_model_choice == "ACE-Step 1.5":
        music_model = "acestep"
    elif music_model_choice == "Lyria RealTime":
        music_model = "lyria"
        # Validate that the API key is available before launching the pipeline
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            return None, (
                "Error: GOOGLE_API_KEY is not set. "
                "Add GOOGLE_API_KEY=<your_key> to the .env file at the project root "
                "and restart the app."
            )
    else:
        music_model = "musicgen"

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

    # Map quality label to engine key (Task 5)
    # "Draft (Turbo / 8-step)" -> "turbo"
    # "Studio (SFT / 60-step)" -> "sft"
    acestep_model_type = "turbo" if "Turbo" in acestep_quality else "sft"

    # Map TTS engine label to key
    tts_engine = "f5" if tts_engine_choice == "F5-TTS" else "kokoro"

    try:
        output_path, status_msg = pipeline.generate(
            generation_mode=generation_mode,
            script=script,
            music_prompt=music_prompt,
            instrumental_duration_m=music_duration,
            voice=kokoro_voice,
            speed=speed,
            music_model=music_model,
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
            acestep_model_type=acestep_model_type,
            lyria_bpm=int(lyria_bpm),
            lyria_density=float(lyria_density),
            lyria_brightness=float(lyria_brightness),
            tts_engine=tts_engine,
            f5_voice_slug=f5_voice_slug if tts_engine == "f5" else None,
        )
        duration = _get_duration(output_path)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        base_status = f"Duration: {minutes}m {seconds}s"
        if status_msg:
            status = f"Warning: {status_msg}. {base_status}"
        else:
            music_label = music_model_choice
            lyria_suffix = f" (BPM {lyria_bpm}, density {lyria_density:.2f})" if music_model == "lyria" else ""
            tts_label = "F5-TTS" if tts_engine == "f5" else "Kokoro"
            status = f"Generated with {tts_label} + {music_label}{lyria_suffix}. {base_status}"
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
            # Background Music Model selector
            music_model_dropdown = gr.Dropdown(
                choices=["MusicGen", "ACE-Step 1.5", "Lyria RealTime"],
                value="MusicGen",
                label="Background Music Model",
                info=(
                    "MusicGen: Meta's established model, reliable ambient generation. "
                    "ACE-Step 1.5: newer DiT model, 48kHz native, coherent long-form via MLX. "
                    "Lyria RealTime: Google's cloud API, native 48kHz stereo, no local VRAM (requires GOOGLE_API_KEY in .env)."
                ),
            )

            # ACE-Step Quality Selector (Visible only when ACE-Step is chosen)
            acestep_quality = gr.Radio(
                choices=["Draft (Turbo / 8-step)", "Studio (SFT / 60-step)"],
                value="Studio (SFT / 60-step)",
                label="Generation Quality",
                info="Studio (default): highest fidelity. Draft: fast preview, lower detail.",
                visible=False,
            )

            # TTS Engine selector
            tts_engine_radio = gr.Radio(
                choices=["Kokoro", "F5-TTS"],
                value="Kokoro",
                label="TTS Voice Engine",
                info=(
                    "Kokoro: 10 curated voices with meditation presets. "
                    "F5-TTS: zero-shot voice cloning — upload a 10–12s reference clip to use any voice."
                ),
            )

            # Kokoro settings (visible by default, hidden when F5-TTS is selected)
            with gr.Accordion("Kokoro Voice Settings", open=False, visible=True) as kokoro_settings:
                kokoro_voice_dropdown = gr.Dropdown(
                    choices=KOKORO_VOICE_CHOICES,
                    value="balanced_calm",
                    label="Voice",
                )

            # F5-TTS settings (hidden by default, shown when F5-TTS engine is selected)
            with gr.Accordion("F5-TTS Voice Settings", open=True, visible=False) as f5_settings:
                f5_voice_dropdown = gr.Dropdown(
                    choices=F5_VOICE_CHOICES if F5_VOICE_CHOICES else ["(no voices registered)"],
                    value=F5_VOICE_DEFAULT,
                    label="Voice Personality",
                    interactive=bool(F5_VOICE_CHOICES),
                    info=(
                        "Select a registered voice personality. "
                        "To add voices, place matching .wav and .txt pairs in "
                        "core/f5_tts/assets/reference_audio/ and reference_transcript/."
                    ),
                )

            # ACE-Step Metadata (Collapsed by default)
            with gr.Accordion("ACE-Step Metadata (BPM / Key)", open=False, visible=False) as acestep_metadata:
                acestep_bpm = gr.Slider(
                    minimum=40,
                    maximum=100,
                    value=50,
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

            # Lyria RealTime Settings (Collapsed, hidden by default)
            with gr.Accordion("Lyria RealTime Settings", open=False, visible=False) as lyria_settings:
                lyria_bpm = gr.Slider(
                    minimum=60,
                    maximum=200,
                    value=70,
                    step=1,
                    label="BPM (Beats Per Minute)",
                    info="60–80 is ideal for meditation. Higher values create more energetic textures.",
                )
                lyria_density = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Density",
                    info="Musical density (0.0 = sparse and minimal, 1.0 = rich and layered). Low values suit meditation.",
                )
                lyria_brightness = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.05,
                    label="Brightness",
                    info="Spectral brightness (0.0 = warm/dark, 1.0 = bright/airy). Keep low for a calming, warm feel.",
                )

            # Common settings
            with gr.Accordion("Audio Settings", open=False):
                speed_slider = gr.Slider(
                    minimum=0.65,
                    maximum=1.0,
                    value=0.70,
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
    def toggle_mode_settings(mode, current_music_model, current_tts_engine):
        is_inst = mode == "Instrumental Only"
        is_voc = mode == "Vocals Only"

        show_acestep = (current_music_model == "ACE-Step 1.5") and not is_voc
        show_lyria = (current_music_model == "Lyria RealTime") and not is_voc
        show_kokoro = (current_tts_engine == "Kokoro") and not is_inst
        show_f5 = (current_tts_engine == "F5-TTS") and not is_inst

        return (
            gr.update(visible=not is_inst),   # script_input
            gr.update(visible=not is_voc),    # music_prompt
            gr.update(visible=is_inst),       # music_duration
            gr.update(visible=not is_voc),    # music_model_dropdown
            gr.update(visible=show_kokoro),   # kokoro_settings
            gr.update(visible=not is_inst),   # speed_slider
            gr.update(visible=not is_voc),    # duck_slider
            gr.update(visible=not is_inst),   # reverb_slider
            gr.update(visible=not is_voc),    # reference_audio
            gr.update(visible=show_acestep),  # acestep_quality
            gr.update(visible=show_acestep),  # acestep_metadata
            gr.update(visible=show_lyria),    # lyria_settings
            gr.update(visible=show_f5),       # f5_settings
        )

    generation_mode.change(
        fn=toggle_mode_settings,
        inputs=[generation_mode, music_model_dropdown, tts_engine_radio],
        outputs=[script_input, music_prompt, music_duration, music_model_dropdown, kokoro_settings, speed_slider, duck_slider, reverb_slider, reference_audio, acestep_quality, acestep_metadata, lyria_settings, f5_settings],
    )

    # Toggle engine-specific settings accordions for ACE-Step and Lyria
    def toggle_music_engine_ui(model, mode):
        is_acestep = model == "ACE-Step 1.5"
        is_lyria = model == "Lyria RealTime"
        is_voc = mode == "Vocals Only"
        return (
            gr.update(visible=is_acestep and not is_voc),  # acestep_quality
            gr.update(visible=is_acestep and not is_voc),  # acestep_metadata
            gr.update(visible=is_lyria and not is_voc),    # lyria_settings
        )

    music_model_dropdown.change(
        fn=toggle_music_engine_ui,
        inputs=[music_model_dropdown, generation_mode],
        outputs=[acestep_quality, acestep_metadata, lyria_settings],
    )

    # Toggle kokoro/f5 settings accordions when TTS engine radio changes
    def toggle_tts_engine_ui(tts_engine, mode):
        is_inst = mode == "Instrumental Only"
        show_kokoro = (tts_engine == "Kokoro") and not is_inst
        show_f5 = (tts_engine == "F5-TTS") and not is_inst
        return gr.update(visible=show_kokoro), gr.update(visible=show_f5)

    tts_engine_radio.change(
        fn=toggle_tts_engine_ui,
        inputs=[tts_engine_radio, generation_mode],
        outputs=[kokoro_settings, f5_settings],
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
            music_model_dropdown,
            kokoro_voice_dropdown,
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
            acestep_quality,
            acestep_bpm,
            acestep_key,
            lyria_bpm,
            lyria_density,
            lyria_brightness,
            tts_engine_radio,
            f5_voice_dropdown,
        ],
        outputs=[audio_output, status_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
