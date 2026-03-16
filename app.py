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
    reverb_ir_choice,
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
    # "Studio (SFT / 50-step)" -> "sft"
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
            reverb_ir=reverb_ir_choice,
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


# CSS — Cosmic Minimalism design system
# Inspired by premium dark-mode apps: Linear, Vercel, Raycast, Arc
CUSTOM_CSS = """
/* ── GOOGLE FONTS ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── DESIGN TOKENS ────────────────────────────────────────── */
:root {
    --c-bg:          #040711;
    --c-surface:     rgba(255,255,255,0.04);
    --c-surface-2:   rgba(255,255,255,0.07);
    --c-border:      rgba(255,255,255,0.18);    /* Increased visibility */
    --c-border-mid:  rgba(255,255,255,0.25);    /* Increased visibility */
    --c-violet:      #8B5CF6;
    --c-indigo:      #6366F1;
    --c-cyan:        #22D3EE;
    --c-text-1:      #F8FAFC;    /* Maximum contrast primary */
    --c-text-2:      #CBD5E1;    /* Bright secondary */
    --c-text-3:      #94A3B8;    /* Readable labels */
    --c-text-4:      #64748B;    /* Legible placeholder */
    --r-sm:  10px;
    --r-md:  14px;
    --r-lg:  20px;
    --r-xl:  24px;
}

/* ── BASE ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body { background: var(--c-bg) !important; }

.gradio-container {
    min-height: 100vh !important;
    font-family: 'Plus Jakarta Sans', 'Inter', system-ui, sans-serif !important;
    background:
        radial-gradient(ellipse 70% 50% at 5%  -5%,  rgba(139,92,246,0.15)  0%, transparent 70%),
        radial-gradient(ellipse 60% 45% at 95% 105%, rgba(99,102,241,0.14)   0%, transparent 70%),
        radial-gradient(ellipse 50% 30% at 50%  50%,  rgba(34,211,238,0.04)  0%, transparent 60%),
        var(--c-bg) !important;
}

/* ── HEADER ───────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 4rem 1rem 2.5rem;
}

.app-header::after {
    content: '';
    display: block;
    width: 140px;
    height: 1.5px;
    margin: 2rem auto 0;
    background: linear-gradient(90deg, transparent, var(--c-violet), var(--c-indigo), transparent);
    opacity: 0.6;
}

.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(3rem, 7vw, 4.5rem) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #FFF 0%, #C4B5FD 40%, #818CF8 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.05em !important;
    line-height: 0.9 !important;
    margin-bottom: 1.2rem !important;
}

.app-header p {
    font-size: 1.05rem !important;
    color: var(--c-text-2) !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    max-width: 520px;
    margin: 0 auto !important;
    opacity: 0.9;
}

/* ── GLASS CARDS ──────────────────────────────────────────── */
.glass-panel {
    background: var(--c-surface) !important;
    backdrop-filter: blur(32px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(180%) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-xl) !important;
    padding: 2rem !important;
    box-shadow: 
        inset 0 0.5px 1px rgba(255,255,255,0.15),
        0 20px 50px rgba(0,0,0,0.5) !important;
}

/* ── DROPDOWNS REFINEMENT ─────────────────────────────────── */
/* Targeting the specific Gradio dropdown container and options list */
.wrap.svelte-iyf88w, .wrap.gradio-dropdown {
    background: rgba(13, 18, 33, 0.95) !important;
    border: 1px solid var(--c-border-mid) !important;
    border-radius: var(--r-md) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.7) !important;
}

ul.options {
    background: #0B0F1A !important;
    border: 1px solid var(--c-border-mid) !important;
    border-radius: var(--r-md) !important;
    margin-top: 6px !important;
    padding: 6px !important;
    z-index: 9999 !important;
    box-shadow: 0 15px 45px rgba(0,0,0,0.8) !important;
}

ul.options li {
    border-radius: 8px !important;
    margin: 2px 0 !important;
    padding: 10px 14px !important;
    color: var(--c-text-2) !important;
    font-weight: 500 !important;
}

ul.options li:hover, ul.options li.selected {
    background: rgba(139,92,246,0.2) !important;
    color: #DDD6FE !important;
}

/* ── MODERNIZED PROGRESS BAR ─────────────────────────────── */
/* Make it a sleek aurora line instead of a box */
.progress-bar {
    height: 4px !important;
    background: linear-gradient(90deg, #8B5CF6, #6366F1, #22D3EE, #8B5CF6) !important;
    background-size: 200% 100% !important;
    animation: aurora-move 1.5s linear infinite !important;
    border-radius: 2px !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(139,92,246,0.4) !important;
}

@keyframes aurora-move {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Remove default loading text container if it looks bulky */
.progress-wrap {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── INPUTS & FORM ELEMENTS ───────────────────────────────── */
textarea, input[type="text"], input[type="number"] {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-md) !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    color: var(--c-text-1) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--c-violet) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

/* ── TABS REFINEMENT ──────────────────────────────────────── */
.tab-nav {
    background: transparent !important;
    border-bottom: 2px solid var(--c-border) !important;
    margin-bottom: 1.5rem !important;
}

.tab-nav button {
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--c-text-3) !important;
    padding: 0.75rem 1.5rem !important;
}

.tab-nav button.selected {
    color: var(--c-violet) !important;
    border-bottom: 2px solid var(--c-violet) !important;
}

/* ── SLIDERS ─────────────────────────────────────────────── */
input[type="range"] {
    height: 6px !important;
}
input[type="range"]::-webkit-slider-thumb {
    width: 18px !important;
    height: 18px !important;
    background: #FFF !important;
    border: 3px solid var(--c-violet) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* ── OTHER IMPROVEMENTS ───────────────────────────────────── */
.block-label {
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    color: var(--c-text-3) !important;
    margin-bottom: 8px !important;
}

button:not(.primary-btn) {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ── BLOCK REFINEMENT ────────────────────────────────────── */
.block {
    border-radius: var(--r-md) !important;
    border-color: var(--c-border) !important;
}

/* Flatten nested blocks inside glass panels — prevents double-border box effect */
.glass-panel .block,
.glass-panel .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
"""

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Plus Jakarta Sans"),
        gr.themes.GoogleFont("Space Grotesk"),
        "ui-sans-serif", "system-ui", "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#060B18",
    body_text_color="#E8EDF5",
    body_text_color_subdued="#9AACCB",
    block_background_fill="rgba(255, 255, 255, 0.03)",
    block_border_color="rgba(255,255,255,0.10)",
    block_border_width="1px",
    block_label_text_size="12px",
    block_label_text_weight="600",
    block_label_text_color="#9AACCB",
    block_label_background_fill="transparent",
    block_title_text_weight="600",
    block_title_text_size="12px",
    block_title_text_color="#9AACCB",
    input_background_fill="rgba(6,11,24,0.55)",
    input_border_color="rgba(255,255,255,0.10)",
    input_border_color_focus="rgba(124,58,237,0.5)",
    input_placeholder_color="#4D6480",
    button_primary_background_fill="linear-gradient(135deg, #7C3AED 0%, #4F46E5 55%, #0E7490 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #8B5CF6 0%, #6366F1 55%, #0891B2 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(255,255,255,0.04)",
    button_secondary_border_color="rgba(255,255,255,0.07)",
    button_secondary_text_color="#94A3B8",
).set(
    body_background_fill_dark="#060B18",
    block_background_fill_dark="rgba(255, 255, 255, 0.03)",
    input_background_fill_dark="rgba(6,11,24,0.55)",
)

# ── Build the Gradio UI ────────────────────────────────────────────────────

with gr.Blocks(
    title="MoodScape — Guided Meditation Generator",
    theme=theme,
    css=CUSTOM_CSS,
) as demo:
    
    gr.HTML("""
        <div class="app-header">
            <h1>MoodScape</h1>
            <p>Premium AI-powered guided meditation audio generator.<br>
            Synthesize professional narration and ambient soundscapes in seconds.</p>
        </div>
    """)

    with gr.Row():
        # ── Left column: script and core inputs ────────────────────────────
        with gr.Column(scale=3, elem_classes="glass-panel"):
            with gr.Group():
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
                    elem_id="script-textbox",
                )
            
            with gr.Row():
                music_prompt = gr.Textbox(
                    label="Music Style Prompt",
                    placeholder="E.g. Evolving pads, gentle rain, minimal soft piano...",
                    value=DEFAULT_MUSIC_PROMPT,
                    lines=3,
                    scale=2
                )
                music_duration = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    value=3.0,
                    step=0.5,
                    label="Instrumental Duration (min)",
                    visible=False,
                    scale=1
                )

        # ── Right column: Settings & Configuration ────────────────────────
        with gr.Column(scale=2):
            with gr.Tabs(elem_id="settings-tabs"):
                
                # Tab 1: Core Voice & Music AI Settings
                with gr.TabItem("AI Engines", elem_classes="glass-panel"):
                    gr.Markdown("### Sound Sources")
                    music_model_dropdown = gr.Dropdown(
                        choices=["MusicGen", "ACE-Step 1.5", "Lyria RealTime"],
                        value="MusicGen",
                        label="Background Music Model",
                    )
                    
                    acestep_quality = gr.Radio(
                        choices=["Draft (Turbo / 8-step)", "Studio (SFT / 50-step)"],
                        value="Studio (SFT / 50-step)",
                        label="Generation Quality",
                        visible=False,
                    )

                    tts_engine_radio = gr.Radio(
                        choices=["Kokoro", "F5-TTS"],
                        value="Kokoro",
                        label="TTS Voice Engine",
                    )

                    with gr.Group(visible=True) as kokoro_settings:
                        kokoro_voice_dropdown = gr.Dropdown(
                            choices=KOKORO_VOICE_CHOICES,
                            value="balanced_calm",
                            label="Voice Model",
                        )

                    with gr.Group(visible=False) as f5_settings:
                        f5_voice_dropdown = gr.Dropdown(
                            choices=F5_VOICE_CHOICES if F5_VOICE_CHOICES else ["(no voices)"],
                            value=F5_VOICE_DEFAULT,
                            label="Voice Personality",
                            interactive=bool(F5_VOICE_CHOICES),
                        )

                # Tab 2: Audio Engineering (Ducking, Reverb, etc)
                with gr.TabItem("Mix Details", elem_classes="glass-panel"):
                    gr.Markdown("### Acoustic Parameters")
                    with gr.Row():
                        speed_slider = gr.Slider(0.65, 1.0, 0.70, step=0.01, label="Speech Speed")
                        duck_slider = gr.Slider(-30, -5, -20, step=1, label="Ducking (dB)")
                    
                    with gr.Row():
                        reverb_slider = gr.Slider(0.0, 0.5, 0.15, step=0.05, label="Voice Reverb")
                        reverb_ir_dropdown = gr.Dropdown(
                            choices=[
                                ("Warm Studio", "warm_studio"),
                                ("Wooden Hall", "wooden_hall"),
                                ("Stone Chapel", "stone_chapel"),
                            ],
                            value="warm_studio",
                            label="Space Type",
                        )
                    
                    with gr.Row():
                        fade_in_slider = gr.Slider(0, 10, 3, step=0.5, label="Fade In (s)")
                        fade_out_slider = gr.Slider(0, 15, 6, step=0.5, label="Fade Out (s)")

                # Tab 3: Model-Specific Extras (BPM, Key, Advanced)
                with gr.TabItem("Advanced", elem_classes="glass-panel"):
                    # ACE-Step Metadata
                    with gr.Group(visible=False) as acestep_metadata:
                        gr.Markdown("#### ACE-Step Tuning")
                        with gr.Row():
                            acestep_bpm = gr.Slider(40, 100, 50, step=1, label="BPM")
                            acestep_key = gr.Dropdown(
                                choices=["Auto", "C Major", "C Minor", "C# Major", "C# Minor", "D Major", "D Minor", "Eb Major", "Eb Minor", "E Major", "E Minor", "F Major", "F Minor", "F# Major", "F# Minor", "G Major", "G Minor", "Ab Major", "Ab Minor", "A Major", "A Minor", "Bb Major", "Bb Minor", "B Major", "B Minor"],
                                value="Auto",
                                label="Key",
                            )

                    # Lyria Settings
                    with gr.Group(visible=False) as lyria_settings:
                        gr.Markdown("#### Lyria RealTime Tuning")
                        lyria_bpm = gr.Slider(60, 200, 70, step=1, label="BPM")
                        with gr.Row():
                            lyria_density = gr.Slider(0, 1.0, 0.1, step=0.05, label="Density")
                            lyria_brightness = gr.Slider(0, 1.0, 0.15, step=0.05, label="Brightness")

                    gr.Markdown("#### Audio Export")
                    with gr.Row():
                        format_radio = gr.Radio(["wav", "mp3"], value="wav", label="Format")
                        seed_input = gr.Number(label="Seed", value=0, precision=0)
                    
                    with gr.Row():
                        upsample_checkbox = gr.Checkbox(label="Hi-Fi (48kHz)", value=False)
                        stems_checkbox = gr.Checkbox(label="Export Stems", value=False)
                    
                    stem_separation_checkbox = gr.Checkbox(label="Clean Music (Source Separation)", value=True)
                    reference_audio = gr.Audio(label="Style/Melody Reference", type="filepath", sources=["upload"])

    # Action Section
    with gr.Row(elem_id="action-row"):
        with gr.Column(scale=2):
            generate_btn = gr.Button("Generate Meditation", variant="primary", size="lg", elem_classes="primary-btn")
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="Generated Audio", type="filepath", elem_classes="glass-panel")
    
    status_text = gr.Textbox(label="Status", interactive=False, elem_classes="glass-panel")

    # ── Visibility Callbacks ───────────────────────────────────────────────

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
        outputs=[script_input, music_prompt, music_duration, kokoro_settings, speed_slider, duck_slider, reverb_slider, reference_audio, acestep_quality, acestep_metadata, lyria_settings, f5_settings],
    )

    def toggle_music_engine_ui(model, mode):
        is_acestep = model == "ACE-Step 1.5"
        is_lyria = model == "Lyria RealTime"
        is_voc = mode == "Vocals Only"
        return gr.update(visible=is_acestep and not is_voc), gr.update(visible=is_acestep and not is_voc), gr.update(visible=is_lyria and not is_voc)

    music_model_dropdown.change(
        fn=toggle_music_engine_ui,
        inputs=[music_model_dropdown, generation_mode],
        outputs=[acestep_quality, acestep_metadata, lyria_settings],
    )

    def toggle_tts_engine_ui(tts_engine, mode):
        is_inst = mode == "Instrumental Only"
        show_kokoro = (tts_engine == "Kokoro") and not is_inst
        show_f5 = (tts_engine == "F5-TTS") and not is_inst
        # Set engine-optimal speed default
        if tts_engine == "F5-TTS":
            speed_val = 0.80
            speed_label = "Speaking Speed (0.75-0.85 = F5 meditation ideal)"
        else:
            speed_val = 0.70
            speed_label = "Speaking Speed (0.65-0.75 = meditation ideal)"
        return (
            gr.update(visible=show_kokoro),
            gr.update(visible=show_f5),
            gr.update(value=speed_val, label=speed_label),
        )

    tts_engine_radio.change(
        fn=toggle_tts_engine_ui,
        inputs=[tts_engine_radio, generation_mode],
        outputs=[kokoro_settings, f5_settings, speed_slider],
    )

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
            reverb_ir_dropdown,
        ],
        outputs=[audio_output, status_text],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False)
