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
    --c-bg:          #060B18;
    --c-surface:     rgba(255,255,255,0.035);
    --c-surface-2:   rgba(255,255,255,0.06);
    --c-border:      rgba(255,255,255,0.10);
    --c-border-mid:  rgba(255,255,255,0.16);
    --c-violet:      #7C3AED;
    --c-indigo:      #4F46E5;
    --c-cyan:        #06B6D4;
    --c-text-1:      #E8EDF5;    /* primary — bright enough to read comfortably */
    --c-text-2:      #9AACCB;    /* secondary — readable but clearly subordinate */
    --c-text-3:      #6B86A8;    /* muted — labels, section headers (≥4.5:1 contrast) */
    --c-text-4:      #4D6480;    /* placeholder — subtle but still legible */
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
    /* Multi-point aurora gradient — very subtle, gives depth */
    background:
        radial-gradient(ellipse 90% 55% at 8%  -5%,  rgba(124,58,237,0.13)  0%, transparent 65%),
        radial-gradient(ellipse 70% 50% at 92% 105%,  rgba(79,70,229,0.11)   0%, transparent 65%),
        radial-gradient(ellipse 55% 35% at 55%  55%,  rgba(6,182,212,0.035)  0%, transparent 55%),
        var(--c-bg) !important;
}

/* ── HEADER ───────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}

/* Thin gradient divider below header */
.app-header::after {
    content: '';
    display: block;
    width: 180px;
    height: 1px;
    margin: 1.75rem auto 0;
    background: linear-gradient(90deg,
        transparent       0%,
        rgba(124,58,237,0.55) 35%,
        rgba(79,70,229,0.55)  65%,
        transparent       100%);
}

.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 4.2rem) !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #C4B5FD 0%, #818CF8 45%, #67E8F9 100%);
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    letter-spacing: -0.04em !important;
    line-height: 1 !important;
    margin: 0 0 0.9rem !important;
    padding: 0 !important;
}

/* Subtitle under the h1 */
.app-header .prose p, .app-header > * p, .app-header p {
    font-size: 0.975rem !important;
    color: var(--c-text-2) !important;
    font-weight: 400 !important;
    letter-spacing: 0.015em !important;
    line-height: 1.65 !important;
    max-width: 500px;
    margin: 0 auto !important;
}

/* ── GLASS CARDS ──────────────────────────────────────────── */
.glass-panel {
    background: var(--c-surface) !important;
    backdrop-filter: blur(28px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(160%) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-xl) !important;
    padding: 1.75rem !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.065),
        0 24px 64px rgba(0,0,0,0.45),
        0 0 0 0.5px rgba(0,0,0,0.2) !important;
    transition: box-shadow 0.35s ease !important;
}

.glass-panel:focus-within {
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 28px 72px rgba(0,0,0,0.5),
        0 0 40px rgba(124,58,237,0.06) !important;
}

/* ── INPUT LABELS ─────────────────────────────────────────── */
label > span:first-child,
.label-wrap span {
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: var(--c-text-2) !important;
}

/* ── MARKDOWN SECTION HEADINGS ────────────────────────────── */
.prose h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: var(--c-text-3) !important;
    margin: 0.25rem 0 1rem !important;
    padding-bottom: 0.6rem !important;
    border-bottom: 1px solid var(--c-border) !important;
}

.prose h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--c-text-3) !important;
    margin: 1.25rem 0 0.7rem !important;
}

/* ── TEXT INPUTS & TEXTAREAS ──────────────────────────────── */
textarea,
input[type="text"],
input[type="number"] {
    background: rgba(6,11,24,0.65) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-md) !important;
    color: var(--c-text-1) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

textarea:focus,
input[type="text"]:focus,
input[type="number"]:focus {
    border-color: rgba(124,58,237,0.45) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.09) !important;
    outline: none !important;
}

textarea::placeholder { color: var(--c-text-4) !important; }

/* ── DROPDOWNS ────────────────────────────────────────────── */
.wrap.svelte-iyf88w,
ul.options {
    background: #0C1221 !important;
    border: 1px solid var(--c-border-mid) !important;
    border-radius: var(--r-md) !important;
    box-shadow: 0 16px 40px rgba(0,0,0,0.6) !important;
}

ul.options li {
    color: var(--c-text-2) !important;
    font-size: 0.875rem !important;
    transition: background 0.12s ease !important;
}

ul.options li:hover,
ul.options li.selected {
    background: rgba(124,58,237,0.14) !important;
    color: #C4B5FD !important;
}

/* ── SLIDERS ──────────────────────────────────────────────── */
input[type="range"] {
    accent-color: var(--c-violet) !important;
    cursor: pointer !important;
}

/* Slider value number input */
.output-number input {
    background: rgba(124,58,237,0.09) !important;
    border: 1px solid rgba(124,58,237,0.22) !important;
    border-radius: 7px !important;
    color: #A78BFA !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── RADIO & CHECKBOX ─────────────────────────────────────── */
input[type="radio"],
input[type="checkbox"] {
    accent-color: var(--c-violet) !important;
    cursor: pointer !important;
}

/* ── TABS ─────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: rgba(0,0,0,0.28) !important;
    border-bottom: 1px solid var(--c-border) !important;
    border-radius: var(--r-lg) var(--r-lg) 0 0 !important;
    padding: 0.35rem 0.35rem 0 !important;
    gap: 2px !important;
}

.tabs > .tab-nav > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    color: var(--c-text-3) !important;
    padding: 0.6rem 1.1rem !important;
    border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    cursor: pointer !important;
    transition: color 0.18s ease, background 0.18s ease !important;
    margin-bottom: -1px !important;
    position: relative !important;
}

.tabs > .tab-nav > button:hover {
    color: var(--c-text-2) !important;
    background: rgba(255,255,255,0.04) !important;
}

.tabs > .tab-nav > button.selected {
    color: #B4A7F5 !important;
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.25) !important;
    border-bottom: none !important;
}

/* Tab content pane */
.tabitem {
    background: rgba(255,255,255,0.014) !important;
    border: 1px solid var(--c-border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--r-lg) var(--r-lg) !important;
    padding: 1.35rem !important;
}

/* ── GENERATE BUTTON ──────────────────────────────────────── */
.primary-btn,
button.primary,
button[variant="primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 55%, #0E7490 100%) !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.875rem 2.25rem !important;
    cursor: pointer !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    box-shadow:
        0 0 0 1px rgba(124,58,237,0.35),
        0 4px 22px rgba(124,58,237,0.38),
        inset 0 1px 0 rgba(255,255,255,0.13) !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
}

/* Shimmer sweep on hover */
.primary-btn::after,
button.primary::after,
button[variant="primary"]::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg,
        transparent 35%,
        rgba(255,255,255,0.08) 50%,
        transparent 65%);
    transform: translateX(-100%);
    transition: transform 0.55s ease;
    pointer-events: none;
}

.primary-btn:hover::after,
button.primary:hover::after,
button[variant="primary"]:hover::after {
    transform: translateX(100%);
}

.primary-btn:hover,
button.primary:hover,
button[variant="primary"]:hover {
    transform: translateY(-3px) !important;
    box-shadow:
        0 0 0 1px rgba(124,58,237,0.45),
        0 8px 36px rgba(124,58,237,0.52),
        inset 0 1px 0 rgba(255,255,255,0.18) !important;
}

.primary-btn:active,
button.primary:active,
button[variant="primary"]:active {
    transform: translateY(-1px) !important;
}

/* ── SECONDARY BUTTONS ────────────────────────────────────── */
button:not(.primary):not([variant="primary"]):not(.tab-nav > button) {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--c-text-2) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}

button:not(.primary):not([variant="primary"]):not(.tab-nav > button):hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: var(--c-border-mid) !important;
}

/* ── AUDIO PLAYER ─────────────────────────────────────────── */
audio {
    border-radius: 10px !important;
    width: 100% !important;
}

/* ── STATUS TEXTBOX (read-only output) ────────────────────── */
#status-box textarea,
.glass-panel textarea[readonly],
textarea[readonly] {
    font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace !important;
    font-size: 0.8rem !important;
    color: var(--c-text-2) !important;
    letter-spacing: 0.01em !important;
    background: rgba(0,0,0,0.25) !important;
}

/* ── FILE UPLOAD ──────────────────────────────────────────── */
.upload-container,
.file-preview-holder,
.gr-file-upload {
    border: 1.5px dashed rgba(124,58,237,0.28) !important;
    border-radius: var(--r-md) !important;
    background: rgba(124,58,237,0.04) !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}

.upload-container:hover {
    border-color: rgba(124,58,237,0.5) !important;
    background: rgba(124,58,237,0.08) !important;
}

/* ── PROGRESS BAR ─────────────────────────────────────────── */
.generating,
.progress-bar {
    background: linear-gradient(90deg, #7C3AED, #4F46E5, #0891B2, #7C3AED) !important;
    background-size: 300% 100% !important;
    animation: aurora-sweep 2.2s ease infinite !important;
    border-radius: 999px !important;
}

@keyframes aurora-sweep {
    0%   { background-position: 100% center; }
    100% { background-position: -100% center; }
}

/* ── ACTION ROW ───────────────────────────────────────────── */
#action-row {
    margin-top: 1.5rem !important;
    align-items: center !important;
}

/* ── CUSTOM SCROLLBAR ─────────────────────────────────────── */
::-webkit-scrollbar          { width: 5px; height: 5px; }
::-webkit-scrollbar-track    { background: transparent; }
::-webkit-scrollbar-thumb    { background: rgba(124,58,237,0.28); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.5); }

/* ── ACCORDION / GROUP ────────────────────────────────────── */
details > summary {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.025em !important;
    color: var(--c-text-2) !important;
    cursor: pointer !important;
    list-style: none !important;
}

/* ── TOOLTIP ──────────────────────────────────────────────── */
.gr-tooltip,
[role="tooltip"] {
    background: #111827 !important;
    border: 1px solid var(--c-border-mid) !important;
    border-radius: 8px !important;
    color: var(--c-text-2) !important;
    font-size: 0.75rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5) !important;
}

/* ── BLOCK CONTAINER REFINEMENT ───────────────────────────── */
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
