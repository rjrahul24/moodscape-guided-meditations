"""MoodScape — Guided Meditation Audio Generator (Gradio UI)."""

import os
import time
import queue
import threading
import warnings

# Prevent HuggingFace tokenizers from warning about fork-after-parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Fix: MacOS Bus Error (SIGBUS) when Python garbage collects MPS tensors on process exit.
# We disable fallback to prevent unsupported ops from quietly failing, and register
# a hard exit hook to bypass the problematic PyTorch C++ destructor sequence.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
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
    "ambient, warm synthesizer pads, gentle drone, slow evolving, "
    "spacious atmosphere, peaceful, soft sustained tones, new age"
)


def _get_duration(path: str) -> float:
    info = sf.info(path)
    return info.duration


def _render_status(message, fraction, detail="", elapsed=None):
    percent = int(fraction * 100)
    elapsed_html = ""
    if elapsed is not None:
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{m}:{s:02d}" if m > 0 else f"{s}s"
        elapsed_html = f'<span class="status-elapsed">{elapsed_str}</span>'

    if fraction >= 1.0:
        # Complete state — checkmark + message + elapsed
        return f"""
        <div class="status-complete">
            <span class="status-check">&#10003;</span>
            <span class="status-msg">{message}</span>
            {elapsed_html}
        </div>
        {f'<div class="status-detail">{detail}</div>' if detail else ''}
        """
    elif fraction <= 0 and ("Error" in message or "Failed" in message):
        # Error state
        return f"""
        <div class="status-bar">
            <span class="status-msg status-error">{message}</span>
            {elapsed_html}
        </div>
        {f'<div class="status-detail status-error">{detail}</div>' if detail else ''}
        """
    else:
        # In-progress state — show progress bar with elapsed timer
        return f"""
        <div class="status-bar">
            <span class="status-msg">{message}</span>
            <span class="status-right">
                <span class="status-pct">{percent}%</span>
                {elapsed_html}
            </span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {percent}%"></div>
        </div>
        {f'<div class="status-detail">{detail}</div>' if detail else ''}
        """


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
    f5_wpm,
    reverb_ir_choice,
):
    # Initial status
    yield None, _render_status("Initializing Pipeline", 0.0)

    # Map music model label to engine key
    if music_model_choice == "ACE-Step 1.5":
        music_model = "acestep"
    elif music_model_choice == "Lyria RealTime":
        music_model = "lyria"
        # Validate that the API key is available before launching the pipeline
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            yield None, _render_status("Error: GOOGLE_API_KEY missing", 0.0, "Please check your .env file")
            return
    elif music_model_choice == "HeartMuLa":
        music_model = "heartmula"
    else:
        music_model = "heartmula"  # safe default

    # Resolve seed: 0 means auto
    seed = int(seed_value) if seed_value and int(seed_value) != 0 else None

    # Map quality label to engine key (Task 5)
    # "Draft (Turbo / 8-step)" -> "turbo"
    # "Studio (SFT / 50-step)" -> "sft"
    acestep_model_type = "turbo" if "Turbo" in acestep_quality else "sft"

    # Map TTS engine label to key
    tts_engine = "f5" if tts_engine_choice == "F5-TTS" else "kokoro"

    # Queue for streaming progress updates from the pipeline thread.
    # Items are (fraction, message) tuples; None is the sentinel for completion.
    update_queue = queue.Queue()
    result_container = {}
    start_time = time.time()

    def progress_cb(fraction, message):
        update_queue.put((fraction, message))

    def run_pipeline():
        try:
            result = pipeline.generate(
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
                bpm=acestep_bpm,
                keyscale=acestep_key,
                acestep_model_type=acestep_model_type,
                lyria_bpm=int(lyria_bpm),
                lyria_density=float(lyria_density),
                lyria_brightness=float(lyria_brightness),
                tts_engine=tts_engine,
                f5_voice_slug=f5_voice_slug if tts_engine == "f5" else None,
                f5_target_wpm=int(f5_wpm) if tts_engine == "f5" and f5_wpm > 0 else None,
                reverb_ir=reverb_ir_choice,
            )
            result_container["result"] = result
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            update_queue.put(None)  # sentinel: pipeline finished

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Stream progress updates until the pipeline signals done
    current_fraction = 0.0
    current_message = "Starting..."
    while True:
        try:
            item = update_queue.get(timeout=0.25)
        except queue.Empty:
            # No new update yet — re-yield current state with fresh elapsed time
            elapsed = time.time() - start_time
            yield None, _render_status(current_message, current_fraction, elapsed=elapsed)
            continue

        if item is None:
            break  # Pipeline finished

        current_fraction, current_message = item
        elapsed = time.time() - start_time
        yield None, _render_status(current_message, current_fraction, elapsed=elapsed)

    thread.join()
    elapsed = time.time() - start_time

    if "error" in result_container:
        yield None, _render_status("Generation Failed", 0.0, result_container["error"], elapsed=elapsed)
        return

    output_path, status_msg, _ = result_container["result"]
    duration = _get_duration(output_path)
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    tts_label = "F5-TTS" if tts_engine == "f5" else "Kokoro"
    detail = f"Synthesized with {tts_label} + {music_model_choice} · {minutes}m {seconds}s"
    yield output_path, _render_status("Generation Complete", 1.0, detail, elapsed=elapsed)


# CSS — Minimal Dark (Inspired by Linear / Raycast / Google Stitch)
CUSTOM_CSS = """
/* ── FONTS ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── DESIGN TOKENS ────────────────────────────────────────── */
:root {
    --bg-primary:     #0a0a0f;
    --bg-surface:     rgba(255,255,255,0.025);
    --bg-elevated:    rgba(255,255,255,0.04);
    --border-subtle:  rgba(255,255,255,0.08);
    --border-focus:   rgba(124,58,237,0.5);
    --accent:         #7c3aed;
    --accent-hover:   #8b5cf6;
    --accent-glow:    rgba(124,58,237,0.15);
    --text-primary:   #f0f0f3;
    --text-secondary: #a1a1aa;
    --text-tertiary:  #71717a;
    --text-muted:     #52525b;
    --radius-sm:      6px;
    --radius-md:      10px;
    --radius-lg:      14px;
    --radius-xl:      20px;
    --font-sans:      'Inter', system-ui, -apple-system, sans-serif;
    --font-mono:      'JetBrains Mono', ui-monospace, monospace;
}

/* ── BASE ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body { background: var(--bg-primary) !important; }

.gradio-container {
    min-height: 100vh !important;
    font-family: var(--font-sans) !important;
    background:
        radial-gradient(ellipse at 0% 0%, rgba(124,58,237,0.07) 0%, transparent 60%),
        var(--bg-primary) !important;
}

/* ── HEADER ───────────────────────────────────────────────── */
.app-header {
    padding: 2.5rem 0.5rem 1.5rem;
}

.app-header h1 {
    font-family: var(--font-sans) !important;
    font-size: clamp(1.8rem, 4vw, 2.4rem) !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #ffffff 0%, #e0d4ff 50%, #c4b5fd 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.3rem !important;
}

.app-header p {
    font-size: 0.9rem !important;
    color: var(--text-secondary) !important;
    font-weight: 300 !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

/* ── CANVAS ZONE (left column) ───────────────────────────── */
.canvas-zone {
    background: var(--bg-surface) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-xl) !important;
    padding: 1.5rem !important;
}

/* ── SETTINGS SIDEBAR (right column) ─────────────────────── */
.settings-sidebar {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* ── PILL RADIO BUTTONS ──────────────────────────────────── */
.pill-radio .wrap {
    display: flex !important;
    gap: 3px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 3px !important;
}

.pill-radio label {
    flex: 1 !important;
    text-align: center !important;
    padding: 7px 12px !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    border: none !important;
    background: transparent !important;
}

.pill-radio label.selected,
.pill-radio label:has(input:checked) {
    background: var(--accent) !important;
    color: white !important;
    box-shadow: 0 2px 8px var(--accent-glow) !important;
}

.pill-radio input[type="radio"] {
    display: none !important;
}

.pill-radio .wrap > label > span {
    cursor: pointer !important;
}

/* ── TOGGLE SWITCHES ─────────────────────────────────────── */
.toggle-switch input[type="checkbox"] {
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 36px !important;
    height: 20px !important;
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    position: relative !important;
    cursor: pointer !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    background-image: none !important;
    flex-shrink: 0 !important;
}

.toggle-switch input[type="checkbox"]::after {
    content: '' !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    background: var(--text-tertiary) !important;
    transition: transform 0.2s ease, background 0.2s ease !important;
}

.toggle-switch input[type="checkbox"]:checked {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    background-image: none !important;
}

.toggle-switch input[type="checkbox"]:checked::after {
    transform: translateX(16px) !important;
    background: white !important;
}

/* ── ACCORDION SECTIONS ──────────────────────────────────── */
.accordion-section {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-surface) !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
}

.accordion-section > .label-wrap {
    padding: 12px 16px !important;
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
    background: transparent !important;
    border-bottom: none !important;
}

.accordion-section > .label-wrap:hover {
    color: var(--text-primary) !important;
}

.accordion-section > .content {
    padding: 0 16px 16px !important;
}

/* ── INPUTS & TEXTAREAS ──────────────────────────────────── */
textarea, input[type="text"], input[type="number"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.875rem !important;
}

textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
    background: rgba(255,255,255,0.05) !important;
}

/* ── DROPDOWNS ───────────────────────────────────────────── */
.gradio-dropdown {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 2px 4px !important;
    transition: border-color 0.2s ease !important;
}

.gradio-dropdown:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

.dropdown-options {
    background: rgba(10, 10, 15, 0.95) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    margin-top: 4px !important;
    z-index: 9999 !important;
}

.dropdown-option {
    padding: 8px 14px !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    transition: background 0.15s ease !important;
}

.dropdown-option:hover {
    background: rgba(124, 58, 237, 0.12) !important;
    color: var(--text-primary) !important;
}

.dropdown-option.selected {
    background: rgba(124, 58, 237, 0.2) !important;
    color: white !important;
    font-weight: 500 !important;
}

/* ── GENERATE BUTTON ─────────────────────────────────────── */
.primary-btn {
    background: linear-gradient(135deg, var(--accent) 0%, #6d28d9 100%) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 14px 24px !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(124,58,237,0.2) !important;
    transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
}

.primary-btn:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(124,58,237,0.3) !important;
}

.primary-btn:active {
    transform: translateY(0) !important;
}

/* ── AUDIO PLAYER ────────────────────────────────────────── */
.music-player-glass {
    background: rgba(10, 10, 15, 0.8) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px !important;
    margin-top: 12px !important;
}

.music-player-glass audio {
    opacity: 0.85 !important;
    width: 100% !important;
}

/* ── SLIDERS ─────────────────────────────────────────────── */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ── STATUS BAR ──────────────────────────────────────────── */
.status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 2px;
}

.status-msg {
    font-family: var(--font-sans);
    font-size: 0.8rem;
    font-weight: 400;
    color: var(--text-secondary);
}

.status-right {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-pct {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 500;
}

.status-elapsed {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-tertiary);
    font-weight: 400;
}

.progress-track {
    width: 100%;
    height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 1.5px;
    overflow: hidden;
    margin: 8px 0;
}

.progress-fill {
    height: 100%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 1.5px;
}

.status-detail {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
    padding: 0 2px;
}

/* ── STATUS: COMPLETE ─────────────────────────────────────── */
.status-complete {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 2px;
}

.status-check {
    color: #4ade80;
    font-size: 0.85rem;
    font-weight: 600;
}

.status-complete .status-msg {
    color: var(--text-secondary);
}

/* ── STATUS: ERROR ────────────────────────────────────────── */
.status-error {
    color: #f87171 !important;
}

/* ── LABELS ──────────────────────────────────────────────── */
.block-label, .block-title {
    font-family: var(--font-sans) !important;
    font-weight: 400 !important;
    font-size: 0.75rem !important;
    color: var(--text-tertiary) !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
}

/* ── OVERFLOW FIX (for dropdowns) ────────────────────────── */
.canvas-zone, .settings-sidebar, .accordion-section,
.gradio-row, .gradio-container {
    overflow: visible !important;
}

/* ── SECTION SPACING ─────────────────────────────────────── */
.dropdown-container {
    margin-bottom: 0.75rem !important;
}

/* ── MARKDOWN INSIDE ACCORDIONS ──────────────────────────── */
.accordion-section h4 {
    font-family: var(--font-sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 0.5rem !important;
}
"""

theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="zinc",
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif", "system-ui", "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0a0a0f",
    body_text_color="#f0f0f3",
    body_text_color_subdued="#a1a1aa",
    block_background_fill="rgba(255,255,255,0.025)",
    block_border_color="rgba(255,255,255,0.08)",
    block_border_width="1px",
    block_label_text_color="#71717a",
    block_label_background_fill="transparent",
    block_title_text_weight="500",
    block_title_text_size="12px",
    block_title_text_color="#71717a",
    input_background_fill="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_color_focus="rgba(124,58,237,0.5)",
    input_placeholder_color="#52525b",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(255,255,255,0.04)",
    button_secondary_border_color="rgba(255,255,255,0.08)",
    button_secondary_text_color="#a1a1aa",
).set(
    body_background_fill_dark="#0a0a0f",
    block_background_fill_dark="rgba(255,255,255,0.025)",
    input_background_fill_dark="rgba(255,255,255,0.04)",
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
            <p>AI-powered meditation audio synthesis</p>
        </div>
    """)

    with gr.Row():
        # ── Left column: Creative Canvas ──────────────────────────────────
        with gr.Column(scale=3, elem_classes="canvas-zone"):
            with gr.Group():
                generation_mode = gr.Radio(
                    choices=["Instrumental Only", "Vocals Only", "Instrumental + Vocal"],
                    value="Instrumental + Vocal",
                    label="Mode",
                    elem_classes="pill-radio",
                )
                script_input = gr.Textbox(
                    label="Meditation Script",
                    placeholder=(
                        "Write your meditation script here...\n"
                        "Use [pause:5s] for timed pauses, [breath] for breath pauses.\n"
                        "Double newlines create natural pauses."
                    ),
                    value=DEFAULT_SCRIPT,
                    lines=14,
                    elem_id="script-textbox",
                )

            with gr.Row():
                music_prompt = gr.Textbox(
                    label="Music Style",
                    placeholder="E.g. Evolving pads, gentle rain, minimal soft piano...",
                    value=DEFAULT_MUSIC_PROMPT,
                    lines=2,
                    scale=2,
                )
                music_duration = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    value=3.0,
                    step=0.5,
                    label="Duration (min)",
                    visible=False,
                    scale=1,
                )

            generate_btn = gr.Button(
                "✦  Generate Meditation",
                variant="primary",
                size="lg",
                elem_classes="primary-btn",
            )
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                elem_classes="music-player-glass",
            )
            status_display = gr.HTML(
                _render_status("Ready", 0.0, "Ready to synthesize."),
                elem_id="status-display",
            )

        # ── Right column: Settings Sidebar ────────────────────────────────
        with gr.Column(scale=2, elem_classes="settings-sidebar"):

            # Section 1: Voice & Sound
            with gr.Accordion("Voice & Sound", open=True, elem_classes="accordion-section"):
                music_model_dropdown = gr.Dropdown(
                    choices=["ACE-Step 1.5", "HeartMuLa", "Lyria RealTime"],
                    value="ACE-Step 1.5",
                    label="Music Engine",
                    info="ACE-Step 1.5 recommended for Apple Silicon (approx. 5 min). HeartMuLa uses MLX/MPS with lazy loading (approx. 8-20 min).",
                    elem_classes="dropdown-container",
                )
                acestep_quality = gr.Radio(
                    choices=["Draft (Turbo / 8-step)", "Studio (SFT / 50-step)"],
                    value="Studio (SFT / 50-step)",
                    label="Quality",
                    visible=True,
                    elem_classes="pill-radio",
                )
                tts_engine_radio = gr.Radio(
                    choices=["Kokoro", "F5-TTS"],
                    value="Kokoro",
                    label="Voice Engine",
                    elem_classes="pill-radio",
                )
                with gr.Group(visible=True, elem_id="kokoro-group") as kokoro_settings:
                    kokoro_voice_dropdown = gr.Dropdown(
                        choices=KOKORO_VOICE_CHOICES,
                        value="balanced_calm",
                        label="Voice",
                        elem_classes="dropdown-container",
                    )
                with gr.Group(visible=False, elem_id="f5-group") as f5_settings:
                    f5_voice_dropdown = gr.Dropdown(
                        choices=F5_VOICE_CHOICES if F5_VOICE_CHOICES else ["(no voices)"],
                        value=F5_VOICE_DEFAULT,
                        label="Voice",
                        interactive=bool(F5_VOICE_CHOICES),
                        elem_classes="dropdown-container",
                    )
                    f5_wpm_slider = gr.Slider(
                        0, 150, 0, step=5,
                        label="Pacing (WPM)",
                        info="0 = natural rhythm (recommended), 90-110 = meditation, 120-150 = narration",
                    )

            # Section 2: Mix & Effects
            with gr.Accordion("Mix & Effects", open=False, elem_classes="accordion-section"):
                with gr.Row():
                    speed_slider = gr.Slider(0.70, 1.20, 1.0, step=0.01, label="Speed")
                    duck_slider = gr.Slider(-30, -5, -20, step=1, label="Ducking (dB)")
                with gr.Row():
                    reverb_slider = gr.Slider(0.0, 0.5, 0.15, step=0.05, label="Reverb")
                    reverb_ir_dropdown = gr.Dropdown(
                        choices=[
                            ("Warm Studio", "warm_studio"),
                            ("Wooden Hall", "wooden_hall"),
                            ("Stone Chapel", "stone_chapel"),
                        ],
                        value="warm_studio",
                        label="Space",
                        elem_classes="dropdown-container",
                    )
                with gr.Row():
                    fade_in_slider = gr.Slider(0, 10, 3, step=0.5, label="Fade In (s)")
                    fade_out_slider = gr.Slider(0, 15, 6, step=0.5, label="Fade Out (s)")

            # Section 3: Advanced
            with gr.Accordion("Advanced", open=False, elem_classes="accordion-section"):
                with gr.Group(visible=True) as acestep_metadata:
                    gr.Markdown("#### ACE-Step Tuning")
                    with gr.Row():
                        acestep_bpm = gr.Slider(40, 100, 50, step=1, label="BPM")
                        acestep_key = gr.Dropdown(
                            choices=["Auto", "C Major", "C Minor", "C# Major", "C# Minor", "D Major", "D Minor", "Eb Major", "Eb Minor", "E Major", "E Minor", "F Major", "F Minor", "F# Major", "F# Minor", "G Major", "G Minor", "Ab Major", "Ab Minor", "A Major", "A Minor", "Bb Major", "Bb Minor", "B Major", "B Minor"],
                            value="Auto",
                            label="Key",
                            elem_classes="dropdown-container",
                        )
                with gr.Group(visible=False) as lyria_settings:
                    gr.Markdown("#### Lyria Tuning")
                    lyria_bpm = gr.Slider(60, 200, 70, step=1, label="BPM")
                    with gr.Row():
                        lyria_density = gr.Slider(0, 1.0, 0.1, step=0.05, label="Density")
                        lyria_brightness = gr.Slider(0, 1.0, 0.15, step=0.05, label="Brightness")

                gr.Markdown("#### Export")
                with gr.Row():
                    format_radio = gr.Radio(
                        ["wav", "mp3"], value="wav", label="Format",
                        elem_classes="pill-radio",
                    )
                    seed_input = gr.Number(label="Seed", value=0, precision=0)
                with gr.Row():
                    upsample_checkbox = gr.Checkbox(
                        label="Hi-Fi (48kHz)", value=True,
                        elem_classes="toggle-switch",
                    )
                    stems_checkbox = gr.Checkbox(
                        label="Export Stems", value=False,
                        elem_classes="toggle-switch",
                    )
                stem_separation_checkbox = gr.Checkbox(
                    label="Clean Music (Source Separation)", value=True,
                    elem_classes="toggle-switch",
                )
                reference_audio = gr.Audio(
                    label="Style/Melody Reference",
                    type="filepath",
                    sources=["upload"],
                )

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
            speed_val = 1.0
            speed_label = "Speaking Speed (1.0 = natural, use slow reference for pace)"
        else:
            speed_val = 0.90
            speed_label = "Speaking Speed (0.85-0.95 = meditation ideal)"
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
            f5_wpm_slider,
            reverb_ir_dropdown,
        ],
        outputs=[audio_output, status_display],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(share=False)
