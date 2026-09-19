"""Gradio TTS Sandbox tab — test, tune, evaluate, and promote open-source TTS models.

Provides two 5-minute pre-written benchmark scripts (Guided Meditation and Sleep Story),
the full pre-processing and post-processing audio engine identical to the Manual screen,
objective QA scorecards, and a one-click workflow to promote winning models into the app.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf

from core.content_profiles import CONTENT_PROFILES, get_profile, normalize_content_type
from core.engine_registry import (
    demote_model,
    get_engine_info,
    get_model_presets,
    list_all_engines,
    load_promoted_models,
    promote_model,
)
from core.f5_tts import voice_registry as _f5_registry
from core.pipeline import MeditationPipeline
from core.qa_monitor import check_clipping, check_lufs, check_silence_ratio, check_spectral_rolloff, run_qa_checks
from core.sandbox_scripts import (
    BENCHMARK_METADATA,
    MEDITATION_5MIN_SCRIPT,
    SLEEP_STORY_5MIN_SCRIPT,
    get_benchmark_script,
    get_script_metadata,
)
from core.upload_music import scan_backgrounds

logger = logging.getLogger("moodscape.sandbox")

# ── Voice and Background Assets ──────────────────────────────────────────────

_F5_REGISTRY = _f5_registry.scan()
F5_VOICE_SLUGS = sorted(_F5_REGISTRY.keys())
F5_VOICE_CHOICES = [(slug.replace("_", " ").title(), slug) for slug in F5_VOICE_SLUGS]
F5_VOICE_DEFAULT = F5_VOICE_CHOICES[0][1] if F5_VOICE_CHOICES else None

KOKORO_VOICE_CHOICES = [
    ("Balanced Calm — natural & human (default)", "balanced_calm"),
    ("Deep Rest — intimate & breathy", "deep_rest"),
    ("Soft Whisper — ASMR relaxation", "soft_whisper"),
    ("Golden Hour — warm & airy", "golden_hour"),
    ("Earth Root — grounding blend", "earth_root"),
    ("Pure Calm — tension-free ultra-soft", "pure_calm"),
    ("Heart — US Female (warm)", "af_heart"),
    ("Nicole — US Female (calm/ASMR)", "af_nicole"),
    ("Emma — UK Female (wise)", "bf_emma"),
    ("Adam — US Male (grounding)", "am_adam"),
    ("George — UK Male (warm)", "bm_george"),
]

BACKGROUND_CHOICES = scan_backgrounds()
BACKGROUND_DEFAULT = BACKGROUND_CHOICES[0][1] if BACKGROUND_CHOICES else None

DEFAULT_MUSIC_PROMPT = (
    "ambient, warm synthesizer pads, gentle drone, slow evolving, "
    "spacious atmosphere, peaceful, soft sustained tones, new age"
)


# ── Render Helpers ───────────────────────────────────────────────────────────

def render_sandbox_status(message: str, fraction: float, detail: str = "", elapsed: float | None = None) -> str:
    """Render sleek HTML status matching app.py theme."""
    percent = int(fraction * 100)
    elapsed_html = ""
    if elapsed is not None:
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{m}:{s:02d}" if m > 0 else f"{s}s"
        elapsed_html = f'<span class="status-elapsed">{elapsed_str}</span>'

    if fraction >= 1.0:
        return f"""
        <div class="status-shell">
            <div class="status-complete">
                <span class="status-check">&#10003;</span>
                <span class="status-msg">{message}</span>
                {elapsed_html}
            </div>
            {f'<div class="status-detail">{detail}</div>' if detail else ''}
        </div>
        """
    elif fraction <= 0 and ("Error" in message or "Failed" in message):
        return f"""
        <div class="status-shell">
            <div class="status-bar">
                <span class="status-msg status-error">{message}</span>
                {elapsed_html}
            </div>
            {f'<div class="status-detail status-error">{detail}</div>' if detail else ''}
        </div>
        """
    else:
        return f"""
        <div class="status-shell">
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
        </div>
        """


def render_qa_scorecard(audio_path: str | None, is_vocals_only: bool = False) -> str:
    """Generate an objective audio quality scorecard HTML from QA checks."""
    if not audio_path or not os.path.isfile(audio_path):
        return """
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 16px; margin-top: 12px;">
            <div style="font-weight: 600; font-size: 0.95em; color: #94a3b8; margin-bottom: 4px;">📊 Audio QA Scorecard</div>
            <div style="font-size: 0.85em; color: #64748b;">Generate audio to run quality metrics (LUFS, True Peak, Clipping, Silence ratio).</div>
        </div>
        """

    try:
        data, sr = sf.read(audio_path)
        mono = data[0] if data.ndim == 2 else data
        duration_sec = len(mono) / sr

        # Audio metrics
        lufs_res = check_lufs(mono, sample_rate=sr, target=-16.0, tolerance=2.0)
        measured_lufs = lufs_res.get("measured_lufs", -99.0)
        lufs_passed = lufs_res.get("passed", False)

        clip_res = check_clipping(mono)
        clip_count = clip_res.get("clipped_samples", 0)
        clip_passed = clip_res.get("passed", True)

        silence_res = check_silence_ratio(mono, sample_rate=sr)
        silence_ratio = silence_res.get("silence_ratio", 0.0) * 100.0
        silence_passed = silence_res.get("passed", True)

        rolloff_res = check_spectral_rolloff(mono, sample_rate=sr)
        rolloff_hz = rolloff_res.get("median_rolloff_hz", 0.0)

        # True peak calculation
        true_peak_db = float(20.0 * np.log10(max(np.max(np.abs(mono)), 1e-6)))
        peak_passed = true_peak_db <= -0.9

        all_passed = lufs_passed and clip_passed and silence_passed and peak_passed

        badge_bg = "#10b981" if all_passed else "#f59e0b"
        badge_text = "PASSED ALL QA CHECKS" if all_passed else "ADVISORY WARNINGS"

        lufs_color = "#10b981" if lufs_passed else "#f59e0b"
        peak_color = "#10b981" if peak_passed else "#ef4444"
        clip_color = "#10b981" if clip_passed else "#ef4444"
        silence_color = "#10b981" if silence_passed else "#f59e0b"

        dur_m = int(duration_sec // 60)
        dur_s = int(duration_sec % 60)

        return f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 8px; padding: 16px; margin-top: 12px; font-family: ui-sans-serif, system-ui, sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; font-size: 1.0em; color: #f8fafc;">📊 Audio Quality Scorecard</span>
                <span style="background: {badge_bg}; color: #ffffff; font-size: 0.75em; font-weight: 700; padding: 3px 8px; border-radius: 9999px; letter-spacing: 0.05em;">{badge_text}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px;">
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">Duration</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: #f8fafc; margin-top: 2px;">{dur_m}:{dur_s:02d}</div>
                    <div style="font-size: 0.7em; color: #64748b;">Target ~5:00 min</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">Loudness</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: {lufs_color}; margin-top: 2px;">{measured_lufs:.1f} LUFS</div>
                    <div style="font-size: 0.7em; color: #64748b;">Target -16 ± 2 LUFS</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">True Peak</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: {peak_color}; margin-top: 2px;">{true_peak_db:.1f} dBTP</div>
                    <div style="font-size: 0.7em; color: #64748b;">Ceiling -1.0 dBTP</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">Clipping</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: {clip_color}; margin-top: 2px;">{clip_count} samples</div>
                    <div style="font-size: 0.7em; color: #64748b;">0 allowed</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">Silence Ratio</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: {silence_color}; margin-top: 2px;">{silence_ratio:.1f}%</div>
                    <div style="font-size: 0.7em; color: #64748b;">Target 15%–75%</div>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 6px;">
                    <div style="font-size: 0.75em; color: #94a3b8; text-transform: uppercase;">Spectral Rolloff</div>
                    <div style="font-size: 1.15em; font-weight: 700; color: #f8fafc; margin-top: 2px;">{rolloff_hz:.0f} Hz</div>
                    <div style="font-size: 0.7em; color: #64748b;">Max 8000 Hz</div>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        logger.error("Error creating QA scorecard: %s", e)
        return f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 16px; margin-top: 12px;">
            <div style="font-weight: 600; font-size: 0.95em; color: #f59e0b;">⚠️ QA Scorecard Unavailable</div>
            <div style="font-size: 0.85em; color: #94a3b8;">{e}</div>
        </div>
        """


def format_promoted_models_html() -> str:
    """Render markdown/HTML view of currently promoted models."""
    models = load_promoted_models()
    if not models:
        return "<div style='color: #94a3b8; font-style: italic; padding: 8px 0;'>No custom models currently promoted. Test models above and click 'Promote Model to App'.</div>"

    rows = []
    for mid, info in sorted(models.items()):
        name = info.get("name", mid)
        base = info.get("base_engine", "f5").upper()
        desc = info.get("description", "")
        presets = info.get("presets", {})
        speed = presets.get("speed", 0.90)
        duck = presets.get("duck_amount_db", -16.0)
        wpm = presets.get("target_wpm", 0)
        cfg = presets.get("cfg_strength", 2.0)
        reverb = presets.get("reverb_amount", 0.15)
        p_str = f"Speed {speed} · Reverb {reverb} · Duck {duck}dB"
        if base == "F5":
            p_str += f" · CFG {cfg} · WPM {wpm}"

        rows.append(f"""
        <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.15);">
            <td style="padding: 8px 6px; font-weight: 600; color: #f8fafc;">{name}</td>
            <td style="padding: 8px 6px; color: #38bdf8;">{base}</td>
            <td style="padding: 8px 6px; font-size: 0.85em; color: #94a3b8;">{p_str}</td>
            <td style="padding: 8px 6px; font-size: 0.85em; color: #64748b;">{desc}</td>
        </tr>
        """)

    table_rows = "\n".join(rows)
    return f"""
    <table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 0.9em; margin-top: 8px;">
        <thead>
            <tr style="border-bottom: 2px solid rgba(148, 163, 184, 0.3); color: #cbd5e1;">
                <th style="padding: 6px;">Model Name</th>
                <th style="padding: 6px;">Base</th>
                <th style="padding: 6px;">Tuned Presets</th>
                <th style="padding: 6px;">Notes</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
    """


# ── Sandbox Handler ──────────────────────────────────────────────────────────

_sandbox_pipeline: MeditationPipeline | None = None

def _get_sandbox_pipeline() -> MeditationPipeline:
    global _sandbox_pipeline
    if _sandbox_pipeline is None:
        _sandbox_pipeline = MeditationPipeline()
    return _sandbox_pipeline


def sandbox_generate_handler(
    benchmark_choice: str,
    generation_mode: str,
    script_text: str,
    content_type_choice: str,
    music_prompt: str,
    music_duration: float,
    model_source: str,
    engine_choice: str,
    kokoro_voice: str,
    f5_voice: str,
    custom_model_id: str,
    custom_display_name: str,
    custom_base_engine: str,
    custom_voice_slug: str,
    f5_wpm: int,
    f5_cfg: float,
    microprosody_flag: bool,
    df_wet_val: float,
    music_model_choice: str,
    uploaded_music_file: str,
    speed: float,
    duck_amount: float,
    spectral_duck_flag: bool,
    shared_reverb_flag: bool,
    reverb_amount: float,
    reverb_ir_choice: str,
    fade_in: float,
    fade_out: float,
    output_format: str,
    seed_val: int,
    upsample_flag: bool,
    export_stems_flag: bool,
    stem_separation_flag: bool,
    quality_mode_flag: bool,
    stereo_output_flag: bool,
    lyria_bpm: int,
    lyria_density: float,
    lyria_brightness: float,
):
    """Orchestrates end-to-end sandbox synthesis with progress updates."""
    start_time = time.time()
    yield None, None, None, render_sandbox_status("Initializing Sandbox Pipeline...", 0.0, elapsed=0), render_qa_scorecard(None)

    content_type = normalize_content_type(content_type_choice)

    # Set research flags
    os.environ["MOODSCAPE_SPECTRAL_DUCK"] = "1" if spectral_duck_flag else "0"
    os.environ["MOODSCAPE_SHARED_REVERB"] = "1" if shared_reverb_flag else "0"
    os.environ["MOODSCAPE_F5_MICROPROSODY"] = "1" if microprosody_flag else "0"
    os.environ["MOODSCAPE_F5_CFG"] = str(float(f5_cfg))
    os.environ["MOODSCAPE_KOKORO_DF_WET"] = str(float(df_wet_val))

    # Resolve active engine and voice
    if model_source == "New Open-Source Test":
        tts_engine = custom_base_engine or "f5"
        f5_voice_slug = custom_voice_slug or F5_VOICE_DEFAULT
        active_voice = kokoro_voice
    else:
        tts_engine = engine_choice or "f5"
        f5_voice_slug = f5_voice or F5_VOICE_DEFAULT
        active_voice = kokoro_voice

    # Resolve music engine
    if music_model_choice == "Lyria RealTime":
        music_model = "lyria"
        if not os.environ.get("GOOGLE_API_KEY", "").strip():
            yield None, None, None, render_sandbox_status("Error: GOOGLE_API_KEY missing", 0.0, "Check your .env file"), render_qa_scorecard(None)
            return
    else:
        music_model = "upload"
        if generation_mode != "Vocals Only" and not uploaded_music_file:
            yield None, None, None, render_sandbox_status("Error: No background track selected", 0.0, "Choose an instrumental track"), render_qa_scorecard(None)
            return

    pipeline = _get_sandbox_pipeline()
    progress_queue: list[tuple[float, str]] = []

    def progress_callback(fraction: float, message: str):
        elapsed = time.time() - start_time
        progress_queue.append((fraction, message))

    # Execute pipeline
    try:
        yield None, None, None, render_sandbox_status("Starting synthesis...", 0.05, elapsed=time.time() - start_time), render_qa_scorecard(None)

        audio_path, status_msg = pipeline.generate(
            script=script_text,
            music_prompt=music_prompt,
            voice=active_voice,
            speed=float(speed),
            duck_amount_db=float(duck_amount),
            reverb_amount=float(reverb_amount),
            fade_in_sec=float(fade_in),
            fade_out_sec=float(fade_out),
            output_format=output_format,
            progress_cb=progress_callback,
            seed=int(seed_val) if seed_val else None,
            do_export_stems=export_stems_flag,
            upsample_48k=upsample_flag,
            generation_mode=generation_mode,
            instrumental_duration_m=float(music_duration),
            music_model=music_model,
            stem_separation=stem_separation_flag,
            lyria_bpm=int(lyria_bpm),
            lyria_density=float(lyria_density),
            lyria_brightness=float(lyria_brightness),
            tts_engine=tts_engine,
            f5_voice_slug=f5_voice_slug,
            f5_target_wpm=int(f5_wpm) if f5_wpm > 0 else None,
            reverb_ir=reverb_ir_choice,
            quality_mode=quality_mode_flag,
            stereo_output=stereo_output_flag,
            uploaded_music_path=uploaded_music_file,
            content_type=content_type,
        )

        elapsed = time.time() - start_time
        scorecard = render_qa_scorecard(audio_path, is_vocals_only=(generation_mode == "Vocals Only"))

        # Check for stems
        voice_stem_path = None
        music_stem_path = None
        if export_stems_flag and pipeline.last_stems:
            voice_stem_path = pipeline.last_stems.get("voice")
            music_stem_path = pipeline.last_stems.get("music")
        elif generation_mode == "Vocals Only":
            voice_stem_path = audio_path

        yield (
            audio_path,
            voice_stem_path,
            music_stem_path,
            render_sandbox_status("Synthesis Complete!", 1.0, f"Ready for evaluation. {status_msg}", elapsed=elapsed),
            scorecard,
        )
    except Exception as e:
        logger.error("Sandbox generation failed: %s", e, exc_info=True)
        yield None, None, None, render_sandbox_status(f"Generation Failed: {str(e)}", 0.0, "Check application logs for details"), render_qa_scorecard(None)


# ── Promotion & Management Callbacks ────────────────────────────────────────

def handle_promote_model(
    model_source: str,
    engine_choice: str,
    custom_model_id: str,
    custom_display_name: str,
    custom_base_engine: str,
    custom_voice_slug: str,
    f5_voice: str,
    speed: float,
    f5_wpm: int,
    f5_cfg: float,
    reverb_amount: float,
    reverb_ir: str,
    duck_amount: float,
    spectral_duck: bool,
    shared_reverb: bool,
    microprosody: bool,
    df_wet: float,
    content_type_choice: str,
):
    """Save tuned model settings to var/promoted_models.json."""
    if model_source == "New Open-Source Test":
        m_id = (custom_model_id or "").strip()
        if not m_id:
            return (
                "<div style='color: #ef4444; font-weight: 600;'>⚠️ Please provide a Model Identifier before promoting.</div>",
                gr.update(),
                format_promoted_models_html(),
            )
        display_name = custom_display_name or m_id.replace("_", " ").title()
        base = custom_base_engine or "f5"
        config = {
            "voice_slug": custom_voice_slug or F5_VOICE_DEFAULT,
            "adapter_type": base,
        }
    else:
        m_id = f"promoted_{engine_choice}_{int(time.time()) % 10000}"
        info = get_engine_info(engine_choice)
        base = info.get("base_engine", "f5") if info else "f5"
        display_name = f"{info.get('name', engine_choice)} (Fine-Tuned)"
        config = {
            "voice_slug": f5_voice or F5_VOICE_DEFAULT,
            "base_engine": base,
        }

    presets = {
        "speed": float(speed),
        "target_wpm": int(f5_wpm),
        "cfg_strength": float(f5_cfg),
        "reverb_amount": float(reverb_amount),
        "reverb_ir": reverb_ir,
        "duck_amount_db": float(duck_amount),
        "spectral_duck": bool(spectral_duck),
        "shared_reverb": bool(shared_reverb),
        "microprosody": bool(microprosody),
        "df_wet": float(df_wet),
        "content_type": normalize_content_type(content_type_choice),
    }

    record = promote_model(
        model_id=m_id,
        display_name=display_name,
        base_engine=base,
        config=config,
        presets=presets,
        description=f"Promoted from Sandbox for {content_type_choice}",
    )

    choices = list_all_engines()
    msg = f"<div style='color: #10b981; font-weight: 600;'>⭐ Successfully promoted '{record['name']}'! It is now active across Manual and Auto-Generate flows.</div>"
    return msg, gr.update(choices=choices, value=record["id"]), format_promoted_models_html()


def handle_discard_model():
    """Clear test state and trigger garbage collection."""
    gc.collect()
    msg = "<div style='color: #94a3b8; font-style: italic;'>Model test discarded. Memory freed and sandbox reset.</div>"
    return None, None, None, msg, render_sandbox_status("Ready", 0.0, "Ready for next test run"), render_qa_scorecard(None)


def handle_demote_model(model_id: str):
    """Demote model from active registry."""
    if not model_id:
        return "<div style='color: #ef4444;'>Select a model to demote.</div>", gr.update(), format_promoted_models_html()
    success = demote_model(model_id)
    choices = list_all_engines()
    if success:
        msg = f"<div style='color: #10b981;'>Model '{model_id}' removed from active registry.</div>"
    else:
        msg = f"<div style='color: #ef4444;'>Could not demote '{model_id}' (not found).</div>"
    return msg, gr.update(choices=choices, value="f5"), format_promoted_models_html()


# ── Tab Builder ──────────────────────────────────────────────────────────────

def build_sandbox_tab() -> dict[str, Any]:
    """Construct the TTS Sandbox tab inside gr.Blocks().

    Returns dict of created components for testing and external event wiring.
    """
    with gr.Tab("TTS Sandbox"):
        gr.Markdown(
            "### 🧪 Open-Source TTS Model Sandbox\n"
            "Experiment with, evaluate, and fine-tune open-source TTS models against 5-minute benchmark scripts. "
            "Runs through MoodScape's full pre/post-processing engine (sinc upsampling, DeepFilterNet denoising, "
            "vocal mastering, adaptive breathing ducking, convolution reverb, and true-peak limiting). "
            "Review the QA Scorecard, dial in optimal voice knobs, then promote winning models directly to the app."
        )

        with gr.Row():
            # ── Left Column: Benchmark Creative Canvas ────────────────────────
            with gr.Column(scale=3, elem_classes="canvas-zone"):
                with gr.Group():
                    with gr.Row():
                        benchmark_script_choice = gr.Radio(
                            choices=["🧘 5-Min Guided Meditation", "🌙 5-Min Sleep Story"],
                            value="🧘 5-Min Guided Meditation",
                            label="Benchmark Script",
                            elem_classes="pill-radio",
                            scale=3,
                        )
                        reset_script_btn = gr.Button("↺ Reset Script", size="sm", scale=1, min_width=110)

                    content_type_dropdown = gr.Dropdown(
                        choices=["Guided Meditation", "Sleep Story"],
                        value="Guided Meditation",
                        label="Content Profile",
                        info="Drives paragraph pause durations and ambient bed ducking behavior.",
                        elem_classes="dropdown-container",
                    )
                    generation_mode = gr.Radio(
                        choices=["Vocals Only", "Instrumental + Vocal", "Instrumental Only"],
                        value="Vocals Only",
                        label="Evaluation Mode",
                        info="Tip: Use 'Vocals Only' to critically inspect raw speech timbre, breathing, and prosody.",
                        elem_classes="pill-radio",
                    )
                    script_input = gr.Textbox(
                        label="Benchmark Script (Editable)",
                        placeholder="Benchmark script content...",
                        value=MEDITATION_5MIN_SCRIPT,
                        lines=14,
                        elem_id="sandbox-script-textbox",
                    )

                with gr.Row():
                    music_prompt = gr.Textbox(
                        label="Atmosphere",
                        placeholder="E.g. warm synthesizer pads, gentle drone...",
                        value=DEFAULT_MUSIC_PROMPT,
                        lines=2,
                        scale=2,
                        visible=False,
                    )
                    music_duration = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=5.0,
                        step=0.5,
                        label="Duration (min)",
                        visible=False,
                        scale=1,
                    )

                generate_btn = gr.Button(
                    "⚡ Test Model in Sandbox",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-btn",
                )

                audio_output = gr.Audio(
                    label="Sandbox Audio Output",
                    type="filepath",
                    elem_classes="music-player-glass",
                )

                with gr.Row():
                    vocal_stem_output = gr.Audio(
                        label="Narration Stem (Solo)",
                        type="filepath",
                        elem_classes="music-player-glass",
                        visible=True,
                    )
                    music_stem_output = gr.Audio(
                        label="Music Bed Stem (Solo)",
                        type="filepath",
                        elem_classes="music-player-glass",
                        visible=False,
                    )

                status_display = gr.HTML(
                    render_sandbox_status("Ready", 0.0, "Ready to synthesize benchmark script."),
                    elem_id="sandbox-status-display",
                )

                # ── Model Decision & Promotion Suite ──────────────────────────
                with gr.Group(elem_classes="accordion-section"):
                    gr.Markdown("#### 🎯 Model Decision Suite")
                    with gr.Row():
                        promote_btn = gr.Button(
                            "⭐ Promote Model to App",
                            variant="primary",
                            size="md",
                            elem_classes="primary-btn",
                        )
                        discard_btn = gr.Button(
                            "🗑️ Discard Model / Reset",
                            variant="secondary",
                            size="md",
                        )
                    promotion_status = gr.HTML(
                        "<div style='color: #94a3b8; font-size: 0.9em;'>Run a sandbox test to evaluate audio quality before promoting or discarding.</div>"
                    )

                # ── Objective Audio QA Scorecard ──────────────────────────────
                scorecard_display = gr.HTML(
                    render_qa_scorecard(None),
                    elem_id="sandbox-scorecard-display",
                )

            # ── Right Column: Settings Sidebar (Identical to Manual + Model Config) ──
            with gr.Column(scale=2, elem_classes="settings-sidebar"):

                # Section 1: Model & Voice Architecture
                with gr.Accordion("Model & Voice Architecture", open=True, elem_classes="accordion-section"):
                    model_source_radio = gr.Radio(
                        choices=["Built-in / Promoted", "New Open-Source Test"],
                        value="Built-in / Promoted",
                        label="Model Source",
                        elem_classes="pill-radio",
                    )

                    with gr.Group(visible=True) as builtin_group:
                        engine_dropdown = gr.Dropdown(
                            choices=list_all_engines(),
                            value="f5",
                            label="TTS Engine",
                            elem_classes="dropdown-container",
                        )
                        with gr.Group(visible=False) as kokoro_group:
                            kokoro_voice_dropdown = gr.Dropdown(
                                choices=KOKORO_VOICE_CHOICES,
                                value="balanced_calm",
                                label="Kokoro Voice Blend",
                                elem_classes="dropdown-container",
                            )
                        with gr.Group(visible=True) as f5_group:
                            f5_voice_dropdown = gr.Dropdown(
                                choices=F5_VOICE_CHOICES if F5_VOICE_CHOICES else ["(no voices)"],
                                value=F5_VOICE_DEFAULT,
                                label="F5 Voice Pool",
                                interactive=bool(F5_VOICE_CHOICES),
                                elem_classes="dropdown-container",
                            )

                    with gr.Group(visible=False) as custom_model_group:
                        custom_model_id = gr.Textbox(
                            label="New Model Identifier / Slug",
                            placeholder="e.g. mystic_sage_v1",
                        )
                        custom_display_name = gr.Textbox(
                            label="Display Name",
                            placeholder="e.g. Mystic Sage (Warm Breathing)",
                        )
                        custom_base_engine = gr.Dropdown(
                            choices=["f5", "kokoro", "chatterbox", "custom"],
                            value="f5",
                            label="Base Framework / Architecture",
                            elem_classes="dropdown-container",
                        )
                        custom_voice_slug = gr.Dropdown(
                            choices=F5_VOICE_CHOICES if F5_VOICE_CHOICES else ["(no voices)"],
                            value=F5_VOICE_DEFAULT,
                            label="Reference Voice Audio",
                            elem_classes="dropdown-container",
                        )

                    f5_wpm_slider = gr.Slider(
                        0, 150, 0, step=5,
                        label="Pacing (WPM)",
                        info="0 = natural rhythm (recommended). 90–110 = meditation. 120–150 = narration.",
                    )
                    f5_cfg_slider = gr.Slider(
                        minimum=1.0, maximum=2.5, value=2.0, step=0.1,
                        label="Voice Expressiveness (F5 CFG)",
                        info="Lower (e.g. 1.2–1.5) = warmer/more expressive. F5 only.",
                    )
                    with gr.Row():
                        microprosody_checkbox = gr.Checkbox(
                            label="Voice Microprosody", value=False,
                            info="F5 only: phrase-final pitch drop + breathiness.",
                            elem_classes="toggle-switch",
                        )
                    df_wet_slider = gr.Slider(
                        0.0, 1.0, 0.25, step=0.05,
                        label="DeepFilterNet Denoising (Wet)",
                        info="Neural denoising strength. 0.25 = Kokoro sweet spot; 1.0 = full F5 denoising.",
                    )

                # Section 2: Music Bed
                with gr.Accordion("Music Bed", open=False, elem_classes="accordion-section"):
                    music_model_dropdown = gr.Dropdown(
                        choices=["Background Music", "Lyria RealTime"],
                        value="Background Music",
                        label="Music Engine",
                        elem_classes="dropdown-container",
                    )
                    with gr.Group(visible=True) as upload_settings:
                        with gr.Row():
                            uploaded_music = gr.Dropdown(
                                choices=BACKGROUND_CHOICES if BACKGROUND_CHOICES else ["(no tracks found)"],
                                value=BACKGROUND_DEFAULT,
                                label="Instrumental Track",
                                interactive=bool(BACKGROUND_CHOICES),
                                elem_classes="dropdown-container",
                                scale=1,
                            )
                            refresh_backgrounds_btn = gr.Button("↻", scale=0, min_width=48)

                    with gr.Group(visible=False) as lyria_settings:
                        gr.Markdown("#### Lyria RealTime Tuning")
                        lyria_bpm = gr.Slider(60, 200, 70, step=1, label="BPM")
                        with gr.Row():
                            lyria_density = gr.Slider(0, 1.0, 0.1, step=0.05, label="Density")
                            lyria_brightness = gr.Slider(0, 1.0, 0.15, step=0.05, label="Brightness")

                # Section 3: Mix & Voice Effects (Identical to Manual)
                with gr.Accordion("Mix & Space Effects", open=False, elem_classes="accordion-section"):
                    with gr.Row():
                        speed_slider = gr.Slider(0.70, 1.20, 0.90, step=0.01, label="Speech Speed", info="0.85–0.95 is ideal for meditation.")
                        duck_slider = gr.Slider(-30, -6, -16, step=1, label="Music Ducking (dB)", info="How low bed drops during speech.")
                    with gr.Row():
                        spectral_duck_checkbox = gr.Checkbox(
                            label="Spectral Ducking", value=False,
                            info="Duck only mid band — preserves bass warmth + air.",
                            elem_classes="toggle-switch",
                        )
                        shared_reverb_checkbox = gr.Checkbox(
                            label="Shared Reverb", value=False,
                            info="Sit music in voice IR room for acoustic cohesion.",
                            elem_classes="toggle-switch",
                        )
                    with gr.Row():
                        reverb_slider = gr.Slider(0.0, 0.50, 0.15, step=0.05, label="Reverb Amount")
                        reverb_ir_dropdown = gr.Dropdown(
                            choices=[
                                ("Warm Studio", "warm_studio"),
                                ("Wooden Hall", "wooden_hall"),
                                ("Stone Chapel", "stone_chapel"),
                            ],
                            value="warm_studio",
                            label="Space / IR",
                            elem_classes="dropdown-container",
                        )
                    with gr.Row():
                        fade_in_slider = gr.Slider(0, 10, 1.5, step=0.5, label="Fade In (s)")
                        fade_out_slider = gr.Slider(0, 15, 6, step=0.5, label="Fade Out (s)")

                # Section 4: Advanced Engine Options
                with gr.Accordion("Advanced Engine Options", open=False, elem_classes="accordion-section"):
                    with gr.Row():
                        format_radio = gr.Radio(["wav", "mp3"], value="wav", label="Format", elem_classes="pill-radio")
                        seed_input = gr.Number(label="Seed", value=0, precision=0)
                    with gr.Row():
                        upsample_checkbox = gr.Checkbox(label="Hi-Fi (48 kHz)", value=True, elem_classes="toggle-switch")
                        stems_checkbox = gr.Checkbox(label="Export Stems", value=True, elem_classes="toggle-switch")
                    stem_separation_checkbox = gr.Checkbox(label="Clean Music (Source Separation)", value=True, elem_classes="toggle-switch")
                    with gr.Row():
                        quality_mode_checkbox = gr.Checkbox(label="High Quality (Best-of-3)", value=False, elem_classes="toggle-switch")
                        stereo_output_checkbox = gr.Checkbox(label="Stereo Output", value=False, elem_classes="toggle-switch")

                # Section 5: Promoted Models Directory
                with gr.Accordion("Promoted Models Directory", open=False, elem_classes="accordion-section"):
                    promoted_models_html = gr.HTML(format_promoted_models_html())
                    with gr.Row():
                        demote_model_dropdown = gr.Dropdown(
                            choices=[mid for _, mid in list_all_engines() if mid not in ("f5", "kokoro")],
                            value=None,
                            label="Select Promoted Model to Remove",
                            elem_classes="dropdown-container",
                            scale=3,
                        )
                        demote_btn = gr.Button("Demote Model", variant="secondary", scale=1, min_width=110)

        # ── Interactive Event Wiring ──────────────────────────────────────────

        def on_benchmark_change(choice: str):
            """Switch between 5-min guided meditation and 5-min sleep story scripts."""
            is_sleep = "Sleep Story" in choice
            content_type = "Sleep Story" if is_sleep else "Guided Meditation"
            script = SLEEP_STORY_5MIN_SCRIPT if is_sleep else MEDITATION_5MIN_SCRIPT
            p = get_profile(normalize_content_type(content_type))
            return (
                script,
                content_type,
                gr.update(value=p["speed"]),
                gr.update(value=p["duck_amount_db"]),
                gr.update(value=p["reverb_amount"]),
                gr.update(value=p["fade_in_sec"]),
                gr.update(value=p["fade_out_sec"]),
            )

        benchmark_script_choice.change(
            fn=on_benchmark_change,
            inputs=[benchmark_script_choice],
            outputs=[script_input, content_type_dropdown, speed_slider, duck_slider, reverb_slider, fade_in_slider, fade_out_slider],
        )

        def on_reset_script(choice: str):
            is_sleep = "Sleep Story" in choice
            return SLEEP_STORY_5MIN_SCRIPT if is_sleep else MEDITATION_5MIN_SCRIPT

        reset_script_btn.click(fn=on_reset_script, inputs=[benchmark_script_choice], outputs=[script_input])

        def on_content_type_change(content_label: str):
            p = get_profile(normalize_content_type(content_label))
            return (
                gr.update(value=p["speed"]),
                gr.update(value=p["duck_amount_db"]),
                gr.update(value=p["reverb_amount"]),
                gr.update(value=p["fade_in_sec"]),
                gr.update(value=p["fade_out_sec"]),
            )

        content_type_dropdown.change(
            fn=on_content_type_change,
            inputs=[content_type_dropdown],
            outputs=[speed_slider, duck_slider, reverb_slider, fade_in_slider, fade_out_slider],
        )

        def on_mode_change(mode: str):
            is_inst = mode == "Instrumental Only"
            is_voc = mode == "Vocals Only"
            return (
                gr.update(visible=not is_inst),   # script_input
                gr.update(visible=not is_voc),    # music_prompt
                gr.update(visible=is_inst),       # music_duration
                gr.update(visible=not is_voc),    # music_stem_output
            )

        generation_mode.change(
            fn=on_mode_change,
            inputs=[generation_mode],
            outputs=[script_input, music_prompt, music_duration, music_stem_output],
        )

        def on_model_source_change(src: str):
            is_custom = src == "New Open-Source Test"
            return (
                gr.update(visible=not is_custom),  # builtin_group
                gr.update(visible=is_custom),      # custom_model_group
            )

        model_source_radio.change(
            fn=on_model_source_change,
            inputs=[model_source_radio],
            outputs=[builtin_group, custom_model_group],
        )

        def on_engine_change(engine: str):
            info = get_engine_info(engine)
            base = info.get("base_engine", engine) if info else engine
            is_kokoro = base == "kokoro"
            presets = get_model_presets(engine)

            # Pre-fill tuned presets if available
            speed_up = gr.update(value=presets.get("speed", 0.90))
            duck_up = gr.update(value=presets.get("duck_amount_db", -16.0))
            reverb_up = gr.update(value=presets.get("reverb_amount", 0.15))
            wpm_up = gr.update(value=presets.get("target_wpm", 0))
            cfg_up = gr.update(value=presets.get("cfg_strength", 2.0))

            return (
                gr.update(visible=is_kokoro),     # kokoro_group
                gr.update(visible=not is_kokoro), # f5_group
                speed_up,
                duck_up,
                reverb_up,
                wpm_up,
                cfg_up,
            )

        engine_dropdown.change(
            fn=on_engine_change,
            inputs=[engine_dropdown],
            outputs=[kokoro_group, f5_group, speed_slider, duck_slider, reverb_slider, f5_wpm_slider, f5_cfg_slider],
        )

        def on_music_engine_change(model: str, mode: str):
            is_lyria = model == "Lyria RealTime"
            is_upload = model == "Background Music"
            is_voc = mode == "Vocals Only"
            return (
                gr.update(visible=is_lyria and not is_voc),
                gr.update(visible=is_upload and not is_voc),
            )

        music_model_dropdown.change(
            fn=on_music_engine_change,
            inputs=[music_model_dropdown, generation_mode],
            outputs=[lyria_settings, upload_settings],
        )

        def on_refresh_backgrounds():
            choices = scan_backgrounds()
            return gr.update(
                choices=choices if choices else ["(no tracks found)"],
                value=choices[0][1] if choices else None,
                interactive=bool(choices),
            )

        refresh_backgrounds_btn.click(fn=on_refresh_backgrounds, outputs=[uploaded_music])

        # Generate click
        generate_btn.click(
            fn=sandbox_generate_handler,
            inputs=[
                benchmark_script_choice,
                generation_mode,
                script_input,
                content_type_dropdown,
                music_prompt,
                music_duration,
                model_source_radio,
                engine_dropdown,
                kokoro_voice_dropdown,
                f5_voice_dropdown,
                custom_model_id,
                custom_display_name,
                custom_base_engine,
                custom_voice_slug,
                f5_wpm_slider,
                f5_cfg_slider,
                microprosody_checkbox,
                df_wet_slider,
                music_model_dropdown,
                uploaded_music,
                speed_slider,
                duck_slider,
                spectral_duck_checkbox,
                shared_reverb_checkbox,
                reverb_slider,
                reverb_ir_dropdown,
                fade_in_slider,
                fade_out_slider,
                format_radio,
                seed_input,
                upsample_checkbox,
                stems_checkbox,
                stem_separation_checkbox,
                quality_mode_checkbox,
                stereo_output_checkbox,
                lyria_bpm,
                lyria_density,
                lyria_brightness,
            ],
            outputs=[audio_output, vocal_stem_output, music_stem_output, status_display, scorecard_display],
            show_progress="full",
        )

        # Promote click
        promote_btn.click(
            fn=handle_promote_model,
            inputs=[
                model_source_radio,
                engine_dropdown,
                custom_model_id,
                custom_display_name,
                custom_base_engine,
                custom_voice_slug,
                f5_voice_dropdown,
                speed_slider,
                f5_wpm_slider,
                f5_cfg_slider,
                reverb_slider,
                reverb_ir_dropdown,
                duck_slider,
                spectral_duck_checkbox,
                shared_reverb_checkbox,
                microprosody_checkbox,
                df_wet_slider,
                content_type_dropdown,
            ],
            outputs=[promotion_status, engine_dropdown, promoted_models_html],
        )

        # Discard click
        discard_btn.click(
            fn=handle_discard_model,
            outputs=[audio_output, vocal_stem_output, music_stem_output, promotion_status, status_display, scorecard_display],
        )

        # Demote click
        demote_btn.click(
            fn=handle_demote_model,
            inputs=[demote_model_dropdown],
            outputs=[promotion_status, engine_dropdown, promoted_models_html],
        )

    return {
        "benchmark_script_choice": benchmark_script_choice,
        "reset_script_btn": reset_script_btn,
        "content_type_dropdown": content_type_dropdown,
        "generation_mode": generation_mode,
        "script_input": script_input,
        "music_prompt": music_prompt,
        "music_duration": music_duration,
        "generate_btn": generate_btn,
        "audio_output": audio_output,
        "vocal_stem_output": vocal_stem_output,
        "music_stem_output": music_stem_output,
        "status_display": status_display,
        "promote_btn": promote_btn,
        "discard_btn": discard_btn,
        "promotion_status": promotion_status,
        "scorecard_display": scorecard_display,
        "model_source_radio": model_source_radio,
        "engine_dropdown": engine_dropdown,
        "kokoro_voice_dropdown": kokoro_voice_dropdown,
        "f5_voice_dropdown": f5_voice_dropdown,
        "custom_model_id": custom_model_id,
        "custom_display_name": custom_display_name,
        "custom_base_engine": custom_base_engine,
        "custom_voice_slug": custom_voice_slug,
        "f5_wpm_slider": f5_wpm_slider,
        "f5_cfg_slider": f5_cfg_slider,
        "microprosody_checkbox": microprosody_checkbox,
        "df_wet_slider": df_wet_slider,
        "speed_slider": speed_slider,
        "duck_slider": duck_slider,
        "reverb_slider": reverb_slider,
        "reverb_ir_dropdown": reverb_ir_dropdown,
        "fade_in_slider": fade_in_slider,
        "fade_out_slider": fade_out_slider,
        "stems_checkbox": stems_checkbox,
        "upsample_checkbox": upsample_checkbox,
    }
