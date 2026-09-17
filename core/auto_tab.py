"""Gradio Auto-Generate tab, built as a callable so it can be tested.

app.py cannot be imported in a test — it loads torch and registers
atexit.register(lambda: os._exit(0)), which would hijack pytest's exit. Keeping
the tab here means its construction is covered; app.py only wires it in.
"""

import os

import gradio as gr

from core.auto_generate import AutoConfig
from core.streaming_run import StreamingRun

DEFAULT_GENERATOR = "ollama:llama3.2:3b"
DEFAULT_JUDGE = "ollama:llama3.2:3b"


def auto_generate_handler(
    prompt,
    content_type,
    tts_engine,
    target_min_min,
    target_max_min,
    generator_spec,
    judge_spec,
):
    """Prompt -> finished meditation, streaming progress to the UI."""
    os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = generator_spec
    os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge_spec

    config = AutoConfig(
        content_type=content_type,
        tts_engine=tts_engine,
        target_min_sec=float(target_min_min) * 60.0,
        target_max_sec=float(target_max_min) * 60.0,
    )

    run = StreamingRun(prompt, config=config)
    for update in run:
        yield None, "", "", update.message

    if run.error:
        yield None, "", "", f"Failed: {run.error}"
        return

    result = run.result
    status = (
        f"Done. Background: {result.background}. "
        f"Estimated {result.estimated_sec / 60:.1f} min."
    )
    advisories = "\n".join(f"- [{v.code}] {v.message}" for v in result.violations)
    if advisories:
        status = f"{status}\n\nAdvisories:\n{advisories}"

    yield result.audio_path, result.script, result.changelog, status


def build_auto_tab() -> dict:
    """Construct the Auto-Generate tab. Call inside a gr.Blocks() context.

    Returns:
        A dict of the created components, keyed by role, so callers (and tests)
        can reach them without depending on layout order.
    """
    with gr.Tab("Auto-Generate"):
        gr.Markdown(
            "Describe how you feel. A script is written, independently "
            "reviewed, checked, and rendered with a random background track — "
            "no further input needed."
        )
        prompt = gr.Textbox(
            label="What do you need?",
            placeholder="I'm feeling anxious. I need a relaxing meditation.",
            lines=3,
        )
        with gr.Row():
            content_type = gr.Dropdown(
                choices=["meditation", "sleep_story"],
                value="meditation",
                label="Content Type",
            )
            tts_engine = gr.Dropdown(
                choices=["f5", "kokoro"], value="f5", label="Voice Engine"
            )
        with gr.Row():
            target_min = gr.Slider(1, 20, value=5, step=1, label="Min minutes")
            target_max = gr.Slider(1, 30, value=7, step=1, label="Max minutes")
        with gr.Row():
            generator = gr.Textbox(
                label="Generator model",
                value=os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR),
            )
            judge = gr.Textbox(
                label="Judge model",
                value=os.environ.get("MOODSCAPE_SCRIPT_JUDGE", DEFAULT_JUDGE),
            )
        button = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Result", type="filepath")
        status = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Accordion("Script", open=False):
            script = gr.Textbox(label="Final script", lines=20, interactive=False)
        with gr.Accordion("Judge changelog", open=False):
            changelog = gr.Textbox(label="Changes", lines=8, interactive=False)

        button.click(
            fn=auto_generate_handler,
            inputs=[prompt, content_type, tts_engine, target_min, target_max,
                    generator, judge],
            outputs=[audio, script, changelog, status],
        )

    return {
        "prompt": prompt, "content_type": content_type, "tts_engine": tts_engine,
        "target_min": target_min, "target_max": target_max,
        "generator": generator, "judge": judge, "button": button,
        "audio": audio, "status": status, "script": script, "changelog": changelog,
    }
