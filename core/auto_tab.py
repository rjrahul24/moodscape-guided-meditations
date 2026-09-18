"""Gradio Auto-Generate tab, built as a callable so it can be tested.

app.py cannot be imported in a test — it loads torch and registers
atexit.register(lambda: os._exit(0)), which would hijack pytest's exit. Keeping
the tab here means its construction is covered; app.py only wires it in.
"""

import os

import gradio as gr

from core.auto_generate import (
    DEFAULT_GENERATOR,
    DEFAULT_JUDGE,
    DEFAULT_PLANNER,
    AutoConfig,
)
from core.genres import genre_choices, load_pack
from core.streaming_run import StreamingRun

BAND_CHOICES = [
    ("3–6 min", "short"),
    ("6–10 min", "medium"),
    ("10–15 min", "long"),
]


def genre_dropdown_choices() -> list[tuple[str, str]]:
    """Flatten the family grouping into Gradio's (label, value) pairs.

    Gradio dropdowns have no option groups, so the family is folded into the
    visible label. Sorting by family keeps 46 entries scannable.
    """
    return [
        (f"{family} — {label}", slug)
        for family, entries in genre_choices()
        for label, slug in entries
    ]


def content_type_for_genre(slug: str) -> str:
    """The audio profile a genre renders as — used to pre-fill the dropdown."""
    return load_pack(slug).content_type


def auto_generate_handler(
    genre,
    band,
    steer,
    content_type,
    tts_engine,
    planner_spec,
    generator_spec,
    judge_spec,
):
    """Genre + length -> finished meditation, streaming progress to the UI."""
    os.environ["MOODSCAPE_SCRIPT_PLANNER"] = planner_spec
    os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = generator_spec
    os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge_spec

    if not genre:
        yield None, "", "", "Pick a genre first."
        return

    # content_type comes from the dropdown, which the genre change handler
    # pre-filled from the pack. The user's override, if any, wins -- run()
    # never re-derives it.
    config = AutoConfig.from_genre(
        load_pack(genre), band=band, content_type=content_type,
        tts_engine=tts_engine,
    )

    run = StreamingRun("", genre=genre, steer=steer, config=config)
    for update in run:
        yield None, "", "", update.message

    if run.result is None:
        # Guard on result, not on the truthiness of run.error: an exception
        # with an EMPTY message sets run.error = "", which is falsy, so a
        # truthiness check here would fall through to `run.result` (still
        # None) and raise AttributeError inside this Gradio generator
        # instead of reporting the failure.
        message = run.error if run.invalid_input else f"Failed: {run.error}"
        yield None, "", "", message
        return

    result = run.result
    status = (
        f"Done. {result.genre} / {result.angle}. "
        f"Background: {result.background}. "
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
            "Pick a genre and length. A script is written, independently "
            "reviewed, checked, and rendered with a random background track — "
            "no further input needed."
        )
        with gr.Row():
            # ── Left column: Creative Canvas ──────────────────────────────
            with gr.Column(scale=3, elem_classes="canvas-zone"):
                genre = gr.Dropdown(
                    choices=genre_dropdown_choices(),
                    value="stress_relief",
                    label="Genre",
                    elem_classes="dropdown-container",
                )
                band = gr.Radio(
                    choices=BAND_CHOICES,
                    value="medium",
                    label="Length",
                )
                with gr.Accordion(
                    "Steer this one", open=False, elem_classes="accordion-section"
                ):
                    steer = gr.Textbox(
                        label="Anything else? (optional)",
                        placeholder="by the ocean · for a night shift",
                        lines=2,
                    )
                button = gr.Button(
                    "Generate", variant="primary", elem_classes="primary-btn"
                )
                audio = gr.Audio(
                    label="Result", type="filepath", elem_classes="music-player-glass"
                )
                status = gr.Textbox(label="Status", lines=4, interactive=False)
                with gr.Accordion("Script", open=False, elem_classes="accordion-section"):
                    script = gr.Textbox(
                        label="Final script", lines=20, interactive=False
                    )
                with gr.Accordion(
                    "Judge changelog", open=False, elem_classes="accordion-section"
                ):
                    changelog = gr.Textbox(
                        label="Changes", lines=8, interactive=False
                    )

            # ── Right column: Settings Sidebar ────────────────────────────
            with gr.Column(scale=2, elem_classes="settings-sidebar"):
                with gr.Accordion(
                    "Content & Voice", open=True, elem_classes="accordion-section"
                ):
                    content_type = gr.Dropdown(
                        choices=["meditation", "sleep_story"],
                        value="meditation",
                        label="Content Type",
                        elem_classes="dropdown-container",
                    )
                    tts_engine = gr.Dropdown(
                        choices=["f5", "kokoro"],
                        value="f5",
                        label="Voice Engine",
                        elem_classes="dropdown-container",
                    )

                with gr.Accordion(
                    "Models", open=False, elem_classes="accordion-section"
                ):
                    planner = gr.Textbox(
                        label="Planner model",
                        value=os.environ.get(
                            "MOODSCAPE_SCRIPT_PLANNER", DEFAULT_PLANNER
                        ),
                    )
                    generator = gr.Textbox(
                        label="Generator model",
                        value=os.environ.get(
                            "MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR
                        ),
                    )
                    judge = gr.Textbox(
                        label="Judge model",
                        value=os.environ.get("MOODSCAPE_SCRIPT_JUDGE", DEFAULT_JUDGE),
                    )

        genre.change(
            fn=content_type_for_genre, inputs=[genre], outputs=[content_type]
        )
        button.click(
            fn=auto_generate_handler,
            inputs=[genre, band, steer, content_type, tts_engine,
                    planner, generator, judge],
            outputs=[audio, script, changelog, status],
        )

    return {
        "genre": genre, "band": band, "steer": steer,
        "content_type": content_type, "tts_engine": tts_engine,
        "planner": planner, "generator": generator, "judge": judge, "button": button,
        "audio": audio, "status": status, "script": script, "changelog": changelog,
    }
