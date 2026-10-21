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
    provider="Local Models",
):
    """Genre + length -> finished meditation, streaming progress to the UI."""
    steer = steer or ""
    planner_spec = planner_spec or os.environ.get("MOODSCAPE_SCRIPT_PLANNER", DEFAULT_PLANNER)
    generator_spec = generator_spec or os.environ.get("MOODSCAPE_SCRIPT_GENERATOR", DEFAULT_GENERATOR)
    judge_spec = judge_spec or os.environ.get("MOODSCAPE_SCRIPT_JUDGE", DEFAULT_JUDGE)

    os.environ["MOODSCAPE_SCRIPT_PLANNER"] = planner_spec
    os.environ["MOODSCAPE_SCRIPT_GENERATOR"] = generator_spec
    os.environ["MOODSCAPE_SCRIPT_JUDGE"] = judge_spec

    if not genre:
        yield None, "", "", "Pick a genre first."
        return

    from core.provider_ui import is_cloud
    if is_cloud(provider):
        tts_engine = "elevenlabs"

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
        f"Voice: {result.voice or 'default'}. "
        f"Estimated {result.estimated_sec / 60:.1f} min."
    )
    advisories = "\n".join(f"- [{v.code}] {v.message}" for v in result.violations)
    if advisories:
        status = f"{status}\n\nAdvisories:\n{advisories}"
    note = billing_note(getattr(result, "meta_path", "") or "")
    if note:
        status = f"{status}\n{note}"

    yield result.audio_path, result.script, result.changelog, status


def billing_note(meta_path: str) -> str:
    """One-line ElevenLabs billing summary from a job's meta.json ('' if none)."""
    import json
    try:
        with open(meta_path, encoding="utf-8") as fh:
            el = json.loads(fh.read()).get("elevenlabs")
    except (OSError, ValueError):
        return ""
    if not el:
        return ""
    return (f"ElevenLabs: {el.get('chars_billed', 0):,} characters billed "
            f"({el.get('cache_hits', 0)} chunks cached).")


def engine_dropdown_visibility(provider: str):
    """Show the local Voice Engine picker only for Local Models."""
    from core.provider_ui import is_cloud
    return gr.update(visible=not is_cloud(provider))


def build_auto_tab(provider=None) -> dict:
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
                        value="",
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
                    from core.engine_registry import list_all_engines
                    # The provider toggle shows/hides this container, not the
                    # dropdown itself: toggling a Dropdown's own visibility while
                    # the tab is open did not bring it back (Gradio 6.8).
                    with gr.Column() as tts_engine_group:
                        tts_engine = gr.Dropdown(
                            choices=[(label, eid) for label, eid in list_all_engines()],
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
                    planner, generator, judge]
            + ([provider] if provider is not None else []),
            outputs=[audio, script, changelog, status],
            # Same queue as the Manual tab's Generate/Remix: one render at a
            # time (32 GB RAM; remix swaps mix env vars process-wide).
            concurrency_id="render",
            concurrency_limit=1,
        )
        if provider is not None:
            provider.change(fn=engine_dropdown_visibility,
                            inputs=[provider], outputs=[tts_engine_group])

    return {
        "genre": genre, "band": band, "steer": steer,
        "content_type": content_type, "tts_engine": tts_engine,
        "tts_engine_group": tts_engine_group,
        "planner": planner, "generator": generator, "judge": judge, "button": button,
        "audio": audio, "status": status, "script": script, "changelog": changelog,
    }
