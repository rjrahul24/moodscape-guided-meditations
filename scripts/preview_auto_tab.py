"""Standalone preview harness for the Auto-Generate tab.

`core/auto_tab.py::build_auto_tab()` is not wired into `app.py` on this
branch — `app.py` currently holds a third party's uncommitted work, so this
branch never touches it. This script mounts `build_auto_tab()` in its own
`gr.Blocks()` so the tab can be run and visually verified in a browser
without booting the full app (and its models).

Usage:
    .venv/bin/python scripts/preview_auto_tab.py

Serves at http://localhost:7861 (NOT 7860 — that's the real app's port, so
both can run side by side).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from core.auto_tab import build_auto_tab  # noqa: E402


def build_preview() -> gr.Blocks:
    """Build the standalone preview harness Blocks."""
    with gr.Blocks(title="MoodScape Auto-Generate — Preview Harness") as demo:
        gr.Markdown(
            "# Auto-Generate Preview Harness\n"
            "This is a **dev preview**, not the real MoodScape app. It mounts "
            "only the Auto-Generate tab for standalone visual verification."
        )
        build_auto_tab()
    return demo


def main() -> None:
    demo = build_preview()
    demo.launch(server_port=7861)


if __name__ == "__main__":
    main()
