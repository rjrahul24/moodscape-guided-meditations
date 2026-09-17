# Wiring the Auto-Generate tab into `app.py`

**Status: not applied.** `app.py` currently holds 37 lines of a third party's
uncommitted work. Staging or editing it here would sweep those unrelated
changes into this branch, so Task 11 stops short of touching it. The tab
itself lives entirely in `core/auto_tab.py` (built and unit-tested there);
`app.py` needs only the two-line change below. Apply it as part of Task 15
Step 2c once `app.py` is clean, or hand this file to the user to apply
themselves.

## The two-line change

**1. Add the import** beside the other `core` imports (near `app.py:62-65`,
alongside `from core.pipeline import MeditationPipeline`):

```python
from core.auto_tab import build_auto_tab
```

**2. Call it inside the existing `gr.Blocks()` context**, after the current
tab's wiring. The `with gr.Blocks(title="MoodScape — Guided Meditation
Generator") as demo:` block currently ends with the `generate_btn.click(...)`
call (`app.py:~1206-1253`), immediately before `if __name__ == "__main__":`
(`app.py:1256`). Add the call there, still indented inside the `with` block:

```python
    build_auto_tab()
```

That's it — `build_auto_tab()` constructs its own `gr.Tab("Auto-Generate")`
and wires its own `button.click(...)` handler internally, so no other part of
`app.py` needs to change. The manual tab (script textbox, sliders, etc.)
continues to work exactly as it does today; the new tab appears alongside it.

## Why this is safe to defer

`core/auto_tab.py` and `core/streaming_run.py` are fully unit-tested without
importing `app.py` (see `tests/unit/test_auto_tab.py` and
`tests/unit/test_streaming_run.py`). `app.py` cannot be imported in a test —
it loads `torch` and registers `atexit.register(lambda: os._exit(0))`, which
would hijack pytest's exit code — so this wiring step can only be verified by
actually running `python app.py` after the two lines above are applied to a
clean `app.py`.
