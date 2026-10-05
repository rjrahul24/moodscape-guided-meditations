# Wiring the Auto-Generate tab into `app.py`

**Status: applied.** Both the tab layout and the UI controls within Auto-Generate have evolved:

1. **Tab structure** (2026-09-17): both workflows as explicit siblings under `gr.Tabs()`.
2. **Auto-Generate controls** (2026-09-18): replaced free-text prompt box with genre dropdown, length band radio, steer accordion.

This doc covers both.

## Current layout

```python
with gr.Blocks(title="MoodScape — Guided Meditation Generator") as demo:

    gr.HTML(""" ...app-header: breath orb, wordmark, tagline... """)

    with gr.Tabs():
        with gr.Tab("Manual"):
            with gr.Row():
                ... # all manual UI: canvas-zone column, settings-sidebar column
                ... # generate_btn.click(...) and all other manual event wiring
        build_auto_tab()

if __name__ == "__main__":
    ...
```

- **The header (`gr.HTML(...)`) sits outside `gr.Tabs()`.** It's app chrome —
  wordmark, breath orb, tagline — not workflow content. Putting it inside a
  tab would either duplicate it per tab or make it disappear when switching
  tabs.
- **`gr.Tabs()` is an explicit container**, not implicit grouping. Both
  `gr.Tab("Manual")` and the tab `build_auto_tab()` creates are siblings
  registered under it, so Gradio renders one tab bar with two equal entries
  instead of a stray tab group below unrelated content.
- **`with gr.Tab("Manual"):` wraps the entire pre-existing manual UI** — the
  `with gr.Row():` containing the `canvas-zone` and `settings-sidebar`
  columns, and every event handler that references components defined inside
  that scope (`generate_btn.click(...)` and the rest). That whole region
  moved two indentation levels deeper (once for `gr.Tabs()`, once for
  `gr.Tab("Manual")`) with no other change — same components, same handlers,
  same order.
- **"Manual" is listed first** so it is Gradio's default-selected tab. The
  manual workflow is the established one; landing users on "Auto-Generate"
  by default would be a surprising behaviour change from what should be a
  pure layout fix.
- **`build_auto_tab()` is called as a sibling of `with gr.Tab("Manual"):`**,
  still inside `with gr.Tabs():`, immediately after the manual block closes
  and before `if __name__ == "__main__":`. It constructs its own
  `gr.Tab("Auto-Generate")` and wires its own `button.click(...)` handler
  internally — `core/auto_tab.py` is untouched by this restructuring and
  still needs no other changes elsewhere in `app.py`.
- The `from core.auto_tab import build_auto_tab` import is unchanged, still
  near the other `core` imports (`app.py:66`, alongside
  `from core.pipeline import MeditationPipeline`).

---

## Auto-Generate UI Controls

**Entry point:** Genre dropdown + Length band radio (2026-09-18). The free-text prompt box is gone.

### Layout (inside `build_auto_tab()`)

```
[Genre]                                    # gr.Dropdown, 46 genres grouped by family
[Length band radio]                        # 3–6 min / 6–10 min / 10–15 min
[Content Type selector]                    # pre-filled from genre pack, still user-changeable
[Voice Engine selector]                    # pre-filled from genre pack, still user-changeable
[Steer this one (accordion)]               # Collapsed by default; empty text box for optional freetext
[Generate button]
```

### Genre dropdown (`genre_dropdown`)

- **Options:** 46 genres in alphabetical order within each of 8 families (Sleep & Rest, Stress & Anxiety, etc.)
- **Source:** `genres.genre_choices()` — loads all packs, extracts labels grouped by family
- **Event:** `genre_dropdown.change(on_genre_change)` → pre-fills Content Type and Voice Engine from the pack

### Length band radio (`duration_band_radio`)

- **Options:** "3–6 min", "6–10 min", "10–15 min" (radio buttons, not dropdown)
- **Maps to:** `DURATION_BANDS` dict in `core/auto_generate.py` → `(target_min_sec, target_max_sec)`
- **Used by:** planner and writer as the duration budget for `pause_ratio` calculations

### Steer accordion (`steer_accordion`)

- **Label:** "Steer this one (optional)"
- **Content:** Single `gr.Textbox` (empty by default, multi-line)
- **Behavior:** If text is provided, appended to the creative brief. Empty = no steer, pure genre-driven flow
- **Use case:** Advanced users who want to nudge a particular run without redesigning the genre

### Content Type & Voice Engine overrides

Both are pre-filled from the genre pack (`pack.content_type`), and still user-editable in a dropdown:
- **Content Type:** Allows switching meditation ↔ sleep_story at render time (bypasses pack, affects pauses and music bed)
- **Voice Engine:** Allows overriding the TTS engine; the selected engine's `prepare_segments()` is still used (not the pack's choice)

Both pre-fills are set by `on_genre_change()` callback when the genre dropdown changes.

### Orchestration inside `auto_generate_handler()`

1. Read `genre_dropdown.value` → `genres.load_pack(genre_slug)` → get pack
2. Read `duration_band_radio.value` → duration band string ("short", "medium", "long")
3. Build `AutoConfig` via `AutoConfig.from_genre(pack, band=band, content_type=content_type, tts_engine=tts_engine)`
4. Read `steer_accordion` text → pass to `StreamingRun` if present
5. Create `StreamingRun` with `genre=genre_slug, steer=steer_text, config=config`
6. Stream progress via `StreamingRun`, yield audio + script on success

## If you need to restructure this again

Preserve the invariants above: header outside `gr.Tabs()`, `gr.Tabs()` as an
explicit container, "Manual" first (so it's the default tab), and
`build_auto_tab()` called as a direct child of `gr.Tabs()` — not nested
inside `gr.Tab("Manual")`, or its tab would appear inside the Manual tab
instead of beside it.

When moving the manual UI block, re-indent it as a pure whitespace change —
don't reflow or "clean up" anything inside it while shifting it. A dropped
`.click(...)` handler still imports fine; it just leaves a dead button. Verify
with more than `python -c "import app"`:

1. Confirm both tabs exist as siblings in the expected order:

   ```python
   import app, gradio as gr
   [b.label for b in app.demo.blocks.values() if isinstance(b, gr.Tab)]
   # ['Manual', 'Auto-Generate']
   ```

2. Compare `len(app.demo.fns)` before and after — it must not decrease (a
   drop means a handler didn't survive the move).

Note: `app.py` registers `atexit.register(lambda: os._exit(0))`, so `print()`
output never reaches the terminal on a normal exit — write results to a file
and read the file back, or you'll see nothing and wrongly conclude the check
failed.

## Why `app.py` itself still isn't imported in tests

`core/auto_tab.py` and `core/streaming_run.py` are fully unit-tested without
importing `app.py` (see `tests/unit/test_auto_tab.py` and
`tests/unit/test_streaming_run.py`). `app.py` cannot be imported in a test —
it loads `torch` and registers `atexit.register(lambda: os._exit(0))`, which
would hijack pytest's exit code — so layout/wiring changes here can only be
verified by actually running `python -c "import app"` plus the checks above,
outside of pytest.
