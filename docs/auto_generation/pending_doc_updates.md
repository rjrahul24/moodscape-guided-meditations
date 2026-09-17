# Pending doc updates — apply once `CLAUDE.md` / `README.md` are clean

`CLAUDE.md` and `README.md` currently carry a third party's uncommitted,
unrelated edits. Staging either file here would sweep those changes into
this branch, so the additions below were written but not applied. Task 15
Step 2d (or whoever cleans up those files) should paste each block into the
named location once the file is safe to touch — check with `git status`
first.

Each block also respects the separate, not-yet-landed RAM-figure fix
(`CLAUDE.md` lines 3 and 61, `docs/ARCHITECTURE.md` line 318, correcting
"36 GB" to "32 GB"). None of these additions touch that number or those
lines; verify that fix's status again before applying, since line numbers
below may have shifted.

---

## 1. `CLAUDE.md` — Folder Map

**Where it goes:** inside the ```` ``` ```` folder-map block, in the `core/`
section, after the `upload_music/` line (currently `CLAUDE.md` line 42).

```
│   ├── script_gen/                    # prompt → validated script (generator + judge + linter)
│   ├── auto_generate.py               # orchestrator: script → music → pipeline
│   ├── background_picker.py           # random pick from assets/backgrounds/
│   └── streaming_run.py               # threaded progress streaming for the UI
```

---

## 2. `CLAUDE.md` — new section after "Pipeline Flow"

**Where it goes:** as its own `##` section, immediately after the existing
"## Pipeline Flow (`core/pipeline.py :: MeditationPipeline.generate()`)"
section ends (currently after `CLAUDE.md` line 72, before "## Code
Conventions").

```markdown
## Auto-Generation Flow (`core/auto_generate.py :: run()`)

Prompt in, finished meditation out, with no human step.

1. **Assemble prompts** → `script_gen/rules.py` reads the engine- and
   content-type-specific guide from `docs/prompting_guides/` at call time,
   plus `content_safety_rules.md`
2. **Draft** → generator model (`MOODSCAPE_SCRIPT_GENERATOR`)
3. **Review** → an *independent* judge model (`MOODSCAPE_SCRIPT_JUDGE`)
   returns a revised script plus a changelog — it revises, it does not score
4. **Validate** → `script_gen/linter.py` (format + mental-health safety) and
   `script_gen/duration.py` (runtime estimate, no rendering)
5. **Repair** → fatal violations go back to the judge as targeted
   instructions, bounded by `MOODSCAPE_SCRIPT_MAX_REPAIRS` (default 2)
6. **Pick music** → `background_picker.pick_background()` reuses
   `upload_music.scan_backgrounds()`, excluding recently used tracks
7. **Render** → `MeditationPipeline.generate()`, unchanged, on the golden path
   (F5 + uploaded background)
8. **Persist** → `<name>.wav`, `<name>.script.txt`, `<name>.meta.json` as siblings

Full detail: [docs/auto_generation/README.md](docs/auto_generation/README.md).
```

---

## 3. `CLAUDE.md` — Top Gotchas addition

**Where it goes:** appended to the end of the "## Top Gotchas" bullet list
(currently after `CLAUDE.md` line 95, before "## Research Experiment
Flags"). Consider also updating the "The six that bite most often" count if
this makes a seventh.

```markdown
- **Fatal vs advisory violations** → `script_gen/linter.py` fails the job for
  safety hard-blocks and malformed markers, but renders anyway (with a warning)
  for duration drift and style issues. Treating every violation as fatal makes
  a weaker local model unusable; treating none as fatal lets a safety failure
  reach audio. Do not flatten this distinction.
```

---

## 4. `README.md` — new "Auto-Generate" section

**Where it goes:** as its own `##` section. The natural spot is right after
the existing "## Usage — CLI" section (currently ends at `README.md` line
182, right before the "## Script Format" `---` divider) — auto-generation is
a third way to run the app, alongside Web UI and CLI.

```markdown
## Usage — Auto-Generate

Skip writing a script by hand: describe how you feel, and a script is
drafted, independently reviewed, checked against the same safety rules as
the manual path, and rendered with a random background track — no further
input needed. This lives in the "Auto-Generate" tab and is built on
`core/auto_generate.py`.

Minimum `.env` for the default configuration (a local Ollama model as both
generator and judge):

```bash
HF_TOKEN=hf_...           # Required — same as the manual path
```

No extra key is needed for `ollama:*` specs — just have `ollama serve`
running locally with the model pulled. Using a hosted model instead requires
that provider's key (e.g. `ANTHROPIC_API_KEY` for `anthropic:*`,
`OPENROUTER_API_KEY` for `openrouter:*`) — see
[docs/auto_generation/README.md](docs/auto_generation/README.md#configuration)
for the full list.

Full subsystem reference — configuration, model spec format, the fatal/
advisory violation split, and how to benchmark model pairings — lives in
[docs/auto_generation/README.md](docs/auto_generation/README.md).
```
