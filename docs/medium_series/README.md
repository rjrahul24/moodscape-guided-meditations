# MoodScape — Build-Journey Article Series (working material)

This folder is **writing scaffolding** for a 3–4 part technical Medium / LinkedIn
series on how MoodScape (an AI-guided-meditation audio generator) was built, with
the focus on the sound-engineering craft: generating good voice audio, generating
good music, and *mixing the two so a synthetic voice sits on a bed and breathes*.

These are **outlines, not drafts** — each file gives a title, a thesis, an ordered
set of narrative beats, and real, verbatim code snippets pulled from the repo with
`file_path:line` citations, plus a "learnings" box. Draft prose on top of them.

## The map

- [`00-timeline-map.md`](00-timeline-map.md) — the master chronological map of the
  whole project's evolution (2026-05-24 → 2026-06-21), phase by phase, with commit
  hashes. Read this first; the four articles are slices of it.

## The four articles

| # | File | Thesis in one line |
|---|------|--------------------|
| 1 | [`01-tts-gauntlet.md`](01-tts-gauntlet.md) | I auditioned six speech engines on a 32 GB Mac and shipped two — here's the selection logic. |
| 2 | [`02-humanizing-the-voice.md`](02-humanizing-the-voice.md) | Raw neural TTS is *too* clean; realism is deliberately engineered back in. |
| 3 | [`03-the-mix.md`](03-the-mix.md) | The hardest problem was sitting narration on a music bed so it breathes. *(centerpiece)* |
| 4 | [`04-architecture.md`](04-architecture.md) | Keeping a multi-engine ML audio pipeline maintainable under a hard RAM ceiling. *(optional / mergeable into #3's tail for a tight 3-part series)* |

## How the articles map to the source (appendix)

Every beat traces back to real commits, docs, or code. Quick index:

| Theme | Primary commits | Primary code | Primary docs |
|-------|-----------------|--------------|--------------|
| Engine selection & culling | `8e8e5b3`, `7aca804`, `f24dcdd` | `core/speech_engine.py`, `core/*_tts/` | `docs/COMPONENT_REGISTRY.md` |
| Voice character (blending, stress) | `c85832b`, `ab041ef`, `e06e70b`, `19abbf3` | `core/kokoro_tts/voice_manager.py`, `preprocessor.py` | `docs/prompting_guides/vocal_meditation_kokoro_instructions.md` |
| Voice realism / post-processing | `400f65c`, `297f62a`, research pass | `core/kokoro_tts/postprocessor.py`, `core/f5_tts/{engine,postprocessor}.py` | `docs/optimization_and_processing/post-processing-pipeline.md` |
| Mixing overhaul | `8a50df0`, `9bc4a5d`, `4efd6ed`, `cd612ad` | `core/mixer.py`, `core/audio_processor.py` | `docs/optimization_and_processing/audio_processing.md`, `docs/GOTCHAS.md` |
| Music-source pivot | `611ae59`, `a0d5ad3`, `f24dcdd` | `core/upload_music/`, `core/lyria/` | `CLAUDE.md` (Pipeline Flow) |
| Architecture / hardware discipline | `95d90f5`, `5753ad7`, `d74d455`, `30f4b93` | `core/pipeline.py` | `docs/ARCHITECTURE.md`, `CLAUDE.md` |

> **Reconstruction note.** The git history starts *already mature*: the first commit
> (`c85832b`, 2026-05-24) is 21,284 lines across 91 files, with six engines already
> scaffolded. So the "evaluate many engines" phase is baked into commit #1 and is
> reconstructed from the code/docs that commit contains (and from what later got
> deleted), not from commit messages. The visible git arc is mostly the *culling and
> refinement* story.
