# The Time-Series Map: How MoodScape Evolved

A phase-by-phase reconstruction of the project's evolution, built from the full git
history (50 commits, **2026-05-24 → 2026-06-21**), the documentation, and the code
itself. This is the backbone the four article outlines slice into.

> **Read this first.** The git history begins *already mature* — the first commit is
> 21,284 lines across 91 files with six engines scaffolded — so "Phase 0" below is
> reconstructed from the code/docs inside commit #1 and from what later got deleted,
> not from commit messages. Everything from Phase 1 on is directly traceable to a
> commit hash.

---

## At a glance

```
2026-05-24  ██████████████  Phase 0+1  Baseline (6 engines) · Kokoro voice character · cull #1 · big refactors
2026-05-25  ▏               pin protobuf
2026-05-31  ████            Phase 2     F5 mastering retune · IndexTTS pacing · QA vocals-mode · UI rework
2026-06-06  ██████          Phase 3a    KokoroV2 · MIXING OVERHAUL · uploaded-instrumental source
2026-06-07  ███             Phase 3b    background-music library + UI
2026-06-10  ████            Phase 3c    adaptive bed calibration · voice-relative VAD · remove Stitch
2026-06-11  ██████          Phase 3d    adaptive bed ON by default · ACE-Step long-form
2026-06-13  ██              Phase 4a    research experiment flags land · cull #2 (IndexTTS-2 removed)
2026-06-21  ██              Phase 4b    cull #3 (ACE-Step removed) → final stack
```

**Final shipping stack:** voice = **Kokoro + F5**; music = **Lyria RealTime** or a
**user-uploaded instrumental**. Everything else was tried and removed.

---

## Phase 0 — The baseline (captured in commit `c85832b`, 2026-05-24)

What already existed at first commit — the "exploration" the later history prunes:

| Category | Present at baseline | Fate |
|----------|--------------------|------|
| Voice TTS | Kokoro, F5, **Chatterbox**, **HeartMuLa** | Chatterbox + HeartMuLa removed same day (`8e8e5b3`) |
| Music | **ACE-Step**, Lyria | ACE-Step removed `f24dcdd` (last commit); Lyria kept |
| Abandoned scaffolding | `neural_enhancer.py` (Apollo, never wired), `session_config.py`, `stitch_client.py` | all deleted (`5962f2b`, `cd612ad`) |
| Core pipeline | `pipeline.py`, `mixer.py`, `qa_monitor.py`, `audio_processor.py` | kept & heavily refined |

The engine-agnostic contract that made all this swappable — every engine returns
mono float32 @ 24 kHz + a boolean voice-activity mask — was already in place
(`core/speech_engine.py`). *That contract is why culling engines was cheap.*

## Phase 1 — Voice identity + first cull + hygiene (2026-05-24, one very large day)

Roughly 20 commits landed this day. Three threads:

**1. Kokoro voice character.**
- `c85832b` — stress reduction/boost markers in the preprocessor: tension words get
  misaki `(-1)`, affirmation words `(+1)`, applied *after* IPA injection so the
  collision guard prevents double-wrapping Sanskrit IPA.
- `ab041ef` / `e06e70b` — **negative-weight voice blending**: the `pure_calm` preset
  *subtracts* 5% of an energetic voice, then L2-renormalizes to the primary voice's
  norm to prevent amplitude drift (`voice_manager.py::blend_with_extrapolation`).
- `19abbf3` — FX tightening from a "research blueprint": compressor −22 → −28 dB
  (catches whisper-level delivery), reverb 15% → 18% wet.

**2. First engine cull.** `8e8e5b3` — remove Chatterbox + HeartMuLa, add **IndexTTS-2**
(zero-shot cloning + emotion control). Net −1,211 lines.

**3. Structural hygiene** (the "make it maintainable" burst): delete dead code
(`5962f2b`); `acestep_engine.py` → `core/acestep/` subpackage (`5753ad7`);
`unit-tests/` → `tests/{unit,integration}` (`5020c05`); unify weights under
`models/` (`d74d455`) and assets under `assets/` (`30f4b93`); slim `CLAUDE.md` from
278 → 98 lines, extracting `COMPONENT_REGISTRY` / `TASK_ROUTING` / `GOTCHAS`
(`95d90f5`).

## Phase 2 — Voice realism pass (2026-05-31)

- `297f62a` — **F5 mastering retune**: two-stage de-esser, an "Abbey Road" parallel
  reverb (HPF 300 Hz on the wet path), master HPF 80 → 60 Hz, anti-mud 300 → 400 Hz,
  tape saturation ~15% wet, crossfade 300 → 150 ms, chunk cap 300 → 250 chars.
- `769469d` — IndexTTS rubberband pacing + DeepFilter wet-blend (full-strength
  DeepFilter stripped breath and naturalness — so it was *blended*, not applied 100%).
- `9edc0e4` — QA gains a **vocals-only mode** and widened pre-export tolerances.
- `120dcbc` — Gradio UI + status rendering rework.

## Phase 3 — The mixing overhaul + music-source pivot (2026-06-06 → 06-11)

The heart of the sound-engineering story.

- `400f65c` — **KokoroV2**: *reduce* over-denoising (the aggressive noise reduction
  was scrubbing breath and naturalness out — "AI voice"), add a de-esser and
  close-mic proximity FX.
- `8a50df0` — **mixing rewrite** fixing three reported defects at once:
  1. *Static/harshness* → remove pedalboard 0.9.23's `Limiter` (it inflates
     sub-threshold signal ~+4.75 dB and adds broadband distortion) and replace it
     with a clean 4×-oversampled `mixer.true_peak_limit()` applied at export.
  2. *Flat/weak ducking* → replace the reactive multiband ducker with
     `apply_breathing_duck` (offline pre-computed S-curve that starts falling ~600 ms
     *before* each phrase and rises in pauses).
  3. *Over-loud bed* → lower baseline `music_volume_db` −14 → −16, deeper default duck.
- `611ae59` / `a0d5ad3` / `0f08cd1` — **music-source pivot**: add uploaded-instrumental
  source, then a background-music library + UI. Stem separation is skipped for uploads
  (the file is already an instrumental).
- `cd612ad` — remove the Stitch client + the dead reactive-ducking implementations.
- `9bc4a5d` → `4efd6ed` — **adaptive per-session bed calibration**: derive
  `music_volume_db` / `duck_amount_db` from *measured* stem short-term LUFS
  (`calibrate_music_bed`) plus a voice-relative VAD threshold; then enable it by default.
- `9905319`+ — ACE-Step long-form loop mode, seed pinning, per-segment QA, seam repair
  (a last push to make ACE-Step viable before it was ultimately cut).

## Phase 4 — Final consolidation (2026-06-13 → 06-21)

- Research experiment flags land as env toggles — two ship **ON** as kill-switches
  (F5 reference-pad, char-based short-phrase pacing), the rest **OFF** with Gradio A/B
  toggles (microprosody, cfg/sway/nfe, spectral duck, shared reverb, ref-preserve-
  dynamics). See `CLAUDE.md` → "Research Experiment Flags".
- `7aca804` — **cull #2**: remove IndexTTS-2 and all artifacts.
- `f24dcdd` — **cull #3**: remove ACE-Step and all artifacts.

The project lands on its final, deliberately small stack.

---

## The through-lines (what the articles are really about)

1. **Subtraction beat addition.** The biggest wins were removals: four engines cut, a
   distorting limiter cut, over-denoising cut, `CLAUDE.md` cut by two-thirds.
2. **Clean neural output is a *starting* point, not the goal.** Realism (breath,
   micro-pitch, room) is engineered *back in* on both voice and music.
3. **The mix is where meditation audio is won or lost** — ducking, loudness staging,
   and shared acoustic space, all under a hard 36 GB memory ceiling that forces
   sequential model loading.
