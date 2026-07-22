# Article 4 (optional) — Architecture for a Multi-Engine ML Audio Pipeline on 36 GB

**Thesis (one line):** Shipping four generative audio models through one pipeline on a
single 36 GB machine is less an ML problem than a systems-discipline problem — memory,
sample-rate contracts, quality gates, and docs.

**Target reader takeaway:** The unglamorous engineering that keeps a multi-model audio
app runnable and maintainable: sequential model loading, a strict sample-rate pipeline,
memory-bounded export, automated QA gates, and treating documentation as a first-class
artifact. *Can be folded into Article 3's tail if you prefer a tight 3-part series.*

**Suggested length:** 1,800–2,400 words.

---

## Narrative beats (in order)

1. **Hook — the 36 GB ceiling is the architect.** You cannot hold a TTS model and a
   music model in memory simultaneously. Every major design choice — sequential
   loading, aggressive unloading, the streaming export — descends from this one number.

2. **Sequential model loading, explicitly.** The pipeline synthesizes all speech, then
   *unloads* the TTS engine (and for F5, `del`s it), forces `gc.collect()` /
   `torch.mps.empty_cache()`, and only then loads the music engine. Show the ordering.

3. **The sample-rate contract.** Engines are born at different rates (Kokoro/F5 24 kHz,
   Lyria 48 kHz). One rule keeps everything phase-clean: *standardize to a single 48 kHz
   mix rate, and never downsample then re-upsample.* TTS is upsampled once with
   `soxr_vhq` (highest-accuracy, zero-crossing safe). The uploaded-instrumental path is
   held to the *same* contract (decode → 48 kHz mono → loop/trim to exact sample count)
   which is exactly why an upload can reuse the entire Lyria mix/duck/master path
   unchanged.

4. **Memory-bounded export.** The mix is not normalized in memory and dumped — it's
   streamed out in ~20 s chunks through Pedalboard's `AudioFile`, with the LUFS gain
   pre-computed as a single scalar so the whole file never has to sit processed in RAM
   at once.

5. **Quality gates, not vibes.** A QA monitor validates the output (LUFS/true-peak/
   spectral checks) before export, with a dedicated vocals-only mode and tolerances
   widened where the earlier gates were too strict. Automated ears.

6. **Documentation as architecture.** The `CLAUDE.md` slim-down (278 → 98 lines) and the
   extraction of `COMPONENT_REGISTRY` / `TASK_ROUTING` / `GOTCHAS` is a real
   engineering decision: keep only what must be in *every* working context, push the
   encyclopedia to on-demand docs. Tie in the "research flags" pattern — hard-won
   lessons encoded as kill-switch and A/B env flags so the codebase remembers *why*.

7. **Refactors that bought leverage.** The subpackage pattern (`core/<engine>/` each
   with its own `engine`/`preprocessor`/`postprocessor`), and consolidating scattered
   weights under `models/` and assets under `assets/`. Boring, and exactly why the
   three later engine culls were one-commit deletes.

8. **Close — constraints are a design tool.** The RAM ceiling didn't limit the project;
   it *shaped* a cleaner architecture than an unconstrained one would have produced.

---

## Code snippets to include

### 1. Sequential loading: unload TTS before touching music (`core/pipeline.py:277`)

```python
if not is_instrumental:
    tts.unload_model()
    if tts_engine == "f5":
        del tts                     # F5 is heavy — drop the reference entirely
# ... then instantiate the selected music engine ...
music_engine.load_model()
gc.collect()                        # reclaim before the music model grows
```

### 2. The sample-rate rule (from `docs/optimization_and_processing/audio_processing.md`)

Quote the rule directly — it's a one-liner worth its own callout:

```
- All music-engine paths mix at 48 kHz.
- TTS (24 kHz) is upsampled to the mix rate with librosa soxr_vhq
  (highest accuracy, minimises zero-crossing errors).
- Rule: never downsample then re-upsample. Always upsample from the lower-rate source.
```

### 3. The upload contract that unlocks reuse (`docs/GOTCHAS.md`)

```
UploadMusicEngine.generate() must return mono float32 @ 48 kHz, exactly
round(total_duration_sec * 48000) samples — that is what lets the upload reuse the
Lyria mix/duck/master path unchanged. Stem separation is skipped for uploads
(guarded by `if stem_separation and not use_upload:`).
```

### 4. Streaming export sketch (`core/mixer.py:938`)

```python
def export_audio(audio, sample_rate, ..., target_lufs=-19.0):
    """Reads the mix in bounded chunks, applies a pre-computed linear LUFS gain
    and the master chain, and streams straight to file to avoid memory spikes."""
    mix_lufs_gain = calculate_loudness_gain(mastered_audio, sample_rate, target_lufs)  # single scalar
    # ... 20-second chunk streaming through pedalboard.io.AudioFile ...
```

---

## Learnings box

> - **Let the hardware ceiling design the pipeline.** Sequential load → unload → GC is a
>   feature, not a workaround.
> - **One mix sample rate, upsample-only.** A strict SR contract eliminates a whole class
>   of phase/aliasing bugs; `soxr_vhq` for the one upsample that matters.
> - **Make new sources conform to the old contract.** The uploaded-instrumental path
>   reuses the entire mix/master chain *because* it's forced to 48 kHz mono at an exact
>   sample count.
> - **Stream the export.** Pre-compute loudness as a scalar; never hold the whole
>   processed file in RAM.
> - **Docs are load-bearing.** A 98-line always-loaded `CLAUDE.md` + on-demand reference
>   docs + env-flag "research memory" is what let a solo dev move fast without regressing.
> - **Refactor for the delete.** Subpackages + consolidated `models/`/`assets/` made every
>   later engine cull a single clean commit.

## Source appendix
Commits: `95d90f5` (CLAUDE.md slim + doc extraction), `5753ad7` (subpackage refactor),
`d74d455` + `30f4b93` (models/ + assets/ consolidation), `9edc0e4` (QA vocals mode),
`5020c05` (tests layout).
Code: `core/pipeline.py` (sequential load/unload), `core/mixer.py` (`export_audio`), `core/qa_monitor.py`.
Docs: `docs/ARCHITECTURE.md`, `docs/optimization_and_processing/audio_processing.md`,
`docs/GOTCHAS.md`, `CLAUDE.md`.
