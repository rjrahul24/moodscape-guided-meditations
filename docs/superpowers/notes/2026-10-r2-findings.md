# R2 findings: F5 memory, chunk budget and speech QA

## R2-0 memory measurement (2026-10-08)

**Setup:**
- F5 (`F5Engine("Brittney")`) only: no music, Demucs or mix.
- Script: the R1 `r1_brittney` deep_sleep sleep story (8 speech chunks from
  `prepare_segments`, 199.7 s of audio), at speed 0.75 and seed 1500.
- Run under `/usr/bin/time -l` and the RAM lock.
- Logged after every `infer`: `torch.mps.driver_allocated_memory()` and
  `current_allocated_memory()`.

| Measure | Value |
|---|---|
| Model load | 8.3 s; driver 3.41 GB, current 1.16 GB |
| TTS wall time | 321.9 s for 199.7 s of audio (RTF 1.61) |
| Driver allocation across chunks | 4.63 → 10.18 → 13.14 → **13.53 GB** (grows) |
| Current (live tensor) allocation | 1.16 GB, flat throughout |
| After synth / after `empty_cache()` | 13.12 GB / 9.85 GB driver |
| Whisper turbo loaded alongside F5 | driver 14.32 GB, current 2.78 GB (+1.6 GB) |
| Whisper on 20 s of speech | 10.8 s on MPS (includes model load) |
| Process peak memory footprint | **17.1 GB** (max RSS 2.2 GB) |

**Readings:**
- **The driver allocation is MPS cache growth, not live memory.** Live
  tensors stay at 1.16 GB while the driver pool grows by about 1.1 GB per
  chunk. → **R2-2 calls `torch.mps.empty_cache()` after each breath group.**
- **The F5 stage alone peaks around 17 GB.** The 33–40 GB whole-pipeline
  peaks measured in Track Q include music loading and resampling, Demucs
  (skipped for uploads), the 48 kHz mix and the export. So the TTS stage
  isn't where the R1 MPS OOM headroom went. Freeing the F5 cache before the
  pipeline unloads TTS still lowers the floor the later stages start from.
- **Whisper co-residency costs about 1.6 GB live**, so 17.1 + 2 is well
  under 30 GB. → **The QA transcriber defaults to `mps`**, and
  `MOODSCAPE_F5_QA_DEVICE=cpu` remains the escape hatch.

## Chunk budget (confirms the roadmap's corrected cause)

- **Reference clip:** Brittney, 100 bytes of transcript over the
  conditioned (ref-padded) **11.44 s** clip. `assets["duration_sec"]`
  equals the conditioned file's duration, so pre-flight ruling 3 is
  satisfied.
- **Upstream's budget at speed 0.75:**
  `int(100 / 11.44 × (22 − 11.44) × 0.75) = 69` bytes. Ours is
  `int(… × 0.9) = 62`.
- **The run's own log shows the hidden re-splitting.** Every
  `prepare_segments` chunk (up to 250 characters) became **2–4 upstream
  batches**, for example "Generating audio in 4 batches…" for a 4-sentence
  paragraph. Each batch was joined with a fixed 0.15 s crossfade, under one
  seed, outside our QA. This is the behaviour `MOODSCAPE_F5_CHUNKING`
  replaces.

## Smoke renders (2026-10-08/09)

**Setup:** box_breathing, short band, seed 1700. The writer produced a
long script (707 s), so this is a stress case.

| Arm | Wall (s) | Peak footprint | LUFS (dual-mono) | Whisper gaps | speech_qa |
|---|---|---|---|---|---|
| base (LLM + F5, flags off) | 1654 | n/a | −16.0 | 5 | — |
| rise (render-only, `BED_RISE_LIMIT=1`) | 68 | n/a | −16.0 | 5 | — (TTS cache hit, no F5) |
| flags-off render-only re-render | 1037 | **43.2 GB** | −16.0 | 10 | — |
| rise_r2, first QA design | 1871 | 42.8 GB | −16.0 | 7 | 29/72 failed, 20 re-rendered, 14 still failing |
| rise_r2, word-gap QA (final) | 1407 | 42.8 GB | −16.0 | 5 | **13/72 failed, 11 re-rendered, 4 still failing** |

- **The loudness fix works.** Every arm measures −16.0 dual-mono, where R1
  measured −13.
- **Chunk ownership works.** Every F5 call in the R2 arms was "1 batch", so
  upstream never re-split.
- **The first QA design misfired.** The signal-only gap rule flagged F5's
  natural comma and sentence pauses (0.6–1.7 s at speed 0.8) and the
  −46…−57 dB ref-pad lead-in. On 20 instrumented chunks it flagged 13; the
  Whisper word-gap rule (`gap_breakdown`) flagged 1, a real 0.61 s stall.
  QA now uses the word-gap rule (commit 2dc20ac).
- **R2 adds no peak memory.** The whole pipeline peaks at about 43 GB with
  or without R2. The F5 stage alone peaks at 17 GB, so the overshoot past
  32 GB comes from the stages after TTS. That pre-existing overshoot is the
  most likely cause of R1's MPS OOM, and it's the top reliability item for a
  later round.
- **R2's cost** is about +40% TTS time on this script (1295 s against 925 s
  for the same script with flags off).
- **F5 is nondeterministic on identical inputs** (5 vs 10 Whisper gaps).
  Single-render gap counts are noise; the A/B decides.

## Found in passing (for R3)

Every F5 chunk begins with about **1.5–2.2 s of near-silence**. It is the
ref-pad leak: `MOODSCAPE_F5_REF_PAD` ships ON, and it happens on the golden
path too. That is dead air before each sentence, on top of the script's own
pauses. It probably contributes to the runtime overshoot and to the pacing
listeners hear. Trimming leading near-silence per chunk (or modelling it in
the timing engine) belongs to R3 and needs its own A/B.
