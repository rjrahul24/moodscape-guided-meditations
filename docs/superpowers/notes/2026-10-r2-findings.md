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
