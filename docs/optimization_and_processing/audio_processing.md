# Audio Processing & Memory Optimization Guide

This document outlines the core architectural logic that powers the generation, standardizations, and export memory management of MoodScape's audio.

## 1. Sample Rate Pipeline Standardization
Generative AI audio models operate natively across a multitude of disparate sample frequencies. 
- **Voice TTS (Kokoro/Parler)**: ~24kHz natively
- **MusicGen**: ~32kHz natively

If these arrays are fed blindly into identical FX Pedals or summed together, extreme timing synchronization and pitch-shift ("chipmunking") degradation occurs. 

**MoodScape enforces an aggressive upsampling strategy:**
All generative outputs are intercepted instantly to upsample to **44.1kHz (44100Hz)**. This is processed securely via TorchAudio's `functional.resample` to maintain precision format boundaries across PyTorch CUDA/MPS configurations before Pedalboard handles FX processing on CPU paths. 

## 2. Dynamic Audio Mixing Processing
MoodScape mitigates digital overlapping via automated **Lookahead Sidechain Ducking** (ducking the music beneath the narration).
- **Offline Lookahead**: Unlike a reactive envelope follower, the complete gain curve is computed from the *entire* voice array in advance, then shifted back by **75ms** so the music starts ducking *before* the first syllable — matching broadcast/DAW behaviour.
- **Attack/Release Swell**: Ducking drops the underlying Music volume based on the Voice RMS. A minimum **500ms release time** after the narrator stops speaking guarantees a slow, peaceful music crescendo returning to baseline without a jarring "pumping" effect.
- **Spectral Masking**: To clarify vocal enunciation, MoodScape targets a `LowpassFilter` on ambient tracks to roll-off frequencies above ~3000Hz. This strips synthetic high-end "shimmer" to preserve a clean pocket for the voice array.

## 3. Streaming Exports (Memory Safeties)
Exporting long audio combinations (e.g. up to 12 hour stretches of 44.1kHz `np.float32`) quickly bottlenecks local compute memory footprint and invokes out-of-memory kernel panics.

**Chunked AudioFile Protocol:**
1. A **parameterised LUFS target** (`-16 LUFS` for Daytime Meditation, `-19 LUFS` for Sleep Journey) calculates a constant **Gain Scalar** via a lightweight meter implementation.
2. MoodScape initiates Pedalboard's underlying `AudioFile` protocol, binding directly to disk buffers.
3. The generation sweeps through the multi-hour array iteratively in **20-second chunk boundaries**.
4. The memory blocks sequentially get normalized, wrapped inside the `Gain(-3.0)` → `Compressor(-18dB, 2:1, 30ms/300ms)` → `Limiter(-0.1)` Master Effect protections, and written to disk without needing simultaneous loading.

This paradigm scales indefinitely for background batch generations.
