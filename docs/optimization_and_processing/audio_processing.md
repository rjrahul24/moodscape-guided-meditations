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
MoodScape mitigates digital overlapping via automated Auto-Ducking (ducking the music beneath the narration). 
- **Envelope Follower**: Instead of primitive mathematical filtering, the mixer calculates the continuous Root Mean Square (RMS) measurement of the Voice Activity frame by 10ms chunked sliding windows.
- **Attack/Release Swell**: Ducking drops the underlying Music volume rapidly linearly based on the Voice RMS. Crucially, MoodScape relies on a minimum **500ms release time** after the narrator stops speaking. This guarantees a slow, peaceful music crescendo returning to baseline without a jarring "pumping" effect. 
- **Spectral Masking**: To clarify vocal enunciation, MoodScape targets a `LowpassFilter` on ambient tracks to roll-off frequencies above ~3000Hz. This strips synthetic high-end "shimmer" to preserve a clean pocket for the voice array.

## 3. Streaming Exports (Memory Safeties)
Exporting long audio combinations (e.g. up to 12 hour stretches of 44.1kHz `np.float32`) quickly bottlenecks local compute memory footprint and invokes out-of-memory kernel panics.

**Chunked AudioFile Protocol:**
1. A preliminary `-16 LUFS` (Loudness Units Full Scale) normalizing standard calculates across a lightweight meter implementation to discover a constant **Gain Scalar**.
2. MoodScape initiates Pedalboard's underlying `AudioFile` protocol, binding directly to disk buffers.
3. The generation sweeps through the multi-hour array iteratively in **20-second chunk boundaries**.
4. The memory blocks sequentially get normalized, wrapped inside the `Gain(-3.0)` and `Limiter(-0.1)` Master Effect protections, and written to disk without needing simultaneous loading.

This paradigm scales indefinitely for background batch generations.
