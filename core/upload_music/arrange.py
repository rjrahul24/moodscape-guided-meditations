"""Length-fitting helpers for uploaded instrumentals.

An uploaded instrumental can be any length, but the pipeline needs a music
array of a specific duration (mono float32 @ 48 kHz, exactly as long as the
ACE-Step / Lyria engines would produce).  `fit_to_length` adapts the upload:

- LONGER than the target → trimmed (the master fade-out covers the cut).
- SHORTER than the target → looped with equal-power crossfades at every seam.
  The crossfade auto-shrinks if the source is too short to support a 500 ms
  seam without smearing content, and falls back to plain tiling for very
  short sources.
- EXACTLY the target → used as-is.

A `FitReport` is returned alongside the audio so the pipeline can surface
"looped N times" / "trimmed" in the status message.

Ported and adapted (mono-first) from the reference meditation_mixer.arrange.
Fades and pre/post-roll are intentionally NOT applied here — `core.mixer.mix`
already handles roll and fades for every music source, so applying them here
would double-fade the uploaded bed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FitReport:
    mode: Literal["used_as_is", "trimmed", "looped", "tiled_no_xfade"]
    loops: int               # how many times the source is repeated (>= 1)
    source_seconds: float
    target_seconds: float
    crossfade_ms: float      # actual crossfade used (may be < requested)


def _equal_power_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power fade-out and fade-in curves of length n."""
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    return (
        np.cos(t * np.pi / 2.0).astype(np.float32),
        np.sin(t * np.pi / 2.0).astype(np.float32),
    )


def fit_to_length(
    bg: np.ndarray,
    sr: int,
    target_samples: int,
    crossfade_ms: float = 500.0,
) -> tuple[np.ndarray, FitReport]:
    """Fit ``bg`` to exactly ``target_samples`` samples.

    Accepts mono 1-D or (channels, samples) 2-D input; output preserves the
    channel layout as (channels, target_samples). The caller is responsible
    for any mono squeeze.

    Returns ``(audio, FitReport)``.
    """
    if bg.ndim == 1:
        bg = bg[np.newaxis, :]
    if bg.dtype != np.float32:
        bg = bg.astype(np.float32, copy=False)
    n_ch, n_bg = bg.shape

    if n_bg == 0:
        raise ValueError("Uploaded instrumental is empty.")
    if target_samples < 0:
        raise ValueError(f"target_samples must be >= 0, got {target_samples}")

    source_s = n_bg / sr
    target_s = target_samples / sr

    # Empty target.
    if target_samples == 0:
        return (
            np.zeros((n_ch, 0), dtype=np.float32),
            FitReport("used_as_is", 1, source_s, target_s, 0.0),
        )

    # Fast path: exactly the right length.
    if n_bg == target_samples:
        return (bg.copy(), FitReport("used_as_is", 1, source_s, target_s, 0.0))

    # Fast path: bg longer than target → just trim.
    if n_bg > target_samples:
        return (
            bg[..., :target_samples].copy(),
            FitReport("trimmed", 1, source_s, target_s, 0.0),
        )

    # bg is shorter than target → loop with crossfades.
    requested_xfade = int(sr * crossfade_ms / 1000.0)
    # Keep at least 50 % of each loop pristine: cap xfade at 25 % of bg.
    xfade = min(requested_xfade, max(0, n_bg // 4))
    # Don't bother with xfades shorter than ~4 ms — gives clicks anyway.
    if xfade < int(sr * 0.004):
        xfade = 0

    out = np.zeros((n_ch, target_samples), dtype=np.float32)

    if xfade <= 0:
        # Source too short for a meaningful crossfade; tile and truncate.
        n_repeats = -(-target_samples // n_bg)  # ceil divide
        tiled = np.tile(bg, (1, n_repeats))
        out[:] = tiled[..., :target_samples]
        return out, FitReport("tiled_no_xfade", n_repeats, source_s, target_s, 0.0)

    # Pre-allocated O(N) construction. Each new loop overwrites the last
    # `xfade` samples of the previous placement with a crossfade, then writes
    # bg[xfade:] as fresh body.
    fade_out, fade_in = _equal_power_curves(xfade)

    first_take = min(n_bg, target_samples)
    out[..., :first_take] = bg[..., :first_take]
    pos = first_take
    loops = 1

    bg_body = bg[..., xfade:]            # length n_bg - xfade
    bg_head = bg[..., :xfade]            # length xfade

    while pos < target_samples:
        loops += 1
        overlap_start = pos - xfade
        out[..., overlap_start:pos] = (
            out[..., overlap_start:pos] * fade_out + bg_head * fade_in
        )
        remaining = target_samples - pos
        take = min(bg_body.shape[-1], remaining)
        out[..., pos:pos + take] = bg_body[..., :take]
        pos += take

    return out, FitReport("looped", loops, source_s, target_s, xfade * 1000.0 / sr)
