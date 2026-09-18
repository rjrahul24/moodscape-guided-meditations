"""Automatic tagging of background instrumentals.

Genre packs name the kind of bed they want ("warm", "sparse", "drone"); this
module works out which tracks match, from the audio itself.

Two tiers of tag, with different trustworthiness:

  * ``measured`` -- written here from librosa features. Regenerating a track's
    entry overwrites these.
  * ``declared`` -- instrument/source identity a human supplies ("piano",
    "flute", "nature"). Feature extraction cannot identify instruments
    reliably, so these are never written or overwritten by code.

Tagging is lazy and incremental: a newly added track is analysed once on first
use and cached by (name, size, mtime). The user's whole workflow is "drop a
file into assets/backgrounds/".
"""

import logging

logger = logging.getLogger(__name__)

# Analysis window. 60s is long enough for stable statistics and short enough
# that a first-use analysis is ~2s; starting 25% in skips intros and fades,
# which are not representative of the bed.
ANALYSIS_SECONDS = 60.0
ANALYSIS_OFFSET_FRACTION = 0.25
ANALYSIS_SR = 22050


def extract_features(path: str) -> dict[str, float]:
    """Measure six spectral/temporal features of one track.

    Returns:
        centroid    -- Hz, brightness.
        flatness    -- spectral flatness x1000 (noise-like vs tonal).
        flux        -- mean onset strength; how much the spectrum moves.
        onset_rate  -- detected onsets per second.
        dynamics    -- RMS p95/p5 ratio; how much the level swings.
        percussive  -- fraction of energy that is percussive (HPSS).
    """
    import numpy as np
    import librosa

    duration = librosa.get_duration(path=path)
    offset = max(0.0, duration * ANALYSIS_OFFSET_FRACTION)
    y, sr = librosa.load(
        path,
        sr=ANALYSIS_SR,
        mono=True,
        offset=offset,
        duration=ANALYSIS_SECONDS,
    )

    spectrum = np.abs(librosa.stft(y))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    rms = librosa.feature.rms(S=spectrum)[0]
    _harmonic, percussive = librosa.effects.hpss(y)

    analysed_sec = len(y) / float(sr) if len(y) else 1.0

    return {
        "centroid": float(
            np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=sr))
        ),
        "flatness": float(
            np.mean(librosa.feature.spectral_flatness(S=spectrum)) * 1000.0
        ),
        "flux": float(
            np.mean(
                librosa.onset.onset_strength(
                    S=librosa.power_to_db(spectrum**2), sr=sr
                )
            )
        ),
        "onset_rate": float(len(onsets) / analysed_sec),
        "dynamics": float(
            np.percentile(rms, 95) / (np.percentile(rms, 5) + 1e-9)
        ),
        "percussive": float(
            np.sum(percussive**2) / (np.sum(y**2) + 1e-9)
        ),
    }
