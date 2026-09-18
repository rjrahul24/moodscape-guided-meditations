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


# --- Tag vocabulary and thresholds --------------------------------------
#
# Thresholds are calibrated against the measured distribution of the 20
# tracks in assets/backgrounds/ as of 2026-09-17, chosen so each band holds a
# meaningful share of the library rather than being empty or catching
# everything. Re-derive with `python scripts/tag_backgrounds.py --report`.

MEASURED_VOCAB = frozenset(
    {
        "dark", "warm", "bright",
        "drone", "evolving",
        "sparse", "busy",
        "tonal", "textured",
        "struck", "sustained",
        "steady", "dynamic",
    }
)

# Instrument/source identity. Feature extraction cannot determine these
# reliably, so they are only ever written by a human.
DECLARED_VOCAB = frozenset(
    {"piano", "flute", "strings", "nature", "voice", "bells"}
)

CENTROID_DARK_BELOW = 600.0
CENTROID_BRIGHT_ABOVE = 1050.0

FLUX_DRONE_BELOW = 0.6
FLUX_EVOLVING_ABOVE = 1.9

ONSET_SPARSE_BELOW = 1.5
ONSET_BUSY_ABOVE = 5.0

FLATNESS_TONAL_BELOW = 0.15
FLATNESS_TEXTURED_ABOVE = 1.5

PERCUSSIVE_STRUCK_ABOVE = 0.040
PERCUSSIVE_SUSTAINED_BELOW = 0.010

DYNAMICS_STEADY_BELOW = 2.0
DYNAMICS_DYNAMIC_ABOVE = 3.3


def tags_from_features(features: dict[str, float]) -> list[str]:
    """Map measured features onto the measured-tag vocabulary.

    Returns a sorted list so output is stable across runs and diffs of
    tags.toml stay readable.
    """
    tags: set[str] = set()

    centroid = features["centroid"]
    if centroid < CENTROID_DARK_BELOW:
        tags.add("dark")
    elif centroid > CENTROID_BRIGHT_ABOVE:
        tags.add("bright")
    else:
        tags.add("warm")

    flux = features["flux"]
    if flux < FLUX_DRONE_BELOW:
        tags.add("drone")
    elif flux > FLUX_EVOLVING_ABOVE:
        tags.add("evolving")

    # Onset detection is meaningless on near-silent drone material: the
    # detector fires on noise floor, so a bed with flux 0.29 can report 5.47
    # onsets/s and would otherwise be tagged the busiest track in the
    # library. Gate on the same threshold that defines a drone.
    if flux >= FLUX_DRONE_BELOW:
        onset_rate = features["onset_rate"]
        if onset_rate < ONSET_SPARSE_BELOW:
            tags.add("sparse")
        elif onset_rate > ONSET_BUSY_ABOVE:
            tags.add("busy")

    flatness = features["flatness"]
    if flatness < FLATNESS_TONAL_BELOW:
        tags.add("tonal")
    elif flatness > FLATNESS_TEXTURED_ABOVE:
        tags.add("textured")

    percussive = features["percussive"]
    if percussive > PERCUSSIVE_STRUCK_ABOVE:
        tags.add("struck")
    elif percussive < PERCUSSIVE_SUSTAINED_BELOW:
        tags.add("sustained")

    dynamics = features["dynamics"]
    if dynamics < DYNAMICS_STEADY_BELOW:
        tags.add("steady")
    elif dynamics > DYNAMICS_DYNAMIC_ABOVE:
        tags.add("dynamic")

    return sorted(tags)
