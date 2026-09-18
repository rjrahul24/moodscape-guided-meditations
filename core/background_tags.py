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
import tomllib
from collections.abc import Sequence
from pathlib import Path

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


TAGS_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets" / "backgrounds" / "tags.toml"
)


def _resolve(tags_path: Path | None) -> Path:
    return tags_path if tags_path is not None else TAGS_PATH


def load_tags(tags_path: Path | None = None) -> dict[str, dict]:
    """Read tags.toml. Returns {} when the file is absent or unreadable.

    A corrupt tags file must not break generation: the worst case is that
    every track is re-analysed, which is slow but correct.
    """
    path = _resolve(tags_path)
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read %s; re-analysing all tracks.", path)
        return {}
    return data.get("tracks", {})


def _toml_string(value: str) -> str:
    """Quote a string for TOML, escaping backslashes and double quotes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def write_tags(entries: dict[str, dict], tags_path: Path | None = None) -> None:
    """Write tags.toml.

    Hand-rolled rather than using a TOML writer: stdlib tomllib is read-only,
    and the structure here is a flat table of string lists and numbers, which
    does not justify a new dependency.
    """
    path = _resolve(tags_path)
    lines = [
        "# Generated by core/background_tags.py.",
        "#",
        "# 'measured' is written from audio analysis and is overwritten when a",
        "# track changes. 'declared' is yours: instrument and source identity",
        "# that analysis cannot determine. It is never overwritten.",
        "",
    ]
    for name in sorted(entries):
        entry = entries[name]
        lines.append(f"[tracks.{_toml_string(name)}]")
        lines.append(f"measured = {_toml_list(entry.get('measured', []))}")
        lines.append(f"declared = {_toml_list(entry.get('declared', []))}")
        lines.append(f"size = {int(entry.get('size', 0))}")
        lines.append(f"mtime = {float(entry.get('mtime', 0.0)):.3f}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def tags_for(
    paths: Sequence[str],
    tags_path: Path | None = None,
    extractor=None,
) -> dict[str, list[str]]:
    """Return combined measured+declared tags for each path, tagging as needed.

    Analyses only tracks with no cache entry or whose (size, mtime) has
    changed, then writes the cache back once. A track that cannot be analysed
    gets an empty tag list rather than raising -- one corrupt file must not
    stop a generation run.
    """
    analyse = extractor if extractor is not None else extract_features
    entries = load_tags(tags_path)
    dirty = False
    result: dict[str, list[str]] = {}

    for raw_path in paths:
        path = Path(raw_path)
        name = path.name
        entry = entries.get(name)

        try:
            stat = path.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            logger.warning("Cannot stat background %s; skipping tags.", raw_path)
            result[raw_path] = []
            continue

        stale = (
            entry is None
            or int(entry.get("size", -1)) != size
            or abs(float(entry.get("mtime", -1.0)) - mtime) > 0.001
        )

        if stale:
            # Preserve the human's declared tags across re-analysis.
            declared = list(entry.get("declared", [])) if entry else []
            try:
                measured = tags_from_features(analyse(raw_path))
            except Exception:
                logger.warning(
                    "Could not analyse background %s; leaving it untagged.",
                    raw_path,
                    exc_info=True,
                )
                result[raw_path] = []
                continue
            entry = {
                "measured": measured,
                "declared": declared,
                "size": size,
                "mtime": mtime,
            }
            entries[name] = entry
            dirty = True

        result[raw_path] = sorted(
            set(entry.get("measured", [])) | set(entry.get("declared", []))
        )

    if dirty:
        write_tags(entries, tags_path)

    return result
