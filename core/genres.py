"""The genre registry: what a genre is, and what it decides.

A genre pack is a TOML file holding two kinds of thing:

  * Curated creative material -- technique, session arc, imagery angles,
    banned phrasings, safety caveats -- that the planner adapts per run. This
    is what lets a mid-sized local model write a credible grief meditation:
    it adapts known-good material rather than inventing doctrine.
  * Deterministic fields the pipeline reads directly -- content_type,
    music_tags, pause_ratio. Because these are decided here rather than by a
    model, the planner's output can stay plain prose with nothing to parse.

Packs are read at call time, so editing one takes effect on the next run with
no restart -- the same contract rules.py uses for the prompting guides.
"""

import os
import random
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.background_tags import DECLARED_VOCAB, MEASURED_VOCAB
from core.content_profiles import CONTENT_PROFILES

# MOODSCAPE_GENRE_PACKS_DIR lets a harness or an experiment point at an
# alternative pack set without copying the repository.
GENRE_PACKS_DIR = Path(
    os.environ.get(
        "MOODSCAPE_GENRE_PACKS_DIR",
        Path(__file__).resolve().parent.parent / "docs" / "genre_packs",
    )
)

MAX_PAUSE_RATIO = 0.6
MIN_ANGLES = 2
MIN_ARC_STEPS = 3

_KNOWN_TAGS = MEASURED_VOCAB | DECLARED_VOCAB

_REQUIRED = (
    "label", "family", "content_type", "music_tags", "pause_ratio",
    "technique", "arc", "safety",
)


class GenrePackError(Exception):
    """A pack is missing, malformed, or references something unknown."""


@dataclass(frozen=True)
class Angle:
    """One distinct treatment of a genre: an arc and imagery set."""

    name: str
    imagery: tuple[str, ...]


@dataclass(frozen=True)
class GenrePack:
    """Everything a genre decides."""

    slug: str
    label: str
    family: str
    content_type: str
    music_tags: tuple[str, ...]
    pause_ratio: float
    technique: str
    arc: tuple[str, ...]
    safety: str
    banned: tuple[str, ...]
    angles: tuple[Angle, ...]


def _resolve(packs_dir: Path | None) -> Path:
    return packs_dir if packs_dir is not None else GENRE_PACKS_DIR


def _require_str(data: dict, key: str, path: Path) -> str:
    """Validate that a field is a string."""
    value = data[key]
    if not isinstance(value, str):
        raise GenrePackError(
            f"{path}: field {key!r} must be a string, got "
            f"{type(value).__name__} ({value!r})."
        )
    return value


def _require_str_list(data: dict, key: str, path: Path) -> tuple[str, ...]:
    """Validate that a field is an array of strings.

    Rejects a bare string explicitly: tuple("abc") yields ('a','b','c'),
    which would silently pass a length check and ship a corrupt pack.
    """
    value = data[key]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise GenrePackError(
            f"{path}: field {key!r} must be an array of strings, got "
            f"{type(value).__name__} ({value!r}). Write it as "
            f"{key} = [\"one\", \"two\"]."
        )
    for item in value:
        if not isinstance(item, str):
            raise GenrePackError(
                f"{path}: every entry of {key!r} must be a string, got "
                f"{type(item).__name__} ({item!r})."
            )
    return tuple(value)


def _require_float(data: dict, key: str, path: Path) -> float:
    """Validate that a field is a number."""
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenrePackError(
            f"{path}: field {key!r} must be a number, got "
            f"{type(value).__name__} ({value!r})."
        )
    return float(value)


def load_pack(slug: str, packs_dir: Path | None = None) -> GenrePack:
    """Load and validate one pack.

    Validation is strict and happens at load, not mid-run: a typo'd tag
    should fail before a model is loaded, not after five minutes of
    generation.
    """
    path = _resolve(packs_dir) / f"{slug}.toml"
    if not path.is_file():
        raise GenrePackError(
            f"No genre pack for {slug!r}. Expected {path}."
        )

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise GenrePackError(f"{path} is not valid TOML: {exc}") from exc

    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        raise GenrePackError(
            f"{path} is missing required field(s): {', '.join(missing)}."
        )

    # Type validation happens first, so type errors are never misreported.
    label = _require_str(data, "label", path)
    family = _require_str(data, "family", path)
    content_type = _require_str(data, "content_type", path)
    music_tags = _require_str_list(data, "music_tags", path)
    pause_ratio = _require_float(data, "pause_ratio", path)
    technique = _require_str(data, "technique", path)
    arc = _require_str_list(data, "arc", path)
    safety = _require_str(data, "safety", path)

    # Membership and range checks now that types are guaranteed.
    if content_type not in CONTENT_PROFILES:
        raise GenrePackError(
            f"{path} names content_type {content_type!r}, which is not one of "
            f"{sorted(CONTENT_PROFILES)}."
        )

    unknown = sorted(set(music_tags) - _KNOWN_TAGS)
    if unknown:
        raise GenrePackError(
            f"{path} uses unknown music tag(s): {', '.join(unknown)}. "
            f"Known tags: {', '.join(sorted(_KNOWN_TAGS))}."
        )

    if not 0.0 <= pause_ratio <= MAX_PAUSE_RATIO:
        raise GenrePackError(
            f"{path} has pause_ratio {pause_ratio}, outside 0-{MAX_PAUSE_RATIO}."
        )

    if len(arc) < MIN_ARC_STEPS:
        raise GenrePackError(
            f"{path} has {len(arc)} arc step(s); at least {MIN_ARC_STEPS} are "
            "needed to shape a session."
        )

    # Handle optional banned field
    banned = ()
    if "banned" in data:
        banned = _require_str_list(data, "banned", path)

    # Validate angles
    raw_angles = data.get("angles", [])
    if len(raw_angles) < MIN_ANGLES:
        raise GenrePackError(
            f"{path} has {len(raw_angles)} angles; at least {MIN_ANGLES} are "
            "needed so repeated runs of this genre can differ."
        )

    angles = []
    for idx, entry in enumerate(raw_angles):
        if not isinstance(entry, dict):
            raise GenrePackError(
                f"{path}: angle at index {idx} must be a table, got "
                f"{type(entry).__name__} ({entry!r})."
            )
        if "name" not in entry:
            raise GenrePackError(
                f"{path}: angle at index {idx} is missing required field 'name'."
            )
        angle_name = entry["name"]
        if not isinstance(angle_name, str):
            raise GenrePackError(
                f"{path}: angle at index {idx}, field 'name' must be a string, got "
                f"{type(angle_name).__name__} ({angle_name!r})."
            )
        imagery = ()
        if "imagery" in entry:
            imagery_value = entry["imagery"]
            if isinstance(imagery_value, str) or not isinstance(imagery_value, (list, tuple)):
                raise GenrePackError(
                    f"{path}: angle {angle_name!r}, field 'imagery' must be an array of strings, got "
                    f"{type(imagery_value).__name__} ({imagery_value!r})."
                )
            for img_idx, item in enumerate(imagery_value):
                if not isinstance(item, str):
                    raise GenrePackError(
                        f"{path}: angle {angle_name!r}, imagery entry {img_idx} must be a string, got "
                        f"{type(item).__name__} ({item!r})."
                    )
            imagery = tuple(imagery_value)
        angles.append(Angle(name=angle_name, imagery=imagery))

    angles = tuple(angles)
    names = [angle.name for angle in angles]
    if len(names) != len(set(names)):
        raise GenrePackError(f"{path} has duplicate angle names: {names}.")

    return GenrePack(
        slug=slug,
        label=label,
        family=family,
        content_type=content_type,
        music_tags=music_tags,
        pause_ratio=pause_ratio,
        technique=technique.strip(),
        arc=arc,
        safety=safety.strip(),
        banned=banned,
        angles=angles,
    )


def load_all_packs(packs_dir: Path | None = None) -> dict[str, GenrePack]:
    """Load every pack in the directory, keyed by slug."""
    root = _resolve(packs_dir)
    if not root.is_dir():
        raise GenrePackError(f"Genre packs directory not found: {root}.")
    return {
        path.stem: load_pack(path.stem, packs_dir=root)
        for path in sorted(root.glob("*.toml"))
    }


def genre_choices(
    packs_dir: Path | None = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Packs grouped by family for the UI dropdown.

    Returns [(family, [(label, slug), ...]), ...], families and labels both
    sorted, so 46 entries stay scannable.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for slug, pack in load_all_packs(packs_dir).items():
        grouped.setdefault(pack.family, []).append((pack.label, slug))
    return [
        (family, sorted(grouped[family])) for family in sorted(grouped)
    ]


def pick_angle(
    pack: GenrePack,
    recent: Sequence[str] = (),
    rng: random.Random | None = None,
) -> Angle:
    """Choose an angle, avoiding recently used ones.

    Mirrors background_picker's exclude-recent rule: if avoiding everything
    would leave nothing, the full set is used. Variety is a preference, not a
    reason to fail a job.
    """
    excluded = set(recent)
    candidates = [a for a in pack.angles if a.name not in excluded] or list(pack.angles)
    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
