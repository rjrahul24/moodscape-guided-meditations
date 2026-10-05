"""Pick a royalty-free background instrumental at random.

Used by the auto-generation path, which has no UI for choosing a track.

Track discovery is delegated to core.upload_music.scan_backgrounds — the
canonical scanner, which already handles every supported format and skips
unreadable files. This module adds only the random choice and the
exclude-recent behaviour that stops a batch landing on the same instrumental
repeatedly.
"""

import logging
import random
from collections.abc import Sequence

from core.upload_music import BACKGROUNDS_DIR, scan_backgrounds

logger = logging.getLogger(__name__)


def pick_background(
    *,
    scan=None,
    exclude: Sequence[str] = (),
    rng: random.Random | None = None,
    prefer_tags: Sequence[str] = (),
    tag_lookup=None,
) -> tuple[str, str]:
    """Choose one background instrumental at random.

    Args:
        scan: Zero-arg callable returning [(label, path), ...]. Defaults to
            scan_backgrounds. Injected in tests to avoid reading real audio.
        exclude: Paths of recently used tracks to avoid. If excluding them
            would leave nothing, the full pool is used instead — variety is a
            preference, not a reason to fail a job.
        rng: Inject a seeded Random for reproducible selection.
        prefer_tags: Only consider tracks carrying ALL of these tags. If that
            leaves nothing, the tag filter is dropped. Genre packs supply
            these so a running meditation does not land on a sleep drone.
        tag_lookup: Callable mapping [path, ...] -> {path: [tag, ...]}.
            Defaults to background_tags.tags_for. Only called when
            prefer_tags is non-empty, so an untagged library costs nothing.

    Returns:
        (label, path) — the label is human-readable, e.g.
        "Healing Forest — 23:12", and goes into the run metadata.

    Raises:
        FileNotFoundError: If the library holds no usable tracks.
    """
    scanner = scan if scan is not None else scan_backgrounds
    pool = scanner()

    if not pool:
        raise FileNotFoundError(
            f"No background instrumentals found in {BACKGROUNDS_DIR}. "
            "Add royalty-free audio files there before auto-generating."
        )

    candidates = list(pool)

    if prefer_tags:
        lookup = tag_lookup
        if lookup is None:
            from core.background_tags import tags_for as lookup
        wanted = set(prefer_tags)
        tags = lookup([path for _label, path in candidates])
        tagged = [
            entry
            for entry in candidates
            if wanted.issubset(set(tags.get(entry[1], ())))
        ]
        if tagged:
            candidates = tagged
        else:
            logger.info(
                "No background matches tags %s; using the full library.",
                sorted(wanted),
            )

    excluded = set(exclude)
    candidates = [entry for entry in candidates if entry[1] not in excluded] or candidates

    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
