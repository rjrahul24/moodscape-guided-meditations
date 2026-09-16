"""Pick a royalty-free background instrumental at random.

Used by the auto-generation path, which has no UI for choosing a track.

Track discovery is delegated to core.upload_music.scan_backgrounds — the
canonical scanner, which already handles every supported format and skips
unreadable files. This module adds only the random choice and the
exclude-recent behaviour that stops a batch landing on the same instrumental
repeatedly.
"""

import random
from collections.abc import Sequence

from core.upload_music import BACKGROUNDS_DIR, scan_backgrounds


def pick_background(
    *,
    scan=None,
    exclude: Sequence[str] = (),
    rng: random.Random | None = None,
) -> tuple[str, str]:
    """Choose one background instrumental at random.

    Args:
        scan: Zero-arg callable returning [(label, path), ...]. Defaults to
            scan_backgrounds. Injected in tests to avoid reading real audio.
        exclude: Paths of recently used tracks to avoid. If excluding them
            would leave nothing, the full pool is used instead — variety is a
            preference, not a reason to fail a job.
        rng: Inject a seeded Random for reproducible selection.

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

    excluded = set(exclude)
    candidates = [entry for entry in pool if entry[1] not in excluded] or pool

    chooser = rng if rng is not None else random
    return chooser.choice(candidates)
