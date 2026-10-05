#!/usr/bin/env python
"""Bulk-tag the background library, or print the measured feature table.

Tagging happens automatically on first use (core/background_tags.tags_for),
so this script is never required. It exists for two cases: re-tagging
everything after changing a threshold, and printing the raw feature
distribution used to calibrate those thresholds.

    python scripts/tag_backgrounds.py            # tag anything new or changed
    python scripts/tag_backgrounds.py --force    # re-analyse every track
    python scripts/tag_backgrounds.py --report   # print the feature table
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.background_tags import (  # noqa: E402
    TAGS_PATH,
    extract_features,
    load_tags,
    tags_for,
    write_tags,
)
from core.upload_music import scan_backgrounds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-analyse every track"
    )
    parser.add_argument(
        "--report", action="store_true", help="print the raw feature table"
    )
    args = parser.parse_args()

    pool = scan_backgrounds()
    if not pool:
        print("No background tracks found.", file=sys.stderr)
        return 1
    paths = [path for _label, path in pool]

    if args.report:
        header = (
            f"{'track':44} {'centroid':>9} {'flatness':>9} {'flux':>7} "
            f"{'onset/s':>8} {'dynamics':>9} {'percussive':>11}"
        )
        print(header)
        for path in paths:
            f = extract_features(path)
            print(
                f"{Path(path).name[:44]:44} {f['centroid']:9.0f} "
                f"{f['flatness']:9.2f} {f['flux']:7.2f} {f['onset_rate']:8.2f} "
                f"{f['dynamics']:9.1f} {f['percussive']:11.3f}"
            )
        return 0

    if args.force:
        # Drop the cache keys so every track re-analyses, but keep declared
        # tags -- those are the human's and are not regenerable.
        entries = load_tags()
        for entry in entries.values():
            entry["size"] = -1
        write_tags(entries)

    tags = tags_for(paths)
    for path in paths:
        print(f"{Path(path).name[:50]:50} {', '.join(tags[path]) or '-'}")
    print(f"\nWrote {TAGS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
