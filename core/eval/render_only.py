# core/eval/render_only.py
"""Load the scripts of an earlier eval run so they can be re-rendered as-is."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceCase:
    case: str
    genre: str
    band: str
    seed: int | None
    script: str
    voice: str
    background_label: str
    background_path: str
    source_dir: Path


def load_source_cases(config_dir: Path) -> tuple[list[SourceCase], list[str]]:
    """Return (cases, skipped_messages) for every case dir under ``config_dir``.

    Seed/band/genre come from ``<config_dir>/../results.json`` when present,
    else from the meta (genre) with band "unknown" and seed None. A background
    file that no longer exists is not checked here.
    """
    config_dir = Path(config_dir)
    results: dict[str, dict] = {}
    results_path = config_dir.parent / "results.json"
    if results_path.is_file():
        try:
            for row in json.loads(results_path.read_text(encoding="utf-8")):
                results.setdefault(row.get("case"), row)
        except (OSError, ValueError):
            results = {}

    cases: list[SourceCase] = []
    skipped: list[str] = []
    if not config_dir.is_dir():
        return cases, [f"{config_dir}: not a directory"]
    for case_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        scripts = sorted(case_dir.glob("*.script.txt"))
        metas = sorted(case_dir.glob("*.meta.json"))
        if len(scripts) != 1 or len(metas) != 1:
            skipped.append(
                f"{case_dir.name}: expected exactly one *.script.txt and *.meta.json "
                f"(found {len(scripts)}/{len(metas)})"
            )
            continue
        try:
            meta = json.loads(metas[0].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            skipped.append(f"{case_dir.name}: unreadable meta ({exc})")
            continue
        row = results.get(case_dir.name, {})
        cases.append(SourceCase(
            case=case_dir.name,
            genre=row.get("genre") or meta.get("genre") or "",
            band=row.get("band") or "unknown",
            seed=row.get("seed"),
            script=scripts[0].read_text(encoding="utf-8"),
            voice=meta.get("voice") or "",
            background_label=meta.get("background") or "",
            background_path=meta.get("background_path") or "",
            source_dir=case_dir,
        ))
    return cases, skipped
