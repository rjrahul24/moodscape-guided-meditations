"""The fixed evaluation test set (eval/testset.toml)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TESTSET = Path(__file__).resolve().parent.parent.parent / "eval" / "testset.toml"


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # stop pytest collecting this dataclass as a test class

    id: str
    genre: str
    band: str
    seed: int


def load_testset(path: Path | None = None) -> list[TestCase]:
    """Expand genres x bands into cases; validate bands and genre packs."""
    from core.auto_generate import DURATION_BANDS
    from core.genres import load_pack

    data = tomllib.loads(Path(path or DEFAULT_TESTSET).read_text(encoding="utf-8"))
    bands = list(data.get("bands", []))
    unknown = [b for b in bands if b not in DURATION_BANDS]
    if unknown:
        raise ValueError(f"Unknown duration band(s) {unknown}; expected {sorted(DURATION_BANDS)}.")
    cases: list[TestCase] = []
    for entry in data.get("cases", []):
        genre = entry["genre"]
        load_pack(genre)  # raises for an unknown or malformed pack
        for offset, band in enumerate(bands):
            cases.append(TestCase(f"{genre}-{band}", genre, band, int(entry["seed"]) + offset))
    return cases
