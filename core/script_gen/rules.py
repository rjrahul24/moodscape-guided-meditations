"""Assemble system prompts from the on-disk prompting guides.

Guides are read at call time, not import time, so edits to
docs/prompting_guides/ take effect on the next generation with no code change
and no restart.
"""

from pathlib import Path

GUIDES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "prompting_guides"
)

GUIDE_TEMPLATE = "vocal_{content_type}_{engine}_instructions.md"
SAFETY_RULES_FILENAME = "content_safety_rules.md"


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompting guide not found: {path}. Expected it at {path.name} "
            f"inside {path.parent}."
        )
    return path.read_text(encoding="utf-8").strip()


def load_guide(
    engine: str, content_type: str, guides_dir: Path | None = None
) -> str:
    """Load the engine- and content-type-specific formatting guide."""
    root = guides_dir if guides_dir is not None else GUIDES_DIR
    filename = GUIDE_TEMPLATE.format(content_type=content_type, engine=engine)
    return _read(root / filename)


def load_safety_rules(guides_dir: Path | None = None) -> str:
    """Load the mental-health content safety rules shared by both passes."""
    root = guides_dir if guides_dir is not None else GUIDES_DIR
    return _read(root / SAFETY_RULES_FILENAME)


def _duration_clause(target_min_sec: float, target_max_sec: float) -> str:
    return (
        f"The finished audio must run between {target_min_sec / 60:.0f} and "
        f"{target_max_sec / 60:.0f} minutes. Spoken delivery is roughly 95-100 "
        "words per minute and explicit [pause:Xs] markers add their full "
        "duration, so budget both words and silence deliberately."
    )


def build_generator_system_prompt(
    engine: str,
    content_type: str,
    target_min_sec: float,
    target_max_sec: float,
    guides_dir: Path | None = None,
) -> str:
    """System prompt for pass 1 — writing the draft script."""
    return "\n\n".join(
        [
            "You write production-ready scripts for a text-to-speech "
            "meditation pipeline. Your output is consumed directly by a TTS "
            "engine, so it must obey the formatting contract exactly.",
            _duration_clause(target_min_sec, target_max_sec),
            "# Formatting guide\n\n" + load_guide(engine, content_type, guides_dir),
            "# Content safety rules\n\n" + load_safety_rules(guides_dir),
            "Output the script and nothing else. No preamble, no explanation, "
            "no markdown, no surrounding quotes.",
        ]
    )


def build_judge_system_prompt(
    engine: str,
    content_type: str,
    target_min_sec: float,
    target_max_sec: float,
    guides_dir: Path | None = None,
) -> str:
    """System prompt for pass 2 — independent review that revises."""
    return "\n\n".join(
        [
            "You are an independent reviewer of scripts written for a "
            "text-to-speech meditation pipeline. You did not write the draft "
            "you are given. Your job is to return a corrected script, not a "
            "score and not a critique.",
            "Read the draft against the formatting guide and the safety rules "
            "below. Fix every violation you find. Preserve what works — do not "
            "rewrite wholesale when a targeted edit will do.",
            _duration_clause(target_min_sec, target_max_sec),
            "# Formatting guide\n\n" + load_guide(engine, content_type, guides_dir),
            "# Content safety rules\n\n" + load_safety_rules(guides_dir),
            "Respond in exactly this format and nothing else:\n\n"
            "<script>\n"
            "the full corrected script\n"
            "</script>\n"
            "<changelog>\n"
            "- one line per change you made, or 'no changes needed'\n"
            "</changelog>",
        ]
    )
