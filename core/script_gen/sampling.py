"""Per-stage sampling parameters for script-generation model calls.

Values come from the model cards (Qwen3.x non-thinking: 0.7 / 0.80 / presence
1.5; Gemma 4: 1.0 / 0.95). top_k and min_p are deliberately absent: the
installed Ollama Modelfiles already set them (qwen top_k 20, gemma top_k 64),
and the OpenAI-compatible /v1 endpoint cannot override them anyway.
"""

import math
import os
from dataclasses import asdict, dataclass

STAGES = ("planner", "writer", "judge", "repair")


@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float | None = None
    presence_penalty: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = SamplingParams(1.0)

# (family prefix, stages, params) — first match wins.
_TABLE: tuple[tuple[str, frozenset[str], SamplingParams], ...] = (
    ("qwen3", frozenset({"planner", "writer"}), SamplingParams(0.7, 0.80, 1.5)),
    ("qwen3", frozenset({"judge", "repair"}), SamplingParams(0.7, 0.80, 0.0)),
    ("gemma4", frozenset(STAGES), SamplingParams(1.0, 0.95, None)),
)

_FIELDS = ("temperature", "top_p", "presence_penalty")


def _bare_model(engine_name: str) -> str:
    model = engine_name.split(":", 1)[1] if ":" in engine_name else engine_name
    return model.rsplit("/", 1)[-1].lower()


def _env_value(var: str, current, *, nullable: bool = True):
    raw = os.environ.get(var)
    if raw is None:
        return current
    text = raw.strip().lower()
    if text in ("", "none", "null"):
        if not nullable:
            raise ValueError(f"Environment variable {var}={raw!r}: temperature cannot be unset; give a number.")
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Environment variable {var}={raw!r} is not a number (or 'none').") from exc
    if not math.isfinite(value):
        raise ValueError(f"Environment variable {var}={raw!r} must be a finite number.")
    return value


def sampling_for(stage: str, engine_name: str) -> SamplingParams:
    """Sampling for one pipeline stage of one model, with env overrides."""
    if stage not in STAGES:
        raise ValueError(f"Unknown sampling stage {stage!r}; expected one of {STAGES}.")
    model = _bare_model(engine_name)
    params = DEFAULT
    for prefix, stages, candidate in _TABLE:
        if model.startswith(prefix) and stage in stages:
            params = candidate
            break
    values = params.as_dict()
    for field in _FIELDS:
        values[field] = _env_value(
            f"MOODSCAPE_SAMPLING_{stage.upper()}_{field.upper()}", values[field],
            nullable=field != "temperature",
        )
    return SamplingParams(**values)
