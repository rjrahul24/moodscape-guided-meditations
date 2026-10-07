# core/eval/report.py
"""Per-run metric collection and the markdown summary table."""

import statistics

from core.eval.mix_metrics import mix_metrics
from core.eval.script_metrics import script_metrics
from core.eval.speech_metrics import speech_metrics


def _guard(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # a metric must never fail a run row
        return {"error": f"{type(exc).__name__}: {exc}"}


def _speech(audio_path: str, script: str, transcriber) -> dict:
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")
    return speech_metrics(audio, sr, script, transcriber=transcriber)


def compute_run_metrics(
    *,
    script: str,
    draft: str | None,
    engine: str,
    content_type: str,
    audio_path: str,
    voice_stem_path: str | None,
    estimated_sec: float | None,
    transcriber,
) -> dict:
    speech_source = voice_stem_path or audio_path
    if transcriber is None:
        speech = {"skipped": "no transcriber"}
    else:
        speech = _guard(_speech, speech_source, script, transcriber)
    return {
        "script": _guard(script_metrics, script, draft=draft, engine=engine, content_type=content_type),
        "mix": _guard(mix_metrics, audio_path, estimated_sec=estimated_sec),
        "speech": speech,
        "speech_input": "voice_stem" if voice_stem_path else "mix",
    }


_COLUMNS = (
    ("slop/150w", "script", "slop_density", "{:.2f}"),
    ("edit ratio", "script", "edit_ratio", "{:.2f}"),
    ("CER", "speech", "cer", "{:.3f}"),
    ("gaps", "speech", "internal_gaps", "{:.1f}"),
    ("|1-dur ratio|", "mix", "duration_ratio", "{:.2f}"),
    ("LUFS (dual-mono)", "mix", "integrated_lufs_dual_mono", "{:.1f}"),
)


def _value(row: dict, group: str, key: str):
    value = (row.get("metrics") or {}).get(group, {}).get(key)
    if value is None or not isinstance(value, (int, float)):
        return None
    return abs(1.0 - value) if key == "duration_ratio" else float(value)


def _on_stem(row: dict) -> bool:
    return (row.get("metrics") or {}).get("speech_input") == "voice_stem"


def summarize(rows: list[dict]) -> str:
    """Markdown table: one line per config with pass rate and metric medians.

    Each cell is `median (n=count)`. Speech columns use only rows scored on the
    voice stem (mix-scored speech metrics are contaminated by the bed);
    `stem/passed` shows how many passed rows that is.
    """
    configs = list(dict.fromkeys(row["config"] for row in rows))
    header = "| config | pass | stem/passed | " + " | ".join(c[0] for c in _COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(_COLUMNS) + 3)]
    for config in configs:
        mine = [r for r in rows if r["config"] == config]
        passed = [r for r in mine if r.get("ok")]
        stem = [r for r in passed if _on_stem(r)]
        cells = []
        for _label, group, key, fmt in _COLUMNS:
            source = stem if group == "speech" else passed
            values = [v for v in (_value(r, group, key) for r in source) if v is not None]
            cells.append(f"{fmt.format(statistics.median(values))} (n={len(values)})" if values else "–")
        lines.append(f"| {config} | {len(passed)}/{len(mine)} | {len(stem)}/{len(passed)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
