"""REVIEW.md and listen.html measurement card for an eval, keyed by an EvalSpec.

Read-only over the eval directory except for REVIEW.md and the injected card in
listen.html. Pure functions over directories (no model or audio imports).
"""

from __future__ import annotations

import html
import json
import re
import statistics
from pathlib import Path

from core.eval.chain import EvalSpec

MARK_START, MARK_END = "<!-- machine-summary:start -->", "<!-- machine-summary:end -->"


def _rows(out: Path, arm: str) -> list[dict]:
    try:
        rows = json.loads((out / arm / "results.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def _get(row: dict, *path):
    cur = row.get("metrics") or {}
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
    return cur


def _med(values) -> str:
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return f"{statistics.median(vals):.2f}" if vals else "-"


def _metas(out: Path, arm: str) -> list[dict]:
    metas = []
    for p in sorted((out / arm / "eval").glob("*/*.meta.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            metas.append(data)
    return metas


def voice_qa_totals(out: Path, arm: str) -> dict:
    tot = {"chunks": 0, "failed_first_pass": 0, "rerendered": 0, "still_failing": 0, "errors": 0, "files": 0}
    for meta in _metas(out, arm):
        qa = meta.get("speech_qa")
        if not isinstance(qa, dict):
            continue
        tot["files"] += 1
        if "error" in qa:
            tot["errors"] += 1
            continue
        for k in ("chunks", "failed_first_pass", "rerendered", "still_failing"):
            tot[k] += int(qa.get(k, 0) or 0)
    return tot


def script_controls(out: Path, arm: str) -> dict | None:
    """Summarise duration_control / content_check from an arm's meta files; None if no meta carries them."""
    dc = [m["duration_control"] for m in _metas(out, arm) if isinstance(m.get("duration_control"), dict) and m["duration_control"]]
    cc = [m["content_check"] for m in _metas(out, arm) if isinstance(m.get("content_check"), dict) and m["content_check"]]
    if not dc and not cc:
        return None
    scales = [d.get("pause_scale") for d in dc if isinstance(d.get("pause_scale"), (int, float))]
    return {
        "dc_files": len(dc),
        "length_repairs": sum(1 for d in dc if d.get("length_repair")),
        "pause_scaled": sum(1 for s in scales if abs(s - 1.0) > 1e-9),
        "pause_range": (min(scales), max(scales)) if scales else None,
        "cc_files": len(cc),
        "cc_ran": sum(1 for c in cc if c.get("ran")),
        "findings_first": sum(int(c.get("findings_first", 0) or 0) for c in cc),
        "findings_final": sum(int(c.get("findings_final", 0) or 0) for c in cc),
    }


def build_review_lines(spec: EvalSpec, out: Path) -> list[str]:
    arms = list(spec.arms)
    page = out / "listen.html"
    lines = [
        f"# {spec.title}: start here", "",
        f"**Open `{page.name}` in this folder** (double-click it; it opens in your browser). Answers save as you go; "
        "press **Download my answers** at the end and tell me.", "",
        "For every pair pick **A**, **lean A**, **same**, **lean B** or **B** - use a lean whenever you have even a slight preference, "
        "and keep **same** for when you truly can't tell. Then tick why.", "",
        "## What each round decides", "",
        "| Round | Pairs | You're comparing | Listen for |", "|---|---|---|---|",
    ]
    for r in spec.rounds:
        key = out / r.id / ".key.json"
        try:
            k = json.loads(key.read_text()) if key.is_file() else None
        except (OSError, json.JSONDecodeError):
            k = None
        if k is not None:
            n = len(k.get("pairs", {}))
            skipped = k.get("skipped_cases") or []
            note = f" (skipped: {', '.join(skipped)})" if skipped else ""
            lines.append(f"| {r.name} | {n}{note} | `{r.a}` vs `{r.b}`, random order | {r.listen_for} |")
        else:
            lines.append(f"| {r.name} | **not built** (an arm failed; see warnings) | `{r.a}` vs `{r.b}` | - |")

    lines += ["", "## What the machine measured (median per file)", "",
              "| Arm | What it adds | Done | Length (min) | LUFS as played | Mid-word gaps | Whisper CER | Voice from cache |",
              "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        rows = _rows(out, a.name)
        ok = [r for r in rows if r.get("ok")]
        cached = sum(1 for r in ok if r.get("tts_cache_hit") is True)
        lines.append(
            f"| `{a.name}` | {a.desc} | {len(ok)}/{len(rows) or '?'} | "
            f"{_med([(_get(r, 'mix', 'actual_sec') or 0) / 60 for r in ok])} | "
            f"{_med([_get(r, 'mix', 'integrated_lufs_dual_mono') for r in ok])} | "
            f"{_med([_get(r, 'speech', 'internal_gaps') for r in ok])} | "
            f"{_med([_get(r, 'speech', 'cer') for r in ok])} | {cached}/{len(ok)} |")

    cached_arms = [a.name for a in arms if a.cached]
    fresh_arms = [a.name for a in arms if not a.cached]
    how = ["", "How to read it:",
           "- **LUFS as played** should be -16 (-18 for a sleep story) in every arm."]
    if cached_arms:
        how.append("- **Voice from cache** should be all-cached for " + ", ".join(f"`{n}`" for n in cached_arms)
                   + " - cached arms reuse an earlier arm's voice render (their `from` / cache source). "
                   + "A round is same-voice only when both its arms share one render. "
                   + ", ".join(f"`{n}`" for n in fresh_arms) + " render a fresh voice (0 cached is expected).")
    how.append("- **Mid-word gaps** vary a lot between two F5 renders of the same script, so small differences mean nothing; "
               "your ears decide.")
    lines += how + [""]

    qa_arms = [(a.name, voice_qa_totals(out, a.name)) for a in arms]
    qa_arms = [(n, q) for n, q in qa_arms if q["files"]]
    lines += ["## Voice QA", ""]
    if qa_arms:
        lines += ["| Arm | Files | Chunks | Failed first check | Re-rendered | Still failing | QA errors |", "|---|---|---|---|---|---|---|"]
        for n, q in qa_arms:
            lines.append(f"| `{n}` | {q['files']} | {q['chunks']} | {q['failed_first_pass']} | {q['rerendered']} | {q['still_failing']} | {q['errors']} |")
    else:
        lines.append("No arm recorded per-chunk voice QA.")

    ctl = [(a.name, script_controls(out, a.name)) for a in arms]
    ctl = [(n, c) for n, c in ctl if c]
    if ctl:
        lines += ["", "## Script controls (duration control / content check)", "",
                  "| Arm | Length repairs | Pauses scaled | Pause scale range | Content check ran | Findings first | Findings final |",
                  "|---|---|---|---|---|---|---|"]
        for n, c in ctl:
            rng = f"{c['pause_range'][0]:.2f}-{c['pause_range'][1]:.2f}" if c["pause_range"] else "-"
            lines.append(f"| `{n}` | {c['length_repairs']}/{c['dc_files']} | {c['pause_scaled']}/{c['dc_files']} | {rng} | "
                         f"{c['cc_ran']}/{c['cc_files']} | {c['findings_first']} | {c['findings_final']} |")

    lines += ["", "## Per case (length in minutes / mid-word gaps)", ""]
    cases = sorted({r.get("case") for a in arms for r in _rows(out, a.name) if r.get("case")})
    lines += ["| Case | " + " | ".join(a.name for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for case in cases:
        cells = []
        for a in arms:
            row = next((r for r in _rows(out, a.name) if r.get("case") == case), None)
            if row is None:
                cells.append("-")
            elif not row.get("ok"):
                cells.append("**failed**")
            else:
                cells.append(f"{(_get(row, 'mix', 'actual_sec') or 0) / 60:.1f} / {_get(row, 'speech', 'internal_gaps')}")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")

    warnings = []
    log = out / "chain.log"
    if log.is_file():
        warnings += [l.strip() for l in log.read_text().splitlines() if "WARNING" in l or ("exit=" in l and "exit=0" not in l)]
    for a in arms:
        rows = _rows(out, a.name)
        if not rows:
            warnings.append(f"{a.name}: produced no results (arm missing or crashed; see {a.name}.log)")
        for r in rows:
            if not r.get("ok"):
                warnings.append(f"{a.name} / {r.get('case')}: {r.get('error', 'failed')}")
    lines += ["", "## Warnings", ""] + ([f"- {w}" for w in warnings] or ["None - every arm completed every case."])

    ids = ", ".join(f"`{r.id}/`" for r in spec.rounds) or "none"
    lines += ["", "## After listening", "",
              "Press **Download my answers**, then tell me. I'll import the answers, score each round "
              "(SHIP needs >=60% of decided pairs and p < 0.05; leans are reported separately), and recommend which flags to turn on.",
              "", f"Files: `listen.html` (the page), {ids} (blind pairs; `.key.json` reveals which is which - please don't open it "
              "before listening), one folder per arm with full renders, scripts and `metrics.json`, `chain.log` (timeline)."]
    return lines


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def md_to_html(md_lines: list[str]) -> str:
    """Tiny converter for the markdown build_review_lines writes: ## headings, | tables |, - bullets, paragraphs."""
    out, i = [], 0
    while i < len(md_lines):
        line = md_lines[i]
        if line.startswith("## "):
            out.append(f"<h3>{_inline(line[3:])}</h3>")
        elif line.startswith("|"):
            rows = []
            while i < len(md_lines) and md_lines[i].startswith("|"):
                cells = [c.strip() for c in md_lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append('<div class="tbl"><table><thead><tr>' + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
            continue
        elif line.startswith("- "):
            items = []
            while i < len(md_lines) and md_lines[i].startswith("- "):
                items.append(f"<li>{_inline(md_lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)


def inject_into_page(page: Path, md_lines: list[str]) -> bool:
    """Put the machine summary before <div id="rounds"> as a collapsed card (idempotent). Returns True if written."""
    if not page.is_file():
        return False
    text = page.read_text(encoding="utf-8")
    if MARK_START in text and MARK_END in text:
        a, b = text.index(MARK_START), text.index(MARK_END) + len(MARK_END)
        text = text[:a] + text[b:]
    anchor = '<div id="rounds"></div>'
    if anchor not in text:
        return False
    style = ("<style>.tbl{overflow-x:auto}.tbl table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}"
             ".tbl th,.tbl td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}"
             "details.card summary{cursor:pointer;font-weight:600}details.card h3{font-size:15px;margin:16px 0 4px}"
             "code{font-size:12px}</style>")
    block = (f"{MARK_START}{style}<details class=\"card\"><summary>What the machine measured "
             f"(optional - read after listening if you prefer to stay unbiased)</summary>{md_to_html(md_lines)}</details>{MARK_END}")
    page.write_text(text.replace(anchor, block + anchor, 1), encoding="utf-8")
    return True


def write_review(spec: EvalSpec, out: Path | None = None) -> Path:
    out = Path(out if out is not None else spec.out)
    lines = build_review_lines(spec, out)
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    start = next(i for i, l in enumerate(lines) if l.startswith("## What the machine measured"))
    end = next(i for i, l in enumerate(lines) if l.startswith("## After listening"))
    inject_into_page(out / "listen.html", lines[start:end])
    return out / "REVIEW.md"
