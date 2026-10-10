# core/eval/listen.py
"""Blind-listening page generation and answer import for A/B rounds."""

import csv
import html
import json
import os
import re
from pathlib import Path

CHOICES = (("A", "A"), ("leanA", "lean A"), ("same", "same"), ("leanB", "lean B"), ("B", "B"))
VALID_CHOICES = {v for v, _ in CHOICES}
TAGS = (
    "voice more natural", "better pacing / pauses", "better wording", "fewer clichés",
    "music balance / clarity", "fewer glitches / gaps", "right length", "calmer / more soothing",
    "wording problem / sensitive",
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "page"


def _pair_ids(ab_dir: Path) -> list[str]:
    with open(Path(ab_dir) / "sheet.csv", newline="", encoding="utf-8") as fh:
        return [row["pair"] for row in csv.DictReader(fh)]


def build_page_data(rounds: list[tuple[str, str, Path]], page_path: Path) -> list[dict]:
    base = Path(page_path).parent
    data = []
    for name, how, ab in rounds:
        ab = Path(ab)
        data.append({
            "round": name, "how": how,
            "pairs": [{"id": p,
                       "a": os.path.relpath(ab / "pairs" / f"{p}_A.wav", base),
                       "b": os.path.relpath(ab / "pairs" / f"{p}_B.wav", base)} for p in _pair_ids(ab)],
        })
    return data


def render_page(title: str, data: list[dict]) -> str:
    blob = json.dumps(data).replace("</", "<\\/")
    return (TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__STORAGE_KEY__", json.dumps("moodscape-listen-" + slug(title)).replace("</", "<\\/"))
            .replace("__CHOICES__", json.dumps([list(c) for c in CHOICES]))
            .replace("__TAGS__", json.dumps(list(TAGS)))
            .replace("__DATA__", blob))


def _read_sheet(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def import_answers(answers: dict, rounds: dict[str, Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, ab in rounds.items():
        sheet = Path(ab) / "sheet.csv"
        rows = _read_sheet(sheet)
        known = {r["pair"] for r in rows}
        prefix = name + "/"
        mine = {k[len(prefix):]: v for k, v in answers.items() if k.startswith(prefix)}
        unknown = sorted(set(mine) - known)
        if unknown:
            raise ValueError(f"Round {name!r}: pair(s) not in sheet.csv: {', '.join(unknown)}")
        n = 0
        for row in rows:
            ans = mine.get(row["pair"])
            if not ans:
                continue
            choice = ans.get("choice") or ""
            if choice and choice not in VALID_CHOICES:
                raise ValueError(f"Round {name!r} pair {row['pair']}: invalid choice {choice!r}")
            reason = (ans.get("reason") or "").strip()
            tags = ans.get("tags") or []
            if tags:
                reason = (reason + " " if reason else "") + f"[tags: {'; '.join(tags)}]"
            row["choice"], row["reason"] = choice, reason
            if choice:
                n += 1
        with open(sheet, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["pair", "choice", "reason"])
            w.writeheader()
            w.writerows({"pair": r["pair"], "choice": r.get("choice", ""), "reason": r.get("reason", "")} for r in rows)
        counts[name] = n
    return counts


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e3e1dc;--acc:#3b5bdb;--ok:#2f9e44}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d21;--ink:#ececf0;--mute:#9a9aa3;--line:#2c2c33;--acc:#7c95ff;--ok:#51cf66}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 80px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.sub{color:var(--mute);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:10px 0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.lbl{font-weight:600;margin-bottom:4px}audio{width:100%}
.choices{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;align-items:center}
.choices label{border:1px solid var(--line);border-radius:999px;padding:4px 12px;cursor:pointer}
.choices input{margin-right:6px}input[type=text]{flex:1;min-width:160px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink)}
.prog{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:8px 14px;font-weight:600;cursor:pointer}
.done{color:var(--ok);font-weight:600}
@media (max-width:640px){.row{grid-template-columns:1fr}}
</style></head><body><main>
<div class="prog"><span id="count"></span><button onclick="save()">Download my answers</button><span id="saved" class="sub"></span></div>
<h1>__TITLE__</h1>
<div class="card"><b>How this works</b> (answers auto-save, so you can stop and resume)
<ol>
<li>Use headphones. Each pair is the <i>same moment</i> from two versions, level-matched, in random order. You don't know which is old or new.</li>
<li><b>For every pair, pick one:</b> <b>A</b> or <b>B</b> if clearly better, <b>lean A</b> or <b>lean B</b> if only slightly, or <b>same</b>. Go with your gut.</li>
<li><b>Then tick why</b> (one or more boxes). That tells me what to fix next.</li>
<li>When everything is answered, press <b>Download my answers</b> and tell me.</li>
</ol></div>
<div id="rounds"></div>
</main>
<script>
const DATA=__DATA__, KEY=__STORAGE_KEY__, CHOICES=__CHOICES__, TAGS=__TAGS__;
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}count()}
function count(){let t=0,d=0;DATA.forEach(r=>r.pairs.forEach(p=>{t++;if((st[r.round+"/"+p.id]||{}).choice)d++}));
document.getElementById("count").innerHTML=d===t?`<span class="done">All ${t} answered</span>`:`${d} / ${t} answered`}
function el(h){const d=document.createElement("div");d.innerHTML=h.trim();return d.firstChild}
const R=document.getElementById("rounds");
DATA.forEach(r=>{R.appendChild(el(`<h2>${esc(r.round)} <span class="sub">(${r.pairs.length} pairs)</span></h2>`));if(r.how)R.appendChild(el(`<p class="sub">${esc(r.how)}</p>`));
r.pairs.forEach(p=>{const k=r.round+"/"+p.id,s=st[k]||{};const c=el(`<div class="card"><div class="lbl">Pair ${esc(p.id)}</div>
<div class="row"><div><div class="lbl">A</div><audio controls preload="none" src="${esc(p.a)}"></audio></div><div><div class="lbl">B</div><audio controls preload="none" src="${esc(p.b)}"></audio></div></div>
<div class="choices">${CHOICES.map(([v,l])=>`<label><input type="radio" name="${esc(k)}" value="${v}" ${s.choice===v?"checked":""}>${l}</label>`).join("")}
</div><div class="choices">${TAGS.map(t=>`<label><input type="checkbox" data-t="${esc(t)}" ${(s.tags||[]).includes(t)?"checked":""}>${esc(t)}</label>`).join("")}
<input type="text" placeholder="anything else (optional)" value="${esc(s.reason||"")}"></div></div>`);
c.querySelectorAll("input[type=checkbox]").forEach(i=>i.onchange=()=>{const cur=new Set((st[k]||{}).tags||[]);i.checked?cur.add(i.dataset.t):cur.delete(i.dataset.t);st[k]={...(st[k]||{}),tags:[...cur]};persist()});
c.querySelectorAll("input[type=radio]").forEach(i=>i.onchange=()=>{st[k]={...(st[k]||{}),choice:i.value};persist()});
c.querySelector("input[type=text]").oninput=e=>{st[k]={...(st[k]||{}),reason:e.target.value};persist()};R.appendChild(c)})});
function save(){const b=new Blob([JSON.stringify(st,null,2)],{type:"application/json"});const a=document.createElement("a");
a.href=URL.createObjectURL(b);a.download="moodscape_listening_answers.json";a.click();document.getElementById("saved").textContent="Saved to Downloads"}
count();
</script></body></html>
"""
