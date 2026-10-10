"""Eval chains from a TOML spec: load/validate the spec and render the zsh chain script.

Pure logic (no model or audio imports). The rendered script mirrors the hand-written
R2 chain: take the heavy-model lock, run the base arm (full LLM run), re-render the other
arms from the base scripts, check the TTS cache for ``cached`` arms, export blind pairs
per round, build the listening page and write REVIEW.md.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_POSITIONS = ("start", "middle", "end")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REPO_ROOT = Path(__file__).resolve().parents[2]


class EvalSpecError(ValueError):
    """The eval spec is malformed or inconsistent."""


@dataclass(frozen=True)
class Arm:
    name: str
    desc: str = ""
    env: dict = field(default_factory=dict)
    cached: bool = False
    from_arm: str | None = None  # scripts-from arm; None = the first arm


@dataclass(frozen=True)
class Round:
    id: str
    name: str
    a: str
    b: str
    how: str = ""
    listen_for: str = ""


@dataclass(frozen=True)
class EvalSpec:
    title: str
    testset: str
    out: str
    configs: str
    positions: str
    arms: tuple
    rounds: tuple
    path: str = ""

    @property
    def base(self) -> Arm:
        return self.arms[0]

    def arm(self, name: str) -> Arm:
        return next(a for a in self.arms if a.name == name)

    def scripts_from(self, arm: Arm) -> str:
        return arm.from_arm or self.base.name

    @property
    def config_name(self) -> str:
        return self.configs.partition("=")[0] or "eval"

    @property
    def ollama_models(self) -> list[str]:
        models = []
        for part in self.configs.partition("=")[2].split("|"):
            if part.startswith("ollama:") and part[7:] not in models:
                models.append(part[7:])
        return models


def _req(data: dict, key: str, where: str = "spec") -> str:
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise EvalSpecError(f"{where}: missing required string field '{key}'")
    return val


def parse_eval_spec(data: dict, path: str = "") -> EvalSpec:
    title = _req(data, "title")
    testset = _req(data, "testset")
    out = _req(data, "out")
    configs = _req(data, "configs")
    if "=" not in configs:
        raise EvalSpecError("configs must look like 'name=provider:model|provider:model'")
    positions = str(data.get("positions", "start,middle")).replace(" ", "")
    bad = [p for p in positions.split(",") if p not in VALID_POSITIONS]
    if bad or not positions:
        raise EvalSpecError(f"positions {bad or positions!r} not in {list(VALID_POSITIONS)}")

    arms: list[Arm] = []
    for i, raw in enumerate(data.get("arms") or []):
        name = raw.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise EvalSpecError(f"arms[{i}]: name must match {_NAME_RE.pattern}, got {name!r}")
        if any(a.name == name for a in arms):
            raise EvalSpecError(f"duplicate arm name '{name}'")
        env = raw.get("env") or {}
        for k in env:
            if not _ENV_RE.match(k):
                raise EvalSpecError(f"arm '{name}': bad env var name {k!r}")
        arm = Arm(name=name, desc=str(raw.get("desc", "")), env={k: str(v) for k, v in env.items()},
                  cached=bool(raw.get("cached", False)), from_arm=raw.get("from"))
        if not arms:
            if arm.from_arm is not None:
                raise EvalSpecError(f"first arm '{name}' is the full LLM run and must not have 'from'")
            if arm.cached:
                raise EvalSpecError(f"first arm '{name}' renders fresh and cannot be 'cached'")
        elif arm.from_arm is not None and arm.from_arm not in {a.name for a in arms}:
            raise EvalSpecError(f"arm '{name}': from='{arm.from_arm}' is not an earlier arm")
        arms.append(arm)
    if not arms:
        raise EvalSpecError("spec needs at least one [[arms]] entry")

    names = {a.name for a in arms}
    rounds: list[Round] = []
    for i, raw in enumerate(data.get("rounds") or []):
        rid = raw.get("id")
        if not isinstance(rid, str) or not _NAME_RE.match(rid):
            raise EvalSpecError(f"rounds[{i}]: id must match {_NAME_RE.pattern}, got {rid!r}")
        if any(r.id == rid for r in rounds):
            raise EvalSpecError(f"duplicate round id '{rid}'")
        if rid in names:
            raise EvalSpecError(f"round id '{rid}' collides with an arm name")
        rname = _req(raw, "name", f"round '{rid}'")
        if "|" in rname:
            raise EvalSpecError(f"round '{rid}': name must not contain '|'")
        for side in ("a", "b"):
            if raw.get(side) not in names:
                raise EvalSpecError(f"round '{rid}': {side}='{raw.get(side)}' is not a known arm")
        how = str(raw.get("how", ""))
        rounds.append(Round(id=rid, name=rname, a=raw["a"], b=raw["b"], how=how,
                            listen_for=str(raw.get("listen_for", "") or how)))
    return EvalSpec(title=title, testset=testset, out=out, configs=configs, positions=positions,
                    arms=tuple(arms), rounds=tuple(rounds), path=path)


def load_eval_spec(path) -> EvalSpec:
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalSpecError(f"cannot read spec {p}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise EvalSpecError(f"spec {p} is not valid TOML: {exc}") from exc
    return parse_eval_spec(data, str(p.resolve()))


_CACHE_CHECK = '''\
for a in {arms}; do
  .venv/bin/python - "$OUT/$a/results.json" "$a" >> $OUT/chain.log 2>&1 <<'PY'
import json, sys
path, arm = sys.argv[1:3]
try:
    rows = json.load(open(path))
except Exception as exc:
    print(f"WARNING: {{arm}} results.json unreadable ({{exc}}) - cache hits unverified")
    raise SystemExit(0)
for r in rows:
    if r.get("ok") and r.get("tts_cache_hit") is not True:
        print(f"WARNING: {{arm}} {{r.get('case')}} re-rendered F5 - rounds are not same-voice for this case")
PY
done
'''


def render_chain(spec: EvalSpec, root: Path | None = None) -> str:
    q = shlex.quote
    root = root or REPO_ROOT
    base = spec.base
    stop = "; ".join(f"ollama stop {q(m)} >/dev/null 2>&1" for m in spec.ollama_models)
    stop_trap = (stop + "; ") if stop else ""
    L = [
        "#!/bin/zsh",
        f"# Generated by scripts/ab_chain.py from {spec.path or 'an eval spec'} - do not edit by hand.",
        f"# {spec.title}: {len(spec.arms)} arms. Start ONLY with the user's go-ahead.",
        f"cd {q(str(root))}",
        f"OUT={q(spec.out)}",
        'mkdir -p "$OUT"',
        'log() { echo "[chain] $* $(date)" >> $OUT/chain.log; }',
        "until mkdir var/heavy-model.lock 2>/dev/null; do sleep 30; done",
        f"echo {q(f'owner: {spec.title} chain ')}\"$(date)\" > var/heavy-model.lock/owner",
        "trap " + q(stop_trap + 'rm -rf var/heavy-model.lock; log "cleanup done"') + " EXIT",
        "EV=(--metrics --export-stems)",
        "arm() {  # arm <name> <scripts-from> [ENV=VAL ...]",
        "  local name=$1 src=$2; shift 2",
        '  log "$name start"',
        f'  env "$@" .venv/bin/python scripts/eval_genres.py --scripts-from $OUT/$src/eval --name {q(spec.config_name)} $EV --out $OUT/$name > $OUT/$name.log 2>&1',
        '  log "$name exit=$?"',
        "}",
        f'log "{base.name} start"',
    ]
    base_env = " ".join(f"{k}={q(v)}" for k, v in base.env.items())
    prefix = f"env {base_env} " if base_env else ""
    L += [
        f"{prefix}.venv/bin/python scripts/eval_genres.py --testset {q(spec.testset)} $EV \\",
        f'  --configs {q(spec.configs)} --out $OUT/{base.name} > $OUT/{base.name}.log 2>&1',
        f'log "{base.name} exit=$?"',
    ]
    if stop:
        L.append(stop)
    for a in spec.arms[1:]:
        env = "".join(f" {k}={q(v)}" for k, v in a.env.items())
        L.append(f"arm {a.name} {spec.scripts_from(a)}{env}")
    cached = [a.name for a in spec.arms if a.cached]
    if cached:
        L.append("# Cached arms must reuse the source arm's voice; warn loudly (never abort) if any case re-rendered F5.")
        L.append(_CACHE_CHECK.format(arms=" ".join(cached)).rstrip("\n"))
    L += ['log "renders done; exporting pairs"', f"P=(--positions {spec.positions} --allow-missing)"]
    for r in spec.rounds:
        L += [
            f'.venv/bin/python scripts/ab_export.py --a $OUT/{r.a}/eval --b $OUT/{r.b}/eval --out $OUT/{r.id} "${{P[@]}}" >> $OUT/export.log 2>&1',
            f'ec=$?; log "export {r.id} exit=$ec"',
        ]
    L += ["# Build the page only from rounds that produced a sheet (a round with no common cases is skipped).", "R=()"]
    for r in spec.rounds:
        L.append(f'[[ -f $OUT/{r.id}/sheet.csv ]] && R+=({q(f"{r.name}|{r.how}=")}"$OUT/{r.id}")')
    L += [
        "ec=0",
        "if (( ${#R} )); then",
        f'  .venv/bin/python scripts/ab_listen_page.py --out $OUT/listen.html --title {q(spec.title)} "${{R[@]}}" >> $OUT/export.log 2>&1',
        "  ec=$?",
        "else",
        '  log "WARNING: no round produced a sheet.csv; no listen.html"',
        "fi",
        'log "listen page exit=$ec"',
        f".venv/bin/python -I scripts/ab_review.py {q(spec.path or 'SPEC.toml')} >> $OUT/export.log 2>&1",
        'ec=$?; log "review exit=$ec"',
        'log "REVIEW.md written - eval complete"',
        "",
    ]
    return "\n".join(L)
