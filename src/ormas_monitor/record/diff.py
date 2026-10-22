"""Run diff: "what changed since the run I approved?"

`diff_runs(approved, proposed)` compares two run records using only their diaries and metrics
files (never weights, never data) and returns a documented dict (brief section 4.6):

    a, b                {path, name, model, architecture_sha256, parameters, steps, intact, complete,
                         sealed, signed, metrics, failed_checks, counts, final_loss}
                        intact = no verification check failed; signed = record.sig is present (a
                        diff takes no key, so the signature itself is not checked here)
    same_architecture   bool: the parameter-shape hashes are equal (None-safe: False if either is unknown)
    config_changes      [{key, a, b}]   optimizer.* and config.* (monitor settings) that differ
    environment_changes [{key, a, b}]   tool, python, torch, platform, model, parameters
    events              {event type: {"a": n, "b": n}}
    layers              [{layer, in_a, in_b, alarms_a, alarms_b, peak_stuck_a, peak_stuck_b,
                          stuck_at_end_a, stuck_at_end_b, repairs_a, repairs_b,
                          channels_repaired_a, channels_repaired_b, certified_a, certified_b,
                          diagnoses_a, diagnoses_b, changed}]
    worse, better, mixed  layer names whose health in B moved one way, the other, or both
    verdict             a few plain-English sentences, each a fact read from the two records

Stuck and certified counts are None for a layer the monitor did not scan for stuck channels
(only Conv/Linear producers are scanned), which is different from 0 (scanned, none stuck).
Integrity comes first: if either record fails verification, the verdict says so before anything
else, because a comparison against an edited record means nothing.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .chain import parse_lines
from .layout import LAYOUT, RecordLayout, locate
from .verify import verify_run

ENVIRONMENT_KEYS = ("tool", "python", "torch", "platform", "model", "parameters")


def _stuck_parts(v: Any) -> list[int]:
    if isinstance(v, dict):
        return [int(x) for x in v.values()]
    return [int(x) for x in v]


def _peak(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _load(path: str | Path, layout: RecordLayout) -> dict:
    loc = locate(path, layout)
    result = verify_run(path, layout=layout)
    events, _ = parse_lines(loc.diary)
    metrics: list[dict] = []
    mpath = loc.file(layout.metrics)
    if mpath is not None and mpath.exists():
        with open(mpath) as fh:
            for text in fh:
                if text.strip():
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        metrics.append(row)
    start = next((e for e in events if e.get("event") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("event") == "run_end"), None)
    losses = [r.get("loss") for r in metrics if r.get("loss") is not None]
    info = {
        "path": str(path),
        "name": start.get("name"),
        "model": start.get("model"),
        "architecture_sha256": start.get("architecture_sha256"),
        "parameters": start.get("parameters"),
        "steps": end.get("steps") if end else None,
        "intact": result.ok,
        "complete": end is not None,
        "sealed": result.check("seal").status == "pass",
        "signed": result.check("signature").status in ("pass", "unchecked"),
        "metrics": mpath is not None and mpath.exists(),
        "failed_checks": [c.name for c in result.checks if c.status == "fail"],
        "counts": end.get("counts") if end else None,
        "final_loss": losses[-1] if losses else None,
    }
    return {"info": info, "start": start, "events": events, "metrics": metrics}


def _flatten(prefix: str, obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(f"{prefix}.{k}", v))
        return out
    return {prefix: obj}


def _changes(a: dict[str, Any], b: dict[str, Any]) -> list[dict]:
    keys = list(a) + [k for k in b if k not in a]
    return [{"key": k, "a": a.get(k), "b": b.get(k)} for k in keys if a.get(k) != b.get(k)]


def _layer_stats(run: dict) -> dict[str, dict]:
    stats: dict[str, dict] = {}

    def entry(layer: str) -> dict:
        return stats.setdefault(layer, {"alarms": 0, "peak_stuck": None, "stuck_at_end": None,
                                        "repairs": 0, "channels_repaired": 0, "certified": None,
                                        "diagnoses": []})

    for row in run["metrics"]:
        for layer in (row.get("grad") or {}):
            entry(layer)
        for layer, parts in (row.get("stuck") or {}).items():
            vals = _stuck_parts(parts)
            s = entry(layer)
            s["peak_stuck"] = _peak(s["peak_stuck"], sum(vals))
            s["stuck_at_end"] = sum(vals)
            s["certified"] = _peak(s["certified"], vals[0] if vals else 0)
    for e in run["events"]:
        layer = e.get("layer")
        if not isinstance(layer, str):
            continue
        s = entry(layer)
        kind = e.get("event")
        if kind == "alarm":
            s["alarms"] += 1
            if e.get("diagnosis") not in s["diagnoses"]:
                s["diagnoses"].append(e.get("diagnosis"))
        elif kind in ("channels", "channels_cleared"):
            n = sum(int(e.get(k) or 0) for k in ("n_certified", "n_flat", "n_observed", "n_decayed"))
            s["peak_stuck"] = _peak(s["peak_stuck"], n)
            s["certified"] = _peak(s["certified"], int(e.get("n_certified") or 0))
        elif kind == "certificate":
            s["certified"] = _peak(s["certified"], int(e.get("count") or 0))
        elif kind == "repair":
            s["repairs"] += 1
            s["channels_repaired"] += int(e.get("count") or 0)
    return stats


def _gt(x: int | None, y: int | None) -> bool:
    return (x or 0) > (y or 0)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _verdict(a: dict, b: dict, same_arch: bool, config: list[dict], layers: list[dict],
             worse: list[str], better: list[str], mixed: list[str]) -> str:
    out = []
    for x, info in (("A", a), ("B", b)):
        if not info["intact"]:
            out.append(f"Run {x}'s record failed verification ({', '.join(info['failed_checks'])}), "
                       "so this comparison cannot be trusted.")
    for x, info in (("A", a), ("B", b)):
        if info["intact"] and not info["complete"]:
            out.append(f"Run {x} has no run_end: it did not finish or its record is truncated.")
    if a["architecture_sha256"] is None or b["architecture_sha256"] is None:
        out.append("The architecture of one run is not recorded.")
    elif same_arch:
        out.append("Same architecture.")
    else:
        out.append("Different architecture: the parameter shapes differ.")
    if config:
        shown = ", ".join(f"{c['key']} {c['a']!r} -> {c['b']!r}" for c in config[:3])
        more = f" and {len(config) - 3} more" if len(config) > 3 else ""
        out.append(f"{_plural(len(config), 'setting')} changed: {shown}{more}.")
    else:
        out.append("No optimizer or monitor settings changed.")

    by_name = {row["layer"]: row for row in layers}
    if worse:
        details = []
        for name in worse[:3]:
            r = by_name[name]
            if _gt(r["certified_b"], r["certified_a"]):
                details.append(f"layer {name}: certified dead channels {r['certified_a'] or 0} -> {r['certified_b']}")
            elif _gt(r["peak_stuck_b"], r["peak_stuck_a"]):
                details.append(f"layer {name}: peak stuck channels {r['peak_stuck_a'] or 0} -> {r['peak_stuck_b']}")
            elif _gt(r["stuck_at_end_b"], r["stuck_at_end_a"]):
                details.append(f"layer {name}: stuck channels at the end {r['stuck_at_end_a'] or 0} -> {r['stuck_at_end_b']}")
            else:
                details.append(f"layer {name}: alarms {r['alarms_a']} -> {r['alarms_b']}")
        more = f" (and {len(worse) - 3} more)" if len(worse) > 3 else ""
        out.append(f"Run B is less healthy than run A in {_plural(len(worse), 'layer')}: "
                   f"{'; '.join(details)}{more}.")
    if better:
        out.append(f"Run B is healthier than run A in {_plural(len(better), 'layer')}: "
                   f"{', '.join(better[:5])}{' and more' if len(better) > 5 else ''}.")
    if mixed:
        out.append(f"{_plural(len(mixed), 'layer')} changed both ways: {', '.join(mixed[:5])}"
                   f"{' and more' if len(mixed) > 5 else ''}.")
    if not worse and not better and not mixed:
        out.append("No layer's alarms, stuck channels or certified dead channels changed.")
    repairs_a = sum(r["repairs_a"] for r in layers)
    repairs_b = sum(r["repairs_b"] for r in layers)
    if repairs_a != repairs_b:
        out.append(f"Repairs: {repairs_a} in run A, {repairs_b} in run B.")
    return " ".join(out)


def diff_runs(run_a: str | Path, run_b: str | Path, layout: RecordLayout = LAYOUT) -> dict:
    """Compare run B (proposed) against run A (approved). Paths are run directories or diary files.

    Reads diaries and metrics only. Raises FileNotFoundError if either diary is missing.
    """
    ra, rb = _load(run_a, layout), _load(run_b, layout)
    a, b = ra["info"], rb["info"]
    same_arch = a["architecture_sha256"] is not None and a["architecture_sha256"] == b["architecture_sha256"]

    settings_a = {**_flatten("optimizer", ra["start"].get("optimizer") or {}),
                  **_flatten("config", ra["start"].get("config") or {})}
    settings_b = {**_flatten("optimizer", rb["start"].get("optimizer") or {}),
                  **_flatten("config", rb["start"].get("config") or {})}
    if (ra["start"].get("optimizer") is None) != (rb["start"].get("optimizer") is None):
        settings_a["optimizer"] = None if ra["start"].get("optimizer") is None else "set"
        settings_b["optimizer"] = None if rb["start"].get("optimizer") is None else "set"
    config = _changes(settings_a, settings_b)
    env = _changes({k: ra["start"].get(k) for k in ENVIRONMENT_KEYS},
                   {k: rb["start"].get(k) for k in ENVIRONMENT_KEYS})

    ca = Counter(e.get("event") for e in ra["events"])
    cb = Counter(e.get("event") for e in rb["events"])
    kinds = list(ca) + [k for k in cb if k not in ca]
    events = {str(k): {"a": ca.get(k, 0), "b": cb.get(k, 0)} for k in kinds}

    sa, sb = _layer_stats(ra), _layer_stats(rb)
    layers: list[dict] = []
    worse: list[str] = []
    better: list[str] = []
    mixed: list[str] = []
    for name in list(sa) + [n for n in sb if n not in sa]:
        x, y = sa.get(name), sb.get(name)
        row = {"layer": name, "in_a": x is not None, "in_b": y is not None}
        for key in ("alarms", "peak_stuck", "stuck_at_end", "repairs", "channels_repaired", "certified",
                    "diagnoses"):
            row[f"{key}_a"] = x[key] if x else None
            row[f"{key}_b"] = y[key] if y else None
        health = ("alarms", "peak_stuck", "stuck_at_end", "certified")
        got_worse = any(_gt(row[f"{k}_b"], row[f"{k}_a"]) for k in health)
        got_better = any(_gt(row[f"{k}_a"], row[f"{k}_b"]) for k in health)
        row["changed"] = (row["in_a"] != row["in_b"] or got_worse or got_better
                          or row["repairs_a"] != row["repairs_b"])
        if row["in_a"] and row["in_b"]:
            if got_worse and not got_better:
                worse.append(name)
            elif got_better and not got_worse:
                better.append(name)
            elif got_better and got_worse:
                mixed.append(name)
        layers.append(row)

    return {
        "a": a, "b": b,
        "same_architecture": same_arch,
        "config_changes": config,
        "environment_changes": env,
        "events": events,
        "layers": layers,
        "worse": worse,
        "better": better,
        "mixed": mixed,
        "verdict": _verdict(a, b, same_arch, config, layers, worse, better, mixed),
    }
