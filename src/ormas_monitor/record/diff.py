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
    layers              [{layer, in_a, in_b, changed, health, silent_change, and for each of
                          alarms, open_alarms, peak_stuck, stuck_at_end, proven_peak, proven_at_end,
                          silent_peak, silent_at_end, repairs, channels_repaired, certified, diagnoses
                          the pair <key>_a, <key>_b}]
                        peak_stuck / stuck_at_end count every kind of stuck channel; proven_* count
                        certified dead + constant + decayed (read from the weights), silent_* count
                        channels with no positive output on the batches seen (unproven);
                        open_alarms = alarms not cleared by the end of the run
                        health: better | worse | mixed | same (None unless the layer is in both runs)
                        silent_change: more | fewer | None
    worse, better, mixed  layers whose health in B moved one way, the other, or both
    more_silent, fewer_silent  layers whose health is the same but whose silent count moved
    verdict             a short headline: integrity first, then the net per-layer result
    verdict_detail      the supporting sentences: architecture, settings, how health is judged, repairs

Health is weighed by strength of evidence. Proven-stuck channels and open alarms at the end of each
run decide; when they are equal, their peaks and the alarm counts do. Silent channels prove nothing
about the weights (a channel can be silent on the batches seen and still recover), so a change in
them is a secondary note and never makes a layer less healthy.

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
from .layout import LAYOUT, RecordLayout, file_problem, locate
from .verify import verify_run

ENVIRONMENT_KEYS = ("tool", "python", "torch", "platform", "model", "parameters")
# per-layer statistics; each becomes the pair <key>_a, <key>_b in a layer row
LAYER_STATS = ("alarms", "open_alarms", "peak_stuck", "stuck_at_end", "proven_peak", "proven_at_end",
               "silent_peak", "silent_at_end", "repairs", "channels_repaired", "certified", "diagnoses")


def _stuck_parts(v: Any) -> list[int] | None:
    """A metrics row's stuck counts for one layer, [certified, constant, silent, decayed]; None if malformed."""
    try:
        vals = [int(x) for x in (v.values() if isinstance(v, dict) else v)]
    except (TypeError, ValueError, OverflowError):
        return None
    return vals


def _split(vals: list[int]) -> tuple[int, int]:
    """(proven, silent): certified + constant + decayed are read from the weights; silent is observed."""
    part = (vals + [0, 0, 0, 0])[:4]
    return part[0] + part[1] + part[3], part[2]


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _peak(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _load(path: str | Path, layout: RecordLayout) -> dict:
    loc = locate(path, layout)
    result = verify_run(path, layout=layout)
    # Files verification refused to follow (links, FIFOs) are not read here either.
    diary_ok = loc.run_dir is None or file_problem(loc.diary) is None
    events = parse_lines(loc.diary)[0] if diary_ok else []
    metrics: list[dict] = []
    mpath = loc.file(layout.metrics)
    has_metrics = mpath is not None and file_problem(mpath) is None
    if has_metrics:
        with open(mpath, encoding="utf-8", errors="replace") as fh:
            for text in fh:
                if text.strip():
                    try:
                        row = json.loads(text)
                    except (ValueError, RecursionError):
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
        "metrics": has_metrics,
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
    from_metrics: set[str] = set()

    def entry(layer: str) -> dict:
        return stats.setdefault(layer, {"alarms": 0, "open": set(), "peak_stuck": None, "stuck_at_end": None,
                                        "proven_peak": None, "proven_at_end": None, "silent_peak": None,
                                        "silent_at_end": None, "repairs": 0, "channels_repaired": 0,
                                        "certified": None, "diagnoses": []})

    for row in run["metrics"]:
        grad, stuck = row.get("grad"), row.get("stuck")
        for layer in (grad if isinstance(grad, dict) else {}):
            entry(layer)
        for layer, parts in (stuck.items() if isinstance(stuck, dict) else ()):
            vals = _stuck_parts(parts)
            if vals is None:
                continue
            s = entry(layer)
            proven, silent = _split(vals)
            from_metrics.add(layer)
            s["peak_stuck"] = _peak(s["peak_stuck"], sum(vals))
            s["stuck_at_end"] = sum(vals)
            s["proven_peak"] = _peak(s["proven_peak"], proven)
            s["proven_at_end"] = proven
            s["silent_peak"] = _peak(s["silent_peak"], silent)
            s["silent_at_end"] = silent
            s["certified"] = _peak(s["certified"], vals[0] if vals else 0)
    for e in run["events"]:
        layer = e.get("layer")
        if not isinstance(layer, str):
            continue
        s = entry(layer)
        kind = e.get("event")
        if kind == "alarm":
            s["alarms"] += 1
            s["open"].add(str(e.get("diagnosis")))
            if e.get("diagnosis") not in s["diagnoses"]:
                s["diagnoses"].append(e.get("diagnosis"))
        elif kind == "alarm_cleared":
            s["open"].discard(str(e.get("diagnosis")))
        elif kind in ("channels", "channels_cleared"):
            proven = sum(_int(e.get(k)) for k in ("n_certified", "n_flat", "n_decayed"))
            silent = _int(e.get("n_observed"))
            s["peak_stuck"] = _peak(s["peak_stuck"], proven + silent)
            s["proven_peak"] = _peak(s["proven_peak"], proven)
            s["silent_peak"] = _peak(s["silent_peak"], silent)
            s["certified"] = _peak(s["certified"], _int(e.get("n_certified")))
            if layer not in from_metrics:        # a diary without metrics: the last level change is the end
                s["proven_at_end"], s["silent_at_end"] = proven, silent
        elif kind == "certificate":
            s["certified"] = _peak(s["certified"], _int(e.get("count")))
        elif kind == "repair":
            s["repairs"] += 1
            s["channels_repaired"] += _int(e.get("count"))
    for s in stats.values():
        s["open_alarms"] = len(s.pop("open"))
    return stats


def _gt(x: int | None, y: int | None) -> bool:
    return (x or 0) > (y or 0)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# Strong evidence, read from the weights and the alarms. The state each run ended in decides; when
# that is the same, the run's history (peaks and alarm counts) does.
END_KEYS = ("proven_at_end", "open_alarms")
HISTORY_KEYS = ("proven_peak", "certified", "alarms")
# Weak evidence: silent channels gave no positive output on the batches seen, which proves nothing.
SILENT_KEYS = ("silent_at_end", "silent_peak")
LABELS = {"proven_at_end": "stuck (from weights)", "open_alarms": "open alarms", "proven_peak": "peak stuck (from weights)",
          "certified": "certified dead", "alarms": "alarms", "silent_at_end": "silent",
          "silent_peak": "peak silent"}


def _moved(row: dict, keys: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(keys that got worse in B, keys that got better in B)."""
    return ([k for k in keys if _gt(row[f"{k}_b"], row[f"{k}_a"])],
            [k for k in keys if _gt(row[f"{k}_a"], row[f"{k}_b"])])


def _health(row: dict) -> tuple[str, str | None, tuple[str, ...]]:
    """(health in B: better | worse | mixed | same, silent change: more | fewer | None, the keys that decided)."""
    worse, better = _moved(row, END_KEYS)
    if not worse and not better:
        worse, better = _moved(row, HISTORY_KEYS)
    health = "mixed" if worse and better else "worse" if worse else "better" if better else "same"
    more, fewer = _moved(row, SILENT_KEYS[:1])
    if not more and not fewer:
        more, fewer = _moved(row, SILENT_KEYS[1:])
    silent = "more" if more and not fewer else "fewer" if fewer and not more else None
    return health, silent, tuple(worse + better)


def _name(layer: str) -> str:
    return f"layer {layer}" if layer.replace(".", "").isdigit() else layer


def _num(v: Any) -> Any:
    return 0 if v is None else v


def _evidence(row: dict) -> str:
    """What moved in one layer, A -> B, strong evidence first."""
    keys = row["_keys"]
    if row["health"] == "worse" and "certified" not in keys and _gt(row["certified_b"], row["certified_a"]):
        keys = keys + ("certified",)
    parts = [f"{LABELS[k]} {_num(row[f'{k}_a'])} to {_num(row[f'{k}_b'])}" for k in keys]
    ra, rb = row["channels_repaired_a"] or 0, row["channels_repaired_b"] or 0
    if rb != ra:
        parts.append(f"{rb} repaired" if ra == 0 else f"repaired {ra} to {rb}")
    return ", ".join(parts)


def _silent_evidence(row: dict) -> str:
    key = "silent_at_end" if row["silent_at_end_a"] != row["silent_at_end_b"] else "silent_peak"
    return f"{LABELS[key]} {_num(row[f'{key}_a'])} to {_num(row[f'{key}_b'])}, unproven"


def _group(names: list[str], by_name: dict[str, dict], limit: int, silent: bool = False) -> str:
    shown = "; ".join(f"{_name(n)}: {(_silent_evidence if silent else _evidence)(by_name[n])}"
                      for n in names[:limit])
    return shown + (f"; +{len(names) - limit} more" if len(names) > limit else "")


def _layer_sentence(by_name: dict[str, dict], worse: list[str], better: list[str], mixed: list[str],
                    more_silent: list[str], fewer_silent: list[str], limit: int) -> str:
    """The net per-layer result, the larger group first (worse first on a tie)."""
    strong = [("healthier", better), ("less healthy", worse)]
    strong.sort(key=lambda g: -len(g[1]))
    if len(better) == len(worse):
        strong.reverse()
    groups = [(word, names, False) for word, names in strong if names]
    if mixed:
        groups.append(("changed both ways", mixed, False))
    if more_silent:
        groups.append(("has more silent channels", more_silent, True))
    if fewer_silent:
        groups.append(("has fewer silent channels", fewer_silent, True))
    if not groups:
        return "No layer's alarms, stuck channels or certified dead channels changed."
    clauses = []
    for i, (word, names, silent) in enumerate(groups):
        count = _plural(len(names), "layer") if i == 0 else str(len(names))
        verb = ("is " if i == 0 and word in ("healthier", "less healthy") else "") + word
        clauses.append(f"{verb} in {count} ({_group(names, by_name, limit, silent)})")
    lead = "Run B " if worse or better or mixed else "No layer changed on proven evidence; run B "
    return lead + ", ".join(clauses[:-1]) + (" and " if len(clauses) > 1 else "") + clauses[-1] + "."


def _verdict(a: dict, b: dict, same_arch: bool, config: list[dict], layers: list[dict], worse: list[str],
             better: list[str], mixed: list[str], more_silent: list[str],
             fewer_silent: list[str]) -> tuple[str, str]:
    """(verdict, verdict_detail): a short headline led by integrity and the net per-layer result,
    and the supporting sentences."""
    integrity = [f"Run {x}'s record failed verification ({', '.join(info['failed_checks'])}), so this "
                 "comparison cannot be trusted." for x, info in (("A", a), ("B", b)) if not info["intact"]]
    by_name = {row["layer"]: row for row in layers}
    head = integrity + [_layer_sentence(by_name, worse, better, mixed, more_silent, fewer_silent, 3)]
    if a["architecture_sha256"] is not None and b["architecture_sha256"] is not None and not same_arch:
        head.append("The architectures differ, so layers may not correspond.")

    detail = []
    for x, info in (("A", a), ("B", b)):
        if info["intact"] and not info["complete"]:
            detail.append(f"Run {x} has no run_end: it did not finish or its record is truncated.")
    if a["architecture_sha256"] is None or b["architecture_sha256"] is None:
        detail.append("The architecture of one run is not recorded.")
    elif same_arch:
        detail.append("Same architecture.")
    else:
        detail.append("Different architecture: the parameter shapes differ.")
    if config:
        shown = ", ".join(f"{c['key']} {c['a']!r} -> {c['b']!r}" for c in config[:3])
        more = f" and {len(config) - 3} more" if len(config) > 3 else ""
        detail.append(f"{_plural(len(config), 'setting')} changed: {shown}{more}.")
    else:
        detail.append("No optimizer or monitor settings changed.")
    if worse or better or mixed or more_silent or fewer_silent:
        if max(len(worse), len(better), len(mixed), len(more_silent), len(fewer_silent)) > 3:
            detail.append(_layer_sentence(by_name, worse, better, mixed, more_silent, fewer_silent, 10))
        detail.append("A layer's health is judged by stuck channels (certified dead, constant or decayed "
                      "to ~0, read from the weights) and open alarms at the end of each run, then by their peaks "
                      "and alarm counts. Silent channels only produced no positive output on the batches seen; "
                      "they are reported separately and never make a layer less healthy on their own.")
    repairs_a = sum(r["repairs_a"] or 0 for r in layers)
    repairs_b = sum(r["repairs_b"] or 0 for r in layers)
    fixed_a = sum(r["channels_repaired_a"] or 0 for r in layers)
    fixed_b = sum(r["channels_repaired_b"] or 0 for r in layers)
    if repairs_a or repairs_b:
        detail.append(f"Repairs: {repairs_a} ({_plural(fixed_a, 'channel')}) in run A, "
                      f"{repairs_b} ({_plural(fixed_b, 'channel')}) in run B.")
    return " ".join(head), " ".join(detail)


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

    ca = Counter(str(e.get("event")) for e in ra["events"])
    cb = Counter(str(e.get("event")) for e in rb["events"])
    kinds = list(ca) + [k for k in cb if k not in ca]
    events = {k: {"a": ca.get(k, 0), "b": cb.get(k, 0)} for k in kinds}

    sa, sb = _layer_stats(ra), _layer_stats(rb)
    layers: list[dict] = []
    groups: dict[str, list[str]] = {"worse": [], "better": [], "mixed": [], "more": [], "fewer": []}
    for name in list(sa) + [n for n in sb if n not in sa]:
        x, y = sa.get(name), sb.get(name)
        row: dict[str, Any] = {"layer": name, "in_a": x is not None, "in_b": y is not None}
        for key in LAYER_STATS:
            row[f"{key}_a"] = x[key] if x else None
            row[f"{key}_b"] = y[key] if y else None
        health, silent, row["_keys"] = _health(row)
        both = row["in_a"] and row["in_b"]
        row["health"] = health if both else None
        row["silent_change"] = silent if both else None
        row["changed"] = (not both or health != "same" or silent is not None
                          or row["repairs_a"] != row["repairs_b"]
                          or any(row[f"{k}_a"] != row[f"{k}_b"] for k in ("peak_stuck", "stuck_at_end")))
        if both and health != "same":
            groups[health].append(name)
        elif both and silent is not None:
            groups[silent].append(name)
        layers.append(row)

    verdict, detail = _verdict(a, b, same_arch, config, layers, groups["worse"], groups["better"],
                               groups["mixed"], groups["more"], groups["fewer"])
    for row in layers:
        del row["_keys"]
    return {
        "a": a, "b": b,
        "same_architecture": same_arch,
        "config_changes": config,
        "environment_changes": env,
        "events": events,
        "layers": layers,
        "worse": groups["worse"],
        "better": groups["better"],
        "mixed": groups["mixed"],
        "more_silent": groups["more"],
        "fewer_silent": groups["fewer"],
        "verdict": verdict,
        "verdict_detail": detail,
    }

