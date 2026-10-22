"""The run-comparison report: renders `ormas_monitor.record.diff.diff_runs(a, b)` (brief §4.6).

Keys beyond the documented ones are listed generically at the end, so the record owner can
extend the diff without touching this file.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from ._html import fmt, layer_html, page, table, tag, verdict

KNOWN = {"a", "b", "same_architecture", "config_changes", "environment_changes", "events", "layers",
         "worse", "better", "mixed", "more_silent", "fewer_silent", "verdict", "verdict_detail"}
# (key, column title); counts are None for a layer that was not scanned, which renders as "–", not 0.
# Diff results that split stuck counts into proven (`proven_*`) and silent (`silent_*`) get the
# split columns; older results, whose `peak_stuck` mixes both, get the original ones.
SPLIT_PAIRS = (("alarms", "Alarms"), ("proven_peak", "Stuck (peak)"), ("proven_at_end", "Stuck at end"),
               ("silent_at_end", "Silent at end"), ("certified", "Proven dead"),
               ("channels_repaired", "Repaired"))
LAYER_PAIRS = (("alarms", "Alarms"), ("peak_stuck", "Peak stuck"), ("stuck_at_end", "Stuck at end"),
               ("certified", "Proven dead"), ("repairs", "Repairs"), ("channels_repaired", "Channels repaired"))
# The diff's own per-layer judgement (`health`, `silent_change`) -> the word in the Change column.
HEALTH_WORD = {"worse": "worse in B", "better": "better in B", "mixed": "both ways"}
SILENT_WORD = {"more": "more silent in B", "fewer": "fewer silent in B"}


def _cell(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return f"<span class='mono'>{escape(json.dumps(v, default=str))}</span>"
    return fmt(v)


def _pair(a: Any, b: Any) -> str:
    """'3 → 0', bold when the two differ, so a change is visible without colour."""
    if isinstance(a, list) or isinstance(b, list):
        a = ", ".join(map(str, a)) if a else None
        b = ", ".join(map(str, b)) if b else None
    if a == b:
        return fmt(a)
    return f"<b>{fmt(a)} → {fmt(b)}</b>"


def _yes(v: Any, yes: str, no: str) -> str:
    if v is None:
        return tag("info", "unknown")
    return tag("healthy", yes) if v else tag("info", no)


def _run_card(label: str, r: dict) -> str:
    intact = r.get("intact")
    failed = r.get("failed_checks") or []
    rows = [("Name", escape(str(r.get("name") or "–"))), ("Model", escape(str(r.get("model") or "–"))),
            ("Parameters", fmt(r.get("parameters"))), ("Steps", fmt(r.get("steps"))),
            ("Final loss", fmt(r.get("final_loss"))),
            ("Record", (tag("healthy", "checks pass") if intact else
                        tag("critical", "FAILED: " + ", ".join(map(str, failed)) if failed else "FAILED")
                        if intact is False else tag("info", "not checked"))),
            ("Finished", _yes(r.get("complete"), "yes", "no run_end")),
            ("Sealed / signed", _yes(r.get("sealed"), "sealed", "not sealed") + " "
             + _yes(r.get("signed"), "signed", "not signed")),
            ("Path", f"<span class='mono'>{escape(str(r.get('path', '–')))}</span>")]
    return (f"<div class='tile' style='padding:10px 14px'><div class='l'><b>Run {label}</b></div>"
            + "".join(f"<div class='small'>{k}: {v}</div>" for k, v in rows) + "</div>")


def _detail(v: Any) -> str:
    """`verdict_detail`: a string, or a list of sentences rendered as a list."""
    if not v:
        return ""
    if isinstance(v, (list, tuple)):
        return "<ul>" + "".join(f"<li>{escape(str(x))}</li>" for x in v) + "</ul>"
    return f"<p>{escape(str(v))}</p>"


def render(d: dict) -> str:
    """The run comparison as one HTML document."""
    a, b = d.get("a") or {}, d.get("b") or {}
    broken = [x for x, r in (("A", a), ("B", b)) if r.get("intact") is False]
    same = d.get("same_architecture")
    worse, better, mixed = (list(d.get(k) or []) for k in ("worse", "better", "mixed"))
    # The headline is the diff's own verdict: this page never computes a second, possibly
    # contradictory one. Only a failed record check changes the box's colour.
    cls, status = ("bad", "Record failed verification") if broken else ("neutral", "Comparison")
    head = str(d.get("verdict") or "Two runs compared.")
    more = _detail(d.get("verdict_detail"))
    if broken:
        more += f"<p>The record of run {' and '.join(broken)} fails verification; its numbers are unconfirmed.</p>"
    if same is False:
        more += "<p>The two runs do not share an architecture, so per-layer rows may not line up.</p>"

    def changes(key: str) -> list[list[str]]:
        return [[f"<span class='mono'>{escape(str(c.get('key')))}</span>", _cell(c.get("a")), _cell(c.get("b"))]
                for c in d.get(key) or []]
    cfg, env = changes("config_changes"), changes("environment_changes")
    ev = [[escape(str(k).replace("_", " ")), _pair((v or {}).get("a"), (v or {}).get("b"))]
          for k, v in (d.get("events") or {}).items()]
    direction = {**{n: "worse in B" for n in worse}, **{n: "better in B" for n in better},
                 **{n: "both ways" for n in mixed}}
    sev = {"worse in B": "warning", "better in B": "healthy", "both ways": "warning"}
    layers = list(d.get("layers") or [])
    split = any("proven_at_end_a" in l or "proven_at_end_b" in l for l in layers)
    pairs = list(SPLIT_PAIRS if split else LAYER_PAIRS)
    lrows: list[tuple[bool, list[str]]] = []
    for l in layers:
        name = str(l.get("layer"))
        if not (l.get("in_a") and l.get("in_b")):
            word = "only in A" if l.get("in_a") else "only in B"
        elif l.get("health") in HEALTH_WORD:
            word = HEALTH_WORD[l["health"]]
        elif l.get("silent_change") in SILENT_WORD:
            word = SILENT_WORD[l["silent_change"]]
        else:
            word = direction.get(name, "changed" if l.get("changed") else "same")
        diag = _pair(l.get("diagnoses_a"), l.get("diagnoses_b"))
        changed = bool(l.get("changed")) if "changed" in l else word != "same"
        lrows.append((changed, [layer_html(name, l.get("kind")), tag(sev.get(word, "info"), word)]
                      + [_pair(l.get(f"{k}_a"), l.get(f"{k}_b")) for k, _ in pairs] + [diag]))
    n_changed = sum(1 for c, _ in lrows if c)
    rows = [r for c, r in lrows if c] + [r for c, r in lrows if not c]
    extra = {k: v for k, v in d.items() if k not in KNOWN}

    body = ("<p class='eyebrow'>Run comparison · A (approved) vs B (proposed)</p>"
            f"<h1>{escape(str(a.get('name') or 'Run A'))} vs {escape(str(b.get('name') or 'Run B'))}</h1>"
            + verdict(cls, status, head, more)
            + f"<div class='tiles'>{_run_card('A', a)}{_run_card('B', b)}</div>"
            + "<h2>Settings that differ</h2>"
            + (table(["Setting", "Run A", "Run B"], cfg) if cfg else "<p>None: optimizer and monitor settings match.</p>")
            + ("<h3>Environment</h3>" + table(["Item", "Run A", "Run B"], env) if env else "")
            + "<h2>Layers</h2>"
            + ((f"<p>{n_changed:,} of {len(rows):,} layers differ between the runs"
                + (" (listed first). Cells read A → B; – means not scanned or not recorded, which is not 0.</p>"
                   if n_changed else ".</p>")
                + ("<p class='sub small'>Stuck: proven dead, constant or decayed to ~0, read from the weights. "
                   "Silent: no positive output for several checks; likely unused, not proven dead, and never "
                   "counted as stuck.</p>" if split else "")
                + table(["Layer", "Change"] + [t for _, t in pairs] + ["Alarm kinds"], rows,
                        num=tuple(range(2, 2 + len(pairs)))))
               if rows else "<p>No per-layer data.</p>")
            + "<h2>Events</h2>"
            + (table(["Event", "Run A → Run B"], ev, num=(1,)) if ev else "<p>No events to compare.</p>")
            + ("<h2>More</h2>" + table(["Key", "Value"], [[escape(k), _cell(v)] for k, v in extra.items()])
               if extra else "")
            + "<p class='sub small'>A comparison of two records, not of two models' quality: it shows what "
              "each monitor saw and did, not which model is better.</p>")
    return page(f"Run comparison: {a.get('name') or 'A'} vs {b.get('name') or 'B'}", body, source="diff")


def write(diff_result: dict, path: str | Path) -> Path:
    """Write the run comparison to `path` and return it."""
    out = Path(path)
    out.write_text(render(diff_result), encoding="utf-8")
    return out
