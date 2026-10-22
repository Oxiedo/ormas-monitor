"""The checkpoint health report: renders `ormas_monitor.checkpoint.run(...)` (brief §4.5).

Unknown finding kinds, severities and extra keys are rendered generically, so the checkpoint
owner can add findings without touching this file.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from ._html import LABEL, fmt, page, plural, table, tag, verdict

LAYER_COLS = (("certified", "Proven dead*"), ("flat", "Constant"), ("decayed", "Decayed to ~0"),
              ("dormant", "Dormant†"))
KNOWN_LAYER_KEYS = {"layer", "batchnorm", "channels", "collapsed_var", "feeds_relu", "wiring"} | {k for k, _ in LAYER_COLS}
KNOWN_FINDING_KEYS = {"severity", "kind", "title", "detail", "layer", "batchnorm", "channels"}
SOURCE_KEYS = ("format", "files", "wrapper", "streamed", "dtypes", "model_type", "norm_layers",
               "batchnorm_layers", "severity_counts")
SEVERITIES = ("critical", "warning", "info")


def _n(v: Any) -> Any:
    """A count from a list (length) or a number; None stays None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        return len(v)
    return v


def _count_cell(v: Any) -> str:
    n = _n(v)
    return "–" if not n else fmt(n)


def _value(v: Any, limit: int = 50) -> str:
    """Any JSON-like value as compact HTML: lists of dicts become a small table."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "–"
    if isinstance(v, list) and v and all(isinstance(r, dict) for r in v):
        cols = list(dict.fromkeys(c for r in v for c in r))
        rows = [[_value(r.get(c)) for c in cols] for r in v[:limit]]
        tail = f"<p class='sub small'>{len(v) - limit:,} more not shown.</p>" if len(v) > limit else ""
        return table([escape(str(c)) for c in cols], rows) + tail
    if isinstance(v, list):
        shown = ", ".join(str(i) for i in v[:limit])
        return escape(shown + (f" … (+{len(v) - limit:,})" if len(v) > limit else "")) if v else "–"
    if isinstance(v, dict):
        return escape(", ".join(f"{k}: {'yes' if i is True else 'no' if i is False else i}" for k, i in v.items())) or "–"
    if isinstance(v, (int, float)):
        return fmt(v)
    return escape(str(v))


def _source_section(result: dict) -> str:
    """How the file was read and how the wiring (which BatchNorms feed a ReLU) was determined."""
    rows = [[escape(k.replace("_", " ")), _value(result[k] if k != "files" else
                                                   [Path(str(p)).name for p in result[k]])]
            for k in SOURCE_KEYS if k in result and result[k] is not None]
    w = result.get("wiring")
    out = ""
    if isinstance(w, dict):
        how = {"traced": "traced by running the named architecture once on random input",
               "heuristic": "guessed from tensor names and file order (no architecture given)"}
        src = str(w.get("source", "unknown"))
        line = how.get(src, src) if not w.get("traced") else how["traced"]
        out += (f"<p><b>How the wiring was determined:</b> {escape(line)}."
                + (f" {escape(str(w['note']))}" if w.get("note") else "") + "</p>")
        extra = {k: v for k, v in w.items() if k not in ("source", "traced", "note")}
        if extra:
            out += f"<p class='sub small'>{_value(extra)}</p>"
    notes = result.get("notes") or []
    if notes:
        out += "<ul>" + "".join(f"<li>{escape(str(n))}</li>" for n in notes) + "</ul>"
    if rows:
        out += "<details><summary>How the file was read</summary>" + table(["Item", "Value"], rows) + "</details>"
    return f"<div class='box'>{out}</div>" if out else ""


def _finding(x: dict) -> str:
    sev = str(x.get("severity", "info"))
    title = x.get("title") or str(x.get("kind", "finding")).replace("_", " ")
    extra = ""
    where = [f"{k}: {x[k]}" for k in ("layer", "batchnorm") if x.get(k)]
    if where:
        extra = f"<p class='mono small'>{escape(' · '.join(where))}</p>"
    more = {k: v for k, v in x.items() if k not in KNOWN_FINDING_KEYS and v not in (None, [], {})}
    if more:
        extra += ("<details><summary>details</summary>"
                  + "".join(f"<p><b>{escape(str(k))}</b>: {_value(v)}</p>" for k, v in more.items())
                  + "</details>")
    return (f"<div class='finding f-{escape(sev)}'>{tag(sev if sev in LABEL else 'info', LABEL.get(sev, sev.title()))} "
            f"<b>{escape(str(title))}</b><p>{escape(str(x.get('detail', '')))}</p>{extra}</div>")


def _diff_section(result: dict) -> str:
    d = result.get("diff")
    if not d:
        return ""
    changed_rows = [r for r in d.get("changed", []) if r.get("change")]
    unchanged = len(d.get("unchanged", []))
    newly = sum(int(n.get("count") or 0) for n in d.get("newly_dead", []))
    tiles = [(len(changed_rows), "tensors changed"), (unchanged, "tensors exactly unchanged"),
             (f"{len(d.get('added', []))} / {len(d.get('removed', []))}", "added / removed"),
             (len(d.get("reshaped", [])), "reshaped"), (newly, "channels newly dead")]
    top = [[f"<span class='mono'>{escape(str(r.get('tensor')))}</span>", fmt(r.get("change")),
            fmt(r.get("abs_change"))] for r in changed_rows[:25]]
    newly_rows = [[f"<span class='mono'>{escape(str(n.get('layer')))}</span>",
                   f"<span class='mono'>{escape(str(n.get('batchnorm')))}</span>", fmt(n.get("count"))]
                  for n in d.get("newly_dead", [])]
    against = Path(str(result.get("against", "the earlier checkpoint"))).name
    out = (f"<h2>What changed since {escape(against)}</h2><div class='tiles'>"
           + "".join(f"<div class='tile'><div class='n'>{fmt(n) if isinstance(n, int) else escape(str(n))}</div>"
                     f"<div class='l'>{escape(l)}</div></div>" for n, l in tiles) + "</div>")
    if newly_rows:
        out += "<h3>Channels that died between the two files</h3>" + table(
            ["Layer", "BatchNorm", "Channels"], newly_rows, num=(2,))
    if top:
        more = "" if len(changed_rows) <= 25 else f"<p class='sub'>{len(changed_rows) - 25:,} more changed tensors not shown.</p>"
        out += ("<h3>Largest relative changes (‖new − old‖ / ‖old‖)</h3>"
                + table(["Tensor", "Relative change", "Absolute change"], top, num=(1, 2)) + more)
    else:
        out += "<p>No tensor changed between the two files.</p>"
    labels = {"added": "added tensors", "removed": "removed tensors", "reshaped": "reshaped tensors",
              "dtype_changed": "tensors whose number type changed",
              "embedding_rows_unchanged": "embedding tables with rows left exactly unchanged"}
    known = {"changed", "unchanged", "newly_dead"}
    for k, v in d.items():
        if k in known or not v:
            continue
        n = len(v) if isinstance(v, (list, dict)) else 1
        out += (f"<details><summary>{n:,} {escape(labels.get(k, k.replace('_', ' ')))}</summary>"
                f"<div class='small'>{_value(v, 200)}</div></details>")
    return out


def render(result: dict) -> str:
    """The checkpoint report as one HTML document."""
    f = list(result.get("findings", []))
    crit = [x for x in f if x.get("severity") == "critical"]
    warn = [x for x in f if x.get("severity") == "warning"]
    if crit:
        cls, status, head = "bad", "Critical", str(crit[0].get("title", "critical finding"))
    elif warn:
        cls, status, head = "warn", "Problems found", f"{plural(len(warn), 'problem')} found in the weights"
    else:
        cls, status, head = "ok", "No problems found", "No problems found in the weights"
    info = [x for x in f if x.get("severity") not in ("critical", "warning")]
    more = f"<p>{plural(len(info), 'further observation')} below, for information.</p>" if info and (crit or warn) \
        else ""
    order = {s: i for i, s in enumerate(SEVERITIES)}
    findings = "".join(_finding(x) for x in sorted(f, key=lambda x: order.get(x.get("severity"), 3)))
    layers = list(result.get("layers", []))
    extra_keys = sorted({k for l in layers for k in l if k not in KNOWN_LAYER_KEYS
                         and isinstance(l.get(k), (list, tuple, int, float)) and not isinstance(l.get(k), bool)})
    wired = any("feeds_relu" in l for l in layers)
    head_cols = ["Layer", "BatchNorm", "Channels"] + [w for _, w in LAYER_COLS] + ["Var ≈ 0"] + \
        [k.replace("_", " ") for k in extra_keys] + (["ReLU follows", "Wiring"] if wired else [])
    rows = [[f"<span class='mono'>{escape(str(l.get('layer', '–')))}</span>",
             f"<span class='mono'>{escape(str(l.get('batchnorm', '–')))}</span>", fmt(l.get("channels"))]
            + [_count_cell(l.get(k)) for k, _ in LAYER_COLS] + [_count_cell(l.get("collapsed_var"))]
            + [_count_cell(l.get(k)) for k in extra_keys]
            + ([_value(l.get("feeds_relu")), escape(str(l.get("wiring") or "–"))] if wired else [])
            for l in layers]
    params = result.get("parameters")
    sub = " · ".join(x for x in (
        f"{params:,} parameters" if isinstance(params, int) else None,
        f"{result['tensors']:,} tensors" if isinstance(result.get("tensors"), int) else None,
        "weights only, no data") if x)
    body = ("<p class='eyebrow'>Checkpoint health report</p>"
            f"<h1>{escape(Path(str(result.get('file', 'checkpoint'))).name)}</h1><p class='sub'>{sub}</p>"
            + verdict(cls, status, head, more)
            + "<p class='sub small'>This check reads the saved weights only. It does not run the model, "
              "look at data or measure accuracy. Proofs of dead channels hold only where a ReLU directly "
              "follows the BatchNorm.</p>"
            + _source_section(result)
            + (f"<h2>Findings</h2>{findings}" if f else "")
            + (f"<h2>BatchNorm layers</h2>{table(head_cols, rows, num=tuple(range(2, 3 + len(LAYER_COLS) + 1 + len(extra_keys))))}"
               "<p class='sub small'>* Proven dead when a ReLU directly follows the BatchNorm. "
               "† Estimated, not proven: shift ≤ −3×|scale|.</p>" if layers else "")
            + _diff_section(result))
    return page(f"Checkpoint health report: {Path(str(result.get('file', 'checkpoint'))).name}", body)


def write(result: dict, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(render(result), encoding="utf-8")
    return out
