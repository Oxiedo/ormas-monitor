"""The engineer report (`report.html`) for a monitored run: verdict, stuck time, loss with
events, lead time before loss spikes, every event, per-layer table, setup and record check."""
from __future__ import annotations

import json
import statistics
import time
from html import escape
from pathlib import Path
from typing import Optional

from ._html import fmt, legend, loss_chart, page, plural, share, duration, table, tag, timeline, verdict
from .facts import COMMON, DISPLAY_SEVERITY, RunFacts, Stuck

GROUPABLE = ("alarm", "alarm_cleared", "channels_cleared")


def describe(e: dict) -> str:
    """One plain sentence for one diary event. Unknown event types are described generically."""
    ev = str(e.get("event", "?"))
    if ev == "alarm":
        word = {"no_gradient": "gradient is exactly zero", "vanishing": "gradient collapsed",
                "exploding": "gradient exploded"}.get(e.get("diagnosis"), str(e.get("diagnosis")))
        if e.get("grad_norm") is None or e.get("baseline") is None:
            return word
        return f"{word}: {fmt(e['grad_norm'])} against its own normal of {fmt(e['baseline'])}"
    if ev == "alarm_cleared":
        return "gradient back in its normal range"
    if ev == "channels":
        st = Stuck.from_event(e)
        parts = [f"{n:,} {w}" for n, w in ((st.certified, "proven dead"), (st.constant, "constant"),
                                            (st.decayed, "decayed to ~0"),
                                            (st.silent, "silent for several checks")) if n]
        of = f"{st.total:,} of {fmt(e.get('channels'))}" if e.get("channels") is not None else f"{st.total:,}"
        return f"{of} channels stuck: " + (", ".join(parts) or str(e.get("level", "")))
    if ev == "channels_cleared":
        return "all channels active again"
    if ev == "certificate":
        return (f"{fmt(e.get('count'))} channels certified dead (proven): zero filter or BatchNorm scale, shift ≤ 0, "
                f"so the ReLU is closed for every input and no gradient can ever reach them")
    if ev == "repair":
        out = f"re-initialised {fmt(e.get('count'))} stuck channels and reset their optimizer state"
        rel, delta = e.get("relative_change"), e.get("delta_norm")
        if rel is not None:
            out += f"; layer weights changed by {100 * float(rel):.3g}% (‖ΔW‖/‖W‖)"
        elif delta is not None:
            out += f"; ‖ΔW‖ = {fmt(delta)} (relative change undefined: the weights were all zero)"
        if "bound" in e:
            out += ("; no bound on the size of the change was enforced" if e["bound"] is None
                    else f"; bound enforced: {e['bound']}")
        return out
    if ev == "loss_spike":
        return f"loss spiked to {e['value']:.4g} (its normal: {e['baseline']:.4g})"
    if ev == "loss_recovered":
        return f"loss back in its normal range ({e['value']:.4g})"
    if ev == "nonfinite_loss":
        return f"loss became {e.get('value')}"
    if ev == "nonfinite_gradient":
        return f"non-finite gradients in {plural(len(e.get('layers') or []), 'layer')}"
    if ev == "sink_error":
        return (f"an output sink failed and was switched off for the rest of the run "
                f"(training continued): {e.get('sink', '?')}: {e.get('error', '?')}")
    if ev == "skipped_step":
        scale = f" (loss scale {fmt(e['scale'])})" if e.get("scale") is not None else ""
        return "optimizer step skipped by the mixed-precision scaler: non-finite gradients" + scale
    return generic(e)


def generic(e: dict) -> str:
    """'<event name>: key=value, …' for event types this version of the report does not know."""
    skip = set(COMMON) | {"step", "severity", "layer"}
    bits = []
    for k, v in e.items():
        if k in skip:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, default=str)
            v = v if len(v) <= 80 else v[:77] + "…"
        bits.append(f"{k}={v}")
    return str(e.get("event", "event")).replace("_", " ") + (": " + ", ".join(bits) if bits else "")


def severity(e: dict) -> str:
    return str(e.get("severity") or DISPLAY_SEVERITY.get(str(e.get("event")), "info"))


def _groups(events: list[dict]) -> list[list[dict]]:
    """Fold runs of same-kind events at one check (8 alarms at step 450) into one row, and every
    `skipped_step` of the run into a single row at the first one."""
    out: list[list[dict]] = []
    skipped: list[dict] = []
    for e in events:
        if e.get("event") == "skipped_step":
            if not skipped:
                out.append(skipped)
            skipped.append(e)
            continue
        if out and e.get("event") in GROUPABLE:
            p = out[-1][-1]
            same = p.get("event") == e.get("event") and p.get("diagnosis") == e.get("diagnosis")
            if same and p.get("step") == e.get("step"):
                out[-1].append(e)
                continue
        out.append([e])
    return out


def _where(group: list[dict]) -> str:
    layers = [str(e.get("layer")) for e in group if e.get("layer")]
    if not layers:
        return "–"
    if len(layers) <= 3:
        return "<br>".join(f"<span class='mono'>{escape(x)}</span>" for x in layers)
    return (f"<span class='mono'>{escape(layers[0])}</span> and {len(layers) - 1} more"
            f"<details><summary>all {len(layers)}</summary><span class='mono'>{escape(', '.join(layers))}</span>"
            f"</details>")


def _step_cell(group: list[dict]) -> str:
    s = [e.get("step") for e in group if isinstance(e.get("step"), (int, float))]
    if not s:
        return "–"
    lo, hi = int(min(s)), int(max(s))
    return f"{lo:,}" if lo == hi else f"{lo:,}–{hi:,}"


def _what(group: list[dict]) -> str:
    e = group[0]
    if len(group) == 1:
        return escape(describe(e))
    ev = e.get("event")
    if ev == "alarm":
        word = {"no_gradient": "gradient is exactly zero", "vanishing": "gradient collapsed",
                "exploding": "gradient exploded"}.get(e.get("diagnosis"), str(e.get("diagnosis")))
        return f"{len(group)} layers: {escape(word)}"
    if ev == "skipped_step":
        scales = [float(x["scale"]) for x in group if isinstance(x.get("scale"), (int, float))]
        rng = f"; loss scale {fmt(max(scales))} → {fmt(min(scales))}" if len(set(scales)) > 1 else ""
        return (f"{len(group)} optimizer steps skipped by the mixed-precision scaler (non-finite gradients, "
                f"the scaler's designed response){rng}")
    return f"{len(group)} layers: {escape(describe(e))}"


def _headline(f: RunFacts) -> tuple[str, str, str, str]:
    """(css class, status word, headline, detail html) for the engineer verdict box."""
    crit = [e for e in f.shown if e.get("severity") == "critical"]
    problems = [e for e in f.shown if severity(e) in ("critical", "warning", "proof")]
    repairs = f.of("repair")
    certs = f.of("certificate")
    still = [ep for ep in f.episodes if ep.open_at_end]
    detail = ""
    if certs:
        detail += ("<p>Proven dead: " + ", ".join(
            f"<span class='mono'>{escape(str(e.get('layer')))}</span> ({fmt(e.get('count'))} channels, "
            f"step {fmt(e.get('step'))})" for e in certs) + ".</p>")
    first = problems[0] if problems else None
    if first is not None:
        where = first.get("layer") or ("the loss" if "loss" in str(first.get("event")) else "the run")
        detail += (f"<p>First sign of trouble at step {fmt(first.get('step'))} in "
                   f"<span class='mono'>{escape(str(where))}</span>: {escape(describe(first))}.</p>")
    if not f.trust.intact:
        return "bad", "Record broken", f"The record fails its integrity check: {f.trust.message}", detail
    if crit:
        return ("bad", "Critical", f"Critical: {describe(crit[0])} at step {fmt(crit[0].get('step'))}",
                detail)
    if still:
        layers = sorted({ep.layer for ep in still})
        return ("bad", "Stuck at end of run",
                f"{plural(len(layers), 'layer')} still had stuck channels when the run ended"
                + (f"; {plural(len(repairs), 'repair')} applied earlier" if repairs else ""), detail)
    if repairs:
        return ("fixed", "Repaired",
                f"{plural(len(repairs), 'repair')} applied ({fmt(sum(int(e.get('count') or 0) for e in repairs))} "
                f"channels); no layer was stuck at the end of the run", detail)
    if problems:
        layers = sorted({str(e.get("layer")) for e in problems if e.get("layer")})
        head = (f"Warnings in {plural(len(layers), 'layer')}" if layers else "Warnings raised") + \
            ", nothing proven broken"
        return "warn", "Warnings", head, detail
    checks = f.checks
    return ("ok", "Healthy", "No problems found" + (f" in {checks:,} checks" if checks else ""),
            "<p>Every layer's gradient stayed within its own normal range and no channel got stuck.</p>")


def _stuck_section(f: RunFacts) -> str:
    c = f.cost
    if not f.episodes:
        silent = "" if not f.silent_peaks else (
            f" Silent channels were seen in {plural(len(f.silent_peaks), 'layer')}; they are an estimate, "
            f"not a weight-level fact, so they are not counted here.")
        return (f"<h2>Time spent stuck</h2><p>None. At no check did any layer have proven-dead, constant "
                f"or decayed channels.{silent}</p>")
    marks = [(int(e["step"]), str(e["event"]), describe(e)) for e in f.of("certificate", "repair")]
    rows = []
    for ep in f.episodes:
        p = ep.peak
        kinds = ", ".join(f"{n:,} {w}" for n, w in ((p.certified, "proven dead"), (p.constant, "constant"),
                                                     (p.decayed, "decayed")) if n)
        if ep.open_at_end:
            outcome = tag("proof" if ep.proven else "warning", "still stuck at end")
        elif ep.repaired:
            outcome = tag("action", f"repaired at step {ep.end:,}")
        else:
            outcome = tag("healthy", f"cleared by step {ep.end:,}")
        rows.append([f"<span class='mono'>{escape(ep.layer)}</span>", f"{ep.start:,}",
                     "end" if ep.open_at_end else f"{ep.end:,}", f"{ep.steps:,}",
                     escape(duration(ep.seconds)), escape(kinds), outcome])
    total_line = (f"<p><b>{c.stuck_steps:,} of {c.total_steps:,} steps</b> ({share(c.stuck_steps, c.total_steps)}) "
                  f"and <b>{escape(duration(c.stuck_seconds))}</b> of wall-clock"
                  + (f" ({share(c.stuck_seconds, c.total_seconds)} of {escape(duration(c.total_seconds))})"
                     if c.total_seconds else "")
                  + " ran with at least one layer stuck"
                  + (f"; {c.proven_steps:,} of those steps with channels proven dead" if c.proven_steps else "")
                  + ".</p>")
    return ("<h2>Time spent stuck</h2>" + total_line
            + timeline(c.total_steps, list(c.spans), marks, "Run timeline with stuck intervals hatched")
            + legend([m[1] for m in marks], True)
            + table(["Layer", "From step", "Until", "Steps", "Wall-clock", "Peak stuck channels", "Outcome"],
                    rows, num=(1, 2, 3, 4))
            + f"<p class='sub small'>Counted from the first check that saw the problem, so each figure is a "
              f"lower bound by up to one check interval ({c.resolution:,} steps). Wall-clock comes from the "
              f"record's timestamps, interpolated between events. Silent channels are not counted.</p>")


def _lead_section(f: RunFacts) -> str:
    if not f.leads:
        return ""
    window = f.config.lead_window_checks * f.check_every
    rows = []
    for ld in f.leads:
        sp = ld.spike
        spike = f"{sp['value']:.4g} (normal {sp['baseline']:.4g})"
        if ld.signal is None:
            sig, lead = "none recorded in the window", "–"
        else:
            s = ld.signal
            where = f"<span class='mono'>{escape(str(s.get('layer') or '–'))}</span>: "
            sig = where + escape(describe(s))
            if ld.n_signals > 1:
                sig += f" <span class='sub small'>(+{ld.n_signals - 1} more in the window)</span>"
            if ld.steps_earlier == 0:
                lead = "same check: no advance warning"
            else:
                lead = (f"<b>{ld.steps_earlier:,} steps</b> earlier (step {int(s['step']):,})"
                        + (f"<br><span class='sub small'>{escape(duration(ld.seconds_earlier))}</span>"
                           if ld.seconds_earlier is not None else ""))
        rows.append([f"{int(sp['step']):,}", escape(spike), sig, lead])
    return ("<h2>Lead time before loss spikes</h2>"
            f"<p>For each loss spike, the earliest gradient alarm, stuck-channel or proof event in the "
            f"window before it: back to the previous recovery of the loss, at most {window:,} steps. "
            f"An earlier signal is not proof that it caused the spike; it shows when the monitor "
            f"could first have told you something was off. Resolution is one check ({f.check_every:,} steps).</p>"
            + table(["Spike at step", "Loss", "Earliest earlier signal", "Lead"], rows, num=(0,)))


def _layer_rows(f: RunFacts) -> list[list[str]]:
    names: list[str] = []
    for m in f.metrics:
        for k in list((m.get("grad") or {})) + list((m.get("stuck") or {})):
            if k not in names:
                names.append(k)
    for e in f.events:
        if e.get("layer") and e["layer"] not in names:
            names.append(str(e["layer"]))
    alarmed = {e.get("layer") for e in f.of("alarm")}
    repaired = {e.get("layer") for e in f.of("repair")}
    proven = {e.get("layer") for e in f.of("certificate")}
    open_end = {ep.layer for ep in f.episodes if ep.open_at_end}
    rows = []
    for name in names:
        series = [float(m["grad"][name]) for m in f.metrics if name in (m.get("grad") or {})
                  and m["grad"][name] is not None]
        p = f.layer_peaks.get(name)
        if name in open_end:
            status = tag("proof", "proven dead at end") if name in proven else tag("warning", "stuck at end")
        elif name in repaired:
            status = tag("action", "repaired")
        elif name in proven:
            status = tag("proof", "proven dead, then cleared")
        elif name in alarmed or (p is not None and p.exact):
            status = tag("warning", "problem found")
        elif p is not None and p.silent:
            status = tag("info", "silent channels (unproven)")
        else:
            status = tag("healthy", "healthy")
        peak = "–"
        if p is not None and p.total:
            peak = ", ".join(f"{n:,} {w}" for n, w in ((p.certified, "proven"), (p.constant, "constant"),
                                                        (p.decayed, "decayed"), (p.silent, "silent")) if n)
        rows.append([f"<span class='mono'>{escape(name)}</span>", fmt(series[-1] if series else None),
                     fmt(statistics.median(series) if series else None), escape(peak), status])
    return rows


def _setup_section(f: RunFacts) -> str:
    s = f.start
    opt = s.get("optimizer") or {}
    items = [("Mode", "repair " + ("all stuck channels" if f.repair_mode == "all" else "on")
              if f.repair_mode else "watch only (no changes to the model)"),
             ("Checks", f"every {f.check_every:,} steps" + (f", {f.checks:,} in total" if f.checks else "")),
             ("Optimizer", json.dumps(opt, default=str) if opt else "–"),
             ("Monitor settings", json.dumps(s.get("config") or {}, default=str)),
             ("Software", f"{s.get('tool', '–')} · torch {s.get('torch', '–')} · python {s.get('python', '–')}"),
             ("Platform", str(s.get("platform", "–"))),
             ("Architecture SHA-256", str(s.get("architecture_sha256", "–")))]
    rows = [[escape(k), f"<span class='mono'>{escape(str(v))}</span>"] for k, v in items]
    out = "<h2>What was active</h2>"
    for m in f.manifests:
        at = f" (recorded at step {int(m['step']):,})" if isinstance(m.get("step"), int) else ""
        out += (f"<p>What the monitor could and could not do in this run{at}, as it recorded it. A layer "
                f"without a certificate gets alarms only; the reason is given.</p>"
                + table(["Mechanism", "State"], _manifest_rows(m)))
    if not f.manifests:
        out += ("<p class='sub small'>This record has no <span class='mono'>manifest</span> event "
                "(written by ormas-monitor 0.2 and later), so the list of active mechanisms is not shown.</p>")
    return out + "<h2>How the monitor was set up</h2>" + table(["Setting", "Value"], rows)


def _plain(v: object) -> str:
    """A manifest value as a short cell: yes/no for booleans, '–' for absent, JSON for nested data."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "–"
    if isinstance(v, (dict, list)):
        return f"<span class='mono'>{escape(json.dumps(v, default=str))}</span>"
    return escape(str(v))


def _manifest_rows(m: dict, prefix: str = "") -> list[list[str]]:
    """Flatten a manifest event into (mechanism, state) rows, whatever fields it carries."""
    rows: list[list[str]] = []
    for k, v in m.items():
        if not prefix and k in COMMON + ("step",):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and len(rows) < 200:
            rows += _manifest_rows(v, key + ".")
            continue
        if key.endswith("bound") and v is None:
            cell = "none enforced: the size of a repair's change is not limited"
        elif isinstance(v, bool):
            cell = tag("healthy", "active") if v else tag("info", "not active")
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            cols = list(dict.fromkeys(c for x in v for c in x))
            cell = table(cols, [[_plain(x.get(c)) for c in cols] for x in v[:200]])
        elif isinstance(v, list):
            cell = escape(", ".join(str(x) for x in v)) if v else "–"
        else:
            cell = escape(str(v)) if v is not None else "–"
        rows.append([f"<span class='mono'>{escape(key)}</span>", cell])
    return rows


STATUS_TAG = {"pass": ("healthy", "pass"), "fail": ("critical", "FAIL"), "absent": ("info", "absent"),
              "unchecked": ("warning", "unchecked"), "skipped": ("info", "skipped")}


def _status_tag(status: Optional[str]) -> str:
    """A pill for a `record.verify` status; colour is backed by the word."""
    sev, word = STATUS_TAG.get(status or "", ("info", "n/a"))
    return tag(sev, word)


def _record_section(f: RunFacts) -> str:
    t = f.trust
    if t.checks:
        rows = [[escape(label), _status_tag(status), escape(msg)] for label, status, msg in t.checks]
    else:
        rows = [["hash chain", _status_tag("pass" if t.intact else "fail"), escape(t.message)],
                ["metrics hash", _status_tag(t.metrics), escape(t.metrics_message)],
                ["seal and signature", _status_tag(None),
                 "not checked: this install has no ormas_monitor.record.verify"]]
    end_note = "" if f.finished else (
        "<p><b>This record has no <span class='mono'>run_end</span> event</b>: the run crashed, is still "
        "running, or its last lines were removed. Totals above are computed from what is there.</p>")
    return ("<h2>The record</h2>"
            f"<p>Every event above is a line in <span class='mono'>diary.jsonl</span>, each sealed with the "
            f"SHA-256 hash of the line before it, so editing, deleting or reordering any line breaks the chain. "
            f"A chain alone cannot detect a file rewritten whole with every hash recomputed; a signature by a "
            f"key the writer does not hold can. Re-check any time with "
            f"<span class='mono'>ormas-monitor verify {escape(f.run_dir.name)}</span>.</p>{end_note}"
            + table(["Check", "Result", "Detail"], rows))


def _about() -> str:
    return ("<h2>How to read this</h2><ul>"
            "<li><b>Gradient alarms</b> compare each layer with its own history, not a fixed threshold: "
            "an alarm means the gradient moved more than <i>sigma</i> standard deviations <i>and</i> at least "
            "10× away from that layer's normal, on two checks in a row.</li>"
            "<li><b>Proven dead</b> channels are a proof, not a guess: the incoming filter or the BatchNorm "
            "scale is exactly zero and the BatchNorm shift is ≤ 0, so the ReLU directly after it is closed "
            "for every input and gradient descent can never revive them. The proof assumes BatchNorm in "
            "training mode and no optimizer momentum left for the channel; the monitor checks both before "
            "issuing it.</li>"
            "<li><b>Decayed to ~0</b> channels have weights under 1e-5 of the layer's typical size and a "
            "shift ≤ 0: effectively dead, the usual end state of a dead channel under weight decay.</li>"
            "<li><b>Constant</b> channels output the same value for every input: alive, but they carry "
            "no information.</li>"
            "<li><b>Silent</b> channels produced no positive output on any input for several checks in a "
            "row. Likely dead, not proven; some silent units are normal in a healthy ReLU network.</li>"
            "<li><b>Repairs</b> (only when switched on) re-initialise stuck channels and reset their "
            "optimizer state, so training can use them again.</li>"
            "<li><b>Time spent stuck</b> counts proven-dead, constant and decayed channels only.</li></ul>")


def render(f: RunFacts) -> str:
    """The engineer report as one HTML document."""
    s = f.start
    cls, status, head, detail = _headline(f)
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["time"])) if s.get("time") else "time unknown"
    sub = " · ".join(escape(str(x)) for x in (
        s.get("model", "model"),
        f"{s['parameters']:,} parameters" if isinstance(s.get("parameters"), int) else None, when,
        duration(f.cost.total_seconds) if f.cost.total_seconds is not None else None,
        "repair on" if f.repair_mode else "watch only") if x)
    alarms = len(f.of("alarm")) + len(f.of("nonfinite_loss", "nonfinite_gradient"))
    proven = sum(int(e.get("count") or 0) for e in f.of("certificate"))
    tiles = [(f"{f.total_steps:,}", "training steps watched"),
             (f"{alarms:,}", "alarms"),
             (f"{proven:,}", "channels proven dead"),
             (f"{sum(int(e.get('count') or 0) for e in f.of('repair')):,}", "channels repaired"),
             (f"{f.cost.stuck_steps:,}" + (f" <span class='small'>({share(f.cost.stuck_steps, f.cost.total_steps)})</span>"
                                           if f.cost.stuck_steps else ""), "steps with a layer stuck"),
             ("intact" if f.trust.intact else "BROKEN", "record integrity")]
    skipped = len(f.of("skipped_step"))
    if skipped:
        tiles.insert(5, (f"{skipped:,}", "steps skipped by the AMP scaler"))
    marks = [(int(e["step"]), str(e.get("event")), severity(e), describe(e)) for e in f.shown
             if isinstance(e.get("step"), (int, float))]
    chart = loss_chart([(int(m["step"]), m.get("loss")) for m in f.metrics if "step" in m], marks,
                       list(f.cost.spans), f.config.max_chart_points)
    groups = _groups(f.shown)
    shown_groups = groups[:f.config.max_event_rows]
    rows = [[_step_cell(g), _where(g), _what(g), tag(severity(g[0]))] for g in shown_groups]
    more = "" if len(groups) <= len(shown_groups) else (
        f"<p class='sub'>{len(groups) - len(shown_groups):,} more rows not shown; every event is in "
        f"<span class='mono'>diary.jsonl</span>.</p>")
    layer_rows = _layer_rows(f)
    body = (
        "<p class='eyebrow'>Training health report · for engineers</p>"
        f"<h1>{escape(str(s.get('name', f.run_dir.name)))}</h1><p class='sub'>{sub}</p>"
        + verdict(cls, status, head, detail)
        + "<div class='tiles'>" + "".join(f"<div class='tile'><div class='n'>{n}</div><div class='l'>{escape(l)}</div></div>"
                                        for n, l in tiles) + "</div>"
        + "<p class='sub small noprint'>A plain-language version for non-engineers is in "
          "<a href='summary.html'>summary.html</a>.</p>"
        + _stuck_section(f)
        + (f"<h2>Loss</h2>{chart}{legend([m[1] for m in marks], bool(f.cost.spans))}"
           f"<p class='sub small'>Hover a marker for details.</p>" if chart else
           ("<h2>Loss</h2><p class='sub'>No per-check loss values in this record "
            "(<span class='mono'>metrics.jsonl</span> is missing or has no loss).</p>"))
        + _lead_section(f)
        + (f"<h2>What happened</h2>{table(['Step', 'Where', 'What', ''], rows, num=(0,))}{more}" if rows else
           "<h2>What happened</h2><p>Nothing to report: no alarms, stuck channels, loss spikes or repairs.</p>")
        + ("<h2>Layers</h2>" + table(["Layer", "Gradient now", "Median over run", "Stuck channels (peak)",
                                      "Status"], layer_rows, num=(1, 2)) if layer_rows else "")
        + _setup_section(f) + _record_section(f) + _about())
    return page(f"Training health report: {s.get('name', f.run_dir.name)}", body)


def write(f: RunFacts, path: Optional[Path] = None) -> Path:
    """Write `report.html` into the run directory (or `path`) and return its path."""
    out = Path(path) if path is not None else f.run_dir / "report.html"
    out.write_text(render(f), encoding="utf-8")
    return out
