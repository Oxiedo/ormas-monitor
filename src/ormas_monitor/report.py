"""Self-contained HTML reports: one for a monitored run, one for a checkpoint check."""
from __future__ import annotations

import json
import statistics
import time
from html import escape
from pathlib import Path

from . import __version__
from .diary import read, verify

CSS = """
:root{--bg:#fff;--fg:#1d2433;--mute:#5a606b;--line:#d5dae0;--card:#f6f8fa;--teal:#0f766e;
--tealbg:#e6f2f0;--red:#c8102e;--redbg:#fbeaec;--amber:#a15c00;--amberbg:#fdf3e1;--blue:#2b6cb0;--bluebg:#e8f0fa}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--fg:#e6e9ee;--mute:#9aa3ae;--line:#2c333d;
--card:#1a2029;--teal:#4fb8ad;--tealbg:#14302d;--red:#ff6b7f;--redbg:#3a1a20;--amber:#f0b45a;
--amberbg:#352712;--blue:#7fb0ea;--bluebg:#16253a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Fira Sans",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 48px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--mute);margin:0}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.verdict{margin:22px 0;padding:16px 18px;border-radius:10px;border:1px solid}
.ok{background:var(--tealbg);border-color:var(--teal)}.bad{background:var(--redbg);border-color:var(--red)}
.warn{background:var(--amberbg);border-color:var(--amber)}
.verdict b{font-size:17px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--card);border-radius:10px;padding:12px 14px}.tile .n{font-size:24px;font-weight:700}
.tile .l{color:var(--mute);font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}
.scroll{overflow-x:auto}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;white-space:nowrap}
.critical{background:var(--redbg);color:var(--red)}.warning{background:var(--amberbg);color:var(--amber)}
.action{background:var(--bluebg);color:var(--blue)}.proof{background:var(--redbg);color:var(--red)}.info{background:var(--card);color:var(--mute)}
.healthy{background:var(--tealbg);color:var(--teal)}
.finding{border-left:4px solid var(--line);padding:8px 14px;margin:10px 0;background:var(--card);border-radius:0 8px 8px 0}
.finding.f-critical{border-color:var(--red)}.finding.f-warning{border-color:var(--amber)}
.finding p{margin:4px 0 0;color:var(--mute)}svg text{fill:var(--mute);font-size:11px}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);color:var(--mute);font-size:13px}
footer a{color:var(--teal)}
"""

LABEL = {"critical": "Critical", "warning": "Warning", "proof": "Proof", "action": "Repair", "info": "Info"}


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>{body}"
            f"<footer>Generated on this machine by <b>ormas-monitor {__version__}</b>. Nothing was "
            f"uploaded anywhere. Dead-channel certificates follow Theorem 2 of R. A. D. Raadh, "
            f"<i>Self-Repair as a Bounded Disturbance</i> (DeepMath 2026). Built by "
            f"<a href='https://oxiedo.com'>Oxiedo</a>. Questions or want help reading this report? "
            f"<a href='mailto:raadh@oxiedo.com'>raadh@oxiedo.com</a></footer></main></body></html>")


def _fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.3g}"
    if isinstance(x, int):
        return f"{x:,}"
    return escape(str(x))


def _chart(points: list[tuple[int, float]], marks: list[tuple[int, str, str]]) -> str:
    pts = [(s, v) for s, v in points if v is not None]
    if len(pts) < 2:
        return ""
    W, H, L, R, T, B = 800, 220, 52, 12, 12, 28
    xs = [s for s, _ in pts]
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    hi = hi if hi > lo else lo + 1
    x0, x1 = min(xs), max(xs) or 1

    def X(s):
        return L + (s - x0) / max(x1 - x0, 1) * (W - L - R)

    def Y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)
    line = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts)
    colours = {"critical": "var(--red)", "warning": "var(--amber)", "action": "var(--blue)", "proof": "var(--red)",
               "info": "var(--mute)"}
    ms = "".join(
        f"<line x1={X(s):.1f} x2={X(s):.1f} y1={T} y2={H - B} stroke='{colours.get(sev, 'var(--mute)')}' "
        f"stroke-width=1.5 stroke-dasharray='4 3'><title>step {s:,}: {escape(t)}</title></line>"
        for s, sev, t in marks if x0 <= s <= x1)
    return (f"<svg viewBox='0 0 {W} {H}' width=100% role=img aria-label='Training loss over steps, "
            f"with dashed lines at each event'>"
            f"<line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke='var(--line)'/>"
            f"<text x={L - 6} y={T + 4} text-anchor=end>{hi:.3g}</text>"
            f"<text x={L - 6} y={H - B} text-anchor=end>{lo:.3g}</text>"
            f"<text x={L} y={H - 8}>step {x0:,}</text><text x={W - R} y={H - 8} text-anchor=end>step {x1:,}</text>"
            f"{ms}<polyline points='{line}' fill=none stroke='var(--teal)' stroke-width=2/></svg>")


def write_all(run_dir: str | Path) -> dict[str, Path]:
    """Write every report for a monitored run. Monitor.close() and `ormas-monitor report` call this.
    Returns {report kind: path}."""
    return {"report": write_run_report(run_dir)}


def write_run_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    events = read(run_dir / "diary.jsonl")
    intact, n_events, integrity = verify(run_dir / "diary.jsonl")
    metrics = [json.loads(line) for line in open(run_dir / "metrics.jsonl") if line.strip()]
    start = next((e for e in events if e["event"] == "run_start"), {})
    end = next((e for e in events if e["event"] == "run_end"), {})
    shown = [e for e in events if e["event"] not in ("run_start", "run_end", "structure")]
    counts = end.get("counts", {})

    critical = [e for e in shown if e.get("severity") == "critical"]
    problems = [e for e in shown if e.get("severity") in ("critical", "warning")]
    repairs = [e for e in shown if e["event"] == "repair"]
    if critical:
        cls, head = "bad", f"Critical problems found: {escape(critical[0]['event'].replace('_', ' '))}"
    elif problems:
        layers = sorted({e.get("layer") for e in problems if e.get("layer")})
        cls = "warn"
        head = (f"Problems found in {len(layers)} layer{'s' if len(layers) != 1 else ''}"
                + (f", {len(repairs)} repair{'s' if len(repairs) != 1 else ''} applied" if repairs else ""))
    else:
        cls, head = "ok", f"No problems found in {len(metrics):,} checks"
    first = problems[0] if problems else None
    proofs = [e for e in shown if e["event"] == "certificate"]
    sub = ""
    if proofs:
        sub = ("Proven dead: " + ", ".join(f"<span class=mono>{escape(e['layer'])}</span> ({e['count']} channels, "
                                          f"step {e['step']:,})" for e in proofs) + ".<br>")
    sub += (f"First sign of trouble at step {first['step']:,} in <span class=mono>{escape(str(first.get('layer') or 'the loss'))}"
            f"</span>: {escape(_describe(first))}" if first
            else "Every layer's gradient stayed within its own normal range and no channel got stuck.")

    tiles = [(f"{end.get('steps', 0):,}", "training steps watched"),
             (f"{counts.get('alarms', 0) + counts.get('nonfinite', 0)}", "alarms"),
             (f"{counts.get('stuck_layers', 0)}", "times a layer got stuck channels"),
             (f"{counts.get('channels_repaired', 0):,}", "channels repaired"),
             ("intact" if intact else "BROKEN", "record integrity")]

    rows = "".join(
        f"<tr><td class=num>{e.get('step', 0):,}</td><td class=mono>{escape(str(e.get('layer') or '–'))}</td>"
        f"<td>{escape(_describe(e))}</td><td><span class='tag {e.get('severity', 'info')}'>"
        f"{LABEL.get(e.get('severity', 'info'), 'Info')}</span></td></tr>" for e in shown)

    layer_rows = _layer_rows(metrics, shown)
    marks = [(e["step"], e.get("severity", "info"), _describe(e)) for e in shown]
    chart = _chart([(m["step"], m.get("loss")) for m in metrics], marks)
    opt = start.get("optimizer") or {}
    body = (
        f"<h1>Training health report</h1><p class=sub>{escape(str(start.get('name', run_dir.name)))} · "
        f"{escape(str(start.get('model', '')))} · {_fmt(start.get('parameters'))} parameters · "
        f"{escape(time.strftime('%Y-%m-%d %H:%M', time.localtime(start.get('time', time.time()))))}</p>"
        f"<div class='verdict {cls}'><b>{head}</b><br>{sub}</div>"
        f"<div class=tiles>" + "".join(f"<div class=tile><div class=n>{n}</div><div class=l>{l}</div></div>"
                                       for n, l in tiles) + "</div>"
        + (f"<h2>Loss</h2>{chart}<p class=sub>Dashed lines mark events; hover for details.</p>" if chart else "")
        + (f"<h2>What happened</h2><div class=scroll><table><tr><th>Step</th><th>Where</th><th>What</th>"
           f"<th></th></tr>{rows}</table></div>" if shown else "")
        + f"<h2>Layers</h2><div class=scroll><table><tr><th>Layer</th><th>Gradient now</th><th>Normal</th>"
          f"<th>Stuck channels (peak)</th><th>Status</th></tr>{layer_rows}</table></div>"
        + "<h2>The record</h2>"
          f"<p>Every event above is in <span class=mono>diary.jsonl</span>, one line each, each line "
          f"sealed with the SHA-256 hash of the line before it. Editing, deleting or reordering any "
          f"line breaks the chain. Check: <b>{escape(integrity)}</b>. The per-check measurements are "
          f"in <span class=mono>metrics.jsonl</span>, whose hash is sealed into the last event. "
          f"Re-check any time with <span class=mono>ormas-monitor verify {escape(run_dir.name)}</span>.</p>"
          f"<details><summary>Run settings</summary><p class=mono>optimizer {escape(json.dumps(opt))}<br>"
          f"monitor {escape(json.dumps(start.get('config', {})))}<br>torch {escape(str(start.get('torch')))}"
          f" · python {escape(str(start.get('python')))} · architecture sha256 "
          f"{escape(str(start.get('architecture_sha256', ''))[:16])}…</p></details>"
        + _about())
    path = run_dir / "report.html"
    path.write_text(_page("Training health report", body))
    return path


def _describe(e: dict) -> str:
    ev = e["event"]
    if ev == "alarm":
        word = {"no_gradient": "gradient is exactly zero", "vanishing": "gradient collapsed",
                "exploding": "gradient exploded"}.get(e.get("diagnosis"), e.get("diagnosis"))
        return f"{word}: {e['grad_norm']:.3g} against a normal of {e['baseline']:.3g}"
    if ev == "alarm_cleared":
        return "gradient back in its normal range"
    if ev == "channels":
        parts = [f"{e.get('n_' + k, len(e.get(k, [])))} {w}" for k, w in
                 (("certified", "certified dead"), ("flat", "constant"), ("decayed", "decayed to ~0"),
                  ("observed", "silent for several checks"))
                 if e.get("n_" + k, len(e.get(k, [])))]
        return f"channels stuck ({e.get('level')} of {e.get('channels')}): " + ", ".join(parts)
    if ev == "channels_cleared":
        return "all channels active again"
    if ev == "certificate":
        return (f"{e['count']} channels proven dead: zero filter or BatchNorm scale, shift ≤ 0, "
                f"ReLU closed for every input, so no gradient can ever reach them")
    if ev == "repair":
        return f"re-initialised {e['count']} stuck channels and reset their optimizer state"
    if ev == "loss_spike":
        return f"loss spiked to {e['value']:.4g} (normal {e['baseline']:.4g})"
    if ev == "loss_recovered":
        return f"loss back to its normal range ({e['value']:.4g})"
    if ev == "nonfinite_loss":
        return f"loss became {e.get('value')}"
    if ev == "nonfinite_gradient":
        return f"non-finite gradients in {len(e.get('layers', []))} layers"
    return ev.replace("_", " ")


def _layer_rows(metrics: list[dict], events: list[dict]) -> str:
    if not metrics:
        return ""
    layers = list(metrics[-1].get("grad", {}))
    for m in metrics:
        for k in m.get("grad", {}):
            if k not in layers:
                layers.append(k)
    status: dict[str, tuple[str, str]] = {}
    for e in events:
        if not e.get("layer"):
            continue
        if e["event"] in ("alarm", "channels"):
            status[e["layer"]] = ("warning", "problem found")
        elif e["event"] == "repair":
            status[e["layer"]] = ("action", "repaired")
    rows = []
    for name in layers:
        series = [m["grad"][name] for m in metrics if name in m.get("grad", {})]
        normal = statistics.median(series) if series else None
        peak = max((sum(m.get("stuck", {}).get(name, [0])) for m in metrics), default=0)
        tag, word = status.get(name, ("healthy", "healthy"))
        rows.append(f"<tr><td class=mono>{escape(name)}</td><td class=num>{_fmt(series[-1] if series else None)}</td>"
                    f"<td class=num>{_fmt(normal)}</td><td class=num>{peak or '–'}</td>"
                    f"<td><span class='tag {tag}'>{word}</span></td></tr>")
    return "".join(rows)


def _about() -> str:
    return ("<h2>How to read this</h2><ul>"
            "<li><b>Gradient alarms</b> compare each layer with its own history, not a fixed threshold: "
            "an alarm means the gradient moved more than 2.5 standard deviations <i>and</i> at least "
            "10× away from that layer's normal, on two checks in a row.</li>"
            "<li><b>Certified dead</b> channels are a proof, not a guess: the incoming filter or the "
            "BatchNorm scale is exactly zero and the BatchNorm shift is ≤ 0, so the ReLU directly after "
            "it is closed for every input and gradient descent can never revive them. It assumes "
            "BatchNorm in training mode and that the optimizer's momentum for the channel has decayed.</li>"
            "<li><b>Decayed to ~0</b> channels have weights under 1e-5 of the layer's typical size and a "
            "shift ≤ 0: effectively dead, the usual end state of a dead channel under weight decay.</li>"
            "<li><b>Constant</b> channels output the same value for every input: alive, but they carry "
            "no information.</li>"
            "<li><b>Silent</b> channels produced no positive output on any input, several checks in a row. "
            "They are likely dead but not proven so.</li>"
            "<li><b>Repairs</b> (only if switched on) re-initialise stuck channels and reset their "
            "optimizer state, so training can use them again.</li></ul>")


def write_check_report(result: dict, path: str | Path) -> Path:
    f = result["findings"]
    crit = [x for x in f if x["severity"] == "critical"]
    warn = [x for x in f if x["severity"] == "warning"]
    if crit:
        cls, head = "bad", crit[0]["title"]
    elif warn:
        cls, head = "warn", f"{len(warn)} problem{'s' if len(warn) != 1 else ''} found"
    else:
        cls, head = "ok", "No problems found in the weights"
    findings = "".join(
        f"<div class='finding f-{x['severity']}'><span class='tag {x['severity']}'>{LABEL[x['severity']]}</span> "
        f"<b>{escape(x['title'])}</b><p>{escape(x.get('detail', ''))}</p></div>" for x in f)
    layer_rows = "".join(
        f"<tr><td class=mono>{escape(l['layer'])}</td><td class=mono>{escape(l['batchnorm'])}</td>"
        f"<td class=num>{l['channels']}</td><td class=num>{len(l['certified']) or '–'}</td>"
        f"<td class=num>{len(l['flat']) or '–'}</td><td class=num>{len(l.get('decayed', [])) or '–'}</td>"
        f"<td class=num>{len(l.get('dormant', [])) or '–'}</td>"
        f"<td class=num>{l['collapsed_var'] or '–'}</td></tr>"
        for l in result["layers"])
    diff_html = ""
    if "diff" in result:
        d = result["diff"]
        top = "".join(f"<tr><td class=mono>{escape(r['tensor'])}</td><td class=num>{r['change']:.3g}</td></tr>"
                      for r in d["changed"][:20])
        diff_html = (f"<h2>What changed since {escape(Path(result['against']).name)}</h2>"
                     f"<div class=tiles><div class=tile><div class=n>{len(d['changed']) - len(d['unchanged'])}</div>"
                     f"<div class=l>tensors changed</div></div><div class=tile><div class=n>{len(d['unchanged'])}</div>"
                     f"<div class=l>tensors exactly unchanged</div></div><div class=tile><div class=n>"
                     f"{len(d['added'])}/{len(d['removed'])}</div><div class=l>added / removed</div></div>"
                     f"<div class=tile><div class=n>{sum(n['count'] for n in d.get('newly_dead', []))}</div>"
                     f"<div class=l>channels newly dead</div></div></div>"
                     f"<h3>Largest relative changes (‖new − old‖ / ‖old‖)</h3><div class=scroll><table>"
                     f"<tr><th>Tensor</th><th>Change</th></tr>{top}</table></div>")
    body = (f"<h1>Checkpoint health report</h1><p class=sub>{escape(Path(result['file']).name)} · "
            f"{result['parameters']:,} parameters · {result['tensors']:,} tensors · weights only, no data</p>"
            f"<div class='verdict {cls}'><b>{escape(head)}</b></div>"
            + (f"<h2>Findings</h2>{findings}" if f else "")
            + (f"<h2>BatchNorm layers</h2><div class=scroll><table><tr><th>Layer</th><th>BatchNorm</th>"
               f"<th>Channels</th><th>Dead*</th><th>Constant</th><th>Decayed to ~0</th><th>Dormant†</th><th>Var ≈ 0</th></tr>{layer_rows}</table></div>"
               f"<p class=sub>* Proven dead when a ReLU directly follows the BatchNorm. † Estimated: shift ≤ −3×|scale|.</p>"
               if result["layers"] else "")
            + diff_html)
    path = Path(path)
    path.write_text(_page("Checkpoint health report", body))
    return path
