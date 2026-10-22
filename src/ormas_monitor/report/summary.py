"""The executive summary (`summary.html` and `summary.md`): one page for a CEO, CFO or compliance
lead. Plain words, no jargon in the prose, every number traceable to the run's record.

The text is built once as a `Summary` (plain strings) and rendered twice, so the HTML and the
Markdown can never disagree.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Optional

from ._html import duration, legend, page, plural, share, timeline, verdict
from .facts import RunFacts

NOT_TOLD = (
    "It does not make anyone compliant with the EU AI Act or any other rule. The record is evidence "
    "for technical documentation, not a certificate.",
    "It does not measure how accurate or fit for purpose the model is. A run with no problems found "
    "can still produce a poor model.",
    "It never reads the training data, so it cannot find mislabelled or harmful examples.",
    "Proofs that a part of the model is dead apply only to one common building block "
    "(convolution or linear layer, then batch normalisation, then ReLU). Transformers and large "
    "language models get warnings only, never proofs.",
    "The tool has so far been tested on one Apple-silicon Mac, not yet on NVIDIA/Linux, multiple GPUs "
    "or at large scale.",
)

GLOSSARY = {
    "layer": ("Layer", "One stage of the neural network. Layers have internal names such as conv2.0."),
    "channel": ("Channel", "One small unit inside a layer. A layer has tens to thousands of them."),
    "stuck": ("Stuck", "A channel that, judging from its weights, outputs nothing useful: proven dead, "
                       "a constant, or shrunk to almost zero."),
    "proven": ("Proven dead", "A mathematical certificate that a channel can never learn again on its "
                              "own, however long training continues."),
    "repair": ("Repair", "Resetting stuck channels to fresh starting values so training can use them "
                         "again. Only happens if switched on."),
    "alarm": ("Warning (alarm)", "A layer's learning signal moved far outside that layer's own normal "
                                 "range."),
    "record": ("Tamper-evident record", "The log of everything the monitor saw. Each line carries a "
                                        "fingerprint of the line before it, so edits show."),
    "signature": ("Signature", "A digital seal made with a private key. It shows who vouched for the "
                               "record and detects a record rewritten from scratch."),
}


@dataclass
class Summary:
    """The executive summary as plain text, before rendering."""

    title: str
    meta: str
    status: str
    tone: str                      # ok | warn | bad | fixed (css class)
    headline: str
    happened: str
    cost_lines: list[str]
    cost_note: str
    estimate: Optional[str]
    done: str
    trust: str
    not_told: tuple[str, ...]
    glossary: list[tuple[str, str]] = field(default_factory=list)


def _names(layers: list[str], limit: int = 4) -> str:
    if len(layers) <= limit:
        return " and ".join([", ".join(layers[:-1]), layers[-1]]) if len(layers) > 1 else layers[0]
    return ", ".join(layers[:limit]) + f" and {len(layers) - limit} more"


def _trust_sentence(f: RunFacts) -> str:
    """One sentence on whether the record can be relied on, from the record check's statuses."""
    t = f.trust
    entries = f"all {t.events:,} entries of the record check out"
    if not t.intact:
        return (f"No: the record has been altered or damaged since it was written ({t.message}), so treat "
                f"everything in this summary as unconfirmed.")
    if t.seal == "fail":
        return ("No: the event log links up, but it no longer matches its seal (record.json), so it was "
                "changed after sealing.")
    if t.signature == "fail":
        return ("No: the event log links up, but its signature does not verify, so the record was changed "
                "after signing or the signature is not from the expected key.")
    if t.metrics == "fail":
        what = ("the measurements file that belongs with it is missing" if "missing" in t.metrics_message
                else "the measurements file no longer matches the fingerprint sealed into the log, so the "
                     "measurements may have been changed")
        return f"Partly: {entries}, but {what}."
    if t.signature == "pass":
        return (f"Yes: {entries} and the record is signed by {t.signer or 'a known key'}, verified with that "
                f"key, so nothing was edited, removed or rewritten after signing (a signature shows who "
                f"vouched for the record, not that training happened as recorded).")
    if t.signature == "unchecked":
        return (f"Partly: {entries}, so no line was edited, removed or reordered after it was written; it "
                f"carries a signature by {t.signer or 'an unnamed key'}, which was not verified here because "
                f"that needs the signer's public key (ormas-monitor report --key).")
    tail = ("it is not signed, so it cannot show who produced it or rule out the whole file being rewritten"
            if t.signature == "absent" else
            "its signature was not checked, so it cannot show who produced it or rule out the whole file "
            "being rewritten")
    return f"Partly: {entries}, so no line was edited, removed or reordered after it was written, but {tail}."


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n:,} times")


def _money(x: float) -> str:
    return "under $0.01" if 0 < x < 0.01 else f"about ${x:,.2f}"


def build(f: RunFacts) -> Summary:
    """Turn the run's facts into the summary text."""
    s = f.start
    c = f.cost
    used = {"layer", "channel", "record"}
    certs = f.of("certificate")
    repairs = f.of("repair")
    crit = [e for e in f.shown if e.get("severity") == "critical"]
    alarms = f.of("alarm")
    spikes = f.of("loss_spike")
    eps = f.episodes
    still = [ep for ep in eps if ep.open_at_end]
    stuck_layers = sorted({ep.layer for ep in eps}, key=lambda x: (min(ep.start for ep in eps if ep.layer == x), x))
    proven_channels = sum(int(e.get("count") or 0) for e in certs)
    fixed_channels = sum(int(e.get("count") or 0) for e in repairs)
    first_stuck = min((ep.start for ep in eps), default=None)
    every = f"every {f.check_every:,} steps"

    # ---- headline
    if not f.trust.intact:
        tone, status = "bad", "Record altered"
        headline = "The record of this run fails its integrity check, so its contents cannot be relied on."
    elif crit:
        tone, status = "bad", "Training broke"
        headline = (f"Training broke at step {int(crit[0].get('step', 0)):,}: it produced invalid numbers "
                    f"(not-a-number or infinity).")
    elif still:
        tone, status = "bad", "Problem, not repaired"
        headline = (f"Part of the model stopped learning (detected at step {first_stuck:,}) and was still "
                    f"stuck when the run ended.")
    elif eps and repairs:
        tone, status = "fixed", "Problem found and repaired"
        headline = (f"Part of the model stopped learning (detected at step {first_stuck:,}); the monitor repaired it at "
                    f"step {int(repairs[0]['step']):,}, and no part was stuck after that.")
    elif eps:
        tone, status = "warn", "Problem, cleared"
        last = max(ep.end for ep in eps)
        headline = f"Part of the model was stuck from step {first_stuck:,} until step {last:,}, then recovered."
    elif alarms or spikes or f.silent_peaks or any(e.get("event") == "sink_error" for e in f.shown):
        tone, status = "warn", "Warnings only"
        headline = "Nothing was proven broken, but the monitor raised warnings worth a look."
    else:
        tone, status = "ok", "Healthy"
        headline = f"No problems found in {f.total_steps:,} training steps."

    # ---- what happened
    p = [f"The monitor checked the model {every} over {f.total_steps:,} training steps"
         + (f" ({duration(c.total_seconds)})." if c.total_seconds is not None else ".")]
    if crit:
        p.append(f"At step {int(crit[0].get('step', 0)):,} the training produced invalid numbers, which "
                 f"stops normal learning.")
    if eps:
        n = len(stuck_layers)
        p.append(f"At step {first_stuck:,} it found {plural(n, 'layer')} ({_names(stuck_layers)}) with stuck "
                 f"channels" + (f"; {proven_channels:,} channels were proven dead, meaning they could never "
                                f"learn again on their own." if proven_channels else "."))
        used |= {"stuck"} | ({"proven"} if proven_channels else set())
        if still:
            p.append("They were still stuck when the run ended." if len(still) == len(eps) or
                     {ep.layer for ep in still} == set(stuck_layers) else
                     f"{plural(len({ep.layer for ep in still}), 'layer')} "
                     f"({_names(sorted({ep.layer for ep in still}))}) were still stuck when the run ended.")
        else:
            p.append(f"By step {max(ep.end for ep in eps):,} no layer was stuck any more.")
    if spikes:
        p.append(f"The training loss (the error the model is trained to reduce) jumped "
                 f"{_times(len(spikes))}, first at step {int(spikes[0]['step']):,}.")
    if alarms:
        al = sorted({str(e.get("layer")) for e in alarms if e.get("layer")})
        p.append(f"The learning signal of {plural(len(al), 'layer')} fell to zero or moved far outside its "
                 f"own normal range ({plural(len(alarms), 'warning')}, first at step {int(alarms[0]['step']):,}).")
        used.add("alarm")
    if f.silent_peaks and not eps:
        layer, n = max(f.silent_peaks.items(), key=lambda kv: kv[1])
        p.append(f"In {plural(len(f.silent_peaks), 'layer')} some channels gave no output for several checks "
                 f"in a row (up to {n:,} in {layer}). They may be inactive, but this is not proven, and some "
                 f"inactive channels are normal in a healthy network.")
    sink_errors = [e for e in f.shown if e.get("event") == "sink_error"]
    if sink_errors:
        p.append(f"{plural(len(sink_errors), 'output plug-in')} failed and "
                 f"{'was' if len(sink_errors) == 1 else 'were'} switched off; training and this record "
                 f"were not affected.")
    if not f.finished:
        p.append("The record has no end-of-run entry: the run crashed, is still running, or the end of the "
                 "log is missing.")
    if tone == "ok":
        p.append("Every layer's learning signal stayed within its own normal range and no part of the model got stuck.")

    # ---- what it cost
    cost_lines: list[str] = []
    note = ""
    estimate = None
    if c.stuck_steps:
        cost_lines.append(f"{c.stuck_steps:,} of {c.total_steps:,} training steps "
                          f"({share(c.stuck_steps, c.total_steps)}) ran with part of the model stuck.")
        if c.stuck_seconds is not None:
            cost_lines.append(f"{duration(c.stuck_seconds)} of wall-clock time"
                              + (f" ({share(c.stuck_seconds, c.total_seconds)} of {duration(c.total_seconds)})"
                                 if c.total_seconds else "") + ".")
        else:
            cost_lines.append("Wall-clock time: unknown (the record's timestamps do not cover the interval).")
        if c.proven_steps and c.proven_steps != c.stuck_steps:
            cost_lines.append(f"{c.proven_steps:,} of those steps had channels proven dead.")
        note = (f"Counted from the first check that saw the problem; it may have started up to "
                f"{c.resolution:,} steps earlier. Time comes from the record's own timestamps.")
        d = c.dollars(c.stuck_seconds)
        if d is not None:
            g = c.inputs.gpu_count
            assumed = "" if c.inputs.gpus is not None else " (GPU count not given; 1 assumed)"
            estimate = (f"ESTIMATE: {_money(d)} of compute. Formula: {c.stuck_seconds / 3600:.3f} h stuck "
                        f"× {g} GPU{'s' if g != 1 else ''}{assumed} × ${c.inputs.gpu_hour_cost:,.2f} per "
                        f"GPU-hour. Built from your price inputs, not measured.")
    else:
        cost_lines.append("Nothing measurable: at no check was any part of the model stuck.")
        if c.dollars(c.total_seconds) is not None:
            g = c.inputs.gpu_count
            estimate = (f"ESTIMATE: the whole run cost {_money(c.dollars(c.total_seconds) or 0.0)} of compute "
                        f"({c.total_seconds / 3600:.3f} h × {g} GPU{'s' if g != 1 else ''} × "
                        f"${c.inputs.gpu_hour_cost:,.2f} per GPU-hour), none of it spent while stuck. "
                        f"Built from your price inputs, not measured.")

    # ---- what was done
    if repairs:
        used.add("repair")
        steps = sorted({int(e["step"]) for e in repairs})
        layers = sorted({str(e.get("layer")) for e in repairs})
        done = (f"The monitor repaired {fixed_channels:,} channels in {plural(len(layers), 'layer')} "
                f"(at step{'s' if len(steps) > 1 else ''} {', '.join(f'{x:,}' for x in steps)}): it reset them to "
                f"fresh starting values and cleared the optimizer's memory of them, so training could use "
                f"them again.")
        if any("bound" in e and e["bound"] is None for e in repairs):
            done += " No limit was enforced on how much a repair could change the layer."
        cleared = [ep for ep in eps if not ep.open_at_end and ep.repaired]
        if cleared and not still:
            done += " No layer was stuck at the end of the run."
        elif still:
            done += f" {plural(len({ep.layer for ep in still}), 'layer')} still had stuck channels at the end."
    elif f.watch_only and (eps or crit):
        done = ("Nothing. The monitor ran in watch-only mode: it recorded the problem but did not change the "
                "model. With repair switched on it re-initialises stuck channels automatically.")
    elif f.watch_only:
        done = "Nothing needed doing. The monitor ran in watch-only mode and did not change the model."
    elif eps:
        done = "Repair was switched on, but no repair was recorded for these channels."
    else:
        done = "Nothing needed doing. Repair was switched on, but no layer qualified for it."

    if f.trust.signature in ("pass", "fail", "unchecked"):
        used.add("signature")
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["time"])) if s.get("time") else "date unknown"
    meta = " · ".join(str(x) for x in (s.get("model"), when,
                                        "repair on" if f.repair_mode else "watch-only mode") if x)
    order = ["layer", "channel", "stuck", "proven", "alarm", "repair", "record", "signature"]
    return Summary(title=str(s.get("name", f.run_dir.name)), meta=meta, status=status, tone=tone,
                   headline=headline, happened=" ".join(p), cost_lines=cost_lines, cost_note=note,
                   estimate=estimate, done=done, trust=_trust_sentence(f), not_told=NOT_TOLD,
                   glossary=[GLOSSARY[k] for k in order if k in used])


def render_html(sm: Summary, f: RunFacts) -> str:
    """One printable page."""
    c = f.cost
    marks = [(int(e["step"]), str(e["event"]), str(e["event"]).replace("_", " "))
             for e in f.of("certificate", "repair")]
    bar = ""
    if c.spans:
        bar = (timeline(c.total_steps, list(c.spans), marks, "Run timeline; hatched where part of the model "
                                                             "was stuck")
               + legend([m[1] for m in marks], True))
    body = (
        "<p class='eyebrow'>Training run summary · for decision-makers</p>"
        f"<h1>{escape(sm.title)}</h1><p class='sub'>{escape(sm.meta)}</p>"
        + verdict(sm.tone, sm.status, sm.headline)
        + f"<h2>What happened</h2><p>{escape(sm.happened)}</p>"
        + "<h2>What it cost</h2>"
        + "".join(f"<p>{escape(x)}</p>" for x in sm.cost_lines) + bar
        + (f"<p class='sub small'>{escape(sm.cost_note)}</p>" if sm.cost_note else "")
        + (f"<div class='estimate'><span class='label'>ESTIMATE</span> {escape(sm.estimate[len('ESTIMATE: '):])}</div>"
           if sm.estimate else "")
        + f"<h2>What was done</h2><p>{escape(sm.done)}</p>"
        + f"<h2>Can you trust this record?</h2><p>{escape(sm.trust)}</p>"
        + "<div class='box'><h3>What this report does not tell you</h3><ul>"
        + "".join(f"<li>{escape(x)}</li>" for x in sm.not_told) + "</ul></div>"
        + "<h2>Words used here</h2><dl class='gloss'>"
        + "".join(f"<dt>{escape(t)}</dt><dd>{escape(d)}</dd>" for t, d in sm.glossary) + "</dl>"
        + "<p class='sub small'>The technical detail behind every sentence is in "
          "<a href='report.html'>report.html</a> (engineer report) and in the record itself, "
          "<span class='mono'>diary.jsonl</span>.</p>")
    return page(f"Run summary: {sm.title}", body, narrow=True)


def render_md(sm: Summary) -> str:
    """The same summary as Markdown, for pasting into a ticket, an email or documentation."""
    lines = [f"# {sm.title}", "", f"*Training run summary · {sm.meta}*", "",
             f"> **{sm.status.upper()}: {sm.headline}**", "",
             "## What happened", "", sm.happened, "", "## What it cost", ""]
    lines += [f"- {x}" for x in sm.cost_lines]
    if sm.cost_note:
        lines += ["", f"_{sm.cost_note}_"]
    if sm.estimate:
        lines += ["", f"**{sm.estimate.split(':', 1)[0]}:**{sm.estimate.split(':', 1)[1]}"]
    lines += ["", "## What was done", "", sm.done, "", "## Can you trust this record?", "", sm.trust, "",
              "## What this report does not tell you", ""]
    lines += [f"- {x}" for x in sm.not_told]
    lines += ["", "## Words used here", ""]
    lines += [f"- **{t}**: {d}" for t, d in sm.glossary]
    lines += ["", "Technical detail: `report.html` (engineer report) and the record itself, `diary.jsonl`.",
              "Generated on this machine by ormas-monitor; nothing was uploaded.", ""]
    return "\n".join(lines)


def write(f: RunFacts, html_path: Optional[Path] = None, md_path: Optional[Path] = None) -> tuple[Path, Path]:
    """Write `summary.html` and `summary.md` (default: into the run directory)."""
    sm = build(f)
    h = Path(html_path) if html_path is not None else f.run_dir / "summary.html"
    m = Path(md_path) if md_path is not None else f.run_dir / "summary.md"
    h.write_text(render_html(sm, f), encoding="utf-8")
    m.write_text(render_md(sm), encoding="utf-8")
    return h, m
