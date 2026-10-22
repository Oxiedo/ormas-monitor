"""Shared HTML pieces for every report: stylesheet (screen, dark mode, print), page frame,
number and time formatting, status tags, and the two inline-SVG figures.

Everything is inline: a report opens offline, from a file, years later, with no fonts or
scripts fetched from anywhere.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Iterable, Sequence

from .. import __version__

CSS = """
:root{--bg:#fff;--fg:#1d2433;--mute:#566070;--line:#d5dae0;--card:#f4f6f8;--teal:#0f766e;
--tealbg:#e3f1ef;--red:#b4152b;--redbg:#fbeaec;--amber:#8f5200;--amberbg:#fdf1dc;--blue:#245f9e;
--bluebg:#e6eef9;--hatch:#b4152b}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--fg:#e6e9ee;--mute:#a1a9b4;--line:#2c333d;
--card:#1a2029;--teal:#4fb8ad;--tealbg:#13302c;--red:#ff7a8a;--redbg:#3a1a20;--amber:#f0b45a;
--amberbg:#352712;--blue:#86b6ee;--bluebg:#16253a;--hatch:#ff7a8a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 20px 48px}
main.narrow{max-width:780px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}
h2{font-size:19px;margin:34px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px}
p{margin:8px 0}
.eyebrow{text-transform:uppercase;letter-spacing:.06em;font-size:12px;color:var(--mute);margin:0 0 6px;font-weight:600}
.sub{color:var(--mute);margin:0}.small{font-size:13px}
.mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
code{background:var(--card);padding:1px 5px;border-radius:4px}
.verdict{margin:22px 0;padding:16px 18px;border-radius:10px;border:1px solid;border-left-width:6px}
.verdict .status{display:block;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.verdict b.head{font-size:18px;line-height:1.35;display:block}
.verdict p{margin:8px 0 0}
.ok{background:var(--tealbg);border-color:var(--teal)}.ok .status{color:var(--teal)}
.bad{background:var(--redbg);border-color:var(--red)}.bad .status{color:var(--red)}
.warn{background:var(--amberbg);border-color:var(--amber)}.warn .status{color:var(--amber)}
.fixed{background:var(--bluebg);border-color:var(--blue)}.fixed .status{color:var(--blue)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--card);border-radius:10px;padding:12px 14px;break-inside:avoid}
.tile .n{font-size:22px;font-weight:700;line-height:1.25;font-variant-numeric:tabular-nums}
.tile .l{color:var(--mute);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr{break-inside:avoid}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;white-space:nowrap;
border:1px solid transparent}
.critical{background:var(--redbg);color:var(--red);border-color:var(--red)}
.warning{background:var(--amberbg);color:var(--amber)}
.action{background:var(--bluebg);color:var(--blue)}
.proof{background:var(--redbg);color:var(--red)}
.info{background:var(--card);color:var(--mute)}
.healthy{background:var(--tealbg);color:var(--teal)}
.finding{border-left:4px solid var(--line);padding:8px 14px;margin:10px 0;background:var(--card);
border-radius:0 8px 8px 0;break-inside:avoid}
.finding.f-critical{border-color:var(--red)}.finding.f-warning{border-color:var(--amber)}
.finding p{margin:4px 0 0;color:var(--mute)}
.box{border:1px solid var(--line);border-radius:10px;padding:12px 18px;margin:14px 0;break-inside:avoid}
.box h2,.box h3{margin-top:4px;border:0;padding:0}
.box ul{margin:6px 0;padding-left:20px}.box li{margin:4px 0}
.estimate{border:1px dashed var(--amber);border-radius:8px;padding:10px 14px;margin:12px 0}
.label{font-size:11px;font-weight:700;letter-spacing:.06em;border:1px solid currentColor;border-radius:4px;
padding:0 5px;color:var(--amber)}
dl.gloss{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:6px 16px;margin:8px 0}
dl.gloss dt{font-weight:600}dl.gloss dd{margin:0;color:var(--mute)}
svg{display:block;max-width:100%;height:auto}
svg text{fill:var(--mute);font-size:11px;font-family:inherit}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--mute);font-size:13px;margin:6px 0 10px}
.legend span{white-space:nowrap}
.sym{font-weight:700}
.s-proof{color:var(--red)}.s-action{color:var(--blue)}.s-warning{color:var(--amber)}.s-critical{color:var(--red)}
.s-info{color:var(--mute)}
.sw{display:inline-block;width:14px;height:10px;vertical-align:-1px;border:1px solid var(--hatch);
background:repeating-linear-gradient(45deg,var(--hatch) 0 2px,transparent 2px 5px);opacity:.75}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute)}
footer{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);color:var(--mute);font-size:13px}
a{color:var(--teal)}
@media (max-width:480px){main{padding:20px 16px 36px}h1{font-size:23px}h2{font-size:17px}
.verdict b.head{font-size:16px}.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}.tile .n{font-size:19px}
dl.gloss{grid-template-columns:1fr}dl.gloss dd{margin-bottom:6px}th,td{padding:6px 5px}svg text{font-size:20px}}
@media print{
:root{--bg:#fff;--fg:#111;--mute:#444;--line:#bbb;--card:#f3f4f6;--teal:#0b5e57;--tealbg:#e3f1ef;
--red:#a3122a;--redbg:#fbeaec;--amber:#7a4600;--amberbg:#fdf1dc;--blue:#1d4f86;--bluebg:#e6eef9;--hatch:#a3122a}
html,body{background:#fff;color:#111}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-size:11pt}main,main.narrow{max-width:none;padding:0}
h1{font-size:20pt}h2{font-size:13pt;margin-top:18pt;break-after:avoid}h3{break-after:avoid}
.scroll{overflow:visible}table{font-size:9pt}
.verdict,.box,.tile,.finding,.estimate,svg,figure{break-inside:avoid}
a{color:inherit;text-decoration:none}
details>summary{display:none}
.noprint{display:none}
@page{margin:16mm 14mm}}
"""

NUM, NARROW = " class='num'", " class='narrow'"
LABEL = {"critical": "Critical", "warning": "Warning", "proof": "Proof", "action": "Repair",
         "info": "Info", "healthy": "Healthy"}

# Event markers on charts: a glyph per kind, so colour is never the only signal.
GLYPH = {"certificate": "◆", "repair": "▲", "alarm": "●", "channels": "■", "loss_spike": "✕",
         "nonfinite_loss": "!", "nonfinite_gradient": "!"}
GLYPH_WORD = {"certificate": "proven dead", "repair": "repair", "alarm": "gradient alarm",
              "channels": "stuck channels", "loss_spike": "loss spike", "nonfinite_loss": "NaN / Inf",
              "nonfinite_gradient": "NaN / Inf"}


def page(title: str, body: str, narrow: bool = False) -> str:
    """Wrap a report body in a complete, self-contained HTML document."""
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<meta name='generator' content='ormas-monitor {escape(__version__)}'>"
            f"<title>{escape(title)}</title><style>{CSS}</style></head><body>"
            f"<main{NARROW if narrow else ''}>{body}{footer()}</main></body></html>")


def footer() -> str:
    """The provenance footer shared by every report."""
    return (f"<footer>Generated on this machine by <b>ormas-monitor {escape(__version__)}</b> from the "
            f"run's own record. Nothing was uploaded anywhere. Dead-channel proofs follow Theorem 2 of "
            f"R. A. D. Raadh, <i>Self-Repair as a Bounded Disturbance</i> (DeepMath 2026). Built by "
            f"<a href='https://oxiedo.com'>Oxiedo</a>. Questions about this report: "
            f"<a href='mailto:raadh@oxiedo.com'>raadh@oxiedo.com</a></footer>")


def fmt(x: Any) -> str:
    """A number or value for a table cell; `None` (absent) renders as an en dash, 0 as 0."""
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if not math.isfinite(x):
            return escape(str(x))
        return f"{x:.3g}"
    if isinstance(x, int):
        return f"{x:,}"
    return escape(str(x))


def plural(n: int, word: str, many: str | None = None) -> str:
    """'1 layer', '2 layers'."""
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def duration(seconds: float | None) -> str:
    """Human wall-clock duration: '42 s', '3 min 05 s', '2 h 14 min'. None renders as 'unknown'."""
    if seconds is None:
        return "unknown"
    s = max(0.0, float(seconds))
    if s < 0.1:
        return "under 0.1 s"
    if s < 10:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        m, sec = divmod(int(round(s)), 60)
        return f"{m} min {sec:02d} s"
    h, rem = divmod(int(round(s)), 3600)
    if h < 48:
        return f"{h} h {rem // 60:02d} min"
    return f"{h / 24:.1f} days"


def share(part: float | None, whole: float | None) -> str:
    """A percentage for plain-language text: '92%', 'under 1%', or '' when it cannot be computed."""
    if part is None or not whole:
        return ""
    p = 100.0 * part / whole
    if p <= 0:
        return "0%"
    if p < 1:
        return "under 1%"
    if p > 99 and part < whole:
        return "over 99%"
    return f"{p:.0f}%"


def tag(severity: str | None, text: str | None = None) -> str:
    """A pill with a word on it (colour is never the only signal)."""
    sev = severity if severity in LABEL else "info"
    return f"<span class='tag {sev}'>{escape(text or LABEL.get(sev, str(severity)))}</span>"


def table(head: Sequence[str], rows: Iterable[Sequence[str]], num: Sequence[int] = ()) -> str:
    """A scrollable table. Cells are already-escaped HTML; columns in `num` align right."""
    th = "".join(f"<th{NUM if i in num else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{NUM if i in num else ''}>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


def verdict(cls: str, status: str, head: str, more: str = "") -> str:
    """The coloured verdict box: a status word, a one-line headline, optional detail HTML."""
    return (f"<div class='verdict {cls}' role='status'><span class='status'>{escape(status)}</span>"
            f"<b class='head'>{escape(head)}</b>{more}</div>")


def _hatch_defs(pid: str) -> str:
    return (f"<defs><pattern id='{pid}' width='6' height='6' patternUnits='userSpaceOnUse' "
            f"patternTransform='rotate(45)'><rect width='6' height='6' fill='var(--hatch)' opacity='.08'/>"
            f"<line x1='0' y1='0' x2='0' y2='6' stroke='var(--hatch)' stroke-width='1.5' opacity='.45'/></pattern></defs>")


def loss_chart(points: list[tuple[int, float]], marks: list[tuple[int, str, str, str]],
               spans: list[tuple[int, int]], max_points: int = 1500) -> str:
    """Loss over steps with stuck intervals hatched and events marked by glyph.

    `marks` are (step, event type, severity, description); `spans` are [start, end) steps.
    Returns '' when there are fewer than two finite loss values.
    """
    pts = [(s, v) for s, v in points if v is not None and math.isfinite(v)]
    if len(pts) < 2:
        return ""
    if len(pts) > max_points:
        stride = math.ceil(len(pts) / max_points)
        pts = pts[::stride] + ([pts[-1]] if (len(pts) - 1) % stride else [])
    W, H, L, R, T, B = 800, 260, 54, 14, 46, 30
    xs = [s for s, _ in pts]
    lo, hi = min(v for _, v in pts), max(v for _, v in pts)
    hi = hi if hi > lo else lo + 1
    x0, x1 = min(xs), max(xs)
    x1 = x1 if x1 > x0 else x0 + 1

    def X(s: float) -> float:
        return L + (min(max(s, x0), x1) - x0) / (x1 - x0) * (W - L - R)

    def Y(v: float) -> float:
        return T + (hi - v) / (hi - lo) * (H - T - B)
    colour = {"critical": "var(--red)", "warning": "var(--amber)", "action": "var(--blue)",
              "proof": "var(--red)", "info": "var(--mute)"}
    band = "".join(f"<rect x='{X(a):.1f}' y='{T}' width='{max(X(b) - X(a), 1.5):.1f}' height='{H - T - B}' "
                   f"fill='url(#lh)'><title>steps {a:,}–{b:,}: a layer had stuck channels</title></rect>"
                   for a, b in spans if b > x0 and a < x1)
    stacked: dict[int, list[str]] = {}
    ms = []
    for s, ev, sev, text in marks:
        if not (x0 <= s <= x1) or ev not in GLYPH:
            continue
        kinds = stacked.setdefault(s, [])
        if ev in kinds:
            continue
        kinds.append(ev)
        c = colour.get(sev, "var(--mute)")
        ms.append(f"<g><title>step {s:,}: {escape(text)}</title>"
                  f"<line x1='{X(s):.1f}' x2='{X(s):.1f}' y1='{T}' y2='{H - B}' stroke='{c}' stroke-width='1' "
                  f"stroke-dasharray='3 3' opacity='.8'/>"
                  f"<text x='{X(s):.1f}' y='{T - 6 - 13 * min(len(kinds) - 1, 2)}' text-anchor='middle' "
                  f"style='fill:{c};font-size:12px'>{GLYPH[ev]}</text></g>")
    line = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts)
    return (f"<figure style='margin:0'><svg viewBox='0 0 {W} {H}' width='100%' role='img' "
            f"aria-label='Training loss over steps; hatched bands mark time when a layer had stuck channels'>"
            f"{_hatch_defs('lh')}{band}"
            f"<line x1='{L}' x2='{W - R}' y1='{H - B}' y2='{H - B}' stroke='var(--line)'/>"
            f"<line x1='{L}' x2='{L}' y1='{T}' y2='{H - B}' stroke='var(--line)'/>"
            f"<text x='{L - 6}' y='{T + 4}' text-anchor='end'>{hi:.3g}</text>"
            f"<text x='{L - 6}' y='{H - B}' text-anchor='end'>{lo:.3g}</text>"
            f"<text x='{L}' y='{H - 10}'>step {x0:,}</text><text x='{W - R}' y='{H - 10}' text-anchor='end'>step {x1:,}</text>"
            f"{''.join(ms)}<polyline points='{line}' fill='none' stroke='var(--teal)' stroke-width='2' "
            f"stroke-linejoin='round'/></svg></figure>")


def legend(kinds: Iterable[str], with_band: bool) -> str:
    """Legend for `loss_chart` / `timeline`: glyph + word for each event kind shown."""
    sev = {"certificate": "proof", "repair": "action", "alarm": "warning", "channels": "warning",
           "loss_spike": "warning", "nonfinite_loss": "critical", "nonfinite_gradient": "critical"}
    words: dict[str, str] = {}
    for k in kinds:
        if k in GLYPH and GLYPH_WORD[k] not in words.values():
            words[k] = GLYPH_WORD[k]
    items = [f"<span><span class='sym s-{sev[k]}'>{GLYPH[k]}</span> {escape(w)}</span>" for k, w in words.items()]
    if with_band:
        items.insert(0, "<span><span class='sw'></span> a layer had stuck channels</span>")
    return f"<div class='legend'>{''.join(items)}</div>" if items else ""


def timeline(total: int, spans: list[tuple[int, int]], marks: list[tuple[int, str, str]],
             label: str) -> str:
    """A one-row bar for the whole run: hatched where a layer was stuck, glyphs for repairs etc.

    `marks` are (step, event type, description). `total` is the number of training steps.
    """
    if total <= 0:
        return ""
    W, H, L, R = 700, 64, 4, 4
    y, h = 20, 18

    def X(s: float) -> float:
        return L + max(0.0, min(s, total)) / total * (W - L - R)
    bars = "".join(f"<rect x='{X(a):.1f}' y='{y}' width='{max(X(b) - X(a), 2):.1f}' height='{h}' fill='url(#th)' "
                   f"stroke='var(--hatch)' stroke-width='1'><title>steps {a:,}–{b:,}</title></rect>"
                   for a, b in spans)
    done: set[tuple[int, str]] = set()
    ms = []
    for s, ev, text in marks:
        if ev not in GLYPH or (s, ev) in done:
            continue
        done.add((s, ev))
        cls = {"repair": "var(--blue)", "certificate": "var(--red)"}.get(ev, "var(--amber)")
        ms.append(f"<text x='{X(s):.1f}' y='{y - 5}' text-anchor='middle' style='fill:{cls};font-size:12px'>"
                  f"{GLYPH[ev]}<title>step {s:,}: {escape(text)}</title></text>")
    return (f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='{escape(label)}'>{_hatch_defs('th')}"
            f"<rect x='{L}' y='{y}' width='{W - L - R}' height='{h}' rx='3' fill='var(--card)' stroke='var(--line)'/>"
            f"{bars}{''.join(ms)}"
            f"<text x='{L}' y='{y + h + 16}'>step 0</text>"
            f"<text x='{W - R}' y='{y + h + 16}' text-anchor='end'>step {total:,} (end)</text></svg>")
