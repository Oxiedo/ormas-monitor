"""`ormas-monitor diff`: what changed between an approved run and a proposed one."""
from __future__ import annotations

import json
import sys


def register(sub) -> None:
    d = sub.add_parser("diff", help="compare two run records: what changed since the run you approved",
                       description="Compare run B against run A using their diaries and metrics only.")
    d.add_argument("run_a", metavar="RUN_A", help="the approved (baseline) run directory or diary")
    d.add_argument("run_b", metavar="RUN_B", help="the proposed run directory or diary")
    d.add_argument("--out", metavar="diff.html", help="write an HTML diff report")
    d.add_argument("--json", action="store_true", help="print the diff result as JSON")
    d.set_defaults(func=run)


def _text(r: dict) -> str:
    lines = [f"A: {r['a']['path']}  ({r['a']['name']}, {r['a']['steps']} steps, "
             f"{'intact' if r['a']['intact'] else 'FAILED VERIFICATION'})",
             f"B: {r['b']['path']}  ({r['b']['name']}, {r['b']['steps']} steps, "
             f"{'intact' if r['b']['intact'] else 'FAILED VERIFICATION'})",
             "", r["verdict"], ""]
    for c in r["config_changes"] + r["environment_changes"]:
        lines.append(f"  changed  {c['key']}: {c['a']!r} -> {c['b']!r}")
    changed = [row for row in r["layers"] if row["changed"]]
    if changed:
        lines.append("")
        lines.append(f"  {'layer':<28}{'alarms':>10}{'peak stuck':>14}{'certified':>13}{'repairs':>11}")
        for row in changed:
            def pair(k: str, row: dict = row) -> str:
                fa, fb = row[f"{k}_a"], row[f"{k}_b"]
                return f"{'-' if fa is None else fa}->{'-' if fb is None else fb}"
            lines.append(f"  {row['layer']:<28}{pair('alarms'):>10}{pair('peak_stuck'):>14}"
                         f"{pair('certified'):>13}{pair('repairs'):>11}")
    return "\n".join(lines)


def run(a) -> int:
    from ..record.diff import diff_runs

    try:
        result = diff_runs(a.run_a, a.run_b)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(_text(result))
    sys.stdout.flush()
    if a.out:
        from .. import report
        writer = getattr(report, "write_diff_report", None)
        if writer is None:
            print("note: this install has no HTML diff report; the text above is the full result",
                  file=sys.stderr)
        else:
            print(f"diff report written to {writer(result, a.out)}", file=sys.stderr)
    return 0
