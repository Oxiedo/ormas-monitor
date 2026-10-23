"""`ormas-monitor check`: health-check a saved model file or Hugging Face folder (weights only)."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

FAIL_ON = {"critical": ("critical",), "warning": ("critical", "warning"), "never": ()}

EPILOG = """\
exit codes:
  0  no finding at or above --fail-on (default: critical)
  1  at least one finding at or above --fail-on
  2  the file could not be read, or was refused (pickled objects without --trust)

examples:
  ormas-monitor check model.safetensors
  ormas-monitor check ./my-hf-model --arch hf
  ormas-monitor check after.pt --against before.pt --fail-on warning --quiet
"""


def register(sub) -> None:
    c = sub.add_parser("check", help="health-check a saved model file or folder (weights only)",
                       description="Weights-only health check of a checkpoint: .pt/.pth/.bin/.ckpt, "
                                   ".safetensors, or a Hugging Face model folder (sharded or not). "
                                   "Tensors are read one at a time; nothing is uploaded.",
                       epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    c.add_argument("file", help="checkpoint file, *.index.json, or Hugging Face model folder")
    c.add_argument("--against", help="an earlier checkpoint (file or folder) to compare with")
    c.add_argument("--out", help="HTML report path (default: <file>.health.html)")
    c.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    c.add_argument("--arch", help="exact wiring from a known model: torchvision:<name> (e.g. "
                                  "torchvision:resnet50), or hf to build the model from config.json "
                                  "(random init, no download). Builds the model without weights to "
                                  "trace it; importing torchvision takes about a second, longer on "
                                  "the first run after installing it")
    c.add_argument("--trust", action="store_true",
                   help="allow loading files that contain Python objects (can run code)")
    c.add_argument("--fail-on", choices=tuple(FAIL_ON), default="critical",
                   help="lowest severity that makes the exit code 1 (default: critical)")
    c.add_argument("--quiet", action="store_true", help="print nothing except errors and --json output")
    c.add_argument("--all", action="store_true", dest="show_all",
                   help="list every info-level finding too (by default they are counted, and listed in the report)")
    c.set_defaults(func=run)


def _finite(obj: object) -> object:
    """JSON has no Infinity/NaN: write them as strings."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def run(a) -> int:
    from ..checkpoint import CheckpointError
    from ..checkpoint import run as check
    from ..report import write_check_report
    t0 = time.perf_counter()
    try:
        result = check(a.file, a.against, a.trust, a.arch)
    except CheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    seconds = time.perf_counter() - t0
    out = Path(a.out or f"{str(a.file).rstrip('/')}.health.html")
    write_check_report(result, out)
    failing = [x for x in result["findings"] if x["severity"] in FAIL_ON[a.fail_on]]
    if a.json:
        print(json.dumps(_finite(result), indent=1, default=str))
    if not a.quiet:
        say = (lambda s: print(s, file=sys.stderr)) if a.json else print
        f = result["findings"]
        dtypes = ", ".join(sorted(result["dtypes"]))
        w = result["wiring"]
        wiring = (f"traced from {w['source']}" if w["traced"] else "name-based heuristic") if w["batchnorms"] \
            else "no BatchNorm layers"
        say(f"{Path(a.file).name}: {result['parameters']:,} parameters in {result['tensors']:,} tensors "
            f"({result['format']}, {dtypes}); wiring: {wiring}; {_count(len(f), 'finding')} in {seconds:.1f} s")
        info = [x for x in f if x["severity"] == "info"]
        brief = len(info) > MAX_INFO_LINES and not getattr(a, "show_all", False)
        for x in f:
            if not (brief and x["severity"] == "info"):
                say(f"  [{x['severity']}] {x['title']}")
        if brief:
            say(f"  plus {_count(len(info), 'info finding')} (no action needed; listed in the report, or use --all)")
        for n in result["notes"]:
            say(f"  note: {n}")
        say(f"report: {out}")
        say(exit_reason(result["findings"], a.fail_on))
    return 1 if failing else 0


MAX_INFO_LINES = 5       # above this many info findings, the terminal shows a count instead


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def exit_reason(findings: list[dict], fail_on: str) -> str:
    """The last line of the summary: which exit code, and why (e.g. warnings that did not count)."""
    n = {s: sum(1 for f in findings if f["severity"] == s) for s in ("critical", "warning")}
    if fail_on == "never":
        found = [_count(n[s], s) for s in n if n[s]]
        return f"exit 0: --fail-on never ({', '.join(found) if found else 'no critical or warning findings'})"
    failing = sum(n[s] for s in FAIL_ON[fail_on])
    if failing:
        parts = ", ".join(_count(n[s], s) for s in FAIL_ON[fail_on] if n[s])
        return f"exit 1: {parts} at or above --fail-on {fail_on}"
    if fail_on == "critical" and n["warning"]:
        return (f"exit 0: no critical findings ({_count(n['warning'], 'warning')}; "
                "use --fail-on warning to fail on them)")
    return "exit 0: no critical findings" if fail_on == "critical" else "exit 0: no critical or warning findings"
