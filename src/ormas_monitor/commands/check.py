"""`ormas-monitor check`: health-check a saved model file (weights only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def register(sub) -> None:
    c = sub.add_parser("check", help="health-check a saved model file (weights only)")
    c.add_argument("file")
    c.add_argument("--against", help="an earlier checkpoint to compare with")
    c.add_argument("--out", help="HTML report path (default: <file>.health.html)")
    c.add_argument("--json", action="store_true", help="print the result as JSON")
    c.add_argument("--arch", help="exact wiring from a known model, e.g. torchvision:resnet50")
    c.add_argument("--trust", action="store_true",
                   help="allow loading files that contain Python objects (can run code)")
    c.set_defaults(func=run)


def run(a) -> int:
    from ..checkpoint import CheckpointError, run as check
    from ..report import write_check_report
    try:
        result = check(a.file, a.against, a.trust, a.arch)
    except CheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.out or f"{a.file}.health.html")
    write_check_report(result, out)
    if a.json:
        print(json.dumps(result, indent=1, default=str))
    else:
        f = result["findings"]
        print(f"{Path(a.file).name}: {result['parameters']:,} parameters, {len(f)} findings")
        for x in f:
            print(f"  [{x['severity']}] {x['title']}")
        print(f"report: {out}")
    return 1 if any(x["severity"] == "critical" for x in result["findings"]) else 0
