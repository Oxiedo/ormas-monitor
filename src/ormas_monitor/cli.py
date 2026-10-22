"""Command line: `ormas-monitor check | verify | report`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ormas-monitor", description="Training health checks that run on your machine.")
    ap.add_argument("--version", action="version", version=f"ormas-monitor {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="health-check a saved model file (weights only)")
    c.add_argument("file")
    c.add_argument("--against", help="an earlier checkpoint to compare with")
    c.add_argument("--out", help="HTML report path (default: <file>.health.html)")
    c.add_argument("--json", action="store_true", help="print the result as JSON")
    c.add_argument("--arch", help="exact wiring from a known model, e.g. torchvision:resnet50")
    c.add_argument("--trust", action="store_true",
                   help="allow loading files that contain Python objects (can run code)")

    v = sub.add_parser("verify", help="check that a run's record has not been edited")
    v.add_argument("run", help="run directory or diary.jsonl")

    r = sub.add_parser("report", help="rebuild the HTML report of a monitored run")
    r.add_argument("run")

    d = sub.add_parser("demo", help="CIFAR-10 demo: two layers die; watch-only vs repair (needs torchvision)")
    d.add_argument("--out", default="ormas_demo")
    d.add_argument("--data", default="data", help="where CIFAR-10 is, or gets downloaded to")
    d.add_argument("--full", action="store_true", help="all 50,000 images, 8 epochs before and 12 after")

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        from .demo import run as demo
        if a.full:
            demo(a.out, a.data, before=8, after=12, train_size=None)
        else:
            demo(a.out, a.data)
        return 0
    if a.cmd == "check":
        from .checkpoint import CheckpointError, run
        from .report import write_check_report
        try:
            result = run(a.file, a.against, a.trust, a.arch)
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

    path = Path(a.run)
    if a.cmd == "verify":
        from .diary import verify
        diary = path / "diary.jsonl" if path.is_dir() else path
        ok, n, msg = verify(diary)
        print(("intact: " if ok else "BROKEN: ") + msg)
        return 0 if ok else 1

    from .report import write_run_report
    print(write_run_report(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
