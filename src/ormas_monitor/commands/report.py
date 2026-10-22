"""`ormas-monitor report`: rebuild the reports of a monitored run."""
from __future__ import annotations


def register(sub) -> None:
    r = sub.add_parser("report", help="rebuild the reports of a monitored run")
    r.add_argument("run")
    r.set_defaults(func=run)


def run(a) -> int:
    from ..report import write_all
    for path in write_all(a.run).values():
        print(path)
    return 0
