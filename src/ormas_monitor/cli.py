"""Command line entry point.

Each subcommand lives in its own module under `ormas_monitor.commands` and exposes
`register(subparsers)`, which adds its parser and sets `func(args) -> int` as the default.
Adding a command means adding a module and listing it in COMMANDS, nothing else.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__

COMMANDS = ("check", "verify", "report", "demo")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ormas-monitor",
                                 description="Training health checks that run on your machine.")
    ap.add_argument("--version", action="version", version=f"ormas-monitor {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        importlib.import_module(f"ormas_monitor.commands.{name}").register(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
