"""`ormas-monitor demo`: the CIFAR-10 dead-layer experiment, watch-only vs repair."""
from __future__ import annotations

import argparse


def register(sub: argparse._SubParsersAction) -> None:
    """Add the `demo` subcommand."""
    d = sub.add_parser("demo", help="CIFAR-10 demo: two layers die; watch-only vs repair (needs torchvision)",
                       description="Train the paper's CIFAR-10 CNN, zero two of its conv layers, then train "
                                   "on twice from the same state: once with the monitor only watching, "
                                   "once with it repairing. Writes both runs' records and reports.")
    d.add_argument("--out", default="ormas_demo",
                   help="directory for the two runs (OUT/watch, OUT/repair); re-running replaces them "
                        "(default: ormas_demo)")
    d.add_argument("--data", default="data", help="where CIFAR-10 is, or gets downloaded to")
    d.add_argument("--full", action="store_true", help="all 50,000 images, 8 epochs before and 12 after")
    d.set_defaults(func=run)


def run(a: argparse.Namespace) -> int:
    """Run the quick demo, or the full one with --full."""
    from ..demo import run as demo
    if a.full:
        demo(a.out, a.data, before=8, after=12, train_size=None)
    else:
        demo(a.out, a.data)
    return 0
