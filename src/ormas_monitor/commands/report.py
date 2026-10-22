"""`ormas-monitor report`: rebuild the reports of a monitored run (engineer report, executive
summary in HTML and Markdown). A dollar ESTIMATE appears only when a GPU-hour price is given."""
from __future__ import annotations

import argparse


def _positive_float(text: str) -> float:
    v = float(text)
    if not v > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return v


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return v


def register(sub) -> None:
    r = sub.add_parser("report", help="rebuild the reports of a monitored run")
    r.add_argument("run", help="run directory (holds diary.jsonl)")
    r.add_argument("--gpu-hour-cost", type=_positive_float, default=None, metavar="PRICE",
                   help="price of one GPU-hour; adds a dollar ESTIMATE of time spent stuck to the summary")
    r.add_argument("--gpus", type=_positive_int, default=None, metavar="N",
                   help="number of GPUs the run used (default 1 when a price is given)")
    r.add_argument("--key", default=None, metavar="PUBLIC.pem",
                   help="signer's public key, to verify a signed record (obtain it from the signer)")
    r.set_defaults(func=run)


def run(a) -> int:
    from ..report import write_all
    for path in write_all(a.run, gpu_hour_cost=a.gpu_hour_cost, gpus=a.gpus, public_key=a.key).values():
        print(path)
    return 0
