"""Measure ormas-monitor's slowdown: the same epochs with and without it, alternating.

Run:  python examples/overhead.py --data DIR [--epochs 3] [--check-every 50]
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time

import torch

from ormas_monitor.demo import StandardCNN, epoch, load_cifar
from ormas_monitor import Monitor


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--check-every", type=int, default=50)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    xtr, ytr, _, _ = load_cifar(a.data, dev)
    times = {"plain": [], "monitored": []}
    for i in range(a.epochs):
        for arm in ("plain", "monitored"):
            torch.manual_seed(i)
            model = StandardCNN().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            gen = torch.Generator().manual_seed(i)
            with tempfile.TemporaryDirectory() as d:
                mon = Monitor(model, opt, out=d, check_every=a.check_every, verbose=False) \
                    if arm == "monitored" else None
                sync(dev)
                t = time.perf_counter()
                epoch(model, opt, xtr, ytr, gen, mon)
                sync(dev)
                times[arm].append(time.perf_counter() - t)
                if mon:
                    mon.close(report=False)
    p, m = statistics.median(times["plain"][1:] or times["plain"]), \
        statistics.median(times["monitored"][1:] or times["monitored"])
    print(f"device {dev}; median epoch: plain {p:.2f}s, monitored {m:.2f}s; "
          f"overhead {100 * (m / p - 1):+.1f}% (check every {a.check_every} steps)")


if __name__ == "__main__":
    main()
