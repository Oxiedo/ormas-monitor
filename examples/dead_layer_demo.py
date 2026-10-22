"""Full-size dead-layer demo: all 50,000 CIFAR-10 images, 8 epochs before the lesion, 12 after.

Same as `ormas-monitor demo --full`. Pass --data to point at an existing CIFAR-10 folder.
"""
import argparse

from ormas_monitor.demo import run

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="demo_runs")
ap.add_argument("--before", type=int, default=8)
ap.add_argument("--after", type=int, default=12)
a = ap.parse_args()
run(a.out, a.data, before=a.before, after=a.after, train_size=None)
