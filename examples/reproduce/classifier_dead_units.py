"""Reproduce: the paper's StandardCNN classifier carries hidden units that never fire.

Trains the demo's StandardCNN on CIFAR-10 exactly as `ormas-monitor demo --full` does before the
lesion (8 epochs, AdamW 1e-3, weight decay 1e-2, batch 128, seed 0), then counts the 256 hidden
units of `classifier.0` whose output is never positive on any test or training image.

Run:  python examples/reproduce/classifier_dead_units.py --data DIR   (needs torchvision)
"""
from __future__ import annotations

import argparse

import torch

from ormas_monitor.demo import StandardCNN, device, epoch, load_cifar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    dev = device()
    xtr, ytr, xte, yte = load_cifar(a.data, dev)
    torch.manual_seed(0)
    model = StandardCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(a.epochs):
        epoch(model, opt, xtr, ytr, gen)
    peak: dict[str, torch.Tensor] = {}

    def hook(module, inputs, out):
        p = out.amax(0)
        peak["p"] = p if "p" not in peak else torch.maximum(peak["p"], p)
    model.classifier[0].register_forward_hook(hook)
    model.eval()
    for split, x in (("test", xte), ("train", xtr)):
        peak.clear()
        with torch.no_grad():
            for i in range(0, len(x), 1000):
                model(x[i:i + 1000])
        print(f"{split}: {int((peak['p'] <= 0).sum())} of 256 hidden units never positive "
              f"on {len(x):,} images")


if __name__ == "__main__":
    main()
