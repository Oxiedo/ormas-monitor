"""`ormas-monitor demo`: the experiment behind the DeepMath 2026 poster, at small scale.

A CIFAR-10 CNN (the paper's baseline: three Conv-BN-ReLU blocks and an MLP head) trains
normally. Then two of its three conv layers are zeroed and the optimizer state cleared, the
paper's "dead" lesion. From that identical state it trains on twice:

  watch   the monitor only watches: it names the dead layers and proves they cannot recover
  repair  the monitor also repairs them

Needs torchvision; downloads CIFAR-10 (~170 MB) once.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .monitor import Monitor


class StandardCNN(nn.Module):
    """The paper's CIFAR-10 baseline: three Conv-BN-ReLU-MaxPool blocks and a two-layer MLP head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.conv1, self.conv2, self.conv3 = block(3, 32), block(32, 64), block(64, 128)
        self.classifier = nn.Sequential(nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv3(self.conv2(self.conv1(x))).flatten(1))


def device() -> torch.device:
    """CUDA if present, else Apple MPS, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def load_cifar(root: str | Path, dev: torch.device,
               train_size: int | None = None) -> list[torch.Tensor]:
    """[x_train, y_train, x_test, y_test] on `dev`, normalised; downloads CIFAR-10 to `root` once."""
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise SystemExit("the demo needs torchvision: pip install torchvision") from e
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    out = []
    for train in (True, False):
        ds = CIFAR10(str(root), train=train, download=True)
        x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div(255).sub(mean).div(std)
        y = torch.tensor(ds.targets)
        if train and train_size:
            x, y = x[:train_size], y[:train_size]
        out += [x.to(dev), y.to(dev)]
    return out


def epoch(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor,
          gen: torch.Generator, mon: Monitor | None = None, bs: int = 128) -> None:
    """One shuffled training epoch; `mon.step(loss)` after every backward when a monitor is given."""
    model.train()
    perm = torch.randperm(len(x), generator=gen).to(x.device)
    for i in range(0, len(x), bs):
        idx = perm[i:i + bs]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        if mon:
            mon.step(loss)
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Test accuracy in percent (eval mode, batches of 1,000)."""
    model.eval()
    hits = sum((model(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(x), 1000))
    return 100 * hits / len(x)


def run(out: str | Path = "ormas_demo", data: str | Path = "data", before: int = 4, after: int = 4,
        train_size: int | None = 10_000, seed: int = 0) -> dict:
    """Train, lesion conv2 and conv3, then train on watch-only and with repair from the same state.

    Returns {"healthy": accuracy, "watch": {...}, "repair": {...}}, each arm with its accuracy
    curve, report path and the monitor's counts. Defaults are the quick demo (10,000 images,
    4 epochs before and 4 after); `ormas-monitor demo --full` uses all images, 8 and 12.
    """
    dev = device()
    xtr, ytr, xte, yte = load_cifar(data, dev, train_size)
    print(f"device {dev}; {len(xtr):,} training images, {len(xte):,} test images")
    torch.manual_seed(seed)
    model = StandardCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    gen = torch.Generator().manual_seed(seed)
    t = time.time()
    for e in range(before):
        epoch(model, opt, xtr, ytr, gen)
        print(f"epoch {e + 1}/{before}: {accuracy(model, xte, yte):.2f}%", flush=True)
    healthy = accuracy(model, xte, yte)
    snap = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    print(f"healthy network: {healthy:.2f}% after {time.time() - t:.0f}s\n")

    results: dict = {"healthy": healthy}
    for arm, repair in (("watch", False), ("repair", True)):
        torch.manual_seed(seed + 1)
        m = StandardCNN().to(dev)
        m.load_state_dict(snap[0])
        o = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        o.load_state_dict(snap[1])
        g = torch.Generator().manual_seed(seed + 1)
        print(f"--- {arm}: the monitor {'watches and repairs' if repair else 'only watches'}")
        mon = Monitor(m, o, out=Path(out) / arm, repair=repair, name=f"CIFAR-10 dead-layer demo ({arm})",
                      overwrite=True)                 # the demo's own directories: re-runnable
        epoch(m, o, xtr, ytr, g, mon)                 # one normal epoch: every layer gets a baseline
        with torch.no_grad():                         # the lesion: conv2 and conv3 zeroed
            for blk in (m.conv2, m.conv3):
                blk[0].weight.zero_()
                blk[0].bias.zero_()
        o.state.clear()                               # as in the paper's experiment harness
        print(f"    step {mon.step_count:,}: conv2 and conv3 zeroed")
        curve = []
        for e in range(after):
            epoch(m, o, xtr, ytr, g, mon)
            curve.append(accuracy(m, xte, yte))
            print(f"    epoch {e + 1}/{after} after the lesion: {curve[-1]:.2f}%", flush=True)
        results[arm] = {"curve": curve, "report": str(mon.close()), **mon.counts}

    print(f"\nhealthy before the lesion: {healthy:.2f}%")
    for arm in ("watch", "repair"):
        r = results[arm]
        print(f"{arm:>7}: {r['curve'][-1]:.2f}% after {after} epochs, "
              f"{r['channels_repaired']} channels repaired. Report: {r['report']}")
    return results
