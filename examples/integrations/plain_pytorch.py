"""Plain PyTorch: break a layer, let ormas-monitor certify and repair it, log events via `logging`.

    python examples/integrations/plain_pytorch.py [--out ormas_runs/plain]
"""
from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.integrations import LoggingSink


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        self.head = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.head(x.mean((2, 3)))


def break_layer(model: SmallCNN) -> None:
    """Zero conv2 and push its BatchNorm shift below 0: every channel's ReLU closes for good."""
    with torch.no_grad():
        model.conv2.weight.zero_()
        model.conv2.bias.zero_()
        model.bn2.bias.fill_(-0.1)


def synthetic(n: int = 512, seed: int = 0) -> torch.utils.data.TensorDataset:
    """Images whose class is the brightest quadrant: learnable in a few hundred steps."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 16, 16, generator=g)
    y = torch.randint(0, 4, (n,), generator=g)
    for i, c in enumerate(y.tolist()):
        r, col = divmod(c, 2)
        x[i, :, r * 8:(r + 1) * 8, col * 8:(col + 1) * 8] += 1.5
    return torch.utils.data.TensorDataset(x, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ormas_runs/plain")
    ap.add_argument("--repair", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    torch.manual_seed(0)
    model = SmallCNN()
    break_layer(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loader = torch.utils.data.DataLoader(synthetic(), batch_size=32, shuffle=True)

    with Monitor(model, opt, out=a.out, repair=a.repair, check_every=10, verbose=False,
                 sinks=[LoggingSink(min_severity="warning")]) as mon:
        for epoch in range(4):
            correct = 0
            for x, y in loader:
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                mon.step(loss)            # after backward, before optimizer.step()
                opt.step()
                correct += int((logits.argmax(1) == y).sum())
            print(f"epoch {epoch}: loss {loss.item():.3f}, train accuracy {correct / 512:.1%}")
    print(f"steps {mon.step_count}, counts {mon.counts}")
    print(f"record: {mon.out / 'diary.jsonl'}")


if __name__ == "__main__":
    main()
