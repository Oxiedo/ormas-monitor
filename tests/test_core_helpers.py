"""Shared helpers for the core-engine tests (models, a training loop, diary schema checks)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from ormas_monitor.schema import RECORD_VERSION, validate_event


def small_cnn() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


def lesion(conv: nn.Module, bn: nn.Module, opt: torch.optim.Optimizer | None = None) -> None:
    """Zero a layer, like the paper's lesion; the paper's harness also clears optimizer state."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        bn.bias.fill_(-0.1)
    if opt is not None:
        opt.state.clear()


def batches(n: int, seed: int = 1, channels: int = 3):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        yield torch.randn(16, channels, 8, 8, generator=g), torch.randint(0, 4, (16,), generator=g)


def train(model, mon, opt, steps: int, after=None, seed: int = 1) -> None:
    for i, (x, y) in enumerate(batches(steps, seed)):
        if after and i == after[0]:
            after[1]()
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()


def events(run: Path) -> list[dict]:
    """Every diary line of a run, after asserting each one matches the schema."""
    lines = [json.loads(l) for l in (Path(run) / "diary.jsonl").read_text().splitlines() if l.strip()]
    assert_valid(lines)
    return lines


def assert_valid(lines: list[dict]) -> None:
    assert lines and lines[0]["event"] == "run_start"
    assert lines[0]["record_version"] == RECORD_VERSION
    problems = {i: validate_event(l) for i, l in enumerate(lines)}
    problems = {i: p for i, p in problems.items() if p}
    assert not problems, problems
