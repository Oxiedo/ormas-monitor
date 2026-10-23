"""The monitor never breaks training (repros r6, r6c, r7, r12 and the fresh-user blocker): empty
batches, sparse gradients, split devices, unreduced losses, host reads per check, step() before
backward(), an existing record, failures while closing, optimizers it cannot read."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor.cli import main as cli
from ormas_monitor.monitor import Monitor
from test_core_helpers import events, small_cnn, train


def loop(model, opt, x, out, steps=3, check_every=1, lossf=lambda y: y.float().pow(2).mean(), **kw):
    mon = Monitor(model, opt, out=out, check_every=check_every, verbose=False, **kw)
    for _ in range(steps):
        loss = lossf(model(x()))
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    return mon


def test_empty_batch_does_not_crash(tmp_path):
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sizes = iter([4, 0, 4, 0])
    loop(m, opt, lambda: torch.randn(next(sizes), 4), tmp_path / "run", steps=4,
         lossf=lambda y: y.sum() + 0 * y.float().pow(2).mean().nan_to_num())
    assert events(tmp_path / "run")[-1]["event"] == "run_end"


def test_sparse_embedding_gradients(tmp_path):
    m = nn.Sequential(nn.Embedding(10, 4, sparse=True), nn.Linear(4, 1))
    loop(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda: torch.randint(0, 10, (8,)), tmp_path / "run")
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert all("0" in r["grad"] and r["grad"]["0"] > 0 for r in rows)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device (MPS)")
def test_model_split_across_devices(tmp_path):
    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 1).to("mps")

        def forward(self, x):
            return self.b(self.a(x).to("mps"))
    m = Split()
    loop(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda: torch.randn(8, 4), tmp_path / "run")
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert set(rows[0]["grad"]) == {"a", "b"}


def test_unreduced_loss_is_averaged_once(tmp_path):
    m = small_cnn()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    mon = Monitor(m, opt, out=tmp_path / "run", check_every=1, verbose=False)
    for _ in range(3):
        per_sample = m(torch.randn(4, 3, 8, 8)).pow(2).mean(1)
        opt.zero_grad()
        per_sample.mean().backward()
        mon.step(per_sample)
        opt.step()
    mon.close(report=False)
    ev = events(tmp_path / "run")
    red = [e for e in ev if e["event"] == "loss_reduced"]
    assert len(red) == 1 and red[0]["elements"] == 4
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    assert all(isinstance(r["loss"], float) for r in rows)


def test_constant_host_reads_per_check(tmp_path, monkeypatch):
    """Device-to-host reads per sampled check do not grow with the number of layers."""
    def reads(model, out):
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        mon = Monitor(model, opt, out=out, check_every=1, verbose=False)
        counted = []
        for _ in range(3):
            loss = model(torch.randn(4, 3, 8, 8)).logsumexp(1).mean()
            opt.zero_grad()
            loss.backward()
            calls = {"n": 0}
            for name in ("cpu", "tolist", "item", "__float__"):
                orig = getattr(torch.Tensor, name)

                def wrapped(self, *a, _orig=orig, _calls=calls, **kw):
                    _calls["n"] += 1
                    return _orig(self, *a, **kw)
                monkeypatch.setattr(torch.Tensor, name, wrapped)
            mon.step(loss)
            monkeypatch.undo()
            counted.append(calls["n"])
            opt.step()
        mon.close(report=False)
        return counted[1:]                 # after the first check (structure detection)

    def deep(k):
        torch.manual_seed(0)
        layers = [nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()]
        for _ in range(k):
            layers += [nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()]
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
    shallow, wide = reads(deep(1), tmp_path / "a"), reads(deep(12), tmp_path / "b")
    assert shallow == wide and max(wide) <= 4, (shallow, wide)


def test_step_before_backward_warns_no_gradients(tmp_path):
    m = small_cnn()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    with torch.no_grad():
        m[3].weight.zero_()
        m[3].bias.zero_()
        m[4].bias.fill_(-1.0)
    mon = Monitor(m, opt, out=tmp_path / "run", check_every=2, verbose=False)
    for _ in range(6):
        loss = nn.functional.cross_entropy(m(torch.randn(8, 3, 8, 8)), torch.randint(0, 4, (8,)))
        mon.step(loss)                     # the mistake: before backward
        loss.backward()
        opt.step()
        opt.zero_grad()
    mon.close(report=False)
    ev = events(tmp_path / "run")
    warn = [e for e in ev if e["event"] == "no_gradients"]
    assert len(warn) == 1 and warn[0]["severity"] == "warning" and "after loss.backward()" in warn[0]["message"]
    assert mon.counts["checks_without_gradients"] == 3 == ev[-1]["counts"]["checks_without_gradients"]
    assert not any(e["event"] == "certificate" for e in ev)     # no gradient to vouch for: no proof


def test_existing_record_is_not_overwritten(tmp_path):
    def once(**kw):
        m = small_cnn()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        mon = Monitor(m, opt, out=tmp_path / "ormas_run", check_every=1, verbose=False, **kw)
        train(m, mon, opt, 2)
        mon.close(report=False)
    once()
    before = (tmp_path / "ormas_run" / "diary.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite=True"):
        once()
    assert (tmp_path / "ormas_run" / "diary.jsonl").read_bytes() == before
    once(overwrite=True)
    assert (tmp_path / "ormas_run" / "diary.jsonl").read_bytes() != before


def test_close_never_masks_the_training_error(tmp_path, monkeypatch, capsys):
    import ormas_monitor.report as report

    def broken(run_dir):
        raise OSError("disk full")
    monkeypatch.setattr(report, "write_all", broken)
    m = small_cnn()
    with pytest.raises(ValueError, match="training blew up"):
        with Monitor(m, out=tmp_path / "a", verbose=False):
            raise ValueError("training blew up")
    assert "disk full" in capsys.readouterr().err
    assert events(tmp_path / "a")[-1]["error"] == "ValueError: training blew up"

    mon = Monitor(m, out=tmp_path / "b", verbose=False)
    with pytest.raises(KeyError):
        try:
            raise KeyError("in the loop")
        finally:
            mon.close()                    # the pattern users write; must not replace the KeyError
    with pytest.raises(OSError, match="disk full"):           # with no pending error, it raises
        Monitor(m, out=tmp_path / "c", verbose=False).close()


def test_non_torch_optimizer_degrades(tmp_path):
    class EngineOptimizer:                 # DeepSpeed-like: not a torch.optim.Optimizer
        param_groups = []

        def zero_grad(self):
            pass

        def step(self):
            pass
    m = small_cnn()
    mon = loop(m, EngineOptimizer(), lambda: torch.randn(4, 3, 8, 8), tmp_path / "run",
               lossf=lambda y: y.pow(2).mean())
    ev = events(tmp_path / "run")
    man = next(e for e in ev if e["event"] == "manifest")
    assert any("EngineOptimizer is not a torch.optim.Optimizer" in n for n in man["notes"])
    assert not any(p["certificate"] for p in man["producers"])
    assert mon.counts["repairs"] == 0
    with pytest.raises(ValueError, match=r"repair needs a torch\.optim\.Optimizer"):
        Monitor(m, EngineOptimizer(), out=tmp_path / "x", repair=True, verbose=False)


def test_demo_cli_help(capsys):
    with pytest.raises(SystemExit):
        cli(["demo", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "re-running replaces them" in out and "only watching" in out
