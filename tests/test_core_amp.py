"""Mixed precision: GradScaler (scaled and unscaled gradients, skipped steps) and bf16 autocast on CPU."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor.monitor import Monitor, _norm32
from test_core_helpers import batches, events, lesion, small_cnn

needs_cpu_scaler = pytest.mark.skipif(not hasattr(torch.amp, "GradScaler"), reason="torch.amp.GradScaler missing")


def run(tmp_path, mode, steps=30, poison_at=None, check_every=1):
    """mode: plain | scaled (no unscale_) | unscaled (unscale_ before step)."""
    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu") if mode != "plain" else None
    mon = Monitor(model, opt, out=tmp_path / mode, check_every=check_every, verbose=False, scaler=scaler)
    for i, (x, y) in enumerate(batches(steps)):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        (scaler.scale(loss) if scaler else loss).backward()
        if i == poison_at:
            model[3].weight.grad[0, 0, 0, 0] = float("inf")
        if mode == "unscaled":
            scaler.unscale_(opt)
        mon.step(loss)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
    mon.close(report=False)
    rows = [json.loads(l) for l in open(tmp_path / mode / "metrics.jsonl")]
    return mon, events(tmp_path / mode), rows


@needs_cpu_scaler
def test_scaled_and_unscaled_gradients_read_the_same(tmp_path):
    _, _, plain = run(tmp_path, "plain")
    for mode in ("scaled", "unscaled"):
        _, ev, rows = run(tmp_path, mode)
        for a, b in zip(plain, rows):
            for layer, g in a["grad"].items():
                assert b["grad"][layer] == pytest.approx(g, rel=1e-5), (mode, a["step"], layer)
        man = next(e for e in ev if e["event"] == "manifest")
        assert man["precision"]["scaler"] and man["precision"]["scale"] == 65536.0


@needs_cpu_scaler
def test_scaler_overflow_is_a_skipped_step_not_an_alarm(tmp_path):
    mon, ev, _ = run(tmp_path, "scaled", steps=30, poison_at=20)
    kinds = [e["event"] for e in ev]
    assert "nonfinite_gradient" not in kinds and "alarm" not in kinds
    skip = [e for e in ev if e["event"] == "skipped_step"]
    assert len(skip) == 1 and skip[0]["step"] == 20 and skip[0]["layers"] == ["3"]
    assert skip[0]["severity"] == "info" and skip[0]["scale"] == 65536.0
    assert mon.counts["skipped_steps"] == 1 and mon.counts["nonfinite"] == 0
    # the skipped step moved no baseline: every layer has one update fewer than checks
    assert {b.n for b in mon._base.values()} == {29}
    assert mon._loss.n == 29


def test_without_scaler_nonfinite_gradient_is_critical(tmp_path):
    mon, ev, _ = run(tmp_path, "plain", steps=25, poison_at=20)
    bad = [e for e in ev if e["event"] == "nonfinite_gradient"]
    assert len(bad) == 1 and bad[0]["severity"] == "critical" and bad[0]["step"] == 20
    # the inf reached the weights (no scaler to skip the step), so the loss went non-finite too
    assert any(e["event"] == "nonfinite_loss" for e in ev) and mon.counts["nonfinite"] == 2
    assert not any(e["event"] == "skipped_step" for e in ev)


def test_scaler_needs_optimizer_and_a_scaler(tmp_path):
    with pytest.raises(ValueError):
        Monitor(small_cnn(), out=tmp_path / "a", scaler=torch.amp.GradScaler("cpu"), verbose=False)
    with pytest.raises(TypeError):
        Monitor(small_cnn(), torch.optim.SGD(small_cnn().parameters(), lr=0.1), out=tmp_path / "b",
                scaler=object(), verbose=False)


def test_norms_are_full_precision():
    g = (torch.rand(4096) * 3).to(torch.bfloat16)
    full = torch.linalg.vector_norm(g.float())
    assert _norm32(g).dtype == torch.float32
    assert _norm32(g).item() == pytest.approx(full.item(), rel=1e-6)


def test_bf16_autocast_certifies_and_repairs(tmp_path):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", repair=True, check_every=5, patience=2, verbose=False)
    for i, (x, y) in enumerate(batches(100)):
        if i == 50:
            lesion(model[3], model[4], opt)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    ev = events(tmp_path / "run")
    assert next(e for e in ev if e["event"] == "structure")["relu_after"] == ["1", "4"]
    assert next(e for e in ev if e["event"] == "certificate")["layer"] == "3"
    assert next(e for e in ev if e["event"] == "repair")["count"] == 16
    assert model[3].weight.abs().amax(dim=(1, 2, 3)).min() > 0
