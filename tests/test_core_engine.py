"""Core engine: config, schema, repair, layers filter, note_loss, sinks, manifest, record verification."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor.config import MonitorConfig
from ormas_monitor.monitor import Monitor
from ormas_monitor.repair import first_moment_mask, repair_channels, repair_generator, unwrap_optimizer
from ormas_monitor.schema import EVENT_FIELDS, RECORD_VERSION, validate_event
from test_core_helpers import events, lesion, small_cnn, train


def lesioned_run(tmp_path, repair=True, **kw):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", repair=repair, check_every=5, patience=2,
                  verbose=False, **kw)
    train(model, mon, opt, 100, after=(50, lambda: lesion(model[3], model[4], opt)))
    mon.close(report=False)
    return model, mon, events(tmp_path / "run")


# ---------------------------------------------------------------- config
def test_config_defaults_are_the_v01_values():
    c = MonitorConfig()
    assert (c.check_every, c.patience, c.sigma, c.warmup) == (50, 4, 2.5, 8)
    assert (c.alarm_min_decades, c.alarm_confirmations, c.certified_repair_streak) == (1.0, 2, 2)
    assert (c.loss_spike_factor, c.decayed_rel_tol, c.list_limit, c.repair_seed) == (1.5, 1e-5, 64, None)


@pytest.mark.parametrize("bad", [dict(check_every=0), dict(patience=True), dict(sigma=0), dict(warmup=-1),
                                 dict(alarm_confirmations=0), dict(loss_spike_factor=0.5),
                                 dict(decayed_rel_tol=0), dict(repair_seed=-1), dict(list_limit=2.0),
                                 dict(baseline_rate_floor=1.5)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        MonitorConfig(**bad)


def test_keyword_shortcuts_override_config(tmp_path):
    cfg = MonitorConfig(check_every=7, sigma=3.0)
    mon = Monitor(small_cnn(), out=tmp_path / "r", config=cfg, check_every=3, verbose=False)
    assert mon.config.check_every == 3 and mon.config.sigma == 3.0 and mon.check_every == 3
    mon.close(report=False)
    start = events(tmp_path / "r")[0]
    assert start["config"]["check_every"] == 3 and start["config"]["sigma"] == 3.0
    with pytest.raises(TypeError):
        Monitor(small_cnn(), out=tmp_path / "x", config={"check_every": 3}, verbose=False)


def test_alarm_confirmations_is_honoured(tmp_path):
    """Default needs 2 consecutive checks; with 1 the same lesion alarms one check earlier."""
    steps = {}
    for n in (1, 2):
        model = small_cnn()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        mon = Monitor(model, opt, out=tmp_path / str(n), check_every=5, verbose=False,
                      config=MonitorConfig(alarm_confirmations=n))
        train(model, mon, opt, 90, after=(60, lambda m=model, o=opt: lesion(m[3], m[4], o)))
        mon.close(report=False)
        steps[n] = min(e["step"] for e in events(tmp_path / str(n)) if e["event"] == "alarm")
    assert steps[1] == steps[2] - 5


# ---------------------------------------------------------------- schema
def test_validate_event_finds_problems():
    good = {"seq": 0, "time": 1.0, "event": "sink_error", "prev": "0" * 64, "hash": "a" * 64,
            "sink": "s", "error": "e"}
    assert validate_event(good) == []
    assert any("missing field 'error'" in p for p in validate_event({k: v for k, v in good.items() if k != "error"}))
    assert any("unknown event" in p for p in validate_event({**good, "event": "nope"}))
    assert any("hash" in p for p in validate_event({**good, "hash": "xyz"}))
    cert = {**good, "event": "certificate", "step": 3, "severity": "warning", "layer": "a", "batchnorm": None,
            "channels": [], "count": 0, "statement": "", "assumes": []}
    assert any("severity must be 'proof'" in p for p in validate_event(cert))
    assert validate_event([]) != []


def test_every_event_type_is_emitted_with_its_fields(tmp_path):
    """All event types the engine can write appear across these runs, and all validate."""
    _, _, ev = lesioned_run(tmp_path)
    seen = {e["event"] for e in ev}
    assert {"run_start", "structure", "manifest", "alarm", "channels", "certificate", "repair",
            "run_end"} <= seen
    assert set(seen) <= set(EVENT_FIELDS)
    assert ev[0]["record_version"] == RECORD_VERSION


def test_record_verifies_strictly(tmp_path):
    """Healthy run, zeroed layer, repair=True: the record passes C's verifier, schema included."""
    verify = pytest.importorskip("ormas_monitor.record.verify")
    lesioned_run(tmp_path)
    res = verify.verify_run(tmp_path / "run")
    assert res.ok, [(c.name, c.status, c.message) for c in res.checks]
    schema = next(c for c in res.checks if c.label == "event schema")
    assert schema.status == "pass", schema.message


# ---------------------------------------------------------------- repair
def test_repair_event_declares_a_discrete_unbounded_logged_change(tmp_path):
    model, _, ev = lesioned_run(tmp_path)
    rep = next(e for e in ev if e["event"] == "repair")
    assert rep["modification_class"] == "discrete" and rep["bound"] is None
    assert rep["weight_norm_before"] == 0 and rep["relative_change"] is None   # undefined, not 0
    assert rep["delta_norm"] == pytest.approx(rep["weight_norm_after"]) and rep["delta_norm"] > 0
    assert rep["rng"] == "torch global RNG"


def test_repair_relative_change_on_partial_layer():
    conv, bn = nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4)
    opt = torch.optim.SGD(list(conv.parameters()) + list(bn.parameters()), lr=0.1)
    with torch.no_grad():
        conv.weight[1] = 0
    before = conv.weight.detach().clone()
    res = repair_channels(conv, bn, torch.tensor([1]), opt, repair_generator(0, 10, "conv"))
    delta = (conv.weight - before).norm().item()
    assert res.count == 1 and res.delta_norm == pytest.approx(delta)
    assert res.relative_change == pytest.approx(delta / before.norm().item())
    assert torch.equal(conv.weight[0], before[0]) and conv.weight[1].abs().max() > 0


def test_seeded_repair_is_deterministic_in_seed_step_layer():
    def fresh(seed, step, layer):
        conv = nn.Conv2d(3, 8, 3)
        with torch.no_grad():
            conv.weight.zero_()
        torch.manual_seed(seed + 123)          # the global RNG must not matter
        opt = torch.optim.SGD(conv.parameters(), lr=0.1)
        repair_channels(conv, None, torch.arange(8), opt, repair_generator(seed, step, layer))
        return conv.weight.detach().clone()
    a = fresh(0, 5, "conv1")
    assert torch.equal(a, fresh(0, 5, "conv1"))
    assert not torch.equal(a, fresh(0, 6, "conv1")) and not torch.equal(a, fresh(1, 5, "conv1"))
    assert a.std().item() == pytest.approx((2 / 27) ** 0.5, rel=0.25)   # Kaiming fan_in, ReLU gain


def test_seeded_repair_in_monitor(tmp_path):
    w = []
    for i in range(2):
        model, _, ev = lesioned_run(tmp_path / str(i), config=MonitorConfig(repair_seed=7))
        assert "sha256(seed=7" in next(e for e in ev if e["event"] == "repair")["rng"]
        w.append(model[3].weight.detach().clone())
    assert torch.equal(w[0], w[1])


def test_repair_resets_optimizer_state_and_momentum_mask():
    conv = nn.Linear(3, 4)
    opt = torch.optim.SGD(conv.parameters(), lr=0.1, momentum=0.9)
    conv(torch.randn(2, 3)).sum().backward()
    opt.step()
    assert first_moment_mask(opt, conv, None).all()
    res = repair_channels(conv, None, torch.tensor([0, 2]), opt)
    assert res.optimizer_slices_reset == 2
    assert first_moment_mask(opt, conv, None).tolist() == [False, True, False, True]


def test_unwrap_optimizer_lightning():
    pytest.importorskip("lightning")
    from lightning.pytorch.core.optimizer import LightningOptimizer
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    assert unwrap_optimizer(LightningOptimizer(opt)) is opt
    assert unwrap_optimizer(opt) is opt and unwrap_optimizer(None) is None
    with pytest.raises(TypeError):
        unwrap_optimizer(object())


def test_unwrap_optimizer_accelerate():
    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    wrapped = Accelerator(cpu=True).prepare_optimizer(opt)
    assert type(wrapped).__name__ == "AcceleratedOptimizer" and unwrap_optimizer(wrapped) is opt


def test_certificate_needs_the_optimizer(tmp_path):
    """Without the optimizer the momentum condition cannot be checked: no certificate."""
    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mon = Monitor(model, out=tmp_path / "run", check_every=5, verbose=False)
    train(model, mon, opt, 80, after=(40, lambda: lesion(model[3], model[4])))
    mon.close(report=False)
    ev = events(tmp_path / "run")
    assert not any(e["event"] == "certificate" for e in ev)
    assert any(e["event"] == "channels" and e["n_flat"] == 16 for e in ev)
    man = next(e for e in ev if e["event"] == "manifest")
    assert not any(p["certificate"] for p in man["producers"])


# ---------------------------------------------------------------- manifest, layers, loss, sinks
def test_manifest_lists_active_mechanisms(tmp_path):
    from test_monitor import Residual
    model = Residual()
    mon = Monitor(model, torch.optim.SGD(model.parameters(), lr=0.01), out=tmp_path / "run",
                  check_every=1, verbose=False)
    model(torch.randn(2, 8, 6, 6)).sum().backward()
    mon.step()
    mon.close(report=False)
    man = next(e for e in events(tmp_path / "run") if e["event"] == "manifest")
    prod = {p["layer"]: p for p in man["producers"]}
    assert prod["conv"]["batchnorm"] == "bn" and not prod["conv"]["certificate"]
    assert "ReLU" in prod["conv"]["reason"]
    assert man["distributed"]["mode"] == "none" and man["precision"]["scaler"] is False
    assert man["repair"]["mode"] is False and man["check_every"] == 1 and man["channel_checks"] is True

    _, _, ev = lesioned_run(tmp_path / "b", repair="all")
    man = next(e for e in ev if e["event"] == "manifest")
    assert all(p["certificate"] for p in man["producers"] if p["kind"] == "Conv2d")
    assert man["repair"]["targets"] == ["certified", "flat", "decayed", "silent"]


def test_layers_filter(tmp_path):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", repair=True, check_every=5, patience=2,
                  verbose=False, layers=["3"])
    assert list(mon.modules) == ["3"]
    train(model, mon, opt, 100, after=(50, lambda: lesion(model[3], model[4], opt)))
    mon.close(report=False)
    ev = events(tmp_path / "run")
    rep = next(e for e in ev if e["event"] == "repair")
    assert rep["layer"] == "3" and rep["batchnorm"] == "4"    # BatchNorm pairing still found
    metrics = [json.loads(l) for l in open(tmp_path / "run" / "metrics.jsonl")]
    assert all(set(m["grad"]) <= {"3"} for m in metrics)
    man = next(e for e in ev if e["event"] == "manifest")
    assert man["layers"] == {"monitored": 1, "total": 5, "filter": ["3"]}
    with pytest.raises(ValueError, match="matches no layer"):
        Monitor(small_cnn(), out=tmp_path / "x", layers="encoder.*", verbose=False)


def test_note_loss(tmp_path):
    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=1, verbose=False)
    for i, loss in enumerate((2.5, 3.5, None)):
        model(torch.randn(2, 3, 8, 8)).sum().backward()
        mon.note_loss(torch.tensor(loss) if loss is not None else None)
        mon.step(1.25 if i == 1 else None)          # an explicit loss wins
    mon.close(report=False)
    rows = [json.loads(l) for l in open(tmp_path / "run" / "metrics.jsonl")]
    assert [r["loss"] for r in rows] == [2.5, 1.25, None]
    with pytest.raises(RuntimeError):
        mon.step()


def test_failing_sink_gives_exactly_one_sink_error(tmp_path):
    got, calls = [], {"n": 0}

    def bad(kind, payload):
        calls["n"] += 1
        raise ConnectionError("dashboard down")

    def good(kind, payload):
        got.append((kind, payload))

    model = small_cnn()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=2, verbose=False, sinks=[bad, good])
    train(model, mon, opt, 10)
    mon.close(report=False)
    ev = events(tmp_path / "run")
    errors = [e for e in ev if e["event"] == "sink_error"]
    assert len(errors) == 1 and "ConnectionError: dashboard down" in errors[0]["error"]
    assert calls["n"] == 1
    sent = [p for k, p in got if k == "event"]
    assert [p["seq"] for p in sent] == [e["seq"] for e in ev]      # every diary line, in order
    assert sum(k == "metrics" for k, _ in got) == 5


def test_sink_failing_on_the_sink_error_itself(tmp_path):
    """A second sink that fails while receiving the first sink_error gets its own, once."""
    def bad(kind, payload):
        raise RuntimeError("first")

    def picky(kind, payload):
        if payload.get("event") == "sink_error":
            raise RuntimeError("second")

    mon = Monitor(small_cnn(), out=tmp_path / "run", verbose=False, sinks=[bad, picky])
    mon.close(report=False)
    errors = [e["error"] for e in events(tmp_path / "run") if e["event"] == "sink_error"]
    assert errors == ["RuntimeError: first", "RuntimeError: second"]


def test_torch_compile(tmp_path):
    """Hooks sit on the compiled module's original submodules; names match the plain model."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    model = small_cnn()
    compiled = torch.compile(model, backend="eager")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(compiled, opt, out=tmp_path / "run", repair=True, check_every=5, patience=2, verbose=False)
    assert "3" in mon.modules
    train(compiled, mon, opt, 100, after=(50, lambda: lesion(model[3], model[4], opt)))
    mon.close(report=False)
    ev = events(tmp_path / "run")
    assert next(e for e in ev if e["event"] == "structure")["relu_after"] == ["1", "4"]
    assert next(e for e in ev if e["event"] == "certificate")["layer"] == "3"
    assert next(e for e in ev if e["event"] == "repair")["layer"] == "3"
    assert next(e for e in ev if e["event"] == "manifest")["distributed"]["wrappers"] == ["torch.compile"]
