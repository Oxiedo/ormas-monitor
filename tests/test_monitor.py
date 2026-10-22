import json

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.checkpoint import run as check_file
from ormas_monitor.cli import main as cli
from ormas_monitor.diary import Diary, verify
from ormas_monitor.structure import pair_by_order, scan_channels


def small_cnn():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 4))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out += x                     # skip connection before the ReLU
        return self.head(self.relu(out).mean((2, 3)))


def lesion(conv, bn, opt=None):
    """Zero a layer, like the paper's lesion; the paper's harness also clears optimizer state."""
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        bn.bias.fill_(-0.1)
    if opt is not None:
        opt.state.clear()


def train(model, mon, opt, steps, after=None):
    g = torch.Generator().manual_seed(1)
    for i in range(steps):
        if after and i == after[0]:
            after[1]()
        x = torch.randn(16, model[0].in_channels if isinstance(model, nn.Sequential) else 8, 8, 8, generator=g)
        y = torch.randint(0, 4, (16,), generator=g)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()


def test_scan_channels():
    w = torch.randn(4, 2, 3, 3)
    w[1] = 0
    w[2] = 0
    gamma = torch.ones(4)
    gamma[3] = 0
    beta = torch.tensor([0.5, -0.2, 0.3, 0.0])
    s = scan_channels(w, None, gamma, beta)
    assert s.certified == [1, 3]       # zero filter / zero gamma with shift <= 0
    assert s.flat == [2]               # zero filter, positive shift: constant output
    s = scan_channels(w, torch.tensor([1.0, -1.0, 0.5, 0.0]), None, None)
    assert s.certified == [1] and s.flat == [2]


def test_pairing():
    kinds = {"a": "Conv2d", "b": "BatchNorm2d", "c": "Linear", "d": "BatchNorm1d"}
    sizes = {"a": 8, "b": 8, "c": 4, "d": 5}
    assert pair_by_order(list("abcd"), kinds, sizes) == {"a": "b"}


def test_diary_chain(tmp_path):
    d = Diary(tmp_path / "d.jsonl")
    for i in range(5):
        d.write("e", i=i)
    d.close()
    assert verify(tmp_path / "d.jsonl")[0]
    lines = (tmp_path / "d.jsonl").read_text().splitlines()
    rec = json.loads(lines[2])
    rec["i"] = 99
    lines[2] = json.dumps(rec)
    (tmp_path / "d.jsonl").write_text("\n".join(lines) + "\n")
    ok, _, msg = verify(tmp_path / "d.jsonl")
    assert not ok and "event 2" in msg


def test_certifies_and_repairs_dead_layer(tmp_path):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", repair=True, check_every=5, patience=2, verbose=False)
    train(model, mon, opt, 120, after=(60, lambda: lesion(model[3], model[4], opt)))
    mon.close()
    events = [json.loads(l) for l in open(tmp_path / "run" / "diary.jsonl")]
    kinds = [e["event"] for e in events]
    cert = next(e for e in events if e["event"] == "certificate")
    assert cert["layer"] == "3" and cert["count"] == 16 and cert["step"] >= 60
    rep = next(e for e in events if e["event"] == "repair")
    assert rep["layer"] == "3" and rep["step"] > cert["step"]
    assert model[3].weight.abs().amax(dim=(1, 2, 3)).min() > 0      # every filter revived
    assert "certificate" not in [k for e, k in zip(events, kinds) if e.get("step", 0) < 60]
    assert verify(tmp_path / "run" / "diary.jsonl")[0]
    assert (tmp_path / "run" / "report.html").read_text().count("certified dead") >= 1


def test_alarm_only_mode_does_not_touch_weights(tmp_path):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=5, verbose=False)
    train(model, mon, opt, 80, after=(40, lambda: lesion(model[3], model[4], opt)))
    mon.close()
    assert model[3].weight.abs().max() == 0
    assert any(json.loads(l)["event"] == "certificate" for l in open(tmp_path / "run" / "diary.jsonl"))


def test_no_certificate_while_momentum_can_revive(tmp_path):
    model = small_cnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=5, verbose=False)
    train(model, mon, opt, 70, after=(60, lambda: lesion(model[3], model[4])))   # state kept
    mon.close(report=False)
    events = [json.loads(l) for l in open(tmp_path / "run" / "diary.jsonl")]
    assert not any(e["event"] == "certificate" for e in events)
    assert any(e["event"] == "channels" and e["flat"] for e in events)


def test_no_certificate_through_skip_connection(tmp_path):
    model = Residual()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with torch.no_grad():
        model.conv.weight.zero_()
        model.bn.bias.fill_(-1.0)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=1, verbose=False)
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        loss = model(torch.randn(4, 8, 6, 6, generator=g)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    events = [json.loads(l)["event"] for l in open(tmp_path / "run" / "diary.jsonl")]
    assert "certificate" not in events and "bn" not in mon._relu_after


def test_checkpoint_check_and_diff(tmp_path):
    model = small_cnn()
    torch.save(model.state_dict(), tmp_path / "before.pt")
    lesion(model[3], model[4])
    torch.save({"state_dict": model.state_dict()}, tmp_path / "after.pt")
    res = check_file(tmp_path / "after.pt", tmp_path / "before.pt")
    dead = [f for f in res["findings"] if f["kind"] == "dead_channels"]
    assert dead and dead[0]["layer"] == "3" and len(dead[0]["channels"]) == 16
    assert res["diff"]["newly_dead"] == [{"layer": "3", "batchnorm": "4", "count": 16}]
    assert cli(["check", str(tmp_path / "after.pt"), "--against", str(tmp_path / "before.pt")]) == 0
    assert (tmp_path / "after.pt.health.html").exists()
    clean = check_file(tmp_path / "before.pt")
    assert not [f for f in clean["findings"] if f["severity"] != "info"]


def test_refuses_pickled_objects(tmp_path):
    torch.save({"model": small_cnn()}, tmp_path / "full.pt")
    assert cli(["check", str(tmp_path / "full.pt")]) == 2


def test_repair_needs_optimizer():
    with pytest.raises(ValueError):
        Monitor(small_cnn(), repair=True, verbose=False)


def test_relu_followers_skips_residual_add():
    from ormas_monitor.structure import relu_followers
    assert relu_followers(small_cnn(), torch.randn(2, 3, 8, 8)) == {"1", "4"}
    assert relu_followers(Residual(), torch.randn(2, 8, 6, 6)) == set()


def test_checkpoint_respects_structure(tmp_path):
    from ormas_monitor.checkpoint import check
    model = Residual()
    with torch.no_grad():
        model.conv.weight.zero_()
        model.bn.bias.fill_(-1.0)
    res = check(model.state_dict(), relu_after=set())
    assert not [f for f in res["findings"] if f["kind"] == "dead_channels"]
    assert [f for f in res["findings"] if f["kind"] == "flat_channels"]


def test_decayed_channels_are_flagged_but_not_certified():
    w = torch.randn(4, 2, 3, 3)
    w[1] *= 1e-9                                   # dead channel shrunk by weight decay
    gamma = torch.ones(4)
    gamma[2] = 1e-10
    beta = torch.tensor([0.1, -1e-8, -0.3, 0.2])
    s = scan_channels(w, None, gamma, beta)
    assert s.certified == [] and s.decayed == [1, 2]
