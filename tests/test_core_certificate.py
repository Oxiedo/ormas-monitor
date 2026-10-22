"""Certificate soundness: each case the V3 review found certified falsely (repros r1-r5) is refused,
and the cases the theorem covers (SGD, AdamW, inplace and shared ReLU modules) still certify."""
from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from ormas_monitor.config import MonitorConfig
from ormas_monitor.monitor import Monitor
from ormas_monitor.repair import sgd_decay_keeps_sign
from ormas_monitor.structure import scan_channels
from test_core_helpers import events


class ConvBnRelu(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=inplace)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.relu(self.bn(self.conv(x))).mean((2, 3))).squeeze(1)


class SecondConsumer(ConvBnRelu):            # r1: the BatchNorm output also feeds a side branch
    def __init__(self):
        super().__init__()
        self.aux = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        y = self.bn(self.conv(x))
        return self.head(self.relu(y).mean((2, 3))).squeeze(1) + self.aux(y).mean((1, 2, 3))


class Branches(nn.Module):                   # r2: both convs run before either BatchNorm
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv_b = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn_a, self.bn_b = nn.BatchNorm2d(4), nn.BatchNorm2d(4)
        self.relu_a, self.relu_b = nn.ReLU(), nn.ReLU()
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        a, b = self.conv_a(x), self.conv_b(x)
        z = torch.cat([self.relu_a(self.bn_a(a)), self.relu_b(self.bn_b(b))], 1)
        return self.head(z.mean((2, 3))).squeeze(1)


class SharedBn(nn.Module):                   # r4: one BatchNorm called twice
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn, self.relu, self.head = nn.BatchNorm2d(4), nn.ReLU(), nn.Linear(4, 1)

    def forward(self, x):
        a = self.relu(self.bn(self.conv(x)))
        return self.head((self.bn(self.conv2(x)) + a).mean((2, 3))).squeeze(1)


class SharedRelu(nn.Module):                 # torchvision BasicBlock style: one ReLU module, used twice
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4)
        self.conv2, self.bn2 = nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return self.head(self.relu(out).mean((2, 3))).squeeze(1)


class Stash(ConvBnRelu):                     # the BatchNorm output escapes through an attribute
    def forward(self, x):
        y = self.bn(self.conv(x))
        self.stashed = y
        return self.head(self.relu(y).mean((2, 3))).squeeze(1)


class HeavyBall(torch.optim.Optimizer):      # momentum under a key no built-in rule knows
    def __init__(self, params, lr=0.1, mu=0.9):
        super().__init__(params, dict(lr=lr, mu=mu))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                v = self.state[p].setdefault("velocity", torch.zeros_like(p))
                if p.grad is not None:
                    v.mul_(g["mu"]).add_(p.grad)
                p.sub_(g["lr"] * v)


def kill(conv, bn, beta=-0.5, ch=0):
    with torch.no_grad():
        conv.weight[ch] = 0
        bn.bias[ch] = beta


def run(tmp_path, model, opt, steps=12, check_every=1, x_shape=(16, 3, 8, 8), loss_fn=None, pass_loss=True,
        **kw):
    torch.manual_seed(0)
    mon = Monitor(model, opt, out=tmp_path / "run", check_every=check_every, verbose=False, **kw)
    for _ in range(steps):
        y = model(torch.randn(*x_shape))
        loss = (y - 1).pow(2).mean() if loss_fn is None else loss_fn(model, y)
        opt.zero_grad()
        loss.backward()
        mon.step(loss if pass_loss else None)
        opt.step()
    mon.close(report=False)
    return mon, events(tmp_path / "run")


def certs(ev):
    return [(e["layer"], e["channels"]) for e in ev if e["event"] == "certificate"]


def refusals(ev):
    return [(e["layer"], e["reason"]) for e in ev if e["event"] == "certificate_refused"]


# ---------------------------------------------------------------- structure (r1, r2, r4)
def test_second_consumer_of_batchnorm_output_is_refused(tmp_path):
    torch.manual_seed(0)
    m = SecondConsumer()
    kill(m.conv, m.bn)
    _, ev = run(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.1))
    assert certs(ev) == []
    assert any(l == "conv" and "2 consumers" in r for l, r in refusals(ev))
    assert m.conv.weight[0].abs().max() > 0          # it was alive all along: the gradient revived it


def test_pairing_follows_dataflow_not_execution_order(tmp_path):
    torch.manual_seed(0)
    m = Branches()
    with torch.no_grad():
        m.conv_b.weight[0] = 0
        m.bn_a.bias[0] = -1.0                        # wrong BatchNorm: closed
        m.bn_b.bias[0] = 1.0                         # conv_b's real BatchNorm: open, so it is alive
    mon, ev = run(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.1), steps=4)
    assert mon.pairs == {"conv_a": "bn_a", "conv_b": "bn_b"}
    assert next(e for e in ev if e["event"] == "structure")["pairs"] == mon.pairs
    assert certs(ev) == []
    man = next(e for e in ev if e["event"] == "manifest")
    assert all(p["certificate"] for p in man["producers"] if p["layer"].startswith("conv"))


def test_shared_batchnorm_is_refused(tmp_path):
    torch.manual_seed(0)
    m = SharedBn()
    with torch.no_grad():
        m.bn.weight[0] = 0
        m.bn.bias[0] = -0.5
    mon, ev = run(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.1), steps=6)
    assert certs(ev) == []
    assert mon.pairs == {}                           # fed by two convs: paired with neither
    man = next(e for e in ev if e["event"] == "manifest")
    assert not any(p["certificate"] for p in man["producers"])


def test_inplace_and_shared_relu_still_certify(tmp_path):
    torch.manual_seed(0)
    m = ConvBnRelu(inplace=True)
    kill(m.conv, m.bn)
    _, ev = run(tmp_path / "a", m, torch.optim.SGD(m.parameters(), lr=0.1))
    assert certs(ev) == [("conv", [0])]

    torch.manual_seed(0)
    m = SharedRelu()
    kill(m.conv1, m.bn1)
    _, ev = run(tmp_path / "b", m, torch.optim.SGD(m.parameters(), lr=0.1), x_shape=(16, 4, 8, 8))
    assert certs(ev) == [("conv1", [0])]
    man = {p["layer"]: p for p in next(e for e in ev if e["event"] == "manifest")["producers"]}
    assert not man["conv2"]["certificate"]           # residual add before the ReLU


def test_gradient_guard_catches_a_consumer_the_graph_walk_missed(tmp_path):
    """Without the loss the graph is walked from the model's outputs, which cannot see a tensor
    that escapes through an attribute; the per-check gradient guard still refuses."""
    torch.manual_seed(0)
    m = Stash()
    kill(m.conv, m.bn)
    mon, ev = run(tmp_path, m, torch.optim.SGD(m.parameters(), lr=0.1), pass_loss=False,
                  loss_fn=lambda model, y: (y - 1).pow(2).mean() + model.stashed.mean())
    assert mon.graph_source == "model outputs"
    assert certs(ev) == []
    assert ("conv", "these channels have a nonzero gradient at this check") in refusals(ev)


class Toggle(ConvBnRelu):                    # from some step on, the BatchNorm also runs a second time
    twice = False

    def forward(self, x):
        out = super().forward(x)
        return out + self.bn(self.conv(x)).mean((1, 2, 3)) if self.twice else out


def test_dynamic_graph_revokes_eligibility(tmp_path):
    torch.manual_seed(0)
    m = Toggle()
    kill(m.conv, m.bn)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    mon = Monitor(m, opt, out=tmp_path / "run", check_every=1, verbose=False)
    for i in range(6):
        m.twice = i >= 3
        loss = (m(torch.randn(16, 3, 8, 8)) - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    ev = events(tmp_path / "run")
    assert [e["step"] for e in ev if e["event"] == "certificate"] == [0]
    refused = [e for e in ev if e["event"] == "certificate_refused"]
    assert len(refused) == 1 and refused[0]["step"] == 3 and "the graph changed" in refused[0]["reason"]
    assert "ran 2 times" in mon._chains[0].reason


def test_eval_mode_batchnorm_is_refused(tmp_path):
    torch.manual_seed(0)
    m = ConvBnRelu()
    kill(m.conv, m.bn)
    m.bn.eval()                                      # eval from the start: never paired
    _, ev = run(tmp_path / "a", m, torch.optim.SGD(m.parameters(), lr=0.1), steps=3)
    assert certs(ev) == []

    torch.manual_seed(0)
    m = ConvBnRelu()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    mon = Monitor(m, opt, out=tmp_path / "b", check_every=1, verbose=False)
    for i in range(4):
        if i == 2:                                   # switched to eval after the structure is known
            kill(m.conv, m.bn)
            m.bn.eval()
        loss = (m(torch.randn(16, 3, 8, 8)) - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    ev = events(tmp_path / "b")
    assert certs(ev) == []
    assert any("eval mode" in r for _, r in refusals(ev))


# ---------------------------------------------------------------- optimizers (r3, r4)
def _kill_and_run(tmp_path, make_opt, steps=40, check_every=1, beta=-0.01, **kw):
    torch.manual_seed(0)
    m = ConvBnRelu()
    kill(m.conv, m.bn, beta=beta)
    _, ev = run(tmp_path, m, make_opt(m.parameters()), steps=steps, check_every=check_every, **kw)
    return m, ev


@pytest.mark.parametrize("label, make, reason", [
    ("Adam coupled L2", lambda p: torch.optim.Adam(p, lr=1e-2, weight_decay=1e-2), "coupled L2"),
    ("RMSprop", lambda p: torch.optim.RMSprop(p, lr=1e-2, weight_decay=1e-2), "not an optimizer the certificate covers"),
    ("Adagrad", lambda p: torch.optim.Adagrad(p, lr=1e-2), "not an optimizer the certificate covers"),
])
def test_uncovered_optimizers_are_refused(tmp_path, label, make, reason):
    _, ev = _kill_and_run(tmp_path, make)
    assert certs(ev) == [], label
    assert any(reason in r for _, r in refusals(ev)), refusals(ev)


@pytest.mark.parametrize("label, make", [
    ("AdamW", lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2)),
    ("SGD momentum + weight decay", lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=5e-4)),
    ("Adam without weight decay", lambda p: torch.optim.Adam(p, lr=1e-2)),
    ("Adam decoupled", lambda p: torch.optim.Adam(p, lr=1e-2, weight_decay=1e-2, decoupled_weight_decay=True)),
])
def test_covered_optimizers_certify_and_the_channel_stays_dead(tmp_path, label, make):
    m, ev = _kill_and_run(tmp_path, make, steps=60)
    assert certs(ev) == [("conv", [0])], label
    assert m.conv.weight[0].abs().max() == 0 and m.bn.bias[0] <= 0     # what the certificate claimed


def test_custom_optimizer_is_refused_and_opt_in_is_strict(tmp_path):
    torch.manual_seed(0)
    m = ConvBnRelu()
    opt = HeavyBall(m.parameters())
    for _ in range(3):                                # momentum builds up, then the channel is killed
        loss = (m(torch.randn(16, 3, 8, 8)) - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    kill(m.conv, m.bn, beta=-0.05)
    _, ev = run(tmp_path / "a", m, opt, steps=4)
    assert certs(ev) == []
    assert any("HeavyBall is not an optimizer the certificate covers" in r for _, r in refusals(ev))
    man = next(e for e in ev if e["event"] == "manifest")
    assert man["optimizer"] == {"type": "HeavyBall", "certificate_family": None}

    # Opted in: every same-shaped state tensor ("velocity") must be zero for the channel.
    kill(m.conv, m.bn, beta=-0.05)                   # the velocity revived it; kill it again
    _, ev = run(tmp_path / "b", m, opt, steps=2, config=MonitorConfig(certify_optimizers=("HeavyBall",)))
    assert certs(ev) == []
    assert ("conv", "the optimizer still holds momentum for these channels") in refusals(ev)


def test_sgd_decay_sign_analysis_matches_torch_sgd():
    """Wherever the analysis says weight decay keeps p <= 0, torch.optim.SGD agrees (from rest)."""
    def crosses(lr, wd, mu, damp, nesterov, zero_buffer):
        p = nn.Parameter(torch.tensor([-1.0], dtype=torch.float64))
        opt = torch.optim.SGD([p], lr=lr, weight_decay=wd, momentum=mu, dampening=damp, nesterov=nesterov)
        if zero_buffer and mu:
            opt.state[p]["momentum_buffer"] = torch.zeros_like(p)
        for _ in range(600):
            p.grad = torch.zeros_like(p)
            opt.step()
            if p.item() > 0:
                return True
        return False
    seen = set()
    for lr, wd, mu, damp, nest in itertools.product([0.1, 1.0, 10.0], [1e-3, 0.05, 0.5], [0, 0.9, 0.99],
                                                    [0, 0.5], [False, True]):
        if nest and (mu == 0 or damp):
            continue
        ok = sgd_decay_keeps_sign(lr, wd, mu, damp, nest)
        seen.add(ok)
        if ok:
            assert not crosses(lr, wd, mu, damp, nest, False) and not crosses(lr, wd, mu, damp, nest, True)
    assert seen == {True, False}
    assert not sgd_decay_keeps_sign(10.0, 0.05, 0.9) and crosses(10.0, 0.05, 0.9, 0, False, False)
    assert sgd_decay_keeps_sign(0.1, 1e-4, 0.9) and sgd_decay_keeps_sign(0.1, 1e-4, 0.9, nesterov=True)


# ---------------------------------------------------------------- floats (r5)
def test_zero_filter_with_nonzero_bias_is_not_certified(tmp_path):
    lin, bn = nn.Linear(5, 2), nn.BatchNorm1d(2)
    with torch.no_grad():
        lin.weight[0] = 0
        lin.bias[0] = 0.3
        bn.bias[0] = 0.0
    s = scan_channels(lin.weight, lin.bias, bn.weight, bn.bias)
    assert s.certified == [] and s.decayed == [0]
    with torch.no_grad():
        bn.weight[0] = 0                               # gamma == 0 removes the rounding: proven again
    assert scan_channels(lin.weight, lin.bias, bn.weight, bn.bias).certified == [0]

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 1))
    with torch.no_grad():
        model[0].weight[0] = 0
        model[0].bias[0] = 0.3
        model[1].bias[0] = 0.0
    _, ev = run(tmp_path, model, torch.optim.SGD(model.parameters(), lr=0.1), steps=5, x_shape=(24, 5))
    assert certs(ev) == []
