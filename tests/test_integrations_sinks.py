"""Sinks: TensorBoard event files, W&B (disabled mode and a recording run), MLflow (fake module),
webhook POSTs to a local HTTP server, logging, and the lazy-import contract."""
from __future__ import annotations

import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor import Monitor
from ormas_monitor.integrations import sinks as S

CERT = {"seq": 3, "time": 0.0, "event": "certificate", "step": 40, "severity": "proof",
        "layer": "conv2", "count": 16, "prev": "0" * 64, "hash": "a" * 64}
INFO = {"seq": 4, "time": 0.0, "event": "alarm_cleared", "step": 50, "layer": "conv1",
        "diagnosis": "vanishing", "grad_norm": 1.0, "baseline": 1.0, "prev": "0" * 64, "hash": "b" * 64}
ROW = {"step": 40, "loss": 1.25, "grad": {"conv1": 0.5, "conv2": 0.0, "head": 2.0},
       "stuck": {"conv1": [0, 0, 0, 0], "conv2": [16, 0, 0, 0]}}


def small_cnn() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))


def run_lesioned(tmp: Path, sinks: list, steps: int = 6) -> Monitor:
    """Plain loop with the second conv zeroed: yields structure, channels and certificate events."""
    model = small_cnn()
    with torch.no_grad():
        model[3].weight.zero_()
        model[3].bias.zero_()
        model[4].bias.fill_(-0.1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    mon = Monitor(model, opt, out=tmp / "run", check_every=2, verbose=False, sinks=sinks)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(torch.randn(8, 3, 8, 8, generator=g)),
                                           torch.randint(0, 4, (8,), generator=g))
        opt.zero_grad()
        loss.backward()
        mon.step(loss)
        opt.step()
    mon.close(report=False)
    return mon


def diary(mon: Monitor) -> list[dict]:
    return [json.loads(x) for x in (mon.out / "diary.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------ helpers
def test_format_and_severity():
    assert S.format_event(CERT) == "step 40: certificate conv2: 16 channels certified dead"
    assert S.severity_of(INFO) == "info"
    assert S.severity_of({"event": "sink_error"}) == "warning"
    with pytest.raises(ValueError):
        S.WebhookSink("http://x", min_severity="loud")
    with pytest.raises(ValueError):
        S.WebhookSink("file:///etc/passwd")


def test_import_never_needs_optional_packages():
    code = ("import sys\n"
            "for m in ('lightning','pytorch_lightning','transformers','tensorboard','wandb','mlflow'):\n"
            "    sys.modules[m] = None\n"
            "import ormas_monitor.integrations as I\n"
            "for cls, extra in ((I.WandbSink, 'wandb'), (I.MLflowSink, 'mlflow')):\n"
            "    try:\n"
            "        cls()\n"
            "    except ImportError as e:\n"
            "        assert f'ormas-monitor[{extra}]' in str(e), e\n"
            "    else:\n"
            "        raise AssertionError(cls)\n"
            "for name, extra in (('LightningCallback', 'lightning'), ('HuggingFaceCallback', 'huggingface')):\n"
            "    try:\n"
            "        getattr(I, name)\n"
            "    except ImportError as e:\n"
            "        assert f'ormas-monitor[{extra}]' in str(e), e\n"
            "    else:\n"
            "        raise AssertionError(name)\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


# ------------------------------------------------------------------ TensorBoard
def test_tensorboard_writes_event_files(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = S.TensorBoardSink(str(tmp_path / "tb"), layers=["3", "8"])
    mon = run_lesioned(tmp_path, [tb])
    tb.close()
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))
    acc = EventAccumulator(str(tmp_path / "tb"))
    acc.Reload()
    scalars = acc.Tags()["scalars"]
    assert {"ormas/loss", "ormas/grad_norm/3", "ormas/grad_norm/8", "ormas/events/certificate",
            "ormas/stuck_channels/3", "ormas/stuck_channels_total"} <= set(scalars)
    assert "ormas/grad_norm/0" not in scalars                     # filtered out by `layers`
    assert [e.step for e in acc.Scalars("ormas/loss")] == [0, 2, 4]
    assert acc.Scalars("ormas/grad_norm/3")[0].value == 0.0
    assert acc.Tags()["tensors"]                                  # event text entries
    assert any(e["event"] == "certificate" for e in diary(mon))


# ------------------------------------------------------------------ W&B
class RecordingRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.defined: list[tuple] = []
        self.alerts: list[dict] = []

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, data):
        self.logged.append(dict(data))

    def alert(self, **kw):
        self.alerts.append(kw)


def test_wandb_sink_logs_metrics_events_and_alerts():
    pytest.importorskip("wandb")
    run = RecordingRun()
    sink = S.WandbSink(run=run, alert_severity="proof")
    sink("metrics", ROW)
    sink("event", CERT)
    sink("event", INFO)
    assert run.defined == [("ormas/step", None), ("ormas/*", "ormas/step")]
    assert run.logged[0]["ormas/step"] == 40 and run.logged[0]["ormas/grad_norm/conv2"] == 0.0
    assert run.logged[0]["ormas/stuck_channels_total"] == 16
    assert run.logged[1] == {"ormas/events/certificate": 1, "ormas/step": 40}
    assert len(run.alerts) == 1 and "16 channels certified dead" in run.alerts[0]["text"]


def test_wandb_sink_with_real_disabled_run(tmp_path):
    # In a subprocess with a timeout, so wandb's own exit-time teardown can never stall the suite.
    pytest.importorskip("wandb")
    code = ("import json, sys, torch, torch.nn as nn, wandb\n"
            "from ormas_monitor import Monitor\n"
            "from ormas_monitor.integrations import WandbSink\n"
            "run = wandb.init(mode='disabled', project='ormas-test')\n"
            "m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.LazyLinear(2))\n"
            "m(torch.randn(2, 3, 6, 6))\n"
            "opt = torch.optim.SGD(m.parameters(), lr=0.1)\n"
            "mon = Monitor(m, opt, out=sys.argv[1], check_every=1, verbose=False, sinks=[WandbSink()])\n"
            "for _ in range(3):\n"
            "    loss = m(torch.randn(4, 3, 6, 6)).square().mean()\n"
            "    opt.zero_grad(); loss.backward(); mon.step(loss); opt.step()\n"
            "mon.close(report=False)\n"
            "run.finish()\n"
            "lines = [json.loads(x) for x in open(sys.argv[1] + '/diary.jsonl')]\n"
            "print(sum(e['event'] == 'sink_error' for e in lines))\n")
    env = {**os.environ, "WANDB_MODE": "disabled", "WANDB_DIR": str(tmp_path), "WANDB_SILENT": "true"}
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "run")], capture_output=True,
                         text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "0"


def test_wandb_sink_without_run_is_disabled_not_fatal(tmp_path):
    wandb = pytest.importorskip("wandb")
    assert wandb.run is None
    mon = run_lesioned(tmp_path, [S.WandbSink()])
    errs = [e for e in diary(mon) if e["event"] == "sink_error"]
    assert len(errs) == 1 and "no active W&B run" in errs[0]["error"]


# ------------------------------------------------------------------ MLflow
def test_mlflow_sink_with_fake_module(monkeypatch, tmp_path):
    calls: list[tuple] = []
    fake = types.ModuleType("mlflow")
    fake.log_metrics = lambda metrics, step=None: calls.append(("metrics", dict(metrics), step))
    fake.log_metric = lambda key, value, step=None: calls.append(("metric", key, value, step))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    sink = S.MLflowSink(layers=["conv*"])
    sink("metrics", ROW)
    sink("event", CERT)
    sink("event", {"event": "sink_error", "sink": "x", "error": "y"})
    kind, metrics, step = calls[0]
    assert step == 40 and metrics["ormas/loss"] == 1.25 and "ormas/grad_norm/head" not in metrics
    assert calls[1] == ("metric", "ormas/events/certificate", 1.0, 40)
    assert calls[2] == ("metric", "ormas/events/sink_error", 1.0, 40)   # no step: last seen
    mon = run_lesioned(tmp_path, [S.MLflowSink()])
    assert not any(e["event"] == "sink_error" for e in diary(mon))
    assert any(c[0] == "metric" and c[1] == "ormas/events/certificate" for c in calls[3:])


# ------------------------------------------------------------------ webhook
class Recorder(http.server.BaseHTTPRequestHandler):
    bodies: list = []
    status = 200

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        type(self).bodies.append((self.headers["Content-Type"], json.loads(self.rfile.read(n))))
        self.send_response(type(self).status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    handler = type("H", (Recorder,), {"bodies": [], "status": 200})
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/hook", handler
    srv.shutdown()
    srv.server_close()


@pytest.mark.parametrize("blocking", [False, True])
def test_webhook_posts_events_at_or_above_severity(tmp_path, server, blocking):
    url, handler = server
    hook = S.WebhookSink(url, min_severity="warning", blocking=blocking)
    mon = run_lesioned(tmp_path, [hook])
    hook.close()
    lines = diary(mon)
    sent = [b["event"]["event"] for _, b in handler.bodies]
    want = [e["event"] for e in lines
            if S.SEVERITY_RANK.get(S.severity_of(e), 0) >= 1 and e["event"] not in ("run_end",)]
    assert sent == want and "certificate" in sent and "structure" not in sent
    ctype, body = handler.bodies[sent.index("certificate")]
    assert ctype == "application/json" and body["source"] == "ormas-monitor"
    assert body["event"]["hash"] == next(e for e in lines if e["event"] == "certificate")["hash"]
    assert hook.sent == len(sent)


def test_webhook_failure_disables_sink_and_is_recorded(tmp_path, server):
    url, handler = server
    handler.status = 500
    hook = S.WebhookSink(url, min_severity="info", blocking=True)
    mon = run_lesioned(tmp_path, [hook])
    errs = [e for e in diary(mon) if e["event"] == "sink_error"]
    assert len(errs) == 1 and "500" in errs[0]["error"]
    assert len(handler.bodies) == 1           # disabled after the first failure


# ------------------------------------------------------------------ logging
def test_logging_sink_levels(caplog):
    sink = S.LoggingSink("ormas_test", metrics_level=logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="ormas_test"):
        sink("event", CERT)
        sink("event", INFO)
        sink("metrics", ROW)
    recs = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert recs[0] == (logging.WARNING, "step 40: certificate conv2: 16 channels certified dead")
    assert recs[1][0] == logging.INFO
    assert recs[2] == (logging.DEBUG, "step 40: loss 1.25, 3 layers with gradients, 16 stuck channels")


def test_console_sink_filters_and_installs_one_handler(capsys):
    a, b = S.ConsoleSink(), S.ConsoleSink(min_severity="info")
    log = logging.getLogger("ormas_monitor.console")
    assert sum(getattr(h, "_ormas_console", False) for h in log.handlers) == 1
    a("event", INFO)                      # below a's threshold
    a("event", CERT)
    b("event", INFO)
    err = capsys.readouterr().err.splitlines()
    assert err == ["[ormas-monitor] step 40: certificate conv2: 16 channels certified dead",
                   "[ormas-monitor] step 50: alarm_cleared conv1: vanishing cleared"]
