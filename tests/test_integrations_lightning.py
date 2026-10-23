"""Lightning integration: a real `lightning.Trainer.fit` on CPU with a tiny Conv-BN-ReLU model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pl = pytest.importorskip("lightning.pytorch")

from ormas_monitor.integrations.lightning import OrmasMonitorCallback  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore:.*does not have many workers"),
              pytest.mark.filterwarnings("ignore:.*LeafSpec"),
              pytest.mark.filterwarnings("ignore:.*GPU available but not used"),
              pytest.mark.filterwarnings("ignore:.*smaller than the logging interval")]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.head(x.mean((2, 3)))


def lesion(net: Net) -> None:
    with torch.no_grad():
        net.conv2.weight.zero_()
        net.conv2.bias.zero_()
        net.bn2.bias.fill_(-0.1)


class Module(pl.LightningModule):
    def __init__(self, opt: str = "sgd", manual: bool = False) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = Net()
        self.opt = opt
        self.automatic_optimization = not manual
        self.losses: list[float] = []
        self.optimizer_steps = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.net(x), y)
        self.losses.append(float(loss.detach()))
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            return None
        return loss

    def configure_optimizers(self):
        if self.opt == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=1e-2)
        else:
            opt = torch.optim.SGD(self.parameters(), lr=0.05)

        def count(*_):
            self.optimizer_steps += 1
        opt.register_step_post_hook(count)
        return opt


def loader(n: int = 64) -> torch.utils.data.DataLoader:
    g = torch.Generator().manual_seed(1)
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3, 8, 8, generator=g), torch.randint(0, 4, (n,), generator=g))
    return torch.utils.data.DataLoader(ds, batch_size=8)


def fit(module: Module, cb: OrmasMonitorCallback, tmp: Path, **trainer_kw) -> pl.Trainer:
    kw = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp))
    kw.update(trainer_kw)
    trainer = pl.Trainer(callbacks=[cb], **kw)
    trainer.fit(module, loader())
    return trainer


def events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "diary.jsonl").read_text().splitlines()]


def metrics(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def test_one_monitor_step_per_optimizer_step_and_loss(tmp_path):
    m = Module()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = fit(m, cb, tmp_path, max_steps=5)
    mon = cb.monitor
    assert mon.step_count == m.optimizer_steps == trainer.global_step == 5
    run = tmp_path / "ormas_run"
    assert mon.out == run and events(run)[-1]["event"] == "run_end"
    assert [r["loss"] for r in metrics(run)] == pytest.approx(m.losses, rel=1e-5)


def test_gradient_accumulation(tmp_path):
    m = Module()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    fit(m, cb, tmp_path, max_steps=4, accumulate_grad_batches=2)
    assert len(m.losses) == 8
    assert cb.monitor.step_count == m.optimizer_steps == 4
    # the monitor sees the loss whose gradient is in .grad: the mean over the micro-batches
    want = [(a + b) / 2 for a, b in zip(m.losses[::2], m.losses[1::2])]
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(want, rel=1e-5)


def test_manual_optimization(tmp_path):
    m = Module(manual=True)
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    fit(m, cb, tmp_path, max_steps=3)
    assert cb.monitor.step_count == m.optimizer_steps == 3
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(m.losses, rel=1e-5)


def test_zeroed_layer_gets_certificate(tmp_path):
    m = Module(opt="adam")
    lesion(m.net)
    cb = OrmasMonitorCallback(check_every=2, verbose=False, write_report=False)
    fit(m, cb, tmp_path, max_steps=6)
    certs = [e for e in events(tmp_path / "ormas_run") if e["event"] == "certificate"]
    assert [c["layer"] for c in certs] == ["net.conv2"] and certs[0]["count"] == 8
    assert float(m.net.conv2.weight.detach().abs().max()) == 0.0      # watch-only: nothing changed


def test_repair_changes_weights_in_lightning_optimizer(tmp_path):
    m = Module(opt="adam")
    lesion(m.net)
    cb = OrmasMonitorCallback(repair=True, check_every=2, verbose=False, out=str(tmp_path / "run"))
    trainer = fit(m, cb, tmp_path, max_steps=8)
    opt = trainer.optimizers[0]
    assert cb.monitor.optimizer is opt and isinstance(opt, torch.optim.Adam)
    assert any(p is m.net.conv2.weight for g in opt.param_groups for p in g["params"])
    reps = [e for e in events(tmp_path / "run") if e["event"] == "repair"]
    assert reps and reps[0]["layer"] == "net.conv2" and reps[0]["count"] == 8
    assert float(m.net.conv2.weight.detach().abs().min(dim=1).values.max()) > 0
    assert float(opt.state[m.net.conv2.weight]["exp_avg"].abs().max()) > 0   # trains after repair
    assert cb.report is not None and Path(cb.report).exists()


def test_closes_on_exception(tmp_path):
    class Boom(Module):
        def training_step(self, batch, batch_idx):
            if batch_idx == 3:
                raise RuntimeError("boom")
            return super().training_step(batch, batch_idx)

    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    with pytest.raises(RuntimeError, match="boom"):
        fit(Boom(), cb, tmp_path, max_steps=10)
    last = events(tmp_path / "ormas_run")[-1]
    assert last["event"] == "run_end" and last["steps"] == 3


def test_bad_monitor_kwargs_fail_at_construction():
    with pytest.raises(TypeError):
        OrmasMonitorCallback(chek_every=3)
    with pytest.raises(TypeError):
        OrmasMonitorCallback(optimizer=None)


def test_to_loggers_writes_into_trainer_tensorboard(tmp_path):
    pytest.importorskip("tensorboard")
    from lightning.pytorch.loggers import TensorBoardLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logger = TensorBoardLogger(str(tmp_path), name="tb")
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False, to_loggers=True)
    fit(Module(), cb, tmp_path, max_steps=3, logger=logger)
    assert cb.monitor.out == Path(logger.log_dir) / "ormas_run"
    acc = EventAccumulator(logger.log_dir)
    acc.Reload()
    tags = acc.Tags()["scalars"]
    assert "ormas/loss" in tags and "ormas/grad_norm/net.conv1" in tags
    assert [e.step for e in acc.Scalars("ormas/loss")] == [0, 1, 2]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="16-bit GradScaler path needs MPS or CUDA")
def test_amp_loss_is_unscaled(tmp_path):
    m = Module()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = fit(m, cb, tmp_path, max_steps=3, precision="16-mixed", accelerator="mps")
    assert trainer.precision_plugin.scaler is not None
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(m.losses, rel=1e-4)
