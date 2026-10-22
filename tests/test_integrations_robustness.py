"""The callbacks must never break training because of the monitor: an optimizer they cannot use
(DeepSpeed-like, custom wrappers) degrades the monitor instead of raising, and a failure while
closing the monitor never replaces the user's own exception."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ormas_monitor.integrations._common import usable_optimizer


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        logits = self.head(x.mean((2, 3)))
        if labels is None:
            return logits
        return {"loss": nn.functional.cross_entropy(logits, labels), "logits": logits}


class FlatMasterOptimizer:
    """Mimics DeepSpeed ZeRO: not a torch optimizer, and the torch optimizer inside it steps one
    flat fp32 copy of the parameters, which is then copied back into the model."""

    def __init__(self, params, lr: float = 0.05) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.flat = nn.Parameter(torch.cat([p.detach().reshape(-1) for p in self.params]))
        self.optimizer = torch.optim.SGD([self.flat], lr=lr, momentum=0.9)

    def inner(self) -> torch.optim.Optimizer:
        return self.optimizer

    @property
    def param_groups(self):
        return self.inner().param_groups

    @property
    def defaults(self):
        return self.inner().defaults

    @property
    def state(self):
        return self.inner().state

    def state_dict(self):
        return self.inner().state_dict()

    def load_state_dict(self, sd) -> None:
        self.inner().load_state_dict(sd)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.flat.grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                    for p in self.params])
        self.inner().step()
        with torch.no_grad():
            off = 0
            for p in self.params:
                p.copy_(self.flat[off:off + p.numel()].view_as(p))
                off += p.numel()
        return loss


class OpaqueWrapper(FlatMasterOptimizer):
    """A custom wrapper that keeps its inner optimizer under a name nobody knows."""

    def __init__(self, params, lr: float = 0.05) -> None:
        super().__init__(params, lr)
        self.engine = self.optimizer
        del self.optimizer

    def inner(self) -> torch.optim.Optimizer:
        return self.engine


def events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "diary.jsonl").read_text().splitlines()]


def failing_report(monkeypatch) -> None:
    def boom(*a, **k):
        raise OSError("disk full while writing the report")
    monkeypatch.setattr("ormas_monitor.report.write_all", boom)


# ---------------------------------------------------------------------------------- unit
def test_usable_optimizer_reasons():
    net = Net()
    sgd = torch.optim.SGD(net.parameters(), lr=0.1)
    assert usable_optimizer(sgd, net) == (sgd, None)
    opt, why = usable_optimizer(FlatMasterOptimizer(net.parameters()), net)
    assert opt is None and "does not hold the model's parameters" in why
    opt, why = usable_optimizer(object(), net)
    assert opt is None and "wraps none" in why


# ---------------------------------------------------------------------------------- Lightning
def _lightning():
    return pytest.importorskip("lightning.pytorch")


def _fit(cb, tmp: Path, optimizer_cls=None, fail_at: int | None = None, steps: int = 6):
    pl = _lightning()

    class Module(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = Net()

        def training_step(self, batch, batch_idx):
            if fail_at is not None and batch_idx == fail_at:
                raise RuntimeError("boom")
            return self.net(*batch)["loss"]

        def configure_optimizers(self):
            if optimizer_cls is None:
                return torch.optim.SGD(self.parameters(), lr=0.05)
            return optimizer_cls(self.parameters())

    g = torch.Generator().manual_seed(1)
    ds = torch.utils.data.TensorDataset(torch.randn(64, 3, 8, 8, generator=g), torch.randint(0, 4, (64,), generator=g))
    module = Module()
    trainer = pl.Trainer(callbacks=[cb], accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp),
                         max_steps=steps)
    trainer.fit(module, torch.utils.data.DataLoader(ds, batch_size=8))
    return module


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("optimizer_cls", [FlatMasterOptimizer, OpaqueWrapper])
def test_lightning_unusable_optimizer_degrades(tmp_path, capsys, optimizer_cls):
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    cb = OrmasMonitorCallback(repair=True, check_every=1, write_report=False)
    before = Net().conv1.weight.detach().clone()
    module = _fit(cb, tmp_path, optimizer_cls)
    out = capsys.readouterr().out
    assert out.count("no usable torch optimizer") == 1 and "ignored repair" in out
    assert not torch.equal(module.net.conv1.weight.detach(), before)      # training really ran
    ev = events(tmp_path / "ormas_run")
    assert ev[-1]["event"] == "run_end" and ev[-1]["steps"] == 6
    assert not any(e["event"] in ("repair", "certificate") for e in ev)
    assert cb.monitor.optimizer is None and not cb.monitor.repair_mode


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_degrade_message_is_logged_when_not_verbose(tmp_path, caplog):
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    cb = OrmasMonitorCallback(check_every=1, write_report=False, verbose=False)
    with caplog.at_level(logging.WARNING, logger="ormas_monitor.integrations"):
        _fit(cb, tmp_path, FlatMasterOptimizer, steps=2)
    assert [r.levelno for r in caplog.records if "no usable torch optimizer" in r.message] == [logging.WARNING]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_close_failure_never_masks_training_error(tmp_path, monkeypatch, capsys):
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    failing_report(monkeypatch)
    cb = OrmasMonitorCallback(check_every=1, verbose=False)
    with pytest.raises(RuntimeError, match="boom"):
        _fit(cb, tmp_path, fail_at=3)
    assert cb.report is None
    assert "could not finish" in capsys.readouterr().err       # reported, not raised
    last = events(tmp_path / "ormas_run")[-1]
    assert last["event"] == "run_end" and "boom" in last.get("error", "")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_close_failure_at_fit_end_does_not_fail_fit(tmp_path, monkeypatch):
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    failing_report(monkeypatch)
    cb = OrmasMonitorCallback(check_every=1, verbose=False)
    _fit(cb, tmp_path, steps=3)
    assert isinstance(cb.close_error, OSError)
    assert events(tmp_path / "ormas_run")[-1]["event"] == "run_end"


# ---------------------------------------------------------------------------------- Hugging Face
class Data(torch.utils.data.Dataset):
    def __init__(self, n: int = 64) -> None:
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, 4, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "labels": self.y[i]}


def _train(cb, tmp: Path, flat: bool = False):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    model = Net()
    optimizers = (None, None)
    if flat:
        opt = FlatMasterOptimizer(model.parameters())
        optimizers = (opt, torch.optim.lr_scheduler.LambdaLR(opt.optimizer, lambda s: 1.0))
    args = transformers.TrainingArguments(
        output_dir=str(tmp), max_steps=4, per_device_train_batch_size=8, learning_rate=1e-2, report_to=[],
        save_strategy="no", logging_strategy="no", use_cpu=True, disable_tqdm=True,
        dataloader_pin_memory=False, seed=0)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=Data(), callbacks=[cb],
                                   optimizers=optimizers)
    trainer.train()
    return trainer


def test_hf_unusable_optimizer_degrades(tmp_path, capsys):
    pytest.importorskip("transformers")
    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback
    cb = OrmasMonitorCallback(repair=True, check_every=1, write_report=False)
    _train(cb, tmp_path, flat=True)
    assert capsys.readouterr().out.count("no usable torch optimizer") == 1
    ev = events(tmp_path / "ormas_run")
    assert ev[-1]["event"] == "run_end" and ev[-1]["steps"] == 4
    assert cb.monitor.optimizer is None and not cb.monitor.repair_mode


def test_hf_close_failure_at_train_end_does_not_fail_train(tmp_path, monkeypatch, caplog):
    pytest.importorskip("transformers")
    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback
    failing_report(monkeypatch)
    cb = OrmasMonitorCallback(check_every=1, verbose=False)
    _train(cb, tmp_path)
    assert isinstance(cb.close_error, OSError)
    assert any("could not finish" in r.message for r in caplog.records)


def test_hf_manual_close_in_except_keeps_user_error(tmp_path, monkeypatch):  # finally: close()
    pytest.importorskip("transformers")
    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback

    class Boom(OrmasMonitorCallback):
        def on_pre_optimizer_step(self, *a, **k):
            super().on_pre_optimizer_step(*a, **k)
            if self.monitor.step_count == 2:
                raise RuntimeError("boom")

    failing_report(monkeypatch)
    cb = Boom(check_every=1, verbose=False)
    with pytest.raises(RuntimeError, match="boom"):
        try:
            _train(cb, tmp_path)
        finally:
            cb.close()
    assert cb.report is None
    assert events(tmp_path / "ormas_run")[-1]["event"] == "run_end"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_existing_run_dir_is_refused_unless_overwrite(tmp_path):
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    _fit(OrmasMonitorCallback(check_every=1, verbose=False, write_report=False), tmp_path, steps=2)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _fit(OrmasMonitorCallback(check_every=1, verbose=False, write_report=False), tmp_path, steps=2)
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False, overwrite=True)
    _fit(cb, tmp_path, steps=3)
    assert events(tmp_path / "ormas_run")[-1]["steps"] == 3


def test_hf_existing_run_dir_is_refused_unless_overwrite(tmp_path):
    pytest.importorskip("transformers")
    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback
    _train(OrmasMonitorCallback(check_every=1, verbose=False, write_report=False), tmp_path)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _train(OrmasMonitorCallback(check_every=1, verbose=False, write_report=False), tmp_path)
    _train(OrmasMonitorCallback(check_every=1, verbose=False, write_report=False, overwrite=True), tmp_path)
