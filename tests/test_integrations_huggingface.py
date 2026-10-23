"""Hugging Face integration: a real `transformers.Trainer.train` on CPU with a tiny Conv-BN-ReLU
model that returns ``{"loss": ...}`` like a transformers model."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from ormas_monitor.integrations.huggingface import OrmasMonitorCallback  # noqa: E402


class Net(nn.Module):
    """Conv-BN-ReLU x2 + linear head; records each training micro-batch loss for the tests."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()
        self.head = nn.Linear(8, 4)
        self.losses: list[float] = []

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.head(x.mean((2, 3)))

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        logits = self.logits(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
            if self.training:
                self.losses.append(float(out["loss"].detach()))
        return out


class TokenNormalisedNet(Net):
    """Accepts `num_items_in_batch`, like recent transformers models: loss = sum / items in step."""

    def forward(self, x, labels=None, num_items_in_batch=None, **kwargs):
        logits = self.logits(x)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        self.per_item = getattr(self, "per_item", []) + [float(loss.detach()) / len(labels)]
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return {"loss": loss, "logits": logits}


class Data(torch.utils.data.Dataset):
    def __init__(self, n: int = 64) -> None:
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, 4, (n,), generator=g)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "labels": self.y[i]}


def lesion(net: Net) -> None:
    with torch.no_grad():
        net.conv2.weight.zero_()
        net.conv2.bias.zero_()
        net.bn2.bias.fill_(-0.1)


def train(model: nn.Module, cb: OrmasMonitorCallback, tmp: Path, eval_dataset: Data | None = None,
          **args_kw) -> transformers.Trainer:
    kw = dict(output_dir=str(tmp), max_steps=4, per_device_train_batch_size=8, learning_rate=1e-2,
              report_to=[], save_strategy="no", logging_strategy="no", use_cpu=True,
              disable_tqdm=True, dataloader_pin_memory=False, seed=0)
    kw.update(args_kw)
    trainer = transformers.Trainer(model=model, args=transformers.TrainingArguments(**kw),
                                   train_dataset=Data(), eval_dataset=eval_dataset,
                                   callbacks=[cb])
    steps = []
    trainer.create_optimizer()
    trainer.optimizer.register_step_post_hook(lambda *_: steps.append(1))
    trainer.train()
    trainer.optimizer_steps = len(steps)
    return trainer


def events(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "diary.jsonl").read_text().splitlines()]


def metrics(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def test_one_monitor_step_per_optimizer_step_and_loss(tmp_path):
    net = Net()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = train(net, cb, tmp_path)
    assert cb.monitor.step_count == trainer.optimizer_steps == trainer.state.global_step == 4
    run = tmp_path / "ormas_run"
    assert events(run)[-1]["event"] == "run_end"
    assert [r["loss"] for r in metrics(run)] == pytest.approx(net.losses, rel=1e-5)


def test_gradient_accumulation(tmp_path):
    net = Net()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = train(net, cb, tmp_path, gradient_accumulation_steps=2)
    assert len(net.losses) == 8
    assert cb.monitor.step_count == trainer.optimizer_steps == 4
    want = [(a + b) / 2 for a, b in zip(net.losses[::2], net.losses[1::2])]
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(want, rel=1e-5)


def test_gradient_accumulation_with_num_items_in_batch(tmp_path):
    net = TokenNormalisedNet()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = train(net, cb, tmp_path, gradient_accumulation_steps=2)
    assert trainer.model_accepts_loss_kwargs
    assert cb.monitor.step_count == 4
    # equal-size micro-batches: the step's loss is the mean of the per-item losses
    want = [(a + b) / 2 for a, b in zip(net.per_item[::2], net.per_item[1::2])]
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(want, rel=1e-5)


def test_zeroed_layer_gets_certificate(tmp_path):
    net = Net()
    lesion(net)
    cb = OrmasMonitorCallback(check_every=2, verbose=False, write_report=False)
    train(net, cb, tmp_path, max_steps=6)
    certs = [e for e in events(tmp_path / "ormas_run") if e["event"] == "certificate"]
    assert [c["layer"] for c in certs] == ["conv2"] and certs[0]["count"] == 8
    assert float(net.conv2.weight.detach().abs().max()) == 0.0


def test_repair_changes_weights_in_trainer_optimizer(tmp_path):
    net = Net()
    lesion(net)
    cb = OrmasMonitorCallback(repair=True, check_every=2, verbose=False, out=str(tmp_path / "run"))
    trainer = train(net, cb, tmp_path, max_steps=8)
    inner = trainer.optimizer.optimizer          # accelerate's AcceleratedOptimizer wraps it
    assert cb.monitor.optimizer is inner
    assert any(p is net.conv2.weight for g in inner.param_groups for p in g["params"])
    reps = [e for e in events(tmp_path / "run") if e["event"] == "repair"]
    assert reps and reps[0]["layer"] == "conv2" and reps[0]["count"] == 8
    assert float(net.conv2.weight.detach().abs().amax(dim=(1, 2, 3)).min()) > 0
    assert float(inner.state[net.conv2.weight]["exp_avg"].abs().max()) > 0
    assert cb.report is not None and Path(cb.report).exists()


def test_eval_forwards_are_not_counted(tmp_path):
    net = Net()
    cb = OrmasMonitorCallback(check_every=1, verbose=False, write_report=False)
    trainer = train(net, cb, tmp_path, eval_dataset=Data(16), max_steps=4, eval_strategy="steps",
                    eval_steps=2, per_device_eval_batch_size=8)
    assert trainer.state.log_history and "eval_loss" in trainer.state.log_history[0]
    assert [r["loss"] for r in metrics(tmp_path / "ormas_run")] == pytest.approx(net.losses, rel=1e-5)
