"""Hugging Face Trainer: add one callback. The model only has to return {"loss": ...} as
transformers models do.

    python examples/integrations/huggingface_example.py [--out hf_out]
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from ormas_monitor.integrations.huggingface import OrmasMonitorCallback


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        self.head = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.head(x.mean((2, 3)))


def break_layer(model: SmallCNN) -> None:
    """Zero conv2 and push its BatchNorm shift below 0: every channel's ReLU closes for good."""
    with torch.no_grad():
        model.conv2.weight.zero_()
        model.conv2.bias.zero_()
        model.bn2.bias.fill_(-0.1)


def synthetic(n: int = 512, seed: int = 0) -> torch.utils.data.TensorDataset:
    """Images whose class is the brightest quadrant: learnable in a few hundred steps."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 16, 16, generator=g)
    y = torch.randint(0, 4, (n,), generator=g)
    for i, c in enumerate(y.tolist()):
        r, col = divmod(c, 2)
        x[i, :, r * 8:(r + 1) * 8, col * 8:(col + 1) * 8] += 1.5
    return torch.utils.data.TensorDataset(x, y)


class WithLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = SmallCNN()

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        logits = self.net(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out


class Dicts(torch.utils.data.Dataset):
    def __init__(self, ds: torch.utils.data.TensorDataset) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> dict:
        x, y = self.ds[i]
        return {"x": x, "labels": y}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hf_out")
    a = ap.parse_args()
    torch.manual_seed(0)
    model = WithLoss()
    break_layer(model.net)
    args = TrainingArguments(output_dir=a.out, num_train_epochs=4, per_device_train_batch_size=16,
                             gradient_accumulation_steps=2, learning_rate=3e-3, report_to=[],
                             save_strategy="no", logging_steps=16, use_cpu=True, disable_tqdm=True,
                             dataloader_pin_memory=False, seed=0)
    cb = OrmasMonitorCallback(repair=True, check_every=10)
    trainer = Trainer(model=model, args=args, train_dataset=Dicts(synthetic()), callbacks=[cb])
    trainer.train()
    print(f"optimizer steps {trainer.state.global_step}, monitor steps {cb.monitor.step_count}")
    print(f"logged losses {[round(h['loss'], 3) for h in trainer.state.log_history if 'loss' in h]}")
    print(f"counts {cb.monitor.counts}")
    print(f"report: {cb.report}")


if __name__ == "__main__":
    main()
