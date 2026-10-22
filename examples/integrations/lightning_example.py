"""PyTorch Lightning: add one callback; the LightningModule is unchanged.

    python examples/integrations/lightning_example.py [--out ormas_runs/lightning]
"""
from __future__ import annotations

import argparse

import lightning as L
import torch
import torch.nn as nn

from ormas_monitor.integrations.lightning import OrmasMonitorCallback


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


class Classifier(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = SmallCNN()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_acc", (logits.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ormas_runs/lightning")
    a = ap.parse_args()
    torch.manual_seed(0)
    model = Classifier()
    break_layer(model.net)
    loader = torch.utils.data.DataLoader(synthetic(), batch_size=32, shuffle=True)
    cb = OrmasMonitorCallback(repair=True, check_every=10, out=a.out)
    trainer = L.Trainer(max_epochs=4, accumulate_grad_batches=2, accelerator="cpu", devices=1,
                        logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, callbacks=[cb])
    trainer.fit(model, loader)
    print(f"optimizer steps {trainer.global_step}, monitor steps {cb.monitor.step_count}")
    print(f"train accuracy (last epoch) {float(trainer.callback_metrics['train_acc_epoch']):.1%}")
    print(f"counts {cb.monitor.counts}")
    print(f"report: {cb.report}")


if __name__ == "__main__":
    main()
