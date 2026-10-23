"""The script behind docs/tutorial.md: an ordinary training loop plus four ormas-monitor lines.

It knocks out two conv layers at epoch 3 on purpose, to show what a failure looks like.
Run:  python examples/walkthrough_train.py   (needs torchvision; downloads CIFAR-10 into ./data once)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from ormas_monitor import Monitor                                   # <- ormas-monitor

# --- the engineer's own model and data (nothing special) ---------------------------
def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
model = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), nn.Flatten(),
                      nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, 10))
ds = CIFAR10("data", train=True, download=True)
x = torch.tensor(ds.data[:10000]).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.25)
y = torch.tensor(ds.targets[:10000])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

mon = Monitor(model, opt, repair=True, out="my_run")                # <- ormas-monitor
for epoch in range(6):
    if epoch == 3:                                                  # simulate a failure mid-training
        with torch.no_grad():
            for b in (model[1], model[2]):
                b[0].weight.zero_()
                b[0].bias.zero_()
        opt.state.clear()
        print("--- epoch 3: two conv layers knocked out (simulated failure)")
    perm = torch.randperm(len(x))
    for i in range(0, len(x), 128):
        idx = perm[i:i + 128]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        mon.step(loss)                                              # <- ormas-monitor
        opt.step()
    with torch.no_grad():
        acc = (model(x[:2000]).argmax(1) == y[:2000]).float().mean().item() * 100
    print(f"epoch {epoch}: train accuracy {acc:.1f}%")
mon.close()                                                         # <- ormas-monitor
