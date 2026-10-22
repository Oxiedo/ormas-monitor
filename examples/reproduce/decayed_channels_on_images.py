"""Reproduce: channels flagged "decayed to ~0" in torchvision models output ~0 on real images.

Runs ResNet-18, VGG-19-BN, DenseNet-169 and MnasNet-1.0 (ImageNet weights) on 64 CIFAR-10 test
images upscaled to 224 px, and prints the largest post-ReLU output of the flagged channels next
to the median peak of the other channels in the same layers.

Run:  python examples/reproduce/decayed_channels_on_images.py --data DIR   (needs torchvision)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10

from ormas_monitor.checkpoint import check
from ormas_monitor.structure import relu_followers

MODELS = ("resnet18", "vgg19_bn", "densenet169", "mnasnet1_0")


def images(root: str, n: int = 64) -> torch.Tensor:
    ds = CIFAR10(root, train=False, download=True)
    x = torch.tensor(ds.data[:n]).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    x = images(a.data)
    for name in MODELS:
        model = torchvision.models.get_model(name, weights="DEFAULT").eval()
        follow = relu_followers(model, x[:1])
        res = check(model.state_dict(), relu_after=follow)
        flagged = {l["batchnorm"]: l["decayed"] for l in res["layers"] if l.get("decayed")}
        modules, peaks = dict(model.named_modules()), {}
        handles = [modules[b].register_forward_hook(
            lambda m, i, o, b=b, peaks=peaks: peaks.__setitem__(b, o.detach().clamp(min=0).amax(dim=(0, 2, 3))))
            for b in flagged]
        with torch.no_grad():
            model(x)
        for h in handles:
            h.remove()
        dead = max(float(peaks[b][c].max()) for b, c in flagged.items())
        alive = torch.cat([peaks[b][[i for i in range(len(peaks[b])) if i not in set(c)]]
                           for b, c in flagged.items()])
        print(f"{name:12s} flagged {sum(map(len, flagged.values())):5d}  "
              f"max post-ReLU output of flagged {dead:.2e}  "
              f"median peak of other channels in the same layers {float(alive.median()):.2f}")


if __name__ == "__main__":
    main()
