"""Reproduce: switching off the channels `check` flags as decayed leaves the model's answers unchanged.

For each torchvision ImageNet model, finds the channels flagged "decayed to ~0" (BatchNorm feeding a
ReLU directly), forces their post-ReLU output to exactly 0 with a hook, and compares the logits with
the untouched model on real photos (CIFAR-10 test images upscaled to 224 px).

This measures whether the channels can be removed without changing outputs. It does not remove them
structurally, so it does not measure speed or memory savings.

Run:  python examples/reproduce/zero_flagged_channels.py --data DIR [--images 256]   (needs torchvision)
      add --control to switch off the same number of random live channels instead
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10

from ormas_monitor.checkpoint import check
from ormas_monitor.structure import relu_followers

MODELS = ("resnet18", "resnet50", "vgg19_bn", "densenet169", "mnasnet1_0", "mobilenet_v2")


def images(root: str, n: int) -> torch.Tensor:
    ds = CIFAR10(root, train=False, download=True)
    x = torch.tensor(ds.data[:n]).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([model(x[i:i + 64]) for i in range(0, len(x), 64)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--images", type=int, default=256)
    ap.add_argument("--control", action="store_true",
                    help="switch off the same number of random LIVE channels instead (should change outputs)")
    a = ap.parse_args()
    x = images(a.data, a.images)
    for name in MODELS:
        model = torchvision.models.get_model(name, weights="DEFAULT").eval()
        follow = relu_followers(model, x[:1])
        res = check(model.state_dict(), relu_after=follow)
        flagged = {l["batchnorm"]: l["decayed"] for l in res["layers"] if l.get("decayed")}
        if a.control:
            rng, mods = random.Random(0), dict(model.named_modules())
            flagged = {b: rng.sample([i for i in range(mods[b].num_features) if i not in set(c)], len(c))
                       for b, c in flagged.items()}
        total = sum(l["channels"] for l in res["layers"] if l["batchnorm"] in follow)
        before = logits(model, x)
        modules = dict(model.named_modules())

        def zero(idx):
            def hook(module, inputs, out):
                out = out.clone()
                out[:, idx] = torch.minimum(out[:, idx], torch.zeros(()))   # post-ReLU these become 0
                return out
            return hook
        handles = [modules[b].register_forward_hook(zero(torch.tensor(c))) for b, c in flagged.items()]
        after = logits(model, x)
        for h in handles:
            h.remove()
        n = sum(map(len, flagged.values()))
        diff = (after - before).abs().max().item()
        same = (after.argmax(1) == before.argmax(1)).float().mean().item() * 100
        print(f"{name:12s} switched off {n:5d} of {total:6,d} channels ({100 * n / max(total, 1):4.1f}%)  "
              f"max logit change {diff:.2e}  same top-1 answer on {same:.1f}% of {len(x)} images")


if __name__ == "__main__":
    main()
