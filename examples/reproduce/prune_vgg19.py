"""Reproduce: removing the channels `check` flags makes VGG-19-BN smaller and faster with the same answers.

1. `check` finds the channels of torchvision's ImageNet VGG-19-BN whose BatchNorm output is a
   constant <= 0 before a ReLU (decayed to ~0, or certified dead). They always output 0.
2. Those channels are cut out of the network: their conv filters, their BatchNorm entries, and
   the matching input slice of the next layer.
3. The cut model is compared with the original on real photos (CIFAR-10 test images upscaled to
   224 px): logits, top-1 answers, multiply-accumulate count and CPU latency.

Run:  python examples/reproduce/prune_vgg19.py --data DIR   (needs torchvision)
"""
from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10

from ormas_monitor.checkpoint import check
from ormas_monitor.structure import relu_followers


def images(root: str, n: int) -> torch.Tensor:
    ds = CIFAR10(root, train=False, download=True)
    x = torch.tensor(ds.data[:n]).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def dead_channels(model: nn.Module) -> dict[str, list[int]]:
    follow = relu_followers(model, torch.randn(1, 3, 224, 224))
    res = check(model.state_dict(), relu_after=follow)
    return {l["batchnorm"]: sorted(set(l["decayed"]) | set(l["certified"])) for l in res["layers"]}


def prune(model: nn.Module, dead: dict[str, list[int]]) -> nn.Module:
    """Cut dead channels out of VGG's conv stack and the first classifier layer."""
    p = copy.deepcopy(model)
    feats = list(p.features)
    prev_keep = None
    for i, mod in enumerate(feats):
        if not isinstance(mod, nn.Conv2d):
            continue
        conv, bn = mod, feats[i + 1]
        gone = set(dead.get(f"features.{i + 1}", []))
        keep = [c for c in range(conv.out_channels) if c not in gone]
        w = conv.weight.data[keep]
        if prev_keep is not None:
            w = w[:, prev_keep]
        nc = nn.Conv2d(w.shape[1], len(keep), conv.kernel_size, conv.stride, conv.padding)
        nc.weight.data, nc.bias.data = w.clone(), conv.bias.data[keep].clone()
        nb = nn.BatchNorm2d(len(keep)).eval()
        for a in ("weight", "bias", "running_mean", "running_var"):
            getattr(nb, a).data = getattr(bn, a).data[keep].clone()
        feats[i], feats[i + 1], prev_keep = nc, nb, keep
    p.features = nn.Sequential(*feats)
    fc = p.classifier[0]
    cols = torch.arange(fc.in_features).view(-1, 49)[prev_keep].flatten()     # 7x7 positions per channel
    nfc = nn.Linear(len(cols), fc.out_features)
    nfc.weight.data, nfc.bias.data = fc.weight.data[:, cols].clone(), fc.bias.data.clone()
    p.classifier[0] = nfc
    return p.eval()


def macs(model: nn.Module) -> int:
    """Multiply-accumulates for one 224x224 image (conv + linear layers)."""
    total = 0

    def hook(mod, inputs, out):
        nonlocal total
        if isinstance(mod, nn.Conv2d):
            total += out.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            total += mod.in_features * mod.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for h in hs:
        h.remove()
    return total


def latency(model: nn.Module, x: torch.Tensor, repeats: int) -> float:
    with torch.no_grad():
        model(x)
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t)
    return sorted(times)[repeats // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = torchvision.models.vgg19_bn(weights="DEFAULT").eval()
    dead = dead_channels(model)
    small = prune(model, dead)
    x = images(a.data, a.images)
    with torch.no_grad():
        before, after = model(x), small(x)
    n_dead = sum(len(v) for v in dead.values())
    n_all = sum(m.out_channels for m in model.features if isinstance(m, nn.Conv2d))
    m0, m1 = macs(model), macs(small)
    print(f"dead channels removed: {n_dead} of {n_all} conv channels")
    print(f"max logit difference: {(after - before).abs().max().item():.2e}; "
          f"same top-1 answer on {(after.argmax(1) == before.argmax(1)).float().mean().item() * 100:.1f}% "
          f"of {len(x)} photos")
    print(f"compute: {m0 / 1e9:.2f} -> {m1 / 1e9:.2f} GMACs per image ({100 * (1 - m1 / m0):.1f}% less)")
    print(f"parameters: {sum(t.numel() for t in model.parameters()) / 1e6:.1f}M -> "
          f"{sum(t.numel() for t in small.parameters()) / 1e6:.1f}M")
    t0, t1 = latency(model, x[:16], a.repeats), latency(small, x[:16], a.repeats)
    print(f"CPU latency, batch 16, {a.threads} threads (median of {a.repeats}): "
          f"{t0 * 1000:.0f} ms -> {t1 * 1000:.0f} ms ({100 * (1 - t1 / t0):.1f}% faster)")


if __name__ == "__main__":
    main()
