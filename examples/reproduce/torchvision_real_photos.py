"""Reproduce: what the channels `check` flags actually do on real photos, across torchvision's models.

Uses ImageNetV2 (matched frequency); the download commands are at the top of vgg19_real_photos.py.

For each BatchNorm -> ReLU model, it counts the flagged channels, records the largest output any of them
gives on the photos, and, if any of them ever fires, compares the answers with those channels switched off.
For the VGG models it also works out how much compute cutting them would save.

Run:  python examples/reproduce/torchvision_real_photos.py --data DIR [--images 2000] [--models vgg19_bn ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import torchvision
from prune_vgg19 import macs, prune
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from vgg19_real_photos import FolderLabel, free_cache, pick_device, predictions

from ormas_monitor.checkpoint import check
from ormas_monitor.structure import relu_followers

MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "wide_resnet50_2",
          "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn", "densenet121", "densenet169", "mobilenet_v2",
          "mobilenet_v3_large", "mobilenet_v3_small", "shufflenet_v2_x1_0", "mnasnet1_0", "regnet_x_400mf",
          "regnet_y_400mf", "regnet_y_1_6gf"]


def run_model(name: str, data_dir: str, images: int, device: str) -> dict:
    weights = torchvision.models.get_model_weights(name).DEFAULT
    model = torchvision.models.get_model(name, weights=weights).eval()
    follow = relu_followers(model, torch.randn(1, 3, 224, 224))
    layers = [l for l in check(model.state_dict(), relu_after=follow)["layers"] if l["batchnorm"] in follow]
    flagged = {l["batchnorm"]: sorted(set(l["decayed"]) | set(l["certified"])) for l in layers}
    flagged = {layer: chans for layer, chans in flagged.items() if chans}
    row = {"model": name, "weights": str(weights), "bn_relu_layers": len(layers),
           "channels": sum(l["channels"] for l in layers), "certified": sum(len(l["certified"]) for l in layers),
           "decayed": sum(len(l["decayed"]) for l in layers), "flagged": sum(map(len, flagged.values())),
           "images": images}
    if name.startswith("vgg") and flagged:
        before, after = macs(model), macs(prune(model, flagged))
        row.update(gmacs=before / 1e9, gmacs_cut=after / 1e9, compute_saving=1 - after / before)

    data = ImageFolder(data_dir, transform=weights.transforms())
    data.target_transform = FolderLabel(data.classes)
    pick = torch.randperm(len(data), generator=torch.Generator().manual_seed(2))[:images].tolist()
    loader = DataLoader(Subset(data, pick), batch_size=16, num_workers=2)
    model.to(device)
    modules, peaks, handles = dict(model.named_modules()), {}, []
    for layer in flagged:
        def watch(_module, _inputs, out, layer=layer):
            peak = out.detach().clamp(min=0).amax(dim=(0, 2, 3)).cpu()
            peaks[layer] = peak if layer not in peaks else torch.maximum(peaks[layer], peak)
        handles.append(modules[layer].register_forward_hook(watch))
    base, labels = predictions(model, loader, device)
    for handle in handles:
        handle.remove()
    row["top1"] = float((base == labels).float().mean())

    dead, others = [], []
    for layer, chans in flagged.items():
        for channel, value in enumerate(peaks[layer].tolist()):
            (dead if channel in set(chans) else others).append(value)
    row["flagged_max_output"] = max(dead) if dead else None
    row["flagged_ever_positive"] = sum(v > 0 for v in dead)
    row["others_median_peak"] = statistics.median(others) if others else None

    if row["flagged_ever_positive"]:
        handles = []
        for layer, chans in flagged.items():
            index = torch.tensor(chans, device=device)

            def switch_off(_module, _inputs, out, index=index):
                out = out.clone()
                out[:, index] = -1.0     # the ReLU that follows turns this into exactly 0
                return out
            handles.append(modules[layer].register_forward_hook(switch_off))
        preds, _ = predictions(model, loader, device)
        for handle in handles:
            handle.remove()
        row["answers_changed_when_off"] = int((preds != base).sum())
        row["top1_when_off"] = float((preds == labels).float().mean())
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="ImageNetV2 folder with one subfolder per class index")
    ap.add_argument("--images", type=int, default=2000)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", help="also write the results to this JSON file")
    a = ap.parse_args()
    device = pick_device()
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.6)   # keep an 8 GB laptop from swapping
    rows = []
    for name in a.models:
        rows.append(run_model(name, a.data, a.images, device))
        print(json.dumps(rows[-1]), flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=2))
        gc.collect()
        free_cache(device)


if __name__ == "__main__":
    main()
