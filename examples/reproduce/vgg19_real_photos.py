"""Reproduce: cutting the flagged channels from VGG-19-BN changes nothing on real ImageNet-style photos.

Uses ImageNetV2 (matched frequency): 10,000 labelled photos across all 1,000 ImageNet classes.

    curl -LO https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
    tar xzf imagenetv2-matched-frequency.tar.gz

Run the comparison, then measure CPU speed in a separate process so no GPU work affects the timing:

    python examples/reproduce/vgg19_real_photos.py --data imagenetv2-matched-frequency-format-val
    python examples/reproduce/vgg19_real_photos.py --latency

The comparison reports, for the published model and the same model with the flagged channels cut out:
top-1 and top-5 accuracy, how many answers differ, the largest logit difference, and the largest output
any flagged channel gives. It repeats that on inputs far from normal photos. As a control, it then
switches off the same number of live channels, to show that the test can fail. Needs torchvision.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torchvision
from prune_vgg19 import dead_channels, macs, prune
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

WEIGHTS = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1


class FolderLabel:
    """ImageNetV2 folders are named by ImageNet class index, and ImageFolder sorts them as strings."""

    def __init__(self, classes: list[str]) -> None:
        self.classes = classes

    def __call__(self, idx: int) -> int:
        return int(self.classes[idx])


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def free_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


class FlaggedPeaks:
    """Tracks the largest post-ReLU output of every channel in the layers that have flagged channels."""

    def __init__(self, model: torch.nn.Module, dead: dict[str, list[int]]) -> None:
        self.dead, self.peaks = dead, {}
        modules = dict(model.named_modules())
        self.handles = [modules[name].register_forward_hook(self._hook(name)) for name in dead]

    def _hook(self, name: str):
        def hook(_module, _inputs, out):
            peak = out.detach().clamp(min=0).amax(dim=(0, 2, 3)).cpu()
            self.peaks[name] = peak if name not in self.peaks else torch.maximum(self.peaks[name], peak)
        return hook

    def reset(self) -> None:
        self.peaks = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def summary(self) -> dict:
        flagged = [float(self.peaks[n][c]) for n, chans in self.dead.items() for c in chans]
        return {"flagged_max_output": max(flagged), "flagged_channels_ever_positive": sum(v > 0 for v in flagged)}


@torch.no_grad()
def compare(model, cut, batches, device: str) -> dict:
    n = same = top1_a = top1_b = top5_a = top5_b = 0
    largest = 0.0
    for i, (x, y) in enumerate(batches):
        x, y = x.to(device), y.to(device)
        a, b = model(x), cut(x)
        largest = max(largest, (a - b).abs().max().item())
        same += (a.argmax(1) == b.argmax(1)).sum().item()
        top1_a += (a.argmax(1) == y).sum().item()
        top1_b += (b.argmax(1) == y).sum().item()
        top5_a += (a.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        top5_b += (b.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        n += len(x)
        if i % 10 == 9:
            free_cache(device)
        if i % 50 == 49:
            print(f"  {n} images: {same} same answers, largest logit difference {largest:.2e}", flush=True)
    return {"images": n, "same_top1": same, "largest_logit_difference": largest,
            "top1": [top1_a / n, top1_b / n], "top5": [top5_a / n, top5_b / n]}


def stress_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    """Inputs far from normal photos, in the network's normalized input space."""
    gen = torch.Generator().manual_seed(seed)
    mean = torch.tensor(WEIGHTS.transforms().mean).view(1, 3, 1, 1)
    std = torch.tensor(WEIGHTS.transforms().std).view(1, 3, 1, 1)
    solid = torch.stack([torch.full((3, 224, 224), float(v)) for v in torch.linspace(0, 1, 16)])
    yy, xx = torch.meshgrid(torch.arange(224), torch.arange(224), indexing="ij")
    boards = torch.stack([((xx // s + yy // s) % 2).float().expand(3, 224, 224) for s in (1, 2, 4, 8, 16, 32, 64, 112)])
    out = {"solid colours": (solid - mean) / std, "checkerboards": (boards - mean) / std,
           "uniform noise": (torch.rand(32, 3, 224, 224, generator=gen) - mean) / std}
    for sigma in (1.0, 3.0, 10.0, 100.0):
        out[f"gaussian sigma {sigma:g}"] = torch.randn(32, 3, 224, 224, generator=gen) * sigma
    return out


@torch.no_grad()
def predictions(model, batches, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    preds, labels = [], []
    for i, (x, y) in enumerate(batches):
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
        if i % 10 == 9:
            free_cache(device)
    return torch.cat(preds), torch.cat(labels)


def live_control(model, dead: dict[str, list[int]], batches, device: str, seeds: int = 3) -> dict:
    """Switch off the same number of randomly chosen live channels in each layer."""
    base, labels = predictions(model, batches, device)
    modules, runs = dict(model.named_modules()), []
    for seed in range(seeds):
        rng, handles = random.Random(seed), []
        for name, chans in dead.items():
            bn = modules[name]
            live = [c for c in range(bn.num_features) if c not in set(chans)]
            mask = torch.ones(bn.num_features, device=device)
            mask[rng.sample(live, min(len(chans), len(live)))] = 0

            def hook(_module, _inputs, out, mask=mask):
                return out.clamp(min=0) * mask[None, :, None, None] + out.clamp(max=0)
            handles.append(bn.register_forward_hook(hook))
        preds, _ = predictions(model, batches, device)
        for handle in handles:
            handle.remove()
        runs.append({"seed": seed, "answers_changed": int((preds != base).sum()),
                     "top1": float((preds == labels).float().mean())})
    return {"images": len(base), "top1_before": float((base == labels).float().mean()), "runs": runs}


@torch.no_grad()
def paired_latency(model, cut, x: torch.Tensor, rounds: int) -> dict:
    """Alternate which model goes first each round, so drift and heat hit both equally."""
    model(x), cut(x)
    times_a, times_b, gains = [], [], []
    for r in range(rounds):
        spent = {}
        for key, net in ((("a", model), ("b", cut)) if r % 2 == 0 else (("b", cut), ("a", model))):
            start = time.perf_counter()
            net(x)
            spent[key] = time.perf_counter() - start
        times_a.append(spent["a"])
        times_b.append(spent["b"])
        gains.append(1 - spent["b"] / spent["a"])
    return {"batch": len(x), "rounds": rounds, "rounds_cut_was_faster": sum(g > 0 for g in gains),
            "median_ms": [statistics.median(times_a) * 1000, statistics.median(times_b) * 1000],
            "median_speedup": statistics.median(gains)}


def latency_only(model, cut, rounds: int) -> list[dict]:
    x = torch.randn(16, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    runs = []
    for threads in (4, 1):
        torch.set_num_threads(threads)
        for batch in (16, 1):
            run = paired_latency(model, cut, x[:batch], rounds)
            runs.append({"threads": threads, **run})
            print(runs[-1], flush=True)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", help="ImageNetV2 folder with one subfolder per class index")
    ap.add_argument("--latency", action="store_true", help="only measure CPU speed; touches no GPU and no data")
    ap.add_argument("--rounds", type=int, default=7, help="paired timing rounds per setting")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only this many photos, chosen with a fixed seed")
    ap.add_argument("--control-images", type=int, default=500)
    ap.add_argument("--out", help="also write the results to this JSON file")
    a = ap.parse_args()
    if not a.latency and not a.data:
        ap.error("pass --data DIR, or --latency")

    model = torchvision.models.vgg19_bn(weights=WEIGHTS).eval()
    dead = {name: chans for name, chans in dead_channels(model).items() if chans}
    cut = prune(model, dead)
    gmacs = macs(model) / 1e9, macs(cut) / 1e9
    params = [sum(p.numel() for p in net.parameters()) / 1e6 for net in (model, cut)]
    result = {"flagged_channels": sum(map(len, dead.values())), "gmacs": gmacs, "params_millions": params,
              "compute_saving": 1 - gmacs[1] / gmacs[0], "torch": torch.__version__}
    print(f"flagged {result['flagged_channels']} channels; compute {gmacs[0]:.2f} -> {gmacs[1]:.2f} GMACs "
          f"({result['compute_saving']:.1%} less); parameters {params[0]:.1f}M -> {params[1]:.1f}M", flush=True)

    if a.latency:
        result["cpu_latency"] = latency_only(model, cut, a.rounds)
    else:
        device = pick_device()
        if device == "mps":
            torch.mps.set_per_process_memory_fraction(0.6)   # keep an 8 GB laptop from swapping
        result["device"] = device
        data = ImageFolder(a.data, transform=WEIGHTS.transforms())
        assert len(data.classes) == 1000, f"expected 1000 class folders, found {len(data.classes)}"
        data.target_transform = FolderLabel(data.classes)
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(0)).tolist()
        photos = Subset(data, order[:a.limit]) if a.limit else data
        model.to(device)
        cut.to(device)
        peaks = FlaggedPeaks(model, dead)

        loader = DataLoader(photos, batch_size=a.batch, num_workers=2)
        result["photos"] = {**compare(model, cut, loader, device), **peaks.summary()}
        print(json.dumps(result["photos"], indent=1), flush=True)

        result["stress"] = {}
        for name, x in stress_inputs().items():
            peaks.reset()
            batches = [(part, torch.zeros(len(part), dtype=torch.long)) for part in x.split(a.batch)]
            row = compare(model, cut, batches, device)
            result["stress"][name] = {"images": row["images"], "same_top1": row["same_top1"],
                                      "largest_logit_difference": row["largest_logit_difference"], **peaks.summary()}
            print(name, result["stress"][name], flush=True)
        peaks.close()

        cut.cpu()
        free_cache(device)
        control = Subset(data, order[:a.control_images])
        result["live_control"] = live_control(model, dead, DataLoader(control, batch_size=a.batch, num_workers=2), device)
        print(result["live_control"], flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
