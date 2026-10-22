"""Health check for a saved model file: no training, no data, weights only.

    ormas-monitor check model.pt                     # one checkpoint
    ormas-monitor check after.pt --against before.pt # what changed between two

Reads .pt/.pth/.bin (PyTorch) and .safetensors files. By default PyTorch files are loaded
with weights_only=True, so a checkpoint cannot run code on your machine.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from pathlib import Path

import torch

from .structure import scan_channels

BN_SUFFIXES = ("weight", "bias", "running_mean", "running_var")
WRAPPER_KEYS = ("state_dict", "model_state_dict", "model", "net", "module", "ema")


class CheckpointError(Exception):
    pass


def load_tensors(path: str | Path, trust: bool = False) -> "OrderedDict[str, torch.Tensor]":
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"{path} does not exist")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise CheckpointError("reading .safetensors needs `pip install safetensors`") from e
        obj = load_file(str(path))
    else:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            if not trust:
                raise CheckpointError(
                    f"{path.name} contains Python objects, not just weights, and loading it could run "
                    "code. Re-run with --trust only if you trust where the file came from, or save "
                    "model.state_dict() instead.") from e
            obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    for _ in range(3):
        if isinstance(obj, dict):
            inner = next((obj[k] for k in WRAPPER_KEYS if k in obj
                          and isinstance(obj[k], (dict, torch.nn.Module))), None)
            if inner is None:
                break
            obj = inner.state_dict() if isinstance(inner, torch.nn.Module) else inner
    if not isinstance(obj, dict):
        raise CheckpointError(f"{path.name} does not hold a state dict")
    tensors = OrderedDict((re.sub(r"^(module\.|_orig_mod\.)", "", k), v.detach())
                          for k, v in obj.items() if isinstance(v, torch.Tensor))
    if not tensors:
        raise CheckpointError(f"no tensors found in {path.name}")
    return tensors


def _prefixes(tensors) -> "OrderedDict[str, dict[str, torch.Tensor]]":
    groups: "OrderedDict[str, dict[str, torch.Tensor]]" = OrderedDict()
    for key, t in tensors.items():
        prefix, _, suffix = key.rpartition(".")
        groups.setdefault(prefix, {})[suffix] = t
    return groups


def _producer_for(bn: str, size: int, groups, order: list[str]) -> str | None:
    """Find the Conv/Linear feeding BatchNorm `bn`: by name first, then by file order."""
    def ok(p):
        w = groups.get(p, {}).get("weight")
        return (p != bn and w is not None and w.dim() >= 2 and w.shape[0] == size
                and "running_mean" not in groups[p])
    parent, _, child = bn.rpartition(".")
    join = (parent + ".") if parent else ""
    candidates = []
    m = re.fullmatch(r"(.*?)(bn|norm|batchnorm|batch_norm)(\d*)", child)
    if m:
        for stem in ("conv", "fc", "linear", "proj"):
            candidates.append(join + m.group(1) + stem + m.group(3))
    if child.isdigit():
        candidates.append(join + str(int(child) - 1))
    for c in candidates:
        if ok(c):
            return c
    i = order.index(bn)
    for p in reversed(order[:i]):
        if ok(p):
            return p
        if "running_mean" in groups[p]:
            break
    return None


def check(tensors, relu_after: set[str] | None = None) -> dict:
    """Weights-only health check. `relu_after`, if known (see structure.relu_followers),
    limits dead/dormant findings to BatchNorms that feed a ReLU directly."""
    groups = _prefixes(tensors)
    order = list(groups)
    findings, layers = [], []

    nonfinite = [k for k, t in tensors.items() if t.is_floating_point() and not torch.isfinite(t).all()]
    if nonfinite:
        findings.append({"severity": "critical", "kind": "nonfinite",
                         "title": f"{len(nonfinite)} tensors contain NaN or Inf",
                         "detail": ", ".join(nonfinite[:12]), "tensors": nonfinite})
    zero = [k for k, t in tensors.items() if t.is_floating_point() and t.numel() > 1
            and not k.endswith(("running_mean", "bias")) and t.abs().max() == 0]
    if zero:
        findings.append({"severity": "warning", "kind": "zero_tensor",
                         "title": f"{len(zero)} weight tensors are entirely zero",
                         "detail": ", ".join(zero[:12]), "tensors": zero})

    paired = set()
    for bn, g in groups.items():
        if not all(s in g for s in BN_SUFFIXES):
            continue
        size = g["weight"].shape[0]
        prod = _producer_for(bn, size, groups, order)
        if relu_after is not None:
            feeds = bn in relu_after
        else:   # wiring unknown: skip-connection branches never feed a ReLU directly
            feeds = not re.search(r"downsample|shortcut|skip|residual", bn)
        w = groups[prod]["weight"] if prod else None
        scan = scan_channels(w, groups[prod].get("bias") if prod else None, g["weight"], g["bias"])
        if prod:
            paired.add(prod)
        var = g["running_var"]
        collapsed = int((var < 1e-12).sum())
        # Dormant: the shift sits more than 3 scales below zero, so with roughly normal
        # normalised inputs the channel is positive on under ~0.13% of positions.
        gamma, beta = g["weight"].float(), g["bias"].float()
        exact = set(scan.certified) | set(scan.flat) | set(scan.decayed)
        dormant = [c for c in ((beta + 3 * gamma.abs()) <= 0).nonzero().flatten().tolist() if c not in exact]
        if not feeds:                      # output joins a skip connection first: no ReLU argument
            scan.flat, scan.certified, scan.decayed, dormant = sorted(set(scan.flat) | set(scan.certified)), [], [], []
        layers.append({"layer": prod or "(none found)", "batchnorm": bn, "channels": size,
                       "certified": scan.certified, "flat": scan.flat, "decayed": scan.decayed,
                       "dormant": dormant,
                       "collapsed_var": collapsed})
        if scan.certified:
            cond = "" if relu_after is not None else f" if a ReLU follows {bn}"
            findings.append({
                "severity": "warning", "kind": "dead_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.certified)}/{size} channels of {prod or bn} are dead for good{cond}",
                "detail": "Their incoming filter or BatchNorm scale is exactly zero and the BatchNorm "
                          "shift is <= 0, so the output is a constant <= 0. A ReLU then closes for "
                          "every input and no gradient can ever reach them again (Theorem 2). "
                          "Does not apply where the BatchNorm output is added to a skip connection "
                          "before the ReLU.",
                "channels": scan.certified})
        if scan.decayed:
            cond = "" if relu_after is not None else f" (if a ReLU follows {bn})"
            findings.append({
                "severity": "warning", "kind": "decayed_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.decayed)}/{size} channels of {prod or bn} are effectively dead{cond}",
                "detail": "Their filter or BatchNorm scale has shrunk to under 1e-5 of the layer's typical "
                          "size and their shift is <= 0, so they output ~0 for every input. This is what a "
                          "dead channel looks like after weight decay has been shrinking it for a while.",
                "channels": scan.decayed})
        if scan.flat:
            findings.append({
                "severity": "info", "kind": "flat_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(scan.flat)}/{size} channels of {prod or bn} output a constant",
                "detail": "Their incoming filter or BatchNorm scale is exactly zero, so they output "
                          "the same value for every input and carry no information.",
                "channels": scan.flat})
        if dormant:
            findings.append({
                "severity": "info", "kind": "dormant_channels", "layer": prod, "batchnorm": bn,
                "title": f"{len(dormant)}/{size} channels of {prod or bn} are probably dormant"
                         + ("" if relu_after is not None else f" (if a ReLU follows {bn})"),
                "detail": "BatchNorm shift <= -3 x |scale|: if a ReLU follows, these fire on under "
                          "~0.13% of positions when normalised inputs are roughly normal. An estimate, "
                          "not a proof.", "channels": dormant})
        if collapsed:
            findings.append({
                "severity": "info", "kind": "collapsed_variance", "batchnorm": bn,
                "title": f"{collapsed}/{size} channels of {bn} never vary (running variance ~0)",
                "detail": "Whatever reaches these channels is the same for every input, so they carry no "
                          "information. Usually a dead filter upstream.",
                "channels": (var < 1e-12).nonzero().flatten().tolist()})

    for prefix, g in groups.items():
        w = g.get("weight")
        if prefix in paired or w is None or w.dim() < 2 or "running_mean" in g:
            continue
        scan = scan_channels(w, g.get("bias"), None, None)
        dead = scan.certified + scan.flat
        if dead and len(dead) < w.shape[0]:
            findings.append({
                "severity": "info", "kind": "zero_filters", "layer": prefix,
                "title": f"{len(dead)}/{w.shape[0]} output units of {prefix} have all-zero weights",
                "detail": "Each outputs only its bias. Followed by a ReLU, those with bias <= 0 "
                          "never fire.", "channels": dead})

    n_params = sum(t.numel() for t in tensors.values() if t.is_floating_point())
    order_rank = {"critical": 0, "warning": 1, "info": 2}
    kind_rank = ["nonfinite", "dead_channels", "decayed_channels", "zero_tensor"]
    findings.sort(key=lambda f: (order_rank[f["severity"]],
                                 kind_rank.index(f["kind"]) if f["kind"] in kind_rank else len(kind_rank)))
    return {"tensors": len(tensors), "parameters": n_params, "batchnorm_layers": len(layers),
            "findings": findings, "layers": layers}


def diff(new, old) -> dict:
    rows, frozen, added, removed, reshaped = [], [], [], [], []
    for k, t in new.items():
        if k not in old:
            added.append(k)
            continue
        o = old[k]
        if o.shape != t.shape:
            reshaped.append(k)
            continue
        if not t.is_floating_point():
            continue
        d = (t.float() - o.float())
        dn, on = float(d.norm()), float(o.float().norm())
        if dn == 0:
            frozen.append(k)
        rows.append({"tensor": k, "change": dn / on if on > 0 else (math.inf if dn else 0.0),
                     "abs_change": dn})
    removed = [k for k in old if k not in new]
    rows.sort(key=lambda r: -r["change"])
    return {"changed": rows, "unchanged": frozen, "added": added, "removed": removed,
            "reshaped": reshaped}


def arch_relu_followers(arch: str) -> set[str]:
    """Wiring for a known architecture, e.g. 'torchvision:resnet50' (needs torchvision)."""
    from .structure import relu_followers
    src, _, name = arch.partition(":")
    if src != "torchvision" or not name:
        raise CheckpointError("--arch takes torchvision:<model name>, e.g. torchvision:resnet50")
    import torchvision
    model = torchvision.models.get_model(name, weights=None)
    size = 299 if name.startswith("inception") else 224
    return relu_followers(model, torch.randn(1, 3, size, size))


def run(path, against=None, trust: bool = False, arch: str | None = None) -> dict:
    new = load_tensors(path, trust)
    follow = arch_relu_followers(arch) if arch else None
    result = {"file": str(path), **check(new, follow)}
    if against:
        old = load_tensors(against, trust)
        result["against"] = str(against)
        result["diff"] = diff(new, old)
        before = {(l["layer"], l["batchnorm"]): set(l["certified"]) for l in check(old, follow)["layers"]}
        newly = []
        for l in result["layers"]:
            gained = set(l["certified"]) - before.get((l["layer"], l["batchnorm"]), set())
            if gained:
                newly.append({"layer": l["layer"], "batchnorm": l["batchnorm"], "count": len(gained)})
        result["diff"]["newly_dead"] = newly
        if newly:
            result["findings"].insert(0, {
                "severity": "warning", "kind": "newly_dead",
                "title": f"{sum(n['count'] for n in newly)} channels died between the two checkpoints",
                "detail": ", ".join(f"{n['layer']}: {n['count']}" for n in newly)})
    return result
