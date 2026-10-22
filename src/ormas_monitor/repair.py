"""Channel repair: re-initialise stuck channels of a Conv/Linear layer and its BatchNorm.

Repair is a DISCRETE modification with no enforced bound on its size: a dead filter is replaced
by a fresh Kaiming-normal one, whatever that changes. The magnitude is measured and logged
(`delta_norm`, `relative_change` = ||dW|| / ||W|| of the layer's weight), never capped. The
record declares this with `modification_class: "discrete"` and `bound: None`.

Determinism: with a `torch.Generator` from `repair_generator(seed, step, layer)` the new filters
depend only on (seed, step, layer), so every process of a distributed run writes the same values.
Without one, they come from torch's global RNG, as in ormas-monitor 0.1.

Also here: `unwrap_optimizer`, which reaches the real torch optimizer through framework wrappers,
and `first_moment_mask`, the per-channel "can the optimizer still move this?" test that the
certificate needs.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

FIRST_MOMENT = ("exp_avg", "momentum_buffer", "grad_avg")   # optimizer state that moves weights
METHOD = ("Kaiming-normal filter, bias 0, BatchNorm scale 1 and shift 0, "
          "running stats reset, optimizer state for these channels zeroed")
MODIFICATION_CLASS = "discrete"


@dataclass(frozen=True)
class RepairResult:
    """What one repair changed. `relative_change` is None when the weight norm before was 0
    (the ratio is undefined), which is the usual case for a fully zeroed layer."""
    count: int
    optimizer_slices_reset: int
    delta_norm: float
    weight_norm_before: float
    weight_norm_after: float
    relative_change: float | None


def unwrap_optimizer(optimizer: Any) -> torch.optim.Optimizer | None:
    """The torch optimizer inside framework wrappers, or `optimizer` itself.

    Follows `.optimizer` / `._optimizer` while they lead to another torch optimizer:
    Lightning's `LightningOptimizer` and Fabric's `_FabricOptimizer` (both `.optimizer`, a
    property over `._optimizer`) and accelerate's `AcceleratedOptimizer` (`.optimizer`).
    Checked against lightning 2.6 and accelerate 1.15. Returns None for None.
    """
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        inner = None
        for attr in ("optimizer", "_optimizer"):
            cand = getattr(optimizer, attr, None)
            if isinstance(cand, torch.optim.Optimizer) and cand is not optimizer:
                inner = cand
                break
        if inner is None:
            break
        optimizer = inner
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"expected a torch.optim.Optimizer (or a wrapper of one), got {type(optimizer).__name__}")
    return optimizer


def repair_generator(seed: int, step: int, layer: str) -> torch.Generator:
    """A CPU generator whose state depends only on (seed, step, layer), the same in every process.

    Uses SHA-256, not Python's `hash()`, which is salted per process.
    """
    digest = hashlib.sha256(f"ormas-repair:{seed}:{step}:{layer}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))


def _channel_params(prod: nn.Module, bn: nn.Module | None) -> list[torch.Tensor]:
    params = [prod.weight, getattr(prod, "bias", None)]
    if bn is not None:
        params += [getattr(bn, "weight", None), getattr(bn, "bias", None)]
    return [p for p in params if p is not None]


def first_moment_mask(optimizer: torch.optim.Optimizer, prod: nn.Module,
                      bn: nn.Module | None) -> torch.Tensor:
    """Per output channel (CPU bool tensor): does the optimizer hold nonzero first-moment
    momentum (Adam `exp_avg`, SGD `momentum_buffer`, ...) for the layer or its BatchNorm?"""
    n = prod.weight.shape[0]
    moving = torch.zeros(n, dtype=torch.bool)
    for p in _channel_params(prod, bn):
        for key, v in optimizer.state.get(p, {}).items():
            if key in FIRST_MOMENT and isinstance(v, torch.Tensor) and v.shape == p.shape:
                moving |= (v.detach().reshape(n, -1).abs().amax(dim=1) > 0).cpu()
    return moving


def _kaiming_normal(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Kaiming-normal (fan_in, ReLU gain) on CPU from `generator`; same law as nn.init.kaiming_normal_."""
    fan_in = math.prod(shape[1:]) if len(shape) > 1 else 1
    std = math.sqrt(2.0) / math.sqrt(fan_in)
    return torch.randn(shape, generator=generator, dtype=torch.float32) * std


def repair_channels(prod: nn.Module, bn: nn.Module | None, idx: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    generator: torch.Generator | None = None) -> RepairResult:
    """Re-initialise output channels `idx` of `prod` (and of its BatchNorm `bn`), in place.

    Filter: Kaiming normal; bias 0; BatchNorm scale 1, shift 0, running mean 0, running var 1;
    every optimizer state tensor shaped like a touched parameter is zeroed for those channels.
    `generator` None draws from torch's global RNG on the weight's device (0.1 behaviour).
    """
    assert idx.dim() == 1 and idx.numel() > 0, "repair needs a non-empty 1-D index of channels"
    w = prod.weight
    dev_idx = idx.to(w.device)
    params = [w]
    with torch.no_grad():
        before = w.detach().float().clone()
        shape = (len(idx),) + tuple(w.shape[1:])
        if generator is None:
            fresh = torch.empty(shape, device=w.device, dtype=w.dtype)
            nn.init.kaiming_normal_(fresh, nonlinearity="relu")
        else:
            fresh = _kaiming_normal(shape, generator).to(device=w.device, dtype=w.dtype)
        w[dev_idx] = fresh
        if getattr(prod, "bias", None) is not None:
            prod.bias[dev_idx] = 0
            params.append(prod.bias)
        if bn is not None:
            if bn.weight is not None:
                bn.weight[dev_idx] = 1
                bn.bias[dev_idx] = 0
                params += [bn.weight, bn.bias]
            if bn.running_mean is not None:
                bn.running_mean[dev_idx] = 0
                bn.running_var[dev_idx] = 1
        after = w.detach().float()
        # One device read for the three norms (full precision, cast before the reduction).
        delta, nb, na = torch.stack([torch.linalg.vector_norm(after - before),
                                     torch.linalg.vector_norm(before),
                                     torch.linalg.vector_norm(after)]).tolist()
    reset = 0
    for p in params:
        for v in optimizer.state.get(p, {}).values():
            if isinstance(v, torch.Tensor) and v.shape == p.shape:
                v[dev_idx] = 0
                reset += 1
    return RepairResult(count=len(idx), optimizer_slices_reset=reset, delta_norm=delta,
                        weight_norm_before=nb, weight_norm_after=na,
                        relative_change=delta / nb if nb > 0 else None)
