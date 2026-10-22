"""Channel repair: re-initialise stuck channels of a Conv/Linear layer and its BatchNorm.

Repair is a DISCRETE modification with no enforced bound on its size: a dead filter is replaced
by a fresh Kaiming-normal one, whatever that changes. The magnitude is measured and logged
(`delta_norm`, `relative_change` = ||dW|| / ||W|| of the layer's weight), never capped. The
record declares this with `modification_class: "discrete"` and `bound: None`.

Determinism: with a `torch.Generator` from `repair_generator(seed, step, layer)` the new filters
depend only on (seed, step, layer), so every process of a distributed run writes the same values.
Without one, they come from torch's global RNG, as in ormas-monitor 0.1.

Also here: `unwrap_optimizer` / `find_optimizer`, which reach the real torch optimizer through
framework wrappers, and the optimizer side of the certificate: which optimizers it covers
(`optimizer_family`), whether their weight decay can push a closed shift above zero
(`decay_reason`, `sgd_decay_keeps_sign`), and the per-channel "can the optimizer still move
this?" test (`moving_mask`).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

FIRST_MOMENT = ("exp_avg", "momentum_buffer", "grad_avg")   # optimizer state that moves weights
# Optimizers the certificate covers, by exact type (a subclass may change the update rule). With a
# zero gradient and zero first moment for a channel, each moves its parameters only by weight decay.
SGD_TYPES: tuple[type, ...] = (torch.optim.SGD,)
ADAM_TYPES: tuple[type, ...] = (torch.optim.Adam, torch.optim.AdamW, torch.optim.NAdam, torch.optim.RAdam,
                                torch.optim.Adamax)
_MOMENTUM_KEYS = {"sgd": ("momentum_buffer",), "adam": ("exp_avg",)}
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


def find_optimizer(optimizer: Any) -> torch.optim.Optimizer | None:
    """The torch optimizer inside framework wrappers, `optimizer` itself, or None when there is
    none (None given, or an object such as a DeepSpeed engine optimizer that is not one).

    Follows `.optimizer` / `._optimizer` while they lead to another torch optimizer:
    Lightning's `LightningOptimizer` and Fabric's `_FabricOptimizer` (both `.optimizer`, a
    property over `._optimizer`) and accelerate's `AcceleratedOptimizer` (`.optimizer`).
    Checked against lightning 2.6 and accelerate 1.15. Never raises.
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
    return optimizer if isinstance(optimizer, torch.optim.Optimizer) else None


def unwrap_optimizer(optimizer: Any) -> torch.optim.Optimizer | None:
    """Like `find_optimizer`, but a non-None object with no torch optimizer inside is a TypeError.
    Returns None for None."""
    found = find_optimizer(optimizer)
    if optimizer is not None and found is None:
        raise TypeError(f"expected a torch.optim.Optimizer (or a wrapper of one), got {type(optimizer).__name__}")
    return found


def optimizer_family(optimizer: torch.optim.Optimizer | None, opt_in: Sequence[str] = ()) -> str | None:
    """"sgd", "adam" (Adam, AdamW, NAdam, RAdam, Adamax), "opt-in" (type name or qualified name
    listed in `opt_in`, i.e. MonitorConfig.certify_optimizers), or None: not covered."""
    if optimizer is None:
        return None
    t = type(optimizer)
    if t in SGD_TYPES:
        return "sgd"
    if t in ADAM_TYPES:
        return "adam"
    if t.__name__ in opt_in or f"{t.__module__}.{t.__qualname__}" in opt_in:
        return "opt-in"
    return None


def sgd_decay_keeps_sign(lr: float, weight_decay: float, momentum: float = 0.0, dampening: float = 0.0,
                         nesterov: bool = False) -> bool:
    """With every gradient exactly zero and a zero (or not yet created) momentum buffer, can
    torch.optim.SGD's weight decay alone ever move a parameter p <= 0 above 0?  False if it can.

    The update is linear in (p, buffer), so p_0 = 1 decides for every p_0 > 0 (and p_0 < 0 by
    symmetry); p_0 = 0 stays 0. With a = lr * weight_decay and k = (1 - dampening) * a, the state
    (p_t, lr * buffer_{t-1}) follows a 2x2 matrix M. From a state (p_s, u_s), p_t keeps its sign
    for every t >= s iff M has two distinct positive real eigenvalues l1 > l2, p_s >= 0, and the
    l1-coefficient A = (p_{s+1} - l2 p_s) / (l1 - l2) is positive. Complex or non-positive
    eigenvalues oscillate, so they are refused, as is the borderline. Both starts torch can be
    in are checked: no buffer yet (the first step stores a * p_0, undampened) and a zero buffer.
    Without momentum p_t = (1 - a)^t p_0.
    """
    a = lr * weight_decay
    if a == 0:
        return True
    if momentum == 0:
        return a < 1                  # strict: at a == 1 rounding can leave p just above 0
    mu, k = momentum, (1 - dampening) * a
    if nesterov:
        m11, m12, first = 1 - a - mu * k, -mu * mu, 1 - a * (1 + mu)
    else:
        m11, m12, first = 1 - k, -mu, 1 - a
    m21, m22 = k, mu
    tr, det = m11 + m22, m11 * m22 - m12 * m21
    disc = tr * tr - 4 * det
    if disc <= 0 or tr <= 0 or det <= 0:
        return False
    l2 = (tr - math.sqrt(disc)) / 2
    for p, u in ((first, a), (1.0, 0.0)):          # after a buffer-creating step; from a zero buffer
        if p < 0 or (m11 * p + m12 * u) - l2 * p <= 0:
            return False
    return True


def decay_reason(optimizer: torch.optim.Optimizer, family: str, group: dict) -> str | None:
    """Why this param group's weight decay could reopen a closed channel, or None if it cannot.

    Applied to the groups holding the shift (BatchNorm beta, or the producer bias without a
    BatchNorm) and gamma. Decay of an exactly-zero filter, bias or gamma is exactly zero, so only
    the shift matters: it must never cross from <= 0 to > 0 under decay alone.
      * Adam family: coupled L2 (weight_decay > 0 without decoupled_weight_decay) feeds
        wd * beta into the moments, and the normalised step overshoots zero: refused. Decoupled
        decay multiplies by (1 - lr * wd), which keeps the sign while lr * wd < 1.
      * SGD: decay without momentum multiplies by (1 - lr * wd). With momentum the certificate
        already requires a zero momentum buffer, so the shift starts from rest and follows a
        heavy-ball recurrence on the decay term alone; `sgd_decay_keeps_sign` checks it never
        overshoots zero (it holds for usual settings, e.g. lr * wd < (1 - sqrt(momentum))^2).
      * opt-in optimizers: only decoupled decay (lr * wd < 1) is accepted.
    """
    wd = float(group.get("weight_decay", 0) or 0)
    if wd == 0:
        return None
    lr = float(group.get("lr", 0))
    decoupled = type(optimizer) is torch.optim.AdamW or bool(group.get("decoupled_weight_decay", False))
    if family == "sgd":
        if sgd_decay_keeps_sign(lr, wd, float(group.get("momentum", 0) or 0),
                                float(group.get("dampening", 0) or 0), bool(group.get("nesterov", False))):
            return None
        return (f"SGD weight decay {wd:g} with lr {lr:g} and momentum {group.get('momentum', 0)} can "
                "push the BatchNorm shift past zero")
    if not decoupled:
        return (f"{type(optimizer).__name__} applies weight decay {wd:g} as coupled L2 on the "
                "BatchNorm shift, which can push it past zero (use AdamW or decoupled_weight_decay=True)")
    if lr * wd >= 1:
        return f"decoupled weight decay with lr * weight_decay = {lr * wd:g} >= 1 can flip the shift's sign"
    return None


def moving_mask(optimizer: torch.optim.Optimizer, family: str, params: Sequence[torch.Tensor | None],
                n: int, device: torch.device) -> torch.Tensor:
    """Per output channel, on `device` (no host read): does the optimizer hold nonzero (or
    non-finite) first-moment state for any of `params`?  For "sgd" that is `momentum_buffer`, for
    "adam" `exp_avg`; for "opt-in" every state tensor shaped like the parameter counts."""
    keys = _MOMENTUM_KEYS.get(family)
    moving = torch.zeros(n, dtype=torch.bool, device=device)
    for p in params:
        if p is None:
            continue
        for key, v in optimizer.state.get(p, {}).items():
            if (keys is None or key in keys) and isinstance(v, torch.Tensor) and v.shape == p.shape:
                moving |= v.detach().reshape(n, -1).ne(0).any(dim=1).to(device)
    return moving


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
