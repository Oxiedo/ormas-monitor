"""Shared plumbing for the framework callbacks: optimizer unwrapping, the per-step loss window,
and validation of the keyword arguments forwarded to `Monitor`."""
from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch

from ..monitor import Monitor

_RESERVED = ("model", "optimizer")


def check_monitor_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Fail at callback construction, not mid-fit, if `kwargs` are not valid `Monitor` options."""
    bad = [k for k in _RESERVED if k in kwargs]
    if bad:
        raise TypeError(f"{', '.join(bad)} is supplied by the framework; do not pass it to the callback")
    try:
        inspect.signature(Monitor).bind_partial(None, **kwargs)
    except TypeError as e:
        raise TypeError(f"invalid Monitor option for the callback: {e}") from None
    return dict(kwargs)


def unwrap_optimizer(opt: Any) -> torch.optim.Optimizer:
    """The plain torch optimizer under Lightning's ``LightningOptimizer`` (``_optimizer``) or
    accelerate's ``AcceleratedOptimizer`` (``optimizer``), so repair edits the state that steps."""
    for _ in range(8):
        inner = getattr(opt, "optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer):
            inner = getattr(opt, "_optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer) or inner is opt:
            break
        opt = inner
    if not isinstance(opt, torch.optim.Optimizer):
        raise TypeError(f"cannot find a torch.optim.Optimizer inside {type(opt).__name__}")
    return opt


class LossWindow:
    """Accumulates the losses of one optimizer step's micro-batches without leaving the device.

    `add` stores a detached tensor sum (no host sync); `take` returns the window's sum or mean
    and resets. ``None`` from `take` means no loss was seen, which is not the same as zero.
    """

    def __init__(self) -> None:
        self._total: torch.Tensor | None = None
        self._n = 0

    def add(self, loss: torch.Tensor | float) -> None:
        t = loss.detach().float() if isinstance(loss, torch.Tensor) else torch.tensor(float(loss))
        if t.numel() != 1:
            t = t.mean()
        self._total = t if self._total is None else self._total + t
        self._n += 1

    def take(self, mean: bool) -> torch.Tensor | None:
        total, n = self._total, self._n
        self._total, self._n = None, 0
        if total is None:
            return None
        return total / n if mean else total
