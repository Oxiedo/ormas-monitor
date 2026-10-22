"""Shared plumbing for the framework callbacks: finding the torch optimizer that steps the model,
the per-step loss window, validation of the keyword arguments forwarded to `Monitor`, and closing
a monitor without ever raising over the user's own training error."""
from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from ..monitor import Monitor
from .sinks import close_sinks

log = logging.getLogger("ormas_monitor.integrations")

_RESERVED = ("model", "optimizer")
_OPTIMIZER_FEATURES = ("repair", "scaler")      # Monitor options that need a usable optimizer


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


def unwrap_optimizer(opt: Any) -> torch.optim.Optimizer | None:
    """The plain torch optimizer under Lightning's ``LightningOptimizer`` (``_optimizer``) or
    accelerate's ``AcceleratedOptimizer`` (``optimizer``), so repair edits the state that steps.

    Returns None when there is no torch optimizer inside `opt` (DeepSpeed engines, custom
    wrappers): an expected absence, not an error.
    """
    for _ in range(8):
        inner = getattr(opt, "optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer):
            inner = getattr(opt, "_optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer) or inner is opt:
            break
        opt = inner
    return opt if isinstance(opt, torch.optim.Optimizer) else None


def usable_optimizer(opt: Any, model: nn.Module) -> tuple[torch.optim.Optimizer | None, str | None]:
    """``(optimizer, None)`` if a torch optimizer inside `opt` steps `model`'s own parameters,
    else ``(None, reason)``.

    Repair resets optimizer state per model parameter and the certificate reads that state, so an
    optimizer that steps copies (DeepSpeed ZeRO's flattened fp32 partitions, fp16 master weights)
    or that cannot be reached at all is treated as absent.
    """
    inner = unwrap_optimizer(opt)
    name = type(opt).__name__
    if inner is None:
        return None, f"{name} is not a torch.optim.Optimizer and wraps none"
    try:
        stepped = {id(p) for g in inner.param_groups for p in g["params"]}
    except (AttributeError, KeyError, TypeError):
        return None, f"cannot read the parameter groups of {name}"
    if not any(id(p) in stepped for p in model.parameters() if p.requires_grad):
        return None, (f"the optimizer inside {name} does not hold the model's parameters "
                      f"(it steps copies, e.g. DeepSpeed's flattened partitions)")
    return inner, None


def without_optimizer(kwargs: dict[str, Any], reason: str, verbose: bool) -> dict[str, Any]:
    """Monitor options for a run with no usable optimizer: repair and the GradScaler are dropped,
    and one line says so (printed when verbose, else logged as a warning)."""
    dropped = [k for k in _OPTIMIZER_FEATURES if kwargs.get(k)]
    kwargs = {k: v for k, v in kwargs.items() if k not in _OPTIMIZER_FEATURES}
    msg = (f"no usable torch optimizer ({reason}); monitoring without it: no repair, no "
           f"dead-channel certificate (it needs the optimizer's momentum); alarms and the record still work"
           + (f"; ignored {', '.join(dropped)}" if dropped else ""))
    if verbose:
        print(f"[ormas-monitor] {msg}", flush=True)
    else:
        log.warning(msg)
    return kwargs


def safe_close(monitor: Monitor, sinks: Sequence[Any], report: bool) -> tuple[Any, BaseException | None]:
    """Close `monitor` and `sinks`; never raises.

    Called from framework end/exception hooks, where an error here would replace the user's own
    training exception (or fail a run whose training finished). While an exception propagates,
    `Monitor.close` itself prints a failure and returns None; otherwise a failure is logged with
    its traceback and returned as ``(None, error)``. On success: ``(report_path, None)``.
    """
    path, error = None, None
    try:
        path = monitor.close(report=report)
    except Exception as e:
        error = e
        log.error("ormas-monitor could not finish the run record or reports in %s (training itself was "
                  "not affected); whatever was written is kept, and `ormas-monitor report %s` rebuilds "
                  "the reports", monitor.out, monitor.out, exc_info=True)
    try:
        close_sinks(sinks)
    except Exception as e:
        error = error or e
        log.error("ormas-monitor: closing a sink failed", exc_info=True)
    return path, error


class LossWindow:
    """Accumulates the losses of one optimizer step's micro-batches without leaving the device.

    `add` stores a detached tensor sum (no host sync); `take` returns the window's sum or mean
    and resets. ``None`` from `take` means no loss was seen, which is not the same as zero.
    ``take(..., keep_graph=True)`` returns the same value still attached to the last loss's
    autograd graph: the monitor's first check walks that graph to confirm which ReLU consumes
    each layer, which the certificate requires.
    """

    def __init__(self) -> None:
        self._total: torch.Tensor | None = None
        self._last: torch.Tensor | None = None
        self._n = 0

    def add(self, loss: torch.Tensor | float) -> None:
        t = loss.detach().float() if isinstance(loss, torch.Tensor) else torch.tensor(float(loss))
        if t.numel() != 1:
            t = t.mean()
        self._total = t if self._total is None else self._total + t
        self._last = loss if isinstance(loss, torch.Tensor) and loss.grad_fn is not None else None
        self._n += 1

    def take(self, mean: bool, keep_graph: bool = False) -> torch.Tensor | None:
        total, last, n = self._total, self._last, self._n
        self._total, self._last, self._n = None, None, 0
        if total is None:
            return None
        value = total / n if mean else total
        if keep_graph and last is not None:
            last = last.float().mean()
            value = value + (last - last.detach())     # adds exactly 0 and keeps last.grad_fn reachable
        return value
