"""PyTorch Lightning integration: one callback, no change to the LightningModule.

    from ormas_monitor.integrations.lightning import OrmasMonitorCallback
    trainer = lightning.Trainer(callbacks=[OrmasMonitorCallback(repair=True)])
    trainer.fit(model, loader)

Hooks used (Lightning 2.6 source, ``plugins/precision/precision.py``):

* ``on_fit_start``: optimizers exist by then (``strategy.setup`` runs first); the `Monitor` is
  created on ``trainer.lightning_module``, the plain torch optimizer, and the precision plugin's
  GradScaler when 16-bit AMP uses one.
* ``on_before_backward(loss)``: called with the exact tensor about to be backpropagated, in
  automatic and manual optimization. Lightning has already divided it by
  ``accumulate_grad_batches``; under 16-bit AMP it is also multiplied by the GradScaler scale,
  which this callback divides back out.
* ``on_before_optimizer_step(optimizer)``: runs after the closure (forward + backward), after
  AMP's ``unscale_``, before gradient clipping and ``optimizer.step()``, and only on steps where
  the optimizer actually steps, so gradient accumulation gives one monitor step per optimizer step.
* ``on_fit_end`` / ``on_exception``: close the monitor (``run_end`` and reports). Closing never
  raises: at fit end a failure is logged and kept in ``callback.close_error``; during an
  exception the monitor prints it to stderr and the training exception propagates unchanged.

If the trainer's optimizer holds no torch optimizer over the model's own parameters (DeepSpeed,
custom wrappers), the monitor runs without it: no repair and no certificate, one line says so.

The loss fed to the monitor is the sum of every loss backpropagated since the previous optimizer
step, i.e. exactly the objective whose gradient is in ``.grad`` (with accumulation, the mean of the
micro-batch losses).
"""
from __future__ import annotations

import os
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl    # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError('ormas_monitor.integrations.lightning needs Lightning: '
                          'pip install "ormas-monitor[lightning]"') from e

from ..monitor import Monitor
from ._common import LossWindow, check_monitor_kwargs, safe_close, unwrap_optimizer, usable_optimizer, \
    without_optimizer
from .sinks import TensorBoardSink, WandbSink


class OrmasMonitorCallback(pl.Callback):
    """Run an ormas-monitor `Monitor` inside ``Trainer.fit``.

    Args:
        optimizer_index: which of ``trainer.optimizers`` the monitor follows (and repairs). The
            monitor steps only when that optimizer steps.
        to_loggers: also send metrics and events to the trainer's own ``TensorBoardLogger`` and
            ``WandbLogger`` experiments (other loggers are ignored). Off by default.
        write_report: write the HTML reports when the monitor closes at fit end.
        **monitor_kwargs: passed to `Monitor` (``repair``, ``check_every``, ``sinks``, ...).
            ``out`` defaults to ``<trainer.log_dir>/ormas_run``.

    ``callback.monitor`` is the current (after fit: the closed) monitor; ``callback.report`` is
    the main report path written at fit end; ``callback.close_error`` is the error that stopped
    the record or reports from being finished, or None.
    """

    def __init__(self, *, optimizer_index: int = 0, to_loggers: bool = False,
                 write_report: bool = True, **monitor_kwargs: Any) -> None:
        super().__init__()
        self.write_report = write_report
        self.monitor_kwargs = check_monitor_kwargs(monitor_kwargs)
        self.optimizer_index = optimizer_index
        self.to_loggers = to_loggers
        self.monitor: Monitor | None = None
        self.report: Any = None
        self.close_error: BaseException | None = None
        self._active = False
        self._target: Any = None                                # the followed trainer optimizer
        self._optimizer: torch.optim.Optimizer | None = None    # the torch optimizer inside it, if usable
        self._losses = LossWindow()
        self._sinks: list[Any] = []

    # ------------------------------------------------------------------ Lightning hooks
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._close()
        optimizers = trainer.optimizers
        kwargs = dict(self.monitor_kwargs)
        self._target, self._optimizer = None, None
        if optimizers and self.optimizer_index >= len(optimizers):
            raise ValueError(f"optimizer_index={self.optimizer_index} but the trainer has "
                             f"{len(optimizers)} optimizer(s)")
        if optimizers:
            self._target = optimizers[self.optimizer_index]
            self._optimizer, reason = usable_optimizer(self._target, trainer.lightning_module)
            if self._optimizer is None:
                kwargs = without_optimizer(kwargs, reason or "", kwargs.get("verbose", True))
        if "out" not in kwargs:
            kwargs["out"] = os.path.join(trainer.log_dir or trainer.default_root_dir, "ormas_run")
        self._sinks = list(kwargs.get("sinks", ()))
        if self.to_loggers:
            self._sinks += _logger_sinks(trainer)
        kwargs["sinks"] = self._sinks
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        if scaler is not None and self._optimizer is not None:
            kwargs.setdefault("scaler", scaler)
        self._losses = LossWindow()
        self.close_error = None
        self.monitor = Monitor(trainer.lightning_module, self._optimizer, **kwargs)
        self._active = True

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           loss: torch.Tensor) -> None:
        if self._active:
            self._losses.add(loss)

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                 optimizer: Any) -> None:
        mon = self.monitor
        if mon is None or not self._active:
            return
        if not self._follows(optimizer):
            return
        loss = self._losses.take(mean=False, keep_graph=mon.step_count == 0)
        if loss is not None:
            mon.note_loss(_unscale(trainer, loss))
        mon.step()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._close()

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                     exception: BaseException) -> None:
        self._close()

    # ------------------------------------------------------------------ internals
    def _follows(self, optimizer: Any) -> bool:
        """Is `optimizer` (as passed to the hook) the one this monitor follows?"""
        if self._target is None or optimizer is self._target:
            return True
        inner = unwrap_optimizer(optimizer)
        return inner is not None and inner is unwrap_optimizer(self._target)

    def _close(self) -> None:
        if not self._active or self.monitor is None:
            return
        self._active = False
        self.report, self.close_error = safe_close(self.monitor, self._sinks, self.write_report)


def _unscale(trainer: "pl.Trainer", loss: torch.Tensor) -> torch.Tensor:
    """Undo AMP loss scaling. The scale is constant within an accumulation window (``update()``
    runs after ``optimizer.step()``), so dividing the window sum once is exact."""
    scaler = getattr(getattr(trainer, "precision_plugin", None), "scaler", None)
    if scaler is None or not scaler.is_enabled():
        return loss
    scale = getattr(scaler, "_scale", None)     # device tensor: avoids a host sync per step
    if isinstance(scale, torch.Tensor):
        return loss / scale.to(loss.device, loss.dtype)
    return loss / scaler.get_scale()


def _logger_sinks(trainer: "pl.Trainer") -> list[Any]:
    """Sinks writing into the trainer's own TensorBoard / W&B experiments."""
    out: list[Any] = []
    for lg in trainer.loggers:
        if isinstance(lg, pl.loggers.TensorBoardLogger):
            out.append(TensorBoardSink(writer=lg.experiment))
        elif isinstance(lg, pl.loggers.WandbLogger):
            out.append(WandbSink(run=lg.experiment))
    return out
