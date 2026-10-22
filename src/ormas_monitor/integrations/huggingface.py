"""Hugging Face ``transformers.Trainer`` integration: one callback, no Trainer subclass.

    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      callbacks=[OrmasMonitorCallback(repair=True)])
    trainer.train()

Hooks used (transformers 4.57 ``Trainer._inner_training_loop``):

* ``on_train_begin``: the optimizer exists and has been through ``accelerator.prepare``; the
  `Monitor` is created on ``kwargs["model"]`` (``Trainer.model``, unwrapped) and the plain torch
  optimizer inside accelerate's ``AcceleratedOptimizer`` (plus its GradScaler under fp16, so the
  monitor knows whether gradients are still scaled: they are unless ``max_grad_norm`` clipping
  unscaled them).
* ``on_pre_optimizer_step``: runs only on gradient-sync steps (one per optimizer step, so gradient
  accumulation is handled), after every micro-batch's backward and **after gradient clipping**
  (``max_grad_norm``, default 1.0), immediately before ``optimizer.step()``.
* ``on_train_end``: close the monitor (``run_end`` and reports). The Trainer has no exception hook;
  if training raises, call ``callback.close()`` yourself.

Loss: callbacks never receive the training loss. This callback reads the ``loss`` the model itself
returns (``outputs["loss"]`` / ``outputs.loss`` / ``outputs[0]`` when labels are given) with a
forward hook active only in training mode, and feeds the monitor the effective loss of the
optimizer step: the mean of the micro-batch losses, or their sum when the Trainer passed
``num_items_in_batch`` (the model then already normalises by the whole step's token count).
If your ``compute_loss`` builds the loss outside the model, call ``callback.note_loss(loss)`` from
it, or the monitor runs without loss checks (its gradient and channel checks are unaffected).
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import torch
import torch.nn as nn

try:
    from transformers import TrainerCallback
except ImportError as e:
    raise ImportError('ormas_monitor.integrations.huggingface needs transformers: '
                      'pip install "ormas-monitor[huggingface]"') from e

from ..monitor import Monitor
from ._common import LossWindow, check_monitor_kwargs, unwrap_optimizer
from .sinks import MLflowSink, TensorBoardSink, WandbSink, close_sinks


def _output_loss(output: Any) -> torch.Tensor | None:
    """The loss a model returned, if it returned one."""
    if isinstance(output, Mapping):
        loss = output.get("loss")
    elif hasattr(output, "loss"):
        loss = output.loss
    elif isinstance(output, (tuple, list)) and output:
        first = output[0]     # HF convention: loss first when labels are given; logits are not scalar
        loss = first if isinstance(first, torch.Tensor) and first.numel() == 1 else None
    else:
        loss = None
    if isinstance(loss, torch.Tensor) and loss.numel() >= 1 and loss.dim() <= 1:
        return loss
    return None


class OrmasMonitorCallback(TrainerCallback):
    """Run an ormas-monitor `Monitor` inside ``Trainer.train``.

    Args:
        to_loggers: also send metrics and events to the trackers in ``args.report_to``:
            TensorBoard (a writer in ``args.logging_dir``), W&B (the active run) and MLflow (the
            active run). Off by default.
        write_report: write the HTML reports when the monitor closes at train end.
        **monitor_kwargs: passed to `Monitor`. ``out`` defaults to
            ``<args.output_dir>/ormas_run`` and ``name`` to ``args.run_name``.
    """

    def __init__(self, *, to_loggers: bool = False, write_report: bool = True,
                 **monitor_kwargs: Any) -> None:
        self.write_report = write_report
        self.monitor_kwargs = check_monitor_kwargs(monitor_kwargs)
        self.to_loggers = to_loggers
        self.monitor: Monitor | None = None
        self.report: Any = None
        self._active = False
        self._losses = LossWindow()
        self._normalised = False
        self._sinks: list[Any] = []
        self._handle: Any = None

    # ------------------------------------------------------------------ public
    def note_loss(self, loss: torch.Tensor | float) -> None:
        """Record a micro-batch loss yourself (for losses computed outside the model's forward)."""
        if self._active:
            self._losses.add(loss)

    def close(self) -> None:
        """Close the monitor and sinks; idempotent. Called automatically at train end."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if not self._active or self.monitor is None:
            return
        self._active = False
        try:
            self.report = self.monitor.close(report=self.write_report)
        finally:
            close_sinks(self._sinks)

    # ------------------------------------------------------------------ Trainer hooks
    def on_train_begin(self, args: Any, state: Any, control: Any, model: nn.Module | None = None,
                       optimizer: Any = None, **kwargs: Any) -> None:
        self.close()
        if model is None:
            raise RuntimeError("OrmasMonitorCallback: the Trainer passed no model to on_train_begin")
        opt = unwrap_optimizer(optimizer) if optimizer is not None else None
        mkw = dict(self.monitor_kwargs)
        mkw.setdefault("out", os.path.join(args.output_dir or ".", "ormas_run"))
        if getattr(args, "run_name", None) and "name" not in mkw:
            mkw["name"] = args.run_name
        self._sinks = list(mkw.get("sinks", ()))
        if self.to_loggers:
            self._sinks += _tracker_sinks(args)
        mkw["sinks"] = self._sinks
        scaler = getattr(optimizer, "scaler", None)      # AcceleratedOptimizer under fp16
        if scaler is not None and opt is not None:
            mkw.setdefault("scaler", scaler)
        self._losses = LossWindow()
        self.monitor = Monitor(model, opt, **mkw)
        self._handle = _register_loss_hook(model, self._on_forward)
        self._active = True

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        mon = self.monitor
        if mon is None or not self._active:
            return
        loss = self._losses.take(mean=not self._normalised)
        self._normalised = False
        if loss is not None:
            mon.note_loss(loss)
        mon.step()

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    def _on_forward(self, module: nn.Module, kwargs: Mapping[str, Any], output: Any) -> None:
        if not module.training or not self._active or not torch.is_grad_enabled():
            return
        loss = _output_loss(output)
        if loss is not None:
            self._losses.add(loss)
            if kwargs.get("num_items_in_batch") is not None:
                self._normalised = True


def _register_loss_hook(model: nn.Module, fn: Any) -> Any:
    """Forward hook that sees the call's kwargs (torch >= 2.0), else only the output."""
    try:
        return model.register_forward_hook(lambda m, a, kw, out: fn(m, kw, out), with_kwargs=True)
    except TypeError:
        return model.register_forward_hook(lambda m, a, out: fn(m, {}, out))


def _tracker_sinks(args: Any) -> list[Any]:
    """Sinks for the trackers named in ``args.report_to``."""
    report_to = args.report_to or []
    if isinstance(report_to, str):
        report_to = [report_to]
    out: list[Any] = []
    if "tensorboard" in report_to:
        out.append(TensorBoardSink(args.logging_dir))
    if "wandb" in report_to:
        out.append(WandbSink())
    if "mlflow" in report_to:
        out.append(MLflowSink())
    return out
