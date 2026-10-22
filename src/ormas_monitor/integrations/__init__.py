"""Integrations: framework callbacks and sinks for the dashboards teams already use.

Importing this package never needs an optional dependency. The pieces load on use:

* ``ormas_monitor.integrations.lightning.OrmasMonitorCallback``: ``pip install "ormas-monitor[lightning]"``
* ``ormas_monitor.integrations.huggingface.OrmasMonitorCallback``: ``pip install "ormas-monitor[huggingface]"``
* sinks (``TensorBoardSink``, ``WandbSink``, ``MLflowSink``, ``WebhookSink``, ``LoggingSink``,
  ``ConsoleSink``): importable here; each needs its extra (``[tensorboard]``, ``[wandb]``,
  ``[mlflow]``) only when constructed. The webhook and logging sinks need nothing extra.

``LightningCallback`` and ``HuggingFaceCallback`` are lazy aliases for the two callbacks.
"""
from __future__ import annotations

import importlib
from typing import Any

from .sinks import (SEVERITY_RANK, ConsoleSink, LoggingSink, MLflowSink, TensorBoardSink,
                    WandbSink, WebhookSink, format_event)

_LAZY = {
    "LightningCallback": ("lightning", "OrmasMonitorCallback"),
    "HuggingFaceCallback": ("huggingface", "OrmasMonitorCallback"),
    "lightning": ("lightning", None),
    "huggingface": ("huggingface", None),
}

__all__ = ["ConsoleSink", "HuggingFaceCallback", "LightningCallback", "LoggingSink", "MLflowSink",
           "SEVERITY_RANK", "TensorBoardSink", "WandbSink", "WebhookSink", "format_event"]


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _LAZY[name]
    mod = importlib.import_module(f"{__name__}.{module}")
    return mod if attr is None else getattr(mod, attr)
