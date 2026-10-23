"""Sinks: send the monitor's events and metrics to the places a team already looks.

A sink is any callable ``sink(kind, payload) -> None`` passed as ``Monitor(..., sinks=[...])``
(brief §4.2). ``kind == "event"`` carries a full diary line; ``kind == "metrics"`` carries one
metrics row ``{"step", "loss", "grad": {layer: norm}, "stuck": {layer: [certified, constant,
silent, decayed]}}``. A sink that raises is disabled for the rest of the run and a ``sink_error``
event is written to the diary; training continues.

    from ormas_monitor.integrations.sinks import TensorBoardSink, LoggingSink
    mon = Monitor(model, opt, sinks=[TensorBoardSink("runs/exp1"), LoggingSink()])

Every sink here imports its optional package (tensorboard, wandb, mlflow) only when it is
constructed, so importing this module never needs them. Only `WandbSink`, `MLflowSink` (when its
tracking URI is remote) and `WebhookSink` can send data off the machine, and only because the user
configured them to.
"""
from __future__ import annotations

import atexit
import fnmatch
import importlib
import json
import logging
import queue
import sys
import threading
import urllib.request
import weakref
from typing import Any, Mapping, Sequence

SEVERITY_RANK: Mapping[str, int] = {"info": 0, "warning": 1, "action": 2, "proof": 3, "critical": 4}
"""Order used by `min_severity` filters. Events without a `severity` field rank as "info", except
`sink_error`, which ranks as "warning"."""

_EXTRAS = {"tensorboard": "tensorboard", "wandb": "wandb", "mlflow": "mlflow"}


def _require(module: str, what: str) -> Any:
    """Import an optional dependency or raise an ImportError that names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = _EXTRAS[module.split(".")[0]]
        raise ImportError(f"{what} needs the `{extra}` package: "
                          f'pip install "ormas-monitor[{extra}]"') from e


def severity_of(line: Mapping[str, Any]) -> str:
    """The severity of a diary line, with the defaults documented on `SEVERITY_RANK`."""
    sev = line.get("severity")
    if sev is None:
        return "warning" if line.get("event") == "sink_error" else "info"
    return str(sev)


def _rank(severity: str) -> int:
    if severity not in SEVERITY_RANK:
        raise ValueError(f"unknown severity {severity!r}; expected one of {sorted(SEVERITY_RANK)}")
    return SEVERITY_RANK[severity]


def format_event(line: Mapping[str, Any]) -> str:
    """One human-readable line for a diary event, e.g. ``step 40: certificate conv2: 16 channels
    certified dead``."""
    ev = line.get("event", "?")
    step = line.get("step")
    head = f"step {step}: {ev}" if step is not None else str(ev)
    layer = line.get("layer")
    if ev == "alarm":
        detail = (f"{line.get('diagnosis')} (gradient {line.get('grad_norm', float('nan')):.3g}, "
                  f"baseline {line.get('baseline', float('nan')):.3g})")
    elif ev == "alarm_cleared":
        detail = f"{line.get('diagnosis')} cleared"
    elif ev == "certificate":
        detail = f"{line.get('count')} channels certified dead"
    elif ev == "repair":
        detail = f"re-initialised {line.get('count')} channels"
    elif ev in ("channels", "channels_cleared"):
        detail = (f"{line.get('n_certified', 0)} certified, {line.get('n_flat', 0)} constant, "
                  f"{line.get('n_decayed', 0)} decayed, {line.get('n_observed', 0)} silent "
                  f"of {line.get('channels')}")
    elif ev in ("loss_spike", "loss_recovered"):
        detail = f"loss {line.get('value')} (baseline {line.get('baseline')})"
    elif ev == "sink_error":
        detail = f"{line.get('sink')}: {line.get('error')}"
    else:
        detail = ""
    out = head
    if layer is not None:
        out += f" {layer}"
    if detail:
        out += f": {detail}"
    return out


class _LayerFilter:
    """fnmatch patterns over layer names; the answer per name is cached (bounded by the model)."""

    def __init__(self, patterns: Sequence[str] | None) -> None:
        if isinstance(patterns, str):
            raise TypeError("layers must be a sequence of patterns, not a single string")
        self.patterns = None if patterns is None else tuple(patterns)
        self._cache: dict[str, bool] = {}

    def __call__(self, name: str) -> bool:
        if self.patterns is None:
            return True
        hit = self._cache.get(name)
        if hit is None:
            hit = any(fnmatch.fnmatchcase(name, p) for p in self.patterns)
            self._cache[name] = hit
        return hit


def _flatten_metrics(row: Mapping[str, Any], prefix: str, keep: _LayerFilter) -> dict[str, float]:
    """Scalars for one metrics row: loss, per-layer gradient norm, per-layer and total stuck channels."""
    out: dict[str, float] = {}
    if row.get("loss") is not None:
        out[f"{prefix}/loss"] = float(row["loss"])
    for layer, g in row.get("grad", {}).items():
        if keep(layer):
            out[f"{prefix}/grad_norm/{layer}"] = float(g)
    total = 0
    for layer, counts in row.get("stuck", {}).items():
        n = int(sum(counts))
        total += n
        if keep(layer):
            out[f"{prefix}/stuck_channels/{layer}"] = float(n)
    out[f"{prefix}/stuck_channels_total"] = float(total)
    return out


# ---------------------------------------------------------------------------------- TensorBoard
class TensorBoardSink:
    """Write metrics and events to TensorBoard.

    Scalars: ``ormas/loss``, ``ormas/grad_norm/<layer>``, ``ormas/stuck_channels/<layer>`` and
    ``ormas/stuck_channels_total``, at the monitor's step. Each event becomes a text entry under
    ``ormas/events`` and a marker scalar ``ormas/events/<event>`` = 1 at its step.

    Args:
        log_dir: directory for a new ``SummaryWriter`` the sink owns (created on first use and
            closed by `close()`). ``None`` uses SummaryWriter's default (``runs/...``).
        writer: an existing ``SummaryWriter`` (for example Lightning's ``TensorBoardLogger.experiment``).
            The sink only flushes it; its owner closes it.
        layers: fnmatch patterns limiting which layers get per-layer scalars (``None`` = all).
        prefix: tag prefix.
    """

    def __init__(self, log_dir: str | None = None, *, writer: Any = None,
                 layers: Sequence[str] | None = None, prefix: str = "ormas") -> None:
        if log_dir is not None and writer is not None:
            raise ValueError("pass log_dir or writer, not both")
        if writer is None:
            self._writer_cls = _require("torch.utils.tensorboard", "TensorBoardSink").SummaryWriter
        self._writer = writer
        self._owns_writer = writer is None
        self.log_dir = log_dir
        self.prefix = prefix
        self._keep = _LayerFilter(layers)
        self._last_step = 0

    @property
    def writer(self) -> Any:
        if self._writer is None:
            self._writer = self._writer_cls(log_dir=self.log_dir)
        return self._writer

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "metrics":
            step = int(payload["step"])
            self._last_step = step
            w = self.writer
            for tag, value in _flatten_metrics(payload, self.prefix, self._keep).items():
                w.add_scalar(tag, value, step)
        elif kind == "event":
            step = payload.get("step")
            step = self._last_step if step is None else int(step)
            w = self.writer
            w.add_text(f"{self.prefix}/events", format_event(payload), step)
            w.add_scalar(f"{self.prefix}/events/{payload.get('event')}", 1.0, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Flush; close the writer only if this sink created it."""
        if self._writer is None:
            return
        if self._owns_writer:
            self._writer.close()
            self._writer = None
        else:
            self._writer.flush()


# ---------------------------------------------------------------------------------- W&B
class WandbSink:
    """Log metrics and events to a Weights & Biases run.

    Logs the same scalars as `TensorBoardSink` plus ``ormas/step``, which is registered with
    ``define_metric`` as the x-axis for every ``ormas/*`` key, so the sink never fights the user's
    own ``step`` counter. Events become marker scalars ``ormas/events/<event>`` = 1.

    Args:
        run: the run to log to; ``None`` uses ``wandb.run`` (the active run) at each call, and
            raises if there is none, which disables the sink with a ``sink_error`` event.
        layers: fnmatch patterns limiting per-layer keys.
        alert_severity: if set (e.g. ``"proof"``), events at or above it also call ``run.alert``,
            which W&B delivers by email or Slack according to the user's W&B settings.
    """

    def __init__(self, run: Any = None, *, layers: Sequence[str] | None = None,
                 prefix: str = "ormas", alert_severity: str | None = None) -> None:
        self._wandb = _require("wandb", "WandbSink")
        self._run = run
        self.prefix = prefix
        self._keep = _LayerFilter(layers)
        self._alert_rank = None if alert_severity is None else _rank(alert_severity)
        self._defined: Any = None      # weak reference to the run the step metric was defined on

    def _target(self) -> Any:
        run = self._run if self._run is not None else self._wandb.run
        if run is None:
            raise RuntimeError("WandbSink: no active W&B run; call wandb.init() first or pass run=")
        if self._defined is None or self._defined() is not run:
            run.define_metric(f"{self.prefix}/step")
            run.define_metric(f"{self.prefix}/*", step_metric=f"{self.prefix}/step")
            self._defined = weakref.ref(run)
        return run

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "metrics":
            run = self._target()
            data: dict[str, float] = {f"{self.prefix}/step": int(payload["step"])}
            data.update(_flatten_metrics(payload, self.prefix, self._keep))
            run.log(data)
        elif kind == "event":
            run = self._target()
            data = {f"{self.prefix}/events/{payload.get('event')}": 1}
            if payload.get("step") is not None:
                data[f"{self.prefix}/step"] = int(payload["step"])
            run.log(data)
            sev = severity_of(payload)
            if self._alert_rank is not None and SEVERITY_RANK.get(sev, 0) >= self._alert_rank:
                level = self._wandb.AlertLevel.ERROR if sev == "critical" else self._wandb.AlertLevel.WARN
                run.alert(title=f"ormas-monitor: {payload.get('event')}", text=format_event(payload),
                          level=level)


# ---------------------------------------------------------------------------------- MLflow
class MLflowSink:
    """Log metrics and events to the active MLflow run with the fluent API.

    Uses ``mlflow.log_metrics(dict, step=...)``; events are marker metrics
    ``ormas/events/<event>`` = 1 at their step. If no run is active, MLflow's fluent API starts
    one (documented MLflow behaviour); call ``mlflow.start_run()`` yourself to control it.

    Args:
        layers: fnmatch patterns limiting per-layer keys.
        prefix: metric key prefix.
    """

    def __init__(self, *, layers: Sequence[str] | None = None, prefix: str = "ormas") -> None:
        self._mlflow = _require("mlflow", "MLflowSink")
        self.prefix = prefix
        self._keep = _LayerFilter(layers)
        self._last_step = 0

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "metrics":
            step = int(payload["step"])
            self._last_step = step
            self._mlflow.log_metrics(_flatten_metrics(payload, self.prefix, self._keep), step=step)
        elif kind == "event":
            step = payload.get("step")
            step = self._last_step if step is None else int(step)
            self._mlflow.log_metric(f"{self.prefix}/events/{payload.get('event')}", 1.0, step=step)


# ---------------------------------------------------------------------------------- webhook
_STOP = object()


def _flush_at_exit(ref: weakref.ReferenceType[WebhookSink]) -> None:
    sink = ref()
    if sink is not None:
        sink.close(timeout=5.0)


class WebhookSink:
    """POST events at or above a severity to an HTTP(S) endpoint as JSON. Opt-in network.

    Body: ``{"source": "ormas-monitor", "text": <one-line summary>, "event": <full diary line>}``
    with ``Content-Type: application/json``. Metrics rows are never sent.

    By default posts happen on a background thread so a slow endpoint never stalls training.
    A failed post is re-raised on the sink's next call, which makes the monitor disable the sink
    and record a ``sink_error`` event. Pending posts are flushed by `close()` and at interpreter
    exit (up to 5 s).

    Args:
        url: ``http://`` or ``https://`` endpoint.
        min_severity: lowest severity sent (see `SEVERITY_RANK`); default ``"warning"``.
        timeout: per-request timeout in seconds.
        headers: extra request headers (for example an auth token).
        blocking: post inline on the training thread instead (failures raise immediately).
        max_pending: queued posts allowed before the sink raises (and so is disabled).
    """

    def __init__(self, url: str, *, min_severity: str = "warning", timeout: float = 2.0,
                 headers: Mapping[str, str] | None = None, blocking: bool = False,
                 max_pending: int = 256) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("WebhookSink url must start with http:// or https://")
        self.url = url
        self.min_rank = _rank(min_severity)
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", "User-Agent": "ormas-monitor",
                        **dict(headers or {})}
        self.blocking = blocking
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_pending)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.sent = 0
        if not blocking:
            atexit.register(_flush_at_exit, weakref.ref(self))

    def __call__(self, kind: str, payload: dict) -> None:
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f"webhook POST to {self.url} failed: {err}") from err
        if kind != "event" or SEVERITY_RANK.get(severity_of(payload), 0) < self.min_rank:
            return
        body = json.dumps({"source": "ormas-monitor", "text": format_event(payload),
                           "event": payload}).encode()
        if self.blocking:
            self._post(body)
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="ormas-webhook", daemon=True)
            self._thread.start()
        try:
            self._queue.put_nowait(body)
        except queue.Full:
            raise RuntimeError(f"webhook {self.url}: {self._queue.maxsize} posts pending; "
                               "endpoint too slow or down") from None

    def _post(self, body: bytes) -> None:
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()
        self.sent += 1

    def _work(self) -> None:
        while True:
            body = self._queue.get()
            try:
                if body is _STOP:
                    return
                if self._error is None:
                    self._post(body)
            except Exception as e:      # surfaced to the monitor on the next call
                self._error = e
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait up to `timeout` seconds for queued posts; True if the queue drained."""
        if self._thread is None or not self._thread.is_alive():
            return self._queue.empty()
        done = threading.Event()

        def waiter() -> None:
            self._queue.join()
            done.set()

        threading.Thread(target=waiter, daemon=True).start()
        return done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending posts and stop the worker thread. The sink restarts it if called again."""
        if self._thread is not None and self._thread.is_alive():
            self.flush(timeout)
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                return
            self._thread.join(timeout)
        self._thread = None


# ---------------------------------------------------------------------------------- logging
_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "action": logging.WARNING,
           "proof": logging.WARNING, "critical": logging.ERROR}


class LoggingSink:
    """Send events to a `logging` logger: critical → ERROR, warning/action/proof → WARNING,
    the rest → INFO. Metrics rows are logged only if `metrics_level` is set.

    Args:
        logger: a logger or a logger name (default ``"ormas_monitor"``).
        min_severity: lowest severity logged.
        metrics_level: e.g. ``logging.DEBUG`` to log a one-line summary per check.
    """

    def __init__(self, logger: logging.Logger | str = "ormas_monitor", *,
                 min_severity: str = "info", metrics_level: int | None = None) -> None:
        self.logger = logging.getLogger(logger) if isinstance(logger, str) else logger
        self.min_rank = _rank(min_severity)
        self.metrics_level = metrics_level

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "event":
            sev = severity_of(payload)
            if SEVERITY_RANK.get(sev, 0) >= self.min_rank:
                self.logger.log(_LEVELS.get(sev, logging.INFO), "%s", format_event(payload))
        elif kind == "metrics" and self.metrics_level is not None \
                and self.logger.isEnabledFor(self.metrics_level):
            stuck = sum(int(sum(c)) for c in payload.get("stuck", {}).values())
            self.logger.log(self.metrics_level, "step %s: loss %s, %d layers with gradients, "
                            "%d stuck channels", payload.get("step"), payload.get("loss"),
                            len(payload.get("grad", {})), stuck)


_CONSOLE_LOGGER = "ormas_monitor.console"
_console_lock = threading.Lock()


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever `sys.stderr` is at emit time (so redirection and capture work)."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self) -> Any:    # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


class ConsoleSink(LoggingSink):
    """`LoggingSink` on the ``ormas_monitor.console`` logger, which gets one stderr handler
    (``[ormas-monitor] ...``) and does not propagate. Useful with ``Monitor(verbose=False)`` when
    you want only events at or above `min_severity` on the console."""

    def __init__(self, *, min_severity: str = "warning", metrics_level: int | None = None) -> None:
        log = logging.getLogger(_CONSOLE_LOGGER)
        with _console_lock:
            if not any(getattr(h, "_ormas_console", False) for h in log.handlers):
                handler = _StderrHandler()
                handler.setFormatter(logging.Formatter("[ormas-monitor] %(message)s"))
                handler._ormas_console = True    # type: ignore[attr-defined]
                log.addHandler(handler)
                log.propagate = False
                if log.level == logging.NOTSET:
                    log.setLevel(logging.DEBUG)
        super().__init__(log, min_severity=min_severity, metrics_level=metrics_level)


def close_sinks(sinks: Sequence[Any]) -> None:
    """Call ``close()`` on every sink that has one (integrations call this at train end)."""
    for s in sinks:
        close = getattr(s, "close", None)
        if callable(close):
            close()
