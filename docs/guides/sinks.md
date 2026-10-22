# Sending events to TensorBoard, W&B, MLflow or a webhook

By default nothing leaves your machine. A **sink** forwards the monitor's events and metrics to another tool. A sink that sends data over the network (W&B, MLflow with a remote tracking server, a webhook) is the **only** way data leaves the machine, and only if you add one.

## Built-in sinks

```python
from ormas_monitor import Monitor
from ormas_monitor.integrations import TensorBoardSink, LoggingSink    # or ormas_monitor.integrations.sinks

mon = Monitor(model, optimizer, sinks=[TensorBoardSink("runs/exp1"), LoggingSink()])
```

| Sink | Install | What it writes | Data off the machine? |
|---|---|---|---|
| `TensorBoardSink(log_dir=None, *, writer=None, layers=None, prefix="ormas")` | `[tensorboard]` | Scalars `ormas/loss`, `ormas/grad_norm/<layer>`, `ormas/stuck_channels/<layer>`, `ormas/stuck_channels_total`; each event as text under `ormas/events` and a marker scalar `ormas/events/<event>`. | No. Local event files. |
| `WandbSink(run=None, *, layers=None, prefix="ormas", alert_severity=None)` | `[wandb]` | The same scalars, with `ormas/step` as their x-axis so it never fights your own step counter. With `alert_severity="proof"`, matching events also raise a W&B alert. | **Yes**, unless W&B runs offline or disabled. |
| `MLflowSink(*, layers=None, prefix="ormas")` | `[mlflow]` | The same scalars as MLflow metrics on the active run; events as marker metrics. | **Yes** with a remote tracking URI; no with a local `mlruns/` folder. |
| `WebhookSink(url, *, min_severity="warning", timeout=2.0, headers=None, blocking=False)` | none | POSTs `{"source": "ormas-monitor", "text": ..., "event": <diary line>}` as JSON for events at or above `min_severity`. Never sends metrics. Posts on a background thread so a slow endpoint never stalls training. | **Yes**, to the URL you give. |
| `LoggingSink(logger="ormas_monitor", *, min_severity="info", metrics_level=None)` | none | Events to a Python `logging` logger: critical → ERROR, warning/action/proof → WARNING, others → INFO. | No. |
| `ConsoleSink(*, min_severity="warning")` | none | Like `LoggingSink`, to stderr. Useful with `Monitor(verbose=False)`. | No. |

`layers=` takes `fnmatch` patterns (for example `["encoder.*"]`) and limits which layers get per-layer scalars.

Severities, lowest to highest: `info`, `warning`, `action` (a repair), `proof` (a certificate), `critical` (non-finite values).

## Write your own

A sink is any callable `sink(kind: str, payload: dict) -> None`.

| `kind` | `payload` |
|---|---|
| `"event"` | The full diary line: `seq`, `time`, `event`, `prev`, `hash`, and the event's fields. See [Record format](../reference/record-format.md). |
| `"metrics"` | One row per check: `{"step": int, "loss": float or None, "grad": {layer: float}, "stuck": {layer: [certified, constant, silent, decayed]}}`. |

```python
def print_alarms(kind, payload):
    if kind == "event" and payload["event"] == "alarm":
        print("ALARM", payload["layer"], payload["diagnosis"])

mon = Monitor(model, optimizer, sinks=[print_alarms])
```

**A sink that raises is disabled for the rest of the run.** The monitor writes a `sink_error` event to the record and training continues. A dashboard outage never stops a training run.

## What a sink receives

Event payloads contain layer names, step numbers, gradient norms and channel indices. They do not contain weights, activations or any training data. The `run_start` event contains the run name, the model class name, parameter count, optimizer settings, Python and PyTorch versions and the platform string. Review that before you send it to a third-party service.
