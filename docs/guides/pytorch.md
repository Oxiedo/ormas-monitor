# Plain PyTorch

## Where `step()` goes

Call `mon.step(loss)` **once per optimizer step**, after `loss.backward()` and before `optimizer.step()`. At that point every gradient for the step exists and nothing has used it yet. Called before `backward()`, the monitor sees no gradients: it writes a `no_gradients` warning and the reports say NOT MONITORED instead of healthy.

```python
from ormas_monitor import Monitor

mon = Monitor(model, optimizer, out="runs/exp1", name="resnet18 baseline")
for epoch in range(epochs):
    for x, y in loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        mon.step(loss)
        optimizer.step()
mon.close()
```

Or as a context manager, which calls `close()` even if training raises:

```python
with Monitor(model, optimizer, out="runs/exp1") as mon:
    for x, y in loader:
        ...
        mon.step(loss)
        optimizer.step()
```

`close()` writes `run_end`, then the reports, and returns the path of the engineer report. Pass `close(report=False)` to skip the reports; you can build them later with `ormas-monitor report runs/exp1`.

A new `Monitor` on a directory that already holds a record raises `FileExistsError`. Use a fresh `out=` per run, or `overwrite=True` to replace the old record on purpose.

## Gradient accumulation

With accumulation, there is one optimizer step per `k` backward passes. Call `step()` only on the backward pass that is followed by `optimizer.step()`:

```python
for i, (x, y) in enumerate(loader):
    loss = loss_fn(model(x), y) / k
    loss.backward()
    if (i + 1) % k == 0:
        mon.step(loss)            # once per optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

If the loss is computed somewhere other than where you call `step()`, pass it with `mon.note_loss(loss)` and call `mon.step()` without an argument.

## Gradient clipping

Call `step()` **before** clipping if you want alarms on the raw gradient, which is usually what you want: clipping hides exploding gradients.

```python
loss.backward()
mon.step(loss)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

## The arguments you will use most

| Argument | Default | Meaning |
|---|---|---|
| `out` | `"ormas_run"` | Run directory for the record, metrics and reports. |
| `repair` | `False` | `True` repairs certified, constant and decayed channels; `"all"` also silent ones. See [Repair](../concepts/repair.md). |
| `check_every` | `50` | Optimizer steps between checks. Hooks do nothing on the other steps. |
| `patience` | `4` | Checks a channel must stay stuck before it counts as silent or is repaired. |
| `sigma` | `2.5` | How far, in standard deviations, a layer may move before an alarm. |
| `warmup` | `8` | Checks before alarms are allowed. |
| `name` | run directory name | A label shown in the reports. |
| `sinks` | `()` | Where else to send events. See [Sinks](sinks.md). |
| `verbose` | `True` | Print one line per event. |
| `config` | `None` | A [MonitorConfig](../reference/config.md); the shortcuts above override its fields. |
| `scaler` | `None` | A `torch.amp.GradScaler`. See [Mixed precision](mixed-precision.md). |
| `layers` | `None` | `fnmatch` patterns restricting which layers are watched. See [Large models](large-models.md). |
| `overwrite` | `False` | Replace a record already in `out`. By default an existing record raises `FileExistsError`. |

All of them, and the configuration object, are in [MonitorConfig](../reference/config.md).

## What the monitor needs from your model

- **Nothing for alarms and loss checks.** Any `nn.Module` works.
- **For channel checks and the certificate:** `Conv1d/2d/3d` or `Linear` layers, optionally followed by `BatchNorm1d/2d/3d` or `SyncBatchNorm`, followed by an `nn.ReLU` or `nn.ReLU6` **module** that is the only consumer of that output, and the optimizer passed to `Monitor` (`SGD` or the Adam family; see [the conditions](../concepts/dead-channels.md#the-conditions-the-monitor-checks-before-it-issues-a-certificate)). A functional `F.relu` call is not a module and cannot be seen by a hook, so those channels are reported as constant, never certified.

The monitor learns the wiring from the first sampled step: which tensor each BatchNorm reads during the forward pass, and the autograd graph of the loss passed to `step()` (or of the model's outputs if no loss with a `grad_fn` is given). Run a training forward pass before each `step()`, which is what a normal loop does. Without an autograd graph at that step, no certificate is issued for the run; the `manifest` event says why.
