# Lightning

```bash
pip install "ormas-monitor[lightning]"
```

Add the callback to your `Trainer`. No change to your `LightningModule`.

```python
import lightning as L
from ormas_monitor.integrations.lightning import OrmasMonitorCallback

callback = OrmasMonitorCallback(repair=False, check_every=50)
trainer = L.Trainer(max_epochs=10, callbacks=[callback])
trainer.fit(model, datamodule=dm)
print(callback.report)          # path of the engineer report
```

Both `lightning` and the older `pytorch_lightning` package are supported.

## What the callback does

| Lightning hook | What happens |
|---|---|
| `on_fit_start` | Builds a `Monitor` on the `LightningModule` and the chosen optimizer. |
| `on_before_backward` | Collects the loss about to be backpropagated. |
| `on_before_optimizer_step` | Calls the monitor once, after backward and after AMP's `unscale_`, before gradient clipping and `optimizer.step()`, and only on steps where the optimizer actually steps. With `accumulate_grad_batches`, that is one monitor step per optimizer step. |
| `on_fit_end`, `on_exception` | Closes the monitor: writes `run_end` and the reports. |

The loss the monitor sees is the objective whose gradient is in `.grad`: with accumulation, the mean of the micro-batch losses. Under 16-bit AMP the callback divides the GradScaler's scale back out, so loss spikes are judged on the true loss.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `optimizer_index` | `0` | Which of `trainer.optimizers` the monitor follows and repairs. The monitor steps only when that optimizer steps. |
| `to_loggers` | `False` | Also send metrics and events to the trainer's own `TensorBoardLogger` and `WandbLogger`. Other loggers are ignored. |
| `write_report` | `True` | Write the reports when the monitor closes at the end of `fit`. |
| `**monitor_kwargs` | | Passed to `Monitor`: `repair`, `check_every`, `sinks`, `config`, `name`, ... |
| `out` | `<trainer.log_dir>/ormas_run` | The run directory, as for `Monitor`. |

After `fit`, `callback.monitor` is the closed monitor (with `counts`) and `callback.report` is the report path.

## Notes

- **Several optimizers:** only the optimizer at `optimizer_index` is monitored and repaired; the monitor steps only when that one steps.
- **Mixed precision:** no scaler argument is needed; see above and [Mixed precision](mixed-precision.md).
- **Multiple GPUs:** see [Distributed training](distributed.md). The callback has been tested on CPU only, not under DDP or on CUDA. **Status: OPEN.**
- **`to_loggers=True` with W&B** sends data to W&B. See [Sinks](sinks.md).

A runnable example is in [`examples/integrations/lightning_example.py`](https://github.com/OWNER/ormas-monitor/blob/main/examples/integrations/lightning_example.py).
