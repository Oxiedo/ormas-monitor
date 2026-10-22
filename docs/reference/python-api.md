# Python API

The public API is small. Everything listed here is stable within a minor version; anything with a leading underscore is not.

```python
from ormas_monitor import Monitor, MonitorConfig, __version__
```

<!-- VERIFY: principal — MonitorConfig exported from ormas_monitor/__init__.py (today it is ormas_monitor.config.MonitorConfig) -->

## `Monitor`

```python
Monitor(
    model, optimizer=None, *,
    out="ormas_run",            # run directory
    repair=False,               # False | True | "all"
    check_every=None, patience=None, sigma=None, warmup=None,   # shortcuts; override config
    name=None,                  # label for the reports; default: the run directory's name
    sinks=(),                   # callables sink(kind, payload)
    verbose=True,               # print one line per event
    config=None,                # MonitorConfig
    scaler=None,                # torch.amp.GradScaler, for mixed precision
    layers=None,                # fnmatch pattern(s): restrict which layers are watched
)
```

| Argument | Meaning |
|---|---|
| `model` | Any `torch.nn.Module`; `torch.compile` and DDP wrappers are unwrapped. The monitor registers forward hooks on its Conv/Linear/BatchNorm and ReLU modules and removes them on `close()`. |
| `optimizer` | The optimizer; Lightning and Accelerate wrappers are unwrapped. Needed for the certificate's momentum condition, for repair and for `scaler`. Without it, alarms and loss checks still run, and no certificate is issued. |
| `repair` | `False`: never change the model. `True`: repair certified-dead, constant and decayed channels. `"all"`: also channels observed silent. Any other value raises `ValueError`; `repair` without an optimizer raises `ValueError`. |
| `check_every`, `patience`, `sigma`, `warmup` | Shortcuts for the [MonitorConfig](config.md) fields of the same name. When given, they override `config`. |
| `config` | A [MonitorConfig](config.md) with every threshold. Default `MonitorConfig()`. Anything else raises `TypeError`. |
| `scaler` | A `torch.amp.GradScaler`. Skipped AMP steps become `skipped_step` events rather than non-finite alarms, and norms are unscaled if `unscale_` was not called. Needs the optimizer. See [Mixed precision](../guides/mixed-precision.md). |
| `layers` | A pattern string or list of `fnmatch` patterns on module names; only matching layers are watched. Matching nothing raises `ValueError`. See [Large models](../guides/large-models.md). |
| `sinks` | See [Sinks](../guides/sinks.md). |

### Methods and attributes

| Name | Meaning |
|---|---|
| `mon.step(loss=None)` | Call once per optimizer step, after `backward()` (and after `scaler.unscale_` if used), before `optimizer.step()`. `loss` may be a tensor, a float or `None`. On unsampled steps it only increments a counter. Raises `RuntimeError` after `close()`. |
| `mon.note_loss(loss)` | Record the loss of the current step, for code that cannot pass it to `step()`. The next `step()` called without a loss uses it; a loss passed to `step()` wins. The callbacks use it. |
| `mon.close(report=True)` | Remove the hooks, write `run_end`, close the files and, with `report=True`, write the reports. Returns the path of the engineer report, or `None` when `report=False`, the monitor was already closed, or this is not rank 0. Safe to call twice. |
| `with Monitor(...) as mon:` | Calls `close()` on exit, including on an exception. |
| `mon.step_count` | Optimizer steps seen so far. |
| `mon.checks` | Checks run so far. |
| `mon.counts` | `{"alarms", "stuck_layers", "repairs", "channels_repaired", "loss_spikes", "nonfinite", "skipped_steps"}` totals. |
| `mon.config` | The effective `MonitorConfig`. |
| `mon.out` | The run directory, as a `Path`. |

## Checkpoint check

```python
from ormas_monitor.checkpoint import run, CheckpointError

result = run("model.pt", against=None, trust=False, arch=None, config=None)
```

Returns a dict with `file`, `tensors`, `parameters`, `batchnorm_layers`, `findings` (a list of `{severity, kind, title, detail, ...}`, see the [findings catalogue](check-findings.md)) and `layers` (a list of `{layer, batchnorm, channels, certified, flat, decayed, dormant, collapsed_var}`). With `against`, also `against` and `diff`. Further keys: `format, files, wrapper, streamed, dtypes, model_type, wiring, notes, norm_layers, severity_counts`. `path` may be a file or a Hugging Face folder; `arch` is `"torchvision:<name>"` or `"hf"`; `config` is a `CheckConfig`. Raises `CheckpointError` when the file is missing, unreadable, or holds pickled objects and `trust=False`. The `--fail-on` threshold is applied by the CLI, not by `run`.

## Record

```python
from ormas_monitor.record import verify_run, seal, sign, keygen, diff_runs
```

| Function | Returns |
|---|---|
| `verify_run(run_dir_or_diary, public_key=None)` | A `VerifyResult`: `.ok` (no check failed), `.complete` (every check passed), `.checks` (a list of `Check(name, status, message)`), `.to_dict()`. |
| `seal(run_dir)` | Writes `record.json`; returns its path. Raises `RecordError` if the run does not verify. |
| `sign(run_dir, private_key_path)` | Seals with the signer's fingerprint and writes `record.sig`; returns its path. Needs the `signing` extra. |
| `keygen(private_path, public_path)` | Writes an Ed25519 key pair as PEM; returns `KeyInfo(private_path, public_path, fingerprint)`. Refuses to overwrite. |
| `diff_runs(run_a, run_b)` | A dict: `a`, `b` (`{path, name, model, architecture_sha256, steps, intact}`), `same_architecture`, `config_changes`, `events`, `layers`, `verdict`. |

`ormas_monitor.diary` remains importable for 0.1 code (`Diary`, `verify`, `read`).

## Reports

```python
from ormas_monitor.report import write_all, write_run_report, write_summary, write_check_report, write_diff_report
```

| Function | Writes |
|---|---|
| `write_all(run_dir, gpu_hour_cost=None, gpus=None, config=None, public_key=None)` | Every report for a run. Returns `{"report": engineer HTML, "summary": executive HTML, "summary_md": executive Markdown}`. A dollar ESTIMATE appears only when `gpu_hour_cost` is given. |
| `write_summary(run_dir, ...)` | The executive summary only. |
| `write_run_report(run_dir)` | The engineer report only. |
| `write_check_report(check_result, path)` | The HTML report for a `checkpoint.run` result. |
| `write_diff_report(diff_result, path)` | The HTML report for a `diff_runs` result. |

## Integrations

| Import | Needs |
|---|---|
| `from ormas_monitor.integrations.lightning import OrmasMonitorCallback` | `[lightning]` |
| `from ormas_monitor.integrations.huggingface import OrmasMonitorCallback` | `[huggingface]` |
| `from ormas_monitor.integrations import TensorBoardSink, WandbSink, MLflowSink, WebhookSink, LoggingSink, ConsoleSink` | each sink's extra, only when constructed |

`ormas_monitor.integrations.LightningCallback` and `HuggingFaceCallback` are lazy aliases for the two callbacks. See the [Lightning](../guides/lightning.md), [Hugging Face](../guides/huggingface.md) and [Sinks](../guides/sinks.md) guides.

## Low-level

`ormas_monitor.structure` holds the weights-only channel test (`scan_channels`) and the wiring probe (`relu_followers(model, example_input)`). `ormas_monitor.schema` holds the event schema (`EVENT_FIELDS`, `validate_event`). These are importable but may change between minor versions.
