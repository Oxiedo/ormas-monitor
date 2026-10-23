# Hugging Face Trainer

```bash
pip install "ormas-monitor[huggingface]"
```

Tested with transformers 4.57 on CPU only; not yet under DDP or on CUDA. **Status: OPEN.**

Add the callback. No `Trainer` subclass is needed.

```python
from transformers import Trainer, TrainingArguments
from ormas_monitor.integrations.huggingface import OrmasMonitorCallback

callback = OrmasMonitorCallback()
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="out", max_steps=1000),
    train_dataset=train_ds,
    callbacks=[callback],
)
trainer.train()
print(callback.report)          # path of the engineer report
```

## What the callback does

| Trainer event | What happens |
|---|---|
| `on_train_begin` | Builds a `Monitor` on the (unwrapped) model and the plain torch optimizer inside Accelerate's wrapper. |
| `on_pre_optimizer_step` | Calls the monitor once per optimizer step, so gradient accumulation is handled, immediately before `optimizer.step()`. |
| `on_train_end` | Closes the monitor: writes `run_end` and the reports. |

!!! warning "Gradients are seen after clipping"
    The Trainer clips gradients (`max_grad_norm`, default `1.0`) **before** the callback event the monitor uses. Per-layer norms are therefore post-clipping. A vanishing or zero gradient is still detected; an exploding one may be capped by clipping before the monitor sees it. Non-finite gradients are still caught.

!!! note "If training raises"
    The Trainer has no exception hook. Call `callback.close()` yourself in a `finally:` block if you want the record closed and the reports written after a crash:

    ```python
    try:
        trainer.train()
    finally:
        callback.close()
    ```

## The loss

Trainer callbacks never receive the training loss. The callback reads the `loss` your model returns (`outputs.loss`, `outputs["loss"]`, or `outputs[0]` when labels are given) and gives the monitor the loss of the whole optimizer step. If your `compute_loss` builds the loss outside the model, call `callback.note_loss(loss)` from it; otherwise the monitor runs without loss checks. Its gradient and channel checks are unaffected.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `to_loggers` | `False` | Also send metrics and events to the trackers in `args.report_to`: TensorBoard, W&B and MLflow. W&B and a remote MLflow send data off the machine. |
| `write_report` | `True` | Write the reports when the monitor closes. |
| `**monitor_kwargs` | | Passed to `Monitor`. `out` defaults to `<args.output_dir>/ormas_run`, `name` to `args.run_name`. Training twice into the same `output_dir` raises `FileExistsError` at `on_train_begin` unless you pass `overwrite=True` or another `out`. |

`callback.note_loss(loss)` and `callback.close()` are public; `callback.monitor` is the `Monitor` and `callback.report` the report path after training. `close()` never raises: a failure while closing is logged and kept in `callback.close_error`, so it cannot replace your own training exception.

If the Trainer's optimizer holds no torch optimizer over the model's own parameters (DeepSpeed, custom wrappers), the monitor runs without it: one line says so, repair and the scaler are dropped, and no certificate is issued; alarms and the record still work.

## What you get on a transformer

Be clear about this before you rely on it:

- **You get:** per-layer gradient alarms against each layer's own baseline, non-finite and loss-spike checks, and the training record.
- **You do not get:** the dead-channel certificate. Transformers use LayerNorm and GELU/SiLU, not BatchNorm → ReLU, so Theorem 2 does not apply. The monitor never issues a certificate it cannot back.

For a large model, restrict the monitor to the layers you care about with `layers=`. See [Large models](large-models.md).

## Checking a saved Hugging Face model

`ormas-monitor check` reads a Hugging Face model folder (sharded or not) directly, tensor by tensor. `--arch hf` builds the model from the folder's `config.json` with random weights, using local files only (no download), to learn the exact BatchNorm → ReLU wiring. Tracing needs an image model (one whose `forward` takes `pixel_values`, such as a Hugging Face ResNet); for a text model the check says why it fell back to the name-based heuristic, which matters little there because transformers rarely use BatchNorm. See [check as a CI gate](ci-gate.md) and the [findings catalogue](../reference/check-findings.md).

```bash
ormas-monitor check ./my-finetuned-model --against ./base-model --arch hf
```

A runnable training example is in [`examples/integrations/huggingface_example.py`](https://github.com/Oxiedo/ormas-monitor/blob/main/examples/integrations/huggingface_example.py).
