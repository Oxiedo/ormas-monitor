# Mixed precision

With a `GradScaler`, gradients after `backward()` are multiplied by the loss scale. A layer alarm compares log-gradients with a baseline, so a changing scale would look like a layer exploding or vanishing, and a step the scaler skips would look like a failure. Pass the scaler to the monitor:

```python
from ormas_monitor import Monitor

scaler = torch.amp.GradScaler("cuda")
mon = Monitor(model, optimizer, scaler=scaler)

for x, y in loader:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)          # recommended; see below
    mon.step(loss)
    scaler.step(optimizer)
    scaler.update()
```

`scaler=` needs the optimizer, and must be a `torch.amp.GradScaler` (anything without `is_enabled()` and `get_scale()` raises `TypeError`).

## What the scaler changes

- **Skipped steps are not failures.** When the scaler finds Inf or NaN gradients, it skips the optimizer step and lowers the scale. That is normal AMP behaviour. The monitor records such a step as a `skipped_step` event (severity `info`) instead of a `nonfinite_gradient` alarm, counts it in `counts["skipped_steps"]`, and does not move any baseline.
- **Norms are true magnitudes.** Gradient norms are computed in float32. If you called `scaler.unscale_(optimizer)` before `step()`, they are used as is; if not, the monitor divides them by the current scale. Calling `unscale_` first is still the clearer pattern, and it is required anyway if you clip gradients.

## bfloat16

`torch.autocast(dtype=torch.bfloat16)` does not need a scaler. Use the plain loop from [Plain PyTorch](pytorch.md).

## Lightning and Hugging Face

- **Lightning:** the callback runs after AMP's `unscale_`, so gradients are true magnitudes, and it divides the scale back out of the loss. You do not pass a scaler.
- **Hugging Face Trainer:** under fp16 the callback passes Accelerate's GradScaler to the monitor. With gradient clipping on (the default `max_grad_norm=1.0`), Accelerate unscales the gradients before clipping, which is before the callback runs, and the monitor uses them as they are. With `max_grad_norm=0` (no clipping) the gradients are still scaled when the callback runs; the monitor sees from the scaler's state that `unscale_` was not called and divides the norms by the current scale. This follows from the code paths; it has not been run end to end, because fp16 training needs a CUDA GPU and the integrations have been tested on CPU only. **Status: OPEN.**
