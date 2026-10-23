# Alarms against each layer's own baseline

A global threshold on gradient norm does not work well: an embedding layer and a classifier head have gradient norms orders of magnitude apart, and both drift during training. The monitor compares each layer with **its own** recent history instead.

## How a layer alarm fires

Every `check_every` optimizer steps (default 50), the monitor reads each layer's gradient norm once, in one device-to-host transfer per device that holds gradients. For each layer it keeps a running mean and variance of log₁₀(gradient norm).

After `warmup` checks (default 8) have built the baseline, a check is *out of range* when the layer's log-gradient is

- more than `sigma` standard deviations (default 2.5) **and** at least one order of magnitude (10×) below its mean: **vanishing**;
- the same distance above its mean: **exploding**;
- exactly zero while the baseline is nonzero: **no gradient**.

An alarm fires only when the same condition holds on **two checks in a row**. It is cleared, with an `alarm_cleared` event, on the first check back in range. Out-of-range values are not added to the baseline, so a failing layer does not teach the monitor that failure is normal.

| Diagnosis | Meaning |
|---|---|
| `no_gradient` | The layer's gradient is exactly zero. Often a dead layer upstream or downstream, or a detached graph. |
| `vanishing` | The gradient collapsed by at least 10× relative to this layer's own normal. |
| `exploding` | The gradient grew by at least 10× relative to this layer's own normal. |

## Loss checks

On every check the monitor also looks at the loss passed to `mon.step(loss)`:

- **Non-finite loss or gradients** (NaN or Inf) raise a `critical` event, once per episode.
- **Loss spikes:** the loss is more than `sigma` standard deviations above its running mean **and** more than 1.5× that mean. One `loss_spike` event per episode, followed by `loss_recovered`.

Code that sees the loss in a different place from the optimizer step can call `mon.note_loss(loss)`; the next `step()` without a loss uses it. A loss passed to `step()` wins. A loss with more than one element (unreduced, per sample) is recorded as its mean, and one `loss_reduced` event says so.

## No gradients

If no watched layer has a gradient at a check, usually because `mon.step()` was called before `loss.backward()`, the monitor writes one `no_gradients` warning, counts every such check in `counts["checks_without_gradients"]`, and the reports say **NOT MONITORED** (no check saw gradients) or **PARTLY MONITORED** (some did not) instead of healthy. A run with no check at all is reported as **UNKNOWN**.

## Why the numbers are what they are

The defaults (`sigma=2.5`, a 10× floor, two consecutive checks, `warmup=8`) are chosen to keep false alarms rare on ordinary training noise. They are design choices, not measured optima. **Status: DESIGNED.** If your run alarms on noise, raise `sigma` or `warmup`; see [MonitorConfig](../reference/config.md).

## What an alarm does not tell you

An alarm says *which* layer left its normal range and in which direction. It does not say *why*. A vanishing gradient in one layer may be caused by a dead layer elsewhere; the report's timeline shows which event came first. For channels that can be proven dead, see [the certificate](dead-channels.md).
