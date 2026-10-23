# Repair

Repair is **off by default**. With `repair=False` the monitor never changes your model or optimizer.

```python
mon = Monitor(model, optimizer, repair=True)    # certified, constant and decayed channels
mon = Monitor(model, optimizer, repair="all")   # also channels observed silent for `patience` checks
```

Repair needs the optimizer, because it resets the optimizer state of the channels it repairs. `Monitor(model, repair=True)` without an optimizer raises `ValueError`.

## What a repair does

For each channel that is due, the monitor:

1. re-initialises the incoming filter (Kaiming-normal, ReLU gain) and sets the producer's bias to 0;
2. resets the BatchNorm: scale γ = 1, shift β = 0, running mean 0, running variance 1;
3. zeroes every optimizer-state tensor shaped like a touched parameter (momentum, second moment) for exactly those channels, and nothing else.

It writes a `repair` event with the layer, the channel indices, the method, the number of optimizer state slices reset, and the measured size of the change: ‖ΔW‖, ‖W‖ before and after, and their ratio. The event also says where the new weights came from: torch's global RNG by default, or a generator seeded from (`repair_seed`, step, layer) when [`repair_seed`](../reference/config.md) is set or the run is distributed. Framework wrappers around the optimizer (Lightning, Accelerate) are unwrapped first.

## When a channel is due

- **Certified-dead channels** are repaired after one confirming check. They cannot recover on their own, so waiting longer gains nothing.
- **Constant and decayed channels** wait until they have been stuck for `patience` consecutive checks (default 4).
- **Silent channels** are repaired only with `repair="all"`, after `patience` checks, because silence is an observation, not a proof.

## Why repair is a discrete modification

The [certificate](dead-channels.md) shows that a certified-dead channel sits in an absorbing set: gradient descent cannot leave it, and neither can any repair that moves the weights by an amount proportional to the weights themselves (‖R(W) − W‖ ≤ ρ‖W‖ keeps a zero filter at zero). The only way out is a jump: replacing the weights with fresh ones.

That has two practical consequences:

- **Repair changes the model you are training.** A repaired run is not the same run with a fix applied invisibly; it is a different trajectory. The record marks every repair with its step and channels so a reviewer can see exactly where the trajectory changed.
- **Repair restores capacity, not what was learnt.** The re-initialised channels start from scratch. In the CIFAR-10 demo, repair recovered 76.77% against a healthy 78.84%, not the full accuracy (one seed, Apple M-series). See [Results](../results.md).

## When not to use it

- During evaluation, fine-tuning runs you must reproduce bit-for-bit, or any run where changing weights outside the optimizer is not acceptable. Use watch-only mode and act on the report.
- When the "dead" channels are intentional, for example after BatchNorm-γ pruning (network slimming). Repair would undo the pruning.
