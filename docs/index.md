# ormas-monitor

**Catch the layer that breaks while your model trains, and get proof.**

`ormas-monitor` is an open-source (Apache-2.0) PyTorch library and command-line tool from [Oxiedo](https://oxiedo.com).

- **It watches each layer against its own history.** A layer's gradient is compared with that layer's normal range, not with a global threshold, so the alarm names the layer that failed.
- **It certifies when a channel can never recover.** For Conv/Linear → BatchNorm → ReLU channels, an exact test on the weights, combined with run-time checks of the wiring, the gradient and the optimizer, proves that no gradient can ever reach the channel again. The monitor states the assumptions the proof needs, checks them before it issues a certificate, and records why when it refuses.
- **It can repair those channels and writes a tamper-evident record.** Repair, off by default, re-initialises stuck channels and their optimizer state. The record is a hash-chained, signable file of everything the monitor saw.

It runs entirely on your machine. The core package makes no network calls and never reads your training data; only the demo downloads CIFAR-10, and only sinks you add send anything.

## Start here

| You want to | Go to |
|---|---|
| Check a saved model file, no code changes | [Quickstart: check a file](quickstart.md#1-check-a-saved-model) |
| Watch a training run | [Quickstart: two lines](quickstart.md#2-watch-a-training-run) |
| See it catch a broken layer | [Quickstart: the demo](quickstart.md#3-run-the-demo) |
| Understand what "certified dead" means | [Dead channels and the certificate](concepts/dead-channels.md) |
| Decide whether your team should use it | [For leaders](for-leaders.md) |
| Know what it does not do | [Limitations](limitations.md) |

## The supporting tools

These exist to support the monitor. They are not separate products.

1. **`ormas-monitor check`:** a weights-only health check of a saved model, and a diff between two checkpoints: what changed and what died.
2. **The training record:** a hash-chained, signable, diffable record of what happened in training. It is evidence for technical documentation, **not** a compliance certificate.
3. **Reports for two readers:** an engineer report, and a plain-language executive summary for a CEO, CFO or compliance lead.

## Verified so far

On one seed (0) and one Apple M-series Mac (MPS, torch 2.11), on CIFAR-10: after two conv layers of a trained CNN were zeroed, a watch-only run stayed at **10.00%** (chance) for all 12 epochs, while the run with repair recovered to **76.77%** (healthy: 78.84%). Monitoring cost **+3.6%** median epoch time at the default `check_every=50`. See [Results](results.md) for every number and the command that reproduces it.

!!! note "Not yet tested on NVIDIA/Linux, multi-GPU or at scale"
    All measured numbers come from one Apple-silicon Mac. A GPU stress test is planned. Until it has run, treat performance on other hardware as unmeasured.
