# ormas-monitor

**Find the layer that broke while your model trains, and prove when it can never come back.**

`ormas-monitor` watches a PyTorch training run layer by layer:
- it raises an alarm when a layer leaves its own normal range;
- it proves when a channel has died for good;
- it can repair the dead channels, if you ask it to;
- it writes a tamper-evident record of everything that happened.

It also checks a saved model file with no training and no data. Everything runs on your machine. The monitor and the checker make no network calls (only the optional `demo` downloads CIFAR-10), and the only dependency is PyTorch.

**See the output before installing anything:** [`sample_reports/`](sample_reports) has the real reports from the runs below.

```bash
pip install git+https://github.com/OWNER/ormas-monitor
ormas-monitor demo        # ~2 min: two layers die; watch-only vs repair (needs torchvision, downloads CIFAR-10)
```

## 1. Check a saved model (no code changes)

```bash
ormas-monitor check model.pt                       # .pt / .pth / .bin / .safetensors
ormas-monitor check finetuned.pt --against base.pt # what changed, and what died, between two
ormas-monitor check model.pt --arch torchvision:resnet50  # exact wiring for a known architecture
```

You get a terminal summary and `model.pt.health.html`. The check flags:
- **NaN/Inf weights** and all-zero tensors.
- **Certified-dead channels:** channels that provably receive zero gradient forever (see below).
- **Channels decayed to ~0:** effectively dead channels that weight decay has shrunk almost, but not exactly, to zero.
- **Constant channels,** and channels that are **probably dormant** (an estimate, labelled as one).
- **BatchNorm layers whose running variance has collapsed.**
- **With `--against`:** which tensors moved most, which never moved at all, and which channels died in between.

PyTorch files load with `weights_only=True`, so a checkpoint can't run code on your machine. A file that holds pickled Python objects is refused unless you pass `--trust`.

## 2. Watch a training run (two lines)

```python
from ormas_monitor import Monitor

mon = Monitor(model, optimizer)          # add repair=True to fix dead channels automatically
for x, y in loader:
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    mon.step(loss)                       # after backward, before optimizer.step()
    optimizer.step()
mon.close()                              # writes ormas_run/report.html
```

Every 50 steps (`check_every`), it does the following:

| What | How |
|---|---|
| **Layer alarms** | Each layer's gradient norm is compared with *its own* history. An alarm fires when it moves more than 2.5 σ **and** at least 10× away from normal, on two checks in a row. It also fires when the gradient is exactly zero. |
| **Dead channels** | The exact certificate below, plus channels that output a constant, and channels that produced no positive output on any input for `patience` checks in a row (likely dead, not proven). |
| **Loss** | Non-finite loss or gradients, and loss spikes (one event per episode). |
| **Repair** (`repair=True`) | Channels stuck for `patience` checks are re-initialised: Kaiming filter, BatchNorm reset, and the optimizer state for exactly those channels zeroed. |
| **The record** | `diary.jsonl`, with each event sealed with the SHA-256 of the one before. Check it with `ormas-monitor verify ormas_run`. |

## Why "dead for good" is a proof, not a guess

Take a Conv → BatchNorm → ReLU channel whose incoming filter (or BatchNorm scale) is exactly zero:
1. BatchNorm then outputs the constant β, its shift, for every input.
2. If β ≤ 0, the ReLU is closed everywhere and every gradient into the channel is exactly zero.
3. Under SGD or AdamW the channel can therefore never move again. Weight decay only shrinks β toward 0, never past it.
4. Any repair proportional to the weights, ‖R(W) − W‖ ≤ ρ‖W‖, leaves it at zero too.

Only a jump (a re-initialisation) gets out. This is Theorem 2 of *Self-Repair as a Bounded Disturbance* (R. A. D. Raadh, DeepMath 2026).

The monitor issues the certificate only when all of these hold:
- a ReLU directly consumes the BatchNorm output. A skip connection added before the ReLU doesn't count, and is detected.
- the optimizer holds no leftover momentum for the channel. Momentum *can* revive it, and the tests check that case.

## Results

### The dead-layer experiment (`ormas-monitor demo`)

This is the experiment from the DeepMath 2026 poster:
- **Network:** the paper's CIFAR-10 CNN (three Conv-BN-ReLU blocks and an MLP head), trained with AdamW.
- **The break:** after training, two of the three conv layers are zeroed and the optimizer state is cleared.
- **The comparison:** from that identical state, one run where the monitor only watches, and one where it also repairs.

| Setting | Healthy | Watch only | With repair |
|---|---:|---:|---:|
| Full: 50k images, 8 epochs, then 12 after the break | 78.84% | **10.00%** (all 12 epochs) | **76.77%** (192 channels repaired) |
| Quick (`ormas-monitor demo`): 10k images, 4 + 4 epochs, ~1.5 min | 61.76% | **10.00%** | **59.12%** (190 channels repaired) |

- **Watch only:** within one check the monitor certified every channel of both layers as dead, then logged exactly zero gradient for every conv layer. The run then sat at chance for good, as Theorem 2 says it must.
- **With repair:** the monitor fixed the proven-dead channels at the next check and the network recovered.

It also flagged something nobody had asked about: before any damage, **160 of the 256 hidden units in the CNN's classifier layer** were already silent. We checked them afterwards: they never fire on any of the 10,000 test images (159 never fire on all 50,000 training images). The paper's own baseline network had been carrying 62% dead units in that layer all along.

Both runs used one seed on an Apple M-series GPU. Reproduce with `ormas-monitor demo --full`.

### Overhead

Median epoch time with the default `check_every=50`:
- **Plain training:** 12.12 s.
- **Monitored:** 12.55 s, which is **+3.6%** (`examples/overhead.py`, Apple M-series GPU).

The checks run only on sampled steps, so `check_every` trades detail for speed.

### Published models carry dead channels

We ran `ormas-monitor check` on 22 ImageNet models from torchvision, using each model's real wiring so that skip connections are handled. 19 of the 21 that use BatchNorm → ReLU contain channels **decayed to ~0**:
- their filter or BatchNorm scale is under 10⁻⁵ of the layer's typical size;
- their shift is ≤ 0.

EfficientNet-B0 has no ReLU, so it's out of scope.

Run on 64 real photos (CIFAR-10 test images upscaled to 224 px), these channels output **exactly 0** in ResNet-18, VGG-19-BN and MnasNet, and at most 1.5 × 10⁻⁶ in DenseNet-169. Other channels in the same layers peak at about 0.3–1.5. It's capacity the model carries and never uses.

| Model | Channels after BN→ReLU | Decayed to ~0 |
|---|---:|---:|
| resnet18 | 1,984 | 7 |
| resnet50 | 7,616 | 1 |
| resnet152 | 24,000 | 8 |
| vgg16_bn | 4,224 | 19 |
| vgg19_bn | 5,504 | 292 (5.3%) |
| mobilenet_v2 | 15,552 | 18 |
| mnasnet1_0 | 17,232 | 734 (4.3%) |
| densenet121 | 40,800 | 2,242 (5.5%) |
| densenet169 | 77,536 | 7,667 (9.9%) |

The full table for all 22 models is in `scan_results/scan.md`; reproduce with `python examples/scan_public_models.py`.

None of these channels is *exactly* zero, so none is "certified". The weights are only near zero, which is what a dead channel looks like after weight decay has spent the rest of training shrinking it.

Shrinking BatchNorm scales toward zero is also a known pruning technique, so the point isn't that such channels exist. The point is that one command finds them.

## What it does not do

- **Dead-channel proofs cover Conv/Linear → BatchNorm → ReLU networks** (ResNets, VGG, MobileNet and similar). On transformers you get the layer alarms, loss checks and the record, but no dead-unit proof.
- **It never reads your data,** so it can't tell you which examples or labels are wrong.
- **The record is evidence, not a certificate of compliance.** It documents what happened during training, for your own technical documentation.

## About

Built by Rokib Al Dhin Raadh at [Oxiedo](https://oxiedo.com). Questions, bugs, or a run you'd like a second pair of eyes on: raadh@oxiedo.com. Licensed under Apache-2.0.
