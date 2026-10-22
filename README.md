# ormas-monitor

**Catch the layer that breaks while your model trains, and get proof.**

[![CI](https://github.com/OWNER/ormas-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/ormas-monitor/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ormas-monitor.svg)](https://pypi.org/project/ormas-monitor/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`ormas-monitor` watches a PyTorch training run layer by layer. Each layer is compared with **its own** history, so the monitor names the layer that failed. When a channel can never recover, it says so with a **certificate**, not a guess. It can repair those channels if you ask, and it writes a tamper-evident record of everything it saw.

Everything runs on your machine. The core makes no network calls and never reads your data.

## Three ways in

**1. Check a saved model (about a minute, no code changes)**

```bash
pip install "ormas-monitor[safetensors]"
ormas-monitor check model.safetensors                  # or .pt / .pth / .bin / .ckpt, or a Hugging Face folder
ormas-monitor check finetuned.pt --against base.pt     # what changed, and what died
ormas-monitor check model.pt --fail-on warning         # non-zero exit for a CI gate
```

**2. Two lines in a training loop, or one callback**

```python
from ormas_monitor import Monitor

mon = Monitor(model, optimizer)        # repair=True to re-initialise certified-dead channels
for x, y in loader:
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    mon.step(loss)                     # after backward, before optimizer.step()
    optimizer.step()
mon.close()                            # writes ormas_run/report.html and the record
```

Using Lightning or the Hugging Face `Trainer`? Add one callback instead, `ormas_monitor.integrations.lightning.OrmasMonitorCallback` or `ormas_monitor.integrations.huggingface.OrmasMonitorCallback`: see [Lightning](docs/guides/lightning.md) and [Hugging Face](docs/guides/huggingface.md).

**3. Watch it catch a broken layer (about two minutes)**

```bash
pip install "ormas-monitor[demo]"
ormas-monitor demo        # CIFAR-10: two layers are zeroed; watch-only vs repair (downloads CIFAR-10 once)
```

Or open the real reports without installing anything: [`sample_reports/`](sample_reports).

## What it found (verified)

All numbers: one seed (0), one Apple M-series Mac, CIFAR-10 unless stated. Not yet measured on NVIDIA/Linux, multi-GPU or at scale.

| Experiment | Result |
|---|---|
| Dead-layer demo, full (50k images): conv2 and conv3 zeroed, optimizer state cleared | healthy 78.84% → watch-only **10.00%** for all 12 epochs → with repair **76.77%** (192 channels repaired) |
| Dead-layer demo, quick (`ormas-monitor demo`) | 61.76% → **10.00%** → **59.12%** (190 channels repaired) |
| Overhead at `check_every=50` | **+3.6%** median epoch time (12.12 s → 12.55 s) |
| `check` on torchvision ImageNet models | **19 of 21** BN→ReLU models carry channels decayed to ~0; DenseNet-169: 7,667 of 77,536 (9.9%) |
| The paper's own StandardCNN | 160 of 256 classifier hidden units never fire on any of the 10,000 test images |

Details and exact reproduce commands: [docs/results.md](docs/results.md).

## The certificate, in one paragraph

In a Conv/Linear → BatchNorm → ReLU channel, if the incoming filter or the BatchNorm scale is exactly zero and the BatchNorm shift is ≤ 0, the ReLU is closed for every input. Every gradient into the channel is then exactly zero, so SGD or AdamW can never revive it (Theorem 2 of the DeepMath 2026 paper below). The monitor issues the certificate only when a ReLU directly consumes the layer's output, the optimizer holds no first-moment momentum for the channel, and BatchNorm is in training mode. [Read more](docs/concepts/dead-channels.md).

## What it is not

- **Not ORMAS.** The ORMAS training architecture is not in this package; this tool needs no architecture change.
- **Not compliance.** The record is evidence for your technical documentation, not a certificate of compliance with the EU AI Act or anything else.
- **No proof for transformers or LLMs.** The certificate needs BN→ReLU. Transformers get alarms, loss checks and the record.
- **Never reads your data.** It cannot find mislabelled examples.
- **No uploads.** Nothing leaves your machine unless you add a sink that sends data (W&B, MLflow, a webhook).
- **Not the first to notice dead units.** ReDo (Sokar et al., ICML 2023) and network slimming came first. We claim the certificate and the packaging.
- **Not yet tested on NVIDIA/Linux, multi-GPU or at scale.**

Full list: [docs/limitations.md](docs/limitations.md).

## Install

```bash
pip install ormas-monitor                     # core: depends only on PyTorch
pip install "ormas-monitor[safetensors]"      # read .safetensors files
pip install "ormas-monitor[signing]"          # Ed25519 signing of the record
pip install "ormas-monitor[lightning]"        # Lightning callback
pip install "ormas-monitor[huggingface]"      # Hugging Face Trainer callback
pip install "ormas-monitor[tensorboard]"      # also: [wandb], [mlflow]
pip install "ormas-monitor[demo]"             # torchvision, for `ormas-monitor demo`
pip install "ormas-monitor[all]"
```

Python 3.9+, PyTorch 2.3+.

## Documentation

[Quickstart](docs/quickstart.md) · [Concepts](docs/concepts/dead-channels.md) · [Guides](docs/guides/pytorch.md) · [CLI reference](docs/reference/cli.md) · [Python API](docs/reference/python-api.md) · [For leaders](docs/for-leaders.md) · [Security and privacy](docs/security-and-privacy.md) · [FAQ](docs/faq.md)

Hosted: https://OWNER.github.io/ormas-monitor/

## Citation

If you use the dead-channel certificate, please cite the paper ([`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{raadh2026selfrepair,
  author    = {Raadh, Rokib Al Dhin},
  title     = {Self-Repair as a Bounded Disturbance: Input-to-State Stability of Neural Network Training Dynamics},
  booktitle = {DeepMath 2026},
  address   = {The Ohio State University, Columbus, OH},
  year      = {2026},
  note      = {Poster}
}
```

## Contact

Built by Rokib Al Dhin Raadh at [Oxiedo](https://oxiedo.com). Questions, bugs, or a run you'd like a second pair of eyes on: raadh@oxiedo.com. Security issues: see [SECURITY.md](SECURITY.md).

Licensed under [Apache-2.0](LICENSE).
