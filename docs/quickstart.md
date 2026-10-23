# Quickstart

Three ways in. Each takes a few minutes and none needs an account. Apart from `pip install`, only the demo uses the network, to download CIFAR-10 (about 170 MB) once.

```bash
pip install ormas-monitor                  # core: depends only on PyTorch
pip install "ormas-monitor[safetensors]"   # if your files are .safetensors
```

See [Install extras](#install-extras) for the integrations.

## 1. Check a saved model

No training, no data, no code changes. The check reads weights only.

```bash
ormas-monitor check model.pt                            # .pt / .pth / .bin / .ckpt / .safetensors
ormas-monitor check ./my-hf-model --arch hf             # a Hugging Face folder, sharded or not
ormas-monitor check finetuned.pt --against base.pt      # what changed, and what died, between two
ormas-monitor check model.pt --arch torchvision:resnet50   # use the exact wiring of a known architecture
```

Real output from a small CNN in which five channels were zeroed between two checkpoints:

```text
finetuned.pt: 1,488 parameters in 14 tensors (pytorch, float32, int64); wiring: name-based heuristic; 3 findings in 0.0 s
  [warning] 5 channels died between the two checkpoints
  [warning] 5 of 16 channels in layer `3` (Conv2d, BatchNorm `4`) are dead (zero filter or scale, shift <= 0) if a ReLU directly follows `4`
  [info] 10 of 12 tensors are exactly unchanged
  note: Wiring not known, so BatchNorm findings say "if a ReLU directly follows". Pass --arch torchvision:<name> or --arch hf for exact wiring.
report: finetuned.pt.health.html
exit 0: no critical findings (2 warnings; use --fail-on warning to fail on them)
```

(An `nn.Sequential` of Conv2d–BatchNorm2d–ReLU twice, ormas-monitor 0.2.0.) The last line explains the exit code: by default only `critical` findings fail the check. The HTML report lists every finding with its explanation. The phrase "if a ReLU directly follows" appears because a bare state dict does not record how layers are wired. Pass `--arch` when the architecture is known, and the check removes the condition. `--arch` builds the model once to trace it: on the reference Mac that adds about a second, but the first run after installing torchvision can take much longer while Python compiles it. If the file does not match the architecture (under half of its BatchNorm layers found), the check stops with an error (exit 2) rather than guess. Every finding is described in the [findings catalogue](reference/check-findings.md).

PyTorch files load with `weights_only=True`, so opening a checkpoint cannot run code on your machine. A file that holds pickled Python objects is refused unless you pass `--trust`.

The exit code is `1` when a finding at or above `--fail-on` is present (default `critical`), `2` when the file cannot be read, and `0` otherwise, which makes the check usable as a [CI gate](guides/ci-gate.md).

## 2. Watch a training run

Two lines in a plain PyTorch loop:

```python
from ormas_monitor import Monitor

mon = Monitor(model, optimizer)          # repair=True to fix certified-dead channels
for x, y in loader:
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    mon.step(loss)                       # after backward, before optimizer.step()
    optimizer.step()
mon.close()                              # writes ormas_run/report.html and the record
```

`Monitor` also works as a context manager (`with Monitor(model, optimizer) as mon:`), which calls `close()` on exit.

Call `step()` after `backward()`: if no watched layer has a gradient at a check, the monitor warns once (`no_gradients`) and the reports say NOT MONITORED rather than healthy.

When something goes wrong, the monitor prints one line per event. This is real output (ormas-monitor 0.2.0, torch 2.11, CPU) from a small CNN, two Conv-BatchNorm-ReLU blocks and a linear head, trained with AdamW on synthetic data; at step 60 the second conv layer was zeroed and the optimizer state cleared, with `repair=True`, `check_every=5` and `out="run"`:

```text
[ormas-monitor] step 60: 3 (Conv2d): 16/16 channels stuck: 16 certified dead
[ormas-monitor] step 65: 0 (Conv2d): gradient is exactly zero (0 vs normal 0.178)
[ormas-monitor] step 65: 1 (BatchNorm2d): gradient is exactly zero (0 vs normal 0.0226)
[ormas-monitor] step 65: 3 (Conv2d): gradient is exactly zero (0 vs normal 0.0798)
[ormas-monitor] step 65: 4 (BatchNorm2d): gradient is exactly zero (0 vs normal 0.0994)
[ormas-monitor] step 65: 3: re-initialised 16 stuck channels
[ormas-monitor] step 70: 0: back to normal
[ormas-monitor] step 70: 1: back to normal
[ormas-monitor] step 70: 3: back to normal
[ormas-monitor] step 70: 4: back to normal
[ormas-monitor] step 70: 3: all channels active again
[ormas-monitor] report written to run/report.html
[ormas-monitor] report written to run/summary.html
[ormas-monitor] report written to run/summary.md
```

The first conv layer's gradient also drops to zero: with every channel of the second layer dead, nothing flows back past it.

The run directory (`ormas_run` by default, `run` here) then holds:

| File | What it is |
|---|---|
| `diary.jsonl` | The training record: one hash-chained event per line. See [the record](concepts/record.md). |
| `metrics.jsonl` | One row per check: loss, per-layer gradient norm, stuck channels per layer. |
| `report.html` | The engineer report. |
| `summary.html`, `summary.md` | The plain-language executive summary. See [For leaders](for-leaders.md). |

Starting a new `Monitor` on a directory that already holds a record raises `FileExistsError`, so a finished (possibly signed) record is never overwritten by accident. Choose another `out=`, or pass `overwrite=True` to replace it.

Check that the record has not been edited:

```bash
ormas-monitor verify ormas_run
# ... one line per check ...
# PASSED: 7 of 9 checks passed (2 absent)
```

The two absent checks are the seal and the signature, which exist only after `ormas-monitor sign`; absent parts do not fail verification unless you pass `--strict`. After signing, the last line reads `PASSED: 9 of 9 checks passed` when you pass the signer's public key with `--key`.

The full checklist, sealing and signing are in [The training record](concepts/record.md).

### With Lightning or the Hugging Face Trainer

Add a callback instead of editing the loop:

=== "Lightning"

    ```python
    from ormas_monitor.integrations.lightning import OrmasMonitorCallback

    trainer = L.Trainer(callbacks=[OrmasMonitorCallback(out="ormas_run")])
    ```

=== "Hugging Face Trainer"

    ```python
    from ormas_monitor.integrations.huggingface import OrmasMonitorCallback

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      callbacks=[OrmasMonitorCallback(out="ormas_run")])
    ```

Full guides: [Lightning](guides/lightning.md), [Hugging Face Trainer](guides/huggingface.md).

## 3. Run the demo

```bash
pip install "ormas-monitor[demo]"
ormas-monitor demo                 # quick: 10,000 images, 4 + 4 epochs
ormas-monitor demo --full          # all 50,000 images, 8 + 12 epochs
```

The demo trains the paper's CIFAR-10 CNN, zeroes two of its three conv layers and clears the optimizer state, then trains on twice from that identical state: once with the monitor only watching, once with repair. With seed 0, torch 2.11 and an Apple M-series GPU (MPS), the quick demo ends at 61.76% healthy, **10.00%** watch-only, **59.12%** with repair (190 channels repaired). The exact numbers vary with the torch version and hardware: a fresh install that resolved torch 2.14.1 got 61.96% / 10.00% / 62.67% (191 repaired). The qualitative result holds: the watch-only run stays at chance and the repaired run recovers. See [Results](results.md).

Pass `--data DIR` to point at an existing CIFAR-10 folder (default `data/`), and `--out DIR` to choose where the two runs are written (default `ormas_demo/`; re-running the demo replaces them). The quick demo took about 50 s on the reference Mac with CIFAR-10 already downloaded.

Or skip the demo and open the real reports in [`sample_reports/`](https://github.com/OWNER/ormas-monitor/tree/main/sample_reports).

## Install extras

| Extra | Adds | For |
|---|---|---|
| `safetensors` | `safetensors` | reading `.safetensors` files |
| `signing` | `cryptography` | Ed25519 signing of the record |
| `lightning` | `lightning` | the Lightning callback |
| `huggingface` | `transformers`, `accelerate`, `safetensors` | the Trainer callback, HF model folders |
| `tensorboard`, `wandb`, `mlflow` | the client library | sending events to those tools |
| `demo` | `torchvision` | `ormas-monitor demo` |
| `all` | everything above | |
| `dev` | `pytest`, `ruff`, `build`, `twine`, `mkdocs-material` | contributing |

```bash
pip install "ormas-monitor[lightning,tensorboard]"
```
