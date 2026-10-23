# Security and privacy

## No network in the core

The core package (the monitor, `check`, the record and the reports) makes **no network calls**. Network appears in exactly two places, both opt-in:

| Where | When |
|---|---|
| `ormas-monitor demo` | Downloads CIFAR-10 once, through torchvision. |
| Sinks that send data | `WandbSink`, `MLflowSink` with a remote tracking URI, `WebhookSink`, and the callbacks' `to_loggers=True` when the trainer reports to W&B or a remote MLflow. Only if you configure them. |

Reports are self-contained HTML: no fonts, scripts or images are fetched when you open one.

## What the monitor reads

- Gradient norms per layer, and the per-channel maximum of Conv/Linear/BatchNorm outputs on sampled steps. These are reduced on the device to one number per layer or channel.
- Weights and optimizer state of layers it checks or repairs.
- **Never your training data.** Inputs pass through the model as usual; the monitor never stores, copies or inspects them.

The record contains layer names, step numbers, gradient norms, channel indices, the model class name, parameter count, optimizer settings, Python and PyTorch versions and the platform string. It contains no weights and no data. Review `run_start` before sharing a record outside your organisation: layer names and the platform string can reveal details of your setup.

## Loading checkpoints safely

- PyTorch files load with `torch.load(..., weights_only=True)`, so a checkpoint cannot run code on your machine.
- A file that holds pickled Python objects is **refused** unless you pass `--trust` (or `trust=True`). Pass it only for files whose origin you trust. Never pass it in CI on untrusted input.
- `.safetensors` files cannot contain code.
- PyTorch zip-format files are memory-mapped (`mmap=True`) and still loaded with `weights_only=True`. Safetensors files and Hugging Face folders are read tensor by tensor.
- `--arch torchvision:<name>` builds the architecture with `weights=None`, and `--arch hf` builds it from the folder's `config.json` with `local_files_only=True` and random weights: neither downloads anything.

## The record's threat model

The hash chain makes unnoticed edits hard; a seal pins the file contents and length; an Ed25519 signature by a key the training machine does not hold makes unauthorised edits detectable. None of them proves the training happened as described. The full table is in [The training record](concepts/record.md#threat-model).

Signing keys: `keygen` writes the private key unencrypted with mode `0600` and never overwrites or writes through a symbolic link. `sign` refuses symlinked record files and never writes through a link. `verify` treats a record as untrusted input: malformed lines, hostile seal values and symlinked files fail a check instead of crashing it. Keep the private key off the training machine where you can, out of the run directory, and out of version control.

## Supply chain

- The core depends only on PyTorch.
- Every optional dependency is behind an extra and imported only when used.
- The release workflow is set up to build in GitHub Actions and publish to PyPI with Trusted Publishing (OIDC), so no long-lived PyPI token is needed. No release has been published through it yet. **Status: OPEN.**

## Reporting a vulnerability

Email **raadh@oxiedo.com** with "SECURITY" in the subject. Please do not open a public issue for a security problem. See [`SECURITY.md`](https://github.com/Oxiedo/ormas-monitor/blob/main/SECURITY.md) for what to include and what to expect.
