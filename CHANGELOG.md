# Changelog

All notable changes to `ormas-monitor` are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) (before 1.0, a minor release may change the API).

## [Unreleased] — 0.2.0

### Core engine
- `MonitorConfig`: every threshold in one frozen, validated dataclass, stored whole in `run_start.config`. The defaults reproduce 0.1 behaviour. The old keyword arguments (`check_every`, `patience`, `sigma`, `warmup`) remain as shortcuts.
- `ormas_monitor.schema`: every event type and its required fields; `record_version` in `run_start`.
- Mixed precision: `Monitor(..., scaler=GradScaler)`; steps skipped by the scaler become `skipped_step` events instead of non-finite alarms; gradient norms are taken in float32.
- Distributed training: only rank 0 writes the record, prints and calls sinks; under DDP, repair uses a seeded generator so every rank writes the same weights; under FSDP, channel checks are disabled and repair is refused. Covered by CPU tests only, not yet run on multi-GPU hardware.
- `layers=` to restrict which layers are watched, for large models.
- `manifest` event: which mechanisms were active (channel checks, repair, precision, distributed) and why any were off.
- `note_loss()` for code that sees the loss and the optimizer step in different places.
- Repair moved to `ormas_monitor.repair`: reaches the real optimizer through framework wrappers, optional deterministic re-initialisation, and logs the measured size of each repair (`delta_norm`, `relative_change`) with `modification_class: "discrete"`.
- The certificate now also requires the optimizer (for the momentum condition) and BatchNorm in training mode, as its statement assumes.

### Integrations
- Lightning callback: `ormas_monitor.integrations.lightning.OrmasMonitorCallback`.
- Hugging Face `Trainer` callback: `ormas_monitor.integrations.huggingface.OrmasMonitorCallback` (tested with transformers 4.57; sees gradients after clipping).
- Runnable examples in `examples/integrations/`.
- Sinks: `TensorBoardSink`, `WandbSink`, `MLflowSink`, `WebhookSink`, `LoggingSink`, `ConsoleSink`. Each optional dependency is imported only when its sink is constructed.

### Training record
- `ormas-monitor verify` now runs a full checklist (chain, sequence, run_start/run_end, schema, metrics hash, seal, signature) with `--key`, `--strict` and `--json`.
- `ormas-monitor keygen` and `ormas-monitor sign`: an Ed25519-signed seal (`record.json`, `record.sig`). Needs the `signing` extra.
- `ormas-monitor diff RUN_A RUN_B`: what changed between an approved run and a proposed one, from the records alone.
- `ormas_monitor.record` package: `verify_run`, `seal`, `sign`, `keygen`, `diff_runs`. `ormas_monitor.diary` stays importable.

### Reports
- Executive summary for a CEO, CFO or compliance lead (`summary.html`, `summary.md`), with an opt-in cost **estimate** from a GPU-hour price and GPU count.
- Engineer report: time spent stuck, and lead time from the first signal to a loss spike.
- HTML diff report for `ormas-monitor diff --out`.
- Print stylesheet for every report.

### Checkpoint check
- Hugging Face model folders, sharded safetensors and `.bin` indexes, `.ckpt`, and streaming reads: one 1.78 GiB sharded checkpoint checked with a 220 MB peak footprint instead of 2,039 MB (one Apple-silicon Mac, one run; `python -m ormas_monitor.checkpoint.bench`).
- `--json` writes only JSON to stdout.
- `--arch hf` and findings relevant to transformers: `norm_scale_zero`, `identical_weights`, `frozen_tensors`, `embedding_rows_unchanged`, `missing_tensors`.
- `--fail-on {critical,warning,never}` and `--quiet`, for CI gates.

### Docs and packaging
- Documentation site (MkDocs Material): quickstart, concepts, guides, reference, a page for leaders, results with reproduce commands, limitations, security and privacy.
- Version single-sourced from `ormas_monitor.__version__`; extras `safetensors`, `signing`, `lightning`, `huggingface`, `tensorboard`, `wandb`, `mlflow`, `demo`, `all`, `dev`; `py.typed`.
- Minimum PyTorch raised from 1.13 to 2.3 (the device-generic `torch.amp.GradScaler`).
- CI on Linux, macOS and Windows with Python 3.9 and 3.12; docs deployment; PyPI release with Trusted Publishing.
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, `NOTICE`, issue and pull-request templates.

## [0.1.0] — 2026-10-22

First release.

- `Monitor`: per-layer gradient alarms against each layer's own baseline; loss-spike and non-finite checks; the dead-channel certificate for Conv/Linear → BatchNorm → ReLU (Theorem 2, DeepMath 2026), withheld when a skip connection precedes the ReLU or the optimizer holds momentum for the channel; constant, decayed and silent channels; optional repair (`repair=True` or `"all"`).
- Hash-chained record (`diary.jsonl`) and `metrics.jsonl`; `ormas-monitor verify`.
- `ormas-monitor check`: weights-only health check of `.pt`, `.pth`, `.bin` and `.safetensors` files, `--against` diff, `--arch torchvision:<name>`, `weights_only=True` loading with `--trust` to opt out.
- HTML run and checkpoint reports.
- `ormas-monitor demo`: the CIFAR-10 dead-layer experiment, watch-only vs repair.
- Examples: full demo, overhead measurement, torchvision scan.

[Unreleased]: https://github.com/OWNER/ormas-monitor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/OWNER/ormas-monitor/releases/tag/v0.1.0
