# CLI reference

```text
ormas-monitor [--version] {check,verify,report,demo,keygen,sign,diff} ...
```

Every command runs locally. Only `demo` uses the network, to download CIFAR-10 once.

## `check`

Health-check a saved model file or Hugging Face folder. Weights only: no training, no data. Tensors are read one at a time.

```text
ormas-monitor check [--against AGAINST] [--out OUT] [--json] [--arch ARCH] [--trust]
                    [--fail-on {critical,warning,never}] [--quiet] file
```

| Option | Meaning |
|---|---|
| `file` | A `.pt`, `.pth`, `.bin`, `.ckpt` or `.safetensors` file, a sharded `*.index.json` (safetensors or `.bin`), or a Hugging Face model folder. Wrapper dicts (`state_dict`, `model_state_dict`, `model`, `net`, `module`, `ema`) are unwrapped and `module.` / `_orig_mod.` prefixes removed. |
| `--against AGAINST` | An earlier checkpoint (file or folder). Adds which tensors changed most, which never moved, which were added, removed or reshaped, and which channels died in between. |
| `--out OUT` | HTML report path. Default: `<file>.health.html`. |
| `--json` | Print the full result as JSON on stdout (the summary then goes to stderr). |
| `--arch ARCH` | Exact wiring: `torchvision:<name>` (e.g. `torchvision:resnet50`, needs torchvision), or `hf` to build the model from the folder's `config.json` (random weights, local files only, no download; tracing works for image models, others fall back and say why). Without it, dead-channel findings say "if a ReLU directly follows". The model is built once to trace it: about a second on the reference Mac, longer on the first run after installing torchvision. If under half of the file's BatchNorm layers are in the model (or of the model's in the file), the check stops with an error (exit `2`); a partial match adds an `arch_partial_match` warning. |
| `--trust` | Allow loading files that contain pickled Python objects. **This can run code.** Use only for files whose origin you trust. |
| `--fail-on` | Lowest severity that makes the exit code 1: `critical` (default), `warning` or `never`. |
| `--quiet` | Print nothing except errors and `--json` output. |

**Exit codes:** `0` no finding at or above `--fail-on`; `1` at least one; `2` the file could not be read or was refused, or `--arch` does not match it. Unless `--quiet`, the last line explains the code, e.g. `exit 0: no critical findings (1 warning; use --fail-on warning to fail on them)`.

To measure time and memory on your machine: `python -m ormas_monitor.checkpoint.bench`. See [Results](../results.md#4-checkpoint-check-time-and-memory).

## `verify`

Check that a run's record is complete and has not been edited.

```text
ormas-monitor verify [--key PUBLIC.pem] [--strict] [--json] run
```

| Option | Meaning |
|---|---|
| `run` | A run directory, or a lone `diary.jsonl`. |
| `--key PUBLIC.pem` | The signer's public key. A valid signature by this key is then required. |
| `--strict` | Also fail when a part is absent or unchecked: unsealed, unsigned, no metrics. |
| `--json` | Print the checklist as JSON. |

**Exit codes:** `0` no check failed (with `--strict`: every check passed); `1` a check failed; `2` the run could not be read. See [The training record](../concepts/record.md#verification) for the checklist.

## `report`

Rebuild the reports of a monitored run from its record and metrics: the engineer report (`report.html`) and the executive summary (`summary.html`, `summary.md`).

```text
ormas-monitor report [--gpu-hour-cost PRICE] [--gpus N] [--key PUBLIC.pem] run
```

| Option | Meaning |
|---|---|
| `run` | The run directory (holds `diary.jsonl`). |
| `--gpu-hour-cost PRICE` | Price of one GPU-hour. Adds a dollar **ESTIMATE** of time spent stuck to the summary. Without it, no dollar figure is printed. |
| `--gpus N` | GPUs the run used. Default 1 when a price is given, and the summary says it was assumed. |
| `--key PUBLIC.pem` | The signer's public key, so the reports verify a signed record instead of reporting the signature as unchecked. |

Prints the path of each report written.

## `demo`

The CIFAR-10 dead-layer experiment: watch-only vs repair. Needs the `demo` extra (torchvision) and downloads CIFAR-10 (about 170 MB) once.

```text
ormas-monitor demo [--out OUT] [--data DATA] [--full]
```

| Option | Meaning |
|---|---|
| `--out OUT` | Where the two runs are written. Default `ormas_demo/` (`watch/` and `repair/`). Re-running replaces them. |
| `--data DATA` | Where CIFAR-10 is, or gets downloaded to. Default `data/`. |
| `--full` | All 50,000 training images, 8 epochs before the lesion and 12 after. Default: 10,000 images, 4 + 4. |

## `keygen`

Create an Ed25519 key pair for signing records. Needs the `signing` extra.

```text
ormas-monitor keygen [--private PATH] [--public PATH]
```

Writes an unencrypted private key with mode `0600` on macOS and Linux (default `ormas-signing.pem`) and its public key (default `ormas-signing.pub.pem`), both PEM. Never overwrites an existing file. Prints the key fingerprint (SHA-256 of the raw public key).

## `sign`

Seal a finished run and sign the seal.

```text
ormas-monitor sign (--key PRIVATE.pem | --seal-only) run
```

Writes `record.json` (the SHA-256 of `diary.jsonl` and `metrics.jsonl`, the diary head and length) and, with `--key`, `record.sig` (Ed25519 over the exact bytes of `record.json`). `--seal-only` writes the seal without a signature: a checksum, not proof of who sealed it. Refuses a run that does not verify, and refuses to write through a symbolic link (`record.json`, `record.sig`) or to seal symlinked record files. Exit `2` on refusal.

## `diff`

Compare two run records: what changed since the run you approved.

```text
ormas-monitor diff [--out diff.html] [--json] RUN_A RUN_B
```

| Option | Meaning |
|---|---|
| `RUN_A` | The approved (baseline) run directory or diary. |
| `RUN_B` | The proposed run directory or diary. |
| `--out diff.html` | Also write an HTML diff report. |
| `--json` | Print the diff result as JSON. |

Uses the diaries and metrics only, never the weights. Prints a verdict (integrity first, then the net per-layer result), a detail line, changed settings and a per-layer table; see [Comparing two runs](../concepts/record.md#comparing-two-runs). To compare weights, use `check --against`.
