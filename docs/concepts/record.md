# The training record

Every event the monitor writes goes to `diary.jsonl` in the run directory: one JSON object per line, each sealed with the SHA-256 of the line before it. The file is the **training record**. It is evidence of what the monitor observed, for your own technical documentation. It is **not** a compliance certificate.

## What it contains

- `run_start`: the tool version, model class, parameter count, a SHA-256 of the architecture (parameter names and shapes), optimizer settings, Python, PyTorch, platform and the monitor configuration.
- One event per thing that happened: alarms, stuck channels, certificates and refused certificates, repairs, loss spikes, non-finite values, checks without gradients.
- `run_end`: step and check counts, totals, and the SHA-256 of `metrics.jsonl`, which binds the metrics file to the record.

Every event type and its fields are listed in the [record format reference](../reference/record-format.md).

## How the chain works

Each line carries `prev`, the hash of the previous line (64 zeros for the first), and `hash`, the SHA-256 of `prev` followed by the canonical JSON of the line's other fields (`sort_keys=True`, separators `,` and `:`). Changing, deleting, inserting or reordering any line breaks every hash after it.

A broken chain is reported at the first line where it fails. See [Verification](#verification) below.

## Verification

```bash
ormas-monitor verify ormas_run                         # exit 0 when no check failed
ormas-monitor verify ormas_run --key reviewer.pub.pem  # also require a valid signature by this key
ormas-monitor verify ormas_run --strict                # also fail when a part is absent (unsealed, unsigned)
ormas-monitor verify ormas_run --json
```

Real output for the signed sample run in `sample_reports/` (run from that folder), verified with its public key (`--key`):

```text
ormas-monitor verify cifar10_with_repair_signed  (37 events)
  [ok  ] JSON lines              37 events parsed
  [ok  ] genesis link            first event links to the genesis hash
  [ok  ] hash chain              all 37 hashes recompute and link
  [ok  ] sequence numbers        seq runs 0..36
  [ok  ] run_start / run_end     run_start is first and run_end is last: the run finished
  [ok  ] event schema            all 37 events match record schema 1
  [ok  ] metrics hash            sha256 93cca3ff1854... as recorded in run_end
  [ok  ] seal (record.json)      diary.jsonl, metrics.jsonl and the diary head match the seal of 2026-10-23T06:47:26Z
  [ok  ] signature (record.sig)  valid Ed25519 signature over record.json by key deb089ff782e...
PASSED: 9 of 9 checks passed
This record is evidence of what the monitor wrote, not a compliance certificate.
```

A record written by ormas-monitor 0.1, which predates the event schema, shows `[skip] event schema` instead.

Each check has one of five statuses: `ok` (pass), `FAIL`, `--` (absent: the part does not exist, for example an unsealed run), `??` (unchecked: it exists but could not be checked as asked, for example a signature with no key given) and `skip` (the check could not run). Without `--strict`, absent and unchecked parts do not fail verification. A lone `diary.jsonl` file can be verified too; its metrics, seal and signature are then reported as absent. Without `--key`, a present signature is `??` (unchecked) and the run passes 8 of 9; an unsealed, unsigned run passes 7 of 9 with 2 absent.

Verification is built to fail cleanly on a hostile record: malformed lines, odd seal values, a seal naming files outside the run directory, symlinked record files and a FIFO in place of a file all produce a failed check, never a crash or a hang.

## Sealing and signing

A **seal** (`record.json` in the run directory) names the SHA-256 of `diary.jsonl` and `metrics.jsonl`, the hash of the last diary line and the number of events. A **signature** (`record.sig`) is a detached Ed25519 signature over the exact bytes of `record.json`. Both refuse a record that does not verify, and both refuse to write through a symbolic link (a planted `record.json` link cannot redirect the write to another file).

```bash
pip install "ormas-monitor[signing]"
ormas-monitor keygen --private reviewer.pem --public reviewer.pub.pem   # defaults: ormas-signing.pem / .pub.pem
ormas-monitor sign ormas_run --key reviewer.pem                         # writes record.json and record.sig
ormas-monitor sign ormas_run --seal-only                                # record.json only, no key
```

Keys are PEM files. The private key is written unencrypted with mode `0600` on macOS and Linux (Windows has no such permission bits), and `keygen` never overwrites an existing file. Keep it out of the run directory and out of version control.

The format can be checked without this package: `shasum -a 256` for the files, and, with OpenSSL 3,

```bash
openssl pkeyutl -verify -pubin -inkey reviewer.pub.pem -rawin -in ormas_run/record.json -sigfile ormas_run/record.sig
```

!!! tip "Who should hold the key"
    A key kept on the machine that trains proves little more than the hash chain does, because whoever can edit the record can also use the key. A key held by a reviewer, who signs after inspecting the run, proves much more.

## Comparing two runs

```bash
ormas-monitor diff runs/approved runs/candidate                   # text summary
ormas-monitor diff runs/approved runs/candidate --out diff.html   # also an HTML report
ormas-monitor diff runs/approved runs/candidate --json
```

The diff reads the two records only, never the weights. It opens with a verdict: integrity first (whether both records verify), then the net per-layer result, for example "Run B is healthier in 2 layers ... and has more silent channels in 1". A detail line follows: whether the architectures match, which optimizer and monitor settings changed, how health is judged, and repairs. Then come the setting changes and a per-layer table of alarms, stuck channels, silent channels, certificates and repairs. A layer's health is judged by channels stuck according to the weights (certified dead, constant or decayed) and open alarms; silent channels are reported separately and never make a layer less healthy on their own. In `--json` the headline is `verdict` and the detail `verdict_detail`.

## Threat model

Be precise about what the record can and cannot show.

| Mechanism | Detects | Does not detect |
|---|---|---|
| **Hash chain** (always on) | Editing, deleting, inserting or reordering any line: verification fails at that line. | Someone who rewrites the whole diary and recomputes every hash. Lines cut from the end: a truncated chain is still a valid chain. |
| **Seal** (`record.json`) | Truncation, and edits to `metrics.jsonl`, as long as the seal itself is trusted. | Anyone who can write the run directory can also write a new seal. An unsigned seal is a checksum, not evidence of who made it. |
| **Signature** (`record.sig`) | Any change to the sealed files after signing, unless the private key was used again. | A misused or stolen private key. The signature proves the key holder signed these bytes; it does not prove the training happened as recorded. |

And for all three:

- **The training process is trusted.** The record is written by the process that trains. If that process, the monitor or the machine is modified, the record faithfully seals what it was told.
- **Clocks are local.** `time` comes from the machine's clock. There is no trusted timestamp.
- **It covers only what the monitor observes.** Not the data, the labels, evaluation results, or anything outside the monitored steps.

In short: the chain makes unnoticed edits hard; a signature by a reviewer makes unauthorised edits detectable; nothing here proves the training happened as described. Treat the record as one piece of evidence, alongside your code, configs, data documentation and evaluation.

!!! note "Not compliance"
    Regulations such as the EU AI Act ask providers of some AI systems to keep technical documentation and logs. A training record can be part of that documentation. Using `ormas-monitor` does not make a system compliant with anything, and no regulator has reviewed this record format. See [For leaders](../for-leaders.md).
