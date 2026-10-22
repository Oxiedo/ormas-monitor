# Security policy

## Reporting a vulnerability

Email **raadh@oxiedo.com** with "SECURITY: ormas-monitor" in the subject. Please do not open a public GitHub issue for a security problem.

Include, where you can:

- the version (`ormas-monitor --version`) and your Python, PyTorch and OS versions;
- what an attacker can do, and what they need (for example: a crafted checkpoint file, write access to a run directory);
- a minimal reproduction.

We aim to acknowledge a report within 5 working days. We will agree a disclosure date with you, credit you in the release notes unless you prefer otherwise, and publish a fix and an advisory on GitHub.

## Supported versions

Security fixes go into the latest minor release. Before 1.0, older minor versions are not patched.

## In scope

- **Code execution from loading a file** with `ormas-monitor check` or `ormas_monitor.checkpoint` without `--trust`. PyTorch files are loaded with `weights_only=True`; a bypass is a vulnerability.
- **Forging a record that verifies**: a modified run directory that passes `ormas-monitor verify --key` for a key the attacker does not hold, or a modified, sealed record that passes the seal check.
- **Unexpected network access** from the core package (the monitor, `check`, the record, the reports). The core is designed to make no network calls.
- **Data in reports or the record** that should not be there, such as training data or weights.
- **HTML injection** in reports through layer names, file names or event fields.

## Out of scope

- Anything that requires `--trust` (`trust=True`): that flag allows pickle loading by design, which can run code.
- Rewriting an *unsigned* record and recomputing its hash chain. The hash chain alone does not claim to resist this; see the threat model in `docs/concepts/record.md`.
- Misuse of a signing key by someone who holds it.
- Network traffic from sinks you configured (W&B, MLflow, webhooks) or from the demo's CIFAR-10 download.
- Vulnerabilities in PyTorch or other dependencies; please report those upstream.
