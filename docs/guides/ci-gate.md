# `check` as a CI gate

`ormas-monitor check` reads weights only, needs no GPU and no data, and returns a non-zero exit code when it finds a problem. That makes it a cheap gate before a checkpoint is published or deployed.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | No finding at or above `--fail-on`. |
| `1` | At least one finding at or above `--fail-on`. The default is `critical` (for example NaN or Inf weights, or tensors missing from the shards). |
| `2` | The file could not be read, or was refused (for example a pickle file without `--trust`). |

`--fail-on` takes `critical` (default), `warning` or `never`. `--quiet` prints nothing except errors and `--json` output.

```bash
ormas-monitor check model.safetensors --fail-on warning --quiet    # also fail on dead or decayed channels
```

## GitHub Actions

```yaml
name: checkpoint health
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install "ormas-monitor[safetensors]"
      - run: ormas-monitor check artifacts/model.safetensors --against artifacts/base.safetensors --fail-on warning
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: health-report
          path: artifacts/*.health.html
```

## Machine-readable output

`--json` prints the full result, including every finding with its `severity`, `kind`, `title` and `detail`, so a script can apply its own policy:

```bash
ormas-monitor check model.pt --json > health.json
```

The finding kinds are listed in the [findings catalogue](../reference/check-findings.md).

## Things to know

- **Never pass `--trust` in CI on files from an untrusted source.** It allows pickle loading, which can run arbitrary code.
- **Pass `--arch` when you know the architecture.** Without it, the check cannot see how layers are wired and qualifies dead-channel findings with "if a ReLU follows".
- **Decayed channels are common in published models.** In our scan, 19 of 21 torchvision BN→ReLU models carried some. Failing on `warning` may be too strict for a model you did not train. Compare against a baseline with `--against` and gate on the `newly_dead` finding instead, for example with `--json` and a short script:

```bash
ormas-monitor check new.safetensors --against base.safetensors --json \
  | python -c "import json,sys; r=json.load(sys.stdin); sys.exit(any(f['kind']=='newly_dead' for f in r['findings']))"
```
