# `check` as a CI gate

`ormas-monitor check` reads weights only, needs no GPU and no data, and returns a non-zero exit code when it finds a problem. That makes it a cheap gate before a checkpoint is published or deployed.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | No finding at or above `--fail-on`. |
| `1` | At least one finding at or above `--fail-on`. The default is `critical` (for example NaN or Inf weights, or tensors missing from the shards). |
| `2` | The file could not be read, or was refused (for example a pickle file without `--trust`). |

`--fail-on` takes `critical` (default), `warning` or `never`. `--quiet` prints nothing except errors and `--json` output.

**For a CI gate, use `--fail-on warning`.** The default, `critical`, fails only on a broken file (NaN or Inf weights, missing shards); dead channels, decayed channels and channels that died since the baseline are warnings, so with the default the check prints them and still exits `0`. The last line of the output says why, for example `exit 0: no critical findings (2 warnings; use --fail-on warning to fail on them)`.

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
- **Pass `--arch` when you know the architecture.** Without it, the check cannot see how layers are wired and qualifies dead-channel findings with "if a ReLU directly follows". A wrong `--arch` (under half of the file's BatchNorm layers found) fails with exit `2`. The first `--arch torchvision:` run after installing torchvision is slow while Python compiles it; later runs add about a second.
- **Decayed channels are common in published models.** In our scan, 19 of 21 torchvision BN→ReLU models carried some. Failing on `warning` may be too strict for a model you did not train. Compare against a baseline with `--against` and gate on the `newly_dead` finding instead, for example with `--json` and a short script:

```bash
ormas-monitor check new.safetensors --against base.safetensors --json \
  | python -c "import json,sys; r=json.load(sys.stdin); sys.exit(any(f['kind']=='newly_dead' for f in r['findings']))"
```
