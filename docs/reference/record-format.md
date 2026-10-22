# Record format

The training record is `diary.jsonl` in the run directory: UTF-8, one JSON object per line. The authoritative list of event types and required fields is `ormas_monitor/schema.py`; this page mirrors it. A line may carry more fields than listed; it may not carry fewer.

## Common fields

Every line has:

| Field | Type | Meaning |
|---|---|---|
| `seq` | int | Position in the record, from 0, contiguous. |
| `time` | float | Unix time from the local clock, rounded to milliseconds. |
| `event` | string | The event type, from the table below. |
| `prev` | hex string | SHA-256 of the previous line (`hash` of line `seq - 1`), or 64 zeros for the first line. |
| `hash` | hex string | SHA-256 of `prev` followed by the canonical JSON of the line without `prev` and `hash`. |

**Canonical JSON** means `json.dumps(body, sort_keys=True, separators=(",", ":"))`. To verify one line by hand:

```python
import hashlib, json
body = {k: v for k, v in line.items() if k not in ("prev", "hash")}
assert line["hash"] == hashlib.sha256(
    (line["prev"] + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
```

`run_start.record_version` (currently `1`) changes when a field is removed or its meaning changes. Adding an event type or an optional field does not change it.

## Event types

| Event | Required fields beyond the common ones | Fixed values |
|---|---|---|
| `run_start` | `name, tool, model, parameters, layers, architecture_sha256, optimizer, python, torch, platform, config, record_version` | |
| `structure` | `step, pairs, relu_after` | |
| `manifest` | `step, check_every, layers, producers, channel_checks, repair, precision, distributed, notes` | |
| `alarm` | `step, severity, layer, kind, diagnosis, grad_norm, baseline, sigma` | `severity = "warning"` |
| `alarm_cleared` | `step, layer, diagnosis, grad_norm, baseline` | |
| `channels` | `step, severity, layer, batchnorm, channels, level, n_certified, n_flat, n_observed, n_decayed, certified, flat, observed` | |
| `channels_cleared` | same as `channels` | |
| `certificate` | `step, severity, layer, batchnorm, channels, count, statement, assumes` | `severity = "proof"` |
| `repair` | `step, severity, layer, batchnorm, channels, count, method, optimizer_slices_reset, modification_class, bound, delta_norm, weight_norm_before, weight_norm_after, relative_change, rng` | `severity = "action"`, `modification_class = "discrete"`, `bound = null` |
| `loss_spike` | `step, severity, value, baseline` | `severity = "warning"` |
| `loss_recovered` | `step, value, baseline` | |
| `nonfinite_loss` | `step, severity, value` | `severity = "critical"` |
| `nonfinite_gradient` | `step, severity, layers` | `severity = "critical"` |
| `skipped_step` | `step, severity, scale, layers` | `severity = "info"` |
| `sink_error` | `sink, error` | |
| `run_end` | `steps, checks, counts, metrics_sha256` | |

## Field meanings

| Field | Meaning |
|---|---|
| `step` | The optimizer step at which the check ran, counting from 0. |
| `severity` | One of `info`, `warning`, `action` (a repair), `proof` (a certificate), `critical` (non-finite values). |
| `layer` | The module name from `model.named_modules()`. |
| `kind` | The module class name, e.g. `Conv2d`. |
| `diagnosis` | `no_gradient`, `vanishing` or `exploding`. See [Alarms](../concepts/alarms.md). |
| `grad_norm`, `baseline` | The layer's gradient norm at this check, and its baseline (10 to the running mean of log₁₀ gradient norm). |
| `batchnorm` | The BatchNorm paired with the layer, or `null`. |
| `channels` | In `channels` events: the layer's channel count. In `certificate` and `repair`: the channel indices (truncated to `channel_list_limit`). |
| `level` | How many of a layer's channels are stuck: `none`, `some`, `most` (half or more) or `all`. |
| `n_certified`, `n_flat`, `n_observed`, `n_decayed` | Counts of certified-dead, constant, silent and decayed channels. |
| `certified`, `flat`, `observed` | The channel indices of each (truncated to `list_limit`). |
| `statement`, `assumes` | The certificate's claim in words, and the assumptions it rests on. |
| `method` | How the repair re-initialised the channels. |
| `optimizer_slices_reset` | How many optimizer-state tensors had the repaired channels zeroed. |
| `modification_class`, `bound` | Always `"discrete"` and `null`: repair is a jump with no enforced bound on its size. See [Repair](../concepts/repair.md#why-repair-is-a-discrete-modification). |
| `delta_norm`, `weight_norm_before`, `weight_norm_after`, `relative_change` | The measured size of the repair: ‖ΔW‖, ‖W‖ before and after, and ‖ΔW‖ / ‖W‖ (`null` when ‖W‖ was 0). |
| `rng` | Where the new weights came from: the global RNG, or a seeded generator. |
| `pairs` | `{producer: batchnorm}` as learnt from the first forward pass. |
| `relu_after` | The layers whose output a ReLU module consumes unmodified. |
| `architecture_sha256` | SHA-256 of `name:shape` for every parameter, joined by `|`. Two runs with the same value have the same parameter names and shapes. |
| `metrics_sha256` | SHA-256 of `metrics.jsonl` at the end of the run. |

## `metrics.jsonl`

One JSON object per check. A real row from the watch-only demo, one check after conv2 and conv3 were zeroed:

```json
{"step":450,"loss":2.305945873260498,
 "grad":{"conv1.0":0.0,"conv1.1":0.0,"conv2.0":0.0,"conv2.1":0.0,"conv3.0":0.0,"conv3.1":0.0,
         "classifier.0":0.0558704249560833,"classifier.3":0.08159016817808151},
 "stuck":{"conv1.0":[0,0,0,0],"conv2.0":[64,0,0,0],"conv3.0":[128,0,0,0],"classifier.0":[0,0,100,0],
          "classifier.3":[0,0,0,0]}}
```

(Wrapped here for reading; in the file each row is one line.)

`stuck` lists, per producer layer, `[certified, constant, silent, decayed]` channel counts.

## Seal and signature

`record.json` and `record.sig`, when present, are described in [The training record](../concepts/record.md#sealing-and-signing).

## Stability

The format is versioned by `record_version`. Readers should ignore fields they do not know. Before 1.0, event types may be added in any minor release; a removed field or changed meaning bumps `record_version`.
