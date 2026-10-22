# MonitorConfig

Every threshold the live monitor uses lives in one frozen, validated dataclass. The defaults are the values ormas-monitor 0.1 used, so `Monitor(model, optimizer)` behaves as it did, and the published demo numbers were measured with them.

```python
from ormas_monitor import Monitor, MonitorConfig

cfg = MonitorConfig(check_every=100, sigma=3.0)
mon = Monitor(model, optimizer, config=cfg)
mon = Monitor(model, optimizer, config=cfg, check_every=20)   # keyword shortcuts override the config
```

The whole configuration is written to the record in `run_start.config`, so a reader can see which thresholds produced each alarm.

<!-- VERIFY: principal — `MonitorConfig` exported from ormas_monitor/__init__.py (today: `from ormas_monitor.config import MonitorConfig`) -->

## Fields

### Sampling and baselines

| Field | Default | Meaning |
|---|---|---|
| `check_every` | `50` | Check every N-th optimizer step. Hooks do nothing on the others. |
| `warmup` | `8` | Checks a layer's baseline needs before it can raise an alarm. |
| `baseline_rate_floor` | `0.1` | Smallest update weight of the running mean and variance of log₁₀(gradient norm). `0.1` means the baseline remembers roughly the last 10 checks. |

### Gradient alarms

| Field | Default | Meaning |
|---|---|---|
| `sigma` | `2.5` | An alarm needs the log₁₀ gradient norm this many standard deviations off its own baseline ... |
| `alarm_min_decades` | `1.0` | ... and at least this many decades away (`1.0` = a factor of 10). |
| `alarm_confirmations` | `2` | Consecutive checks showing the same condition before an alarm is raised. One alarm per episode; `alarm_cleared` ends it. |

### Loss

| Field | Default | Meaning |
|---|---|---|
| `loss_spike_factor` | `1.5` | A loss spike also needs the loss above this multiple of its baseline mean. |

### Channels and repair

| Field | Default | Meaning |
|---|---|---|
| `patience` | `4` | Checks a channel must stay stuck before it counts as silent, or is repaired when it is not proven dead. |
| `certified_repair_streak` | `2` | Checks a certified-dead channel must stay stuck before it is repaired. A proven-dead channel cannot recover, so waiting longer gains nothing. |
| `decayed_rel_tol` | `1e-5` | A filter or BatchNorm scale below this multiple of the layer median (and not exactly zero), with shift ≤ 0, is flagged as decayed. |
| `repair_seed` | `None` | `None`: repaired filters come from torch's global RNG, as in 0.1. An int: they come from a generator seeded from (`repair_seed`, step, layer), so the repair is identical in every process. Under `torch.distributed`, `None` is treated as `0`. |

### Record size

| Field | Default | Meaning |
|---|---|---|
| `list_limit` | `64` | Channel lists in `channels` events are truncated to this length. |
| `channel_list_limit` | `256` | Channel lists in `certificate` and `repair` events. |
| `layer_list_limit` | `32` | Layer lists in `nonfinite_gradient` and `skipped_step` events. |

## Validation

Every field is validated on construction and a bad value raises `ValueError` naming the field. For example, `check_every`, `patience` and `alarm_confirmations` must be integers ≥ 1, `sigma` must be > 0, and `loss_spike_factor` must be ≥ 1.

## Methods

| Method | Returns |
|---|---|
| `cfg.replace(**changes)` | A validated copy with `changes` applied. Unknown names raise `TypeError`. |
| `cfg.to_dict()` | A plain dict of every field, as stored in `run_start.config`. |

## Tuning

- **Too many alarms on noisy training:** raise `sigma` or `warmup`, or `alarm_confirmations`.
- **Overhead too high:** raise `check_every`, or restrict the monitor with `layers=` (see [Large models](../guides/large-models.md)).
- **Repairs too eager:** raise `patience` and `certified_repair_streak`.

None of these defaults has been tuned on a benchmark. **Status: DESIGNED.**
