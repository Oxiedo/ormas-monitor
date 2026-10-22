"""The diary line schema: every event type the monitor writes, and the fields each must carry.

A diary line is a JSON object with the common fields `seq, time, event, prev, hash` (see
`diary.py`) plus the event's own fields below. Lines may carry more fields than listed; they may
not carry fewer. `validate_event(line)` returns the problems it finds; an empty list means valid.

RECORD_VERSION is written into `run_start.record_version`. It changes when a field is removed or
its meaning changes; adding an event type or an optional field does not change it.
"""
from __future__ import annotations

import re
from typing import Any

RECORD_VERSION = 1

COMMON_FIELDS: tuple[str, ...] = ("seq", "time", "event", "prev", "hash")

SEVERITIES = ("info", "warning", "critical", "proof", "action")
DIAGNOSES = ("no_gradient", "vanishing", "exploding")
LEVELS = ("none", "some", "most", "all")

_CHANNEL_FIELDS = ("step", "severity", "layer", "batchnorm", "channels", "level", "n_certified",
                   "n_flat", "n_observed", "n_decayed", "certified", "flat", "observed")

EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "run_start": ("name", "tool", "model", "parameters", "layers", "architecture_sha256", "optimizer",
                  "python", "torch", "platform", "config", "record_version"),
    "structure": ("step", "pairs", "relu_after"),
    "manifest": ("step", "check_every", "layers", "producers", "channel_checks", "repair",
                 "precision", "distributed", "notes"),
    "alarm": ("step", "severity", "layer", "kind", "diagnosis", "grad_norm", "baseline", "sigma"),
    "alarm_cleared": ("step", "layer", "diagnosis", "grad_norm", "baseline"),
    "channels": _CHANNEL_FIELDS,
    "channels_cleared": _CHANNEL_FIELDS,
    "certificate": ("step", "severity", "layer", "batchnorm", "channels", "count", "statement", "assumes"),
    # Channels whose weights alone would be certified, but a condition of the certificate failed.
    # Once per (layer, reason); `count` is how many candidate channels that reason refused.
    "certificate_refused": ("step", "severity", "layer", "batchnorm", "count", "reason"),
    # A sampled check found no .grad on any watched layer (step() before backward?). Once per run;
    # run_end.counts.checks_without_gradients counts every such check.
    "no_gradients": ("step", "severity", "message"),
    # step() got a loss with more than one element; the monitor records its mean. Once per run.
    "loss_reduced": ("step", "severity", "elements"),
    "repair": ("step", "severity", "layer", "batchnorm", "channels", "count", "method",
               "optimizer_slices_reset", "modification_class", "bound", "delta_norm",
               "weight_norm_before", "weight_norm_after", "relative_change", "rng"),
    "loss_spike": ("step", "severity", "value", "baseline"),
    "loss_recovered": ("step", "value", "baseline"),
    "nonfinite_loss": ("step", "severity", "value"),
    "nonfinite_gradient": ("step", "severity", "layers"),
    "skipped_step": ("step", "severity", "scale", "layers"),
    "sink_error": ("sink", "error"),
    "run_end": ("steps", "checks", "counts", "metrics_sha256"),
}

# Fields whose value is fixed for an event type.
FIXED_VALUES: dict[str, dict[str, Any]] = {
    "certificate": {"severity": "proof"},
    "certificate_refused": {"severity": "info"},
    "no_gradients": {"severity": "warning"},
    "loss_reduced": {"severity": "info"},
    "repair": {"severity": "action", "modification_class": "discrete", "bound": None},
    "nonfinite_loss": {"severity": "critical"},
    "nonfinite_gradient": {"severity": "critical"},
    "skipped_step": {"severity": "info"},
    "loss_spike": {"severity": "warning"},
    "alarm": {"severity": "warning"},
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_event(line: dict) -> list[str]:
    """Problems with one diary line, as human-readable strings. Empty list: the line is valid.

    Checks the common fields and their types, that the event type is known, that every required
    field is present, fixed values (e.g. a certificate's severity is "proof"), and a few
    enumerations (severity, diagnosis, level). Hash-chain integrity is `diary.verify`'s job.
    """
    if not isinstance(line, dict):
        return [f"line is a {type(line).__name__}, not an object"]
    problems = [f"missing common field {k!r}" for k in COMMON_FIELDS if k not in line]
    if "seq" in line and (not _is_int(line["seq"]) or line["seq"] < 0):
        problems.append(f"seq must be an int >= 0, got {line['seq']!r}")
    if "time" in line and (not isinstance(line["time"], (int, float)) or isinstance(line["time"], bool)):
        problems.append(f"time must be a number, got {line['time']!r}")
    for k in ("prev", "hash"):
        if k in line and not (isinstance(line[k], str) and _HEX64.match(line[k])):
            problems.append(f"{k} must be 64 lowercase hex characters")
    event = line.get("event")
    if not isinstance(event, str):
        return problems + [f"event must be a string, got {event!r}"]
    if event not in EVENT_FIELDS:
        return problems + [f"unknown event type {event!r}"]
    problems += [f"{event}: missing field {k!r}" for k in EVENT_FIELDS[event] if k not in line]
    for k, want in FIXED_VALUES.get(event, {}).items():
        if k in line and line[k] != want:
            problems.append(f"{event}: {k} must be {want!r}, got {line[k]!r}")
    if "severity" in line and line["severity"] not in SEVERITIES:
        problems.append(f"{event}: unknown severity {line['severity']!r}")
    if event in ("alarm", "alarm_cleared") and line.get("diagnosis") not in DIAGNOSES:
        problems.append(f"{event}: unknown diagnosis {line.get('diagnosis')!r}")
    if event in ("channels", "channels_cleared") and line.get("level") not in LEVELS:
        problems.append(f"{event}: unknown level {line.get('level')!r}")
    if "step" in EVENT_FIELDS[event] and "step" in line and (not _is_int(line["step"]) or line["step"] < 0):
        problems.append(f"{event}: step must be an int >= 0, got {line['step']!r}")
    if event == "run_start" and "record_version" in line and line["record_version"] != RECORD_VERSION:
        problems.append(f"run_start: record_version {line['record_version']!r} is not {RECORD_VERSION}")
    return problems
