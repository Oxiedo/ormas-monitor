"""MonitorConfig: every threshold the live monitor uses, in one frozen, validated dataclass.

The defaults are the values ormas-monitor 0.1 hard-coded, so `Monitor(model, opt)` behaves exactly
as it did; the published demo numbers were measured with them. The record stores the whole config
in `run_start`, so a reader can see which thresholds produced each alarm.

    from ormas_monitor import Monitor, MonitorConfig
    cfg = MonitorConfig(check_every=100, sigma=3.0)
    mon = Monitor(model, opt, config=cfg)
    mon = Monitor(model, opt, config=cfg, check_every=20)   # keyword shortcuts override the config
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .structure import DECAYED_REL_TOL


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds of the live monitor. All fields are validated on construction.

    Sampling and baselines
        check_every       check every N-th optimizer step; hooks do nothing on the others.
        warmup            checks a layer's baseline needs before it can raise an alarm.
        baseline_rate_floor  smallest update weight of the running mean/variance of
                          log10(gradient norm); 0.1 means the baseline remembers ~10 checks.

    Gradient alarms
        sigma             an alarm needs the log10 gradient norm this many standard deviations
                          off its own baseline ...
        alarm_min_decades ... and at least this many decades (1.0 = a factor of 10).
        alarm_confirmations  consecutive checks showing the same condition before an alarm is
                          raised. One alarm per episode; `alarm_cleared` ends it.

    Loss
        loss_spike_factor a loss spike also needs the loss above this x its baseline mean.

    Channels and repair
        patience          checks a channel must stay stuck before it counts as silent, or is
                          repaired when it is not proven dead.
        certified_repair_streak  checks a certified-dead channel must stay stuck before it is
                          repaired. A proven-dead channel cannot recover, so waiting longer
                          gains nothing.
        decayed_rel_tol   a filter or BatchNorm scale below this x the layer median (and not
                          exactly zero), with shift <= 0, is flagged as decayed.
        repair_seed       None: repaired filters are drawn from torch's global RNG, as in 0.1.
                          An int: drawn from a torch.Generator seeded from (repair_seed, step,
                          layer), so the repair is identical in every process. Under
                          torch.distributed, None is treated as 0.

    Record size
        list_limit        channel lists in `channels` events are truncated to this length.
        channel_list_limit  channel lists in `certificate` and `repair` events.
        layer_list_limit  layer lists in `nonfinite_gradient` and `skipped_step` events.
    """

    check_every: int = 50
    patience: int = 4
    sigma: float = 2.5
    warmup: int = 8
    baseline_rate_floor: float = 0.1
    alarm_min_decades: float = 1.0
    alarm_confirmations: int = 2
    loss_spike_factor: float = 1.5
    certified_repair_streak: int = 2
    decayed_rel_tol: float = DECAYED_REL_TOL
    repair_seed: int | None = None
    list_limit: int = 64
    channel_list_limit: int = 256
    layer_list_limit: int = 32

    def __post_init__(self) -> None:
        positive_ints = ("check_every", "patience", "alarm_confirmations", "certified_repair_streak",
                         "list_limit", "channel_list_limit", "layer_list_limit")
        for name in positive_ints:
            v = getattr(self, name)
            if not _is_int(v) or v < 1:
                raise ValueError(f"MonitorConfig.{name} must be an int >= 1, got {v!r}")
        if not _is_int(self.warmup) or self.warmup < 0:
            raise ValueError(f"MonitorConfig.warmup must be an int >= 0, got {self.warmup!r}")
        if not _is_number(self.sigma) or not self.sigma > 0:
            raise ValueError(f"MonitorConfig.sigma must be a number > 0, got {self.sigma!r}")
        if not _is_number(self.alarm_min_decades) or self.alarm_min_decades < 0:
            raise ValueError(f"MonitorConfig.alarm_min_decades must be >= 0, got {self.alarm_min_decades!r}")
        if not _is_number(self.loss_spike_factor) or self.loss_spike_factor < 1:
            raise ValueError(f"MonitorConfig.loss_spike_factor must be >= 1, got {self.loss_spike_factor!r}")
        if not _is_number(self.baseline_rate_floor) or not 0 < self.baseline_rate_floor <= 1:
            raise ValueError("MonitorConfig.baseline_rate_floor must be in (0, 1], "
                             f"got {self.baseline_rate_floor!r}")
        if not _is_number(self.decayed_rel_tol) or not 0 < self.decayed_rel_tol < 1:
            raise ValueError(f"MonitorConfig.decayed_rel_tol must be in (0, 1), got {self.decayed_rel_tol!r}")
        if self.repair_seed is not None and (not _is_int(self.repair_seed) or self.repair_seed < 0):
            raise ValueError(f"MonitorConfig.repair_seed must be None or an int >= 0, got {self.repair_seed!r}")

    def replace(self, **changes: Any) -> MonitorConfig:
        """A copy with `changes` applied (validated again). Unknown names raise TypeError."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of every field, as stored in the record's `run_start.config`."""
        return dataclasses.asdict(self)
