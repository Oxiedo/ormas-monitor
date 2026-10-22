"""Facts about one monitored run, computed once from its record and shared by every report.

Inputs are the run directory's `diary.jsonl` (required) and `metrics.jsonl` (optional: older
records shipped without it). Nothing here guesses: a quantity that the record cannot support
is `None`, and the reports say "unknown" rather than print a number.

Definitions the reports rely on:

* **Stuck** (counted as cost): a layer has at least one channel that is *proven dead*,
  *constant* or *decayed to ~0*. These three states are read from the weights, not inferred.
  *Silent* channels (no positive output for several checks) are an estimate and are reported,
  but never counted as cost.
* **Steps stuck**: a check that finds a stuck layer covers the steps up to the next check
  (or the end of the run); a repair at a check ends the interval at that check. Counting
  starts at the first check that saw the problem, so the figure is a lower bound by up to
  one check interval.
* **Wall-clock**: timestamps of diary events, linearly interpolated between the nearest
  events on either side. Exact when an interval starts and ends on recorded events (the usual
  case: a `channels` event starts it; a `repair`, `channels_cleared` or `run_end` ends it).
* **Coverage**: a check "saw gradients" when at least one watched layer had a `.grad`. A run
  whose checks never saw one (typically `mon.step()` called before `loss.backward()`) was not
  monitored for learning, whatever else its record says; a run with no check at all is unknown.
* **Lead time**: for each loss spike, the earliest `alarm`, `channels` or `certificate` event
  in the window before it. An earlier signal is not evidence that it caused the spike.
"""
from __future__ import annotations

import bisect
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..diary import file_sha256, read
from ..diary import verify as chain_verify

COMMON = ("seq", "time", "event", "prev", "hash")
SIGNAL_EVENTS = ("alarm", "channels", "certificate")
HIDDEN_EVENTS = ("run_start", "run_end", "structure", "manifest")
DISPLAY_SEVERITY = {"sink_error": "warning", "skipped_step": "info", "loss_recovered": "info",
                    "alarm_cleared": "info", "channels_cleared": "info", "no_gradients": "warning",
                    "certificate_refused": "info"}


@dataclass(frozen=True)
class ReportConfig:
    """Knobs for report rendering. Defaults suit every record written so far."""

    lead_window_checks: int = 20       # a loss spike looks back at most this many checks for a signal
    max_chart_points: int = 1500       # loss chart is thinned above this many points
    default_check_every: int = 50      # used only when an old record does not state its interval
    max_event_rows: int = 400          # engineer report: event table rows before truncation

    def __post_init__(self) -> None:
        assert self.lead_window_checks >= 1 and self.max_chart_points >= 2
        assert self.default_check_every >= 1 and self.max_event_rows >= 1


@dataclass(frozen=True)
class CostInputs:
    """Optional price inputs for the dollar ESTIMATE. Absent (None) means: print no dollars."""

    gpu_hour_cost: Optional[float] = None
    gpus: Optional[int] = None

    def __post_init__(self) -> None:
        if self.gpu_hour_cost is not None and not self.gpu_hour_cost > 0:
            raise ValueError("gpu_hour_cost must be a positive price per GPU-hour")
        if self.gpus is not None and (not isinstance(self.gpus, int) or self.gpus < 1):
            raise ValueError("gpus must be a whole number of at least 1")

    @property
    def priced(self) -> bool:
        return self.gpu_hour_cost is not None

    @property
    def gpu_count(self) -> int:
        """GPUs used for the estimate: as given, else 1 (and the reports say it was assumed)."""
        return self.gpus if self.gpus is not None else 1


@dataclass(frozen=True)
class Stuck:
    """Channel counts of one layer at one check."""

    certified: int = 0
    constant: int = 0
    silent: int = 0
    decayed: int = 0

    @classmethod
    def from_metrics(cls, row: Any) -> "Stuck":
        """`[certified, constant, silent, decayed]`; v0.1 records wrote 3 elements, no `decayed`."""
        vals = [int(v) for v in (row or [])][:4] + [0] * 4
        return cls(*vals[:4])

    @classmethod
    def from_event(cls, e: dict) -> "Stuck":
        """From a `channels` event; `n_*` fields when present, else lengths of the (truncated) lists."""
        def n(key: str) -> int:
            v = e.get("n_" + key)
            return int(v) if v is not None else len(e.get(key) or [])
        return cls(n("certified"), n("flat"), n("observed"), n("decayed"))

    @property
    def exact(self) -> int:
        """Channels stuck by a weight-level fact: proven dead, constant, or decayed to ~0."""
        return self.certified + self.constant + self.decayed

    @property
    def total(self) -> int:
        return self.exact + self.silent

    def minus(self, n: int) -> "Stuck":
        """Counts after `n` channels were repaired (repairs take proven, constant, decayed, silent)."""
        out = []
        for v in (self.certified, self.constant, self.decayed, self.silent):
            take = min(v, n)
            out.append(v - take)
            n -= take
        return Stuck(certified=out[0], constant=out[1], decayed=out[2], silent=out[3])

    def peak(self, other: "Stuck") -> "Stuck":
        return Stuck(*(max(a, b) for a, b in zip(
            (self.certified, self.constant, self.silent, self.decayed),
            (other.certified, other.constant, other.silent, other.decayed))))


@dataclass(frozen=True)
class Episode:
    """One continuous interval [start, end) in which one layer had stuck channels."""

    layer: str
    start: int
    end: int
    open_at_end: bool           # still stuck when the record ends
    peak: Stuck
    proven: bool                # a channel was proven dead at some check in the interval
    repaired: int               # channels repaired during or at the end of the interval
    seconds: Optional[float]

    @property
    def steps(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Cost:
    """Training spent while part of the model was stuck. See the module docstring for definitions."""

    total_steps: int
    total_seconds: Optional[float]
    stuck_steps: int
    stuck_seconds: Optional[float]
    proven_steps: int
    proven_seconds: Optional[float]
    spans: tuple[tuple[int, int], ...]          # union of stuck intervals, [start, end)
    resolution: int                             # steps between checks
    inputs: CostInputs = field(default_factory=CostInputs)

    def dollars(self, seconds: Optional[float]) -> Optional[float]:
        """ESTIMATE: seconds / 3600 x GPUs x price per GPU-hour; None without a price or a time."""
        if not self.inputs.priced or seconds is None:
            return None
        return seconds / 3600.0 * self.inputs.gpu_count * float(self.inputs.gpu_hour_cost)


@dataclass(frozen=True)
class Lead:
    """For one loss spike: the earliest earlier signal in its look-back window, if any."""

    spike: dict
    window_from: int            # first step of the look-back window
    signal: Optional[dict]
    n_signals: int
    steps_earlier: Optional[int]
    seconds_earlier: Optional[float]


@dataclass(frozen=True)
class Coverage:
    """What the monitor could actually see. `None` means the record cannot say (older records)."""

    checks: Optional[int]                   # sampled checks, from run_end or the metrics rows
    without_gradients: Optional[int]        # checks at which no watched layer had a gradient
    gradients_seen: Optional[bool]          # at least one check saw a gradient
    first_without: Optional[int]            # step of the first check without gradients
    monitored_layers: Optional[int]         # layers the manifest says were watched

    @property
    def state(self) -> str:
        """'unknown' (no check ran), 'blind' (no gradient ever seen), 'partial' (some checks saw
        none), 'full' (every check saw gradients) or 'unstated' (the record cannot tell)."""
        if self.checks == 0:
            return "unknown"
        if self.gradients_seen is False:
            return "blind"
        if self.without_gradients:
            return "partial"
        return "full" if self.gradients_seen else "unstated"

    @property
    def advice(self) -> str:
        """The fix, in a few words, for a headline."""
        if self.monitored_layers == 0:
            return "no layer was watched (the layer filter matched nothing)"
        return "call mon.step() after loss.backward()"

    @property
    def cause(self) -> str:
        """The likely reason no gradients were seen, and the fix, as one sentence."""
        if self.monitored_layers == 0:
            return "No layer was watched: the layer filter matched nothing."
        return ("The usual cause is calling mon.step() before loss.backward() (or after "
                "optimizer.zero_grad()); call it after loss.backward() and before optimizer.step().")


@dataclass(frozen=True)
class Trust:
    """Can the record be trusted? Statuses follow `record.verify`: pass, fail, absent, unchecked,
    skipped; `None` means this install could not run that check at all."""

    intact: bool                        # parse, genesis, chain and sequence checks all passed
    events: int
    message: str                        # the chain's own one-line result
    finished: Optional[bool]            # run_start first and run_end last
    metrics: Optional[str]              # status of the metrics-hash check
    metrics_message: str
    seal: Optional[str]
    signature: Optional[str]
    signer: Optional[str]               # key fingerprint named in the seal, if signed
    source: str
    checks: tuple[tuple[str, str, str], ...] = ()     # (label, status, message), in verify order

    @property
    def metrics_ok(self) -> Optional[bool]:
        return {"pass": True, "fail": False}.get(self.metrics or "")

    @property
    def signed(self) -> Optional[bool]:
        if self.signature is None:
            return None
        return self.signature != "absent"


@dataclass
class RunFacts:
    """Everything the run reports need, derived once."""

    run_dir: Path
    events: list[dict]
    metrics: list[dict]
    has_metrics: bool
    start: dict
    end: Optional[dict]
    manifests: list[dict]
    check_every: int
    repair_mode: Any
    total_steps: int
    shown: list[dict]
    episodes: list[Episode]
    silent_peaks: dict[str, int]
    layer_peaks: dict[str, Stuck]
    cost: Cost
    leads: list[Lead]
    trust: Trust
    time_at: Callable[[int], Optional[float]]
    config: ReportConfig
    coverage: Coverage
    silent_at_end: dict[str, int]           # silent channels per layer in the last state recorded
    last_check: Optional[int]               # step of the last check
    kinds: dict[str, str]                   # layer -> module type (Conv2d, Linear, ...), where recorded

    def kind(self, layer: Any) -> Optional[str]:
        return self.kinds.get(str(layer))

    def later_checks(self, step: int) -> Optional[int]:
        """Checks recorded strictly after `step`; None for a record without metrics."""
        if not self.has_metrics:
            return None
        return sum(1 for r in self.metrics if isinstance(r.get("step"), (int, float)) and r["step"] > step)

    @property
    def finished(self) -> bool:
        return self.end is not None

    def of(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e.get("event") in kinds]

    @property
    def watch_only(self) -> bool:
        return not self.repair_mode

    @property
    def checks(self) -> Optional[int]:
        if self.end is not None and self.end.get("checks") is not None:
            return int(self.end["checks"])
        return len(self.metrics) if self.has_metrics else None


# --------------------------------------------------------------------------- loading
def load_metrics(path: Path) -> list[dict]:
    """Rows of metrics.jsonl, or [] when the file is absent (older records, lone diaries)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def analyse(run_dir: str | Path, config: ReportConfig | None = None,
            inputs: CostInputs | None = None, public_key: str | Path | None = None) -> RunFacts:
    """Read a run directory and compute every fact the reports print. `public_key` (PEM) lets the
    record check verify a signature; without it a signature is reported as present but unchecked."""
    config = config or ReportConfig()
    inputs = inputs or CostInputs()
    run_dir = Path(run_dir)
    events = read(run_dir / "diary.jsonl")
    metrics = load_metrics(run_dir / "metrics.jsonl")
    start = next((e for e in events if e.get("event") == "run_start"), {})
    end = next((e for e in reversed(events) if e.get("event") == "run_end"), None)
    manifests = [e for e in events if e.get("event") == "manifest"]
    cfg = start.get("config") or {}
    check_every = int(cfg.get("check_every") or config.default_check_every)
    repair_mode = cfg.get("repair", False)
    for m in manifests:     # a manifest, when present, is the authoritative statement of the mode
        for key in ("repair", "repair_mode"):
            if key in m:
                v = m[key]
                repair_mode = v.get("mode", repair_mode) if isinstance(v, dict) else v

    stepped = [int(e["step"]) for e in events if isinstance(e.get("step"), (int, float))]
    m_steps = [int(r["step"]) for r in metrics if isinstance(r.get("step"), (int, float))]
    if end is not None and end.get("steps") is not None:
        total = int(end["steps"])
    else:
        last = max(stepped + m_steps, default=-1)
        total = last + 1 if last >= 0 else 0

    time_at = _clock(events, total, end)
    timeline = _timeline(metrics, events)
    episodes = _episodes(timeline, total, events, time_at)
    spans = _union([(e.start, e.end) for e in episodes])
    proven = _union([(e.start, e.end) for e in episodes if e.proven])
    total_seconds = None
    if end is not None and start.get("time") is not None:
        total_seconds = float(end["time"]) - float(start["time"])
    cost = Cost(total_steps=total, total_seconds=total_seconds,
                stuck_steps=sum(b - a for a, b in spans), stuck_seconds=_seconds(spans, time_at),
                proven_steps=sum(b - a for a, b in proven), proven_seconds=_seconds(proven, time_at),
                spans=tuple(spans), resolution=check_every, inputs=inputs)

    silent: dict[str, int] = {}
    peaks: dict[str, Stuck] = {}
    for _, state in timeline:
        for layer, st in state.items():
            peaks[layer] = peaks.get(layer, Stuck()).peak(st)
            if st.silent:
                silent[layer] = max(silent.get(layer, 0), st.silent)

    shown = [e for e in events if e.get("event") not in HIDDEN_EVENTS]
    final = timeline[-1][1] if timeline else {}
    silent_end = {k: st.silent for k, st in final.items() if st.silent}
    checks_at = sorted(int(r["step"]) for r in metrics if isinstance(r.get("step"), (int, float)))
    return RunFacts(run_dir=run_dir, events=events, metrics=metrics, has_metrics=bool(metrics),
                    start=start, end=end, manifests=manifests, check_every=check_every,
                    repair_mode=repair_mode, total_steps=total, shown=shown, episodes=episodes,
                    silent_peaks=silent, layer_peaks=peaks, cost=cost,
                    leads=_leads(events, check_every, config), trust=trust(run_dir, end, public_key),
                    time_at=time_at, config=config, coverage=_coverage(events, metrics, end, manifests),
                    silent_at_end=silent_end, last_check=checks_at[-1] if checks_at else None,
                    kinds=_kinds(events))


def _kinds(events: list[dict]) -> dict[str, str]:
    """Module type per layer, from the manifest's producers and any event naming one."""
    out: dict[str, str] = {}
    for e in events:
        rows = e.get("producers") if e.get("event") == "manifest" else [e]
        for r in rows or []:
            if isinstance(r, dict) and r.get("layer") and isinstance(r.get("kind"), str) \
                    and r["kind"][:1].isupper():          # a class name, not an alarm/diagnosis word
                out.setdefault(str(r["layer"]), r["kind"])
    return out


def _coverage(events: list[dict], metrics: list[dict], end: Optional[dict],
              manifests: list[dict]) -> Coverage:
    """Did the checks see gradients? From the metrics rows; `run_end` counts and `no_gradients`
    events fill in for records without metrics."""
    counts = (end or {}).get("counts") or {}
    rows = [r for r in metrics if isinstance(r.get("step"), (int, float))]
    checks: Optional[int] = None
    if end is not None and end.get("checks") is not None:
        checks = int(end["checks"])
    elif rows:
        checks = len(rows)
    blind_rows = [int(r["step"]) for r in rows if not r.get("grad")]
    flagged = [e for e in events if e.get("event") == "no_gradients"]
    without: Optional[int] = None
    if counts.get("checks_without_gradients") is not None:
        without = int(counts["checks_without_gradients"])
    elif rows:
        without = len(blind_rows)
    elif flagged:
        without = len(flagged)
    seen: Optional[bool] = None
    if rows:
        seen = len(blind_rows) < len(rows)
    elif without is not None and checks:
        seen = without < checks
    first = [int(e["step"]) for e in flagged if isinstance(e.get("step"), (int, float))] + blind_rows[:1]
    monitored = None
    for m in manifests:
        layers = m.get("layers")
        if isinstance(layers, dict) and isinstance(layers.get("monitored"), int):
            monitored = layers["monitored"]
    return Coverage(checks=checks, without_gradients=without, gradients_seen=seen,
                    first_without=min(first) if first else None, monitored_layers=monitored)


# --------------------------------------------------------------------------- time
def _clock(events: list[dict], total: int, end: Optional[dict]) -> Callable[[int], Optional[float]]:
    """step -> unix time, interpolated between diary events (earliest event at each step)."""
    first: dict[int, float] = {}
    for e in events:
        s, t = e.get("step"), e.get("time")
        if isinstance(s, (int, float)) and isinstance(t, (int, float)):
            first[int(s)] = min(first.get(int(s), float(t)), float(t))
    if end is not None and isinstance(end.get("time"), (int, float)):
        first[total] = max(first.get(total, float(end["time"])), float(end["time"]))
    anchors = sorted(first.items())
    steps = [s for s, _ in anchors]

    def at(step: int) -> Optional[float]:
        if len(anchors) < 2 or step < steps[0] or step > steps[-1]:
            return dict(anchors).get(step) if anchors else None
        i = bisect.bisect_left(steps, step)
        if steps[i] == step:
            return anchors[i][1]
        (s0, t0), (s1, t1) = anchors[i - 1], anchors[i]
        return t0 + (t1 - t0) * (step - s0) / (s1 - s0)
    return at


def _seconds(spans: list[tuple[int, int]], at: Callable[[int], Optional[float]]) -> Optional[float]:
    total = 0.0
    for a, b in spans:
        ta, tb = at(a), at(b)
        if ta is None or tb is None:
            return None
        total += max(0.0, tb - ta)
    return total


# --------------------------------------------------------------------------- stuck state
def _timeline(metrics: list[dict], events: list[dict]) -> list[tuple[int, dict[str, Stuck]]]:
    """Per-layer channel state, as a list of (step, state from this step on).

    With metrics, every check row is a full snapshot. Without (older records), state changes
    only at `channels` / `channels_cleared` events. Either way a `repair` at a step reduces the
    repaired layer's counts from that step on.
    """
    by_step: dict[int, dict[str, list]] = {}

    def at(s: int) -> dict[str, list]:
        return by_step.setdefault(s, {"rows": [], "chan": [], "repair": []})
    for r in metrics:
        if isinstance(r.get("step"), (int, float)):
            at(int(r["step"]))["rows"].append(r)
    for e in events:
        if not isinstance(e.get("step"), (int, float)) or not e.get("layer"):
            continue
        if e.get("event") in ("channels", "channels_cleared") and not metrics:
            at(int(e["step"]))["chan"].append(e)
        elif e.get("event") == "repair":
            at(int(e["step"]))["repair"].append(e)

    state: dict[str, Stuck] = {}
    out: list[tuple[int, dict[str, Stuck]]] = []
    for s in sorted(by_step):
        g = by_step[s]
        for r in g["rows"]:
            state = {k: Stuck.from_metrics(v) for k, v in (r.get("stuck") or {}).items()}
        for e in g["chan"]:
            state[e["layer"]] = Stuck() if e["event"] == "channels_cleared" else Stuck.from_event(e)
        if g["repair"]:
            snap = dict(state)
            out.append((s, snap))       # what the check saw, before the repair took effect
            for e in g["repair"]:
                state[e["layer"]] = state.get(e["layer"], Stuck()).minus(int(e.get("count") or 0))
            out.append((s, dict(state)))
        else:
            out.append((s, dict(state)))
    return out


def _episodes(timeline: list[tuple[int, dict[str, Stuck]]], total: int, events: list[dict],
              at: Callable[[int], Optional[float]]) -> list[Episode]:
    """Merge consecutive stuck checks of each layer into intervals."""
    layers: list[str] = []
    for _, st in timeline:
        for k in st:
            if k not in layers:
                layers.append(k)
    repairs: dict[str, list[tuple[int, int]]] = {}
    for e in events:
        if e.get("event") == "repair" and e.get("layer") and isinstance(e.get("step"), (int, float)):
            repairs.setdefault(e["layer"], []).append((int(e["step"]), int(e.get("count") or 0)))

    out: list[Episode] = []
    for layer in layers:
        cur: Optional[dict] = None
        for s, st in timeline:
            x = st.get(layer, Stuck())
            if x.exact > 0:
                if cur is None:
                    cur = {"start": s, "peak": Stuck(), "proven": False}
                cur["peak"] = cur["peak"].peak(x)
                cur["proven"] = cur["proven"] or x.certified > 0
            elif cur is not None:
                out.append(_close(layer, cur, s, False, repairs, at))
                cur = None
        if cur is not None:
            out.append(_close(layer, cur, max(total, cur["start"]), True, repairs, at))
    out.sort(key=lambda e: (e.start, e.layer))
    return out


def _close(layer: str, cur: dict, end: int, open_: bool, repairs: dict[str, list[tuple[int, int]]],
           at: Callable[[int], Optional[float]]) -> Episode:
    fixed = sum(n for s, n in repairs.get(layer, []) if cur["start"] <= s <= end)
    span = [(cur["start"], end)]
    return Episode(layer=layer, start=cur["start"], end=end, open_at_end=open_, peak=cur["peak"],
                   proven=cur["proven"], repaired=fixed, seconds=_seconds(span, at))


def _union(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# --------------------------------------------------------------------------- lead time
def _leads(events: list[dict], check_every: int, config: ReportConfig) -> list[Lead]:
    window = config.lead_window_checks * check_every
    out: list[Lead] = []
    recovered_at = 0
    for e in events:
        ev = e.get("event")
        if ev == "loss_recovered" and isinstance(e.get("step"), (int, float)):
            recovered_at = int(e["step"])
        if ev != "loss_spike" or not isinstance(e.get("step"), (int, float)):
            continue
        s = int(e["step"])
        lo = max(recovered_at, s - window, 0)
        signals = [x for x in events if x.get("event") in SIGNAL_EVENTS
                   and isinstance(x.get("step"), (int, float)) and lo <= int(x["step"]) <= s
                   and int(x.get("seq", 0)) != int(e.get("seq", -1))]
        signals.sort(key=lambda x: (int(x["step"]), int(x.get("seq", 0))))
        first = signals[0] if signals else None
        secs = None
        if first is not None and isinstance(first.get("time"), (int, float)) and isinstance(e.get("time"), (int, float)):
            secs = float(e["time"]) - float(first["time"])
        out.append(Lead(spike=e, window_from=lo, signal=first, n_signals=len(signals),
                        steps_earlier=(s - int(first["step"])) if first else None,
                        seconds_earlier=secs))
    return out


# --------------------------------------------------------------------------- trust
def _record_verifier() -> Optional[Callable[..., Any]]:
    """`ormas_monitor.record.verify.verify_run`, or None in an install without the record package."""
    name = "ormas_monitor.record.verify"
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name in ("ormas_monitor.record", name):
            return None
        raise
    return mod.verify_run


def _signer(run_dir: Path) -> Optional[str]:
    """The key fingerprint a seal names as its signer (record.json), shortened for display."""
    path = run_dir / "record.json"
    if not path.exists():
        return None
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None         # the seal check reports the unreadable seal; nothing to name here
    who = seal.get("signer") if isinstance(seal, dict) else None
    fp = who.get("public_key_sha256") if isinstance(who, dict) else None
    return f"the key with fingerprint {str(fp)[:16]}…" if fp else None


def trust(run_dir: Path, end: Optional[dict], public_key: str | Path | None = None) -> Trust:
    """Verify the record with `record.verify.verify_run` when installed, else the bare hash chain
    plus the metrics hash sealed in `run_end`."""
    fn = _record_verifier()
    if fn is None:
        intact, n, message = chain_verify(run_dir / "diary.jsonl")
        mpath, sealed = run_dir / "metrics.jsonl", (end or {}).get("metrics_sha256")
        if not sealed:
            metrics, mmsg = "skipped", "no metrics hash recorded in run_end"
        elif not mpath.exists():
            metrics, mmsg = "fail", "metrics.jsonl is missing"
        elif file_sha256(mpath) == sealed:
            metrics, mmsg = "pass", "matches the hash recorded in run_end"
        else:
            metrics, mmsg = "fail", "metrics.jsonl changed after the run ended"
        return Trust(intact=bool(intact), events=int(n), message=str(message), finished=end is not None,
                     metrics=metrics, metrics_message=mmsg, seal=None, signature=None, signer=None,
                     source="ormas_monitor.diary.verify")
    res = fn(run_dir, public_key=public_key)
    by = {c.name: c for c in res.checks}
    core = [by[k] for k in ("parse", "genesis", "chain", "seq") if k in by]
    intact = all(c.status != "fail" for c in core)
    broken = next((c for c in core if c.status == "fail"), None)
    chain = by.get("chain")
    message = broken.message if broken else (chain.message if chain else f"{res.events} events")
    sig = by.get("signature")
    signature = sig.status if sig else None
    return Trust(intact=intact, events=int(res.events), message=message,
                 finished=(by["bookends"].status == "pass") if "bookends" in by else None,
                 metrics=by["metrics"].status if "metrics" in by else None,
                 metrics_message=by["metrics"].message if "metrics" in by else "",
                 seal=by["seal"].status if "seal" in by else None, signature=signature,
                 signer=_signer(run_dir) if signature in ("pass", "fail", "unchecked") else None,
                 source="ormas_monitor.record.verify",
                 checks=tuple((c.label, c.status, c.message) for c in res.checks))
