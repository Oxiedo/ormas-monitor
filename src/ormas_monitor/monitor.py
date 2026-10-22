"""Live training monitor: attach to any PyTorch model, call `step()` after `backward()`.

    mon = Monitor(model, optimizer, repair=True)
    for x, y in loader:
        loss = loss_fn(model(x), y)
        loss.backward()
        mon.step(loss)          # after backward, before optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
    mon.close()                 # writes ormas_run/report.html

Every `check_every` steps it reads per-layer gradient norms and per-channel activity, raises
alarms against each layer's own baseline, certifies dead channels, optionally repairs them,
and writes every event to the hash-chained Diary. Nothing leaves the machine.

Thresholds live in `MonitorConfig` (config.py); the event schema in schema.py; repair in repair.py.
At the first check a `manifest` event records which mechanisms are actually active for this
model and run (certificate per layer, repair mode, mixed precision, distributed mode).

Changes in 0.2:
  * thresholds moved to MonitorConfig; keyword shortcuts still override it;
  * `scaler=` (torch.amp.GradScaler): non-finite gradients are a skipped step, not an alarm;
    norms are taken in float32 and divided by the scale when `unscale_` was not called;
  * torch.distributed: only rank 0 writes files, prints or calls sinks; under DDP repair
    targets weight-detected channels only and uses a seeded generator; FSDP disables channel
    checks and refuses repair;
  * `layers=` fnmatch filter; `note_loss()`; torch.compile / DDP wrappers are unwrapped;
  * the certificate now also requires the optimizer (to check momentum) and BatchNorm in
    training mode, both of which the statement assumes;
  * sinks receive every diary line, run_start and run_end included; each failing sink
    produces exactly one `sink_error`.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from . import __version__
from .config import MonitorConfig
from .diary import Diary, file_sha256
from .repair import METHOD, MODIFICATION_CLASS, first_moment_mask, repair_channels, repair_generator, \
    unwrap_optimizer
from .schema import RECORD_VERSION
from .structure import BN_KINDS, CONV_KINDS, RELU_KINDS, pair_by_order, scan_channels

_LOG_EPS = 1e-30          # log10(0 + eps) = -30: a zero gradient norm, kept finite
_NO_GRAD_FLOOR = -29.0    # a baseline above this was not itself built from zero gradients
_HALF = (torch.float16, torch.bfloat16)

Sink = Callable[[str, dict], None]


def _out_channels(module: nn.Module) -> int | None:
    for attr in ("out_channels", "out_features", "num_features"):
        if hasattr(module, attr):
            return int(getattr(module, attr))
    return None


def _level(stuck: int, total: int) -> str:
    if stuck == 0:
        return "none"
    if stuck == total:
        return "all"
    return "most" if stuck * 2 >= total else "some"


def _norm32(g: torch.Tensor) -> torch.Tensor:
    # Telemetry stays full precision: cast before the reduction, never after.
    return torch.linalg.vector_norm(g, dtype=torch.float32) if g.dtype in _HALF else torch.linalg.vector_norm(g)


def _unwrap_model(model: nn.Module) -> tuple[nn.Module, list[str]]:
    """The user's module inside torch.compile / DDP / DataParallel wrappers, and the wrappers seen."""
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    wrappers = []
    while True:
        inner = getattr(model, "_orig_mod", None)
        if type(model).__name__ == "OptimizedModule" and isinstance(inner, nn.Module):
            wrappers.append("torch.compile")
            model = inner
        elif isinstance(model, (DistributedDataParallel, DataParallel)):
            wrappers.append(type(model).__name__)
            model = model.module
        else:
            return model, wrappers


def _uses_sharded_parameters(model: nn.Module) -> bool:
    """FSDP (1 or 2) or DTensor parameters: weights are sharded/flattened across ranks."""
    kinds: list[type] = []
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        kinds.append(FullyShardedDataParallel)
    except ImportError:
        pass
    try:
        from torch.distributed.fsdp import FSDPModule
        kinds.append(FSDPModule)
    except ImportError:
        pass
    if kinds and any(isinstance(m, tuple(kinds)) for m in model.modules()):
        return True
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None
    for p in model.parameters():
        if getattr(p, "_is_flat_param", False) or (DTensor is not None and isinstance(p, DTensor)):
            return True
    return False


def _distributed(model: nn.Module, wrappers: list[str]) -> tuple[str, int, int]:
    """(mode, rank, world_size). mode is none | ddp | distributed | fsdp."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return "none", 0, 1
    rank, world = dist.get_rank(), dist.get_world_size()
    if _uses_sharded_parameters(model):
        return "fsdp", rank, world
    return ("ddp" if "DistributedDataParallel" in wrappers else "distributed"), rank, world


class _Baseline:
    """Running mean and variance of log10(gradient norm) for one layer (or of the loss)."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self.alarm: str | None = None
        self.pending: str | None = None
        self.pending_count = 0

    def update(self, x: float, floor: float) -> None:
        a = max(floor, 1.0 / (self.n + 1))
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)
        self.n += 1


class Monitor:
    """Watch a model while it trains. See the module docstring for the loop.

    Args:
        model: the model being trained (a torch.compile or DDP wrapper is fine).
        optimizer: needed for repair and for the certificate's momentum check. Framework
            wrappers (Lightning, accelerate) are unwrapped.
        out: run directory for diary.jsonl, metrics.jsonl and the reports.
        repair: False (watch only), True (repair certified, flat and decayed channels) or
            "all" (also channels silent for `patience` checks; not under torch.distributed).
        check_every, patience, sigma, warmup: shortcuts that override `config`.
        name: run name in the record (default: the run directory's name).
        sinks: callables `sink(kind, payload)`, kind "event" (a diary line) or "metrics" (a row).
        verbose: print events as they happen.
        config: a MonitorConfig with every threshold (default: MonitorConfig()).
        scaler: the torch.amp.GradScaler used for mixed precision, if any.
        layers: fnmatch pattern(s) on module names; only matching layers are monitored.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *,
                 out: str | Path = "ormas_run", repair: bool | str = False,
                 check_every: int | None = None, patience: int | None = None,
                 sigma: float | None = None, warmup: int | None = None, name: str | None = None,
                 sinks: Sequence[Sink] = (), verbose: bool = True,
                 config: MonitorConfig | None = None, scaler: Any = None,
                 layers: str | Sequence[str] | None = None) -> None:
        if config is not None and not isinstance(config, MonitorConfig):
            raise TypeError(f"config must be a MonitorConfig, got {type(config).__name__}")
        overrides = {k: v for k, v in (("check_every", check_every), ("patience", patience),
                                       ("sigma", sigma), ("warmup", warmup)) if v is not None}
        self.config = (config or MonitorConfig()).replace(**overrides)
        if repair and optimizer is None:
            raise ValueError("repair needs the optimizer, so its state for repaired channels can be reset")
        if repair not in (False, True, "all"):
            raise ValueError("repair must be False, True (certified and flat channels) or 'all'")
        if scaler is not None:
            if not (callable(getattr(scaler, "is_enabled", None)) and callable(getattr(scaler, "get_scale", None))):
                raise TypeError(f"scaler must be a torch.amp.GradScaler, got {type(scaler).__name__}")
            if optimizer is None:
                raise ValueError("scaler needs the optimizer: its per-optimizer state says whether "
                                 "unscale_ was called")

        self.model, self.optimizer = model, optimizer
        self._opt = unwrap_optimizer(optimizer)
        self._net, self._wrappers = _unwrap_model(model)
        self.dist_mode, self.rank, self.world_size = _distributed(self._net, self._wrappers)
        if self.dist_mode == "fsdp" and repair:
            raise ValueError("repair is not supported under FSDP: parameters are sharded and flattened, "
                             "so per-channel filters cannot be read or rewritten. Use repair=False; "
                             "gradient alarms and the record still work")
        self.out = Path(out)
        self.repair_mode = repair
        self.scaler = scaler
        self.verbose = verbose
        self._writer = self.rank == 0
        # Sinks receive every diary line and metrics row: sink(kind, payload). A failing sink is
        # reported once as a `sink_error` event and then skipped (the one sanctioned swallow).
        self.sinks = list(sinks)
        self._sink_failed: set[int] = set()

        net = self._net
        self._all_modules: dict[str, nn.Module] = {
            n: m for n, m in net.named_modules()
            if any(p.requires_grad for p in m.parameters(recurse=False))}
        self._all_kinds = {n: type(m).__name__ for n, m in self._all_modules.items()}
        self._all_sizes = {n: _out_channels(m) for n, m in self._all_modules.items()}
        self.layer_patterns: list[str] | None = None
        if layers is not None:
            self.layer_patterns = [layers] if isinstance(layers, str) else list(layers)
            if not self.layer_patterns or not all(isinstance(p, str) for p in self.layer_patterns):
                raise ValueError("layers must be a pattern string or a non-empty list of them")
            self.modules = {n: m for n, m in self._all_modules.items()
                            if any(fnmatch.fnmatchcase(n, p) for p in self.layer_patterns)}
            if not self.modules:
                raise ValueError(f"layers={self.layer_patterns!r} matches no layer with trainable "
                                 f"parameters; names look like {list(self._all_modules)[:6]}")
        else:
            self.modules = dict(self._all_modules)
        self.kinds = {n: self._all_kinds[n] for n in self.modules}
        self.sizes = {n: self._all_sizes[n] for n in self.modules}
        self.producers = [n for n, k in self.kinds.items() if k in CONV_KINDS]
        self._channel_checks = self.dist_mode != "fsdp"

        self.step_count = 0
        self.checks = 0
        self._sampling = True
        self._noted_loss: torch.Tensor | float | None = None
        self._order: list[str] = []
        self._relu_after: set[str] = set()      # layers whose output goes straight into a ReLU
        self._last_out = None
        self._act_max: dict[str, torch.Tensor] = {}
        self._act_watch: set[str] | None = None  # None until the structure is known: watch all
        self.pairs: dict[str, str] | None = None
        self._base = {n: _Baseline() for n in self.modules}
        self._loss = _Baseline()
        self._loss_bad = False
        self._spiking = False
        self._grad_bad = False
        self._streak: dict[str, torch.Tensor] = {}
        self._silent: dict[str, torch.Tensor] = {}
        self._channel_level: dict[str, str] = {}
        self._certified_seen: set[str] = set()
        self.counts = {"alarms": 0, "stuck_layers": 0, "repairs": 0, "channels_repaired": 0,
                       "loss_spikes": 0, "nonfinite": 0, "skipped_steps": 0}
        self._closed = False

        self._handles = []
        for n, m in self._all_modules.items():
            if self._all_kinds[n] in CONV_KINDS or self._all_kinds[n] in BN_KINDS:
                self._handles.append(m.register_forward_hook(self._hook(n)))
        for m in net.modules():
            if isinstance(m, RELU_KINDS):
                self._handles.append(m.register_forward_pre_hook(self._relu_hook))

        self.diary: Diary | None = None
        self._metrics = None
        if self._writer:
            self.out.mkdir(parents=True, exist_ok=True)
            self.diary = Diary(self.out / "diary.jsonl")
            self._metrics = open(self.out / "metrics.jsonl", "w")
        shapes = "|".join(f"{n}:{tuple(p.shape)}" for n, p in net.named_parameters())
        opt = None
        if self._opt is not None:
            g = self._opt.param_groups[0]
            opt = {"type": type(self._opt).__name__,
                   **{k: g[k] for k in ("lr", "weight_decay", "momentum", "betas") if k in g}}
        self._write(
            "run_start", name=name or self.out.name, tool=f"ormas-monitor {__version__}",
            model=type(net).__name__,
            parameters=sum(p.numel() for p in net.parameters()),
            layers=len(self.modules), architecture_sha256=hashlib.sha256(shapes.encode()).hexdigest(),
            optimizer=opt, python=sys.version.split()[0], torch=torch.__version__,
            platform=platform.platform(), record_version=RECORD_VERSION,
            config={**self.config.to_dict(), "repair": repair, "layers": self.layer_patterns})

    # ------------------------------------------------------------------ settings, read-only
    @property
    def check_every(self) -> int:
        """Steps between checks (from the config)."""
        return self.config.check_every

    @property
    def patience(self) -> int:
        """Checks a channel must stay stuck (from the config)."""
        return self.config.patience

    @property
    def sigma(self) -> float:
        """Standard deviations off baseline for an alarm (from the config)."""
        return self.config.sigma

    @property
    def warmup(self) -> int:
        """Checks before a baseline can alarm (from the config)."""
        return self.config.warmup

    # ------------------------------------------------------------------ hooks
    def _hook(self, name: str):
        kind = self._all_kinds[name]

        def fn(module, inputs, output):
            if not self._sampling or not module.training or not isinstance(output, torch.Tensor):
                return
            if self.pairs is None:
                self._order.append(name)
                self._last_out = (name, output, output._version)
            elif name not in self._act_watch:
                return
            with torch.no_grad():
                if kind == "Linear":
                    dims = tuple(range(output.dim() - 1))
                else:
                    dims = (0,) + tuple(range(2, output.dim()))
                peak = output.detach().amax(dim=dims) if dims else output.detach()
                prev = self._act_max.get(name)
                self._act_max[name] = peak if prev is None else torch.maximum(prev, peak)
        return fn

    def _relu_hook(self, module, inputs):
        # A ReLU directly consumes a layer's output only if it receives the very same tensor,
        # unmodified. A residual `out += identity` bumps the version counter, so it doesn't count.
        last = self._last_out
        if self.pairs is None and last is not None and inputs and inputs[0] is last[1] \
                and inputs[0]._version == last[2]:
            self._relu_after.add(last[0])

    # ------------------------------------------------------------------ public API
    def note_loss(self, loss: torch.Tensor | float | None) -> None:
        """Record the loss of the current step, for integrations that cannot pass it to `step()`.

        The next `step()` called without a loss uses it; a loss passed to `step()` wins. The
        value is read from the device only on sampled steps.
        """
        self._noted_loss = loss.detach() if isinstance(loss, torch.Tensor) else loss

    def step(self, loss: torch.Tensor | float | None = None) -> None:
        """Call once per optimizer step, after loss.backward() (and scaler.unscale_, if used)
        and before optimizer.step()."""
        if self._closed:
            raise RuntimeError("Monitor.step() called after close()")
        if loss is None:
            loss = self._noted_loss
        self._noted_loss = None
        step, sampled = self.step_count, self._sampling
        self.step_count += 1
        if sampled:
            self._check(step, loss)
        self._sampling = self.step_count % self.config.check_every == 0

    def close(self, report: bool = True) -> Path | None:
        """Remove the hooks, write `run_end`, and (rank 0, report=True) write the reports.

        Returns the engineer report's path, or None (no report, already closed, or not rank 0).
        """
        if self._closed:
            return None
        self._closed = True
        for h in self._handles:
            h.remove()
        if not self._writer:
            return None
        self._metrics.close()
        self._write("run_end", steps=self.step_count, checks=self.checks, counts=self.counts,
                    metrics_sha256=file_sha256(self.out / "metrics.jsonl"))
        self.diary.close()
        if not report:
            return None
        from .report import write_all
        paths = write_all(self.out)
        for p in paths.values():
            self._say(f"report written to {p}")
        return paths["report"]

    def __enter__(self) -> Monitor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ record
    def _say(self, msg: str) -> None:
        if self.verbose and self._writer:
            print(f"[ormas-monitor] {msg}", flush=True)

    def _emit(self, kind: str, payload: dict) -> None:
        failed = []
        for i, sink in enumerate(self.sinks):
            if i in self._sink_failed:
                continue
            try:
                sink(kind, payload)
            except Exception as e:      # a dashboard outage must not stop a training run
                self._sink_failed.add(i)
                failed.append((sink, e))
        for sink, e in failed:
            self._say(f"sink {sink!r} failed and is disabled for this run: {e}")
            self._write("sink_error", sink=repr(sink), error=f"{type(e).__name__}: {e}")

    def _write(self, event: str, **fields: Any) -> dict | None:
        """Append one diary line and pass it to the sinks. Only rank 0 writes; others return None."""
        if not self._writer:
            return None
        line = self.diary.write(event, **fields)
        self._emit("event", line)
        return line

    def _event(self, event: str, step: int, msg: str | None = None, **fields: Any) -> None:
        self._write(event, step=step, **fields)
        if msg:
            self._say(f"step {step:,}: {msg}")

    # ------------------------------------------------------------------ checks
    def _check(self, step: int, loss: torch.Tensor | float | None) -> None:
        self.checks += 1
        if self.pairs is None:
            order = self._order or list(self._all_modules)
            self.pairs = pair_by_order(order, self._all_kinds, self._all_sizes)
            self._last_out = None
            self._act_watch = {self.pairs.get(p, p) for p in self.producers}
            self._event("structure", step, pairs=self.pairs, relu_after=sorted(self._relu_after))
            self._event("manifest", step, **self._manifest())

        loss_val = None
        if loss is not None:
            loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        names, values, scale = self._grad_norms()
        skipped = scale is not None and any(not math.isfinite(g) for g in values)
        # A step the scaler skips does not move the loss baseline either; a non-finite loss is
        # still reported.
        if loss_val is not None and not (skipped and math.isfinite(loss_val)):
            self._check_loss(step, loss_val)

        grads = self._check_gradients(step, names, values, skipped, scale)
        stuck = self._check_channels(step) if self._channel_checks else {}
        row = {"step": step, "loss": loss_val, "grad": grads, "stuck": stuck}
        if self._writer:
            self._metrics.write(_dumps(row) + "\n")
            self._emit("metrics", row)
        self._act_max.clear()

    def _scaler_on(self) -> bool:
        return self.scaler is not None and bool(self.scaler.is_enabled())

    def _scale_state(self) -> tuple[float | None, bool]:
        """(current loss scale or None, are the .grad tensors still multiplied by it?)."""
        if not self._scaler_on() or getattr(self.scaler, "_scale", None) is None:
            return None, False             # no scaler, or scale() never called: plain gradients
        states = getattr(self.scaler, "_per_optimizer_states", {})
        unscaled = False
        for opt in (self.optimizer, self._opt):
            st = states.get(id(opt))      # .get: indexing the defaultdict would insert a state
            if st is not None and getattr(st.get("stage"), "name", None) == "UNSCALED":
                unscaled = True
        return float(self.scaler.get_scale()), not unscaled

    def _grad_norms(self) -> tuple[list[str], list[float], float | None]:
        names, norms = [], []
        for n, m in self.modules.items():
            gs = [p.grad for p in m.parameters(recurse=False) if p.grad is not None]
            if gs:
                names.append(n)
                norms.append(torch.linalg.vector_norm(torch.stack([_norm32(g) for g in gs])))
        if not names:
            return [], [], None
        stacked = torch.stack(norms).float()
        scale, scaled = self._scale_state()
        if scaled:
            stacked = stacked / scale
        return names, stacked.cpu().tolist(), scale

    def _check_loss(self, step: int, x: float) -> None:
        cfg = self.config
        if not math.isfinite(x):
            if not self._loss_bad:
                self.counts["nonfinite"] += 1
                self._event("nonfinite_loss", step, f"loss is {x}", severity="critical", value=str(x))
            self._loss_bad = True
            return
        self._loss_bad = False
        b = self._loss
        if b.n >= cfg.warmup:
            sd = math.sqrt(b.var) + 1e-12
            if x > b.mean + cfg.sigma * sd and x > cfg.loss_spike_factor * b.mean:
                if not self._spiking:     # one event per episode, not one per check
                    self.counts["loss_spikes"] += 1
                    self._event("loss_spike", step, f"loss spiked to {x:.4g} (baseline {b.mean:.4g})",
                                severity="warning", value=x, baseline=b.mean)
                self._spiking = True
                return
            if self._spiking:
                self._event("loss_recovered", step, f"loss back to normal ({x:.4g})", value=x, baseline=b.mean)
            self._spiking = False
        b.update(x, cfg.baseline_rate_floor)

    def _check_gradients(self, step: int, names: list[str], values: list[float], skipped: bool,
                         scale: float | None) -> dict[str, float]:
        if not names:
            return {}
        out = dict(zip(names, values))
        bad = [n for n, g in zip(names, values) if not math.isfinite(g)]
        limit = self.config.layer_list_limit
        if skipped:
            # With a GradScaler, non-finite gradients mean the scale was too high: the scaler
            # skips this optimizer step and lowers the scale. Normal; baselines are not updated.
            self.counts["skipped_steps"] += 1
            self._event("skipped_step", step, f"gradient scaler skips this step "
                        f"(non-finite gradients in {len(bad)} layers, scale {scale:g})",
                        severity="info", scale=scale, layers=bad[:limit])
            return out
        if bad and not self._grad_bad:
            self.counts["nonfinite"] += 1
            self._event("nonfinite_gradient", step, f"non-finite gradients in {len(bad)} layers",
                        severity="critical", layers=bad[:limit])
        self._grad_bad = bool(bad)
        for n, g in out.items():
            if math.isfinite(g):
                self._update_layer(step, n, g)
        return out

    def _update_layer(self, step: int, name: str, g: float) -> None:
        cfg = self.config
        b = self._base[name]
        x = math.log10(g + _LOG_EPS)
        cond = None
        if b.n >= cfg.warmup:
            sd = math.sqrt(b.var) + 1e-6
            margin = max(cfg.sigma * sd, cfg.alarm_min_decades)
            if g == 0 and b.mean > _NO_GRAD_FLOOR:
                cond = "no_gradient"
            elif x < b.mean - margin:
                cond = "vanishing"
            elif x > b.mean + margin:
                cond = "exploding"
        if cond:
            b.pending_count = b.pending_count + 1 if b.pending == cond else 1
            b.pending = cond
            if b.pending_count >= cfg.alarm_confirmations and b.alarm != cond:
                b.alarm = cond
                self.counts["alarms"] += 1
                words = {"no_gradient": "gradient is exactly zero",
                         "vanishing": "gradient collapsed",
                         "exploding": "gradient exploded"}[cond]
                self._event("alarm", step, f"{name} ({self.kinds[name]}): {words} "
                            f"({g:.3g} vs normal {10 ** b.mean:.3g})",
                            severity="warning", layer=name, kind=self.kinds[name],
                            diagnosis=cond, grad_norm=g, baseline=10 ** b.mean, sigma=cfg.sigma)
            return
        b.pending, b.pending_count = None, 0
        if b.alarm:
            self._event("alarm_cleared", step, f"{name}: back to normal", layer=name,
                        diagnosis=b.alarm, grad_norm=g, baseline=10 ** b.mean)
            b.alarm = None
        b.update(x, cfg.baseline_rate_floor)

    def _check_channels(self, step: int) -> dict[str, list[int]]:
        cfg = self.config
        replicated = self.dist_mode in ("ddp", "distributed")
        out = {}
        for name in self.producers:
            prod = self.modules[name]
            bn = self._all_modules.get(self.pairs.get(name, ""))
            weight = getattr(prod, "weight", None)
            if weight is None or weight.dim() < 2:
                continue
            n = weight.shape[0]
            scan = scan_channels(weight, getattr(prod, "bias", None),
                                 getattr(bn, "weight", None) if bn is not None else None,
                                 getattr(bn, "bias", None) if bn is not None else None,
                                 rel_tol=cfg.decayed_rel_tol)
            out_name = self.pairs.get(name) if bn is not None else name
            feeds_relu = out_name in self._relu_after
            # The certificate needs a ReLU directly after the layer, BatchNorm in training mode,
            # and no optimizer momentum left that could still move the channel (so it needs the
            # optimizer to check). Otherwise a zero channel is only reported as constant.
            can_certify = feeds_relu and self._opt is not None and (bn is None or bn.training)
            certified = set(scan.certified) if can_certify else set()
            if certified:
                moving = first_moment_mask(self._opt, prod, bn)
                certified = {c for c in certified if not moving[c]}
            flat = set(scan.flat) | (set(scan.certified) - certified)
            decayed = set(scan.decayed) if feeds_relu else set()
            # "Silent" needs no positive output on any input for `patience` checks in a row,
            # not just one batch.
            observed: set[int] = set()
            act = self._act_max.get(out_name)
            if feeds_relu and act is not None and act.numel() == n:
                quiet = self._silent.setdefault(name, torch.zeros(n, dtype=torch.long))
                now = (act.float().cpu() <= 0)
                quiet[now] += 1
                quiet[~now] = 0
                observed = set((quiet >= cfg.patience).nonzero().flatten().tolist()) - certified - flat - decayed

            # Under torch.distributed every rank must make the same repair decision. Weights are
            # identical across DDP replicas; activations are not, so "silent" is never a target.
            silent_target = self.repair_mode == "all" and not replicated
            mask = torch.zeros(n, dtype=torch.bool)
            targets = certified | flat | decayed | (observed if silent_target else set())
            if targets:
                mask[list(targets)] = True
            streak = self._streak.setdefault(name, torch.zeros(n, dtype=torch.long))
            streak[mask] += 1
            streak[~mask] = 0
            out[name] = [len(certified), len(flat), len(observed), len(decayed)]

            stuck = len(certified | flat | observed | decayed)
            level = _level(stuck, n)
            if level != self._channel_level.get(name, "none"):
                if level != "none":
                    self.counts["stuck_layers"] += 1
                parts = []
                if certified:
                    parts.append(f"{len(certified)} certified dead")
                if flat:
                    parts.append(f"{len(flat)} flat")
                if decayed:
                    parts.append(f"{len(decayed)} decayed to ~0")
                if observed:
                    parts.append(f"{len(observed)} silent for several checks")
                msg = (f"{name} ({self.kinds[name]}): {stuck}/{n} channels stuck: " + ", ".join(parts)
                       if level != "none" else f"{name}: all channels active again")
                lim = cfg.list_limit
                self._event("channels" if level != "none" else "channels_cleared", step, msg,
                            severity="warning" if level != "none" else "info", layer=name,
                            batchnorm=self.pairs.get(name), channels=n, level=level,
                            n_certified=len(certified), n_flat=len(flat), n_observed=len(observed),
                            n_decayed=len(decayed),
                            certified=sorted(certified)[:lim], flat=sorted(flat)[:lim],
                            observed=sorted(observed)[:lim])
                self._channel_level[name] = level
            if certified and name not in self._certified_seen:
                self._certified_seen.add(name)
                self._event("certificate", step, severity="proof", layer=name, batchnorm=self.pairs.get(name),
                            channels=sorted(certified)[:cfg.channel_list_limit], count=len(certified),
                            statement="incoming filter or BatchNorm scale is exactly zero and the "
                                      "BatchNorm shift is <= 0: the ReLU is closed for every input, "
                                      "so every gradient into these channels is exactly zero "
                                      "(Theorem 2, absorbing dead set)",
                            assumes=["ReLU follows", "BatchNorm in training mode",
                                     "optimizer momentum for the channel has decayed"])

            if self.repair_mode:
                # Proven-dead channels cannot recover, so waiting longer than one confirming
                # check gains nothing; unproven cases wait `patience` checks.
                proven = torch.zeros(n, dtype=torch.bool)
                if certified:
                    proven[list(certified)] = True
                due = ((streak >= cfg.patience) | (proven & (streak >= cfg.certified_repair_streak))
                       ).nonzero().flatten()
                if due.numel():
                    self._repair(step, name, prod, bn, due)
                    streak[due] = 0
        return out

    def _repair_seed(self) -> int | None:
        seed = self.config.repair_seed
        if seed is None and self.dist_mode != "none":
            return 0            # every rank must draw the same filters
        return seed

    def _rng_description(self) -> str:
        seed = self._repair_seed()
        if seed is None:
            return "torch global RNG"
        return f"torch.Generator seeded from sha256(seed={seed}, step, layer)"

    def _repair(self, step: int, name: str, prod: nn.Module, bn: nn.Module | None,
                idx: torch.Tensor) -> None:
        seed = self._repair_seed()
        gen = repair_generator(seed, step, name) if seed is not None else None
        res = repair_channels(prod, bn, idx, self._opt, gen)
        self.counts["repairs"] += 1
        self.counts["channels_repaired"] += res.count
        rel = f", ||dW||/||W|| = {res.relative_change:.3g}" if res.relative_change is not None else ""
        self._event("repair", step, f"{name}: re-initialised {res.count} stuck channels{rel}",
                    severity="action", layer=name, batchnorm=self.pairs.get(name),
                    channels=idx.tolist()[:self.config.channel_list_limit], count=res.count,
                    method=METHOD, optimizer_slices_reset=res.optimizer_slices_reset,
                    modification_class=MODIFICATION_CLASS, bound=None, delta_norm=res.delta_norm,
                    weight_norm_before=res.weight_norm_before, weight_norm_after=res.weight_norm_after,
                    relative_change=res.relative_change, rng=self._rng_description())

    def _manifest(self) -> dict[str, Any]:
        """Which mechanisms are actually active for this model and run."""
        notes = []
        if self._channel_checks:
            reason_off = None
        else:
            reason_off = ("parameters are sharded (FSDP): per-channel weight scans, the dead-channel "
                          "certificate and repair are disabled")
            notes.append(reason_off)
        producers = []
        for name in self.producers:
            prod = self.modules[name]
            bn_name = self.pairs.get(name) if self.pairs.get(name) in self._all_modules else None
            relu = (bn_name or name) in self._relu_after
            weight = getattr(prod, "weight", None)
            if reason_off:
                why = reason_off
            elif weight is None or weight.dim() < 2:
                why = "no 2-D weight to scan"
            elif not relu:
                why = "no ReLU module directly consumes the output (residual add, other activation, " \
                      "or a functional relu)"
            elif self._opt is None:
                why = "no optimizer given, so the momentum condition cannot be checked"
            else:
                why = None
            producers.append({"layer": name, "kind": self.kinds[name], "batchnorm": bn_name,
                              "relu_after": relu, "certificate": why is None, "reason": why})
        if any(p["certificate"] for p in producers):
            notes.append("a certificate is issued per channel only if, at that check, the optimizer "
                         "holds no first-moment momentum for it and its BatchNorm is in training mode")
        replicated = self.dist_mode in ("ddp", "distributed")
        targets = []
        if self.repair_mode:
            targets = ["certified", "flat", "decayed"]
            if self.repair_mode == "all":
                if replicated:
                    notes.append("repair='all' under torch.distributed: silent channels are not repaired, "
                                 "because activations differ between ranks")
                else:
                    targets.append("silent")
        scale, _ = self._scale_state()
        precision = {"scaler": self.scaler is not None, "scaler_enabled": self._scaler_on(),
                     "scale": scale, "gradient_norms": "float32, cast before the reduction"}
        if self._scaler_on():
            notes.append("with a gradient scaler, non-finite gradients are recorded as skipped_step; "
                         "skipped steps are seen only on sampled steps")
        if self.dist_mode == "fsdp":
            notes.append("under FSDP each rank sees gradient norms of its own shards; rank 0's are recorded")
        if self.dist_mode != "none":
            notes.append("only rank 0 writes the record and calls sinks; every rank runs the hooks")
        return {
            "check_every": self.config.check_every,
            "layers": {"monitored": len(self.modules), "total": len(self._all_modules),
                       "filter": self.layer_patterns},
            "producers": producers,
            "channel_checks": self._channel_checks,
            "repair": {"mode": self.repair_mode, "targets": targets,
                       "rng": self._rng_description() if self.repair_mode else None,
                       "modification_class": MODIFICATION_CLASS, "bound": None},
            "precision": precision,
            "distributed": {"mode": self.dist_mode, "rank": self.rank, "world_size": self.world_size,
                            "wrappers": self._wrappers},
            "notes": notes,
        }


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))
