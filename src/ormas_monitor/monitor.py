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

Thresholds live in `MonitorConfig` (config.py); the event schema in schema.py; repair and the
optimizer rules in repair.py. At the first check a `manifest` event records which mechanisms are
actually active for this model and run (certificate per layer, repair mode, mixed precision,
distributed mode).

The certificate (Theorem 2) is issued for a channel of a Conv/Linear -> BatchNorm -> ReLU chain
only when all of these hold, and `certificate_refused` says which one failed:
  * structure, from the first sampled step: the BatchNorm's input *is* the producer's output
    tensor, unmodified (dataflow, not execution order); in the autograd graph of that step the
    producer's output has exactly one consumer, the BatchNorm, and the BatchNorm's exactly one,
    a ReLU module (dtype casts in between are allowed); producer and BatchNorm each ran exactly
    once per forward pass, then and at every later check (otherwise the layer is refused for good);
  * weights: zero filter with zero or no producer bias, or zero BatchNorm scale; shift <= 0;
  * optimizer: SGD or the Adam family (or an opt-in type), weight decay that cannot lift the
    shift above 0, and no first-moment state for the channel;
  * at every check: BatchNorm in training mode, and the gradient of the channel's filter row,
    producer bias, scale and shift exactly zero.

Changes in 0.2:
  * thresholds moved to MonitorConfig; keyword shortcuts still override it;
  * `scaler=` (torch.amp.GradScaler): non-finite gradients are a skipped step, not an alarm;
    norms are taken in float32 and divided by the scale when `unscale_` was not called;
  * torch.distributed: only rank 0 writes files, prints or calls sinks; under DDP repair
    targets weight-detected channels only and uses a seeded generator; FSDP disables channel
    checks and refuses repair;
  * `layers=` fnmatch filter; `note_loss()`; torch.compile / DDP wrappers are unwrapped;
  * the certificate conditions above (dataflow pairing, autograd consumer check, optimizer
    allow-list, per-check gradient guard); `certificate_refused`, `no_gradients`, `loss_reduced`;
  * never breaks training: empty batches, sparse gradients, models split across devices and
    unreduced losses are handled; a constant number of host reads per check; `close()` never
    raises over a pending exception;
  * `overwrite=False`: a run directory that already holds a record is refused;
  * sinks receive every diary line, run_start and run_end included; each failing sink
    produces exactly one `sink_error`.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from . import __version__
from .config import MonitorConfig
from .diary import Diary, file_sha256
from .repair import METHOD, MODIFICATION_CLASS, decay_reason, find_optimizer, moving_mask, optimizer_family, \
    repair_channels, repair_generator
from .schema import RECORD_VERSION
from .structure import BN_KINDS, CONV_KINDS, RELU_KINDS, channel_masks

_LOG_EPS = 1e-30          # log10(0 + eps) = -30: a zero gradient norm, kept finite
_NO_GRAD_FLOOR = -29.0    # a baseline above this was not itself built from zero gradients
_HALF = (torch.float16, torch.bfloat16)
_TRANSPARENT = ("ToCopyBackward0",)   # dtype casts (autocast) between two modules: elementwise
_EXTERNAL = object()                  # graph "parent" of a root: consumed outside the model

# Per-channel flag bits, packed into one uint8 tensor per layer so a check reads the host once.
_CAND, _FLAT, _DECAYED, _CERT, _R_MOMENTUM, _R_GRADIENT, _QUIET, _ACT = 1, 2, 4, 8, 16, 32, 64, 128

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


def _dense(g: torch.Tensor) -> torch.Tensor:
    """The stored values of a sparse gradient (nn.Embedding(sparse=True)), else `g` itself."""
    if g.layout == torch.sparse_coo:
        return g.coalesce().values()
    if g.layout != torch.strided:
        return g.values()
    return g


def _norm32(g: torch.Tensor) -> torch.Tensor:
    # Telemetry stays full precision: cast before the reduction, never after.
    g = _dense(g)
    return torch.linalg.vector_norm(g, dtype=torch.float32) if g.dtype in _HALF else torch.linalg.vector_norm(g)


def _grad_fns(obj: Any, depth: int = 0) -> list:
    """Autograd nodes of the tensors in a model output (tensor, sequence, mapping), bounded."""
    if isinstance(obj, torch.Tensor):
        return [obj.grad_fn] if obj.grad_fn is not None else []
    if depth >= 4:
        return []
    items = obj.values() if isinstance(obj, Mapping) else obj if isinstance(obj, (list, tuple)) else ()
    return [fn for x in list(items)[:64] for fn in _grad_fns(x, depth + 1)]


def _graph_parents(roots: list) -> dict:
    """node -> list of nodes that consume it (one entry per edge), over the graph under `roots`.
    Each root also gets the `_EXTERNAL` parent: its value leaves the walked graph."""
    parents: dict = {}
    for r in roots:
        parents.setdefault(r, []).append(_EXTERNAL)
    seen, stack = set(), list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for nxt, _ in node.next_functions:
            if nxt is not None:
                parents.setdefault(nxt, []).append(node)
                stack.append(nxt)
    return parents


def _sole_consumer(parents: dict, node: Any) -> tuple[Any, str | None]:
    """(the one node consuming `node`'s output, looking through dtype casts, None) or (None, why)."""
    while True:
        ps = parents.get(node)
        if ps is None:
            return None, "is not part of the autograd graph the monitor walked"
        if len(ps) != 1:
            return None, f"has {len(ps)} consumers in the autograd graph"
        if ps[0] is _EXTERNAL:
            return None, "is returned from the model, so its consumers are unknown"
        if type(ps[0]).__name__ not in _TRANSPARENT:
            return ps[0], None
        node = ps[0]


def _dense_grad(g: torch.Tensor) -> torch.Tensor:
    return g.to_dense() if g.layout != torch.strided else g


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


@dataclass
class _Chain:
    """One producer (Conv/Linear with a 2-D+ weight) and what follows it, fixed at the first check.
    `reason` is None while the layer may be certified, else why it may not (structural, or a
    module-call change seen later; both permanent). `offset` locates its channels in the
    monitor's concatenated per-channel counters."""
    name: str
    prod: nn.Module
    bn: nn.Module | None
    bn_name: str | None
    out_name: str
    n: int
    offset: int
    relu: bool
    reason: str | None


class Monitor:
    """Watch a model while it trains. See the module docstring for the loop.

    Args:
        model: the model being trained (a torch.compile or DDP wrapper is fine).
        optimizer: needed for repair and for the certificate's momentum check. Framework
            wrappers (Lightning, accelerate) are unwrapped. An object that is not a torch
            optimizer (e.g. DeepSpeed's) is accepted for watching: no certificate, no repair.
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
        overwrite: replace a record already in `out`. False (default) raises FileExistsError
            rather than destroy a previous, possibly signed, record.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *,
                 out: str | Path = "ormas_run", repair: bool | str = False,
                 check_every: int | None = None, patience: int | None = None,
                 sigma: float | None = None, warmup: int | None = None, name: str | None = None,
                 sinks: Sequence[Sink] = (), verbose: bool = True,
                 config: MonitorConfig | None = None, scaler: Any = None,
                 layers: str | Sequence[str] | None = None, overwrite: bool = False) -> None:
        if config is not None and not isinstance(config, MonitorConfig):
            raise TypeError(f"config must be a MonitorConfig, got {type(config).__name__}")
        overrides = {k: v for k, v in (("check_every", check_every), ("patience", patience),
                                       ("sigma", sigma), ("warmup", warmup)) if v is not None}
        self.config = (config or MonitorConfig()).replace(**overrides)
        if repair and optimizer is None:
            raise ValueError("repair needs the optimizer, so its state for repaired channels can be reset")
        if repair not in (False, True, "all"):
            raise ValueError("repair must be False, True (certified and flat channels) or 'all'")
        self._opt = find_optimizer(optimizer)
        if repair and self._opt is None:
            raise ValueError(f"repair needs a torch.optim.Optimizer to reset the repaired channels' state; "
                             f"{type(optimizer).__name__} is not one. Use repair=False to watch only")
        if scaler is not None:
            if not (callable(getattr(scaler, "is_enabled", None)) and callable(getattr(scaler, "get_scale", None))):
                raise TypeError(f"scaler must be a torch.amp.GradScaler, got {type(scaler).__name__}")
            if optimizer is None:
                raise ValueError("scaler needs the optimizer: its per-optimizer state says whether "
                                 "unscale_ was called")

        self.model, self.optimizer = model, optimizer
        self._family = optimizer_family(self._opt, self.config.certify_optimizers)
        self._net, self._wrappers = _unwrap_model(model)
        self.dist_mode, self.rank, self.world_size = _distributed(self._net, self._wrappers)
        if self.dist_mode == "fsdp" and repair:
            raise ValueError("repair is not supported under FSDP: parameters are sharded and flattened, "
                             "so per-channel filters cannot be read or rewritten. Use repair=False; "
                             "gradient alarms and the record still work")
        self.out = Path(out)
        self._writer = self.rank == 0
        if self._writer and not overwrite and (self.out / "diary.jsonl").exists():
            raise FileExistsError(f"{self.out} already holds a run record (diary.jsonl); starting a new run "
                                  f"there would destroy it. Choose another out= directory, or pass "
                                  f"overwrite=True to replace it")
        self.repair_mode = repair
        self.scaler = scaler
        self.verbose = verbose
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
        self._noted_fn = None
        # Structure detection, during the first sampled step only.
        self._detecting = True
        self._outs: dict[int, tuple[str, torch.Tensor, int]] = {}   # this forward's outputs, by id
        self._feeds: dict[str, set[str | None]] = {}   # BatchNorm -> producers whose output it got
        self._relu_after: set[str] = set()      # layers whose output goes straight into a ReLU
        self._fn: dict[str, Any] = {}           # module -> autograd node of its last output
        self._relu_fns: set = set()             # autograd nodes made by ReLU modules
        self._root_fns: list = []               # autograd nodes of the model's last outputs
        self.graph_source: str | None = None    # "loss" | "model outputs" | None
        # Module calls per forward pass, on sampled steps (shared modules, dynamic graphs).
        self._fwd: Counter | None = None
        self._profiles: list[Counter] = []
        self._loose: Counter = Counter()        # calls outside the root module's forward
        self._act_max: dict[str, torch.Tensor] = {}
        self._act_watch: set[str] | None = None  # None until the structure is known: watch all
        self.pairs: dict[str, str] | None = None
        self._chains: list[_Chain] = []
        self._silent_all = self._streak_all = self._layer_id = torch.zeros(0, dtype=torch.long)
        self._base = {n: _Baseline() for n in self.modules}
        self._loss = _Baseline()
        self._loss_bad = False
        self._spiking = False
        self._grad_bad = False
        self._channel_level: dict[str, str] = {}
        self._certified_seen: set[str] = set()
        self._refused_seen: set[tuple[str, str]] = set()
        self._warned: set[str] = set()
        self.counts = {"alarms": 0, "stuck_layers": 0, "repairs": 0, "channels_repaired": 0,
                       "loss_spikes": 0, "nonfinite": 0, "skipped_steps": 0, "checks_without_gradients": 0}
        self._closed = False
        self._error: str | None = None

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
        elif optimizer is not None:
            opt = {"type": type(optimizer).__name__}
        self._write(
            "run_start", name=name or self.out.name, tool=f"ormas-monitor {__version__}",
            model=type(net).__name__,
            parameters=sum(p.numel() for p in net.parameters()),
            layers=len(self.modules), architecture_sha256=hashlib.sha256(shapes.encode()).hexdigest(),
            optimizer=opt, python=sys.version.split()[0], torch=torch.__version__,
            platform=platform.platform(), record_version=RECORD_VERSION,
            config={**self.config.to_dict(), "repair": repair, "layers": self.layer_patterns})

        self._handles = [net.register_forward_pre_hook(self._root_pre), net.register_forward_hook(self._root_post)]
        for n, m in self._all_modules.items():
            if self._all_kinds[n] in CONV_KINDS or self._all_kinds[n] in BN_KINDS:
                self._handles.append(m.register_forward_hook(self._hook(n)))
            if self._all_kinds[n] in BN_KINDS:
                self._handles.append(m.register_forward_pre_hook(self._bn_pre_hook(n)))
        self._relu_handles = []
        for m in net.modules():
            if isinstance(m, RELU_KINDS):
                self._relu_handles += [m.register_forward_pre_hook(self._relu_pre_hook),
                                       m.register_forward_hook(self._relu_hook)]
        self._handles += self._relu_handles

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
    # Every hook returns at once on unsampled steps. On sampled steps the root hooks delimit
    # forward passes so module calls can be counted per pass; during the first sampled step the
    # module hooks also record outputs and autograd nodes for structure detection.
    def _root_pre(self, module, inputs):
        if self._sampling and module.training and torch.is_grad_enabled():
            self._fwd = Counter()

    def _root_post(self, module, inputs, output):
        if self._fwd is None:
            return
        self._profiles.append(self._fwd)
        self._fwd = None
        if self._detecting:
            self._outs.clear()           # pairing and ReLU checks happen within one forward
            fns = _grad_fns(output)
            if fns:
                self._root_fns = fns

    def _hook(self, name: str):
        kind = self._all_kinds[name]

        def fn(module, inputs, output):
            if not self._sampling or not isinstance(output, torch.Tensor) or output.numel() == 0:
                return
            if torch.is_grad_enabled():
                (self._fwd if self._fwd is not None else self._loose)[name] += 1
            if not module.training:
                return
            if self._detecting:
                self._outs[id(output)] = (name, output, output._version)
                if output.grad_fn is not None:
                    self._fn[name] = output.grad_fn
            elif name not in self._act_watch:
                return
            with torch.no_grad():
                if kind == "Linear":
                    dims = tuple(range(output.dim() - 1))
                else:
                    dims = (0,) + tuple(range(2, output.dim()))
                peak = output.detach().amax(dim=dims) if dims else output.detach()
                prev = self._act_max.get(name)
                self._act_max[name] = peak if prev is None or prev.shape != peak.shape \
                    else torch.maximum(prev, peak)
        return fn

    def _same_output(self, x: Any) -> str | None:
        """Name of the module whose output, in this forward, is `x` itself and unmodified."""
        rec = self._outs.get(id(x)) if isinstance(x, torch.Tensor) else None
        if rec is not None and rec[1] is x and x._version == rec[2]:
            return rec[0]
        return None

    def _bn_pre_hook(self, name: str):
        def fn(module, inputs):
            # Pair by dataflow: the BatchNorm's input must be the producer's output tensor itself.
            # A tensor modified in place since (version counter bumped) does not count.
            if self._detecting and self._sampling and module.training:
                src = self._same_output(inputs[0]) if inputs else None
                self._feeds.setdefault(name, set()).add(src if self._all_kinds.get(src) in CONV_KINDS else None)
        return fn

    def _relu_pre_hook(self, module, inputs):
        # A ReLU directly consumes a layer's output only if it receives the very same tensor,
        # unmodified. A residual `out += identity` bumps the version counter, so it doesn't count.
        if self._detecting and self._sampling and inputs:
            src = self._same_output(inputs[0])
            if src is not None:
                self._relu_after.add(src)

    def _relu_hook(self, module, inputs, output):
        if self._detecting and self._sampling and isinstance(output, torch.Tensor) and output.grad_fn is not None:
            self._relu_fns.add(output.grad_fn)

    # ------------------------------------------------------------------ public API
    def note_loss(self, loss: torch.Tensor | float | None) -> None:
        """Record the loss of the current step, for integrations that cannot pass it to `step()`.

        The next `step()` called without a loss uses it; a loss passed to `step()` wins. The
        value is read from the device only on sampled steps. On the first sampled step a loss
        that still has its autograd graph also serves the certificate's consumer check.
        """
        self._noted_fn = loss.grad_fn if isinstance(loss, torch.Tensor) and self._detecting else None
        self._noted_loss = loss.detach() if isinstance(loss, torch.Tensor) else loss

    def step(self, loss: torch.Tensor | float | None = None) -> None:
        """Call once per optimizer step, after loss.backward() (and scaler.unscale_, if used)
        and before optimizer.step().

        A loss with more than one element (unreduced) is recorded as its mean; a `loss_reduced`
        event says so once.
        """
        if self._closed:
            raise RuntimeError("Monitor.step() called after close()")
        if loss is None:
            loss, root = self._noted_loss, self._noted_fn
        else:
            root = loss.grad_fn if isinstance(loss, torch.Tensor) else None
        self._noted_loss = self._noted_fn = None
        step, sampled = self.step_count, self._sampling
        self.step_count += 1
        if sampled:
            self._check(step, loss, root)
        self._sampling = self.step_count % self.config.check_every == 0

    def close(self, report: bool = True) -> Path | None:
        """Remove the hooks, write `run_end`, and (rank 0, report=True) write the reports.

        Returns the engineer report's path, or None (no report, already closed, not rank 0, or
        the record could not be finished while another exception was propagating). Called while
        an exception is propagating (a `finally:` block, `__exit__`), a failure here is printed
        to stderr and does not replace the original exception.
        """
        if self._closed:
            return None
        self._closed = True
        for h in self._handles:
            h.remove()
        if not self._writer:
            return None
        pending = sys.exc_info()[1]
        if pending is not None and self._error is None:
            self._error = f"{type(pending).__name__}: {pending}"
        try:
            return self._finish(report)
        except Exception as e:
            if pending is None:
                raise
            print(f"[ormas-monitor] could not finish the record in {self.out} "
                  f"({type(e).__name__}: {e}); the training error propagates unchanged",
                  file=sys.stderr, flush=True)
            return None

    def _finish(self, report: bool) -> Path | None:
        self._metrics.close()
        extra = {"error": self._error} if self._error is not None else {}
        self._write("run_end", steps=self.step_count, checks=self.checks, counts=self.counts,
                    metrics_sha256=file_sha256(self.out / "metrics.jsonl"), **extra)
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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
            return
        self._error = f"{exc_type.__name__}: {exc}"
        try:
            self.close()
        except Exception as e:           # never mask the exception that ended training
            print(f"[ormas-monitor] could not finish the record in {self.out} ({type(e).__name__}: {e})",
                  file=sys.stderr, flush=True)

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

    # ------------------------------------------------------------------ structure
    def _runs_once(self, name: str) -> int | None:
        """None if `name` ran exactly once in every forward pass of this sampled window, else the
        offending call count. With no root forward seen, the window's total count is used."""
        counts = [f[name] for f in self._profiles] if self._profiles else [self._loose[name]]
        bad = [c for c in counts if c != 1]
        return bad[0] if bad else None

    def _finish_structure(self, step: int, loss_root: Any) -> None:
        """Fix pairs, ReLU followers and per-layer certificate eligibility from the first sampled step."""
        owners: dict[str, list[str]] = {}
        for bn, srcs in self._feeds.items():
            src = next(iter(srcs)) if len(srcs) == 1 else None
            if src is not None and self._all_sizes.get(src) == self._all_sizes.get(bn):
                owners.setdefault(src, []).append(bn)
        self.pairs = {p: bns[0] for p, bns in owners.items() if len(bns) == 1}

        roots = [loss_root] if loss_root is not None else self._root_fns
        self.graph_source = None if not roots else "loss" if loss_root is not None else "model outputs"
        parents = _graph_parents(roots) if roots else {}

        offset = 0
        for name in self.producers:
            prod = self.modules[name]
            weight = getattr(prod, "weight", None)
            if weight is None or weight.dim() < 2:
                continue
            bn_name = self.pairs.get(name)
            relu = (bn_name or name) in self._relu_after
            reason = self._static_reason(name, bn_name, relu, parents) if self._channel_checks else None
            self._chains.append(_Chain(name=name, prod=prod, bn=self._all_modules[bn_name] if bn_name else None,
                                       bn_name=bn_name, out_name=bn_name or name, n=weight.shape[0],
                                       offset=offset, relu=relu, reason=reason))
            offset += weight.shape[0]
        self._silent_all = torch.zeros(offset, dtype=torch.long)
        self._streak_all = torch.zeros(offset, dtype=torch.long)
        self._layer_id = torch.repeat_interleave(torch.arange(len(self._chains)),
                                                 torch.tensor([c.n for c in self._chains], dtype=torch.long))
        self._act_watch = {self.pairs.get(p, p) for p in self.producers}
        self._event("structure", step, pairs=self.pairs, relu_after=sorted(self._relu_after),
                    graph=self.graph_source)
        self._event("manifest", step, **self._manifest())
        # Detection is over: drop the references it held and the ReLU hooks it needed.
        self._detecting = False
        self._outs, self._feeds, self._fn, self._relu_fns, self._root_fns = {}, {}, {}, set(), []
        for h in self._relu_handles:
            h.remove()

    def _static_reason(self, name: str, bn_name: str | None, relu: bool, parents: dict) -> str | None:
        """Why a producer can never be certified in this run, or None."""
        if not relu:
            return ("no ReLU module directly consumes the output (residual add, other activation, "
                    "or a functional relu)")
        if self.optimizer is None:
            return "no optimizer given, so the momentum condition cannot be checked"
        if self._opt is None:
            return (f"{type(self.optimizer).__name__} is not a torch.optim.Optimizer, so its momentum "
                    "cannot be read")
        if self._family is None:
            return (f"{type(self._opt).__name__} is not an optimizer the certificate covers (SGD, Adam, "
                    "AdamW, NAdam, RAdam, Adamax; others via MonitorConfig.certify_optimizers)")
        for mod in (name, bn_name):
            if mod is not None:
                calls = self._runs_once(mod)
                if calls is not None:
                    return f"{mod} ran {calls} times in one forward pass (a shared or skipped module)"
        if not parents:
            return ("no autograd graph at the first check: step() got no loss with a grad_fn and the "
                    "model's outputs had none")
        node = self._fn.get(name)
        if node is None:
            return f"no autograd node recorded for {name} (its forward ran without gradients)"
        consumer, why = _sole_consumer(parents, node)
        if why is not None:
            return f"the output of {name} {why}"
        if bn_name is not None:
            if consumer is not self._fn.get(bn_name):
                return f"the output of {name} does not go into {bn_name} alone"
            consumer, why = _sole_consumer(parents, consumer)
            if why is not None:
                return f"the output of {bn_name} {why}"
        if consumer not in self._relu_fns:
            return f"the output of {bn_name or name} is not consumed by a ReLU module alone"
        return None

    # ------------------------------------------------------------------ checks
    def _check(self, step: int, loss: torch.Tensor | float | None, loss_root: Any) -> None:
        self.checks += 1
        if self._detecting:
            self._finish_structure(step, loss_root)
        if isinstance(loss, torch.Tensor) and loss.numel() != 1:
            if loss.numel() > 0 and "loss_reduced" not in self._warned:
                self._warned.add("loss_reduced")
                self._event("loss_reduced", step, f"step() got a loss with {loss.numel()} elements; "
                            "recording its mean (pass loss.mean() to silence this)",
                            severity="info", elements=loss.numel())
            loss = loss.detach().float().mean() if loss.numel() > 0 else None

        names, values, loss_val, scale = self._read_scalars(loss)
        skipped = scale is not None and any(not math.isfinite(g) for g in values)
        # A step the scaler skips does not move the loss baseline either; a non-finite loss is
        # still reported.
        if loss_val is not None and not (skipped and math.isfinite(loss_val)):
            self._check_loss(step, loss_val)

        if not names:
            self.counts["checks_without_gradients"] += 1
            if "no_gradients" not in self._warned:
                self._warned.add("no_gradients")
                msg = "no gradients on any watched layer — call mon.step() after loss.backward()"
                self._event("no_gradients", step, msg, severity="warning", message=msg)
        grads = self._check_gradients(step, names, values, skipped, scale)
        stuck = self._check_channels(step) if self._channel_checks else {}
        row = {"step": step, "loss": loss_val, "grad": grads, "stuck": stuck}
        if self._writer:
            self._metrics.write(_dumps(row) + "\n")
            self._emit("metrics", row)
        self._act_max.clear()
        self._profiles.clear()
        self._loose.clear()

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

    def _read_scalars(self, loss: torch.Tensor | float | None
                      ) -> tuple[list[str], list[float], float | None, float | None]:
        """(layers with gradients, their gradient norms, the loss, the scaler's scale), with one
        host read per device holding gradients. Norms are stacked per device, so a model split
        across devices never stacks tensors from two devices."""
        names, norms = [], []
        for n, m in self.modules.items():
            gs = [p.grad for p in m.parameters(recurse=False) if p.grad is not None]
            if gs:
                names.append(n)
                dev = gs[0].device
                norms.append(torch.linalg.vector_norm(torch.stack([_norm32(g).to(dev) for g in gs])))
        loss_t = loss.detach() if isinstance(loss, torch.Tensor) else None
        loss_val = float(loss) if loss is not None and loss_t is None else None
        scale, scaled = self._scale_state() if names else (None, False)
        by_dev: dict[torch.device, list[int]] = {}
        for i, t in enumerate(norms):
            by_dev.setdefault(t.device, []).append(i)
        values: list[float] = [0.0] * len(norms)
        for dev, idx in by_dev.items():
            stacked = torch.stack([norms[i] for i in idx]).float()
            if scaled:
                stacked = stacked / scale
            # The loss rides along with the norms when that is exact (same device, <= 32 bits).
            with_loss = loss_t is not None and loss_t.device == dev and loss_t.dtype in (torch.float32,) + _HALF
            if with_loss:
                stacked = torch.cat([stacked, loss_t.reshape(1).float()])
            got = stacked.cpu().tolist()
            if with_loss:
                loss_val, loss_t = got.pop(), None
            for i, v in zip(idx, got):
                values[i] = v
        if loss_t is not None:
            loss_val = float(loss_t)
        return names, values, loss_val, scale

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

    @staticmethod
    def _channel_params(ch: _Chain) -> list[torch.Tensor]:
        ps = [ch.prod.weight, getattr(ch.prod, "bias", None)]
        if ch.bn is not None:
            ps += [getattr(ch.bn, "weight", None), getattr(ch.bn, "bias", None)]
        return [p for p in ps if p is not None]

    def _layer_reason(self, ch: _Chain, step: int, groups: dict[int, dict]) -> str | None:
        """Why `ch` may not be certified at this check, or None. Structural reasons stick."""
        if ch.reason is None and ch.bn is not None and not ch.bn.training:
            return f"BatchNorm {ch.bn_name} is in eval mode (running statistics can reopen the ReLU)"
        if ch.reason is None and (self._profiles or self._loose):
            for mod in (ch.name, ch.bn_name):
                calls = self._runs_once(mod) if mod is not None else None
                if calls is not None:
                    ch.reason = (f"{mod} ran {calls} times in one forward pass at step {step:,}, unlike "
                                 "at the first check (the graph changed)")
                    break
        if ch.reason is not None:
            return ch.reason
        shift = getattr(ch.bn, "bias", None) if ch.bn is not None else getattr(ch.prod, "bias", None)
        gamma = getattr(ch.bn, "weight", None) if ch.bn is not None else None
        for p in (shift, gamma):
            group = groups.get(id(p)) if p is not None else None
            if group is not None:
                why = decay_reason(self._opt, self._family, group)
                if why is not None:
                    return why
        for p in self._channel_params(ch):
            if p.requires_grad and p.grad is None:
                return f"no gradient on {ch.name} at this check"
        return None

    def _flags(self, ch: _Chain, may_certify: bool) -> torch.Tensor:
        """Per-channel flag bits for one layer, as a uint8 tensor on the weights' device."""
        prod, bn, n = ch.prod, ch.bn, ch.n
        bias = getattr(prod, "bias", None)
        gamma = getattr(bn, "weight", None) if bn is not None else None
        beta = getattr(bn, "bias", None) if bn is not None else None
        m = channel_masks(prod.weight, bias, gamma, beta, rel_tol=self.config.decayed_rel_tol)
        dev = m.certified.device

        def bit(mask: torch.Tensor, value: int) -> torch.Tensor:
            return mask.to(torch.uint8) * value

        flags = bit(m.certified, _CAND) | bit(m.flat, _FLAT)
        if ch.relu:
            flags |= bit(m.decayed, _DECAYED)
        if may_certify:
            # Each way of being constant depends on different state: a zero filter needs the
            # filter, bias and shift to stay put; a zero scale needs the scale and shift to.
            shift = beta if bn is not None else bias
            zero_ok = m.zero_case & ~moving_mask(self._opt, self._family, [prod.weight, bias, shift], n, dev)
            gamma_ok = m.gamma_case & ~moving_mask(self._opt, self._family, [gamma, shift], n, dev)
            passes = zero_ok | gamma_ok
            # Defence in depth: whatever the structure analysis concluded, a channel whose filter
            # row, bias, scale or shift has a nonzero (or non-finite) gradient now is not dead.
            live = torch.zeros(n, dtype=torch.bool, device=dev)
            for p in self._channel_params(ch):
                if p.grad is not None:
                    live |= _dense_grad(p.grad).reshape(n, -1).ne(0).any(dim=1).to(dev)
            flags |= bit(passes & ~live, _CERT) | bit(m.certified & ~passes, _R_MOMENTUM) | \
                bit(passes & live, _R_GRADIENT)
        act = self._act_max.get(ch.out_name)
        if ch.relu and act is not None and act.numel() == n:
            flags |= bit(act.reshape(-1).to(dev) <= 0, _QUIET) | _ACT
        return flags

    def _check_channels(self, step: int) -> dict[str, list[int]]:
        cfg = self.config
        chains = self._chains
        if not chains:
            return {}
        groups = {id(p): g for g in self._opt.param_groups for p in g["params"]} if self._family else {}
        reasons = [self._layer_reason(ch, step, groups) for ch in chains]
        # Device side: every layer's flags, then one host read per device.
        flags = [self._flags(ch, r is None) for ch, r in zip(chains, reasons)]
        by_dev: dict[torch.device, list[int]] = {}
        for i, f in enumerate(flags):
            by_dev.setdefault(f.device, []).append(i)
        host: list[torch.Tensor] = list(flags)
        for idx in by_dev.values():
            got = torch.cat([flags[i] for i in idx]).cpu().split([chains[i].n for i in idx])
            for i, t in zip(idx, got):
                host[i] = t
        F = torch.cat(host).long()

        # Host side, vectorised over every channel of every layer (CPU tensors, no device reads).
        def has(b: int) -> torch.Tensor:
            return (F & b) != 0
        cert, cand, act, quiet = has(_CERT), has(_CAND), has(_ACT), has(_QUIET)
        flat = has(_FLAT) | (cand & ~cert)
        decayed = has(_DECAYED)
        silent = self._silent_all
        silent[act & quiet] += 1
        silent[act & ~quiet] = 0
        # "Silent" needs no positive output on any input for `patience` checks in a row.
        observed = (silent >= cfg.patience) & act & ~cert & ~flat & ~decayed
        # Under torch.distributed every rank must make the same repair decision. Weights are
        # identical across DDP replicas; activations are not, so "silent" is never a target.
        silent_target = self.repair_mode == "all" and self.dist_mode not in ("ddp", "distributed")
        targets = cert | flat | decayed | (observed if silent_target else torch.zeros_like(cert))
        streak = self._streak_all
        streak[targets] += 1
        streak[~targets] = 0
        if self.repair_mode:
            # Proven-dead channels cannot recover, so waiting longer than one confirming
            # check gains nothing; unproven cases wait `patience` checks.
            due = (streak >= cfg.patience) | (cert & (streak >= cfg.certified_repair_streak))
        else:
            due = torch.zeros_like(cert)
        per_layer = torch.zeros(len(chains), 8, dtype=torch.long).index_add_(
            0, self._layer_id, torch.stack([cert, flat, observed, decayed, due, cand,
                                            has(_R_MOMENTUM), has(_R_GRADIENT)], 1).long()).tolist()

        out = {}
        lim = cfg.list_limit
        for ch, reason, counts in zip(chains, reasons, per_layer):
            n_cert, n_flat, n_obs, n_dec, n_due, n_cand, n_mom, n_grad = counts
            name, n, sl = ch.name, ch.n, slice(ch.offset, ch.offset + ch.n)

            def ids(mask: torch.Tensor, sl: slice = sl) -> list[int]:
                return mask[sl].nonzero().flatten().tolist()
            out[name] = [n_cert, n_flat, n_obs, n_dec]
            stuck = n_cert + n_flat + n_obs + n_dec      # the four sets are disjoint
            level = _level(stuck, n)
            if level != self._channel_level.get(name, "none"):
                if level != "none":
                    self.counts["stuck_layers"] += 1
                parts = []
                if n_cert:
                    parts.append(f"{n_cert} certified dead")
                if n_flat:
                    parts.append(f"{n_flat} flat")
                if n_dec:
                    parts.append(f"{n_dec} decayed to ~0")
                if n_obs:
                    parts.append(f"{n_obs} silent for several checks")
                msg = (f"{name} ({self.kinds[name]}): {stuck}/{n} channels stuck: " + ", ".join(parts)
                       if level != "none" else f"{name}: all channels active again")
                self._event("channels" if level != "none" else "channels_cleared", step, msg,
                            severity="warning" if level != "none" else "info", layer=name,
                            batchnorm=ch.bn_name, channels=n, level=level,
                            n_certified=n_cert, n_flat=n_flat, n_observed=n_obs, n_decayed=n_dec,
                            certified=ids(cert)[:lim], flat=ids(flat)[:lim], observed=ids(observed)[:lim])
                self._channel_level[name] = level
            if n_cert and name not in self._certified_seen:
                self._certified_seen.add(name)
                self._event("certificate", step, severity="proof", layer=name, batchnorm=ch.bn_name,
                            channels=ids(cert)[:cfg.channel_list_limit], count=n_cert,
                            statement="incoming filter or BatchNorm scale is exactly zero and the "
                                      "BatchNorm shift is <= 0: the ReLU is closed for every input, "
                                      "so every gradient into these channels is exactly zero "
                                      "(Theorem 2, absorbing dead set)",
                            assumes=["ReLU follows", "BatchNorm in training mode",
                                     "the optimizer holds no first-moment state for the channel",
                                     "the ReLU is the BatchNorm output's only consumer",
                                     "the channel's gradient is exactly zero at this check",
                                     "weight decay cannot lift the shift above zero"])
            refusals = [(n_cand, reason)] if reason is not None else [
                (n_mom, "the optimizer still holds momentum for these channels"),
                (n_grad, "these channels have a nonzero gradient at this check")]
            for count, why in refusals:
                if count and (name, why) not in self._refused_seen:
                    self._refused_seen.add((name, why))
                    self._event("certificate_refused", step, severity="info", layer=name,
                                batchnorm=ch.bn_name, count=count, reason=why)
            if n_due:
                idx = due[sl].nonzero().flatten()
                self._repair(step, name, ch.prod, ch.bn, idx)
                streak[idx + ch.offset] = 0
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
        reason_off = None
        if not self._channel_checks:
            reason_off = ("parameters are sharded (FSDP): per-channel weight scans, the dead-channel "
                          "certificate and repair are disabled")
            notes.append(reason_off)
        if self.optimizer is not None and self._opt is None:
            notes.append(f"{type(self.optimizer).__name__} is not a torch.optim.Optimizer: no certificate "
                         "and no repair; alarms and the record work")
        elif self._opt is not None and self._family is None:
            notes.append(f"{type(self._opt).__name__} is not an optimizer the certificate covers: no "
                         "certificate, so no channel is repaired early as proven dead; flat and decayed "
                         "channels are still found (and repaired, if enabled)")
        chains = {c.name: c for c in self._chains}
        producers = []
        for name in self.producers:
            ch = chains.get(name)
            bn_name = self.pairs.get(name)
            why = reason_off or (ch.reason if ch is not None else "no 2-D weight to scan")
            producers.append({"layer": name, "kind": self.kinds[name], "batchnorm": bn_name,
                              "relu_after": (bn_name or name) in self._relu_after,
                              "certificate": why is None, "reason": why})
        if any(p["certificate"] for p in producers):
            notes.append("a certificate is issued per channel only if, at that check, the optimizer "
                         "holds no first-moment momentum for it, its gradient is exactly zero, its "
                         "BatchNorm is in training mode and weight decay cannot lift its shift above 0")
            notes.append("the producer -> BatchNorm -> ReLU structure is read from the autograd graph of "
                         f"the first check (from the {self.graph_source}); a later change in how often a "
                         "module runs revokes eligibility, other graph changes are caught by the "
                         "per-check gradient guard")
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
            "optimizer": {"type": type(self.optimizer).__name__ if self.optimizer is not None else None,
                          "certificate_family": self._family},
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
