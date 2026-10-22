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
"""
from __future__ import annotations

import hashlib
import math
import platform
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from . import __version__
from .diary import Diary, file_sha256
from .structure import BN_KINDS, CONV_KINDS, pair_by_order, scan_channels

RELU_KINDS = (nn.ReLU, nn.ReLU6)
FIRST_MOMENT = ("exp_avg", "momentum_buffer", "grad_avg")   # optimizer state that moves weights


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


class _Baseline:
    """Running mean and variance of log10(gradient norm) for one layer."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 0.0
        self.n = 0
        self.alarm: str | None = None
        self.pending: str | None = None
        self.pending_count = 0

    def update(self, x: float) -> None:
        a = max(0.1, 1.0 / (self.n + 1))
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)
        self.n += 1


class Monitor:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *,
                 out: str | Path = "ormas_run", repair: bool | str = False,
                 check_every: int = 50, patience: int = 4, sigma: float = 2.5,
                 warmup: int = 8, name: str | None = None,
                 sinks: "tuple[Callable[[str, dict], None], ...] | list" = (),
                 verbose: bool = True) -> None:
        if repair and optimizer is None:
            raise ValueError("repair needs the optimizer, so its state for repaired channels can be reset")
        if repair not in (False, True, "all"):
            raise ValueError("repair must be False, True (certified and flat channels) or 'all'")
        self.model, self.optimizer = model, optimizer
        self.out = Path(out)
        self.repair_mode = repair
        self.check_every, self.patience, self.sigma, self.warmup = check_every, patience, sigma, warmup
        self.verbose = verbose
        # Sinks receive every record the monitor writes: sink(kind, payload) with kind in
        # {"event", "metrics"}. They must not break training: a failing sink is reported once
        # as a `sink_error` event and then skipped.
        self.sinks = list(sinks)
        self._sink_failed: set[int] = set()

        self.modules: dict[str, nn.Module] = {
            n: m for n, m in model.named_modules()
            if any(p.requires_grad for p in m.parameters(recurse=False))}
        self.kinds = {n: type(m).__name__ for n, m in self.modules.items()}
        self.sizes = {n: _out_channels(m) for n, m in self.modules.items()}
        self.producers = [n for n, k in self.kinds.items() if k in CONV_KINDS]

        self.step_count = 0
        self.checks = 0
        self._sampling = True
        self._order: list[str] = []
        self._relu_after: set[str] = set()      # layers whose output goes straight into a ReLU
        self._last_out = None
        self._act_max: dict[str, torch.Tensor] = {}
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
                       "loss_spikes": 0, "nonfinite": 0}
        self._closed = False

        self._handles = []
        for n, m in self.modules.items():
            if self.kinds[n] in CONV_KINDS or self.kinds[n] in BN_KINDS:
                self._handles.append(m.register_forward_hook(self._hook(n)))
        for m in model.modules():
            if isinstance(m, RELU_KINDS):
                self._handles.append(m.register_forward_pre_hook(self._relu_hook))

        self.out.mkdir(parents=True, exist_ok=True)
        self.diary = Diary(self.out / "diary.jsonl")
        self._metrics = open(self.out / "metrics.jsonl", "w")
        shapes = "|".join(f"{n}:{tuple(p.shape)}" for n, p in model.named_parameters())
        opt = None
        if optimizer is not None:
            g = optimizer.param_groups[0]
            opt = {"type": type(optimizer).__name__,
                   **{k: g[k] for k in ("lr", "weight_decay", "momentum", "betas") if k in g}}
        self.diary.write(
            "run_start", name=name or self.out.name, tool=f"ormas-monitor {__version__}",
            model=type(model).__name__,
            parameters=sum(p.numel() for p in model.parameters()),
            layers=len(self.modules), architecture_sha256=hashlib.sha256(shapes.encode()).hexdigest(),
            optimizer=opt, python=sys.version.split()[0], torch=torch.__version__,
            platform=platform.platform(),
            config={"check_every": check_every, "patience": patience, "sigma": sigma,
                    "warmup": warmup, "repair": repair})

    # ------------------------------------------------------------------ hooks
    def _hook(self, name: str):
        kind = self.kinds[name]

        def fn(module, inputs, output):
            if not self._sampling or not module.training or not isinstance(output, torch.Tensor):
                return
            if self.pairs is None:
                self._order.append(name)
                self._last_out = (name, output, output._version)
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
    def step(self, loss: torch.Tensor | float | None = None) -> None:
        """Call once per training step, after loss.backward() and before optimizer.step()."""
        step, sampled = self.step_count, self._sampling
        self.step_count += 1
        if sampled:
            self._check(step, loss)
        self._sampling = self.step_count % self.check_every == 0

    def close(self, report: bool = True) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        for h in self._handles:
            h.remove()
        self._metrics.close()
        self.diary.write("run_end", steps=self.step_count, checks=self.checks, counts=self.counts,
                         metrics_sha256=file_sha256(self.out / "metrics.jsonl"))
        self.diary.close()
        if not report:
            return None
        from .report import write_all
        paths = write_all(self.out)
        for p in paths.values():
            self._say(f"report written to {p}")
        return paths["report"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ internals
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(f"[ormas-monitor] {msg}", flush=True)

    def _emit(self, kind: str, payload: dict) -> None:
        for i, sink in enumerate(self.sinks):
            if i in self._sink_failed:
                continue
            try:
                sink(kind, payload)
            except Exception as e:      # a dashboard outage must not stop a training run
                self._sink_failed.add(i)
                line = self.diary.write("sink_error", sink=repr(sink), error=f"{type(e).__name__}: {e}")
                self._say(f"sink {sink!r} failed and is disabled for this run: {e}")
                for j, other in enumerate(self.sinks):
                    if j not in self._sink_failed:
                        try:
                            other("event", line)
                        except Exception:
                            self._sink_failed.add(j)

    def _event(self, event: str, step: int, msg: str | None = None, **fields) -> None:
        line = self.diary.write(event, step=step, **fields)
        self._emit("event", line)
        if msg:
            self._say(f"step {step:,}: {msg}")

    def _check(self, step: int, loss) -> None:
        self.checks += 1
        if self.pairs is None:
            order = self._order or list(self.modules)
            self.pairs = pair_by_order(order, self.kinds, self.sizes)
            self._last_out = None
            self._event("structure", step, pairs=self.pairs, relu_after=sorted(self._relu_after))

        loss_val = None
        if loss is not None:
            loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
            self._check_loss(step, loss_val)

        grads = self._check_gradients(step)
        stuck = self._check_channels(step)
        row = {"step": step, "loss": loss_val, "grad": grads, "stuck": stuck}
        self._metrics.write(_dumps(row) + "\n")
        self._emit("metrics", row)
        self._act_max.clear()

    def _check_loss(self, step: int, x: float) -> None:
        if not math.isfinite(x):
            if not self._loss_bad:
                self.counts["nonfinite"] += 1
                self._event("nonfinite_loss", step, f"loss is {x}", severity="critical", value=str(x))
            self._loss_bad = True
            return
        self._loss_bad = False
        b = self._loss
        if b.n >= self.warmup:
            sd = math.sqrt(b.var) + 1e-12
            if x > b.mean + self.sigma * sd and x > 1.5 * b.mean:
                if not self._spiking:     # one event per episode, not one per check
                    self.counts["loss_spikes"] += 1
                    self._event("loss_spike", step, f"loss spiked to {x:.4g} (baseline {b.mean:.4g})",
                                severity="warning", value=x, baseline=b.mean)
                self._spiking = True
                return
            if self._spiking:
                self._event("loss_recovered", step, f"loss back to normal ({x:.4g})", value=x, baseline=b.mean)
            self._spiking = False
        b.update(x)

    def _check_gradients(self, step: int) -> dict[str, float]:
        names, norms = [], []
        for n, m in self.modules.items():
            gs = [p.grad for p in m.parameters(recurse=False) if p.grad is not None]
            if gs:
                names.append(n)
                norms.append(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs])))
        if not names:
            return {}
        values = torch.stack(norms).float().cpu().tolist()
        out = {}
        bad = [n for n, g in zip(names, values) if not math.isfinite(g)]
        if bad and not self._grad_bad:
            self.counts["nonfinite"] += 1
            self._event("nonfinite_gradient", step, f"non-finite gradients in {len(bad)} layers",
                        severity="critical", layers=bad[:32])
        self._grad_bad = bool(bad)
        for n, g in zip(names, values):
            out[n] = g
            if math.isfinite(g):
                self._update_layer(step, n, g)
        return out

    def _update_layer(self, step: int, name: str, g: float) -> None:
        b = self._base[name]
        x = math.log10(g + 1e-30)
        cond = None
        if b.n >= self.warmup:
            sd = math.sqrt(b.var) + 1e-6
            if g == 0 and b.mean > -29:
                cond = "no_gradient"
            elif x < b.mean - max(self.sigma * sd, 1.0):
                cond = "vanishing"
            elif x > b.mean + max(self.sigma * sd, 1.0):
                cond = "exploding"
        if cond:
            b.pending_count = b.pending_count + 1 if b.pending == cond else 1
            b.pending = cond
            if b.pending_count >= 2 and b.alarm != cond:
                b.alarm = cond
                self.counts["alarms"] += 1
                words = {"no_gradient": "gradient is exactly zero",
                         "vanishing": "gradient collapsed",
                         "exploding": "gradient exploded"}[cond]
                self._event("alarm", step, f"{name} ({self.kinds[name]}): {words} "
                            f"({g:.3g} vs normal {10 ** b.mean:.3g})",
                            severity="warning", layer=name, kind=self.kinds[name],
                            diagnosis=cond, grad_norm=g, baseline=10 ** b.mean, sigma=self.sigma)
            return
        b.pending, b.pending_count = None, 0
        if b.alarm:
            self._event("alarm_cleared", step, f"{name}: back to normal", layer=name,
                        diagnosis=b.alarm, grad_norm=g, baseline=10 ** b.mean)
            b.alarm = None
        b.update(x)

    def _check_channels(self, step: int) -> dict[str, list[int]]:
        out = {}
        for name in self.producers:
            prod = self.modules[name]
            bn = self.modules.get(self.pairs.get(name, ""))
            weight = getattr(prod, "weight", None)
            if weight is None or weight.dim() < 2:
                continue
            n = weight.shape[0]
            scan = scan_channels(weight, getattr(prod, "bias", None),
                                 getattr(bn, "weight", None) if bn is not None else None,
                                 getattr(bn, "bias", None) if bn is not None else None)
            out_name = self.pairs.get(name) if bn is not None else name
            feeds_relu = out_name in self._relu_after
            # The certificate needs a ReLU directly after the layer and no optimizer momentum
            # left that could still move the channel; otherwise the channel is only constant.
            certified = set(scan.certified) if feeds_relu else set()
            if certified and self.optimizer is not None:
                moving = self._momentum(name, prod, bn)
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
                observed = set((quiet >= self.patience).nonzero().flatten().tolist()) - certified - flat - decayed

            mask = torch.zeros(n, dtype=torch.bool)
            targets = certified | flat | decayed | (observed if self.repair_mode == "all" else set())
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
                self._event("channels" if level != "none" else "channels_cleared", step, msg,
                            severity="warning" if level != "none" else "info", layer=name,
                            batchnorm=self.pairs.get(name), channels=n, level=level,
                            n_certified=len(certified), n_flat=len(flat), n_observed=len(observed),
                            n_decayed=len(decayed),
                            certified=sorted(certified)[:64], flat=sorted(flat)[:64],
                            observed=sorted(observed)[:64])
                self._channel_level[name] = level
            if certified and name not in self._certified_seen:
                self._certified_seen.add(name)
                self._event("certificate", step, severity="proof", layer=name, batchnorm=self.pairs.get(name),
                            channels=sorted(certified)[:256], count=len(certified),
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
                due = ((streak >= self.patience) | (proven & (streak >= 2))).nonzero().flatten()
                if due.numel():
                    self._repair(step, name, prod, bn, due)
                    streak[due] = 0
        return out

    def _momentum(self, name: str, prod: nn.Module, bn: nn.Module | None) -> torch.Tensor:
        """Per channel: does the optimizer still hold first-moment momentum for it?"""
        n = prod.weight.shape[0]
        moving = torch.zeros(n, dtype=torch.bool)
        params = [prod.weight, getattr(prod, "bias", None)]
        if bn is not None:
            params += [bn.weight, bn.bias]
        for p in params:
            if p is None:
                continue
            for key, v in self.optimizer.state.get(p, {}).items():
                if key in FIRST_MOMENT and isinstance(v, torch.Tensor) and v.shape == p.shape:
                    moving |= (v.detach().reshape(n, -1).abs().amax(dim=1) > 0).cpu()
        return moving

    def _repair(self, step: int, name: str, prod: nn.Module, bn: nn.Module | None,
                idx: torch.Tensor) -> None:
        w = prod.weight
        dev_idx = idx.to(w.device)
        params = [w]
        with torch.no_grad():
            fresh = torch.empty((len(idx),) + tuple(w.shape[1:]), device=w.device, dtype=w.dtype)
            nn.init.kaiming_normal_(fresh, nonlinearity="relu")
            w[dev_idx] = fresh
            if getattr(prod, "bias", None) is not None:
                prod.bias[dev_idx] = 0
                params.append(prod.bias)
            if bn is not None:
                if bn.weight is not None:
                    bn.weight[dev_idx] = 1
                    bn.bias[dev_idx] = 0
                    params += [bn.weight, bn.bias]
                if bn.running_mean is not None:
                    bn.running_mean[dev_idx] = 0
                    bn.running_var[dev_idx] = 1
        reset = 0
        for p in params:
            for v in self.optimizer.state.get(p, {}).values():
                if isinstance(v, torch.Tensor) and v.shape == p.shape:
                    v[dev_idx] = 0
                    reset += 1
        self.counts["repairs"] += 1
        self.counts["channels_repaired"] += len(idx)
        self._event("repair", step, f"{name}: re-initialised {len(idx)} stuck channels",
                    severity="action", layer=name, batchnorm=self.pairs.get(name),
                    channels=idx.tolist()[:256], count=len(idx),
                    method="Kaiming-normal filter, bias 0, BatchNorm scale 1 and shift 0, "
                           "running stats reset, optimizer state for these channels zeroed",
                    optimizer_slices_reset=reset)


def _dumps(obj) -> str:
    import json
    return json.dumps(obj, separators=(",", ":"))
